"""One test per bug this repository has already shipped once.

These are the highest-value tests in the suite: each encodes a specific failure that actually
happened, so the next occurrence is caught here rather than by a user. Each test carries the story
of what went wrong, because "why is this asserted" is the part that gets lost first.

The bugs, roughly in the order they were found:

1. **State leaking across episodes** -- ``trajectory_prediction`` rewrites ``f_bar`` / ``mu_bar``
   / ``Sigma_bar`` in place, so clearing only the prediction outputs was not enough.
2. **``np.linalg.eig`` dtype drift** -- the same real symmetric matrix comes back ``float64`` on
   numpy 2.3.5 and ``complex128`` on 2.5.4, and the cone program then refuses to solve.
3. **``NLink`` silently ignoring ``period``** -- ``-p 100`` and ``-p 1`` gave byte-identical
   output, so a figure claiming slowly-varying parameters was showing i.i.d. ones.
4. **The multiprocessing start method degrading silently** -- NLink's ambiguity set holds
   lambdas, so ``spawn`` cannot pickle the context.
5. **Hardcoded plot indices crashing the plotting stage** -- data already computed and written to
   disk, then ``IndexError``, and a single figure made the whole experiment exit non-zero.
6. **A mis-shaped input broadcast silently** -- ``u`` of shape ``(1, 1)`` turned ``B @ u`` into
   ``(dim_x, 1)`` and adding the ``(dim_x,)`` bias broadcast *that* to ``(dim_x, dim_x)``; the
   error only surfaced later in ``predict()`` with an incomprehensible message.
"""

from __future__ import annotations

import multiprocessing as mp
import warnings

import numpy as np
import pytest

from _systems import control_at, linear_ambiguity_set, nonlinear_ambiguity_set, true_step
from ppsds.numerics.linalg import eig_real
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (
    EigDrpp,
    NominalPredictor,
    eigvec_fixed_minmax,
)
from ppsds.scoring_rule.logarithm_score import LogarithmScore
from ppsds.visualization.bounds import clamp_index, clamp_len, clamp_time_indexes

T = 4


def _trajectory(T=T, seed=5):
    """A deterministic true trajectory plus inputs, both of shape ``(T+1, ...)``."""
    rng = np.random.default_rng(seed)
    x = np.array([2.0, 1.0])
    states, inputs = [x], []
    for i in range(T):
        u = control_at(i)
        inputs.append(u)
        x = true_step(x, u, rng)
        states.append(x)
    inputs.append(control_at(T))
    return np.stack(states), np.stack(inputs)


def _run_episode(pred, params, x_traj, u_traj):
    """One ``trajectory_prediction`` call, with the deprecation warning suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        return pred.trajectory_prediction(LogarithmScore(), x_traj, u_traj, params)


def _sampled_params(abg, seed=11):
    np.random.seed(seed)
    return abg.sample_trajectory(T, period=1)


# ----------------------------------------------------------------------
# 1. State leaking across episodes
# ----------------------------------------------------------------------

@pytest.mark.parametrize('factory', [NominalPredictor, EigDrpp])
def test_reusing_a_predictor_matches_two_fresh_ones(factory):
    """Bug 1. A second episode on a reused instance must be indistinguishable from a first."""
    abg = linear_ambiguity_set()
    x_traj, u_traj = _trajectory()
    params = _sampled_params(abg)

    reused = factory(abg.f_bar, abg.mu_bar, abg.Sigma_bar) if factory is NominalPredictor \
        else factory(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma)
    _run_episode(reused, params, x_traj, u_traj)            # warm-up episode
    second = _run_episode(reused, params, x_traj, u_traj)   # the episode under test

    fresh = NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar) if factory is NominalPredictor \
        else factory(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma)
    reference = _run_episode(fresh, params, x_traj, u_traj)

    for key in ('x_pred', 'Sigma_pred', 'score'):
        np.testing.assert_array_equal(
            second[key], reference[key],
            err_msg=f'{factory.__name__}: episode 2 on a reused instance differs from a fresh one')


def test_trajectory_prediction_restores_the_nominal_model():
    """Bug 1, the mechanism. ``reset()`` must undo the in-place rewrites, not just clear outputs.

    Clearing ``x_pred`` / ``Sigma_pred`` / ``lamb_pred`` alone would leave the previous episode's
    ``f_bar`` / ``mu_bar`` / ``Sigma_bar`` in place -- and ``lamb`` / ``v`` are derived from
    ``Sigma_bar``, so they go stale too.
    """
    abg = linear_ambiguity_set()
    x_traj, u_traj = _trajectory()
    params = _sampled_params(abg)

    pred = NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar)
    _run_episode(pred, params, x_traj, u_traj)

    np.testing.assert_array_equal(pred.mu_bar, abg.mu_bar)
    np.testing.assert_array_equal(pred.Sigma_bar, abg.Sigma_bar)
    np.testing.assert_array_equal(pred.lamb, eig_real(abg.Sigma_bar)[0])
    assert pred.bias is None
    for attr in ('x_pred', 'Sigma_pred', 'lamb_pred'):
        assert getattr(pred, attr) is None, f'{attr} survived the episode'


def test_trajectory_prediction_warns_that_it_is_deprecated():
    """The migration target does not fully exist (the linear A/B path has no env-layer
    equivalent), so the warning is load-bearing information rather than noise."""
    abg = linear_ambiguity_set()
    x_traj, u_traj = _trajectory()
    pred = NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar)
    with pytest.warns(FutureWarning, match='migrate to the Gymnasium'):
        pred.trajectory_prediction(LogarithmScore(), x_traj, u_traj, None)


# ----------------------------------------------------------------------
# 2. np.linalg.eig dtype drift
# ----------------------------------------------------------------------

def test_eig_real_keeps_the_eigenvalues_real():
    """Bug 2. Taking the real part must be lossless, and must actually produce a real dtype."""
    lamb, vec = eig_real(linear_ambiguity_set().Sigma_bar)
    assert lamb.dtype == np.float64
    assert vec.dtype == np.float64


@pytest.mark.filterwarnings('ignore:Casting complex values')
def test_complex_eigenvalues_are_what_break_the_cone_program():
    """Bug 2, the failure itself: hand the cone program complex eigenvalues and it refuses.

    This is what makes ``eig_real`` load-bearing rather than stylistic. The assertion used to be
    "numpy sometimes hands back ``complex128`` here"; this reproduces that input unconditionally,
    so the test does not silently stop covering the failure on a numpy build that happens to
    return ``float64``.
    """
    abg = linear_ambiguity_set()
    lamb, _ = eig_real(abg.Sigma_bar)
    complex_lamb = lamb.astype(np.complex128)

    with pytest.raises(ValueError, match='real'):
        eigvec_fixed_minmax(complex_lamb, abg.gamma, np.zeros(2), np.zeros(1))

    # ... while the real version solves, so the wrapper is exactly the difference
    lamb_opt, value, j = eigvec_fixed_minmax(lamb, abg.gamma, np.zeros(2), np.zeros(1))
    assert np.all(np.isfinite(lamb_opt)) and np.isfinite(value) and j >= 0


# ----------------------------------------------------------------------
# 3. NLink ignoring `period`
# ----------------------------------------------------------------------

def test_sampling_honours_the_period():
    """Bug 3. ``period`` used to be accepted and then not read, so p=100 and p=1 agreed exactly.

    The failure was silent and self-flattering: a figure labelled "slowly varying parameters"
    was drawn from i.i.d. draws.
    """
    abg = nonlinear_ambiguity_set()

    np.random.seed(3)
    slow = abg.sample_trajectory(T=6, period=100)
    np.random.seed(3)
    fast = abg.sample_trajectory(T=6, period=1)

    # period > T: every step reuses the first draw, so the parameters are constant
    for key in ('delta_f', 'mu'):
        np.testing.assert_array_equal(
            slow[key], np.broadcast_to(slow[key][0], slow[key].shape),
            err_msg=f'{key} is not constant with period=100 over 7 steps')

    # period = 1: resampled every step, so the two trajectories must differ
    assert not np.array_equal(slow['delta_f'], fast['delta_f']), \
        'period=100 and period=1 produced identical drift parameters'
    assert not np.array_equal(
        fast['delta_f'], np.broadcast_to(fast['delta_f'][0], fast['delta_f'].shape)), \
        'period=1 did not resample the drift parameters'


def test_a_block_of_period_is_internally_constant():
    """The piecewise-constant structure itself, not just "the two differ"."""
    abg = nonlinear_ambiguity_set()
    np.random.seed(4)
    traj = abg.sample_trajectory(T=5, period=2)     # steps 0..5 -> blocks {0,1}, {2,3}, {4,5}
    for start in (0, 2, 4):
        np.testing.assert_array_equal(traj['delta_f'][start], traj['delta_f'][start + 1])
    assert not np.array_equal(traj['delta_f'][0], traj['delta_f'][2])


# ----------------------------------------------------------------------
# 4. The multiprocessing start method
# ----------------------------------------------------------------------

def test_configure_refuses_a_non_fork_start_method(monkeypatch):
    """Bug 4. With ``spawn`` the nonlinear experiment used to fail *silently*.

    ``NLink``'s ambiguity set holds ``f_bar`` / ``gamma[0]`` as lambdas, which cannot be
    pickled, so the context can only be inherited from the parent via ``fork``. The guard turns
    that into a loud failure at startup instead of a wrong result.

    ``get_start_method`` is monkeypatched rather than ``set_start_method``-ed, because the start
    method is a process-wide setting that can only be chosen once and changing it here would
    affect every other test.
    """
    mp_runner = pytest.importorskip(
        'ppsds.experiment.drpp.additive_drift.conic_moment',
        reason='the experiment driver needs the `experiments` extra (tqdm)')

    monkeypatch.setattr(mp, 'get_start_method', lambda allow_none=False: 'spawn')
    with pytest.raises(RuntimeError) as excinfo:
        # the guard runs before the context is stored, so the argument is never used
        mp_runner.configure(object())

    message = str(excinfo.value)
    assert 'fork' in message
    assert 'not picklable' in message, 'the message must state the reason, not just the rule'


# ----------------------------------------------------------------------
# 5. Hardcoded plot indices
# ----------------------------------------------------------------------

def test_plot_indices_are_clamped_and_warn():
    """Bug 5. A small ``-t`` must degrade to "draws a different trajectory", not to a crash.

    Clamping **warns** rather than adjusting quietly, because it changes *which* trajectory is
    drawn: silently substituting the last one would leave you believing the figure shows
    trajectory 43. Each entry point names itself in the message so an out-of-range site can be
    located from the warning alone.
    """
    with pytest.warns(RuntimeWarning, match='trajectory index 43 out of range'):
        assert clamp_index(43, 2) == 1
    with pytest.warns(RuntimeWarning, match='trajectory index -4 out of range'):
        assert clamp_index(-4, 3) == 0
    with pytest.warns(RuntimeWarning, match=r'time index \[1, 16, 31\] has out-of-range entries'):
        assert clamp_time_indexes([1, 16, 31], 5) == [1, 4, 4]
    with pytest.warns(RuntimeWarning, match='trajectory length 31 out of range'):
        assert clamp_len(31, 10) == 10


def test_clamping_is_idempotent_and_silent_in_range():
    with warnings.catch_warnings():
        warnings.simplefilter('error')       # any warning becomes a failure
        assert clamp_index(1, 2) == 1
        assert clamp_time_indexes([0, 4], 5) == [0, 4]
        assert clamp_len(5, 10) == 5


def test_a_broken_dataset_still_raises_rather_than_being_clamped_away():
    """Clamping must not paper over "there is no data at all" -- that is a different problem and
    needs to be visible."""
    with pytest.raises(ValueError):
        clamp_index(0, 0)
    with pytest.raises(ValueError):
        clamp_len(1, 0)
    with pytest.raises(ValueError):
        clamp_time_indexes([0], 0)


# ----------------------------------------------------------------------
# 6. Silent broadcasting of a mis-shaped input
# ----------------------------------------------------------------------

def test_update_rejects_a_mis_shaped_input():
    """Bug 6. A column-vector ``u`` used to broadcast all the way into ``Sigma_pred``."""
    abg = linear_ambiguity_set()
    pred = NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar)

    with pytest.raises(ValueError, match='u_observe'):
        pred.update(np.zeros(2), np.zeros((2, 1)))
    with pytest.raises(ValueError, match='x_observe'):
        pred.update(np.zeros((2, 1)), np.zeros(1))
    with pytest.raises(ValueError, match='x_observe'):
        pred.update(np.zeros(3), np.zeros(1))       # right rank, wrong length
    with pytest.raises(ValueError, match='x_observe'):
        pred.update(2.0, np.zeros(1))               # a scalar is not a state
