"""Contract tests for the Gymnasium environment layer and the four dynamics modes.

``Env`` is **assembled, not subclassed**: an ambiguity set supplies the nominal model,
``transition`` and the legal ``sds_params`` key set, while a ``DynamicsMode`` decides which
``sds_params`` each step uses. So the interesting surfaces are the two seams:

- "the mode's output is legal ``sds_params``", checked at the ``step()`` entry point and meant
  to fail **loudly** (the old code guessed the format from ``'A' in params`` and silently took
  the wrong branch when the guess was wrong);
- "the mode can be driven from the agent side through exactly one channel",
  ``inject_eigvec_fixed_info``, which is a no-op for every non-adversarial mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from _systems import LINEAR_X0
from ppsds.environment import Env
from ppsds.environment.dynamics_mode import (
    DynamicsMode,
    EigvecFixedAdversarialDynamics,
    FixedDynamics,
    PreSampledDynamics,
    RandomDynamics,
)

U0 = np.array([0.1])


def make_env(abg, mode, x0=None, max_steps=5, bounds=None):
    return Env(abg, mode, x0=LINEAR_X0 if x0 is None else x0,
               max_steps=max_steps, bounds=bounds)


def rollout(env, steps, u=U0):
    """Step ``steps`` times, returning the states and the per-step ``info`` dicts."""
    states, infos = [], []
    for _ in range(steps):
        x, _reward, _terminated, _truncated, info = env.step(u)
        states.append(np.array(x))
        infos.append(info)
    return states, infos


# ----------------------------------------------------------------------
# The Gymnasium API
# ----------------------------------------------------------------------

def test_spaces_match_the_ambiguity_set_dimensions(linear_set):
    env = make_env(linear_set, FixedDynamics())
    assert env.observation_space.shape == (linear_set.x_dim,)
    assert env.action_space.shape == (linear_set.u_dim,)
    assert env.observation_space.dtype == np.float64


def test_reset_starts_from_the_declared_initial_state(linear_set):
    env = make_env(linear_set, FixedDynamics())
    x, info = env.reset(seed=0)
    np.testing.assert_array_equal(x, LINEAR_X0)
    assert info == {}
    assert env._step_count == 0


def test_reset_can_override_the_initial_state(linear_set):
    env = make_env(linear_set, FixedDynamics())
    x, _ = env.reset(seed=0, options={'x0': np.array([9.0, -9.0])})
    np.testing.assert_array_equal(x, np.array([9.0, -9.0]))


def test_step_returns_the_gymnasium_five_tuple(linear_set):
    env = make_env(linear_set, FixedDynamics(), max_steps=2)
    env.reset(seed=0)
    out = env.step(U0)
    assert len(out) == 5
    x, reward, terminated, truncated, info = out
    assert x.shape == (linear_set.x_dim,)
    assert reward == 0.0, 'prediction quality is assessed externally, so the reward is inert'
    assert terminated is False and truncated is False
    assert info['mode'] == 'FixedDynamics'
    assert info['sds_params'] == {}


def test_truncation_happens_after_max_steps(linear_set):
    env = make_env(linear_set, FixedDynamics(), max_steps=3)
    env.reset(seed=0)
    flags = [env.step(U0)[3] for _ in range(3)]
    assert flags == [False, False, True]


def test_bounds_terminate_early_and_none_never_terminates(linear_set):
    bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    env = make_env(linear_set, FixedDynamics(), max_steps=50, bounds=bounds)
    env.reset(seed=0)
    # the nominal two-state system wanders away from x0 = (2, 1), which is already outside
    assert env.step(U0)[2] is True

    unbounded = make_env(linear_set, FixedDynamics(), max_steps=50)
    unbounded.reset(seed=0)
    assert unbounded.step(U0)[2] is False


def test_the_step_count_is_not_the_episode_length(linear_set):
    """``truncated`` is driven by ``max_steps``, not by how many steps were actually taken."""
    env = make_env(linear_set, FixedDynamics(), max_steps=10)
    env.reset(seed=0)
    assert [env.step(U0)[3] for _ in range(3)] == [False, False, False]


# ----------------------------------------------------------------------
# The loud contract on sds_params
# ----------------------------------------------------------------------

class _BogusMode(DynamicsMode):
    """A mode that returns a key the ambiguity set does not accept."""

    def reset(self, rng, T):
        pass

    def get_params(self, step_idx):
        return {'not_a_real_key': 1.0}


def test_env_step_rejects_an_illegal_sds_params_key(linear_set):
    """``Env.step`` validates the key set at the entry point, so a bad mode fails at the seam
    rather than deep inside the dynamics with a confusing error."""
    env = make_env(linear_set, _BogusMode())
    env.reset(seed=0)
    with pytest.raises(ValueError, match='not_a_real_key'):
        env.step(U0)


def test_dynamics_mode_is_abstract():
    with pytest.raises(TypeError):
        DynamicsMode()


def test_the_base_set_info_is_a_no_op():
    """Defined on the base class so the caller does not have to sniff with ``hasattr``."""

    class _Trivial(DynamicsMode):
        def reset(self, rng, T):
            pass

        def get_params(self, step_idx):
            return {}

    assert _Trivial().set_info(np.ones(2), np.zeros(2), np.zeros(1)) is None


# ----------------------------------------------------------------------
# FixedDynamics
# ----------------------------------------------------------------------

def test_fixed_dynamics_never_perturbs_anything(linear_set):
    env = make_env(linear_set, FixedDynamics())
    env.reset(seed=0)
    for info in rollout(env, 5)[1]:
        assert info['sds_params'] == {}, 'an empty dict means "all nominal"'

    # and the state advances with the nominal drift plus noise -- so it is not simply frozen
    env2 = make_env(linear_set, FixedDynamics())
    x0, _ = env2.reset(seed=0)
    assert not np.array_equal(env2.step(U0)[0], x0)


# ----------------------------------------------------------------------
# RandomDynamics
# ----------------------------------------------------------------------

def test_random_dynamics_is_seed_reproducible(linear_set):
    """Same seed, same parameter trajectory -- "sample once, run many"."""
    runs = []
    for _ in range(2):
        env = make_env(linear_set, RandomDynamics(linear_set, period=1))
        env.reset(seed=7)
        runs.append(rollout(env, 4)[0])
    np.testing.assert_array_equal(runs[0], runs[1])


def test_random_dynamics_differs_across_seeds(linear_set):
    states = []
    for seed in (1, 2):
        env = make_env(linear_set, RandomDynamics(linear_set, period=1))
        env.reset(seed=seed)
        states.append(rollout(env, 4)[0])
    assert not np.array_equal(states[0], states[1])


def test_random_dynamics_honours_the_period(linear_set):
    """``period=100`` over a 5-step episode must look constant; ``period=1`` must not."""
    env_slow = make_env(linear_set, RandomDynamics(linear_set, period=100))
    env_slow.reset(seed=3)
    slow = [info['sds_params']['mu'].copy() for info in rollout(env_slow, 5)[1]]

    env_fast = make_env(linear_set, RandomDynamics(linear_set, period=1))
    env_fast.reset(seed=3)
    fast = [info['sds_params']['mu'].copy() for info in rollout(env_fast, 5)[1]]

    np.testing.assert_array_equal(slow[0], slow[-1])
    assert not np.array_equal(fast[0], fast[-1])


def test_random_dynamics_does_not_pollute_the_global_numpy_rng(linear_set):
    """``reset()`` seeds the global RNG from the env's own generator to reproduce the
    ambiguity set's global-RNG sampling, and must put the previous state back."""
    np.random.seed(1234)
    before = np.random.random()
    np.random.seed(1234)

    env = make_env(linear_set, RandomDynamics(linear_set, period=1))
    env.reset(seed=5)
    after = np.random.random()

    assert before == after, 'the global numpy RNG was left in a different state'


def test_random_dynamics_refuses_to_run_before_reset(linear_set):
    mode = RandomDynamics(linear_set)
    with pytest.raises(RuntimeError, match='reset'):
        mode.get_params(0)


# ----------------------------------------------------------------------
# PreSampledDynamics
# ----------------------------------------------------------------------

def test_pre_sampled_dynamics_replays_the_given_parameters(linear_set):
    np.random.seed(11)
    params = linear_set.sample_trajectory(5, period=1)

    env = make_env(linear_set, PreSampledDynamics(params))
    env.reset(seed=0)
    for i, info in enumerate(rollout(env, 5)[1]):
        for key in linear_set.param_keys:
            np.testing.assert_array_equal(info['sds_params'][key], params[key][i])


def test_pre_sampled_dynamics_ignores_the_seed(linear_set):
    """The parameters are already fixed, so the seed must not change them -- only the noise."""
    np.random.seed(11)
    params = linear_set.sample_trajectory(4, period=1)

    mus = []
    for seed in (1, 2):
        env = make_env(linear_set, PreSampledDynamics(params))
        env.reset(seed=seed)
        mus.append([info['sds_params']['mu'].copy() for info in rollout(env, 4)[1]])
    np.testing.assert_array_equal(mus[0], mus[1])


# ----------------------------------------------------------------------
# inject_eigvec_fixed_info: a no-op for every non-adversarial mode
# ----------------------------------------------------------------------

@pytest.mark.parametrize('mode_factory', [
    lambda abg: FixedDynamics(),
    lambda abg: RandomDynamics(abg, period=1),
    lambda abg: PreSampledDynamics(_presampled(abg)),
])
def test_injection_is_a_no_op_for_non_adversarial_modes(linear_set, mode_factory):
    """The call is unconditional at the call site, so a mode that ignores it must ignore it
    *completely* -- not merely tolerate it, and certainly not consume randomness."""
    plain = make_env(linear_set, mode_factory(linear_set))
    plain.reset(seed=4)
    expected = rollout(plain, 4)[0]

    injected = make_env(linear_set, mode_factory(linear_set))
    injected.reset(seed=4)
    lamb_pred = np.array([1.0, 2.0])
    states = []
    for _ in range(4):
        injected.inject_eigvec_fixed_info(lamb_pred, U0)
        states.append(np.array(injected.step(U0)[0]))

    np.testing.assert_array_equal(np.stack(states), np.stack(expected))


def _presampled(abg):
    np.random.seed(11)
    return abg.sample_trajectory(10, period=1)


@pytest.mark.parametrize('mode_factory', [
    lambda abg: FixedDynamics(),
    lambda abg: EigvecFixedAdversarialDynamics(abg),
])
def test_injecting_none_skips_the_injection(linear_set, mode_factory):
    """``lamb_pred=None`` means "no prediction yet" (the first step), and is documented as a
    skip -- so it must be safe for every mode."""
    env = make_env(linear_set, mode_factory(linear_set))
    env.reset(seed=0)
    env.inject_eigvec_fixed_info(None, U0)
    env.step(U0)          # must not raise


# ----------------------------------------------------------------------
# EigvecFixedAdversarialDynamics
# ----------------------------------------------------------------------

def test_adversarial_mode_falls_back_to_nominal_on_the_first_step(linear_set):
    """There is no prediction before the first ``step()``, so the mode returns ``{}``."""
    mode = EigvecFixedAdversarialDynamics(linear_set)
    env = make_env(linear_set, mode)
    env.reset(seed=0)
    assert env.step(U0)[4]['sds_params'] == {}
    assert mode.j == -1, 'no worst direction has been chosen yet'


def test_adversarial_mode_uses_the_injected_prediction(linear_set):
    """Pins both the *content* and the *argument order* of the injection.

    ``Env.inject_eigvec_fixed_info(lamb_pred, u, x=None)`` calls
    ``mode.set_info(lamb_pred, x, u)`` -- a different order, and the mode forwards x and u to
    ``worst_case_params(x, u, lamb_pred)``. Passing them through in the wrong order would still
    produce plausible-looking arrays, so this re-derives the expected result from the fixture's
    own inputs instead of comparing against a second call of the same code path.
    """
    mode = EigvecFixedAdversarialDynamics(linear_set)
    env = make_env(linear_set, mode)
    env.reset(seed=0)
    env.step(U0)                                  # consume the nominal first step

    lamb_pred = np.array([1.0, 2.0])
    u = np.array([0.3])
    x_at_injection = env.x.copy()
    env.inject_eigvec_fixed_info(lamb_pred, u, x=x_at_injection)
    params = env.step(u)[4]['sds_params']

    assert 'j' not in params, (
        'the worst-direction index is diagnostic and must be popped by the mode -- Env.step '
        'would reject it as an illegal sds_params key')
    assert 0 <= mode.j < linear_set.x_dim

    expected = linear_set.worst_case_params(x_at_injection, u, lamb_pred)
    expected.pop('j')
    assert set(params) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(params[key], value)

    linear_set.validate(params)                   # the adversarial output is legal by construction


def test_adversarial_mode_reset_clears_the_injection(linear_set):
    mode = EigvecFixedAdversarialDynamics(linear_set)
    env = make_env(linear_set, mode, max_steps=5)
    env.reset(seed=0)
    env.step(U0)
    env.inject_eigvec_fixed_info(np.array([1.0, 2.0]), U0)
    env.step(U0)
    assert mode.j >= 0

    env.reset(seed=0)
    assert mode.j == -1
    assert env.step(U0)[4]['sds_params'] == {}


# ----------------------------------------------------------------------
# The nonlinear family goes through the same env
# ----------------------------------------------------------------------

def test_env_works_with_the_nonlinear_family(nonlinear_set):
    """The env is generic: the family is carried entirely by the ambiguity set."""
    env = Env(nonlinear_set, RandomDynamics(nonlinear_set, period=1),
              x0=np.array([0.3, 0.0]), max_steps=4)
    env.reset(seed=0)
    states, infos = rollout(env, 4, u=np.zeros(1))
    assert np.all(np.isfinite(np.stack(states)))
    for info in infos:
        assert set(info['sds_params']) == set(nonlinear_set.param_keys)
        nonlinear_set.validate(info['sds_params'])


def test_env_exposes_its_ambiguity_set_and_mode(linear_set):
    mode = FixedDynamics()
    env = make_env(linear_set, mode)
    assert env.ambiguity_set is linear_set
    assert env.dynamics_mode is mode
