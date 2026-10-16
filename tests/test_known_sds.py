"""The known system, and its Gymnasium environment.

Two properties carry the weight here. The first is reproducibility: every rollout in this part
of the library has to be reproducible from a ``Generator``, because the same trajectory is
supposed to be replayable by a different implementation (a browser, a script, a test) and land
on the same score. The second is that the noise law really is interchangeable -- the noise in
the predictability work is *uniform*, which no ambiguity set can express, so the environment has
to accept a law the rest of the package never handed it before.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from _systems import PAPER_A, paper_noise_law
from ppsds.dynamics.known_sds import AdditiveDriftSds
from ppsds.environment import Env, KnownSdsEnv
from ppsds.numerics.atom_pmf import AtomPmf
from ppsds.numerics.random_variables import multivariate_uniform


def _shear(x, u):
    """The paper's drift: ``A @ x``, with no control channel."""
    return PAPER_A @ np.asarray(x, dtype=float)


def _sds(x0=None):
    return AdditiveDriftSds(_shear, paper_noise_law(), x0=x0)


def test_a_step_is_the_drift_plus_a_noise_draw():
    """``x' = f(x) + w`` -- the additive-drift assumption, in executable form."""
    sds = _sds()
    state = np.array([0.5, -0.25])
    expected_drift = _shear(state, None)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        noise = paper_noise_law().rvs(size=1, rng=np.random.default_rng(seed))[0]
        assert sds.transition(state, None, rng=rng) == pytest.approx(expected_drift + noise)


def test_a_rollout_returns_states_and_the_noise_that_drove_them():
    """States cover ``T + 1`` points, noise ``T``, and they are consistent with each other."""
    sds = _sds(x0=np.array([1.0, 2.0]))
    rollout = sds.sample_trajectory(7, rng=np.random.default_rng(0))
    assert rollout['x'].shape == (8, 2)
    assert rollout['w'].shape == (7, 2)
    assert rollout['x'][0] == pytest.approx(np.array([1.0, 2.0]))
    for step in range(7):
        assert rollout['x'][step + 1] == pytest.approx(
            _shear(rollout['x'][step], None) + rollout['w'][step]
        )


def test_a_rollout_is_reproducible_from_its_seed():
    """Same seed, same trajectory -- the property the game/experiment round-trip rests on."""
    first = _sds().sample_trajectory(5, rng=np.random.default_rng(42))
    again = _sds().sample_trajectory(5, rng=np.random.default_rng(42))
    assert np.array_equal(first['x'], again['x'])
    assert np.array_equal(first['w'], again['w'])


def test_the_noise_stays_inside_the_law_s_it_given():
    """The whole point of naming a law: the noise is *that* law, not a Gaussian stand-in."""
    noise = _sds().sample_trajectory(500, rng=np.random.default_rng(1))['w']
    assert np.all(noise >= -1.0) and np.all(noise <= 1.0)


def test_a_scipy_noise_law_is_accepted_too():
    """``scipy`` takes its random source as ``random_state``, ours as ``rng``; both must work.

    The keyword is detected from the signature, so a scipy law is usable without an adapter --
    which matters because that is what a user with their own system will reach for.
    """
    gaussian = AdditiveDriftSds(_shear, multivariate_normal(mean=np.zeros(2), cov=np.eye(2)))
    rollout = gaussian.sample_trajectory(4, rng=np.random.default_rng(0))
    assert rollout['w'].shape == (4, 2)
    assert np.all(np.abs(rollout['w']) < 6), 'a standard normal should not be out at 6 sigma'

    pmf = AdditiveDriftSds(_shear, AtomPmf(np.zeros((1, 2)), np.ones(1)))
    assert pmf.sample_trajectory(3, rng=np.random.default_rng(0))['w'].shape == (3, 2)


def test_a_control_may_be_constant_or_per_step():
    """A control channel is passed through to the drift, in either shape."""
    seen = []

    def drift(x, u):
        seen.append(np.asarray(u, dtype=float).copy())
        return np.asarray(x, dtype=float) + np.asarray(u, dtype=float)

    sds = AdditiveDriftSds(drift, multivariate_uniform(low=-0.1 * np.ones(1), high=0.1 * np.ones(1)),
                           x0=np.zeros(1))
    sds.sample_trajectory(3, rng=np.random.default_rng(0), u=np.array([2.0]))
    assert all(u == pytest.approx(np.array([2.0])) for u in seen)

    seen.clear()
    sds.sample_trajectory(3, rng=np.random.default_rng(0), u=np.array([[1.0], [2.0], [3.0]]))
    assert [float(u[0]) for u in seen] == [1.0, 2.0, 3.0]


def test_a_control_of_the_wrong_length_is_rejected():
    """Rather than broadcasting a control over the horizon."""
    with pytest.raises(ValueError, match='constant'):
        _sds().sample_trajectory(3, rng=np.random.default_rng(0), u=np.zeros((2, 1)))


def test_a_misshaped_initial_state_is_rejected():
    """The state moves in the noise's dimension, and a mismatch is a mistake worth naming."""
    with pytest.raises(ValueError, match='coordinate'):
        _sds(x0=np.zeros(3))


def test_a_noise_law_without_rvs_is_rejected():
    """The one thing a noise law must be able to do."""
    with pytest.raises(TypeError, match='rvs'):
        AdditiveDriftSds(_shear, np.zeros(2))


def test_the_environment_has_the_gymnasium_contract():
    """``reset`` gives ``(state, info)``; ``step`` gives the five-tuple; it truncates, never ends."""
    env = KnownSdsEnv(_sds(), x0=np.array([1.0, 1.0]), max_steps=4)
    state, info = env.reset(seed=0)
    assert state == pytest.approx(np.array([1.0, 1.0])) and isinstance(info, dict)

    for step in range(4):
        state, reward, terminated, truncated, info = env.step(np.zeros(0))
        assert reward == 0.0 and terminated is False
        assert truncated is (step == 3), 'truncation lands exactly at max_steps'
        assert isinstance(info, dict)


def test_the_environment_is_reproducible_from_its_reset_seed():
    """Gymnasium's seeding cycle must actually drive the noise."""
    def rollout():
        env = KnownSdsEnv(_sds(), x0=np.zeros(2), max_steps=5)
        env.reset(seed=7)
        return np.array([env.step(np.zeros(0))[0] for _ in range(5)])

    assert np.array_equal(rollout(), rollout())


def test_the_environment_can_start_elsewhere():
    """``options['x0']`` overrides, as it does on the ambiguity-composing ``Env``."""
    env = KnownSdsEnv(_sds(), x0=np.zeros(2))
    state, _ = env.reset(seed=0, options={'x0': np.array([3.0, 4.0])})
    assert state == pytest.approx(np.array([3.0, 4.0]))
    assert env.x0 == pytest.approx(np.zeros(2)), 'the default is not overwritten'


def test_the_environment_exposes_the_system_it_steps():
    """The noise law and drift stay reachable, so a scoring rule can be built from the same object."""
    env = KnownSdsEnv(_sds())
    assert env.noise_law is env.sds.noise_law
    assert env.f_bar is _shear


def test_stepping_before_reset_fails_loudly():
    """There is no state yet, and ``None`` arithmetic would fail somewhere less informative."""
    with pytest.raises(RuntimeError, match='reset'):
        KnownSdsEnv(_sds()).step(np.zeros(0))


def test_the_two_environments_are_not_interchangeable():
    """Handing an ambiguity set to the known-system environment (or the reverse) is a typo."""
    with pytest.raises(TypeError, match='AdditiveDriftSds'):
        KnownSdsEnv(object())
    assert Env is not KnownSdsEnv, 'the ambiguity-composing Env is a separate class'
