"""Contract tests for the conic-moment ambiguity sets.

The two claims under test throughout are the ones the module's own docstrings make:

- the ambiguity set **contains its own generator** (every sample is a genuine member, otherwise
  "the set is the ground truth" contradicts itself);
- an unknown ``sds_params`` key is a **loud** error, because the previous "guess the format from
  ``'A' in params``" quietly swallowed exactly that mistake.
"""

from __future__ import annotations

import numpy as np
import pytest

from _systems import linear_ambiguity_set, nonlinear_ambiguity_set
from ppsds.ambiguity.additive_drift.conic_moment import (
    AdditiveDriftConicMomentSet,
    LinearAmbiguitySet,
    NonlinearAmbiguitySet,
)


# ----------------------------------------------------------------------
# Parameterization
# ----------------------------------------------------------------------

def test_param_keys_are_exactly_what_the_two_families_declare():
    assert LinearAmbiguitySet.param_keys == frozenset({'A', 'B', 'mu', 'Sigma'})
    assert NonlinearAmbiguitySet.param_keys == frozenset({'delta_f', 'mu', 'Sigma'})


def test_dimension_aliases_agree(linear_set):
    assert (linear_set.dim_x, linear_set.dim_u) == (linear_set.x_dim, linear_set.u_dim)


def test_gamma_exposes_the_three_radii_in_the_predictor_format(linear_set):
    gamma = linear_set.gamma
    assert len(gamma) == 3
    assert gamma[0](np.array([1.0, 1.0]), np.array([0.0])) == pytest.approx(
        linear_set.f_err(np.array([1.0, 1.0]), np.array([0.0])))
    assert gamma[1] == linear_set.alphas['mu']
    assert gamma[2] == linear_set.alphas['Sigma']
    # and gamma[0] returns a **squared** radius while f_err_max is un-squared
    assert gamma[0](np.zeros(2), np.zeros(1)) == 0.0


# ----------------------------------------------------------------------
# validate()
# ----------------------------------------------------------------------

def test_validate_accepts_a_subset_of_keys(linear_set):
    """A missing key means "take the nominal value", which is legitimate -- the first step of
    adversarial mode has no prediction yet and passes ``{}``."""
    linear_set.validate({})
    linear_set.validate({'mu': np.zeros(2)})
    linear_set.validate({'A': np.eye(2), 'Sigma': np.eye(2)})


def test_validate_rejects_an_unknown_key_loudly(linear_set):
    with pytest.raises(ValueError) as excinfo:
        linear_set.validate({'delta_f': np.zeros(2)})
    message = str(excinfo.value)
    assert 'delta_f' in message, 'the offending key must be named'
    assert 'LinearAmbiguitySet' in message, 'the rejecting class must be named'
    # the accepted keys are listed, so the message is actionable rather than just a refusal
    for key in LinearAmbiguitySet.param_keys:
        assert key in message


def test_nonlinear_set_rejects_the_linear_format(nonlinear_set):
    """The two families use **different** key sets, so crossing them must not pass."""
    with pytest.raises(ValueError, match='delta_f'):
        nonlinear_set.validate({'A': np.eye(2), 'B': np.eye(2)})
    nonlinear_set.validate({'delta_f': np.zeros(2)})


def test_both_subclasses_are_additive_drift_sets():
    assert issubclass(LinearAmbiguitySet, AdditiveDriftConicMomentSet)
    assert issubclass(NonlinearAmbiguitySet, AdditiveDriftConicMomentSet)


# ----------------------------------------------------------------------
# transition(): x' = mean(x, u; params) + w
# ----------------------------------------------------------------------

def test_transition_is_mean_plus_noise(linear_set):
    """``transition`` is where the "additive" assumption is enforced, so it is worth pinning
    the mean assembly exactly -- including that a supplied ``mu`` is added *after* the drift."""
    abg = linear_set
    A, B = abg.A_bar + 0.05, abg.B_bar + 0.1
    mu, Sigma = np.array([0.1, -0.2]), np.eye(2) * 0.5
    x, u = np.array([1.0, 2.0]), np.array([0.3])
    params = {'A': A, 'B': B, 'mu': mu, 'Sigma': Sigma}

    got = abg.transition(x, u, params, np.random.default_rng(3))
    expected_mean = A @ x + B @ u + mu
    expected = np.random.default_rng(3).multivariate_normal(expected_mean, Sigma)

    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(expected_mean, abg.mean(x, u, params) + mu)


def test_transition_falls_back_to_the_nominal_values(linear_set):
    abg = linear_set
    x, u = np.array([1.0, 2.0]), np.array([0.3])
    got = abg.transition(x, u, {}, np.random.default_rng(4))
    expected = np.random.default_rng(4).multivariate_normal(
        abg.f_bar(x, u) + abg.mu_bar, abg.Sigma_bar)
    np.testing.assert_array_equal(got, expected)


def test_nonlinear_transition_adds_delta_f_to_the_nominal_drift(nonlinear_set):
    abg = nonlinear_set
    x, u = np.array([0.3, 0.0]), np.array([0.1])
    delta_f = np.array([0.05, -0.02])
    params = {'delta_f': delta_f}
    got = abg.transition(x, u, params, np.random.default_rng(6))
    expected = np.random.default_rng(6).multivariate_normal(
        abg.f_bar(x, u) + delta_f + abg.mu_bar, abg.Sigma_bar)
    np.testing.assert_array_equal(got, expected)


# ----------------------------------------------------------------------
# sample_trajectory()
# ----------------------------------------------------------------------

@pytest.mark.parametrize('factory', [linear_ambiguity_set, nonlinear_ambiguity_set])
def test_sample_trajectory_shapes_follow_the_declared_fields(factory):
    abg = factory()
    T = 5
    traj = abg.sample_trajectory(T, period=1)
    assert sorted(traj) == sorted(k for k, _ in abg._traj_fields)
    for key, shape in abg._traj_fields:
        assert traj[key].shape == (T + 1,) + shape, f'{key} has the wrong shape'


def test_sample_trajectory_rejects_a_nonpositive_period(linear_set):
    with pytest.raises(ValueError, match='period'):
        linear_set.sample_trajectory(3, period=0)


def test_sample_trajectory_treats_none_as_period_one(linear_set):
    """``sample_trajectory(T, None)`` is the old calling convention and must keep working."""
    np.random.seed(2)
    legacy = linear_set.sample_trajectory(4, period=None)
    np.random.seed(2)
    explicit = linear_set.sample_trajectory(4, period=1)
    np.testing.assert_array_equal(legacy['mu'], explicit['mu'])


# ----------------------------------------------------------------------
# The set contains its own generator
# ----------------------------------------------------------------------

def test_linear_samples_stay_inside_the_declared_set(linear_set):
    """Every component of a sample must obey the radius the ambiguity set declares."""
    abg = linear_set
    np.random.seed(17)
    for _ in range(25):
        d = abg.sample()

        # A: only the superdiagonal may move, and only within alphas['A']
        offset = d['A'] - abg.A_bar
        expected = np.zeros_like(offset)
        idx = np.arange(abg.x_dim - 1)
        expected[idx, idx + 1] = offset[idx, idx + 1]
        np.testing.assert_array_equal(offset, expected)
        assert np.all(np.abs(offset) <= abg.alphas['A'] + 1e-15)

        # B: only B[0, -1] may move, and only within alphas['B']
        b_offset = d['B'] - abg.B_bar
        assert abs(b_offset[0, -1]) <= abg.alphas['B'] + 1e-15
        b_offset[0, -1] = 0.0
        np.testing.assert_array_equal(b_offset, np.zeros_like(b_offset))

        # mu: the **squared** radius is alphas['mu'], and exhaustible per component
        mu_a = (d['mu'] - abg.mu_bar).reshape(-1, 1)
        assert np.sum(mu_a ** 2) <= abg.alphas['mu'] + 1e-12

        # Sigma: derived from the mu draw, and still a covariance
        expected_sigma = abg.alphas['Sigma'] * abg.Sigma_bar - mu_a @ mu_a.T
        np.testing.assert_allclose(d['Sigma'], expected_sigma, rtol=0, atol=0)
        assert np.linalg.eigvalsh((d['Sigma'] + d['Sigma'].T) / 2).min() > -1e-12


def test_nonlinear_samples_stay_inside_the_declared_set(nonlinear_set):
    abg = nonlinear_set
    np.random.seed(29)
    for _ in range(25):
        d = abg.sample()
        # the drift is sampled uniformly inside the ball of radius f_err_max
        assert np.linalg.norm(d['delta_f']) <= abg.f_err_max + 1e-12
        # mu is drawn from a box of half-width sqrt(gamma[1]) in **every** component, so the
        # bound on the norm is d * gamma[1] rather than gamma[1]
        mu_a = d['mu'] - abg.mu_bar
        assert np.all(np.abs(mu_a) <= np.sqrt(abg.gamma[1]) + 1e-12)
        assert np.sum(mu_a ** 2) <= abg.x_dim * abg.gamma[1] + 1e-12
        # the rejection sampler only ever returns positive-definite covariances
        assert np.linalg.eigvalsh(d['Sigma']).min() > 0


def test_sampling_does_not_mutate_the_ambiguity_set(linear_set):
    """``sample()`` must not write through to the nominal parameters.

    ``B = self.B_bar`` instead of ``B = self.B_bar.copy()`` would alias, and since
    ``sample_trajectory`` calls ``sample()`` once per period the nominal ``B_bar`` would take a
    random walk -- silently, and permanently.
    """
    abg = linear_set
    before_A, before_B = abg.A_bar.copy(), abg.B_bar.copy()
    mu_bar, Sigma_bar = abg.mu_bar.copy(), abg.Sigma_bar.copy()
    np.random.seed(23)
    for _ in range(20):
        abg.sample()
    np.testing.assert_array_equal(abg.A_bar, before_A)
    np.testing.assert_array_equal(abg.B_bar, before_B)
    np.testing.assert_array_equal(abg.mu_bar, mu_bar)
    np.testing.assert_array_equal(abg.Sigma_bar, Sigma_bar)


# ----------------------------------------------------------------------
# worst_case_params()
# ----------------------------------------------------------------------

@pytest.mark.parametrize('factory', [linear_ambiguity_set, nonlinear_ambiguity_set])
def test_worst_case_params_are_accepted_by_validate(factory):
    """Whatever the adversarial path produces must be legal ``sds_params`` for that set.

    The returned dict carries one extra key, ``'j'`` -- the index of the worst direction, which
    is diagnostic rather than part of the parameterization. ``EigvecFixedAdversarialDynamics``
    pops it before handing the parameters to ``Env.step``, which is what this reproduces; note
    that ``Env.step`` would reject a leftover ``'j'``, so the pop is load-bearing.
    """
    abg = factory()
    lamb_pred = np.array([1.0, 2.0])[:abg.x_dim]
    params = abg.worst_case_params(np.array([0.3, 0.0]), np.array([0.1]), lamb_pred)
    assert set(params) == set(abg.param_keys) | {'j'}
    j = params.pop('j')
    assert 0 <= j < abg.x_dim
    abg.validate(params)


def test_worst_case_covariance_is_the_shrunk_nominal_one(linear_set):
    """``Sigma = gamma[2]*Sigma_bar - gamma[1]*lamb[j]*v_j v_j^T`` -- shared by both families."""
    abg = linear_set
    lamb_pred = np.array([0.5, 0.9])
    params = abg.worst_case_params(np.array([1.0, 1.0]), np.array([0.0]), lamb_pred)
    j = params['j']
    lamb, v = np.linalg.eig(abg.Sigma_bar)
    lamb, v = lamb.real, v.real
    v_j = v[:, j:j + 1]
    expected = abg.gamma[2] * abg.Sigma_bar - abg.gamma[1] * lamb[j] * v_j @ v_j.T
    np.testing.assert_allclose(params['Sigma'], expected, rtol=1e-12, atol=1e-12)
