"""Contract tests for the three DRPP predictors.

All three share the private ``_ConicMomentPredictor`` skeleton, and their only substantive
difference is where the predictive covariance comes from -- so that is what most of this file
pins down, one experiment per predictor.
"""

from __future__ import annotations

import numpy as np
import pytest

from _systems import linear_ambiguity_set
from ppsds.numerics.linalg import eig_real
from ppsds.probabilistic_predictor.base import ProbabilisticPredictor
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (
    EigDrpp,
    NominalPredictor,
    NoiseDrpp,
)

X0 = np.array([2.0, 1.0])
U0 = np.array([0.1])

ALL_PREDICTORS = ('nominal', 'noise_drpp', 'eig_drpp')


def build(name, abg):
    """Construct one predictor, taking every model quantity off the ambiguity set."""
    if name == 'nominal':
        return NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar)
    if name == 'noise_drpp':
        return NoiseDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma)
    if name == 'eig_drpp':
        return EigDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma)
    raise AssertionError(f'unknown predictor {name!r}')


@pytest.mark.parametrize('name', ALL_PREDICTORS)
def test_outputs_start_empty_and_are_filled_by_update(name):
    pred = build(name, linear_ambiguity_set())
    assert pred.x_pred is None
    assert pred.Sigma_pred is None
    assert pred.lamb_pred is None

    pred.update(X0, U0)
    assert pred.x_pred.shape == (2,)
    assert pred.lamb_pred.shape == (2,)
    assert pred.Sigma_pred.shape == (2, 2)
    for attr in ('x_pred', 'Sigma_pred', 'lamb_pred'):
        assert np.all(np.isfinite(getattr(pred, attr)))


@pytest.mark.parametrize('name', ALL_PREDICTORS)
def test_predictive_mean_is_the_nominal_drift_plus_the_noise_mean(name):
    abg = linear_ambiguity_set()
    pred = build(name, abg)
    pred.update(X0, U0)
    np.testing.assert_array_equal(pred.x_pred, abg.f_bar(X0, U0) + abg.mu_bar)


@pytest.mark.parametrize('name', ALL_PREDICTORS)
def test_predict_returns_the_documented_distribution(name):
    """Pins the predictive law and the log density's normalization.

    The density at the mean is ``-1/2 * (d*log(2*pi) + log det Sigma)``. Note the absence of a
    ``+d`` term here: that term belongs to the **expected** score, which is the form the
    theoretical bounds (and ``EigDrpp.val_wst``) use. Mixing the two up shifts everything by
    ``d/2``, so it is worth an explicit assertion.
    """
    abg = linear_ambiguity_set()
    pred = build(name, abg)
    pred.update(X0, U0)

    dist = pred.predict()
    np.testing.assert_allclose(dist.mean, pred.x_pred, rtol=0, atol=0)
    np.testing.assert_allclose(dist.cov, pred.Sigma_pred, rtol=0, atol=0)

    d = abg.x_dim
    expected = -0.5 * (d * np.log(2 * np.pi) + np.log(np.linalg.det(pred.Sigma_pred)))
    assert dist.logpdf(pred.x_pred) == pytest.approx(expected, rel=1e-10)


def test_nominal_predictor_uses_the_nominal_covariance_verbatim():
    abg = linear_ambiguity_set()
    pred = NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar)
    pred.update(X0, U0)
    np.testing.assert_array_equal(pred.Sigma_pred, abg.Sigma_bar)
    np.testing.assert_array_equal(pred.lamb_pred, eig_real(abg.Sigma_bar)[0])


def test_noise_drpp_inflates_the_covariance_by_gamma2():
    """The inflation is computed as a product **before** the eigendecomposition.

    That order is not arbitrary: ``eig_real(g2 * Sigma_bar)`` and
    ``g2 * eig_real(Sigma_bar)`` may differ in their last bits, and ``lamb_pred`` is fed to the
    adversarial dynamics, where a last-bit difference gets amplified. The assertion below, with
    ``assert_array_equal``, is what keeps that order from being "simplified" away.
    """
    abg = linear_ambiguity_set()
    pred = NoiseDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma)
    pred.update(X0, U0)
    np.testing.assert_array_equal(pred.Sigma_pred, abg.gamma[2] * abg.Sigma_bar)
    np.testing.assert_array_equal(pred.lamb_pred, eig_real(abg.gamma[2] * abg.Sigma_bar)[0])


def test_eig_drpp_rebuilds_its_covariance_in_the_fixed_eigenbasis():
    """``Sigma_pred = v @ diag(lamb_pred) @ v.T``, with ``v`` fixed to ``Sigma_bar``'s
    eigenvectors -- that fixed basis is what "eigenvectors fixed" in the method's name means."""
    abg = linear_ambiguity_set()
    pred = EigDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma)
    pred.update(X0, U0)

    _, v = eig_real(abg.Sigma_bar)
    np.testing.assert_allclose(pred.Sigma_pred, v @ np.diag(pred.lamb_pred) @ v.T,
                               rtol=1e-12, atol=1e-12)
    assert isinstance(pred.j, int) and 0 <= pred.j < abg.x_dim
    assert np.isfinite(pred.val_wst)


@pytest.mark.parametrize('name', ALL_PREDICTORS)
def test_reset_restores_a_reusable_state(name):
    abg = linear_ambiguity_set()
    pred = build(name, abg)
    pred.update(X0, U0)
    pred.reset()

    for attr in ('x_pred', 'Sigma_pred', 'lamb_pred'):
        assert getattr(pred, attr) is None, f'{attr} survived reset()'

    expected_Sigma_bar = (abg.gamma[2] * abg.Sigma_bar) if name == 'noise_drpp' else abg.Sigma_bar
    np.testing.assert_array_equal(pred.mu_bar, abg.mu_bar)
    np.testing.assert_array_equal(pred.Sigma_bar, expected_Sigma_bar)
    np.testing.assert_array_equal(pred.lamb, eig_real(expected_Sigma_bar)[0])

    pred.update(X0, U0)
    assert np.all(np.isfinite(pred.lamb_pred)), 'the instance is not reusable after reset()'


def test_multi_step_mode_reuses_the_previous_prediction_and_input():
    """``update(None, None)`` extrapolates: the previous *prediction* becomes the state."""
    abg = linear_ambiguity_set()
    pred = NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar)
    pred.update(X0, U0)
    first = pred.x_pred.copy()

    pred.update()
    np.testing.assert_array_equal(pred.u, U0, 'the previous input must be reused')
    np.testing.assert_array_equal(pred.x, first, 'the previous prediction becomes the state')
    np.testing.assert_array_equal(pred.x_pred, abg.f_bar(first, U0) + abg.mu_bar)


def test_multi_step_mode_before_any_prediction_fails_loudly():
    abg = linear_ambiguity_set()
    pred = NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar)
    with pytest.raises(ValueError):
        pred.update()


@pytest.mark.parametrize('name', ALL_PREDICTORS)
def test_predictors_satisfy_the_contract_by_structure_only(name):
    """``ProbabilisticPredictor`` is a ``typing.Protocol``, and the concrete classes
    deliberately do **not** inherit from it.

    The repository tried an ABC once: it declared the attributes as ``@property`` (data
    descriptors) while the implementing classes assign plain attributes, so genuinely
    inheriting it would raise ``AttributeError`` at construction time -- and
    ``ABCMeta.register()`` hid exactly that. If someone reintroduces inheritance, this test
    fails and points at the docstring in ``ppsds/probabilistic_predictor/base.py``.
    """
    pred = build(name, linear_ambiguity_set())
    for method in ('predict', 'update', 'reset'):
        assert callable(getattr(pred, method)), f'{method} missing'
    for attr in ('x_pred', 'Sigma_pred', 'lamb_pred'):
        assert hasattr(pred, attr), f'{attr} missing'
    assert ProbabilisticPredictor not in type(pred).__mro__, \
        'the predictors must satisfy the Protocol structurally, not by inheritance'
