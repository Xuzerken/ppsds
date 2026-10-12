"""Seeded numerical regression against the committed ``golden.json``.

The scenario lives in :mod:`_golden`; read its docstring for why the assertions are split into
"sorted eigenvalues within a relative tolerance" and "invariants exactly".

A failure here means the numbers moved. That is not automatically a bug -- but it does mean the
change must be explained, and if it is intended, ``golden.json`` must be regenerated in the same
commit so that the diff of the fixture is part of the review.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import _golden
from _systems import linear_ambiguity_set
from ppsds.numerics.linalg import eig_real
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (
    EigDrpp,
    NoiseDrpp,
    eigvec_fixed_minmax,
)

FIXTURE = Path(__file__).resolve().parent / 'golden.json'

#: Relative tolerance on eigenvalues and the predicted state. Loose enough to survive a flipped
#: near-tie in the discrete choice of ``j`` (which moves the score by ~1e-9) and ordinary
#: solver/platform noise; tight enough that a real change to the optimization cannot hide.
RTOL = 1e-6

#: Absolute tolerance on the log scores, which sit around -3.
ATOL = 1e-6

#: Absolute tolerance for quantities this suite recomputes through the *same* code path as the
#: fixture, where only floating-point non-determinism is expected.
ORACLE_ATOL = 1e-9


@pytest.fixture(scope='module')
def recorded():
    return _golden.deserialize(json.loads(FIXTURE.read_text(encoding='utf-8')))


@pytest.fixture(scope='module')
def actual():
    """Run the scenario once for the whole module (the cone programs cost ~0.4 s in total)."""
    return _golden.compute()


def test_scenario_metadata_is_pinned(recorded, actual):
    assert recorded['seed'] == actual['seed'] == _golden.SEED
    assert recorded['steps'] == actual['steps'] == _golden.STEPS


@pytest.mark.parametrize('name', _golden.PREDICTOR_NAMES)
def test_predictions_match_the_fixture(recorded, actual, name):
    """The predictive covariance, its eigenvalues, the predictive mean and the score."""
    rec, act = recorded['predictors'][name], actual['predictors'][name]
    assert len(rec['score']) == _golden.STEPS

    for i in range(_golden.STEPS):
        # Eigenvalues first, sorted: a near-tie in `j` permutes them without changing the
        # covariance as a matrix, so comparing them unsorted would be a flaky assertion.
        np.testing.assert_allclose(
            np.sort(act['lamb_pred'][i]), np.sort(rec['lamb_pred'][i]),
            rtol=RTOL, atol=0, err_msg=f'{name}: lamb_pred moved at step {i}')
        np.testing.assert_allclose(
            act['Sigma_pred'][i], rec['Sigma_pred'][i],
            rtol=RTOL, atol=0, err_msg=f'{name}: Sigma_pred moved at step {i}')
        np.testing.assert_allclose(
            act['x_pred'][i], rec['x_pred'][i],
            rtol=RTOL, atol=0, err_msg=f'{name}: x_pred moved at step {i}')
        assert act['score'][i] == pytest.approx(rec['score'][i], rel=RTOL, abs=ATOL), \
            f'{name}: score moved at step {i}'


@pytest.mark.parametrize('name', _golden.PREDICTOR_NAMES)
def test_predictive_covariances_are_valid_distributions(actual, name):
    """A log score is only defined for a genuine density, so these have to hold.

    ⚠️ **Symmetry holds up to rounding, and that is all that can be asked for.** Eig-DRPP's
    predictive covariance is rebuilt as ``v @ diag(lamb) @ v.T``, which is symmetric in exact
    arithmetic but not bit-for-bit: the two off-diagonal entries are sums accumulated in a
    different order. This assertion originally demanded ``atol=0`` and passed on the development
    machine while failing on every CI runner by exactly 1 ULP (2.2e-16) -- a test that decided
    by BLAS rather than by code, and the first thing the CI ever caught.

    The tolerance is ``1e-12`` relative to the largest entry: four orders of magnitude above the
    rounding it must tolerate, and far below a genuine asymmetry (a transposed index moves these
    entries by their whole value, not by their last bits).
    """
    for i, Sigma in enumerate(actual['predictors'][name]['Sigma_pred']):
        assert np.all(np.isfinite(Sigma)), f'{name}: non-finite covariance at step {i}'
        scale = max(1.0, float(np.abs(Sigma).max()))
        np.testing.assert_allclose(
            Sigma, Sigma.T, rtol=0, atol=1e-12 * scale,
            err_msg=f'{name}: covariance is not symmetric at step {i}')
        assert np.linalg.eigvalsh(Sigma).min() > 0, \
            f'{name}: covariance not positive definite at step {i}'
        assert np.all(np.isfinite(actual['predictors'][name]['x_pred'][i]))


def test_eig_drpp_worst_direction_is_a_valid_index(actual):
    for i, j in enumerate(actual['eig_drpp_j']):
        assert 0 <= j < actual['x_obs'][i].shape[0], f'j out of range at step {i}'


def test_eig_drpp_matches_the_cone_program_oracle(recorded):
    """Eig-DRPP must feed ``eigvec_fixed_minmax`` exactly the nominal eigenvalues and the
    observed state and input.

    This is the check that a golden comparison cannot give: if the predictor started passing
    ``lamb_pred`` instead of the nominal ``lamb``, or swapped ``x`` for ``u``, the numbers would
    still be self-consistent from run to run -- but they would no longer be the numbers the
    fixture was built from, so re-running the cone program on the *fixture's own inputs* must
    reproduce the fixture's outputs.
    """
    abg = linear_ambiguity_set()
    lamb_nominal, _ = eig_real(abg.Sigma_bar)

    for i in range(recorded['steps']):
        lamb_opt, val_opt, j_opt = eigvec_fixed_minmax(
            lamb_nominal, abg.gamma, recorded['x_obs'][i], recorded['u_obs'][i])

        assert val_opt == pytest.approx(recorded['eig_drpp_val_wst'][i], abs=ORACLE_ATOL), \
            f'worst-case value disagrees with the cone program at step {i}'
        np.testing.assert_allclose(
            np.sort(lamb_opt),
            np.sort(recorded['predictors']['eig_drpp']['lamb_pred'][i]),
            rtol=RTOL, atol=0,
            err_msg=f'eigenvalues disagree with the cone program at step {i}')
        assert j_opt == recorded['eig_drpp_j'][i], \
            f'worst direction disagrees with the cone program at step {i}'


def test_eig_drpp_inflates_the_prediction(recorded):
    """For this fixture, Eig-DRPP buys spread: the covariance carries more volume than nominal.

    Deliberately **not** a componentwise / Loewner-order assertion, and deliberately scoped to
    this fixture's parameters. The cone constraint ``coefficient[j]*x_j >= coefficient[i]*x_i``
    permits coordinates in which ``lamb_pred < gamma[2]*lamb``; probing 360 randomized parameter
    draws of this same family turned up 34 such cases, so "Eig-DRPP dominates ``gamma[2]*Sigma_bar``
    entrywise" is simply false, and a test asserting it would pass only by accident here.
    """
    for i in range(recorded['steps']):
        lamb_eig = recorded['predictors']['eig_drpp']['lamb_pred'][i]
        lamb_nom = recorded['predictors']['nominal']['lamb_pred'][i]
        assert np.prod(lamb_eig) > np.prod(lamb_nom), \
            f'Eig-DRPP lost volume against the nominal covariance at step {i}'


def test_zero_drift_radius_collapses_eig_drpp_onto_noise_drpp():
    """With ``f_err_max = 0`` there is no drift uncertainty, and the cone program has a
    closed-form answer.

    The coefficients ``coefficient[i] = 2*sqrt(z*gamma[1]*lamb[i]) + z`` all vanish when the
    drift radius ``z`` is 0, so the constraints become trivial and the objective is minimized at
    ``x = 1/(gamma[2]*lamb)``. Three things must then hold exactly (to solver tolerance):

    - the predictive covariance is ``gamma[2] * Sigma_bar`` -- i.e. Eig-DRPP degenerates to
      Noise-DRPP rather than to something merely close to it;
    - the predictive mean is unchanged (robustification touches only the covariance);
    - ``val_wst`` equals ``value_upper_bound_true_transition()``, because with no drift
      uncertainty the worst case *is* the nominal one -- and that bound is what pins the
      objective's normalization, which is the easiest thing here to get wrong by a factor or an
      additive term.

    Note what ``val_wst`` is: an **expected** log score, i.e. ``-(d*log2pi + d + log det Sigma)/2``.
    It is not the log density at the mean (that one has no ``+d``) and so is not directly
    comparable with ``LogarithmScore.score`` on a single state; the two differ by ``d/2``.
    """
    abg = linear_ambiguity_set(f_err_max=0.0)
    gamma2 = abg.gamma[2]
    pred_eig = EigDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma)
    pred_noise = NoiseDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma)

    x, u = np.array([2.0, 1.0]), np.array([0.1])
    pred_eig.update(x, u)
    pred_noise.update(x, u)

    np.testing.assert_allclose(
        pred_eig.Sigma_pred, gamma2 * abg.Sigma_bar, rtol=1e-6, atol=1e-9,
        err_msg='Eig-DRPP did not collapse onto gamma[2] * Sigma_bar')
    np.testing.assert_allclose(
        pred_eig.Sigma_pred, pred_noise.Sigma_pred, rtol=1e-6, atol=1e-9,
        err_msg='Eig-DRPP and Noise-DRPP disagree with no drift uncertainty')
    np.testing.assert_allclose(pred_eig.x_pred, pred_noise.x_pred, rtol=0, atol=0,
                               err_msg='robustification changed the predictive mean')
    assert pred_eig.val_wst == pytest.approx(
        abg.value_upper_bound_true_transition(), rel=1e-6, abs=1e-6), \
        'with no drift uncertainty the worst case must attain the upper bound'
