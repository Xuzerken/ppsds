"""Predictors that predict with a pmf you supply.

These mirror the DRPP predictors' cycle -- ``predict`` / ``update`` / ``reset``, a multi-step
fallback, moment summaries alongside the law -- but the law is handed in rather than optimized
for. The tests therefore concentrate on the two things that can go wrong: the prediction
drifting away from the pmf that was supplied, and the summary attributes disagreeing with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppsds.numerics.atom_pmf import AtomPmf
from ppsds.probabilistic_predictor.base import ProbabilisticPredictor
from ppsds.probabilistic_predictor.pmf.atom import (
    DriftShiftedAtomPmfPredictor,
    StaticAtomPmfPredictor,
)

PREDICTORS = (StaticAtomPmfPredictor, DriftShiftedAtomPmfPredictor)


def _pmf():
    return AtomPmf(np.array([[0.0, 0.0], [2.0, 1.0]]), np.array([0.25, 0.75]))


def _drift(x, u):
    """A shear drift, so the shifted atoms cannot be confused with the originals."""
    return np.array([x[0] + x[1], x[1]]) + np.array([u[0], 0.0])


def _build(cls):
    return cls(_pmf()) if cls is StaticAtomPmfPredictor else cls(_pmf(), _drift)


@pytest.mark.parametrize('cls', PREDICTORS)
def test_the_predictor_satisfies_the_contract_by_structure_only(cls):
    """Same structural-Protocol discipline as the DRPP predictors: no inheritance."""
    predictor = _build(cls)
    assert ProbabilisticPredictor not in type(predictor).__mro__
    for name in ('predict', 'update', 'reset'):
        assert callable(getattr(predictor, name)), name
    for name in ('x_pred', 'Sigma_pred', 'lamb_pred'):
        assert hasattr(predictor, name), name
        assert getattr(predictor, name) is None, f'{name} should start empty'


@pytest.mark.parametrize('cls', PREDICTORS)
def test_predicting_before_any_update_fails_loudly(cls):
    """There is no law yet, and an obscure downstream failure is what silence would buy."""
    with pytest.raises(ValueError, match='before update'):
        _build(cls).predict()


@pytest.mark.parametrize('cls', PREDICTORS)
def test_the_moment_summaries_describe_the_predicted_law(cls):
    """``x_pred`` / ``Sigma_pred`` / ``lamb_pred`` are the law's moments, not a second opinion."""
    predictor = _build(cls)
    predictor.update(np.array([1.0, 0.5]), np.array([0.0]))
    law = predictor.predict()

    assert predictor.x_pred == pytest.approx(law.mean)
    assert predictor.Sigma_pred == pytest.approx(law.cov)
    # Sorted: `lamb_pred` comes from the shared `eig_real` entry point, which does not order its
    # output, while `eigvalsh` returns ascending. Same set of values either way, and here the
    # order carries no meaning -- unlike in the DRPP predictors, where the adversarial dynamics
    # picks a worst *direction* by index.
    assert np.sort(predictor.lamb_pred) == pytest.approx(np.sort(np.linalg.eigvalsh(law.cov)))


def test_a_static_predictor_returns_the_supplied_pmf_every_step():
    """The atoms you place are where you say they are: no drift, no adaptation."""
    pmf = _pmf()
    predictor = StaticAtomPmfPredictor(pmf)
    for state in (np.array([0.0, 0.0]), np.array([5.0, -3.0])):
        predictor.update(state, np.array([0.0]))
        assert predictor.predict() is pmf, 'the very same object, not a copy that could drift'


def test_a_drift_shifted_predictor_moves_the_atoms_by_the_drift():
    """The atoms live in the noise frame, so predicting adds the drift to each of them."""
    pmf = _pmf()
    predictor = DriftShiftedAtomPmfPredictor(pmf, _drift)
    state, control = np.array([1.0, 0.5]), np.array([0.25])
    predictor.update(state, control)

    expected_shift = _drift(state, control)
    assert predictor.predict().atoms == pytest.approx(pmf.atoms + expected_shift)
    assert predictor.x_pred == pytest.approx(pmf.mean + expected_shift)
    assert predictor.Sigma_pred == pytest.approx(pmf.cov), 'a shift must not change the spread'


def test_the_supplied_pmf_is_not_mutated_by_a_shifted_prediction():
    """The noise-frame law is the caller's object and must survive the rollout untouched."""
    pmf = _pmf()
    before = pmf.atoms.copy()
    predictor = DriftShiftedAtomPmfPredictor(pmf, _drift)
    predictor.update(np.array([3.0, 3.0]), np.array([0.0]))
    assert pmf.atoms == pytest.approx(before)


@pytest.mark.parametrize('cls', PREDICTORS)
def test_multi_step_mode_carries_the_previous_prediction_forward(cls):
    """``update()`` with no arguments extrapolates, as the DRPP predictors' cycle does."""
    predictor = _build(cls)
    predictor.update(np.array([1.0, 0.0]), np.array([0.0]))
    first = predictor.predict()
    predictor.update()  # no observation: take the last prediction as the new state
    second = predictor.predict()
    assert second.atoms.shape == first.atoms.shape
    if cls is StaticAtomPmfPredictor:
        assert second.atoms == pytest.approx(first.atoms)


@pytest.mark.parametrize('cls', PREDICTORS)
def test_reset_clears_the_per_trajectory_state_but_keeps_the_law(cls):
    """A second trajectory must not inherit the first one's state."""
    predictor = _build(cls)
    predictor.update(np.array([1.0, 1.0]), np.array([0.0]))
    predictor.reset()
    assert predictor.x_pred is None and predictor.Sigma_pred is None
    assert predictor.lamb_pred is None
    with pytest.raises(ValueError, match='before update'):
        predictor.predict()
    predictor.update(np.array([0.0, 0.0]), np.array([0.0]))  # reusable straight away
    assert predictor.predict() is not None


@pytest.mark.parametrize('cls', PREDICTORS)
def test_a_misshaped_state_is_rejected(cls):
    """The same silent-broadcast guard the DRPP predictors use."""
    predictor = _build(cls)
    with pytest.raises(ValueError, match='x_observe'):
        predictor.update(np.zeros((1, 2)), np.array([0.0]))


def test_a_predictor_needs_an_atom_pmf():
    """Handing it something else is a mistake worth naming."""
    with pytest.raises(TypeError, match='AtomPmf'):
        StaticAtomPmfPredictor(np.zeros((3, 2)))


def test_a_drift_shifted_predictor_needs_a_drift():
    """Without a drift there is nothing to shift by, and defaulting to zero would hide that."""
    with pytest.raises(TypeError, match='f_bar'):
        DriftShiftedAtomPmfPredictor(_pmf(), None)


def test_a_misshaped_drift_is_rejected():
    """A wrong-shaped drift would broadcast against the atoms and misplace every one of them."""
    predictor = DriftShiftedAtomPmfPredictor(_pmf(), lambda x, u: np.zeros((2, 1)))
    with pytest.raises(ValueError, match='f_bar returned shape'):
        predictor.update(np.zeros(2), np.array([0.0]))
