"""The discrete predictive law: a pmf whose support need not contain the truth's.

The load-bearing test here is that the vectorised box integral agrees with a naive
atom-by-atom one. The fast path builds a boolean matrix and contracts it; the reference just
loops. They must give the same numbers, since the whole point of a pmf predictor is that its
score is the weight that landed inside the box.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ppsds.numerics.atom_pmf import AtomPmf
from ppsds.scoring_rule.epsilon_logarithm_score import EpsilonLogarithmScore
from ppsds.scoring_rule.logarithm_score import LogarithmScore


def _naive_box_integral(atoms, weights, x, eps):
    """``sum of weights within the L-infinity box``, written the obvious slow way."""
    return sum(
        weight for atom, weight in zip(atoms, weights) if np.max(np.abs(atom - x)) <= eps
    )


def _scattered_pmf(seed=0, n_atoms=12):
    rng = np.random.default_rng(seed)
    return AtomPmf(rng.uniform(-2, 2, size=(n_atoms, 2)), rng.random(n_atoms) + 0.1)


def test_weights_are_normalised_on_construction():
    """Raw stakes or counts are accepted; the law uses the normalised weights."""
    pmf = AtomPmf(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([2.0, 6.0]))
    assert pmf.weights == pytest.approx(np.array([0.25, 0.75]))
    assert pmf.weights.sum() == pytest.approx(1.0)


def test_box_integral_agrees_with_the_naive_loop():
    """The vectorised box integral must equal the atom-by-atom one, to the last bit."""
    pmf = _scattered_pmf()
    rng = np.random.default_rng(1)
    outcomes = rng.uniform(-3, 3, size=(200, 2))
    for eps in (0.25, 0.5, 1.0):
        fast = pmf.box_integral(outcomes, eps)
        slow = np.array([_naive_box_integral(pmf.atoms, pmf.weights, x, eps) for x in outcomes])
        assert fast == pytest.approx(slow, rel=1e-15, abs=1e-15), f'eps={eps}'


def test_box_integral_of_a_single_outcome_matches_the_batch():
    """``(d_x,)`` and ``(1, d_x)`` are the same query."""
    pmf = _scattered_pmf()
    point = np.array([0.4, -0.6])
    batch = pmf.box_integral(point[None, :], 0.5)
    assert batch.shape == (1,), 'a (1, d_x) query is a batch of one, not a single outcome'
    assert pmf.box_integral(point, 0.5) == pytest.approx(batch[0])


def test_a_pmf_whose_support_misses_the_truth_still_scores():
    """The reason the neighbourhood exists: no atom sits on the outcome, yet the score is finite.

    With the ordinary logarithm score the same predictor is ``-inf``; with a neighbourhood it
    scores the weight it put close enough, which is what a player placing a few bets gets.
    """
    pmf = AtomPmf(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0.5, 0.5]))
    outcome = np.array([0.1, 0.0])          # not an atom, and not the mean either

    assert EpsilonLogarithmScore(0.25).score(pmf, outcome) == pytest.approx(math.log(0.5))
    assert LogarithmScore().score(pmf, outcome) == -np.inf


def test_the_score_uses_the_box_integral_path():
    """A pmf is scored through ``box_integral``, not through a CDF it does not have."""
    pmf = _scattered_pmf()
    outcome = np.array([0.2, 0.2])
    eps = 0.5
    assert EpsilonLogarithmScore(eps).score(pmf, outcome) == pytest.approx(
        math.log(pmf.box_integral(outcome, eps))
    )


def test_disjoint_bets_score_minus_infinity():
    """Weight that all sits outside the box is worth nothing -- ``log 0``."""
    pmf = AtomPmf(np.array([[10.0, 10.0]]), np.array([1.0]))
    assert EpsilonLogarithmScore(0.5).score(pmf, np.array([0.0, 0.0])) == -np.inf


def test_entropy_is_the_shannon_entropy_of_the_weights():
    """A single atom is certain (0 nats); four equal atoms are 2 bits' worth (``log 4``)."""
    assert AtomPmf(np.array([[0.0, 0.0]]), np.array([1.0])).entropy() == pytest.approx(0.0)
    square = AtomPmf(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.ones(4))
    assert square.entropy() == pytest.approx(math.log(4.0))


def test_moments_are_the_weighted_ones():
    """``mean`` / ``cov`` summarise the law; the environment layer reads them as moments."""
    pmf = AtomPmf(np.array([[0.0, 0.0], [2.0, 0.0]]), np.array([0.25, 0.75]))
    assert pmf.mean == pytest.approx(np.array([1.5, 0.0]))
    assert pmf.cov == pytest.approx(np.array([[0.75, 0.0], [0.0, 0.0]]))
    assert pmf.dimensions == 2


def test_rvs_draws_atoms_and_is_reproducible_from_an_rng():
    """Samples land on atoms, with the documented shape, reproducibly from a ``Generator``."""
    pmf = _scattered_pmf()
    draws = pmf.rvs(size=50, rng=np.random.default_rng(3))
    assert draws.shape == (50, 2)
    assert np.all([np.any(np.all(np.isclose(atom, pmf.atoms), axis=1)) for atom in draws])
    again = pmf.rvs(size=50, rng=np.random.default_rng(3))
    assert np.array_equal(draws, again)


@pytest.mark.parametrize(
    'atoms, weights, message',
    [
        (np.zeros((3, 2)), np.ones(4), 'weights'),
        (np.zeros((3, 2)), np.array([1.0, -1.0, 1.0]), 'non-negative'),
        (np.zeros((3, 2)), np.zeros(3), 'positive'),
    ],
)
def test_a_law_that_is_not_a_law_is_rejected(atoms, weights, message):
    """Structural mistakes fail at construction rather than producing a silent non-law."""
    with pytest.raises(ValueError, match=message):
        AtomPmf(atoms, weights)


def test_a_query_of_the_wrong_dimension_is_rejected():
    """A 3-vector against a 2-D law is a mistake, not something to broadcast."""
    with pytest.raises(ValueError, match='coordinate'):
        _scattered_pmf().box_integral(np.zeros((2, 3)), 0.5)


@pytest.mark.parametrize('bad', (-0.1, np.inf, np.nan))
def test_a_radius_that_is_not_a_radius_is_rejected(bad):
    """The same radius guard the scoring rule applies."""
    with pytest.raises(ValueError, match='radius'):
        _scattered_pmf().box_integral(np.zeros(2), bad)
