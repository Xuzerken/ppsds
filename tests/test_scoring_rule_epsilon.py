"""The epsilon-logarithm scoring rule, path by path and against itself.

:meth:`~ppsds.scoring_rule.epsilon_logarithm_score.EpsilonLogarithmScore.score` picks a
different algorithm depending on the law it is handed, and the algorithms must agree -- a
dispatched fast path that quietly disagrees with the general one is exactly the kind of bug
that shows up later as an unexplained line on a figure. So the closed form for a diagonal
Gaussian is checked *against* the general inclusion-exclusion, not against a stored constant.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from ppsds.numerics.random_variables import multivariate_uniform
from ppsds.scoring_rule.epsilon_logarithm_score import EpsilonLogarithmScore

EPSILONS = (0.25, 0.5, 0.75)


def _diagonal_gaussian():
    """A Gaussian whose covariance is diagonal -- the case the closed-form path handles."""
    return multivariate_normal(mean=np.array([0.3, -0.7]), cov=np.diag([0.5, 2.0]))


def _query_points():
    return np.random.default_rng(0).normal(size=(6, 2))


def test_closed_form_agrees_with_inclusion_exclusion():
    """The diagonal-Gaussian fast path must equal the ``2 ** d_x``-corner general one.

    These are two independent derivations of the same box integral; they agree to floating-point
    noise, so any disagreement is a real defect rather than a tolerance question.
    """
    law, points = _diagonal_gaussian(), _query_points()
    for eps in EPSILONS:
        scorer = EpsilonLogarithmScore(eps)
        fast = scorer._box_probability(law, points)
        general = scorer._cdf_inclusion_exclusion(law.cdf, points)
        assert fast == pytest.approx(general, rel=1e-12, abs=1e-14), f'eps={eps}'


def test_a_batch_matches_point_by_point():
    """``(n, d)`` input scores the same as ``n`` separate ``(d,)`` calls."""
    scorer, law, points = EpsilonLogarithmScore(0.5), _diagonal_gaussian(), _query_points()
    assert scorer.score(law, points) == pytest.approx([scorer.score(law, p) for p in points])


def test_call_is_score():
    """``__call__`` is the same entry point, so a scorer can be used like a function."""
    scorer, law, points = EpsilonLogarithmScore(0.5), _diagonal_gaussian(), _query_points()
    assert scorer(law, points) == pytest.approx(scorer.score(law, points))


def test_one_dimensional_value_matches_the_textbook_difference():
    """In 1-D the box integral is a plain CDF difference, which pins down the corner signs."""
    scorer = EpsilonLogarithmScore(0.5)
    unit = multivariate_normal(mean=[0.0], cov=[[1.0]])
    expected = math.log(norm.cdf(0.5) - norm.cdf(-0.5))
    assert scorer.score(unit, np.array([0.0])) == pytest.approx(expected)


def test_eps_zero_is_the_ordinary_logarithm_score():
    """At exactly ``eps = 0`` the rule must hand back ``logpdf`` and nothing else."""
    law, points = _diagonal_gaussian(), _query_points()
    assert EpsilonLogarithmScore(0.0).score(law, points) == pytest.approx(law.logpdf(points))


def test_subtracting_the_box_volume_recovers_the_logarithm_score():
    """``L_eps - d_x log(2 eps) -> log p(x)`` as ``eps -> 0``.

    The score itself does *not* converge to the ordinary logarithm score -- it runs off to
    ``-inf``, because the box integral shrinks like ``(2 eps) ** d_x``. Subtracting that volume
    is what recovers it, and it is why the published formulas carry an ``I_{eps > 0}``
    indicator rather than taking a limit.
    """
    law, points = _diagonal_gaussian(), _query_points()
    eps = 1e-4
    corrected = EpsilonLogarithmScore(eps).score(law, points) - 2 * math.log(2 * eps)
    assert corrected == pytest.approx(law.logpdf(points), abs=1e-3)


def test_a_predictor_with_no_mass_near_the_outcome_scores_minus_infinity():
    """Zero box probability is ``-inf``, silently -- that *is* the score.

    A uniform predictor on ``[0, 1] ** 2`` evaluated at a point more than ``eps`` outside it
    puts no mass in the box at all.
    """
    law = multivariate_uniform(low=np.zeros(2), high=np.ones(2))
    assert EpsilonLogarithmScore(0.5).score(law, np.array([5.0, 5.0])) == -np.inf


def test_a_uniform_law_scores_through_the_cdf_path():
    """A law with no ``mean`` / ``cov`` falls through to inclusion-exclusion and is right."""
    law = multivariate_uniform(low=-np.ones(2), high=np.ones(2))
    # The box sits strictly inside the support, so it holds exactly (2 eps) ** 2 of the unit
    # density 1/4.
    assert EpsilonLogarithmScore(0.5).score(law, np.zeros(2)) == pytest.approx(math.log(0.25))


def test_a_pmf_without_a_density_cannot_be_scored_at_eps_zero():
    """``eps = 0`` needs a density, and a pmf has none -- the error says so.

    This is the whole reason the neighbourhood exists, so the failure has to be legible rather
    than a silent ``-inf``.
    """
    law = multivariate_uniform(low=-np.ones(2), high=np.ones(2))
    with pytest.raises(TypeError, match='eps == 0 needs logpdf'):
        EpsilonLogarithmScore(0.0).score(law, np.zeros(2))


def test_a_law_that_offers_nothing_to_integrate_is_rejected():
    """Neither ``box_integral``, nor a Gaussian mean/cov, nor ``cdf`` -- nothing to dispatch on."""
    class Opaque:
        pass

    with pytest.raises(TypeError, match='box_integral'):
        EpsilonLogarithmScore(0.5).score(Opaque(), np.zeros(2))


@pytest.mark.parametrize('bad', (-0.1, np.inf, np.nan))
def test_a_radius_that_is_not_a_radius_is_rejected(bad):
    """A negative or non-finite radius is a mistake at construction, not at scoring time."""
    with pytest.raises(ValueError, match='radius'):
        EpsilonLogarithmScore(bad)
