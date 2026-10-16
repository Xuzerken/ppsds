"""The predictability reference values, checked against the **published** closed forms.

The point of this file is that the oracle is external. The paper
("Probabilistic predictability of stochastic dynamical systems", *Automatica* **174** (2025)
112160) prints its one figure with two horizontal lines whose values are known in closed form
for its system, so the library is checked against the paper rather than against itself.

The two claims worth separating:

- the **upper bound** ``d_x log(2 eps) - h_d(q)`` is exact and cheap -- a property of the system;
- the **approximation** ``d_x log(2 eps) - H(q || p)`` is only ``O(eps)`` away from a *given*
  predictor's true expected score, so it is checked both as a formula (exactly) and as a claim
  about a simulated score (within ``O(eps)``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from _systems import (
    PAPER_CROSS_ENTROPY_GAUSSIAN,
    PAPER_ENTROPY_NOISE,
    PAPER_EPSILONS,
    paper_approximation,
    paper_noise_draws,
    paper_noise_law,
    paper_predictor_law,
    paper_upper_bound,
)
from ppsds.predictability.bounds import approximation_value, predictability_upper_bound
from ppsds.predictability.entropy import cross_entropy, differential_entropy
from ppsds.predictability.trajectory import across_trajectories, cumulative_average
from ppsds.scoring_rule.epsilon_logarithm_score import EpsilonLogarithmScore


def test_noise_entropy_is_the_log_volume():
    """``h_d(U([-1,1]^2)) = log 4`` -- a uniform density is ``1 / volume``."""
    assert differential_entropy(paper_noise_law()) == pytest.approx(math.log(4.0))


def test_cross_entropy_matches_the_closed_form():
    """``H(U([-1,1]^2) || N(0, I)) = log(2 pi) + 1/3``.

    The library integrates rather than using this closed form, so the tolerance is the
    quadrature's own accuracy (tensor trapezoid, ``O(h ** 2)``), not machine epsilon.
    """
    measured = cross_entropy(paper_noise_law(), paper_predictor_law())
    assert measured == pytest.approx(PAPER_CROSS_ENTROPY_GAUSSIAN, abs=1e-4)


@pytest.mark.parametrize('eps', PAPER_EPSILONS)
def test_upper_bound_matches_the_paper(eps):
    """The red dotted line: ``2 log(2 eps) - log 4``."""
    measured = predictability_upper_bound(paper_noise_law(), eps)
    assert measured == pytest.approx(paper_upper_bound(eps), abs=1e-12)


@pytest.mark.parametrize('eps', PAPER_EPSILONS)
def test_approximation_matches_the_paper(eps):
    """The blue dotted line: ``2 log(2 eps) - log(2 pi) - 1/3``."""
    measured = approximation_value(paper_noise_law(), paper_predictor_law(), eps)
    assert measured == pytest.approx(paper_approximation(eps), abs=1e-4)


@pytest.mark.parametrize('eps', PAPER_EPSILONS)
def test_the_upper_bound_dominates_the_approximation(eps):
    """The best-possible score cannot be below a given predictor's -- so the lines are ordered.

    Both are expectations under the same true law, and the upper bound is the maximum over
    predictors while the approximation is one predictor's value, up to its ``O(eps)`` error.
    """
    q, p = paper_noise_law(), paper_predictor_law()
    assert predictability_upper_bound(q, eps) > approximation_value(q, p, eps)


def test_eps_zero_drops_the_volume_term():
    """At ``eps = 0`` both formulas lose ``d_x log(2 eps)`` and keep the entropy term.

    The alternative -- treating ``eps -> 0`` as a limit -- would send both to ``-inf``, which is
    what the score itself does; the indicator in the published formulas is what makes exactly
    ``eps = 0`` a different, finite case.
    """
    q = paper_noise_law()
    assert predictability_upper_bound(q, 0.0) == pytest.approx(-PAPER_ENTROPY_NOISE)
    assert approximation_value(q, paper_predictor_law(), 0.0) == pytest.approx(
        -PAPER_CROSS_ENTROPY_GAUSSIAN, abs=1e-4
    )


@pytest.mark.parametrize('eps', PAPER_EPSILONS)
def test_the_simulated_score_tracks_the_approximation_to_order_eps(eps):
    """The paper's Theorem 2, measured: ``|mean score - approximation|`` is ``O(eps)``.

    This is the claim the paper's figure is *for*, so it is checked on the paper's own system
    with the paper's own deliberately misspecified predictor. The realised outcome the score is
    evaluated at is the noise draw: the predictor predicts the noise, so the drift cancels.
    """
    draws = paper_noise_draws(n_traj=2000, n_step=30, seed=9)
    scores = EpsilonLogarithmScore(eps).score(paper_predictor_law(), draws)
    error = abs(scores.mean() - approximation_value(paper_noise_law(), paper_predictor_law(), eps))
    # Measured on this seed: 0.014 at eps=0.25, 0.056 at 0.5, 0.124 at 0.75 -- i.e. ~0.06*eps,
    # ~0.11*eps, ~0.16*eps, growing with eps exactly as the paper reports ("still around 0.1"
    # at the largest radius). 0.25*eps leaves room for the sampling noise without admitting a
    # failure of the order claim.
    assert error <= 0.25 * eps, f'approximation error {error:.4f} is not O(eps) for eps={eps}'


def test_a_single_trajectory_converges_to_the_expectation():
    """The paper's Theorem 3, measured: the running per-trajectory mean converges like ``1/sqrt(T)``.

    This is why the figures need one trajectory per curve rather than repeated samplings.

    Not marked ``slow``: the predictor is a diagonal Gaussian, so the whole sweep goes through
    the closed form in milliseconds. A filter that skipped this would be skipping the cheapest
    test in the file.
    """
    eps = 0.5
    scores = EpsilonLogarithmScore(eps).score(
        paper_predictor_law(), paper_noise_draws(n_traj=2000, n_step=30, seed=11)
    ).reshape(2000, 30)
    deviation = np.abs(cumulative_average(scores) - scores.mean())
    first, last = deviation[:, 0].mean(), deviation[:, -1].mean()
    # Measured: 0.162 -> 0.028, a ratio of 0.175 against the 1/sqrt(30) = 0.183 the rate
    # predicts. The bound below is deliberately loose -- it checks the *rate*, not the constant.
    assert last < first / 3, f'deviation fell only from {first:.4f} to {last:.4f}'


def test_each_panel_draws_its_own_trajectories():
    """Panels must not share a sample.

    The figure's three thin curves are *samples*: drawing the same three in every panel makes it
    look as though the individual curves were a property of the radius rather than of the draw.
    The published figure picks three at random per radius, and the caption says so. This pins
    both halves: different radii get different indices, and the same radius is reproducible.
    """
    from ppsds.visualization.predictability import choose_trajectories

    panels = {eps: choose_trajectories(1000, eps) for eps in PAPER_EPSILONS}
    for eps, chosen in panels.items():
        assert len(set(chosen)) == 3, f'eps={eps} drew a repeated index: {chosen}'
        assert all(0 <= i < 1000 for i in chosen)
    assert len({frozenset(v) for v in panels.values()}) == len(panels), (
        f'panels share a sample: {panels}')
    assert choose_trajectories(1000, 0.5) == panels[0.5], 'the choice must be reproducible'


def test_a_short_rollout_thins_the_figure_instead_of_breaking_it():
    """Fewer trajectories than curves asked for degrades to what exists."""
    from ppsds.visualization.predictability import choose_trajectories

    assert choose_trajectories(2, 0.5) == (0, 1)
    with pytest.raises(ValueError, match='no trajectories'):
        choose_trajectories(0, 0.5)


def test_cumulative_average_is_a_running_mean():
    """Entry ``[n, k]`` averages the first ``k + 1`` steps, so a length-1 block is unchanged."""
    scores = np.array([[1.0, 3.0, 5.0]])
    assert cumulative_average(scores) == pytest.approx(np.array([[1.0, 2.0, 3.0]]))


def test_across_trajectories_bands_the_spread():
    """The band brackets the mean, and a single trajectory collapses it rather than erroring."""
    spread = across_trajectories(np.array([[0.0, 1.0], [10.0, 11.0]]))
    assert spread['mean'] == pytest.approx(np.array([5.0, 6.0]))
    assert np.all(spread['low'] < spread['mean']) and np.all(spread['mean'] < spread['high'])
    assert spread['quantile_levels'] == (0.025, 0.975)

    single = across_trajectories(np.array([[2.0, 4.0]]))
    assert single['mean'] == pytest.approx(np.array([2.0, 4.0]))
    assert single['low'] == pytest.approx(single['mean'])
    assert single['high'] == pytest.approx(single['mean'])
