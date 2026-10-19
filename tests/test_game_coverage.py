"""The coverage arithmetic: can this hand of chips miss, and how much does it catch?

Two implementations of this live in the repository -- ``experiments/pmf_betting/bet_coverage.py``
and the page's ``coverage.js`` -- and they must agree, because the page's safety badge is a claim
about money. The pages' side is checked against the exported goldens in ``test_game_page.py``;
this file is the authority's own tests, plus the one cross-check that keeps "exact" honest: the
closed-form tiling against a brute-force fine grid, where brute force is allowed to be slow and
stupid because it is only ever a witness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / 'experiments' / 'pmf_betting'
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import bet_coverage  # noqa: E402  (after the path insert above)
from ppsds.predictability.bounds import predictability_upper_bound  # noqa: E402
import presets  # noqa: E402


def brute_force_grid(atoms, weights, low, high, eps, n=601):
    """The payout share on a fine grid, the slow way. A witness, never the judge."""
    axis_x = np.linspace(low[0], high[0], n)
    axis_y = np.linspace(low[1], high[1], n)
    mass = np.zeros((n, n))
    for atom, weight in zip(np.asarray(atoms, dtype=float), weights):
        inside = ((np.abs(axis_x[:, None] - atom[0]) <= eps)
                  & (np.abs(axis_y[None, :] - atom[1]) <= eps))
        mass += weight * inside
    return mass


@pytest.mark.parametrize('key', ['shear', 'damped', 'stacked_deck', 'house_rules'])
def test_a_covering_layout_leaves_nothing_uncovered(key):
    """The layout the page offers as "safe" has to actually be safe, on every radius on offer."""
    preset = presets.get(key)
    low, high = presets.support_box(preset)
    for eps in preset.epsilons:
        atoms, weights = presets.covering_bet(preset, eps)
        report = bet_coverage.uncovered_report(atoms, weights, low, high, eps)
        assert report['possible_neg_inf'] is False
        assert report['uncovered_area_fraction'] < 1e-12, (
            f'{key} at eps={eps} leaves {report["uncovered_area_fraction"]} of the table open')
        assert report['worst_case_roi'] > 0


def test_a_covering_layout_tiles_exactly_when_the_numbers_allow():
    """When the support's half-width is a whole multiple of eps the tiles fit, every landing pays
    the same share ``1 / n``, and the worst case is that share rather than something smaller."""
    cases = [('shear', 0.25, 16), ('shear', 0.5, 4), ('damped', 0.1, 9)]
    for key, eps, count in cases:
        preset = presets.get(key)
        low, high = presets.support_box(preset)
        atoms, weights = presets.covering_bet(preset, eps)
        meta = presets.covering_meta(preset, eps)
        assert len(atoms) == count
        assert meta['n_atoms'] == count
        assert meta['guaranteed'] is True
        assert meta['worst_case_roi'] == pytest.approx(1 / count)
        report = bet_coverage.uncovered_report(atoms, weights, low, high, eps)
        assert report['worst_case_roi'] == pytest.approx(1 / count)


def test_the_safe_layout_meets_the_closed_form_where_the_page_offers_it():
    """The three layouts the game puts behind a button, pinned to the numbers the page shows."""
    expected = {
        ('shear', 0.5): (4, 0.25),
        ('shear', 0.25): (16, 1 / 16),
        ('damped', 0.1): (9, 1 / 9),
        ('damped', 0.4): (1, 1.0),
    }
    for (key, eps), (count, share) in expected.items():
        preset = presets.get(key)
        atoms, _ = presets.covering_bet(preset, eps)
        meta = presets.covering_meta(preset, eps)
        assert (len(atoms), meta['worst_case_roi']) == pytest.approx((count, share))


def test_covering_is_safe_but_no_longer_the_best_on_a_biased_table():
    """What the two new tables are for.

    Covering the support still guarantees a payout -- that is a property of the support, not of
    the density, so a biased table does not weaken it. What changes is that it is no longer the
    *best* hand: the table's entropy is lower, par is higher, and the gap is real money. The
    numbers are pinned because they are what a player sees when they press the button and read
    the scoreboard.
    """
    expected = {
        ('stacked_deck', 0.5): (4, np.log(0.25), -0.926713932101958),
        ('house_rules', 0.5): (4, np.log(0.25), -1.0627375681569962),
    }
    for (key, eps), (count, covering, par) in expected.items():
        preset = presets.get(key)
        low, high = presets.support_box(preset)
        atoms, weights = presets.covering_bet(preset, eps)
        report = bet_coverage.uncovered_report(atoms, weights, low, high, eps)

        assert len(atoms) == count
        assert report['possible_neg_inf'] is False, 'a biased table is still safe to cover'
        assert np.log(report['worst_case_roi']) == pytest.approx(covering)
        assert predictability_upper_bound(preset.noise(), eps) == pytest.approx(par)
        assert covering < par - 0.25, 'the whole point is that covering is now clearly worse'


def test_a_hand_that_leaves_a_hole_says_so():
    """The badge has to be able to say "not safe", or a badge that always says "safe" would pass."""
    preset = presets.get('shear')
    low, high = presets.support_box(preset)
    # One chip in the middle of a 2x2 table with eps = 0.5 covers a quarter of it.
    report = bet_coverage.uncovered_report([[0.0, 0.0]], [1.0], low, high, 0.5)
    assert report['possible_neg_inf'] is True
    assert report['worst_case_roi'] == 0.0
    assert report['uncovered_area_fraction'] == pytest.approx(0.75)

    # Two chips 1.5 apart: their boxes miss each other, and each covers a quarter.
    report = bet_coverage.uncovered_report([[-0.75, 0.0], [0.75, 0.0]], [0.5, 0.5], low, high, 0.5)
    assert report['possible_neg_inf'] is True
    assert report['uncovered_area_fraction'] == pytest.approx(0.625)


def test_the_cell_scan_agrees_with_a_brute_force_grid():
    """The scan is exact by construction; this is the witness that the construction is right."""
    rng = np.random.default_rng(20261015)
    preset = presets.get('shear')
    low, high = presets.support_box(preset)
    eps = 0.4
    worst = 0.0
    for _ in range(6):
        count = int(rng.integers(1, 6))
        atoms = rng.uniform(-1.0, 1.0, size=(count, 2))
        weights = rng.uniform(0.2, 1.0, size=count)

        report = bet_coverage.uncovered_report(atoms, weights, low, high, eps)
        grid = brute_force_grid(atoms, weights, low, high, eps)
        covered = float((grid > 0).mean())
        assert covered == pytest.approx(1 - report['uncovered_area_fraction'], abs=2e-3)
        # The worst case the scan reports must be attained somewhere on the grid.
        assert grid.min() / weights.sum() >= report['worst_case_roi'] - 1e-9
        worst = max(worst, abs(grid.min() / weights.sum() - report['worst_case_roi']))
    assert worst < 1e-3, 'the scan and the grid disagree about the worst landing'


def test_a_gaussian_landing_can_never_be_guaranteed():
    """Unbounded noise: no hand is safe, and the honest number is a probability, not a worst case."""
    preset = presets.get('rotation')
    law = preset.noise()
    sigma = np.sqrt(np.diag(law.cov))
    for eps in preset.epsilons:
        meta = presets.covering_meta(preset, eps, reach=presets.SAFE_WINDOW_SIGMA * sigma)
        assert meta['guaranteed'] is False
        assert meta['worst_case_roi'] == 0.0
        assert meta['kind'] == 'gaussian'


def test_the_catch_probability_of_the_three_sigma_window():
    """Covering +-3 sigma exactly leaves the tails, and the tails are a closed form."""
    preset = presets.get('rotation')
    law = preset.noise()
    sigma = np.sqrt(np.diag(law.cov))
    reach = presets.SAFE_WINDOW_SIGMA * sigma
    atoms, weights = presets.covering_bet(preset, 0.5, reach=reach)
    miss = bet_coverage.gaussian_miss_probability(atoms, weights, law.mean, law.cov, 0.5)
    # The window tiles exactly, so what is missed is exactly the mass outside +-3 sigma.
    expected = 1 - (2 * 0.9986501019683699 - 1) ** 2
    assert miss == pytest.approx(expected, abs=1e-9)


def test_a_covering_layout_needs_a_radius():
    """``eps = 0`` has no area to cover: the page must say so rather than divide by zero."""
    preset = presets.get('shear')
    with pytest.raises(ValueError, match='positive radius'):
        presets.covering_bet(preset, 0.0)
    with pytest.raises(ValueError, match='positive radius'):
        presets.covering_bet(preset, -0.5)


def test_unbounded_noise_needs_a_window():
    """The safe layout cannot invent a window for a table whose noise has no edges."""
    with pytest.raises(ValueError, match='unbounded noise'):
        presets.covering_bet(presets.get('rotation'), 0.5)
