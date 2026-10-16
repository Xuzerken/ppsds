"""The block noise law: an exact entropy, and the reason the biased tables exist.

``BlockNoise`` is what makes the game's "safe but not best" tables possible, and its whole
justification is the closed-form entropy -- par is ``d log 2 eps - h_d(q)``, so if the entropy were
merely approximate, every par line on those tables would be approximate too. So the entropy is
checked two independent ways: against an algebraically different rearrangement of the same
integral, and against a Monte-Carlo estimate that shares no code with it.

The laws themselves are also interface-tested: they stand in for the library's noise laws, whose
consumers ask for ``low``/``high`` (the support box), ``dimensions``/``mean`` (the dimension),
``rvs(size=, rng=)`` (sampling) and ``entropy()`` (par). Missing any of them fails somewhere far
away from here, which is the kind of failure that costs an afternoon.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / 'examples' / 'understand_ppsds_by_pmf_betting' / 'experiment'
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import presets  # noqa: E402  (after the path insert above)
from noise_laws import BlockNoise  # noqa: E402
from ppsds.predictability.bounds import predictability_upper_bound  # noqa: E402

BANDS = BlockNoise(
    boxes=[((-1.0, -1.0), (1.0, -1.0 / 3.0)),
           ((-1.0, -1.0 / 3.0), (1.0, 1.0 / 3.0)),
           ((-1.0, 1.0 / 3.0), (1.0, 1.0))],
    weights=[0.10, 0.80, 0.10],
)

QUADRANTS = BlockNoise(
    boxes=[((-1.0, 0.0), (0.0, 1.0)), ((0.0, 0.0), (1.0, 1.0)),
           ((-1.0, -1.0), (0.0, 0.0)), ((0.0, -1.0), (1.0, 0.0))],
    weights=[0.60, 0.20, 0.15, 0.05],
)


@pytest.mark.parametrize('law', [BANDS, QUADRANTS], ids=['bands', 'quadrants'])
def test_the_entropy_matches_the_same_integral_rearranged(law):
    """``h = -sum w log(w / V)`` against ``h = H(w) + sum w log V``: same value, different algebra."""
    rearranged = float(
        -np.sum(law.weights * np.log(law.weights)) + np.sum(law.weights * np.log(law.volumes)))
    assert law.entropy() == pytest.approx(rearranged, abs=1e-12)


@pytest.mark.parametrize('law', [BANDS, QUADRANTS], ids=['bands', 'quadrants'])
def test_the_entropy_matches_a_monte_carlo_estimate(law):
    """An independent estimate from samples: ``h = -E log q(X)``. Shares no code with the formula.

    The tolerance is the Monte-Carlo error, not the formula's: with 200k draws and a log-density
    that takes a handful of values, the standard error is of order 1e-3.
    """
    draws = law.rvs(size=200_000, rng=np.random.default_rng(20261015))
    estimate = float(-np.mean(np.log(law.pdf(draws))))
    assert estimate == pytest.approx(law.entropy(), abs=5e-3)


@pytest.mark.parametrize('law', [BANDS, QUADRANTS], ids=['bands', 'quadrants'])
def test_the_density_is_normalised_and_zero_outside(law):
    """A grid sum of ``pdf x cell area`` is one, to within the grid's own boundary counting.

    The law is a mixture of uniforms, so a point on a face two blocks share belongs to both
    components and the density there is their sum. That face is measure zero, so the integral is
    unaffected -- but a grid sum counts those cells twice, which is where the tolerance comes
    from (a per-block integral is exact by construction and would prove nothing).
    """
    axis = np.linspace(-1.0, 1.0, 801)
    grid = np.stack(np.meshgrid(axis, axis, indexing='ij'), axis=-1).reshape(-1, 2)
    cell = (axis[1] - axis[0]) ** 2
    assert float(np.sum(law.pdf(grid))) * cell == pytest.approx(1.0, abs=0.02)

    far = np.array([[3.0, 3.0], [-2.5, 0.5]])
    assert np.all(law.pdf(far) == 0.0)

    # Inside a block -- strictly inside, off every shared face -- the density is that block's
    # weight over its volume, and on a face it is the sum of the blocks that meet there.
    for weight, (low, high), volume in zip(law.weights, law.boxes, law.volumes):
        middle = 0.5 * (low + high)
        assert float(law.pdf(middle)) == pytest.approx(weight / volume)


@pytest.mark.parametrize('law', [BANDS, QUADRANTS], ids=['bands', 'quadrants'])
def test_draws_are_reproducible_and_land_in_the_boxes(law):
    """Reproducible from an explicit ``rng`` (this is what makes a run replayable), and supported."""
    first = law.rvs(size=500, rng=np.random.default_rng(7))
    second = law.rvs(size=500, rng=np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert first.shape == (500, 2)

    inside = np.zeros(len(first), dtype=bool)
    for low, high in law.boxes:
        inside |= np.all((first >= low) & (first <= high), axis=1)
    assert inside.all(), 'a draw landed outside every box'


@pytest.mark.parametrize('law', [BANDS, QUADRANTS], ids=['bands', 'quadrants'])
def test_the_support_and_moments_are_what_consumers_read(law):
    """``low``/``high`` are the support box and ``mean``/``cov`` are exact: the game's covering
    layout, its dimension inference and the page's prose all read these."""
    assert law.low.tolist() == [-1.0, -1.0]
    assert law.high.tolist() == [1.0, 1.0]
    assert law.dimensions == 2

    draws = law.rvs(size=400_000, rng=np.random.default_rng(11))
    assert np.allclose(draws.mean(axis=0), law.mean, atol=2e-3)
    assert np.allclose(np.cov(draws.T), law.cov, atol=5e-3)

    # The covariance is the within-box spread plus the spread of the box centres; check it against
    # the definition on a one-box law, where the within-box term is the only one.
    single = BlockNoise(boxes=[((-1.0, -1.0), (1.0, 1.0))], weights=[1.0])
    assert np.allclose(single.cov, np.diag([1.0 / 3.0, 1.0 / 3.0]))
    assert single.mean.tolist() == [0.0, 0.0]
    assert single.entropy() == pytest.approx(np.log(4.0))


def test_the_par_line_of_a_biased_table_is_higher_and_the_gap_is_the_entropy_deficit():
    """The identity the biased tables teach: with the support fixed, the covering layout's score
    sits below par by exactly ``h_d(uniform on the support) - h_d(q)``.

    It is worth pinning as a number, because it is what a player is looking at: the covering hand
    still cannot miss, and it is strictly worse than what this table's entropy allows.
    """
    uniform_on_support = float(np.log(4.0))  # a 2 x 2 box
    for law, eps, covering, par in (
        (BANDS, 0.5, -1.3862943611198906, -0.926713932101958),
        (QUADRANTS, 0.5, -1.3862943611198906, -1.0627375681569962),
    ):
        assert predictability_upper_bound(law, eps) == pytest.approx(par, abs=1e-12)
        assert covering == pytest.approx(np.log(0.25), abs=1e-12)
        deficit = uniform_on_support - law.entropy()
        assert (par - covering) == pytest.approx(deficit, abs=1e-12)
        assert covering < par, 'covering is supposed to be strictly worse on a biased table'


def test_the_shipped_biased_tables_are_the_laws_this_file_tests():
    """The registry and these tests must not drift: the numbers above are the shipped ones."""
    assert presets.get('stacked_deck').noise().entropy() == pytest.approx(BANDS.entropy())
    assert presets.get('house_rules').noise().entropy() == pytest.approx(QUADRANTS.entropy())
    assert presets.get('house_rules').hidden_noise is True
    assert presets.get('stacked_deck').hidden_noise is False


def test_weights_are_normalised_but_never_silently_wrong():
    """Raw counts are accepted (the library's laws do the same); nonsense is refused."""
    law = BlockNoise(boxes=[((-1.0, -1.0), (0.0, 0.0)), ((0.0, 0.0), (1.0, 1.0))], weights=[3, 1])
    assert law.weights.tolist() == [0.75, 0.25]

    with pytest.raises(ValueError, match='non-negative'):
        BlockNoise(boxes=[((-1.0, -1.0), (1.0, 1.0))], weights=[-1.0])
    with pytest.raises(ValueError, match='positive'):
        BlockNoise(boxes=[((-1.0, -1.0), (1.0, 1.0))], weights=[0.0])
    with pytest.raises(ValueError, match='pair up'):
        BlockNoise(boxes=[((-1.0, -1.0), (1.0, 1.0))], weights=[0.5, 0.5])
    with pytest.raises(ValueError, match='empty'):
        BlockNoise(boxes=[((1.0, 1.0), (1.0, 1.0))], weights=[1.0])
    with pytest.raises(ValueError, match='dimensional'):
        BlockNoise(boxes=[((-1.0, -1.0), (1.0, 1.0)), ((0.0,), (1.0,))], weights=[0.5, 0.5])
    with pytest.raises(ValueError, match='overlap'):
        BlockNoise(boxes=[((-1.0, -1.0), (0.5, 0.5)), ((0.0, 0.0), (1.0, 1.0))], weights=[0.5, 0.5])


def test_touching_boxes_are_not_overlapping():
    """Sharing a face is how a tiling is built; only a shared *interior* is refused."""
    law = BlockNoise(
        boxes=[((-1.0, -1.0), (0.0, 1.0)), ((0.0, -1.0), (1.0, 1.0))], weights=[0.4, 0.6])
    assert law.entropy() == pytest.approx(-(0.4 * np.log(0.4 / 2.0) + 0.6 * np.log(0.6 / 2.0)))
