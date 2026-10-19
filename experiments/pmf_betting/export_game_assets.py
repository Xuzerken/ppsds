"""Export the game's data from ppsds -- the "outward" half of the game/experiment contract.

Run::

    python experiments/pmf_betting/export_game_assets.py

Writes into the game page's ``assets/`` directory -- ``public/docs/examples/understand_ppsds_by_pmf_betting/game/assets/``
in this repository, ``docs/...`` on the public side (where ``public/**`` is the repository root):

- ``presets.json`` -- the systems, their noise laws, the radii they offer and what the **safe**
  layout achieves on each of them, straight from :mod:`presets`;
- ``trajectories.json`` -- pre-generated rollouts, one list per preset. The page **replays**
  these by index; it never samples outcomes itself. That is what makes a hand played in the
  browser reproducible here: same preset, same trajectory index, same noise;
- ``selfcheck.json`` -- (preset, radius, chips) -> score, plus the coverage and nominal-dynamics
  goldens, all computed by ppsds. The page evaluates the same cases with its own JavaScript on
  load and complains if they disagree.

Why the page must replay rather than sample
-------------------------------------------
Two independent implementations of one scoring rule will drift apart eventually. Replaying a
recorded trajectory does not remove that risk, but it turns the risky part into a *number the
two sides can compare* -- and the self-check vectors make the comparison happen on every page
load, in the open, instead of being discovered later as "the game and the script disagree".

Three kinds of case, because the page now decides three things in JavaScript
-----------------------------------------------------------------------------------
``score``
    What a hand pays. Includes a case with **unnormalised** chip weights: a pmf is normalised,
    so the score is ``log(hit / total)`` and not ``log(hit)``. Only a hand whose weights happen
    to sum to one cannot tell those apart, which is why every other case here looked fine while
    the page was occasionally reporting a number the library would not.
``coverage``
    Whether a hand can miss entirely (bounded noise), and how much it catches (unbounded noise).
    The safe layout itself is checked here too, by making the page **generate** it and compare.
``drift``
    Where the table's own dynamics takes the puck -- the path the page draws both on the felt
    and in the table card. Without a case pinning it, that drawing would be a second, invented
    dynamics rather than the one in :mod:`presets`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bet_coverage import gaussian_miss_probability, uncovered_report
from presets import (
    EXPORT_SEED,
    EXPORT_STEPS,
    PRESETS,
    SAFE_WINDOW_SIGMA,
    covering_bet,
    covering_meta,
    default_bets,
    fixture_trajectory,
    sample_trajectories,
    support_box,
)
from ppsds.numerics.atom_pmf import AtomPmf
from ppsds.predictability.bounds import predictability_upper_bound
from ppsds.scoring_rule.epsilon_logarithm_score import EpsilonLogarithmScore

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_ROOT.parents[1]

#: Where the page lives. The published tree is ``public/docs/`` here, while on the public side
#: ``public/**`` **is** the repository root, so the same page sits under ``docs/``. Probe rather
#: than assume: this script runs in both trees, and a wrong index does not raise -- it writes to
#: a directory that looks plausible and leaves the page loading stale data. That is why
#: ``tests/test_game_contract.py`` compares the committed assets against a fresh export.
DOCS_ROOT = ((REPO_ROOT / 'public' / 'docs')
             if (REPO_ROOT / 'public' / 'docs').is_dir()
             else (REPO_ROOT / 'docs'))
GAME_ASSETS = (DOCS_ROOT / 'examples'
               / 'understand_ppsds_by_pmf_betting' / 'game' / 'assets')

#: Rollouts per preset. Enough for a session; each is ~31 states, so the file stays small.
TRAJECTORIES_PER_PRESET = 12

#: Steps in an exported page replay -- see :data:`presets.EXPORT_STEPS` for why it is its own
#: constant, and why everything that regenerates a replay generates this many before slicing.
TRAJECTORY_STEPS = EXPORT_STEPS

#: Decimals kept in the JSON. Six is far below the tolerance the self-check uses and cuts the
#: file size by roughly a third.
DECIMALS = 6

#: Steps the nominal-dynamics golden walks. Long enough to tell the three tables apart (an arc,
#: a spiral, a slide), short enough that the JSON stays a few hundred bytes.
DRIFT_STEPS = 6


def _noise_description(preset):
    """A JSON-able description of the noise law, for the page to display (never to sample).

    The law is asked what it is (``law.kind``) before its attributes are inspected. Guessing from
    ``hasattr(law, 'low')`` used to label *any* bounded law "uniform", which would have had the
    page describe a block law as flat -- wrong, silently, in the one place a player looks to
    understand the table.

    A table that does not publish its law publishes nothing about it: no boxes, no weights, no
    support. (The support is withheld as well because the coverage arithmetic is built from it --
    publishing it while calling the law a secret would be a contradiction, and the self-check
    cases for such a table are dropped below for the same reason.)
    """
    if preset.hidden_noise:
        return {'kind': 'unpublished'}
    law = preset.noise()
    if getattr(law, 'kind', None) == 'blocks':
        return {
            'kind': 'blocks',
            'boxes': [[[float(v) for v in low], [float(v) for v in high]] for low, high in law.boxes],
            'weights': [float(w) for w in law.weights],
        }
    if hasattr(law, 'low') and hasattr(law, 'high'):
        return {'kind': 'uniform', 'low': [float(v) for v in law.low],
                'high': [float(v) for v in law.high]}
    mean, cov = getattr(law, 'mean', None), getattr(law, 'cov', None)
    if mean is not None and cov is not None:
        return {'kind': 'gaussian', 'mean': [float(v) for v in np.asarray(mean).reshape(-1)],
                'cov': [[float(v) for v in row] for row in np.asarray(cov)]}
    return {'kind': type(law).__name__}


def support_description(preset):
    """Whether the noise can land anywhere, and (if not) the box it is confined to.

    The page needs this to decide what "safe" can even mean: a bounded box can be covered, an
    unbounded one cannot. ``gaussian`` is the page's assumption for the unbounded case -- it
    computes an uncatchable mass from ``mean``/``cov``, so an unbounded law that is *not*
    Gaussian would need its own branch there rather than silently reusing this one.
    """
    if preset.hidden_noise:
        return {'kind': 'unpublished'}
    box = support_box(preset)
    if box is not None:
        return {'kind': 'bounded', 'low': [float(v) for v in box[0]],
                'high': [float(v) for v in box[1]]}
    law = preset.noise()
    if getattr(law, 'mean', None) is None or getattr(law, 'cov', None) is None:
        raise ValueError(
            f'preset {preset.key!r} has unbounded noise that the page cannot describe: it has '
            'neither a low/high box nor a mean/cov. Teach the page about it before exporting.'
        )
    return {'kind': 'gaussian'}


def drift_matrix(preset):
    """The preset's drift as a 2x2 matrix, **verified** linear.

    Evaluated on the unit basis rather than at the origin -- a linear drift maps the origin to the
    origin, so sampling there would hand the page a matrix of zeros and every nominal-next-state
    arrow would point at the origin. The verification is the point: presets are declared as
    callables, and a nonlinear one would make the page quietly draw the wrong thing.
    """
    basis = np.eye(2)
    columns = [np.asarray(preset.drift(basis[i], np.zeros(0)), dtype=float) for i in range(2)]
    matrix = np.column_stack(columns)
    probe = np.array([0.7, -1.3])
    if not np.allclose(preset.drift(probe, np.zeros(0)), matrix @ probe, atol=1e-10):
        raise ValueError(
            f'preset {preset.key!r} is not linear, so the page cannot draw its nominal next '
            'state from a matrix. Encode it differently before exporting.'
        )
    return matrix


def _safe_window(preset, eps):
    """The ``reach`` a safe layout covers on this table, and whether that is a guarantee.

    Bounded noise: the support itself, so the answer is a guarantee and ``reach`` is ``None``
    (the layout does not need to be told where to look). Unbounded noise: a window of
    :data:`~presets.SAFE_WINDOW_SIGMA` standard deviations, which is a choice, not a guarantee.
    """
    if support_box(preset) is not None:
        return None
    sigma = np.sqrt(np.diag(np.asarray(preset.noise().cov, dtype=float)))
    return SAFE_WINDOW_SIGMA * sigma


def _safe_summary(preset, eps):
    """What the page's "Cover the table" button achieves here -- for the table card."""
    meta = covering_meta(preset, eps, reach=_safe_window(preset, eps))
    summary = {'n_atoms': meta['n_atoms'], 'guaranteed': meta['guaranteed'],
               'worst_case_roi': meta['worst_case_roi']}
    if meta['kind'] == 'gaussian':
        summary['window_sigma'] = SAFE_WINDOW_SIGMA
    return summary


def export_presets():
    """The systems, plus the **par line** the page draws.

    The par line is the predictability upper bound for that (system, radius), computed here
    rather than in the page: it needs the truth's differential entropy, and re-deriving it in
    JavaScript would be a second implementation of a number the page only displays.
    """
    out = {}
    for key, preset in PRESETS.items():
        noise = preset.noise()
        out[key] = {
            'key': preset.key,
            'title': preset.title,
            'blurb': preset.blurb,
            'motion': preset.motion,
            # Withheld along with the law: par is `d log 2 eps - h_d(q)`, so publishing it on a
            # table that keeps `q` to itself would hand over the truth's entropy -- one number,
            # but a real hint, and the rule on such a table is that nothing derived from the
            # hidden law is published. The player still has the payout, the score, the stake and
            # their own past landings to learn from.
            'upper_bounds': {} if preset.hidden_noise
                             else {str(float(eps)): float(predictability_upper_bound(noise, eps))
                                   for eps in preset.epsilons},
            'drift_matrix': [[float(v) for v in row] for row in drift_matrix(preset)],
            'noise': _noise_description(preset),
            'hidden_noise': bool(preset.hidden_noise),
            'support': support_description(preset),
            # Withheld for a table that does not publish its law: the summary says how many
            # tiles the covering layout uses and how well it does, and both are functions of the
            # support this table is keeping to itself (tiles of 2 eps over the support *are* the
            # support). The page does not show them there anyway -- its "cover the table" button
            # is off -- so this closes a leak without costing anything.
            'safe': {} if preset.hidden_noise else
                    {str(float(eps)): _safe_summary(preset, eps) for eps in preset.epsilons},
            'x0': [float(v) for v in preset.x0],
            'epsilons': [float(eps) for eps in preset.epsilons],
        }
    return out


def export_trajectories():
    out = {}
    for index, preset in enumerate(PRESETS.values()):
        states, noise = sample_trajectories(
            preset, TRAJECTORIES_PER_PRESET, TRAJECTORY_STEPS, seed=EXPORT_SEED + index)
        out[preset.key] = {
            'states': np.round(states, DECIMALS).tolist(),
            'noise': np.round(noise, DECIMALS).tolist(),
        }
    return out


def _bet_for_case(preset, eps, noise):
    """The hand a scoring case is settled with, and the box that hand was built to cover.

    Bounded noise gets the safe layout over its own support, which is also the only kind of hand
    that cannot miss. Unbounded noise has no support to cover, so the window is widened to
    include whatever the recorded fixture actually landed on -- the case then still has finite
    scores to compare, while the page's own safe window stays at
    :data:`~presets.SAFE_WINDOW_SIGMA` sigma.
    """
    box = support_box(preset)
    if box is not None:
        return covering_bet(preset, eps), box
    law = preset.noise()
    mean = np.asarray(law.mean, dtype=float).reshape(-1)
    sigma = np.sqrt(np.diag(np.asarray(law.cov, dtype=float)))
    reach = np.maximum(SAFE_WINDOW_SIGMA * sigma,
                       np.max(np.abs(np.asarray(noise) - mean), axis=0))
    window = (mean - reach, mean + reach)
    return covering_bet(preset, eps, reach=reach), window


def _score_case(preset, eps, noise, atoms, weights, bet_name):
    """``(chips, landings) -> score``, from ppsds, with the landings included.

    The outcomes are exported as well as the scores they produced. Without them the page could
    only check its arithmetic against the *recorded* scores -- a tautology inside one file. With
    them it recomputes each score from the chips and the landing, which is the
    cross-implementation check this file exists for.
    """
    scores = EpsilonLogarithmScore(eps).score(AtomPmf(atoms, weights), noise)
    if not np.all(np.isfinite(scores)):
        raise ValueError(
            f'the bet for preset {preset.key!r} at eps={eps} still misses some landings, so the '
            'self-check would contain -inf and the JSON would be invalid'
        )
    return {
        'kind': 'score',
        'preset': preset.key,
        'eps': float(eps),
        'bet_name': bet_name,
        'atoms': np.round(atoms, DECIMALS).tolist(),
        'weights': np.round(weights, DECIMALS).tolist(),
        'noise': np.round(noise, DECIMALS).tolist(),
        'per_step_scores': np.round(scores, DECIMALS).tolist(),
        'mean_score': float(np.round(scores.mean(), DECIMALS)),
    }


def _coverage_case(preset, eps, atoms, weights, layout, window):
    """What a hand covers, from ppsds. ``layout='covering'`` asks the page to rebuild the hand.

    The page generates the safe layout itself (it has to: the button has to work with no server),
    so the case carries the atoms it must produce *and* the numbers it must then compute -- and
    the ``low``/``high`` window those atoms were built over, which is the only thing the page
    needs to rebuild them. A case whose ``layout`` is ``given`` is a hand the page only measures.
    """
    low, high = window
    case = {
        'kind': 'coverage',
        'preset': preset.key,
        'eps': float(eps),
        'layout': layout,
        'low': [float(v) for v in low],
        'high': [float(v) for v in high],
        'atoms': np.round(atoms, DECIMALS).tolist(),
        'weights': np.round(weights, DECIMALS).tolist(),
    }
    if support_box(preset) is not None:
        report = uncovered_report(atoms, weights, low, high, eps)
        case.update({
            'uncovered_area_fraction': report['uncovered_area_fraction'],
            'worst_case_roi': report['worst_case_roi'],
            'possible_neg_inf': report['possible_neg_inf'],
        })
        return case
    law = preset.noise()
    mean = np.asarray(law.mean, dtype=float).reshape(-1)
    cov = np.asarray(law.cov, dtype=float)
    if not np.allclose(cov, np.diag(np.diag(cov))):
        raise ValueError(
            f'preset {preset.key!r} has correlated noise, and the page evaluates the catch '
            'probability for independent axes only. Give the page a bivariate CDF before '
            'exporting a preset like this.'
        )
    case.update({
        'mean': [float(v) for v in mean],
        'cov': [[float(v) for v in row] for row in cov],
        'miss_probability': gaussian_miss_probability(atoms, weights, mean, cov, eps),
    })
    return case


def _drift_case(preset):
    """Where the table's own drift takes the puck -- the path the page draws.

    The start is deliberately **not** ``x0``: the shear table's ``x0`` is the origin, which is a
    fixed point of its drift, so a golden path from there would be all zeros and would pass for
    any matrix at all. Offsetting by a fixed amount keeps the walk in the informative part of
    every table's motion.
    """
    start = np.asarray(preset.x0, dtype=float) + np.array([0.7, 0.4])
    path = [start]
    for _ in range(DRIFT_STEPS):
        path.append(np.asarray(preset.drift(path[-1], np.zeros(0)), dtype=float))
    return {
        'kind': 'drift',
        'preset': preset.key,
        'start': [float(v) for v in start],
        'steps': DRIFT_STEPS,
        'path': np.round(np.asarray(path), DECIMALS).tolist(),
    }


def export_selfcheck():
    """Every case the page re-derives in JavaScript, with ppsds' answers attached.

    Each scoring case settles the **first trajectory's** landings, so the page's answer depends
    on the same numbers it displays -- a self-check that scored unseen data would pass while the
    visible scoring was wrong. Every emitted number is finite, and checked to be: a non-finite
    float here would be written as invalid JSON rather than failing.
    """
    cases = []
    for index, preset in enumerate(PRESETS.values()):
        _, noise = fixture_trajectory(preset)
        if preset.hidden_noise:
            # Only the dynamics. A scoring or coverage case would carry the atoms, the support and
            # the recorded landings -- which together *are* the noise law this table keeps to
            # itself, in a file the page serves to anyone who asks. The mathematics stays pinned
            # by every other table (the scoring code is shared, not per table).
            cases.append(_drift_case(preset))
            continue
        for eps in preset.epsilons[:2]:
            (atoms, weights), window = _bet_for_case(preset, eps, noise)
            cases.append(_score_case(preset, eps, noise, atoms, weights,
                                     f'safe layout, {len(atoms)} stacks'))
            cases.append(_coverage_case(preset, eps, atoms, weights, 'covering', window))

        # On an unbounded table the safe layout covers a *chosen* window, and the page offers the
        # same one -- so the hand the button builds there gets its own case. Above, the window had
        # to be widened to reach the recorded landings; here it is the game's own choice.
        if support_box(preset) is None:
            eps = preset.epsilons[0]
            reach = _safe_window(preset, eps)
            cases.append(_coverage_case(
                preset, eps, *covering_bet(preset, eps, reach=reach), layout='covering',
                window=(np.asarray(preset.noise().mean).reshape(-1) - reach,
                        np.asarray(preset.noise().mean).reshape(-1) + reach)))

        cases.append(_drift_case(preset))

    # A hand whose weights do not sum to one. The library normalises a pmf, so what settles is
    # the *share* that landed -- log(hit / total) -- and a page that reported log(hit) would be
    # right only here, where the sum happens to be 1.
    unnormalised = PRESETS['shear']
    _, noise = fixture_trajectory(unnormalised)
    atoms, _ = default_bets(unnormalised, eps=0.5)
    cases.append(_score_case(unnormalised, 0.5, noise, atoms, np.full(len(atoms), 2.0),
                             'weights of 2.0 each: what settles is the share, not the sum'))

    # A hand that covers *part* of the table: two chips far enough apart that their boxes do not
    # meet. The page must say "this one can miss", which is what the safety badge is for --
    # without a case like this, a badge that said "safe" no matter what would still pass.
    scattered = PRESETS['shear']
    cases.append(_coverage_case(scattered, 0.5, np.array([[-0.75, 0.0], [0.75, 0.0]]),
                                np.array([0.5, 0.5]), layout='given',
                                window=support_box(scattered)))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=GAME_ASSETS,
                        help='where to write the assets (default: %(default)s)')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    payloads = {
        'presets.json': export_presets(),
        'trajectories.json': export_trajectories(),
        'selfcheck.json': export_selfcheck(),
    }
    for name, payload in payloads.items():
        path = args.out / name
        # `allow_nan=False` because the default writes `NaN` / `Infinity` / `-Infinity`, which are
        # **not** valid JSON: the browser's `JSON.parse` would reject the entire file, and the
        # failure would look like "the game is broken" rather than "an export emitted a non-finite
        # number". Better to fail here, where the reason is visible.
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False),
                        encoding='utf-8')
        print(f'wrote {path}  ({path.stat().st_size / 1024:.1f} KiB)')
    print()
    kinds = {}
    for case in payloads['selfcheck.json']:
        kinds[case['kind']] = kinds.get(case['kind'], 0) + 1
    summary = ', '.join(f'{count} {kind}' for kind, count in sorted(kinds.items()))
    print(f'{len(payloads["selfcheck.json"])} self-check cases ({summary}). The page recomputes '
          'all of them in JavaScript on load and warns in the console if it disagrees.')


if __name__ == '__main__':
    main()
