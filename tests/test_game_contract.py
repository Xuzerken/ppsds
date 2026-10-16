"""The game/experiment contract: the page and this package must be the same world.

The interactive page re-implements several of the package's numbers in JavaScript -- the scoring
rule, the coverage arithmetic, the safe layout, the table's own dynamics. Two independent
implementations of one rule will drift eventually, so the agreement is pinned rather than assumed:

- the exported assets must be **strict, finite JSON** (the browser's ``JSON.parse`` is not
  Python's ``json.loads``);
- every self-check case the page validates itself against must **reproduce here** -- if the
  recorded number and ppsds disagree, the page would be checking itself against a wrong answer,
  which is worse than not checking at all;
- the assets directory must contain **nothing but** the exporter's output, because
  ``public/**`` is copied into the public repository wholesale and a stray file would be published;
- a bet file exported from the page must score the same when fed back through ``run.py``.

These run against the **committed** assets, not freshly generated ones: what ships is what the
page will load.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ppsds.numerics.atom_pmf import AtomPmf
from ppsds.scoring_rule.epsilon_logarithm_score import EpsilonLogarithmScore

from _paths import public_root

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / 'examples' / 'understand_ppsds_by_pmf_betting' / 'experiment'

#: The page's data, in whichever layout this tree uses -- see :mod:`_paths`.
ASSETS = ((public_root() or REPO_ROOT / 'public') / 'docs'
          / 'understand_ppsds_by_pmf_betting' / 'game' / 'assets')

pytestmark = pytest.mark.skipif(
    public_root() is None,
    reason='this tree has no published content (for instance an unpacked sdist), so the page this '
           'contract is about is not here to check',
)

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import bet_coverage  # noqa: E402  (after the path insert above)
import presets  # noqa: E402
import run as pmf_run  # noqa: E402

ASSET_FILES = ('presets.json', 'trajectories.json', 'selfcheck.json')


def _load(name):
    return json.loads((ASSETS / name).read_text(encoding='utf-8'))


def _reject_constant(token):
    """Reject the non-standard ``NaN`` / ``Infinity`` / ``-Infinity`` tokens.

    ``json.loads`` accepts them by default, so a plain parse would pass on a file the browser
    refuses to read at all.
    """
    raise AssertionError(f'non-standard JSON token {token!r} -- JSON.parse would reject this file')


@pytest.mark.parametrize('name', ASSET_FILES)
def test_assets_are_strict_json_with_no_non_finite_numbers(name):
    """A file the browser cannot parse is a broken game, however right the numbers are."""
    path = ASSETS / name
    assert path.is_file(), f'{path} is missing; run examples/understand_ppsds_by_pmf_betting/experiment/export_game_assets.py'
    json.loads(path.read_text(encoding='utf-8'), parse_constant=_reject_constant)


def test_the_assets_directory_holds_only_the_exporters_output():
    """``public/**`` is exported wholesale, so anything dropped in here gets published.

    Not hypothetical: the game's data lands in ``public/docs/game/assets/``, someone saving a hand
    they copied out of the page beside it is the obvious next step, and ``.json`` is not on the
    exporter's excluded-suffix list. A file the page does not load is a file nobody would notice
    shipping.
    """
    present = sorted(path.name for path in ASSETS.iterdir() if path.is_file())
    assert present == sorted(ASSET_FILES), (
        f'{ASSETS} holds {present}; it must hold exactly {sorted(ASSET_FILES)}. Player data belongs '
        'in the reader\'s downloads, not in the repository.'
    )


def test_every_selfcheck_case_reproduces_in_ppsds():
    """The page checks itself against these; ppsds must agree with them exactly."""
    cases = _load('selfcheck.json')
    assert cases, 'the self-check file is empty, so the page would validate nothing'

    kinds = set()
    for case in cases:
        kinds.add(case['kind'])
        if case['kind'] == 'score':
            _check_score_case(case)
        elif case['kind'] == 'coverage':
            _check_coverage_case(case)
        elif case['kind'] == 'drift':
            _check_drift_case(case)
        else:
            raise AssertionError(f'unknown self-check case kind {case["kind"]!r}')
    assert kinds == {'score', 'coverage', 'drift'}, (
        f'only {sorted(kinds)} are covered: a case kind with no case proves nothing'
    )


def _check_score_case(case):
    preset = presets.get(case['preset'])
    _, noise = presets.fixture_trajectory(preset, len(case['per_step_scores']))
    scores = EpsilonLogarithmScore(case['eps']).score(
        AtomPmf(np.asarray(case['atoms']), np.asarray(case['weights'])), noise)
    assert np.allclose(scores, case['per_step_scores'], atol=1e-5), (
        f"{case['preset']} eps={case['eps']}: per-step scores differ from the exported ones")
    assert scores.mean() == pytest.approx(case['mean_score'], abs=1e-5)


def _check_coverage_case(case):
    preset = presets.get(case['preset'])
    if case['layout'] == 'covering':
        expected_atoms, expected_weights = presets.covering_bet(
            preset, case['eps'], reach=_reach_from_window(preset, case))
        assert np.allclose(np.asarray(case['atoms']), expected_atoms, atol=1e-6), (
            f"{case['preset']} eps={case['eps']}: the exported layout is not the one the safe "
            'layout produces')
        assert np.allclose(np.asarray(case['weights']), expected_weights, atol=1e-6)

    atoms, weights = np.asarray(case['atoms']), np.asarray(case['weights'])
    if 'uncovered_area_fraction' in case:
        report = bet_coverage.uncovered_report(atoms, weights, case['low'], case['high'], case['eps'])
        assert report['uncovered_area_fraction'] == pytest.approx(
            case['uncovered_area_fraction'], abs=1e-12)
        assert report['worst_case_roi'] == pytest.approx(case['worst_case_roi'], abs=1e-12)
        assert report['possible_neg_inf'] == case['possible_neg_inf']
    else:
        miss = bet_coverage.gaussian_miss_probability(
            atoms, weights, case['mean'], case['cov'], case['eps'])
        assert miss == pytest.approx(case['miss_probability'], abs=1e-9)


def _reach_from_window(preset, case):
    """How far the case's window reaches, for the unbounded case where it is a choice."""
    if presets.support_box(preset) is not None:
        return None
    mean = np.asarray(preset.noise().mean, dtype=float).reshape(-1)
    return np.asarray(case['high'], dtype=float) - mean


def _check_drift_case(case):
    preset = presets.get(case['preset'])
    path = [np.asarray(case['start'], dtype=float)]
    for _ in range(case['steps']):
        path.append(np.asarray(preset.drift(path[-1], np.zeros(0)), dtype=float))
    assert np.allclose(np.asarray(path), np.asarray(case['path']), atol=1e-6), (
        f"{case['preset']}: the exported drift path is not the one the preset's drift walks. "
        'The page draws this path, so a mismatch is a table drawn with invented dynamics.'
    )


def test_the_unnormalised_case_settles_the_share_and_not_the_sum():
    """A pmf is normalised, so the score is ``log(hit / total)``.

    This is the case that would have caught the page reporting ``log(hit)``: with a hand whose
    weights sum to one the two agree, which is why every other case here stayed quiet. Both
    numbers are asserted -- the right one matches, the wrong one does not -- so the regression is
    pinned from both sides.
    """
    case = next(item for item in _load('selfcheck.json')
                if item.get('bet_name', '').startswith('weights of 2.0'))
    weights = np.asarray(case['weights'])
    assert weights.sum() > 1.5, 'this case is only interesting if the weights do not sum to one'

    _, noise = presets.fixture_trajectory(presets.get(case['preset']), len(case['per_step_scores']))
    pmf = AtomPmf(np.asarray(case['atoms']), weights)
    scores = EpsilonLogarithmScore(case['eps']).score(pmf, noise)
    assert np.allclose(scores, case['per_step_scores'], atol=1e-5)

    hit_only = np.log(np.asarray([
        pmf.weights[np.max(np.abs(pmf.atoms - draw), axis=1) <= case['eps']].sum() * weights.sum()
        for draw in noise]))
    assert not np.allclose(hit_only, case['per_step_scores'], atol=1e-5), (
        'the unnormalised reading agrees with the exported scores, so this case is not a guard')


def test_the_export_is_reproducible():
    """Running the exporter twice must give byte-identical assets, or the diff is unreadable."""
    import export_game_assets

    first = export_game_assets.export_selfcheck()
    second = export_game_assets.export_selfcheck()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_a_bet_file_scores_the_same_read_back(tmp_path):
    """The round trip: a hand exported from the page and re-scored here must be one number.

    ``run.py --bets`` is the endpoint the page's "copy my bets" button aims at, so the file's
    contents -- not the flags on the command line -- have to determine the answer.
    """
    preset = presets.get('shear')
    eps = 0.5
    atoms, weights = presets.default_bets(preset, eps)
    bet_file = tmp_path / 'my_bets.json'
    bet_file.write_text(json.dumps({
        'preset': preset.key, 'eps': eps,
        'atoms': atoms.tolist(), 'weights': weights.tolist(),
    }), encoding='utf-8')

    file_preset, file_eps, pmf = pmf_run.read_bets(bet_file)
    assert file_preset == preset.key
    assert file_eps == pytest.approx(eps)
    assert pmf.atoms == pytest.approx(atoms)

    _, noise = presets.sample_trajectories(preset, 200, 10, seed=9)
    _, aggregate = pmf_run.score(pmf, noise, eps, 200, 10)
    direct = EpsilonLogarithmScore(eps).score(pmf, noise.reshape(-1, 2)).reshape(200, 10).mean()
    assert aggregate['mean'][-1] == pytest.approx(direct)


def test_a_bet_file_missing_a_field_is_rejected(tmp_path):
    """A hand without its preset has no meaning -- guessing one would score a different world."""
    bet_file = tmp_path / 'broken.json'
    bet_file.write_text(json.dumps({'eps': 0.5, 'atoms': [[0.0, 0.0]], 'weights': [1.0]}),
                        encoding='utf-8')
    with pytest.raises(ValueError, match='preset'):
        pmf_run.read_bets(bet_file)


def test_the_exported_par_lines_match_the_library():
    """The page draws the par line; it must be the library's number, not a JavaScript guess."""
    from ppsds.predictability.bounds import predictability_upper_bound

    for key, payload in _load('presets.json').items():
        preset = presets.get(key)
        for eps_text, value in payload['upper_bounds'].items():
            assert value == pytest.approx(predictability_upper_bound(preset.noise(), float(eps_text)))


def test_the_presets_exported_match_the_presets_declared():
    """The page's picker and this registry cannot diverge -- they are one list."""
    exported = _load('presets.json')
    assert set(exported) == set(presets.PRESETS)
    for key, payload in exported.items():
        declared = presets.PRESETS[key]
        assert payload['title'] == declared.title
        assert payload['blurb'] == declared.blurb
        assert payload['motion'] == declared.motion
        assert payload['hidden_noise'] is declared.hidden_noise
        assert payload['epsilons'] == [float(v) for v in declared.epsilons]
        assert payload['x0'] == pytest.approx([float(v) for v in declared.x0])


def test_a_biased_table_exports_a_block_law():
    """The page describes the noise from the export, so the export has to carry the blocks.

    A law with ``low``/``high`` used to be labelled "uniform" by attribute-guessing, which would
    have had the page describe a three-band law as flat -- wrong, silently, in the one place a
    player looks to understand the table.
    """
    preset = presets.get('stacked_deck')
    noise = _load('presets.json')[preset.key]['noise']
    law = preset.noise()
    assert noise['kind'] == 'blocks'
    assert len(noise['boxes']) == len(law.boxes)
    assert noise['weights'] == pytest.approx([float(w) for w in law.weights])
    for (low, high), (want_low, want_high) in zip(noise['boxes'], law.boxes):
        assert low == pytest.approx([float(v) for v in want_low])
        assert high == pytest.approx([float(v) for v in want_high])


def test_a_table_that_keeps_its_law_publishes_nothing_derived_from_it():
    """Withholding has to be consistent, or it is theatre.

    The support (the covering layout is built from it), the safe-layout summary (same), the par
    line (par is ``d log 2 eps - h_d(q)``) and the scoring and coverage self-check cases (they
    carry the atoms, the support and the recorded landings, which together *are* the law) are all
    functions of the hidden law, so none of them ships. What does ship is the drift -- the puck's
    motion is visible on the table anyway -- so the page can still draw the table and check its
    dynamics against ppsds.
    """
    secret = _load('presets.json')[presets.get('house_rules').key]
    assert secret['hidden_noise'] is True
    assert secret['noise'] == {'kind': 'unpublished'}
    assert secret['support'] == {'kind': 'unpublished'}
    assert secret['upper_bounds'] == {}
    assert secret['safe'] == {}

    kinds = {case['kind'] for case in _load('selfcheck.json') if case['preset'] == 'house_rules'}
    assert kinds == {'drift'}, f'a secret table leaked {sorted(kinds - {"drift"})} cases'


def test_the_replays_are_long_enough_for_the_longest_run():
    """A run's length is the player's choice, so the replays have to cover the longest one -- and
    the first N draws of a replay must not depend on how long it was exported."""
    import export_game_assets

    trajectories = _load('trajectories.json')
    for key, payload in trajectories.items():
        assert len(payload['noise'][0]) == export_game_assets.TRAJECTORY_STEPS, key
        assert len(payload['states'][0]) == export_game_assets.TRAJECTORY_STEPS + 1, key

    # A shorter run must replay the *same* wake, not a fresh one. It does, because `run.py`
    # regenerates the full export length and slices -- generating a short one directly would give
    # different draws (scipy draws a whole axis per call), and the page and the script would
    # quietly be playing different runs.
    import run as pmf_run

    preset = presets.get('shear')
    for index, n_step in ((0, 5), (3, presets.FIXTURE_STEPS)):
        states, noise = pmf_run.trajectory_of(preset, index, n_step)
        # Absolute tolerance: the exported arrays are rounded to six decimals, and a draw near zero
        # makes a relative one meaningless.
        assert np.allclose(noise, np.asarray(trajectories['shear']['noise'][index])[:n_step], atol=1e-6)
        assert np.allclose(states, np.asarray(trajectories['shear']['states'][index])[:n_step + 1], atol=1e-6)

    # And generating a short one directly is *not* the same thing: the day this changes, the
    # slicing above becomes unnecessary rather than wrong, and this test will say so.
    short_noise = presets.sample_trajectories(
        preset, 1, presets.FIXTURE_STEPS, seed=presets.EXPORT_SEED)[1][0]
    assert not np.allclose(
        short_noise, np.asarray(trajectories['shear']['noise'][0])[:presets.FIXTURE_STEPS], atol=1e-6)


def test_the_exported_safe_layouts_match_the_library():
    """The number the page's safety badge leads with is the library's, per table and radius."""
    for key, payload in _load('presets.json').items():
        preset = presets.get(key)
        for eps_text, summary in payload['safe'].items():
            eps = float(eps_text)
            reach = (None if presets.support_box(preset) is not None
                     else presets.SAFE_WINDOW_SIGMA * np.sqrt(np.diag(preset.noise().cov)))
            meta = presets.covering_meta(preset, eps, reach=reach)
            assert summary['n_atoms'] == meta['n_atoms']
            assert summary['guaranteed'] == meta['guaranteed']
            assert summary['worst_case_roi'] == pytest.approx(meta['worst_case_roi'])
