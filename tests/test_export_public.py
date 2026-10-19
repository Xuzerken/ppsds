"""The export whitelist's *scope* rules -- which files travel to the public repository.

``tools/export_public.py`` answers "is what would be exported allowed to be public?", and it is a
release gate rather than a test: it is not shipped and it is not part of the package. This file
checks the smaller, sharper thing that can be asserted: that the suffix exclusion means what it
is supposed to mean, and in particular that it is **scoped**.

That scope is worth a test because getting it wrong fails *silently in the dangerous direction*.
Before 2026-10-16 the exclusion applied to ``public/`` as well, so a PNG placed there for a
published page was dropped without a word: the page shipped with a broken image, ``--check``
compared the trees and reported a match, and every gate stayed green. A rule whose failure mode is
"nothing happens, quietly" is a rule that needs an assertion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from _paths import public_root

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / 'tools'


@pytest.fixture(scope='module')
def ep():
    """``tools/export_public.py``, imported **lazily**.

    Lazily, not at module scope, and that distinction is the whole point of this fixture. A
    module-level ``import export_public`` under a ``pytestmark`` skipif **still runs** -- the
    skipif only decides whether the tests execute, and by then the module has already failed to
    import. This script is not shipped to the public repository (only ``ab_compare.py``,
    ``check_links.py`` and ``README.md`` are), so the import raised ``ModuleNotFoundError`` there
    and took the *entire* pytest run down as a collection error -- every job that runs the suite,
    for a file none of them needed. The guard has to be inside the thing being guarded.
    """
    if not (TOOLS / 'export_public.py').is_file():
        pytest.skip('tools/export_public.py is not in this tree -- the public repository ships '
                    'only part of tools/, so there is no export whitelist to test against')
    sys.path.insert(0, str(TOOLS))
    import export_public
    return export_public


# ----------------------------------------------------------------------
# The rule itself
# ----------------------------------------------------------------------

def test_public_release_assets_are_exported_but_experiment_output_is_not(ep):
    """The whole point of the carve-out: it depends on **where** the file is, not what it is."""
    # A hand-authored release asset: exported.
    assert not ep.should_skip('public/docs/assets/figure.png')
    # Regenerated experiment output, wherever it sits: never exported.
    assert ep.should_skip('examples/understand_predictability/predictability.png')
    assert ep.should_skip('examples/understand_drpp_on_simulated_ambiguity/data/gym/plot.png')
    assert ep.should_skip('experiments/drpp/data/gym/plot.png')
    # The carve-out is for `public/` only -- a PNG at the repository root is not swept in with it.
    assert ep.should_skip('figure.png')


def test_the_carve_out_is_narrow(ep):
    """``.png`` was added; nothing else was. Binary formats the leak scan cannot read stay out."""
    for quiet in ('.pdf', '.pkl', '.npy', '.npz', '.pyc'):
        assert ep.should_skip(f'public/docs/assets/thing{quiet}'), \
            f'{quiet} must not be exported just because it sits under public/'
    assert ep.should_skip('public/docs/assets/thing.pdf')


def test_ordinary_files_are_still_exported(ep):
    """Guards against a carve-out that accidentally swallows the normal case."""
    for rel in ('src/ppsds/__init__.py',
                'examples/apply_drpp_to_your_own_env/quickstart.py',
                'experiments/drpp/run.py',
                'public/README.md',
                'public/docs/research/index.md'):
        assert not ep.should_skip(rel), f'{rel} should be exported'


# ----------------------------------------------------------------------
# The rule as it is actually applied
# ----------------------------------------------------------------------

def test_a_png_under_public_really_does_reach_the_export(ep):
    """End to end through :func:`iter_export`, not just the predicate.

    Writes a throwaway asset, asks the exporter what it would ship, and removes it again. A
    predicate that is right while the walk that calls it still filters by name would pass the
    tests above and fail here -- which is exactly the bug this is here to catch.
    """
    asset_dir = (public_root() or REPO_ROOT / 'public') / 'docs' / 'assets'
    stray_public = asset_dir / '__export_probe__.png'
    stray_experiment = REPO_ROOT / 'examples' / '__export_probe__.png'
    stray_experiments_tree = REPO_ROOT / 'experiments' / '__export_probe__.png'
    stray_public.write_bytes(b'\x89PNG\r\n\x1a\n')
    stray_experiment.write_bytes(b'\x89PNG\r\n\x1a\n')
    stray_experiments_tree.write_bytes(b'\x89PNG\r\n\x1a\n')
    try:
        exported = ep.iter_export()
        assert 'docs/assets/__export_probe__.png' in exported, (
            'a PNG under public/ was dropped from the export -- the published page would 404 it '
            'while `--check` reported a match')
        assert 'examples/__export_probe__.png' not in exported, (
            'a PNG under examples/ reached the export -- experiment output must never ship')
        assert 'experiments/__export_probe__.png' not in exported, (
            'a PNG under experiments/ reached the export -- experiment output must never ship')
    finally:
        stray_public.unlink(missing_ok=True)
        stray_experiment.unlink(missing_ok=True)
        stray_experiments_tree.unlink(missing_ok=True)


def test_every_committed_release_asset_is_in_the_export(ep):
    """The non-vacuous version of the check above: whatever is really there must really travel.

    This is the assertion that would have caught the original bug against the *existing* tree
    rather than against a probe, and it keeps working as figures are added.
    """
    asset_dir = REPO_ROOT / 'public' / 'docs' / 'assets'
    if not asset_dir.is_dir():
        pytest.skip('no release-asset directory in this tree')

    exported = ep.iter_export()
    for path in sorted(asset_dir.iterdir()):
        if not path.is_file() or path.name.startswith('sidebars.'):
            continue        # sidebars.{css,js} travel as ordinary web assets
        rel = f'docs/assets/{path.name}'
        assert rel in exported, f'{rel} is committed but would not be exported'


def test_the_game_page_is_scanned_for_embargoed_tokens(ep):
    """The leak scan has to *read* the front end, not just copy it around it.

    Until 2026-10-16 ``.js``/``.html`` were missing from ``TEXT_SUFFIXES``, so the game page --
    the largest hand-written thing in the repository -- was exported to the public repository
    without any gate ever looking inside it.
    """
    assert '.js' in ep.TEXT_SUFFIXES
    assert '.html' in ep.TEXT_SUFFIXES

    scanned = ep.scanned_files(ep.iter_export())
    assert any(name.endswith('.js') for name in scanned), \
        'no JavaScript is being scanned, so the suffix list has drifted again'
