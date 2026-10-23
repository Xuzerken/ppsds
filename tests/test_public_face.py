"""The shape of the public front door: the README and the documentation home page.

These are the two things a stranger reads first, and they are the two things with no natural
assertion attached to them -- a test suite proves the library works, not that anyone can tell what
it is for. Everything here is a rule someone wrote down on purpose (see
``docs/strategy/MARKETING.md``), so everything here is a rule that can silently rot. The checks are
deliberately about **shape**, not about wording: a test that pinned the sentences would make
editing them a chore, which is the opposite of what a front door needs.

The one rule worth spelling out is the first: the README and the site home page open with the
**same** centre sentence, and it is asserted rather than hoped for, because two hand-maintained
copies of the same paragraph drift. They had already drifted once before this existed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from _paths import public_root

#: ``public/`` in the development repository, the repository root on the public side, and
#: ``None`` in a tree that has neither (an sdist) -- see :mod:`_paths`.
PUBLIC = public_root()
README = PUBLIC / 'README.md' if PUBLIC else None
SITE_HOME = PUBLIC / 'docs' / 'index.md' if PUBLIC else None
ASSETS = PUBLIC / 'docs' / 'assets' if PUBLIC else None

pytestmark = pytest.mark.skipif(
    PUBLIC is None,
    reason='this tree has no published content (for instance an unpacked sdist)',
)

#: The reader types the front page is organised around. Prose may change; dropping one silently
#: would leave a reader with nowhere to go. The four sections (Examples / Experiments / Research /
#: Tutorials) are reached through these rows.
BRANCH_LABELS = ('just curious', 'build it yourself', 'your system',
                 'only data', 'reproduce a paper', 'use it on your system', 'add a family',
                 'add a predictor kind', 'do research on the floor', 'work on the ceiling',
                 'found a problem while playing', 'contribute what you built',
                 'discuss a research collaboration')

#: How far into the README the first installable command has to appear. Not a style preference:
#: a reader who has not seen a command by here has no evidence the thing exists.
INSTALL_DEADLINE = 40

IMAGE_RE = re.compile(r'!?\[[^\]]*\]\((?:<)?([^)>\s]+)(?:>)?\)|<img[^>]+src="([^"]+)"')
#: ``src``/``poster`` attributes of the elements that carry media. The README cannot embed a
#: ``<video>`` at all (GitHub's sanitiser strips the tag unless the file was uploaded as an
#: attachment), so it shows the poster and links to the MP4, while the site plays it. Both
#: spellings still have to resolve to a file the export actually carries.
MEDIA_SRC_RE = re.compile(r'<(?:img|source|video)[^>]+(?:src|poster)="([^"]+)"')
FENCE_RE = re.compile(r'^```(\w*)', re.MULTILINE)
ASSET_SUFFIXES = ('.png', '.gif', '.svg', '.mp4', '.webm')


def text_of(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def opening_paragraph(text: str) -> str:
    """The first paragraph after the H1, whitespace-normalised."""
    body = text.split('\n', 1)[1] if text.startswith('#') else text
    for block in re.split(r'\n\s*\n', body):
        block = block.strip()
        if block and not block.startswith(('#', '![', '<', '|', '-')):
            return ' '.join(block.split())
    raise AssertionError('no opening paragraph found')


# ----------------------------------------------------------------------
# The two front doors agree
# ----------------------------------------------------------------------

def test_the_readme_and_the_site_home_open_with_the_same_sentence():
    """Two hand-maintained copies of one paragraph is a drift bug waiting to happen."""
    assert opening_paragraph(text_of(README)) == opening_paragraph(text_of(SITE_HOME)), (
        'the README and docs/index.md open with different text. They are meant to be the same '
        'centre sentence in both places; edit one and copy it to the other.')


@pytest.mark.parametrize('label', BRANCH_LABELS)
def test_both_front_doors_offer_every_branch(label):
    """A reader who is not the default reader has to be able to see themselves on the front page."""
    for path in (README, SITE_HOME):
        assert label in text_of(path).lower(), f'{path.name} no longer offers the "{label}" branch'


@pytest.mark.parametrize('path', [README, SITE_HOME], ids=['README', 'index'])
def test_both_front_doors_link_to_the_game(path):
    """The game is the entry point for the largest group of readers; a dead link is the whole
    funnel gone."""
    text = text_of(path)
    assert 'understand_ppsds_by_pmf_betting' in text, f'{path.name} does not link to the game'
    assert 'game-demo' in text, f'{path.name} does not show the demo'


# ----------------------------------------------------------------------
# The README is a front door, not a whitepaper
# ----------------------------------------------------------------------

def test_the_install_command_arrives_early():
    lines = text_of(README).splitlines()
    for index, line in enumerate(lines[:INSTALL_DEADLINE], start=1):
        if line.strip().startswith('pip install ppsds'):
            return
    raise AssertionError(
        f'no `pip install ppsds` in the first {INSTALL_DEADLINE} lines of the README; it is at '
        f'line {next((i for i, line in enumerate(lines, 1) if "pip install ppsds" in line), None)}')


def test_the_first_code_block_is_something_you_can_run():
    """The first block a reader meets must be a command, not an abstract formula.

    This is the concrete form of "say what it is good for before saying what it is": an equation
    as the first block is the whitepaper opening this README was rewritten to stop being.
    """
    text = text_of(README)
    match = FENCE_RE.search(text)
    assert match, 'the README has no fenced code block at all'
    block = text[match.end():text.index('```', match.end())]
    first_line = next((line.strip() for line in block.splitlines() if line.strip()), '')
    assert first_line.startswith(('pip install', 'python ', 'conda ')), (
        f'the README opens with a code block starting "{first_line[:40]}". The first block should '
        f'be a command a reader can paste.')


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def referenced_assets(text: str) -> set[str]:
    """Every asset a page references, as a path relative to ``docs/assets/``.

    The README points at absolute URLs (GitHub and PyPI resolve nothing relative) while site
    pages point at ``assets/...``; both spellings are folded to the same path so the existence
    check below tests the file the export actually carries. Paths matter now that the demo lives
    in ``docs/assets/media/``: keeping only the basename would look for it at the top of the
    asset directory and report a false failure.
    """
    targets = [match.group(1) or match.group(2) for match in IMAGE_RE.finditer(text)]
    targets += MEDIA_SRC_RE.findall(text)
    names = set()
    for target in targets:
        if not target.lower().endswith(ASSET_SUFFIXES):
            continue
        for marker in ('docs/assets/', '/assets/', 'assets/'):
            if marker in target:
                target = target.split(marker, 1)[1]
                break
        else:
            continue
        names.add(target)
    return names


def test_every_figure_the_front_doors_reference_exists():
    """A broken image on the front page is worse than no image."""
    for path in (README, SITE_HOME):
        for name in referenced_assets(text_of(path)):
            assert (ASSETS / name).is_file(), f'{path.name} references {name}, which does not exist'


#: The front page's hero. Not a figure: it is a narrated recording made outside this repository
#: (see the README of the talks project), so it has no generator here and lives under ``media/``.
DEMO_MEDIA = ('game-demo.mp4', 'game-demo-poster.png')


@pytest.mark.parametrize('name', DEMO_MEDIA)
def test_the_demo_media_ships_with_the_front_doors(name):
    path = ASSETS / 'media' / name
    assert path.is_file(), f'{name} is missing from docs/assets/media'
    assert path.stat().st_size > 10_000, f'{name} is suspiciously small'


def test_the_site_plays_the_demo_and_the_readme_offers_it():
    """One demo, two renderers: the site plays the video; GitHub strips ``<video>``, so the README
    shows its poster and links to the MP4. Both front doors must offer the demo one way or the
    other -- a front door that lost it is the whole funnel gone."""
    site = text_of(SITE_HOME)
    assert '<video' in site, 'the site home page no longer embeds the demo video'
    assert 'media/game-demo.mp4' in site, 'the site video does not point at the MP4'
    readme = text_of(README)
    assert 'media/game-demo.mp4' in readme, 'the README no longer links to the demo video'
    assert 'media/game-demo-poster.png' in readme, 'the README shows no poster for the demo'


def assets_mkdocs_loads() -> set[str]:
    """The ``docs/assets/`` files ``mkdocs.yml`` pulls in through ``extra_css``/``extra_javascript``.

    Read as text rather than parsed: the only thing wanted is which asset filenames appear, and a
    YAML dependency for that would be a dependency added to the test environment for nothing.
    """
    config = (PUBLIC / 'mkdocs.yml').read_text(encoding='utf-8')
    return set(re.findall(r'assets/([A-Za-z0-9_.-]+\.(?:css|js))', config))


#: A stylesheet is pulled in by `mkdocs.yml`, not by a link in the prose, so "no page references
#: it" is not a fault for these -- but "nothing loads it" is, and that is what the test below
#: checks instead. The old version of this test hardcoded `sidebars.`, which meant the next file
#: added to `extra_css` would have failed as an orphan until someone edited the test.
LOADED_BY_CONFIG = assets_mkdocs_loads()


def test_no_figure_is_orphaned():
    """The other direction: an asset nothing points at is dead weight in the published site."""
    if not ASSETS.is_dir():
        pytest.skip('no asset directory')
    referenced = set()
    for path in [README, SITE_HOME, *(PUBLIC / 'docs').rglob('*.md')]:
        referenced |= referenced_assets(text_of(path))
    for asset in sorted(ASSETS.iterdir()):
        if not asset.is_file() or asset.name in LOADED_BY_CONFIG:
            continue
        assert asset.name in referenced, (
            f'{asset.name} is committed but nothing uses it -- neither a page nor mkdocs.yml; '
            f'either use it or remove it')


def test_the_front_page_cards_stylesheet_still_matches_its_heading():
    """``cards.css`` selects the card table by the heading's id, which nothing else enforces.

    The rules that give the front page's three pictures equal columns are anchored on
    ``#where-do-you-want-to-start``, the slug MkDocs derives from the heading. Rename the heading
    and the stylesheet goes on existing, goes on being loaded, and silently stops applying -- the
    pictures simply go back to three different widths, which is the bug it was written for. The
    slug is derived here the same way MkDocs derives it, so the failure names the new id.
    """
    heading = next((line for line in text_of(SITE_HOME).splitlines()
                    if line.startswith('## ') and 'start' in line.lower()), None)
    assert heading, 'the front page no longer has a "where do you want to start" heading'
    slug = re.sub(r'[^a-z0-9 -]', '', heading[3:].strip().lower()).replace(' ', '-')

    stylesheet = (ASSETS / 'cards.css').read_text(encoding='utf-8')
    assert f'#{slug}' in stylesheet, (
        f'the heading now slugs to "{slug}" but cards.css still selects '
        f'{re.findall(r"#[a-z0-9-]+", stylesheet)}; update the selector or the heading')
    assert 'cards.css' in LOADED_BY_CONFIG, 'cards.css is not loaded by mkdocs.yml any more'


# ----------------------------------------------------------------------
# The site keeps its promises
# ----------------------------------------------------------------------

def nav_pages() -> list[Path]:
    """Every ``.md`` the navigation points at."""
    config = (PUBLIC / 'mkdocs.yml').read_text(encoding='utf-8')
    nav = config.split('\nnav:', 1)[1]
    return [PUBLIC / 'docs' / name for name in re.findall(r'[\w./-]+\.md', nav)]


def test_every_navigable_page_says_where_to_go_next():
    """The sidebar is a map; a page that ends without a next step drops the reader."""
    missing = [p.name for p in nav_pages() if '## Next' not in text_of(p)]
    assert not missing, f'pages with no "Where to go next" section: {missing}'


def test_the_map_page_is_reachable_from_the_navigation():
    """It is the one page whose whole job is to be found when the shelf order does not fit."""
    assert any(p.name == 'map.md' for p in nav_pages()), 'map.md is not in the navigation'


def test_every_page_in_the_tree_is_in_the_navigation():
    """A page no nav entry points at is built but unreachable.

    The migration to the four-section tree moves files; without this, a page could be moved and
    left out of the nav, and the site would build green with a hole in it.
    """
    docs = PUBLIC / 'docs'
    pages = {p.relative_to(docs).as_posix() for p in docs.rglob('*.md')}
    nav = {p.relative_to(docs).as_posix() for p in nav_pages()}
    assert pages == nav, (
        f'pages missing from the nav: {sorted(pages - nav)}; '
        f'nav entries with no page: {sorted(nav - pages)}')
