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

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC = REPO_ROOT / 'public'
README = PUBLIC / 'README.md'
SITE_HOME = PUBLIC / 'docs' / 'index.md'
ASSETS = PUBLIC / 'docs' / 'assets'

pytestmark = pytest.mark.skipif(
    not README.is_file(),
    reason='public/ is not present in this tree (for instance in an unpacked sdist)',
)

#: The three answers to "where do you want to start?". Prose may change; these are the reader
#: types the front page is organised around, and dropping one silently would leave a reader with
#: nowhere to go.
BRANCH_LABELS = ('just play with it', 'put it on a system of mine', 'check it against the papers')

#: How far into the README the first installable command has to appear. Not a style preference:
#: a reader who has not seen a command by here has no evidence the thing exists.
INSTALL_DEADLINE = 40

IMAGE_RE = re.compile(r'!?\[[^\]]*\]\((?:<)?([^)>\s]+)(?:>)?\)|<img[^>]+src="([^"]+)"')
FENCE_RE = re.compile(r'^```(\w*)', re.MULTILINE)


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
    assert 'demo.gif' in text, f'{path.name} does not show the animation'


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
    names = set()
    for match in IMAGE_RE.finditer(text):
        target = match.group(1) or match.group(2) or ''
        name = target.rsplit('/', 1)[-1]
        if name.endswith(('.png', '.gif', '.svg')):
            names.add(name)
    return names


def test_every_figure_the_front_doors_reference_exists():
    """A broken image on the front page is worse than no image."""
    for path in (README, SITE_HOME):
        for name in referenced_assets(text_of(path)):
            assert (ASSETS / name).is_file(), f'{path.name} references {name}, which does not exist'


def test_no_figure_is_orphaned():
    """The other direction: an asset nothing points at is dead weight in the published site."""
    if not ASSETS.is_dir():
        pytest.skip('no asset directory')
    referenced = set()
    for path in [README, SITE_HOME, *(PUBLIC / 'docs').rglob('*.md')]:
        referenced |= referenced_assets(text_of(path))
    for asset in sorted(ASSETS.iterdir()):
        if not asset.is_file() or asset.name.startswith('sidebars.'):
            continue        # sidebars.{css,js} are loaded by mkdocs.yml, not referenced by prose
        assert asset.name in referenced, (
            f'{asset.name} is committed but no page references it; either use it or remove it')


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
    missing = [p.name for p in nav_pages() if '## Where to go next' not in text_of(p)]
    assert not missing, f'pages with no "Where to go next" section: {missing}'


def test_the_map_page_is_reachable_from_the_navigation():
    """It is the one page whose whole job is to be found when the shelf order does not fit."""
    assert any(p.name == 'map.md' for p in nav_pages()), 'map.md is not in the navigation'
