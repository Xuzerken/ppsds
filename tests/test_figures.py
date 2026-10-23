"""The figures the documentation ships.

Figure assets failed silently once already: before 2026-10-16 a PNG under ``public/`` was dropped
from the export without a word, so a page could reference an image that would never reach the
published site. The default tests here cover the parts that are cheap and absolute -- the file is
a real image, and the exporter will actually carry it. The regeneration test is marked ``slow``
because it runs the example figure scripts for real; two assets are exempt from it, for reasons
that have nothing to do with this file -- see :data:`MAINTAINER_BUILT`.

**What this file deliberately does not do** is compare the committed figures byte for byte against
a fresh render. Matplotlib's PNG bytes depend on the platform's freetype and font stack, and CI
runs a Windows job as well as Linux ones, so a byte comparison would be permanently red on one of
them -- and a gate that is always red is a gate nobody reads. The regeneration test therefore
checks that the scripts still run and still produce a figure of the same **dimensions** (see
:data:`FIGURES` and the note above :func:`image_size` for why dimensions can be checked exactly).
That catches a changed layout, but not a changed number. A figure whose numbers went stale while
its shape stayed the same is caught by eyes or not at all; the numbers themselves are asserted by
the example smoke tests.

The cheap assertions stay available without the plotting extra, because the failure they exist for
-- a committed asset that the exporter silently leaves behind -- has nothing to do with matplotlib.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import pytest

from _paths import public_root

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Resolved for both layouts -- see :mod:`_paths`.
ASSETS = (public_root() or REPO_ROOT / 'public') / 'docs' / 'assets'

#: The figures generated into ``public/docs/assets/``, and the script under ``examples/`` that
#: writes each. Note that one script writes two of them -- the slow test below is parametrized by
#: **generator**, not by figure, so a generator is run once rather than once per file it produces.
FIGURES = {
    'prediction_on_your_system.png': 'examples/apply_drpp_to_your_own_env/make_figures.py',
    'when_drpp_pays.png': 'examples/apply_drpp_to_your_own_env/make_figures.py',
    'acrobot_simulation.gif': 'examples/apply_drpp_to_your_own_env/make_acrobot_gif.py',
    'drpp_ellipses.png': 'examples/understand_drpp_on_simulated_ambiguity/make_figures.py',
}

#: Assets that are committed and published but that **nothing in CI can regenerate**, because
#: their builders are maintainer tools rather than example scripts. They still get the two cheap
#: assertions below -- a real image, and carried by the exporter -- since the failure those exist
#: for (a committed asset the exporter silently leaves behind) has nothing to do with the builder.
#: They are kept out of :data:`FIGURES` only so the slow shape test does not try to run them:
#:
#: * ``papers.png``, ``paper-predictability.png`` and ``paper-drpp.png`` are **tiled from the
#:   papers' own PDFs** (``tools/make_paper_figures.py``), and need ghostscript plus the private
#:   ``papers/`` tree, which the published repository does not carry. Their inputs are vector
#:   figures that cannot go stale, which is why they are rasterisations of them rather than a
#:   re-drawing of our own.
#: * ``map.png`` is drawn by ``tools/make_map_figure.py``. It is a design asset, not a result.
#: * ``media/game-demo-poster.png`` is a frame of the narrated demo, recorded in the talks project
#:   (outside this repository). The MP4 beside it is checked by
#:   :func:`test_the_demo_video_reaches_the_export`.
MAINTAINER_BUILT = ('papers.png', 'paper-predictability.png', 'paper-drpp.png', 'map.png',
                    'media/game-demo-poster.png')

#: Everything that has to be a real, exportable image, whichever way it was built.
IMAGES = tuple(sorted(set(FIGURES) | set(MAINTAINER_BUILT)))

#: Unique generators, in a stable order.
GENERATORS = tuple(dict.fromkeys(FIGURES.values()))

#: The figures each generator owns.
OUTPUTS = {generator: [name for name, g in FIGURES.items() if g == generator]
           for generator in GENERATORS}

pytestmark = pytest.mark.skipif(
    public_root() is None,
    reason='this tree has no published content (for instance an unpacked sdist)',
)


def image_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` from an image header, without needing an image library.

    Both formats the generators write, because both are checked the same way: PNG stores them
    big-endian at byte 16, GIF little-endian at byte 6. Reading the header rather than the image
    keeps this test free of a plotting dependency, which is the point -- it has to run in the job
    that installs no extras.
    """
    header = path.read_bytes()[:24]
    if header[:8] == b'\x89PNG\r\n\x1a\n':
        return struct.unpack('>II', header[16:24])
    if header[:6] in (b'GIF87a', b'GIF89a'):
        return struct.unpack('<HH', header[6:10])
    raise AssertionError(f'{path.name} is neither a PNG nor a GIF')


#: Why the dimension check below can be exact.
#:
#: It can, but only because the generators were changed to make it possible. Matplotlib's PNG
#: dimensions are ``figsize * dpi`` **only when nothing crops the canvas** -- and the natural
#: thing to write is ``bbox_inches='tight'``, which crops to the drawn content. How much gets
#: cropped depends on font metrics, so a tight-bbox figure has platform-dependent dimensions: it
#: would come out a few pixels different on a headless Linux runner than on the machine that
#: generated it, and this test would be red there for no reason at all.
#:
#: The generators therefore pass ``bbox_inches=None`` and rely on ``tight_layout()``, which lays
#: the axes out inside a canvas of exactly ``figsize``. That trades a little whitespace trimming
#: for a figure whose size is a property of the code rather than of the machine that ran it --
#: which is what makes the assertion below worth having.


@pytest.mark.parametrize('name', IMAGES)
def test_each_figure_is_a_real_image_with_sane_dimensions(name):
    """A truncated or empty file would still be *present*, which is what a weaker check sees."""
    path = ASSETS / name
    assert path.is_file(), f'{name} is missing; run its generator'
    width, height = image_size(path)
    assert width > 400 and height > 300, f'{name} is {width}x{height}, too small to be the figure'
    assert path.stat().st_size > 10_000, f'{name} is suspiciously small'


@pytest.mark.parametrize('name', IMAGES)
def test_each_figure_would_actually_be_exported(name):
    """The failure this file exists for: a committed asset the exporter silently leaves behind."""
    tools = REPO_ROOT / 'tools'
    if not (tools / 'export_public.py').is_file():
        pytest.skip('tools/export_public.py is not in this tree')
    sys.path.insert(0, str(tools))
    import export_public as ep

    assert f'docs/assets/{name}' in ep.iter_export(), (
        f'{name} is committed but would not be exported -- the published page would show a broken '
        f'image while `--check` reported a match')


def test_the_demo_video_reaches_the_export():
    """The front page's demo is not an image, so it is outside the parametrised checks above --
    but it has the same silent failure: ``.mp4`` is not excluded today, and the day someone adds
    it to ``EXCLUDED_SUFFIXES`` (binary formats the leak scan cannot read) the published page
    would show a dead player while ``--check`` reported a match."""
    tools = REPO_ROOT / 'tools'
    if not (tools / 'export_public.py').is_file():
        pytest.skip('tools/export_public.py is not in this tree')
    sys.path.insert(0, str(tools))
    import export_public as ep

    assert 'docs/assets/media/game-demo.mp4' in ep.iter_export(), (
        'the demo video is committed but would not be exported')


@pytest.mark.slow
@pytest.mark.parametrize('generator', GENERATORS, ids=lambda value: Path(value).parent.name)
def test_the_generator_still_produces_figures_of_the_same_shape(generator):
    """Runs the real generator once and compares the shape of everything it wrote.

    Same shape, not same bytes -- see the module docstring for why. A generator that has stopped
    working, or a figure whose layout changed, both fail here.
    """
    # The generators draw, so they need the plotting extra -- and this file's *other* tests do
    # not, which is deliberate: "the committed figure is exportable" is a core-only concern and
    # has to keep working in the job that installs no extras.
    pytest.importorskip('matplotlib', reason='the figure generators need ppsds[experiments]')

    script = REPO_ROOT / generator
    assert script.is_file(), f'{generator} is missing'
    before = {name: image_size(ASSETS / name) for name in OUTPUTS[generator]}

    proc = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f'{generator} exited with {proc.returncode}\n'
        f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}')

    for name, was in before.items():
        now = image_size(ASSETS / name)
        assert now == was, (
            f'{name} came back as {now[0]}x{now[1]} but is committed as {was[0]}x{was[1]}. '
            f'Dimensions are meant to be deterministic -- if this failed on another machine, '
            f'check that the generator still passes `bbox_inches=None` (see the note at the top '
            f'of this file). If the layout changed on purpose, commit the regenerated file.')
