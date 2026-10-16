"""The figures the documentation ships.

Figure assets failed silently once already: before 2026-10-16 a PNG under ``public/`` was dropped
from the export without a word, so a page could reference an image that would never reach the
published site. The default tests here cover the parts that are cheap and absolute -- the file is
a real PNG, and the exporter will actually carry it. The regeneration test is marked ``slow``
because it runs the two figure scripts for real.

**What this file deliberately does not do** is compare the committed figures byte for byte against
a fresh render. Matplotlib's PNG bytes depend on the platform's freetype and font stack, and CI
runs a Windows job as well as Linux ones, so a byte comparison would be permanently red on one of
them -- and a gate that is always red is a gate nobody reads. The regeneration test therefore
checks that the scripts still run and still produce the same *dimensions*, which catches a changed
layout but not a changed number. A figure whose numbers went stale while its size stayed the same
is caught by eyes or not at all; the numbers themselves are asserted by the example smoke tests.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / 'public' / 'docs' / 'assets'

#: The figures generated into ``public/docs/assets/``, and the script that writes each. Note that
#: one script writes two of them -- the slow test below is parametrized by **generator**, not by
#: figure, so a generator is run once rather than once per file it happens to produce.
#: ``demo.gif`` is not here: the betting story generates it, and it is a GIF rather than a PNG.
FIGURES = {
    'prediction_on_your_system.png': 'examples/apply_drpp_to_your_own_env/make_figures.py',
    'when_drpp_pays.png': 'examples/apply_drpp_to_your_own_env/make_figures.py',
    'predictability.png': 'examples/understand_predictability/make_figures.py',
}

#: Unique generators, in a stable order.
GENERATORS = tuple(dict.fromkeys(FIGURES.values()))

#: The figures each generator owns.
OUTPUTS = {generator: [name for name, g in FIGURES.items() if g == generator]
           for generator in GENERATORS}

pytestmark = pytest.mark.skipif(
    not ASSETS.is_dir(),
    reason='public/ is not present in this tree (for instance when the tests run against an '
           'unpacked sdist)',
)


def png_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` from a PNG's header, without needing an image library."""
    header = path.read_bytes()[:24]
    assert header[:8] == b'\x89PNG\r\n\x1a\n', f'{path.name} is not a PNG'
    width, height = struct.unpack('>II', header[16:24])
    return width, height


@pytest.mark.parametrize('name', sorted(FIGURES))
def test_each_figure_is_a_real_png_with_sane_dimensions(name):
    """A truncated or empty file would still be *present*, which is what a weaker check sees."""
    path = ASSETS / name
    assert path.is_file(), f'{name} is missing; run its generator'
    width, height = png_size(path)
    assert width > 400 and height > 300, f'{name} is {width}x{height}, too small to be the figure'
    assert path.stat().st_size > 10_000, f'{name} is suspiciously small'


@pytest.mark.parametrize('name', sorted(FIGURES))
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


@pytest.mark.slow
@pytest.mark.parametrize('generator', GENERATORS, ids=lambda value: Path(value).parent.name)
def test_the_generator_still_produces_figures_of_the_same_shape(generator):
    """Runs the real generator once and compares the dimensions of everything it wrote.

    Same shape, not same bytes -- see the module docstring for why. A generator that has stopped
    working, or a figure whose layout changed, both fail here.
    """
    script = REPO_ROOT / generator
    assert script.is_file(), f'{generator} is missing'
    before = {name: png_size(ASSETS / name) for name in OUTPUTS[generator]}

    proc = subprocess.run([sys.executable, str(script)], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        f'{generator} exited with {proc.returncode}\n'
        f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}')

    for name, was in before.items():
        now = png_size(ASSETS / name)
        assert now == was, (
            f'{name} came back as {now} but is committed as {was}. If the layout changed on '
            f'purpose, commit the regenerated file.')
