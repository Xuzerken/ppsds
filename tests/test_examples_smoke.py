"""End-to-end smoke tests: run the shipped examples as real subprocesses.

This is the only place that checks "does the thing actually run", which unit tests structurally
cannot: they import the package in-process from the source tree, so they never see a broken
entry point, a path that only resolves from one working directory, or an import that a user's
install would not have.

Deliberately **not** run with ``-I``: that option implies ``-E`` and ``-s``, which strip
``PYTHONPATH`` -- and under a src layout that is exactly how the package is found in a
development checkout, so ``-I`` would leave the examples unable to import anything at all. A
subprocess is already a clean namespace, so the extra isolation buys nothing here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'

#: The teaching examples: one per scene, small and quick. These run by default.
FAST = sorted(EXAMPLES.rglob('quickstart.py'))

#: The heavier scene-1 pieces: they identify a model from rollouts of an off-the-shelf gym
#: environment (Acrobot is a 6-dimensional chaotic system), which costs tens of seconds. Marked
#: ``slow`` so the inner development loop stays fast, and selected explicitly in CI.
SLOW = sorted(p for name in ('with_standard_gym_env.py', 'pendulum_singular_covariance.py')
              for p in EXAMPLES.rglob(name))

FAST_TIMEOUT = 180
SLOW_TIMEOUT = 900

pytestmark = pytest.mark.skipif(
    not EXAMPLES.is_dir(),
    reason='examples/ is not next to the source tree (for instance when the tests are run '
           'against an unpacked sdist)',
)


def run_example(script: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run one example and fail loudly, quoting both streams, if it does not exit cleanly."""
    assert script.is_file(), f'{script} does not exist'
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(EXAMPLES.parent),
        capture_output=True, text=True, timeout=timeout,
    )
    assert proc.returncode == 0, (
        f'{script.name} exited with {proc.returncode}\n'
        f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}')
    assert proc.stdout.strip(), (
        f'{script.name} exited cleanly but printed nothing, which usually means it quietly '
        f'stopped early rather than that it is silent by design')
    return proc


def test_a_quickstart_exists_for_every_scene():
    """Keeps the parametrized tests below from silently becoming empty."""
    assert len(FAST) >= 2, f'expected at least one quickstart per scene, found {FAST}'
    for scene in ('scene1_own_environment', 'scene2_simulated_ambiguity'):
        assert any(scene in str(p) for p in FAST), f'no quickstart under {scene}'


@pytest.mark.parametrize('script', FAST, ids=lambda p: p.parent.name)
def test_quickstart_runs(script):
    run_example(script, FAST_TIMEOUT)


@pytest.mark.slow
@pytest.mark.parametrize('script', SLOW, ids=lambda p: p.stem)
def test_heavier_scene1_example_runs(script):
    """These are the two entry points the README sends readers to, so "it runs" matters."""
    run_example(script, SLOW_TIMEOUT)
