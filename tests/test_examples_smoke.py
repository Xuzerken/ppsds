"""End-to-end smoke tests: run the shipped examples and experiments as real subprocesses.

This is the only place that checks "does the thing actually run", which unit tests structurally
cannot: they import the package in-process from the source tree, so they never see a broken
entry point, a path that only resolves from one working directory, or an import that a user's
install would not have.

The example directories are **classified explicitly** rather than discovered with ``rglob``.
There are two kinds now, and the distinction is a design decision, not an accident:

* a **runnable** example ships a teaching ``quickstart.py``;
* a **README-only** example's runnable half lives elsewhere -- the betting example's front is the
  no-install game page, and its code is the ``experiments/pmf_betting`` experiment.

Discovery would silently accept a new example that belongs to neither kind; the table forces the
next person to say which one it is.

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

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples'
EXPERIMENTS = ROOT / 'experiments'

#: Example directory -> the teaching script(s) it ships. Keep this in step with the site's
#: Examples section; every directory under ``examples/`` must appear in exactly one table.
RUNNABLE_EXAMPLES = {
    'apply_drpp_to_your_own_env': ('quickstart.py',),
    'understand_drpp_on_simulated_ambiguity': ('quickstart.py',),
    'understand_predictability': ('quickstart.py',),
}

#: Examples whose runnable half lives elsewhere: the betting example is the game page (no
#: install) plus ``experiments/pmf_betting``. The directory exists for its README.
README_ONLY_EXAMPLES = ('understand_ppsds_by_pmf_betting',)

#: The teaching examples: small and quick. These run by default.
FAST = sorted(
    script
    for name, scripts in RUNNABLE_EXAMPLES.items()
    for script in (EXAMPLES / name / s for s in scripts)
    if script.is_file()
)

#: The heavier pieces of the "apply DRPP to your own environment" example: they identify a model
#: from rollouts of an off-the-shelf gym environment (Acrobot is a 6-dimensional chaotic system),
#: which costs tens of seconds. Marked ``slow`` so the inner development loop stays fast, and
#: selected explicitly in CI.
SLOW = sorted(p for name in ('with_standard_gym_env.py', 'pendulum_singular_covariance.py')
              for p in EXAMPLES.rglob(name))

#: Experiment smoke cases: ``(directory, script, arguments)``. The heavy DRPP configurations are
#: deliberately **not** here -- they cost thousands of cone programs. What is checked is that each
#: entry point starts from this checkout and produces output.
EXPERIMENT_SMOKE = (
    ('pmf_betting', 'quickstart.py', ()),
    ('predictability', 'run.py', ('--n_traj', '10')),
)

#: Entry points whose module imports the plotting periphery, so ``--help`` is enough and the test
#: must skip when the ``experiments`` extra is absent (the core-only CI job).
EXPERIMENT_HELP = (('drpp', 'run.py'),)

FAST_TIMEOUT = 180
SLOW_TIMEOUT = 900

pytestmark = pytest.mark.skipif(
    not EXAMPLES.is_dir(),
    reason='examples/ is not next to the source tree (for instance when the tests are run '
           'against an unpacked sdist)',
)


def run_example(script: Path, timeout: int, args: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    """Run one entry point and fail loudly, quoting both streams, if it does not exit cleanly."""
    assert script.is_file(), f'{script} does not exist'
    proc = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(ROOT),
        capture_output=True, text=True, timeout=timeout,
    )
    assert proc.returncode == 0, (
        f'{script.name} exited with {proc.returncode}\n'
        f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}')
    assert proc.stdout.strip(), (
        f'{script.name} exited cleanly but printed nothing, which usually means it quietly '
        f'stopped early rather than that it is silent by design')
    return proc


# ----------------------------------------------------------------------
# examples/
# ----------------------------------------------------------------------

def test_every_example_directory_is_classified():
    """The tables above are exhaustive, and a README-only example really has no quickstart.

    Without this, a new example could be added and never run, or a moved script could leave a
    directory that looks runnable and is not.
    """
    present = {p.name for p in EXAMPLES.iterdir()
               if p.is_dir() and not p.name.startswith(('__', '.'))}
    classified = set(RUNNABLE_EXAMPLES) | set(README_ONLY_EXAMPLES)
    assert present == classified, (
        f'example directories and the classification tables disagree: '
        f'unclassified={sorted(present - classified)}, missing={sorted(classified - present)}')

    for name in README_ONLY_EXAMPLES:
        assert (EXAMPLES / name / 'README.md').is_file(), f'{name} has no README'
        assert not list((EXAMPLES / name).rglob('quickstart.py')), (
            f'{name} is classified README-only but now ships a quickstart; classify it as '
            f'runnable or move the script to experiments/')

    for name, scripts in RUNNABLE_EXAMPLES.items():
        for script in scripts:
            assert (EXAMPLES / name / script).is_file(), f'{name}/{script} is missing'


@pytest.mark.parametrize('script', FAST, ids=lambda p: p.parent.name)
def test_quickstart_runs(script):
    run_example(script, FAST_TIMEOUT)


@pytest.mark.slow
@pytest.mark.parametrize('script', SLOW, ids=lambda p: p.stem)
def test_heavier_scene1_example_runs(script):
    """These are the two entry points the README sends readers to, so "it runs" matters."""
    run_example(script, SLOW_TIMEOUT)


# ----------------------------------------------------------------------
# experiments/
# ----------------------------------------------------------------------

@pytest.mark.skipif(not EXPERIMENTS.is_dir(), reason='experiments/ is not in this tree')
@pytest.mark.parametrize(
    ('directory', 'script', 'args'), EXPERIMENT_SMOKE,
    ids=[d for d, _, _ in EXPERIMENT_SMOKE])
def test_experiment_entry_point_runs(directory, script, args):
    """The batch experiments' cheap paths: each starts from this checkout and prints its report."""
    if directory == 'drpp':
        pytest.importorskip('matplotlib', reason='the DRPP runner imports the plotting periphery')
    run_example(EXPERIMENTS / directory / script, FAST_TIMEOUT, args)


@pytest.mark.skipif(not EXPERIMENTS.is_dir(), reason='experiments/ is not in this tree')
@pytest.mark.parametrize(('directory', 'script'), EXPERIMENT_HELP,
                         ids=[d for d, _ in EXPERIMENT_HELP])
def test_experiment_help_runs(directory, script):
    """``--help`` proves the CLI parses even when the full run is too expensive for a smoke test."""
    pytest.importorskip('matplotlib', reason='the DRPP runner imports the plotting periphery')
    run_example(EXPERIMENTS / directory / script, FAST_TIMEOUT, ('--help',))
