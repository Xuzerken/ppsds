"""The one architectural invariant that is otherwise enforced only by convention.

``ppsds`` is a single distribution with a **two-layer** split inside it:

=========================================  ==============================================
core                                       experimental periphery
=========================================  ==============================================
``ambiguity`` ``dynamics``                 ``experiment`` ``visualization``
``probabilistic_predictor``
``environment`` ``numerics``
``scoring_rule`` ``control_policy``
deps: numpy / scipy / cvxpy / gymnasium    extra deps: matplotlib / pandas / seaborn / tqdm
=========================================  ==============================================

The criterion is the **dependency closure**: if a core module imported a periphery module, then
``pip install ppsds`` would drag matplotlib into every user's closure. Nothing stops that today
except this test and the ``core-only`` CI job, which installs the package *without* the extra.

Two independent kinds of evidence below: a static AST scan (which catches an import that is never
executed in the common path) and a subprocess check (which catches an import that happens
transitively, through some other module).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / 'src' / 'ppsds'

#: Subpackages allowed to depend on the plotting/experiment extra.
PERIPHERY = ('experiment', 'visualization')

#: Third-party distributions that only the periphery may import.
FORBIDDEN = ('matplotlib', 'pandas', 'seaborn', 'tqdm')

#: Importing any of these must not pull the above in. ``cvxpy`` is the interesting entry: it is a
#: hard dependency of the *distribution* but is imported lazily inside ``eigvec_fixed_minmax``,
#: so every user who only wants NominalPredictor or the environment layer does not pay for it
#: (nor for its solvers: osqp, clarabel, scs, ecos).
CORE_ENTRY_POINTS = (
    'ppsds',
    'ppsds.environment',
    'ppsds.ambiguity.additive_drift.conic_moment',
    'ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment',
    'ppsds.scoring_rule.logarithm_score',
    'ppsds.scoring_rule.epsilon_logarithm_score',
    'ppsds.numerics.linalg',
    'ppsds.numerics.atom_pmf',
    'ppsds.dynamics.known_sds',
    'ppsds.probabilistic_predictor.pmf.atom',
    'ppsds.predictability.bounds',
    'ppsds.predictability.entropy',
    'ppsds.predictability.trajectory',
)


def _top_level_imports(path):
    """Top-level module names imported by ``path`` (relative imports resolve to ``ppsds``)."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split('.')[0])
    return found


def test_the_scan_actually_looks_at_something():
    """Guards against the scan silently going vacuous (a moved source root, a renamed package).

    A dependency-boundary test that examines zero files passes forever and protects nothing.
    """
    files = list(SOURCE_ROOT.rglob('*.py'))
    assert len(files) > 30, f'only found {len(files)} modules under {SOURCE_ROOT}; is the path right?'
    assert (SOURCE_ROOT / '__init__.py').is_file()


def test_only_the_periphery_imports_the_experimental_dependencies():
    offenders = {}
    for path in sorted(SOURCE_ROOT.rglob('*.py')):
        relative = path.relative_to(SOURCE_ROOT)
        if relative.parts[0] in PERIPHERY:
            continue
        hit = _top_level_imports(path) & set(FORBIDDEN)
        if hit:
            offenders[str(relative)] = sorted(hit)

    assert offenders == {}, (
        'these core modules import the experimental periphery, which would put matplotlib and '
        'friends into the dependency closure of `pip install ppsds`:\n'
        + '\n'.join(f'  {path}: {mods}' for path, mods in sorted(offenders.items()))
    )


def test_the_periphery_does_import_them():
    """The converse, so the check above cannot pass by the extra becoming unused.

    If the periphery stopped importing matplotlib, ``FORBIDDEN`` would be measuring nothing and
    the boundary would be vacuous in the other direction. (Deliberately not asserted per-file:
    which periphery module imports what is free to change.)
    """
    seen = set()
    for path in SOURCE_ROOT.rglob('*.py'):
        if path.relative_to(SOURCE_ROOT).parts[0] in PERIPHERY:
            seen |= _top_level_imports(path)
    assert seen & set(FORBIDDEN), (
        'no periphery module imports anything from the experimental extra any more; either the '
        'boundary moved or this test is no longer measuring what it claims')


@pytest.mark.parametrize('module', CORE_ENTRY_POINTS)
def test_importing_the_core_does_not_load_the_heavy_dependencies(module):
    """The runtime half, in a clean interpreter.

    Note the absence of ``-I``: it implies ``-E`` and ``-s``, which strip ``PYTHONPATH`` -- and
    under a src layout that is exactly how the package is found in a development checkout, so
    ``-I`` would make this test unable to import anything at all. Isolation is not needed here:
    what is being measured is the *set of loaded modules* after one import, and a fresh
    subprocess is already a clean namespace.
    """
    code = (
        'import importlib, sys\n'
        f'importlib.import_module({module!r})\n'
        'heavy = [name for name in ("matplotlib", "pandas", "seaborn", "tqdm", "cvxpy")\n'
        '         if name in sys.modules]\n'
        'print(",".join(heavy))\n'
    )
    proc = subprocess.run([sys.executable, '-c', code],
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, f'importing {module} failed:\n{proc.stderr}'
    assert proc.stdout.strip() == '', (
        f'importing {module} pulled in {proc.stdout.strip()}; the core must stay importable '
        f'with only numpy/scipy/gymnasium installed')
