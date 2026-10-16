"""Distribution-artifact tests: build the sdist and the wheel, then actually use them.

Marked ``packaging`` because building shells out to ``python -m build``, which creates an
isolated build environment and downloads the backend -- far slower than the rest of the suite,
and dependent on the network. CI runs these in the ``build`` job, not in the matrix.

What these catch that nothing else can: the wheel is a **different artifact from the source
tree**. An editable install resolves every import through ``src/``, so it can never reveal a
file that is not packaged, a module the wheel omits, or a README that a PyPI page would render
into something else.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.packaging

BUILD_TIMEOUT = 1800
RUN_TIMEOUT = 300

#: The embargo list lives outside every file that gets scanned, and outside everything that gets
#: published -- see the header of the file itself. It is absent in the public repository, so this
#: gate cannot run there; that is expected rather than a gap, since there is nothing to forbid.
TOKEN_FILE = 'tools/embargo_tokens.txt'


def forbidden_tokens(repo_root: Path) -> list[str]:
    path = repo_root / TOKEN_FILE
    if not path.is_file():
        return []
    return [line.strip().lower() for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.strip().startswith('#')]


def _source_version(repo_root: Path) -> str:
    text = (repo_root / 'src' / 'ppsds' / '__init__.py').read_text(encoding='utf-8')
    match = re.search(r'^__version__ = "([^"]+)"', text, re.MULTILINE)
    assert match, 'could not find __version__ in src/ppsds/__init__.py'
    return match.group(1)


def _run(cmd, **kwargs):
    kwargs.setdefault('capture_output', True)
    kwargs.setdefault('text', True)
    kwargs.setdefault('timeout', RUN_TIMEOUT)
    proc = subprocess.run(cmd, **kwargs)
    assert proc.returncode == 0, (
        f'{cmd} exited with {proc.returncode}\n'
        f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}')
    return proc


@pytest.fixture(scope='module')
def artifacts(tmp_path_factory, repo_root):
    """Build once per module and hand back ``(wheel, sdist)``."""
    out = tmp_path_factory.mktemp('dist')
    proc = subprocess.run([sys.executable, '-m', 'build', '--outdir', str(out)],
                          cwd=str(repo_root), capture_output=True, text=True,
                          timeout=BUILD_TIMEOUT)
    assert proc.returncode == 0, (
        f'python -m build failed\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}')

    wheels, sdists = sorted(out.glob('*.whl')), sorted(out.glob('*.tar.gz'))
    assert len(wheels) == 1, f'expected one wheel, got {wheels}'
    assert len(sdists) == 1, f'expected one sdist, got {sdists}'
    return wheels[0], sdists[0]


# ----------------------------------------------------------------------
# The wheel: package payload only
# ----------------------------------------------------------------------

def test_the_wheel_contains_only_the_package(artifacts):
    """No docs, no examples, no tests, no dev records -- the wheel is the installed payload."""
    wheel, _ = artifacts
    names = zipfile.ZipFile(wheel).namelist()
    top_level = {name.split('/')[0] for name in names}
    dist_info = next(name for name in top_level if name.endswith('.dist-info'))

    assert top_level == {'ppsds', dist_info}, f'unexpected top-level entries: {top_level}'
    assert any(name == f'{dist_info}/licenses/LICENSE' for name in names), \
        'the licence text must travel inside the wheel'
    assert all(name.startswith(('ppsds/', f'{dist_info}/')) for name in names)


def test_the_wheel_metadata_declares_what_the_project_does(artifacts, repo_root):
    wheel, _ = artifacts
    metadata = zipfile.ZipFile(wheel).read(
        next(n for n in zipfile.ZipFile(wheel).namelist() if n.endswith('METADATA'))
    ).decode('utf-8')

    assert f'Version: {_source_version(repo_root)}' in metadata
    assert 'Requires-Python: >=3.10' in metadata
    assert 'License-Expression: MIT' in metadata
    # PyPI renders the README as the project page, so it has to be markdown
    assert 'Description-Content-Type: text/markdown' in metadata
    # the dependency floors this release was actually reasoned about
    for requirement in ('Requires-Dist: numpy>=2.0', 'Requires-Dist: cvxpy>=1.6',
                        'Requires-Dist: gymnasium>=1.0', 'Requires-Dist: scipy>=1.14'):
        assert requirement in metadata, f'{requirement} missing from the metadata'
    # and the periphery must stay optional. The quote style around the extra name depends on the
    # metadata version, so match either.
    assert 'Provides-Extra: experiments' in metadata
    assert re.search(r"Requires-Dist: matplotlib>=3\.8; extra == ['\"]experiments['\"]", metadata)
    # the project page has no links at all without this block
    assert 'Project-URL: Repository,' in metadata


# ----------------------------------------------------------------------
# The sdist: what a source consumer receives
# ----------------------------------------------------------------------

def test_the_sdist_carries_the_examples_and_tests_the_readme_points_at(artifacts):
    """The README is the long description and tells readers to run an example; an sdist
    consumer who does not receive the examples gets instructions that point at nothing."""
    _, sdist = artifacts
    with tarfile.open(sdist) as tar:
        names = [name.split('/', 1)[1] for name in tar.getnames() if '/' in name]

    for required in ('README.md', 'LICENSE', 'pyproject.toml',
                     'examples/apply_drpp_to_your_own_env/quickstart.py',
                     'examples/understand_drpp_on_simulated_ambiguity/quickstart.py',
                     'examples/understand_predictability/quickstart.py',
                     'examples/understand_ppsds_by_pmf_betting/experiment/quickstart.py',
                     'tests/conftest.py'):
        assert required in names, f'{required} is missing from the sdist'
    assert any(name.startswith('src/ppsds/') for name in names)


def test_the_sdist_carries_nothing_private(artifacts):
    """The whitelist in ``pyproject.toml`` is only as good as this assertion.

    ``docs/`` is excluded wholesale rather than file by file: the contributor guide is left out
    on purpose (it is written for a collaborator's unpublished method), so the remaining
    tutorials could never make the archive self-contained, and shipping half a docs tree that
    links at a missing file is worse than shipping none.
    """
    _, sdist = artifacts
    with tarfile.open(sdist) as tar:
        names = [name.split('/', 1)[1] for name in tar.getnames() if '/' in name]

    for prefix in ('docs/', 'papers/', 'tools/', 'build/', 'dist/'):
        leaked = [n for n in names if n.startswith(prefix)]
        assert not leaked, f'{prefix} must not be in the sdist, found {leaked[:5]}'
    # The sdist contains ``examples/`` but **not** ``public/``, so this never sees a release asset
    # from ``public/docs/assets/`` -- which *is* allowed to be a PNG (see
    # ``tools/export_public.py``). A PNG that reaches the sdist still means experiment output
    # leaked in, which is what this checks.
    for suffix in ('.pkl', '.npy', '.npz', '.pdf', '.png'):
        leaked = [n for n in names if n.endswith(suffix)]
        assert not leaked, f'{suffix} files must not be in the sdist, found {leaked[:5]}'
    assert not [n for n in names if n.endswith('EXTENDING.md')], \
        'the contributor guide names an unpublished method and must not ship'


def test_the_shipped_code_carries_no_private_references(artifacts, repo_root):
    """A text scan of the code a user actually receives.

    Two genuine leaks were found this way before the release: docstrings in
    ``experiment/cache.py`` and ``numerics/random_variables.py`` cited the private development
    records **by path**, which resolves for nobody outside this repository.

    The scope is deliberately the **code** -- the wheel's ``ppsds/**`` and the sdist's ``src/``
    and ``examples/``. The README, and the long description it produces, are covered by the
    export gate in ``tools/export_public.py``, which scans the whole published tree: the README
    in *this* repository is a development document that the public one replaces, so scanning it
    here would report the same expected hits on every run and the gate would be ignored.
    """
    tokens = forbidden_tokens(repo_root)
    if not tokens:
        pytest.skip(f'{TOKEN_FILE} is not present, so there is no embargo list to check against '
                    f'(expected in the public repository, where tools/ holds only the scripts '
                    f'that are published)')

    wheel, sdist = artifacts
    hits = []

    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            if name.startswith('ppsds/') and name.endswith('.py'):
                hits += _scan(zf.read(name).decode('utf-8', 'replace'), f'wheel:{name}', tokens)

    with tarfile.open(sdist) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            relative = member.name.split('/', 1)[-1]
            # ``.js``/``.html`` joined this list on 2026-10-16. They are not in the sdist today
            # (the game page lives under ``public/``), but they *were* exported to the public
            # repository for months without any gate reading them, so the suffix list now covers
            # what a future example might ship rather than only what today's does.
            if relative.startswith(('src/', 'examples/')) and \
                    relative.endswith(('.py', '.sh', '.ps1', '.js', '.html')):
                text = tar.extractfile(member).read().decode('utf-8', 'replace')
                hits += _scan(text, f'sdist:{relative}', tokens)

    assert not hits, 'private-repository references found in shipped code:\n' + '\n'.join(hits)


def _scan(text: str, where: str, tokens: list[str]):
    lowered = text.lower()
    return [f'  {where}: {token}' for token in tokens if token in lowered]


# ----------------------------------------------------------------------
# The wheel, actually used
# ----------------------------------------------------------------------

def test_the_wheel_payload_imports_in_isolation(artifacts, tmp_path, repo_root):
    """Import the wheel's own files, with ``-I`` so that neither the working directory nor
    ``PYTHONPATH`` can supply a package of the same name.

    ``-I`` is right *here* precisely because this is an extracted payload rather than a
    checkout: there is no ``PYTHONPATH`` to strip and no src layout to resolve.
    """
    wheel, _ = artifacts
    extract = tmp_path / 'unpacked'
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(extract)

    code = (
        'import sys\n'
        f'sys.path.insert(0, {str(extract)!r})\n'
        'import ppsds\n'
        'print(ppsds.__version__)\n'
        'print(ppsds.__file__)\n'
    )
    proc = _run([sys.executable, '-I', '-c', code], cwd=str(tmp_path))
    version, file = proc.stdout.strip().splitlines()

    assert version == _source_version(repo_root)
    assert str(extract) in file, f'imported {file}, which is not the wheel payload'


def test_the_wheel_installs_with_pip(artifacts, tmp_path):
    """The installation machinery itself: metadata, ``RECORD``, and the ``.dist-info`` layout.

    Dependencies are skipped (``--no-deps``) and the virtualenv inherits the parent's
    site-packages, so this stays offline and quick; what it verifies is that pip accepts the
    artifact and that the result is importable, not that the dependency solver works.
    """
    wheel, _ = artifacts
    venv = tmp_path / 'venv'
    _run([sys.executable, '-m', 'venv', '--system-site-packages', str(venv)], timeout=600)

    pip = venv / 'bin' / 'pip'
    _run([str(pip), 'install', '--no-deps', '--quiet', str(wheel)], timeout=600)

    proc = _run([str(venv / 'bin' / 'python'), '-c', 'import ppsds; print(ppsds.__version__)'],
                cwd=str(tmp_path))
    assert proc.stdout.strip()
