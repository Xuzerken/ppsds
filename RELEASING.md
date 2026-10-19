# Releasing

The runbook for cutting a release. It is written for the maintainer, but it is public on purpose:
the checks below are the reason a release is trustworthy, and there is no reason to keep them
secret.

## Before anything: mirror the development repository

The public repository is a projection of the development repository, so the release is cut
**here**. In the development repository:

```bash
python tools/export_public.py --target ../ppsds --check    # must say the two match
python tools/export_public.py --target ../ppsds --commit --push
```

`--check` is not optional. If it refuses, it is because something changed on the public side that
this repository does not have, and exporting would revert it — the script says which paths and
what to do.

## Two channels: private hand-outs and PyPI pre-releases

A version number on PyPI is permanent, so the version string decides the channel:

- **Unqualified `0.0.x`** (for example `0.0.2`) is a private hand-out. It stays a GitHub Release
  attachment; the `publish` workflow builds the artifacts and deliberately skips PyPI.
- **A PEP 440 pre-release suffix** (`aN` / `bN` / `rcN`, for example `0.0.3a0`) is published to
  PyPI as a pre-release. PyPI and pip both treat it as a pre-release: it is never chosen over a
  stable version, and users pin it (`ppsds==0.0.3a0`) or pass `--pre`.
- **A version without either** (for example `0.1.0`) is a stable release and publishes normally.

### Private hand-out (unqualified `0.0.x`)

A private hand-out goes to collaborators, not to PyPI.

```bash
# in the public clone, on main, with __version__ already at the hand-out number
git tag -a v0.0.2 -m "ppsds 0.0.2 (private hand-out)"
git push origin v0.0.2
```

The `publish` workflow runs, builds the sdist and the wheel, and then **deliberately does not
publish**: it reports that the version has no pre-release suffix and stops short of PyPI. That is
the guard, not a failure — see the job's notice.

The artifacts are still attached to the run. Download `dist` from the workflow run's artifacts
and attach both files to a GitHub Release cut from the tag. Then hand out the release, not the
repository URL: whoever receives it installs with

```bash
pip install ./ppsds-0.0.2-py3-none-any.whl
```

### Pre-release on PyPI

Tag `v0.0.3a0` (the suffix is part of the version and must match `__version__` exactly). The
guard sees the pre-release suffix and publishes. Mark the GitHub Release **Set as a
pre-release**, and tell testers the exact pin:

```bash
pip install ppsds==0.0.3a0     # or: pip install --pre ppsds
```

**Invite collaborators to the release, not to the development.** The repository they can see is
the public one. Tell them plainly to **fork it and work in the fork** rather than pushing
branches here: everything in this repository — all branches, all open pull requests — becomes
visible the moment it is made public, and work that is not ready to be seen must not be in it.

## 1. Confirm everything is green

All workflows on `main`: `tests`, `core-only`, `lower-bounds`, `lint`, `build`, `docs`.

`core-only` is the one that is easy to dismiss and should not be: it installs the package
*without* the `experiments` extra, which is the only way to prove that a plain
`pip install ppsds` works.

## 2. Bump the version and the changelog

- `src/ppsds/__init__.py` — `__version__`. It is the single source of truth; `pyproject.toml`
  reads it dynamically. Use the PEP 440 form: a pre-release keeps its suffix (`0.0.3a0`), and
  the tag must match it exactly.
- `CHANGELOG.md` — move `[Unreleased]` to the new version with today's date.

```bash
grep -rn "0\.0\." src pyproject.toml     # no stale version strings left behind
```

## 3. Build and inspect

```bash
rm -rf dist build
python -m build
python -m twine check dist/*
python -m readme_renderer README.md > /dev/null   # what PyPI will render, not what GitHub does
unzip -l dist/*.whl && tar -tzf dist/*.tar.gz     # eyeball both lists
```

The wheel must contain the package and nothing else. The sdist carries `examples/` and `tests/`
as well, because the README — which is the long description — tells readers to run an example.

## 4. Rehearse on TestPyPI

The repository is public by the time the first upload happens, so the `pypi` and `testpypi`
GitHub environments exist (Settings → Environments, each with yourself as a required reviewer),
and both PyPI and TestPyPI have a Trusted Publisher entry for this repository. Create those
before tagging.

First rehearse the artifact itself, without the network:

```bash
# build from the public clone, then rehearse the artifact
rm -rf dist build && python -m build
python -m venv /tmp/rehearse
/tmp/rehearse/bin/pip install dist/ppsds-*.whl
/tmp/rehearse/bin/python -c "import ppsds; print(ppsds.__version__)"

# and the check that actually proves the sdist is complete: install from it and run an example
tar -xzf dist/ppsds-*.tar.gz -C /tmp
/tmp/rehearse/bin/pip install /tmp/ppsds-<version>/
/tmp/rehearse/bin/python /tmp/ppsds-<version>/examples/apply_drpp_to_your_own_env/quickstart.py
```

Then rehearse the upload path itself with a manual `publish` run (`target: testpypi`). Never
rehearse an upload on PyPI. A version number there is permanent.

```bash
python -m venv /tmp/rehearse
/tmp/rehearse/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  ppsds==<version>
```

The extra index is required: TestPyPI does not host numpy, scipy, cvxpy or gymnasium. TestPyPI
is a separate service with its own account, and its own Trusted Publisher entry.

## 5. Tag, and let CI publish

Tag on the **public** repository, since that is the commit PyPI and any archive will reference.
The `publish` workflow compares the tag against `__version__` **exactly, including the
pre-release suffix** (`v0.0.3a0`, not `v0.0.3-alpha`), and refuses to upload if they disagree.

```bash
git -C ../ppsds tag -a v<version> -m "ppsds <version>"
git -C ../ppsds push origin v<version>
```

Publishing uses PyPI Trusted Publishing: the workflow exchanges a short-lived OIDC token for an
upload token, so there is no credential stored anywhere. The one-time setup on PyPI trusts
exactly this owner, repository, workflow filename and environment.

## 6. Verify the release

- The PyPI page renders: no literal `$$`, no links that 404, correct metadata and classifiers.
- `pip install ppsds==<version>` in a clean environment on more than one platform, then run an
  example. For a pre-release, also check `pip install --pre ppsds`; a plain `pip install ppsds`
  selects the pre-release only while no stable version exists.
- The documentation site is live at https://xuzerken.github.io/ppsds/.

## 7. Archive and cite

Cut a GitHub Release from the tag, using the changelog section as the description. For a
pre-release, check **Set as a pre-release**. If Zenodo is watching the repository, tagging
archives it and mints a DOI; put that DOI into `CITATION.cff` and the citation page, and commit.
That is a documentation change and needs no version bump. Add the DOI after the first **stable**
release.

## Afterwards

Add a fresh `[Unreleased]` heading to the changelog. Leave `__version__` at the released value
until the next bump.

## If something is wrong with a published release

**Yank it on PyPI; never delete a file, and never reuse a version number.** Yanking hides the
release from resolution while leaving it available to anyone who already pinned it — which is
what you want, because somebody's environment already depends on it. Fix forward with a patch
release.
