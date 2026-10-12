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

## 1. Confirm everything is green

All workflows on `main`: `tests`, `core-only`, `lower-bounds`, `lint`, `build`, `docs`.

`core-only` is the one that is easy to dismiss and should not be: it installs the package
*without* the `experiments` extra, which is the only way to prove that a plain
`pip install ppsds` works.

## 2. Bump the version and the changelog

- `src/ppsds/__init__.py` — `__version__`. It is the single source of truth; `pyproject.toml`
  reads it dynamically.
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

Never rehearse this on PyPI. A version number there is permanent.

```bash
python -m venv /tmp/rehearse
/tmp/rehearse/bin/pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  ppsds==<version>
```

The extra index is required: TestPyPI does not host numpy, scipy, cvxpy or gymnasium.

Then the part that actually proves the sdist is complete — run an example **from the sdist**,
not from a checkout:

```bash
tar -xzf dist/ppsds-*.tar.gz -C /tmp
/tmp/rehearse/bin/python /tmp/ppsds-<version>/examples/scene1_own_environment/quickstart.py
```

## 5. Tag, and let CI publish

Tag on the **public** repository, since that is the commit PyPI and any archive will reference.
The `publish` workflow compares the tag against `__version__` and refuses to upload if they
disagree.

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
  example.
- The documentation site is live at https://xuzerken.github.io/ppsds/ — Pages cannot be served
  from a private repository, so the first release is also when the site goes up.

## 7. Archive and cite

Cut a GitHub Release from the tag, using the changelog section as the description. If Zenodo is
watching the repository, tagging archives it and mints a DOI; put that DOI into `CITATION.cff`
and the citation page, and commit. That is a documentation change and needs no version bump.

## Afterwards

Add a fresh `[Unreleased]` heading to the changelog. Leave `__version__` at the released value
until the next bump.

## If something is wrong with a published release

**Yank it on PyPI; never delete a file, and never reuse a version number.** Yanking hides the
release from resolution while leaving it available to anyone who already pinned it — which is
what you want, because somebody's environment already depends on it. Fix forward with a patch
release.
