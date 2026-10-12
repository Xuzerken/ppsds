# tools/

Scripts that guard **procedures**, not behaviour. They are deliberately not tests, and
deliberately not in the package: `pytest` asserts what the library does, while these answer
"was this repository released in a state that is allowed?" -- questions with a manual,
judgement-shaped answer that no assertion can carry.

They are not shipped in the sdist and they are not part of the wheel. `src/ppsds/` is the
distribution; everything here is about the repository.

| script | the question it answers | when to run it |
|---|---|---|
| `ab_compare.py` | "did this change alter a single number?" | before merging anything that touches numerics, the cache fingerprint, or the experiment pipeline |
| `check_links.py` | "do the links in the documents that ship actually resolve?" (plus `--pypi-readme` for the two rendering rules PyPI adds) | before a release; in CI |
| `export_public.py` | "does the public repository match this one, and does what would be exported carry anything private?" | before every push to the public repository |

`check_devlog.py` is **not** here on purpose, and is the one script in this directory that does
not travel to the public repository: it enforces this repository's development-record workflow,
which would be permanently red for an outside contributor.

## `ab_compare.py` in one paragraph

It checks out a baseline commit into a `git worktree`, runs the same small simulation
configurations in both trees (each with its own `PYTHONPATH`, so one environment can host both),
and compares the resulting pickles element by element with `equal_nan=True`. The acceptance line
is **every maximum absolute difference equal to 0.0** -- not "the run succeeded", and not "the
numbers look close". Bit-identical is the right bar here because the cone program's discrete
choice of worst direction can flip on a 1e-14 perturbation, and a change that flips it is a
change worth seeing rather than tolerating.

`--base` is required. A baseline is a judgement about what a change is allowed to affect, so it
belongs at the call site rather than in a default that suits one machine.

## `export_public.py` in one paragraph

The whitelist lives at the top of the script and is short by design. The public repository is a
projection of this one -- shared code copied verbatim, and the contents of `public/` promoted to
the repository root -- so there is exactly one place where each published file is written, and
never a question of whether something was edited on the public side and forgotten here.
`--check` turns "the two repositories have drifted apart" into a non-zero exit code, which is
worth wiring into a pre-push hook: drift is cheap to fix the moment it happens and expensive to
untangle at the next release.

## The round trip

### Outgoing: this repository → the public one

```bash
python tools/export_public.py --target ../ppsds --commit --push
```

One command. `--dry-run` prints the plan, `--check` only compares, and both write nothing.

### Incoming: a change that arrived on the public side

This is the direction that needs care, because **this repository is the source of truth for the
shared files** -- so an export run over a contribution that arrived on the public side would
quietly revert it. The export refuses to do that; it detects the situation and prints what to do.

How to bring a contribution back, per path:

| what the contribution touches | how to absorb it |
|---|---|
| `src/`, `examples/`, `tests/`, `tools/`, `pyproject.toml`, `.github/workflows/` | `git fetch <the public clone> main` then `git cherry-pick -x FETCH_HEAD` — the path is the same here |
| the public-only documents (`README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.gitignore`, …) | **copy the content** into `public/<the same name>`; these live at a different path here, and this repository has its own, different, root `README.md` — cherry-picking would land on the wrong file |
| something that is not exported at all | decide whether it belongs here, and add it to the whitelist if it does |

Then export again. `-x` on the cherry-pick records the original commit in the message, which is
what keeps the contribution traceable after it travels through here; the author is preserved
either way.

Which merge style the public repository used changes only the cherry-pick:

```bash
git cherry-pick -x <sha>            # squash merge: one commit
git cherry-pick -x <first>^..<last> # rebase merge: a range
git cherry-pick -m 1 -x <sha>       # merge commit: follow the first parent
```

### What the export checks before writing

Four things, in this order, and any of them stops it:

1. **The embargo scan** — no forbidden token in anything that would be exported.
2. **The destination is the right repository** — otherwise pruning deletes somebody's files.
3. **The destination has no uncommitted changes** — except an export's own output that was not
   committed yet, which `--commit` will pick up.
4. **Nothing changed on the public side since the last export** — the check that makes the
   incoming direction safe. It works from the destination's own history (the last export left a
   commit), so it survives a fresh clone, and it does not depend on the clone being behind its
   remote: the moment anyone pulls, "is it behind?" would report all clear while a merged
   contribution sat in the working tree.
