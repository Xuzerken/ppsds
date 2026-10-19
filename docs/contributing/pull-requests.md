# Pull requests

A change can be a fix, a new example, a new family of ambiguity sets and its DRPP, a new kind of
probabilistic predictor, or a benchmark. The setup is the same for all of them.

## Setting up

```bash
git clone https://github.com/Xuzerken/ppsds.git
cd ppsds
python -m venv .venv && . .venv/bin/activate

pip install -e ".[dev]"                  # core only
pip install -e ".[dev,experiments]"      # plus the experiment driver and the figures
pip install -e ".[docs]"                 # plus the documentation site toolchain
```

`.[dev]` is the configuration a plain `pip install ppsds` produces, so it is the one that catches
a core module which has grown a dependency on the plotting periphery.

## Running the tests

```bash
pytest                                   # everything
pytest -m "not slow and not packaging"   # the inner loop
ruff check .                             # lint
```

Two markers select the slower checks: `slow` runs the examples that identify a model from
rollouts, and `packaging` builds and installs the artifacts. `tests/golden.json` pins the
predictions of all three predictors on a fixed system; when a numerical change is intended,
regenerate it and **read the diff**.

## The one rule that is not ordinary

**A change that touches numerics must survive an A/B comparison.**

```bash
python tools/ab_compare.py --base <the commit you branched from>
```

The tool checks the baseline out into a separate `git worktree`, runs the same small
configurations in both trees, and compares the pickles element by element. The acceptance line is
**every maximum absolute difference equal to `0.0`** — bit-identical, not "close". A cone program
keeps the best of several candidate worst directions, and a perturbation of `1e-14` can flip
which one wins; the scores then move by about `1e-9`, which looks fine but is a different
predictor. That is the change this gate exists to make visible.

If a difference is intended, say so in the pull request and include the measured magnitudes.

## The boundary to respect

```
src/ppsds/          the distribution
examples/           runnable examples
experiments/        the batch experiments
tests/              the test suite
tools/              procedure gates
```

Inside the package, the core (`ambiguity`, `dynamics`, `probabilistic_predictor`, `environment`,
`numerics`, `scoring_rule`, `control_policy`) must never import the periphery (`experiment`,
`visualization`). If it does, `pip install ppsds` drags matplotlib into every user's dependency
closure. `tests/test_import_hygiene.py` checks it both statically and at runtime, and CI installs
without the extra to check it the honest way.

## Conventions

- **Code is in English** — comments, docstrings, warnings, exception messages, example output.
- **Module paths carry meaning.** `ambiguity/additive_drift/conic_moment` reads as *assumption* /
  *family*. A new ambiguity-set family is a new module beside the existing one, not a branch
  inside it. [Extending `ppsds`](../tutorials/developer-guide.md) walks through that end to end.
- **Errors are loud.** A wrong `sds_params` key raises and names the accepted keys; a `spawn`
  start method raises rather than producing wrong results. Please keep it that way.
- Pull requests should say what changed and how it was verified. Numbers are welcome.

## Next

- **The full extension walkthrough**: [Extending `ppsds`](../tutorials/developer-guide.md)
- **Finding a problem to fix**: [Reporting a problem](reporting.md)
- **Research and collaboration**: [Research collaborations](research.md)
