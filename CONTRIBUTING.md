# Contributing

Thanks for considering it. This is a research package, so the bar that matters most is that a
change does not silently alter a number — the rest is ordinary.

## Setting up

```bash
git clone https://github.com/Xuzerken/ppsds.git
cd ppsds
python -m venv .venv && . .venv/bin/activate      # or any environment you like

pip install -e ".[dev]"                  # core only
pip install -e ".[dev,experiments]"      # plus the experiment driver and the figures
pip install -e ".[docs]"                 # plus the toolchain that builds the documentation site
```

`pip install -e ".[dev]"` is the more useful of the first two: it is the configuration a plain
`pip install ppsds` produces, so it is the one that catches a core module that has grown a
dependency on the plotting periphery.

The `docs` extra is separate because nobody writing library code needs it — GitHub renders the
Markdown as it stands. Install it to preview the site locally (`mkdocs serve`) or to change the
site itself; CI installs it on its own.

## Running the tests

```bash
pytest                                   # everything, about 30 seconds
pytest -m "not slow and not packaging"   # the inner loop, about 4 seconds
ruff check .                             # lint
```

Two markers select the slower checks:

- `slow` — the examples that identify a model from rollouts of an off-the-shelf gymnasium
  environment. They are the entry points the README sends readers to, so they do run in CI, just
  on one configuration.
- `packaging` — builds the sdist and the wheel and installs the wheel. Needs the network (the
  build backend is fetched into an isolated environment).

`tests/golden.json` pins the predictions of all three predictors on a fixed system over eight
steps. When a numerical change is intended, regenerate it and **read the diff** — the diff is the
review:

```bash
python tests/regenerate_fixtures.py
```

## The one rule that is not ordinary

**A change that touches numerics must survive an A/B comparison.** `tools/ab_compare.py` checks
a baseline commit out into a separate `git worktree`, runs the same small configurations in both
trees, and compares the resulting pickles element by element:

```bash
python tools/ab_compare.py --base <the commit you branched from>
```

The acceptance line is **every maximum absolute difference equal to `0.0`** — bit-identical, not
"close". That bar is deliberate: the cone program keeps the best of several candidate worst
directions, and a perturbation of 1e-14 can flip which one wins. The scores then differ by about
1e-9, so the run still looks fine, but the choice changed — and that is exactly the kind of
change worth seeing rather than tolerating.

If a difference is intended, say so in the pull request and include the measured magnitudes.

## Layout, and the boundary to respect

```
src/ppsds/          the distribution
examples/           runnable examples
tests/              the test suite
tools/              procedure gates (see tools/README.md)
```

Inside the package there is a boundary that is not visible from the import paths:

| core | periphery |
|---|---|
| `ambiguity`, `dynamics`, `probabilistic_predictor`, `environment`, `numerics`, `scoring_rule`, `control_policy` | `experiment`, `visualization` |
| numpy, scipy, cvxpy, gymnasium | additionally matplotlib, pandas, seaborn, tqdm |

**A core module must never import a periphery module.** If it does, `pip install ppsds` drags
matplotlib into every user's dependency closure. `tests/test_import_hygiene.py` checks this
statically and at runtime, and CI installs the package without the extra to check it the only
honest way.

## Conventions

- **Code is in English** — comments, docstrings, warnings, exception messages, example output.
- **Module paths carry meaning.** `ambiguity/additive_drift/conic_moment` reads as
  *assumption* / *family*. A new ambiguity-set family is a new module beside the existing one,
  not a branch inside it, and the same holds for the adversarial dynamics modes. The
  [developer guide](https://xuzerken.github.io/ppsds/extending/) walks through that end to end,
  including what each layer expects from a new family.
- **Errors are loud.** A wrong `sds_params` key raises and names the accepted keys; a
  `spawn` start method raises rather than producing wrong results. Please keep it that way: the
  failures this package has had to fix were mostly silent ones.
- Pull requests should say what changed and how it was verified. Numbers are welcome.

## Reporting a problem

Open an issue at https://github.com/Xuzerken/ppsds/issues. If it is a numerical problem, the
versions matter a great deal — the two failure modes this package has hit were both specific to
a numpy or cvxpy version — so please include the output of:

```bash
python -c "import numpy, scipy, cvxpy, gymnasium, sys; print(sys.version); print(numpy.__version__, scipy.__version__, cvxpy.__version__, gymnasium.__version__)"
```
