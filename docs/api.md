# API reference

Generated from the docstrings in the source, so this page cannot drift from the code.

## What each subpackage answers

One line each. These are the first sentence of each subpackage's own docstring, so they are also
what `help()` shows you:

| Subpackage | It answers |
|---|---|
| `ppsds.ambiguity` | *What am I willing to admit about the system?* — the belief, written as a set |
| `ppsds.dynamics` | *The system itself* — known dynamics and mechanical models, plus identifying a nominal model from rollouts |
| `ppsds.probabilistic_predictor` | *A prediction is a distribution, not a point.* The kinds sit side by side: `drpp/` derives one from an ambiguity set, `pmf/` takes one you supply |
| `ppsds.scoring_rule` | *How well did a prediction do?* — one number |
| `ppsds.predictability` | *How predictable is the system at all?* — the ceiling, as a reference line |
| `ppsds.environment` | *Run it as an environment* — for a system you declared an ambiguity set for |
| `ppsds.numerics` | Small numerical tools, and the weighted-atom pmf |
| `ppsds.control_policy` | A control policy, for the experiments that need one |
| `ppsds.experiment` | *Running experiments at scale*: the batch driver and its validated cache |
| `ppsds.visualization` | *Drawing the results*: score curves, covariance ellipses, trajectory overlays |

The module path then states the scope layer by layer: `ambiguity/additive_drift/conic_moment` reads
as *assumption* (additive drift) / *family* (conic moment), which is why the class names need not
repeat it. Swapping the family means adding a module at that level, not editing this one.

## Two layers inside one package

It matters which one you are importing:

| | modules | dependencies |
|---|---|---|
| **core** | `ambiguity`, `dynamics`, `probabilistic_predictor`, `environment`, `numerics`, `scoring_rule`, `control_policy` | numpy, scipy, cvxpy, gymnasium |
| **periphery** | `experiment`, `visualization` | additionally matplotlib, pandas, seaborn, tqdm (`pip install "ppsds[experiments]"`) |

A core module never imports the periphery, so `pip install ppsds` does not pull matplotlib in.

## Predictors

The DRPP over conic-moment ambiguity sets:

::: ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment

Predictors whose law is a pmf **you** supply:

::: ppsds.probabilistic_predictor.pmf.atom

The contract both satisfy. Three methods: `predict()` returns the predictive distribution for the
current step, `update(x, u)` advances the internal state after observing the state and control, and
`reset()` restores it for reuse — clearing the outputs alone is not enough, because a predictor
rewrites the nominal parameters in place. Three attributes must exist on every implementation —
`x_pred`, `Sigma_pred`, `lamb_pred` — because the environment layer's adversarial mode reads them
to pick the worst-case perturbation direction.

<!-- `members: false` is load-bearing: griffe raises "Empty strings are not supported" while
     reading this module's members. The class documents its three attributes with `#:` comments
     above bare annotations, which is what it trips on; rendering the module and describing the
     contract here says the same thing without depending on that parser path. Removing the option
     breaks `mkdocs build --strict`. -->

::: ppsds.probabilistic_predictor.base
    options:
      members: false

## Ambiguity sets

::: ppsds.ambiguity.additive_drift.conic_moment

## Scoring rules

::: ppsds.scoring_rule.logarithm_score

::: ppsds.scoring_rule.epsilon_logarithm_score

## Predictability: the ceiling

::: ppsds.predictability.bounds

::: ppsds.predictability.entropy

::: ppsds.predictability.trajectory

## The system itself

::: ppsds.dynamics.known_sds

::: ppsds.dynamics.identification

::: ppsds.dynamics.mechanical

## The environment layer

Serves the "simulate your own ambiguity" path, plus one exception: `KnownSdsEnv` steps a system
that is known exactly. If you are applying a predictor to an environment you already have, none of
this is needed.

::: ppsds.environment.base

::: ppsds.environment.dynamics_mode

::: ppsds.environment.known_sds

## Numerical helpers

::: ppsds.numerics.atom_pmf

::: ppsds.numerics.linalg

::: ppsds.numerics.random_variables

## Control policy

::: ppsds.control_policy.lqr

## Experiments and figures

Both need `pip install "ppsds[experiments]"`.

::: ppsds.experiment.cache

::: ppsds.visualization.predictability

## Next

This page is a reference, not a route. If you arrived here from a search:

- **What the guarantee costs, before you rely on it**: [Research](research/index.md)
- **Get something running first**: [Getting Started](tutorials/getting-started.md)
- **How the pieces fit together**: [the map](tutorials/map.md) — the routes through the examples, and which
  subpackage each one uses
- **Every example, with its real output**: [Examples](examples/index.md)
