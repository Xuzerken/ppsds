# API reference

Generated from the docstrings in the source, so this page cannot drift from the code. The
module path states the scope layer by layer: `ambiguity/additive_drift/conic_moment` reads as
*assumption* (additive drift) / *family* (conic moment), which is why the class names need not
repeat it.

Two layers live inside the one package, and it matters which one you are importing:

| | modules | dependencies |
|---|---|---|
| **core** | `ambiguity`, `dynamics`, `probabilistic_predictor`, `environment`, `numerics`, `scoring_rule`, `control_policy` | numpy, scipy, cvxpy, gymnasium |
| **periphery** | `experiment`, `visualization` | additionally matplotlib, pandas, seaborn, tqdm (`pip install "ppsds[experiments]"`) |

A core module never imports the periphery, so `pip install ppsds` does not pull matplotlib in.

## Predictors: the DRPP over conic-moment ambiguity sets

::: ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment

## Ambiguity sets

::: ppsds.ambiguity.additive_drift.conic_moment

## Scoring rule

::: ppsds.scoring_rule.logarithm_score

## The environment layer

Serves the "simulate your own ambiguity" path only. If you are applying a predictor to an
environment you already have, none of this is needed.

::: ppsds.environment.base

::: ppsds.environment.dynamics_mode

## Numerical helpers

::: ppsds.numerics.linalg

::: ppsds.numerics.random_variables
