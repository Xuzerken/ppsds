# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) — with the usual pre-1.0 caveat that
a minor version may change the API.

**How the version numbers are used here.** During development, `0.0.x` are private beta
artifacts published to collaborators only, never to PyPI: a version number on PyPI is permanent
and can be yanked but never reused, so it is not spent on half-finished releases. The first
version on PyPI is `0.1.0`.

## [0.0.1] - 2026-10-15

**A private beta, not on PyPI.** Everything below is the first release; `0.0.1` is handed to
collaborators through a GitHub Release so that problems are found before the version number that
matters is spent. The first public version will be `0.1.0`, and it will contain this.

### Added

- `ppsds.ambiguity` — conic-moment ambiguity sets over an additive-drift assumption
  (`LinearAmbiguitySet`, `NonlinearAmbiguitySet`), covering drift, noise-mean and covariance
  uncertainty through a small set of radii.
- `ppsds.probabilistic_predictor` — `NominalPredictor`, `NoiseDrpp` and `EigDrpp`, where
  `EigDrpp` solves the paper's cone program (`eigvec_fixed_minmax`, equation 19) with CVXPY/SCS.
- `ppsds.scoring_rule` — the logarithmic scoring rule.
- `ppsds.environment` — a gymnasium `Env` assembled from an ambiguity set plus a dynamics mode
  (`FixedDynamics`, `RandomDynamics`, `PreSampledDynamics`,
  `EigvecFixedAdversarialDynamics`), for simulating ambiguity rather than consuming it.
- `ppsds.dynamics` — robotic-arm models and system identification.
- `ppsds.experiment` and `ppsds.visualization` — a multiprocess experiment driver with a
  validating cache, and the paper's figures. These need the optional `experiments` extra.
- Examples for applying a predictor to an environment you already have
  (`examples/scene1_own_environment/`) and for building a simulated one
  (`examples/scene2_simulated_ambiguity/`).

### Known limitations

Recorded here rather than only in the issue tracker, because they affect whether the package
fits a given problem:

- `alphas['A']` and `alphas['B']` perturb only the superdiagonal of `A` and only `B[0, -1]`,
  which is the structure of the paper's own example. Uncertainty in arbitrary entries of `A` and
  `B` is not expressible through them.
- The batch experiment driver requires the `fork` start method, because one ambiguity set holds
  lambdas and cannot be pickled. It asserts this at startup and fails loudly.
- The `A`/`B` time-varying path of `Predictor.trajectory_prediction` is deprecated and has no
  equivalent in the environment layer yet.
- Four modules ship with no callers (`visualization/box.py`, `visualization/contour.py`,
  `scoring_rule/epsilon_logarithm_score.py`, `numerics/random_variables.multivariate_gmm`). They
  are kept deliberately for now.
