# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) — with the usual pre-1.0 caveat that
a minor version may change the API.

**How the version numbers are used here.** During development, `0.0.x` are private beta
artifacts published to collaborators only, never to PyPI: a version number on PyPI is permanent
and can be yanked but never reused, so it is not spent on half-finished releases. The first
version on PyPI is `0.1.0`.

## [Unreleased]

## [0.0.2] - 2026-10-19

**A private beta, not on PyPI.** The second beta: the package stops being only about DRPP. DRPP
answers *what can a predictor guarantee when the model is uncertain* — a lower bound. This release
adds the other side of that question, *how well could anything predict a system known exactly* —
an upper bound — and a way to feel the difference before reading any of it. It also rebuilds the
documentation around the question of who each page is for.

### Added

- `ppsds.predictability` — the `epsilon`-logarithm score's reference values:
  `predictability_upper_bound` (the best expected score any predictor could reach, a property of
  the system alone), `approximation_value` (the `O(epsilon)` approximation for a given
  predictor), `differential_entropy` / `cross_entropy`, and trajectory aggregation
  (`cumulative_average`, `across_trajectories`).
- `EpsilonLogarithmScore` — the neighbourhood scoring rule, replacing the previous zero-caller
  stub. It is vectorised, callable, and dispatches on the law it is given: `logpdf` at
  `epsilon = 0`, a `box_integral` method for pmfs, a closed form for diagonal Gaussians, and CDF
  inclusion-exclusion as the general fallback. These paths are tested against each other, not
  against stored constants.
- `ppsds.numerics.atom_pmf.AtomPmf` — a predictive law as weighted atoms. Its support need not
  contain the true distribution's, which is the point: under the neighbourhood score a sparse
  pmf is a usable prediction, while under the ordinary logarithm score it is `-inf` almost
  surely.
- `ppsds.probabilistic_predictor.pmf` — a second **kind** of predictor beside `drpp`:
  `StaticAtomPmfPredictor` and `DriftShiftedAtomPmfPredictor`, which predict with a pmf you
  supply rather than one a model derives.
- `ppsds.dynamics.known_sds.AdditiveDriftSds` and `ppsds.environment.KnownSdsEnv` — a system that
  is known exactly (drift plus a named noise law, non-Gaussian included) and a gymnasium
  environment that steps it. `Env` still serves scene 2; `KnownSdsEnv` is the exception.
- Examples reproducing the predictability paper's figure (`examples/understand_predictability/`) and
  running your own pmf bets as an experiment (`experiments/pmf_betting/`).
- An interactive page at <https://xuzerken.github.io/ppsds/examples/understand_ppsds_by_pmf_betting/> that scores a hand of bets live,
  and a README animation of it. The page replays trajectories exported from
  `experiments/pmf_betting/presets.py`, and checks its own JavaScript against vectors computed by
  `ppsds` on every load.
- `EpsilonLogarithmScore`, `AtomPmf`, the pmf predictors and the predictability functions are now
  re-exported from the top level.
- `docs/assets/map.png` — a route map across the four sections, used by the README, the site home
  and `tutorials/map.md`.
- `docs/assets/paper-drpp.png` and `docs/assets/paper-predictability.png` — each paper's own
  vector figure rasterised separately for its Research page.

### Changed

- **The documentation site is organised as four kinds of node.** Home, then **Examples** (stories
  you can run), **Experiments** (the same questions at paper scale), **Research** (the common
  preliminaries and one page per paper, each with its own citation), **Tutorials** (topics treated
  on their own terms), and **API**. The old flat shelf (`Concepts`, `Predictability`, `Examples`,
  `Developer Guide`, `Citation`) is gone; `concepts.md` became the Research preliminaries and the
  separate citation page was retired because every paper page now ends with its citation.
- **The batch experiments moved out of `examples/` into `experiments/`**:
  `experiments/drpp/`, `experiments/pmf_betting/` and `experiments/predictability/`. Each example
  and experiment directory now carries its own `environment.yml`, so one page can be run without
  installing the rest. `examples/understand_ppsds_by_pmf_betting/` keeps its README: its runnable
  front is the no-install game page.
- **The README sells routes, not a page list.** It opens with a two-column game block (the
  animation beside the rules that map onto the package), a one-row-per-reader entry table, and
  cross-section routes through Examples, Experiments, Research and Tutorials. The old
  `The examples` and `A predictor on your own system` sections were removed: both duplicated pages
  the site now owns.
- The citation and licence sections were merged; contributing is a reader type in the entry table
  rather than a separate section.

### Notes

- **No number moved.** The DRPP experiment was compared against the pre-move commit with
  `tools/ab_compare.py`; every compared array is bit-identical.
- **The `epsilon`-logarithm score is discontinuous at `epsilon = 0`**, and this is a property of
  the definition rather than an artefact: at exactly zero it is `log p(x)`, while for any
  `epsilon > 0` it carries a `d log(2 epsilon)` volume term and runs to `-inf` as `epsilon`
  shrinks. Subtract that term to recover the ordinary logarithm score; do not take a limit.
- `multivariate_uniform` gained a batch-aware `pdf` / `cdf` and an optional `rng` for `rvs`. The
  default `rvs` path is unchanged, and the DRPP numbers are bit-identical.

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
  (`examples/apply_drpp_to_your_own_env/`) and for building a simulated one
  (`examples/understand_drpp_on_simulated_ambiguity/`).

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
