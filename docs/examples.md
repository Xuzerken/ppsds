# Examples

Every example lives under
[`examples/`](https://github.com/Xuzerken/ppsds/tree/main/examples), split into two scenes:

- **`scene1_own_environment/`** — DRPP on a gymnasium environment you already have. This is the
  main path: three examples of increasing realism.
- **`scene2_simulated_ambiguity/`** — declare an ambiguity set by hand, generate a stochastic
  dynamical system from it, and compare the predictors on it. A single-process teaching example,
  plus a multi-process batch runner.

The commands below are run from the repository root with `ppsds` installed editable
(`pip install -e .`), and the output blocks are what the scripts actually print.

## scene1_own_environment/quickstart.py

```bash
python examples/scene1_own_environment/quickstart.py
```

[The file](https://github.com/Xuzerken/ppsds/blob/main/examples/scene1_own_environment/quickstart.py)
stands up a two-state linear system as a stand-in for "your own environment", declares a nominal
model and a conic-moment ambiguity set, and scores `NominalPredictor` and `EigDrpp` over 50 steps
of zero control. About one second.

```
Running 50 steps on **one ordinary trajectory** (your own environment, zero control)

predictor     mean log score   note
----------------------------------------------------------------
nominal               -2.867   trusts the nominal model
Eig-DRPP              -3.366   made robust to the worst case
```

**What to conclude:** `nominal` scores higher than Eig-DRPP on this trajectory, and that is the
expected outcome rather than a failure. DRPP optimizes the worst case; on a single ordinary
rollout that conservatism is a cost with nothing yet to collect. The example's real subject is
the workflow: the environment is stepped but never modified, and both predictors take their model
from the same ambiguity set, so they differ only in whether they account for the ambiguity.

## scene1_own_environment/with_standard_gym_env.py

```bash
python examples/scene1_own_environment/with_standard_gym_env.py
```

About 15 seconds. This one puts DRPP on **Acrobot-v1** — 6-dimensional, underactuated, chaotic,
and not an environment you wrote — so the nominal model and the ambiguity set have to be
identified from data. It then answers the question `quickstart.py` leaves open: **if the model is
this bad, does DRPP win?**

```
1. System identification: run a few trajectories and fit a **linear** nominal model
  residual/increment     =    73.7%
  lag-1 autocorrelation  =    0.033
  covariance condition number = 1.36e+02
  numerically zero eigenvalues =        0

2. Declare the ambiguity set (same order: pick the family, then fill in its language)
  f_err_max = 6.619  (99th percentile of the residual norm)
  alphas    = A=0.283 (scales with ||x||), B=1.687 (scales with ||u||), mu=1e-6, Sigma=1.0

3. First question: the model is this bad, so did DRPP win?
  identification data is plentiful (20 episodes), evaluation uses **another 5** trajectories (different seeds):
    nominal =   -1.143      Eig-DRPP =   -7.099      -> nominal wins

4. What actually decides it: whether your declared bounds are optimistic or conservative
   ident. data est. resid std    nominal   Eig-DRPP   winner
  --------------------------------------------------------------
     1 episode         0.1782     -6.062     -2.515   **Eig-DRPP**
    2 episodes         0.7952     -1.308     -9.479   nominal
    4 episodes         0.8277     -0.947     -9.201   nominal
   20 episodes         0.6949     -1.143     -7.099   nominal
```

(The script's own paragraphs explaining each block are elided here; the measurements are verbatim.)

**What to conclude:** a bad model is *not* what makes DRPP win. Section 3 shows DRPP losing by a
wide margin even though the linear residual is 74% of the state increment, because `Sigma_bar`
was estimated from your own residuals and is therefore calibrated to the errors the predictor will
actually meet. Section 4 isolates the variable that does decide it: with 20 identification
episodes the estimate is calibrated and nominal wins, while one episode underestimates it by about
3.9x, the nominal prediction is too narrow, and DRPP wins. **Too little identification data is the
most common source of optimism** — and optimism, not model error, is what DRPP is paid for.

## scene1_own_environment/pendulum_singular_covariance.py

```bash
python examples/scene1_own_environment/pendulum_singular_covariance.py
```

About 4 seconds. A second pitfall on the same path, on **Pendulum-v1**: the residual covariance
you estimate has a **structural** zero eigenvalue. Not because the residuals are small, but
because the observation lies on the cylinder `cos^2 + sin^2 = 1` and one state update is exactly
linear in the observation — so that component's residual is identically zero.

```
1. System identification (run a few trajectories, fit a linear nominal model)
  covariance condition number = 7.04e+11
  numerically zero eigenvalues =        1
  eigenvalues            = [5.065e-15 7.349e-04 3.566e-03]
  residual std / increment std = [0.982 0.333 0.   ]

2. ...feeding the unregularized covariance to the score:
    measured: the realized value is 5.48e-02 away from the predictive mean, and the score is -inf

3. Regularization: this step **is itself a modeling statement**
      floor  min eigenvalue    nominal   Eig-DRPP
  ----------------------------------------------
      1e-06        1.00e-06      9.164      3.659
      1e-04        1.00e-04      6.879      3.648
      1e-03        1.00e-03      5.685      3.556
      1e-02        1.00e-02      3.711      2.851
```

**What to conclude:** a degenerate predictive distribution scores `-inf` for almost every value,
and a lower bound of `-inf` guarantees nothing — so `Sigma_bar` must be regularized before DRPP
can use it. But the regularization floor is not a cosmetic constant: moving it from `1e-6` to
`1e-2` moves the score by 5.5 points, because it sets the width of the predictive distribution.
Lifting a structurally dead direction to positive-definite **is** a modelling statement — "how
much uncertainty am I willing to admit here" — and should come from physics or from a first run
you scrutinize, not from a number invented to make the code run. Throughout, Eig-DRPP sits below
nominal, because here the true system is close to the nominal model and robustness is pure cost.

## scene2_simulated_ambiguity/quickstart.py

```bash
python examples/scene2_simulated_ambiguity/quickstart.py
```

A few seconds, single process, **core API only** — no cache, no command line, no model table. It
declares a conic-moment ambiguity set by hand, then runs the same rollout loop under two
semantics: `random` (one SDS sampled from the set and shared by every predictor, so only the noise
differs) and `worst` (each predictor faces the SDS chosen specifically against it — a best
response, which is the paper's central semantics). Eig-DRPP dominates the runtime because it
solves a cone program at every step; lower `N_TRAJ` or `T` in the file to make it faster.

```
Mean logarithmic score (higher is better), 4 trajectories x 50 steps

predictor       random     worst
--------------------------------
Nominal         -4.726   -23.365
Noise-DRPP      -3.974   -10.186
Eig-DRPP        -4.566    -5.541

Theoretical bounds (per step): upper -4.048, lower -5.833
```

**What to conclude:** under `random` SDS the three predictors sit close together — the nominal
model is roughly right, so robustness buys little and costs a little. Under `worst` they separate
sharply: `Nominal` is driven down to `-23.4` because it is attacked on purpose, while `Eig-DRPP`
shrinks its covariance along the worst direction and holds near its lower bound of `-5.833`. That
gap is the guarantee, measured — average performance given up under `random`, worst-case floor
collected under `worst`. `Noise-DRPP` sits in between, which is what "robust to noise only"
buys.

## scene2_simulated_ambiguity/batch_experiment/run.py

The teaching example above is deliberately small. Reproducing the paper's sweeps needs more:
seven models, a validated cache, and multiple processes. That is what `run.py` is, and it is
honest about being a **batch runner** rather than a tutorial — a model table in `models.py`, a
cache with a shape contract, and a `fork` requirement.

```bash
cd examples/scene2_simulated_ambiguity/batch_experiment

# linear model, randomly-sampled SDS
python run.py -n 8 -s 3 -p 100 -m final_round -c none -t 100

# nonlinear model (6-link arm), worst-case adversarial SDS
python run.py -n 8 -s 3 -m NLink -c none -t 100 -e worst
```

The second command prints the following (progress bars and the worker `FutureWarning`s about the
deprecated full-trajectory helper are omitted; the output paths are abbreviated — the runner
prints them in full):

```
======================================================================
DRPP Gymnasium [worst]: NLink, period=N/A, control=none
======================================================================
Evaluating Worst-Nominal...
  Saved Worst-Nominal to .../data/gym/model=NLink_n=6/control=none/.../seed=3/worst-case/nominal.pkl
Evaluating Worst-Eig-DRPP...
  Saved Worst-Eig-DRPP to .../data/gym/model=NLink_n=6/control=none/.../seed=3/worst-case/eig_drpp.pkl
Evaluating Worst-Noise-DRPP...
  Saved Worst-Noise-DRPP to .../data/gym/model=NLink_n=6/control=none/.../seed=3/worst-case/noise_drpp.pkl

cache: recomputed 3, reused 0
  - nominal.pkl: parameters or code changed (fingerprint mismatch)
  - eig_drpp.pkl: parameters or code changed (fingerprint mismatch)
  - noise_drpp.pkl: parameters or code changed (fingerprint mismatch)
Done! Results saved to .../data/gym/model=NLink_n=6/control=none/.../seed=3/worst-case/
```

Run the identical command again and the cache answers instead of the solver:

```
cache: recomputed 0, reused 3
```

Each model runs through the **same generic `Env`**; what differs between systems lives entirely in
the ambiguity-set subclass, which is why one table can cover all seven models. Three files are
written per run (`nominal.pkl`, `eig_drpp.pkl`, `noise_drpp.pkl`), plus `traj.pkl`, `optimal.pkl`
and the figures for `random` runs.

### Command-line flags

| flag | meaning |
|---|---|
| `-n` | number of worker processes (default `8`) |
| `-s` | random seed (default `3`) |
| `-p` | SDS re-sampling period: `100` = slowly varying, `1` = i.i.d. (default `100`) |
| `-m` | model — linear: `final_round`, `original`, `larger_x0`, `different_Sigma_bar`, `different_system_convergence`; nonlinear: `NLink`, `higher_dim` |
| `-c` | control: `none` or `regulation` (default `none`); **`regulation` is valid for the linear models only** |
| `-e` | experiment: `random` (default) or `worst` |
| `-t` | number of trajectories (default `100`) |
| `-f`, `--force` | ignore the cache and recompute everything |

`-p` distinguishes the two mechanisms of the paper: a large period (slowly varying SDS) versus
`1` (i.i.d. per step). `higher_dim` is the 4-dimensional linear system — "nonlinear" above means
"not one of the five paper linear models", not "uses the nonlinear ambiguity set".

### Where the output goes

Results always land in `examples/scene2_simulated_ambiguity/data/gym/...`, the **scene root**,
not under `batch_experiment/`. The runner resolves that root from its own file location rather
than the working directory, so both entry points share one result tree and a run can be launched
from anywhere. The pickles and the figures (`error_bar.pdf`, plus an ellipse figure for the
linear models) are written there.

### Caching and safety

- The cache is **validating**: a step is reused only if its manifest still matches (source
  hashes, a parameter snapshot, each dependency's sha256, and the array shapes). If a parameter or
  the code changed it recomputes and says why. `-f` forces a full recompute; a directory with no
  manifest recomputes on first use.
- `-c regulation` on a nonlinear model **exits immediately with an error** rather than failing
  halfway through a run.
- Multiprocessing relies on the `fork` start method, because the nonlinear ambiguity set holds
  lambdas that cannot be pickled. The runner asserts this and fails loudly otherwise.
- A `-t` below the largest fixed plot index emits a `RuntimeWarning` and clamps instead of
  crashing — see [Getting Started](getting-started.md) for the message.

For convenience, `run.sh` (Linux/WSL) and `run.ps1` (Windows) drive `run.py` through six
experiments in sequence under one model table.
