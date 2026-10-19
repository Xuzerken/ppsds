# Put DRPP on your own system

> **Put the predictor on your own system — and find out when it pays for itself and when it charges you.**

| | |
|---|---|
| **Who it is for** | You have an environment and you intend to use the result, not just read about it |
| **Prerequisites** | A Gymnasium environment and the idea of a nominal model. No paper required |
| **Runs on** | `pip install ppsds` alone — nothing here needs the environment layer |

Three scripts, in this order. The first is the workflow; the other two are the two ways it
surprises people.

## Act 1 — Four steps, and the order matters

```bash
python examples/apply_drpp_to_your_own_env/quickstart.py     # about one second
```

`ppsds` never asks your environment for anything: you step it, and you hand the observed states to
the predictor. The whole integration is a **declaration**, in this order — the model, the family of
ambiguity sets, that family's parameters, then the predictors. The first script stands up a
two-state linear system as a stand-in for "your own environment" and scores the nominal predictor
and Eig-DRPP over 50 steps of zero control:

```
Running 50 steps on **one ordinary trajectory** (your own environment, zero control)

predictor     mean log score   note
----------------------------------------------------------------
nominal               -2.867   trusts the nominal model
Eig-DRPP              -3.366   made robust to the worst case
```

**What to conclude:** `nominal` scores higher than Eig-DRPP on this trajectory, and that is the
expected outcome rather than a failure. DRPP optimizes the worst case; on a single ordinary
rollout that conservatism is a cost with nothing yet to collect. The example's real subject is the
workflow: the environment is stepped but never modified, and both predictors take their model from
the same ambiguity set, so they differ only in whether they account for the ambiguity.

The full declaration, with the code and the pitfalls, is the
[Getting Started tutorial](../../tutorials/getting-started.md).

## Act 2 — “My model is bad” is not a reason DRPP wins

```bash
python examples/apply_drpp_to_your_own_env/with_standard_gym_env.py     # about 15 seconds
```

This one puts DRPP on **Acrobot-v1**: six-dimensional, underactuated, chaotic, and not an
environment you wrote. The nominal model and the ambiguity set therefore have to be identified from
data. Then it answers the question Act 1 leaves open: **if the model is this bad, does DRPP win?**

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

**What to conclude:** a bad model is *not* what makes DRPP win. Section 3 shows DRPP losing by a
wide margin even though the linear residual is 74% of the state increment, because `Sigma_bar` was
estimated from your own residuals and is therefore calibrated to the errors the predictor will
actually meet. Section 4 isolates the variable that does decide it: with 20 identification episodes
the estimate is calibrated and nominal wins, while one episode underestimates it by about 3.9×, the
nominal prediction is too narrow, and DRPP wins. **Too little identification data is the most common
source of optimism**, and optimism — not model error — is what DRPP is paid for.

## Act 3 — The covariance you computed may be unusable

```bash
python examples/apply_drpp_to_your_own_env/pendulum_singular_covariance.py     # about 4 seconds
```

A second pitfall on the same path, on **Pendulum-v1**: the residual covariance you estimate has a
**structural** zero eigenvalue. Not because the residuals are small, but because the observation
lies on the cylinder `cos^2 + sin^2 = 1` and one state update is exactly linear in the observation,
so that component's residual is identically zero.

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
and a lower bound of `-inf` guarantees nothing, so `Sigma_bar` must be regularized before DRPP can
use it. But the regularization floor is not a cosmetic constant: moving it from `1e-6` to `1e-2`
moves the score by 5.5 points, because it sets the width of the predictive distribution. Lifting a
structurally dead direction to positive-definite **is** a modelling statement — “how much
uncertainty am I willing to admit here” — and should come from physics or from a first run you
scrutinize, not from a number invented to make the code run.

## Next

- **The workflow, step by step**: [Getting Started](../../tutorials/getting-started.md)
- **What the guarantee costs, and when it is vacuous**: [Research](../../research/index.md)
- **The same questions on a system you do *not* have**: [what “worst case” means](../understand_drpp_on_simulated_ambiguity/index.md)
- **When a number looks wrong**: [Troubleshooting](../../tutorials/troubleshooting.md)
