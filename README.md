# ppsds

**Distributionally robust probabilistic prediction for stochastic dynamical systems.**

`ppsds` turns "predict the next state" from a point estimate into a **predictive distribution
with a worst-case guarantee**: as long as the true system lies inside the ambiguity set you
declared, the expected score of the prediction cannot fall below a stated lower bound.

This is the reference implementation accompanying

> T. Xu and J. He, *Distributionally Robust Probabilistic Prediction for Stochastic Dynamical
> Systems*, IEEE Transactions on Automatic Control.

---

## Why not just fit a model and predict?

An ordinary predictor trusts its own model: it predicts using the dynamics and noise it believes
in. When the system behaves as expected that is hard to beat — and it is also **overconfident**
the moment it does not. The dynamics drift, the noise has a different covariance, and the
prediction is quietly too narrow, scoring badly on the states that actually occur.

DRPP changes the objective. Instead of the nominal model it optimizes for the **worst system
allowed by the ambiguity set you declared**:

```
    maximize over predictors p_hat,
    minimize over distributions P in the ambiguity set,
        E_P[ score(p_hat) ]
```

The price is a little average performance — the predictive covariance becomes more
conservative. What it buys is a floor that holds in the worst case.

**This is the part that surprises people:** a worse model does *not* by itself make DRPP win.
If your declared covariance came from your own residuals, the ordinary predictor is already
calibrated to the errors it will meet, and it wins. DRPP pays off when your declaration is
**optimistic** — and too little identification data is the most common reason for that. See
[When DRPP actually wins](#when-drpp-actually-wins) below, which reports the measurement.

---

## Install

```bash
pip install ppsds
```

Core dependencies are `numpy`, `scipy`, `cvxpy` and `gymnasium`. Only the experiment and
plotting periphery needs more:

```bash
pip install "ppsds[experiments]"     # adds matplotlib, pandas, seaborn, tqdm
```

`pip install ppsds` does not import `cvxpy` until a cone program is actually solved, so using
the nominal predictor or the environment layer costs nothing extra.

Requires Python 3.10 or newer.

---

## Quick start

```bash
pip install ppsds
python -c "import ppsds; print(ppsds.__version__)"
```

Then, on any stochastic dynamical system you already have as a gymnasium environment:

```python
import numpy as np
from ppsds.ambiguity.additive_drift.conic_moment import LinearAmbiguitySet
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (
    NominalPredictor, EigDrpp,
)
from ppsds.scoring_rule.logarithm_score import LogarithmScore

# 1. The nominal model -- "what I believe my system is". Independent of the family you pick.
A_bar = np.array([[1.0, 0.10], [0.0, 1.0]])
B_bar = np.array([[1.0], [0.0]])
mu_bar = np.zeros(2)
Sigma_bar = np.array([[1.0, 0.45], [0.45, 1.30]])
f_bar = lambda x, u: A_bar @ x + B_bar @ u

# 2. The family of the ambiguity set -- conic moment is the one implemented here.
#    Decide this **before** step 3: different families describe uncertainty in different
#    languages, so the family determines what step 3 has to supply.

# 3. That family's parameterization -- note which of these are squared.
alphas = {
    'A': 0.30,       # largest perturbation of A's superdiagonal  (un-squared)
    'B': 0.30,       # largest perturbation of B[0, -1]           (un-squared)
    'mu': 0.25,      # SQUARED radius of the noise mean: |true mean - mu_bar| <= sqrt(0.25)
    'Sigma': 3.0,    # covariance scaling: true covariance is dominated by 3.0 * Sigma_bar
}
f_err_max = 0.30     # hard bound on the drift **norm**             (un-squared)

abg = LinearAmbiguitySet(A_bar, B_bar, mu_bar, Sigma_bar, alphas, f_err_max)

# 4. Predictors take their model from the ambiguity set -- never retype the literals.
predictors = {
    'nominal':  NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar),
    'Eig-DRPP': EigDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma),
}
scorer = LogarithmScore()

x, _ = env.reset(seed=0)                  # `env` is yours; DRPP never needs to modify it
for _ in range(50):
    u = policy(x)
    for pred in predictors.values():
        pred.update(x, u)                 # update from the current observation
    x, *_ = env.step(u)
    for name, pred in predictors.items():
        score = scorer.score(pred.predict(), x)
```

Runnable versions: [scene1 quickstart][s1quick] (about one second) and
[scene2 quickstart][s2quick] (about two seconds, comparing `random` and `worst` SDS).

[s1quick]: https://github.com/Xuzerken/ppsds/blob/main/examples/scene1_own_environment/quickstart.py
[s2quick]: https://github.com/Xuzerken/ppsds/blob/main/examples/scene2_simulated_ambiguity/quickstart.py

---

## The two ways to use this

| | **Your own environment** | **Simulated ambiguity** |
|---|---|---|
| You have | an existing gymnasium environment | only the paper, and want to reproduce it |
| You declare | nominal model + ambiguity set | additionally the ambiguity set and dynamics generator |
| You use | `predictor` + `scoring_rule` | plus `environment` |
| Start at | [scene1 quickstart][s1quick] | [scene2 quickstart][s2quick] |

`ppsds.environment` exists **for the second row only**. If you are putting DRPP on a system you
already have, you never touch it.

### Declaring an ambiguity set is the whole job

DRPP never asks your environment for anything. The environment is only stepped; the predictor
takes what it needs from the set you declared. The one thing worth real thought is step 3 above:
**if the set does not contain the true system, the guarantee is vacuous, and if it is too wide,
the predictions become needlessly conservative.** Those radii matter more than the precision of
your nominal model.

If your environment is deterministic — most classic gymnasium environments are — it has no
process noise, so the covariance is something you **declare** rather than measure. That is not a
defect: the model is a statement about what you believe, and a deterministic environment simply
means the noise is entirely unmodelled. One practical caveat: a residual covariance computed
from data is often numerically singular, and the logarithmic score requires a density, so it has
to be regularized — and the scale of that regularization *is* the modelling decision. The
[Acrobot example][s1gym] works through this, and [Pendulum][s1pend] through the structurally
singular case.

[s1gym]: https://github.com/Xuzerken/ppsds/blob/main/examples/scene1_own_environment/with_standard_gym_env.py
[s1pend]: https://github.com/Xuzerken/ppsds/blob/main/examples/scene1_own_environment/pendulum_singular_covariance.py

---

## When DRPP actually wins

"It's the worst case" is a property of the guarantee, not a claim about every trajectory. On a
single ordinary trajectory the nominal predictor usually scores **higher** — that is
conservatism being paid for and not yet collected.

Measured on Acrobot (a 6-dimensional chaotic system, fixed policy, evaluation on trajectories
**different** from the identification data):

| identification data | estimated `Sigma_bar` | nominal | Eig-DRPP | |
|---|---|---|---|---|
| 20 episodes | calibrated (residual std 0.69) | **-1.14** | -7.10 | nominal wins |
| 1 episode | **underestimated 3.9x** (0.18) | -6.06 | **-2.51** | DRPP wins |

In the first row the declaration is accurate, so robustification is pure cost. In the second the
declaration is optimistic, the nominal prediction is too narrow, and DRPP's covariance already
accounts for that.

Run either regime yourself: `python examples/scene1_own_environment/with_standard_gym_env.py`.

To see DRPP on the problem it was designed for, run the adversarial experiment, where each
predictor faces the worst SDS chosen **specifically against it**:

```bash
cd examples/scene2_simulated_ambiguity/batch_experiment
python run.py -n 8 -s 3 -m NLink -c none -t 100 -e worst
```

---

## The three predictors

| predictor | predictive covariance |
|---|---|
| `NominalPredictor` | the nominal `Sigma_bar` |
| `NoiseDrpp` | `gamma[2] * Sigma_bar` — robust to noise only |
| `EigDrpp` | one cone program per step (the paper's method) |

`EigDrpp` optimizes the **eigenvalues** of the predictive covariance while its **eigenvectors
are fixed** to those of `Sigma_bar`, which is what makes the minimax problem solvable as a cone
program. It is provably suboptimal, and the paper bounds the gap.

---

## What is in the package

| path | contents |
|---|---|
| `ppsds.ambiguity` | ambiguity sets, laid out as assumption / family |
| `ppsds.probabilistic_predictor` | the predictors above (the paper's contribution) |
| `ppsds.scoring_rule` | the logarithmic scoring rule |
| `ppsds.environment` | the gymnasium layer — for simulated ambiguity only |
| `ppsds.dynamics` | robotic-arm models and system identification |
| `ppsds.numerics` | small numerical helpers |
| `ppsds.experiment`, `ppsds.visualization` | experiment driver and figures — need `ppsds[experiments]` |

The module path states the scope: `ambiguity/additive_drift/conic_moment` is
*assumption* (additive drift) / *family* (conic moment). A different family is a new module
beside it rather than a branch inside it.

---

## Known limitations

Stated plainly, because a research release that hides its edges is worse than one that names
them:

- **The conic-moment parameterization is example-specific in one respect.** `alphas['A']` and
  `alphas['B']` perturb only the superdiagonal of `A` and only `B[0, -1]` — the structure of the
  paper's own example. It does not describe arbitrary uncertainty in `A` and `B`.
- **The batch experiment runner requires the `fork` start method**, because one ambiguity set
  holds lambdas and cannot be pickled. It asserts this and fails loudly rather than degrading
  silently.
- **The README is ahead of PyPI's renderer in one place by design:** the formula above is a
  fenced code block because PyPI does not render LaTeX.

---

## Documentation and contributing

Full guides, the API reference and worked examples: **https://xuzerken.github.io/ppsds/**

- Issues and questions: https://github.com/Xuzerken/ppsds/issues
- Contributing: [CONTRIBUTING.md](https://github.com/Xuzerken/ppsds/blob/main/CONTRIBUTING.md)
- Release notes: [CHANGELOG.md](https://github.com/Xuzerken/ppsds/blob/main/CHANGELOG.md)

---

## Citation

```bibtex
@article{xuDRPP,
  title   = {Distributionally Robust Probabilistic Prediction for Stochastic Dynamical Systems},
  author  = {Xu, Tao and He, Jianping},
  journal = {IEEE Transactions on Automatic Control}
}
```

GitHub's "Cite this repository" button reads
[CITATION.cff](https://github.com/Xuzerken/ppsds/blob/main/CITATION.cff) and produces the same
entry.

---

## License

[MIT](https://github.com/Xuzerken/ppsds/blob/main/LICENSE) — use it freely, keep the copyright
and license notice.
