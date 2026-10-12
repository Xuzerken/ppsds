# ppsds

**Probabilistic prediction for stochastic dynamical systems.**

`ppsds` predicts the next state of a stochastic dynamical system as a **distribution** rather
than a point — and it ships predictors that stay honest when your model of the system is wrong.

Predictors come in families, and the package is built to hold more than one:

- **distributionally robust** (**DRPP**) — optimizes for the *worst* system an ambiguity set
  allows, and comes with a worst-case guarantee on the expected score. This release implements it
  over **conic-moment** ambiguity sets.
- **nominal** and **noise-robust** — the ordinary baselines, so the guarantee can be measured
  against what it costs.

The guarantee is conditional, and understanding the condition is most of using the package well:
a predictor does not become better by having a *worse* model, and the declaration you write
matters more than how accurate the model is.
[Concepts](https://xuzerken.github.io/ppsds/concepts/) is about exactly that.

## Install

```
pip install ppsds
```

Core dependencies are `numpy`, `scipy`, `cvxpy` and `gymnasium`; Python 3.10 or newer. Only the
experiment and plotting periphery needs more:

```
pip install "ppsds[experiments]"     # adds matplotlib, pandas, seaborn, tqdm
```

`import ppsds` does not import `cvxpy` until a cone program is actually solved, so the nominal
predictor and the environment layer cost nothing extra.

## A predictor on your own system

`ppsds` never asks your environment for anything: the whole integration is a **declaration**.
Four steps, and the order matters.

```python
import numpy as np
from ppsds.ambiguity.additive_drift.conic_moment import LinearAmbiguitySet
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (
    NominalPredictor, EigDrpp,
)
from ppsds.scoring_rule.logarithm_score import LogarithmScore

# 1. The nominal model -- what you believe your system is. Independent of the family you pick.
A_bar = np.array([[1.0, 0.10], [0.0, 1.0]])
B_bar = np.array([[1.0], [0.0]])
mu_bar = np.zeros(2)
Sigma_bar = np.array([[1.0, 0.45], [0.45, 1.30]])
f_bar = lambda x, u: A_bar @ x + B_bar @ u

# 2. The family of the ambiguity set. Decide this **before** step 3: families describe
#    uncertainty in different languages, so the family determines what step 3 must supply.

# 3. That family's parameterization -- and note which of these are squared.
alphas = {
    'A': 0.30,       # largest perturbation of A's superdiagonal   (un-squared)
    'B': 0.30,       # largest perturbation of B[0, -1]            (un-squared)
    'mu': 0.25,      # SQUARED radius of the noise mean: |true mean - mu_bar| <= sqrt(0.25)
    'Sigma': 3.0,    # covariance scaling: true covariance is dominated by 3.0 * Sigma_bar
}
f_err_max = 0.30     # hard bound on the drift **norm**              (un-squared)

abg = LinearAmbiguitySet(A_bar, B_bar, mu_bar, Sigma_bar, alphas, f_err_max)

# 4. Predictors take their model from the ambiguity set -- never retype the literals.
predictors = {
    'nominal':  NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar),
    'Eig-DRPP': EigDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma),
}
scorer = LogarithmScore()

x, _ = env.reset(seed=0)                 # `env` is yours; ppsds never modifies it
for _ in range(50):
    u = policy(x)
    for pred in predictors.values():
        pred.update(x, u)                # update from the current observation
    x, *_ = env.step(u)
    for name, pred in predictors.items():
        score = scorer.score(pred.predict(), x)
```

## Documentation

**https://xuzerken.github.io/ppsds/**

| | |
|---|---|
| [Getting Started](https://xuzerken.github.io/ppsds/getting-started/) | install it, run something, put it on your own system, and the pitfalls |
| [Concepts](https://xuzerken.github.io/ppsds/concepts/) | what is optimized, what the worst case may touch, what the guarantee costs |
| [Examples](https://xuzerken.github.io/ppsds/examples/) | every example, with the command and the real output |
| [API](https://xuzerken.github.io/ppsds/api/) | generated from the docstrings |
| [Citation](https://xuzerken.github.io/ppsds/citation/) | the software citation, and the references it implements |

## Citation

If you use `ppsds`, cite the software:

```bibtex
@software{ppsds,
  title   = {ppsds: probabilistic prediction for stochastic dynamical systems},
  author  = {Xu, Tao and He, Jianping},
  license = {MIT},
  url     = {https://github.com/Xuzerken/ppsds}
}
```

GitHub's **Cite this repository** button reads
[`CITATION.cff`](https://github.com/Xuzerken/ppsds/blob/main/CITATION.cff) and produces the same
entry; an archived release with a DOI will be added once the first version is tagged.

The **papers** that the predictors implement are listed as references on the
[citation page](https://xuzerken.github.io/ppsds/citation/) rather than here — this repository is
the software, and it will carry predictors from more than one paper.

## Contributing, licence

- [Contributing](https://github.com/Xuzerken/ppsds/blob/main/CONTRIBUTING.md) — setup, tests, and
  the rule that a change must not move a number
- [MIT](https://github.com/Xuzerken/ppsds/blob/main/LICENSE) — use it freely, keep the copyright
  and licence notice
