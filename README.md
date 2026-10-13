# ppsds: probabilistic prediction for stochastic dynamical systems

`ppsds` predicts the next state of a stochastic dynamical system as a **distribution** rather
than a point. It is for systems you can **step** (a simulator, or a Gymnasium environment) and can
model, but not exactly: an underactuated robot, a vehicle or drone held near its operating point, a
plant inside a process simulator.

An ordinary predictor trusts that model. That works until the real system drifts away from it, and
then the prediction is quietly too narrow: **overconfident** exactly when you needed it to be
honest.

`ppsds` ships one remedy today: **DRPP**, distributionally robust probabilistic prediction. You
declare an **ambiguity set**, the systems you are willing to admit, and DRPP returns the
predictive distribution whose *worst-case* expected score over that set is as high as it can be: a
**floor that holds for every system in the set**, for the price of a little average performance.
[Concepts](https://xuzerken.github.io/ppsds/concepts/) is about what that costs and what it
assumes.

That DRPP is the one for **conic-moment** ambiguity sets, on an **additive-drift** system. It is
the first of a template rather than a single method: each family of ambiguity sets yields its own
DRPP, and the `drpp` subpackage is built to hold more.

Zoom out one level. DRPP is one *kind* of **probabilistic predictor** (PP). The kinds sit side by
side in `ppsds.probabilistic_predictor`, one subpackage each, and `drpp` is the first of them. A
data-driven setting would add a **learning-based PP** there; a theoretical one would add the
**optimal PP** that attains **probabilistic predictability**, the best a prediction of a *known*
system can do. That upper bound, and DRPP's guaranteed lower bound for a system known only through
an ambiguity set, answer the same question from two sides.

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
| [Developer Guide](https://xuzerken.github.io/ppsds/extending/) | write your own family of ambiguity sets, end to end |
| [Troubleshooting](https://xuzerken.github.io/ppsds/extending-faq/) | the traps, and what to do about them |
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
[citation page](https://xuzerken.github.io/ppsds/citation/) rather than here. This repository is
the software, and it will carry predictors from more than one paper.

## Contributing, licence

- [Contributing](https://github.com/Xuzerken/ppsds/blob/main/CONTRIBUTING.md): setup, tests, and
  the rule that a change must not move a number
- [MIT](https://github.com/Xuzerken/ppsds/blob/main/LICENSE): use it freely, keep the copyright
  and licence notice
