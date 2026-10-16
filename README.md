# ppsds: probabilistic prediction for stochastic dynamical systems

**How predictable is a system you only partly know?** `ppsds` answers from both sides: the
**ceiling** any prediction could reach on a system you know exactly, and a predictor that comes
with a **floor** over every system you are willing to admit.

![Placing stacks where a stochastic system will go next, then watching the share that paid and its score settle](https://raw.githubusercontent.com/Xuzerken/ppsds/main/docs/assets/demo.gif)

**[Play it](https://xuzerken.github.io/ppsds/understand_ppsds_by_pmf_betting/)** — thirty seconds
on that table explains what this package is better than a paragraph can.

<details>
<summary>What you are looking at, and how it maps onto the package</summary>

A puck moves around a table. You put stacks where you think it will land next, the dealer drops a
ball, and every stack within `epsilon` of the landing pays its share of your bet. That is the whole
idea with the abstraction removed — the three things on screen are the three things the package is
made of:

| on screen | in the package |
|---|---|
| the moving puck | the **system** — a drift plus a known noise law |
| the stacks you place | the **probabilistic predictor** — a pmf over where the puck will be |
| how they pay | the **scoring rule** — the `epsilon`-logarithm score |

Playing it is free, needs no install, and keeps nothing: the page has no server behind it. When you
want the same thing done rigorously, the bets you place there export to a file the examples read
back — that is the second half of that story.

</details>

## Install

```
pip install ppsds
```

Core dependencies are `numpy`, `scipy`, `cvxpy` and `gymnasium`; Python 3.10 or newer. Only the
experiment and plotting periphery needs more:

```
pip install "ppsds[experiments]"     # adds matplotlib, pandas, seaborn, tqdm
```

## Where do you want to start?

| …just play with it | …put it on a system of mine | …check it against the papers |
|---|---|---|
| <img src="https://raw.githubusercontent.com/Xuzerken/ppsds/main/docs/assets/demo.gif" width="260" alt="the betting game"> | <img src="https://raw.githubusercontent.com/Xuzerken/ppsds/main/docs/assets/prediction_on_your_system.png" width="260" alt="a true trajectory and the predicted distribution at two steps"> | <img src="https://raw.githubusercontent.com/Xuzerken/ppsds/main/docs/assets/predictability.png" width="260" alt="score curves against the predictability ceiling"> |
| Nothing to install. It is a static page, and it is the fastest way to see what a probabilistic prediction *is*. | You have a simulator or a Gymnasium environment and want a forecast that stays honest when your model drifts. **This is the main path.** | Read the papers, reproduce their figures, or build a predictor of your own kind. |
| **[Play it →](https://xuzerken.github.io/ppsds/understand_ppsds_by_pmf_betting/)** | **[Put it on your system →](https://xuzerken.github.io/ppsds/examples/)** | **[Concepts →](https://xuzerken.github.io/ppsds/concepts/)** |

## The examples

Four stories, each self-contained. Every one prints the numbers it talks about.

| Story | For | What it is |
|---|---|---|
| [`understand_ppsds_by_pmf_betting`](https://github.com/Xuzerken/ppsds/tree/main/examples/understand_ppsds_by_pmf_betting) | anyone | The game above, then the same move over thousands of trajectories, measured against the ceiling no bettor can beat |
| [`apply_drpp_to_your_own_env`](https://github.com/Xuzerken/ppsds/tree/main/examples/apply_drpp_to_your_own_env) | you have an environment | Four steps to put the predictor on your own system, then the two ways it goes wrong — including why *"my model is bad"* is **not** by itself a reason it wins |
| [`understand_drpp_on_simulated_ambiguity`](https://github.com/Xuzerken/ppsds/tree/main/examples/understand_drpp_on_simulated_ambiguity) | you want the semantics | Declare an ambiguity set by hand, grow a system out of it, and watch three predictors face two meanings of "worst" |
| [`understand_predictability`](https://github.com/Xuzerken/ppsds/tree/main/examples/understand_predictability) | you want the ceiling | The best any prediction could do on a *known* system, and the paper's figure |

## A predictor on your own system

`ppsds` never asks your environment for anything. The whole integration is a **declaration**, and
the order matters: the model, then the family of ambiguity sets, then that family's parameters,
then the predictors.

```python
import numpy as np
from ppsds.ambiguity.additive_drift.conic_moment import LinearAmbiguitySet
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (
    NominalPredictor, EigDrpp,
)
from ppsds.scoring_rule.logarithm_score import LogarithmScore

# 1-2. The nominal model, and the family. Decide the family before its parameters:
#      families describe uncertainty in different languages, so it decides what step 3 supplies.
A_bar, B_bar = np.array([[1.0, 0.10], [0.0, 1.0]]), np.array([[1.0], [0.0]])
mu_bar, Sigma_bar = np.zeros(2), np.array([[1.0, 0.45], [0.45, 1.30]])

# 3. That family's parameters -- note which of these are squared.
alphas = {'A': 0.30, 'B': 0.30, 'mu': 0.25, 'Sigma': 3.0}   # only `mu` is squared
abg = LinearAmbiguitySet(A_bar, B_bar, mu_bar, Sigma_bar, alphas, f_err_max=0.30)

# 4. Predictors take their model from the ambiguity set -- never retype the literals.
nominal = NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar)
robust = EigDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma)
scorer = LogarithmScore()

x, _ = env.reset(seed=0)                 # `env` is yours; ppsds never modifies it
for _ in range(50):
    u = policy(x)
    nominal.update(x, u)
    robust.update(x, u)
    x, *_ = env.step(u)
    print(scorer.score(nominal.predict(), x), scorer.score(robust.predict(), x))
```

What comes out of that loop is a distribution per step, not a number — and the robust one is
deliberately **wider**. Here it is on an off-the-shelf environment (`Acrobot-v1`, after
identifying a linear model from data), with the true path drawn over it:

![The true trajectory through one plane of Acrobot's state space, with the 90% predictive distribution at two steps: nominal in blue, Eig-DRPP in red, and the robust one visibly wider](https://raw.githubusercontent.com/Xuzerken/ppsds/main/docs/assets/prediction_on_your_system.png)

That width is what you pay for the floor. How much it is worth depends on how confidently you
declared — not on how accurate your model is. That is the second half of the
[`apply_drpp_to_your_own_env`](https://github.com/Xuzerken/ppsds/tree/main/examples/apply_drpp_to_your_own_env)
story, and it is the part that surprises people.

## Documentation

**https://xuzerken.github.io/ppsds/**

| | |
|---|---|
| [Getting Started](https://xuzerken.github.io/ppsds/getting-started/) | install it, run something, put it on your own system, and the pitfalls |
| [Concepts](https://xuzerken.github.io/ppsds/concepts/) | what is optimized, what the worst case may touch, what the guarantee costs |
| [Examples](https://xuzerken.github.io/ppsds/examples/) | every example, with the command and the real output |
| [Predictability](https://xuzerken.github.io/ppsds/predictability/) | the other side: the ceiling, which needs no predictor at all |
| [Developer Guide](https://xuzerken.github.io/ppsds/extending/) | write your own family of ambiguity sets, end to end |
| [Troubleshooting](https://xuzerken.github.io/ppsds/extending-faq/) | the traps, and what to do about them |
| [API](https://xuzerken.github.io/ppsds/api/) | generated from the docstrings |
| [Citation](https://xuzerken.github.io/ppsds/citation/) | the software citation, and the references it implements |

If you only read one page and you intend to rely on the guarantee, make it
[Concepts](https://xuzerken.github.io/ppsds/concepts/): the guarantee is **conditional**, and the
condition is a declaration you write, not something the library can check for you.

<details>
<summary>A little more about what is inside</summary>

`ppsds` predicts the next state of a stochastic dynamical system as a **distribution** rather than
a point. It is for systems you can **step** (a simulator, or a Gymnasium environment) and can
model, but not exactly: an underactuated robot, a vehicle or drone held near its operating point, a
plant inside a process simulator.

An ordinary predictor trusts that model. That works until the real system drifts away from it, and
then the prediction is quietly too narrow: **overconfident** exactly when you needed it to be
honest.

`ppsds` ships one remedy today: **DRPP**, distributionally robust probabilistic prediction. You
declare an **ambiguity set** — the systems you are willing to admit — and DRPP returns the
predictive distribution whose *worst-case* expected score over that set is as high as it can be:
a floor that holds for every system in the set, for the price of a little average performance.

That DRPP is the one for **conic-moment** ambiguity sets, on an **additive-drift** system. It is
the first of a template rather than a single method: each family of ambiguity sets yields its own
DRPP, and the `drpp` subpackage is built to hold more. Zoom out one level and DRPP is one *kind*
of **probabilistic predictor**; `ppsds.predictability` answers the same question from the other
side, for a system you *do* know exactly.

</details>

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
entry; an archived release with a DOI will be added once the first version is tagged. The
**papers** the predictors implement are listed on the
[citation page](https://xuzerken.github.io/ppsds/citation/) rather than here.

## Contributing, licence

- [Contributing](https://github.com/Xuzerken/ppsds/blob/main/CONTRIBUTING.md): setup, tests, and
  the rule that a change must not move a number
- [MIT](https://github.com/Xuzerken/ppsds/blob/main/LICENSE): use it freely, keep the copyright
  and licence notice
