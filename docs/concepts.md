# Concepts

`ppsds` predicts the next state of a stochastic dynamical system as a **distribution**, and it
does so in a way that survives being wrong about the system. This page is about the **DRPP** that
ships today, the one for **conic-moment** ambiguity sets, and what that sentence means for it:
what is optimized, what the worst case is allowed to touch, what the numbers in an ambiguity set
physically are, and what the method gives up. It assumes no familiarity with the paper.

For the mechanics of putting it on a system, see [Getting Started](getting-started.md); for
runnable code, [Examples](examples.md).

---

## What DRPP optimizes

An ordinary probabilistic predictor answers "given where I am, what is the distribution of the
next state?" by trusting its own model. DRPP answers the same question but is judged by a
different objective. It maximizes, over predictors, the expected score under the system that is
**worst for that predictor**:

```
    maximize over predictors p_hat
        minimize over systems P in the ambiguity set
            E_P[ log p_hat(x_{k+1} | x_k, u_k) ]
```

Read it inside out. The inner minimization is an adversary that picks a system (a dynamics and a
noise distribution) out of the ambiguity set you declared. The outer maximization is your
predictor, which may be designed knowing the set but not which member will be chosen. The
quantity being scored is the logarithmic score of the predictive density at the state that
actually occurs: a density that puts mass where the state lands scores well.

What this buys, compared with fitting a model and predicting from it, is a **guarantee that is
uniform over the set**: for every system the set admits, the predictor's expected score is at
least the value the maximin attains. A fitted model gives a point estimate of the dynamics; the
maximin gives a floor that holds across a whole neighborhood of models you were willing to
write down. The cost is that the floor is not the highest score available to a predictor that
happens to know the system. That is the trade, and the rest of this page is about how it is
paid.

---

## What the worst case actually ranges over

**The ambiguity set you declared, and nothing else.** This is the single most important thing to
understand about using the package correctly, and it cuts both ways.

- The adversary cannot invent a system you did not allow. If you declared three radii, the worst
  case is the worst combination of dynamics and noise those three radii admit, and no more.
- If the true system lies **outside** the set, the guarantee says nothing about it. The bound is
  a statement over the set, and a system outside the set is simply not covered.

The package never inspects your environment. The predictor takes everything it needs (the
nominal model and the radii) from the set you declared, and the environment (when you use one)
is only stepped. So the declaration is not a description of your system discovered from data; it
is a statement about what you believe, and the quality of the guarantee is exactly the quality
of that statement.

In the environment layer the adversary is a **best response**: the worst system is selected
*after* the predictor emits its distribution, using the predictor's own predicted covariance.
That ordering is the paper's semantics, not an implementation accident: the worst case reacts
to the prediction it is trying to defeat.

---

## The conic-moment family: three radii

The family implemented here constrains the **first two moments** of the noise, plus the norm of
the drift error, through a small number of scalars called `gamma`. Each one answers a different
question about how wrong you are willing to be.

| quantity | where it lives | what it bounds | squared? |
|---|---|---|---|
| drift radius | `gamma[0]`, the callable behind `f_err(x, u)` | the **squared norm** of the drift error: `||delta_f||^2 <= f_err(x, u)` | **squared** |
| drift cap | `f_err_max` | the norm of the drift error itself: `||delta_f|| <= f_err_max` | **un-squared** |
| noise-mean radius | `gamma[1]`, set by `alphas['mu']` | the **squared norm** of the noise-mean deviation: `||mu - mu_bar||^2 <= gamma[1]` | **squared** |
| covariance scale | `gamma[2]`, set by `alphas['Sigma']` | the true noise covariance stays dominated by `gamma[2] * Sigma_bar` | neither; a scale factor |

> **Which are squared is the easiest thing to get wrong**, because the conventions here are not
> uniform, and they are deliberate. `f_err_max` is **un-squared**: it is a hard cap on the drift
> norm, and it doubles as the ball radius the sampler draws from, so it has to be a norm.
> `f_err(x, u)` and `gamma[1]` are **squared**: they are radii inside the cone program, which
> works with squared norms. In the linear parameterization the matrix perturbations
> `alphas['A']` and `alphas['B']` are also **un-squared**: they are the largest perturbation of
> an entry, so the sampler draws uniformly on `[-alphas['A'], alphas['A']]`.

Physically, the three numbers are three independent ways for your belief to be wrong:

- **Drift radius (`gamma[0]`).** How far the true drift may sit from the nominal drift, measured
  as a norm. It is a function of the state and control rather than a constant, because the size
  of a drift error is proportional to how far the system has moved and how hard you are pushing
  it. It enters the predictive **mean only**.
- **Noise-mean radius (`gamma[1]`).** How far the true noise mean may sit from the nominal one.
  It enters **both** the predictive mean and the covariance.
- **Covariance scale (`gamma[2]`).** How much larger the true noise covariance may be than the
  nominal one. It enters the predictive **covariance only**.

The reason the noise radii carry the weight is one identity worth memorizing:

> **The predictive covariance is the noise covariance.**

The prediction is a distribution whose spread is not modeled separately: it *is* the noise
covariance of your belief. So "how wrong can the noise be" directly determines how wide the
prediction has to be, and the covariance scale `gamma[2]` sets that width. The adversary's
freedom on the noise is the prediction's width; the adversary's freedom on the drift is a shift
of the mean.

Concretely, the worst case is built by selecting **one** worst eigen-direction `j` of the nominal
covariance, then shifting the predictive mean along it and reshaping the covariance around the
same axis. The mean shift combines the drift and noise-mean radii along that direction, and the
covariance is scaled by `gamma[2]` while being tightened along the chosen axis. One direction,
chosen to hurt most, not a wholesale inflation of the uncertainty.

One clarification that is easy to misread: **the ambiguity set does not assume the noise is
Gaussian.** It constrains moments. Gaussianity appears only downstream, in the kernel used to
sample noise and in the logarithmic scoring rule. "Moment constraints" is not shorthand for
"distributional assumptions".

---

## The ordinary predictor, and why "my model is bad" is not the reason DRPP wins

`NominalPredictor` is the baseline that does no robustification at all. Its predictive mean comes
from the nominal drift and the nominal noise mean, and its predictive covariance is exactly
`Sigma_bar`. Against a system that behaves as expected, that is the right answer.

It is tempting to read DRPP as "for when your model is bad". That is not the mechanism, and
getting this wrong leads people to declare radii for the wrong reason. The point is
**calibration**, not model accuracy:

> The score is decided by whether the predictive spread matches the spread of the error that
> actually occurs, not by how small that error is.

If `Sigma_bar` was estimated from the residuals of the very system you are predicting, then it
already reflects however wrong your model is. A structurally wrong model with a covariance fitted
to its own residuals is still well calibrated: its errors are large, and its declared spread is
large to match. Robustifying means widening the prediction, and a wider density scores worse
when it was already the right width. So when the declaration is accurate, robustification is
pure cost, and the nominal predictor wins, on average and usually on every ordinary trajectory.

DRPP pays off in the **other** regime: when the declaration is **optimistic**: the declared
spread is narrower than the errors that actually occur. Then the nominal prediction is too sharp,
and it pays a large penalty on the states that fall outside it. The most common reason a
declaration is optimistic is **too little identification data**: a covariance estimated from a
handful of residuals is itself biased low, so the "accurate" model was never accurate about its
own error. There the robust covariance is not a premium being paid; it is covering an error the
nominal covariance already had.

This is why the honest framing is: DRPP trades average performance for a worst-case floor, and
the trade is favorable exactly when your declared set is doing real work.

---

## What `EigDrpp` gives up

The exact DRPP is **hard**. The outer step, choosing the predictive covariance that maximizes
the worst-case score, is a nonconvex semidefinite program whose global optimum is NP-hard, and a
single-step DRPP is equivalent to finding a global minimax point, which is NP-hard even in very
simple settings. Something has to be relaxed.

The relaxation is a restriction of the predictor to a **smaller family**. The eigenvectors of the
predictive covariance are **fixed** to those of the nominal covariance `Sigma_bar`, and only the
**eigenvalues** are optimized. "Eigenvectors fixed" is exactly what the name says:

```
    eigenvectors of Sigma_pred  :=  eigenvectors of Sigma_bar   (fixed, not optimized)
    eigenvalues  of Sigma_pred  :=  optimized each step
```

With the eigenvectors pinned, the functional maximin collapses into a **Euclidean cone program**,
solved per step by `eigvec_fixed_minmax` using CVXPY and SCS. That collapse is the whole reason
the problem becomes tractable: without fixing the basis there is no cone program to write down.

What it gives up is **optimality, not the guarantee**. `EigDrpp` returns a **lower bound** on the
achievable worst-case value. The bound still holds for every system in the set (that property is
untouched), but its value may sit below the true optimum, because the predictor was not allowed to
rotate its covariance. In one line: *Eig-DRPP is not a weaker guarantee; it is the same guarantee
at a more conservative value.* That is a design trade, not a defect.

The package's three predictors are points along this spectrum:

| predictor | predictive covariance | what it is robust to |
|---|---|---|
| `NominalPredictor` | `Sigma_bar` | nothing; it trusts the nominal model |
| `NoiseDrpp` | `gamma[2] * Sigma_bar` | the noise only; the drift radius is ignored |
| `EigDrpp` | solved per step | both, within the fixed-eigenvector relaxation |

### Two bounds bracket the answer

Two quantities in the ambiguity set frame the achievable score from either side, and comparing
them is the clearest way to see what the relaxation costs:

- `value_upper_bound_true_transition()`: the score achievable when the **true transition is
  known**. No predictor can beat this, so it is an upper bound on what any predictor can score
  inside that set.
- `value_lower_bound_eigvec_fixed()`: the score `EigDrpp` **guarantees** under the fixed-eigenvector
  relaxation. It is computed *by* that minmax, which is why the ambiguity set depends on the
  predictor: the lower bound has no meaning without it.

The gap between them is the price of making the problem solvable.

### Two bounds one level up, and they are not these two

The pair above brackets the score for **one declared ambiguity set**. A different pair sits one
level up, and "upper" and "lower" must not be read across the gap:

- **Probabilistic predictability** asks how well **any** predictor could do for a system that is
  **known exactly**. That is an upper bound on the achievable expected score.
- **DRPP** asks what a predictor can **guarantee** for a system that is **not** known exactly,
  only bounded by an ambiguity set. That is a lower bound, and it holds for every system in the
  set, at the cost of average performance.

`value_upper_bound_true_transition()` is the first idea applied to one set: the score a predictor
could reach if it knew the true transition *inside that set*. It is a case of the predictability
upper bound, not the DRPP lower bound: the DRPP lower bound is
`value_lower_bound_eigvec_fixed()`, and within a single set the gap between those two is the price
of the relaxation.

The predictability half of that pair now has its own page — see **[Predictability](predictability.md)**
for the `epsilon`-logarithm score, the upper bound `d log(2 eps) - h_d(q)`, and how to evaluate a
predictor you wrote yourself, including one whose prediction is a pmf.

---

## The two ways to use the package

There are two distinct jobs, and they use different parts of the package. Which one you are doing
decides what you have to supply.

| | Your own environment | Simulated ambiguity |
|---|---|---|
| You have | an existing gymnasium environment | only the paper, and want to reproduce it |
| You supply | nominal model + ambiguity set (agent side) | additionally the ambiguity set and the dynamics generator (environment side) |
| You use | `ppsds.probabilistic_predictor` + `ppsds.scoring_rule` | plus `ppsds.environment` |

`ppsds.environment` exists **for the second column only**. If you are putting DRPP on a system you
already have, you never touch it: you step your own environment and hand the observed states to
the predictor. The environment layer is for the case where you also need to *generate* the
dynamics from a declared set, so that the ground truth and the declaration are the same object.

Either way, the one part that deserves real thought is the declaration: the radii matter more
than the precision of the nominal model.

---

## Honest limits

- **The guarantee is conditional on the declaration.** It says "for every system in the declared
  set, the expected score is at least this much". If the true system falls outside the set, the
  statement is vacuous: nothing is claimed about it. The set has to contain reality for the
  guarantee to mean anything.
- **A set that is too wide costs you average performance.** Admitting systems that never occur
  forces a wider predictive covariance than the situation needs, and the score rewards sharpness.
  The premium is only returned if the extra width was actually required.
- **A set that is too narrow voids the guarantee**, and in simulation it shows up as the
  guarantee apparently failing. That is a mis-declared set, not a broken algorithm. When a
  simulated experiment seems to violate a bound, look first at whether the declaration covers the
  truth.
- **The set must contain its own generator.** If you write a sampler for simulated dynamics, its
  samples must land inside the declared set, or your ground truth is outside the set by
  construction. This is a design invariant, not a detail. A concrete instance: the linear drift
  bound is the **sum** of the control and state contributions, because that sum is the tight
  bound for the perturbation the sampler produces; a smaller expression would place some samples
  outside the set: the set could not contain its own generator.
- **What the family can express.** The conic-moment family covers **additive** drift error and
  **additive, homogeneous** noise. State-dependent or multiplicative noise, or noise correlated
  with the state, are outside it. And in the linear parameterization the matrix perturbation is
  example-specific: `alphas['A']` touches only the superdiagonal of `A` and `alphas['B']` only the
  single entry `B[0, -1]`. That is the structure of the paper's own example; it does not describe
  arbitrary uncertainty in `A` and `B`.

## Where to go next

- **See it declared, step by step**: [Getting Started](getting-started.md) — and the pitfalls that
  come with the declaration
- **See the guarantee behave**: [Examples](examples.md), or the simulated-ambiguity story for the
  two meanings of "worst"
- **The other side of the question**: [Predictability](predictability.md) — the ceiling, which
  needs no predictor at all
- **Widen what the family can express**: [the Developer Guide](extending.md)
