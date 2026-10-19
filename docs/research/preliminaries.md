# Preliminaries

The two papers share a vocabulary. This page defines it once, without assuming that you have
opened either paper. It deliberately stops before the details that belong to one of them; those
live on the [ceiling](probabilistic-predictability.md) and [floor](drpp-conic.md) pages.

## A probabilistic prediction is a distribution

An ordinary predictor returns a point: the state it thinks comes next. A **probabilistic
predictor** returns a distribution over the next state instead, which is what lets you talk about
uncertainty rather than a single guess. A **scoring rule** turns that distribution and the state
that actually occurred into one number; higher is better.

The rule used throughout `ppsds` is the **`epsilon`-logarithm score**. Instead of the density at
the exact outcome, it takes the probability of a neighbourhood of radius `epsilon` around it:

$$
L_\epsilon(\hat p, x)
  = \log \int_{\{\,s \,:\, \lVert s - x \rVert_\infty \le \epsilon\,\}} \hat p(s)\,\mathrm{d}s .
$$

At `epsilon = 0` this is the ordinary logarithm score. For any `epsilon > 0` it is finite even
when the prediction is a **pmf** with a handful of atoms, which the ordinary score is not. The
score is discontinuous at zero: it carries a `d log(2 epsilon)` volume term, and the ordinary
score is recovered by subtracting that term, not by taking a limit.

Both sides of the library are stated in this score. The ceiling asks how large its expectation
can be; the floor asks what value a predictor can guarantee.

## Ambiguity sets and the maximin objective

You do not know the system exactly. What you can write down is an **ambiguity set**
$\mathcal{P}$: the systems you are willing to admit, built from a nominal model and the size of
the errors you are willing to allow. A predictor is then judged against the worst member of that
set:

$$
\max_{\hat p}\ \min_{P \in \mathcal{P}}\
  \mathbb{E}_P\!\left[\log \hat p(x_{k+1} \mid x_k, u_k)\right] .
$$

Read it inside out. The inner minimisation is an adversary that picks a system out of the set you
declared. The outer maximisation is your predictor, which may be designed knowing the set but not
which member will be chosen. The quantity scored is the predictive density at the state that
actually occurs.

**The worst case ranges over your declaration, and nothing else.** The adversary cannot invent a
system you did not allow. If the true system lies outside the set, the guarantee says nothing
about it. The package never inspects your environment: the predictor takes its model and its
radii from the set you declared, and the environment, when there is one, is only stepped. So the
declaration is a statement about what you believe, and the quality of the guarantee is exactly the
quality of that statement.

In the environment layer the adversary is a **best response**: the worst system is chosen *after*
the predictor emits its distribution, using the predictor's own predicted covariance. That
ordering is the paper's semantics, not an implementation accident.

## Two pairs of bounds

Two different pairs of "upper" and "lower" bounds appear in this material, and they must not be
read across the gap.

**Inside one ambiguity set.** `value_upper_bound_true_transition()` is the score achievable when
the true transition is known; no predictor in that set can beat it.
`value_lower_bound_eigvec_fixed()` is the score Eig-DRPP guarantees under its fixed-eigenvector
relaxation. The gap between them is the price of making the problem solvable. These belong to the
[floor](drpp-conic.md) and are derived there.

**One level up, between two kinds of prediction.** **Probabilistic predictability** asks how well
*any* predictor could do on a system known **exactly**; it is an upper bound and a property of the
system. **DRPP** asks what a predictor can **guarantee** for a system known only through an
ambiguity set; it is a lower bound and a property of the predictor. This pair is the frame of the
whole library, and each half has its own paper page.

The two pairs are different even though both use the words "upper" and "lower". Inside one set,
the upper bound is a case of the predictability idea applied to that set; the lower bound is the
DRPP guarantee. The gap in the second pair is model uncertainty; the gap in the first is the
relaxation.

## The two ways to use the package

| | Your own environment | Simulated ambiguity |
|---|---|---|
| You have | an existing Gymnasium environment | only the paper, and want to reproduce it |
| You supply | nominal model + ambiguity set (agent side) | additionally the ambiguity set and the dynamics generator (environment side) |
| You use | `ppsds.probabilistic_predictor` + `ppsds.scoring_rule` | plus `ppsds.environment` |

`ppsds.environment` exists **for the second column only**. If you are putting a predictor on a
system you already have, you never touch it: you step your own environment and hand the observed
states to the predictor.

## Honest limits

- **The guarantee is conditional on the declaration.** It says: for every system in the declared
  set, the expected score is at least this much. If the true system falls outside the set, the
  statement is vacuous.
- **A set that is too wide costs you average performance.** Admitting systems that never occur
  forces a wider predictive covariance than the situation needs, and the score rewards sharpness.
- **A set that is too narrow voids the guarantee.** In simulation it shows up as the guarantee
  apparently failing. That is a mis-declared set, not a broken algorithm: look first at whether
  the declaration covers the truth.
- **The set must contain its own generator.** If you write a sampler for simulated dynamics, its
  samples must land inside the declared set, or your ground truth is outside it by construction.
- **The conic-moment family covers additive drift error and additive, homogeneous noise.** The
  current family does not express state-dependent or multiplicative noise. The paper-specific
  parameterization and its example-specific structure are on the [conic DRPP
  page](drpp-conic.md).

## Next

- **The ceiling**: [probabilistic predictability paper](probabilistic-predictability.md)
- **The floor**: [conic DRPP paper](drpp-conic.md)
- **Put it on your system**: [Getting Started](../tutorials/getting-started.md)
- **See the guarantee in one script**: [the simulated-ambiguity example](../examples/understand_drpp_on_simulated_ambiguity/index.md)
