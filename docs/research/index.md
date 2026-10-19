# Research

`ppsds` answers one question from two sides: **how well can a system you only partly know be
predicted?** The **ceiling** is a property of a system you know exactly; the **floor** is what a
predictor can guarantee over a set of systems you are willing to admit. These are the papers the
package implements.

| Paper | Venue | Question | On this site |
|---|---|---|---|
| *Probabilistic Predictability of Stochastic Dynamical Systems* | Automatica | How well could **any** predictor do on a system known exactly? | [Paper](probabilistic-predictability.md) · [Experiment](../experiments/predictability.md) · [Example](../examples/understand_predictability/index.md) |
| *Distributionally Robust Probabilistic Prediction for Stochastic Dynamical Systems* | IEEE Transactions on Automatic Control | What can a predictor **guarantee** when the system is known only through an ambiguity set? | [Paper](drpp-conic.md) · [Experiment](../experiments/drpp.md) · [Example](../examples/understand_drpp_on_simulated_ambiguity/index.md) |

## Two things to take away

1. **A predictor does not become better by having a worse model.** What decides how well this works
   is the declaration you write, not how accurate your model is.
2. **The guarantee is conditional, and the condition is that declaration.** Nothing in the library
   can check it for you. `diagnose()` can tell you what your residuals look like; it cannot tell
   you whether the set you wrote down contains the system you are actually running.

## Before the papers

Both papers assume the same small vocabulary. In brief:

- A **probabilistic prediction** is a distribution over the next state, judged by a scoring rule.
- An **ambiguity set** is the set of systems you are willing to admit. The worst case ranges over
  that set and nothing else.
- There are **two different pairs of bounds**: one pair sits inside a single ambiguity set, the
  other compares a system known exactly against one known only through a set.
- The guarantee is **conditional on the declaration**. Every limit follows from that.

The full treatment, written for a reader who has not opened either paper, is
[Preliminaries](preliminaries.md). Each paper page then takes one side and goes all the way down.

## Next

- **The shared vocabulary**: [Preliminaries](preliminaries.md)
- **The ceiling**: [Probabilistic predictability paper](probabilistic-predictability.md)
- **The floor**: [Conic DRPP paper](drpp-conic.md)
- **Put either on your system**: [Getting Started](../tutorials/getting-started.md)
