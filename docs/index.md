# ppsds

**Probabilistic prediction for stochastic dynamical systems.**

`ppsds` predicts the next state of a stochastic dynamical system as a **distribution** rather
than a point — and it ships predictors that stay honest when your model of the system is wrong.

Predictors come in families, and the package is built to hold more than one. This release
implements the **distributionally robust** family (**DRPP**) over **conic-moment** ambiguity
sets, alongside the ordinary nominal and noise-robust baselines that it can be measured against.

```
pip install ppsds
```

## Start here

| | |
|---|---|
| **[Getting Started](getting-started.md)** | install it, run something, put it on your own system, and the pitfalls |
| **[Concepts](concepts.md)** | what is optimized, what the worst case may touch, what the guarantee costs |
| **[Examples](examples.md)** | every example, with the command and the real output |
| **[API](api.md)** | generated from the docstrings |
| **[Citation](citation.md)** | the software citation, and the references it implements |

If you only read one page and you intend to rely on the guarantee, make it
[Concepts](concepts.md): the guarantee is conditional, and the condition is a declaration you
write, not something the library can check for you.

## What "distributionally robust" changes

An ordinary predictor trusts the model it was given. That is hard to beat while the system
behaves as expected, and it is **overconfident** the moment it does not: the dynamics drift, the
noise has a different covariance, and the prediction is quietly too narrow.

DRPP changes the objective. Instead of the nominal model, it optimizes for the worst system the
ambiguity set allows — the predictive distribution whose *worst-case* expected score is as high
as it can be:

```
    max over the predictive distribution p̂   of   min over P in the ambiguity set   of   E_P[ score(p̂) ]
```

The price is a little average performance — a more conservative predictive covariance. What it
buys is a floor that holds in the worst case, and that floor is what the
[Concepts](concepts.md) page is about, including the part that surprises people: **a predictor
does not become better by having a worse model**, and the declaration matters more than the
model's accuracy.

## Two ways to use it

- **You already have a gymnasium environment.** Declare a nominal model and an ambiguity set, and
  let a predictor run. This is the main path, and it does not require the environment layer at
  all.
- **You want to simulate your own ambiguity** — to build a benchmark, or to reproduce a paper.
  Then you also use the environment layer, which supplies the ambiguity set and a choice of
  dynamics modes, including an adversarial one.

[Getting Started](getting-started.md) covers the first; [Concepts](concepts.md) and
[Examples](examples.md) cover both.
