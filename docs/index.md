# ppsds

**Distributionally robust probabilistic prediction for stochastic dynamical systems.**

`ppsds` turns "predict the next state" from a point estimate into a predictive distribution
**with a worst-case guarantee**: as long as the true system lies inside the ambiguity set you
declared, the expected score of the prediction cannot fall below a stated lower bound.

This is the reference implementation accompanying T. Xu and J. He, *Distributionally Robust
Probabilistic Prediction for Stochastic Dynamical Systems*, IEEE Transactions on Automatic
Control.

```
pip install ppsds
```

## Start here

| | |
|---|---|
| **[Getting Started](getting-started.md)** | install it, run something, put it on your own system, and the pitfalls |
| **[Concepts](concepts.md)** | what DRPP optimizes, what the worst case ranges over, what the guarantee costs |
| **[Examples](examples.md)** | every example in the repository, with the command and the real output |
| **[API](api.md)** | generated from the docstrings |
| **[Citation](citation.md)** | how to cite the paper and the software |

## The short version

An ordinary predictor trusts its own model. That is hard to beat while the system behaves as
expected, and it is **overconfident** the moment it does not: the model drifts, the noise has a
different covariance, and the prediction is quietly too narrow.

DRPP changes the objective — instead of the nominal model, it optimizes for the worst system
allowed by the set you declared:

```
    maximize over predictors p_hat,
    minimize over distributions P in the ambiguity set,
        E_P[ score(p_hat) ]
```

The price is a little average performance. What it buys is a floor that holds in the worst case.

**The part that surprises people:** a worse model does not by itself make DRPP win. If your
declared covariance came from your own residuals, the ordinary predictor is already calibrated to
the errors it will meet. DRPP pays off when the declaration is *optimistic* — and too little
identification data is the most common reason for that. [Concepts](concepts.md) explains why, and
the [examples](examples.md) measure both regimes.

## Two ways to use it

- **You already have a gymnasium environment.** Declare a nominal model and an ambiguity set, and
  let a predictor run. This is the main path, and it does not require the environment layer at
  all.
- **You want to simulate your own ambiguity** — to build a benchmark, or to reproduce the paper.
  Then you also use the environment layer, which supplies the ambiguity set and a choice of
  dynamics modes, including an adversarial one.

[Getting Started](getting-started.md) covers the first; [Concepts](concepts.md) and
[Examples](examples.md) cover both.
