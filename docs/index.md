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
[Concepts](concepts.md) is about what that costs and what it assumes.

That DRPP is the one for **conic-moment** ambiguity sets, on an **additive-drift** system. It is
the first of a template rather than a single method: each family of ambiguity sets yields its own
DRPP, and the `drpp` subpackage is built to hold more.

Zoom out one level. DRPP is one *kind* of **probabilistic predictor** (PP). The kinds sit side by
side in `ppsds.probabilistic_predictor`, one subpackage each, and `drpp` is the first of them. A
data-driven setting would add a **learning-based PP** there; a theoretical one would add the
**optimal PP** that attains **probabilistic predictability**, the best a prediction of a *known*
system can do. That upper bound, and DRPP's guaranteed lower bound for a system known only through
an ambiguity set, answer the same question from two sides.

```
pip install ppsds
```

## Start here

| | |
|---|---|
| **[Getting Started](getting-started.md)** | install it, run something, put it on your own system, and the pitfalls |
| **[Concepts](concepts.md)** | what is optimized, what the worst case may touch, what the guarantee costs |
| **[Examples](examples.md)** | every example, with the command and the real output |
| **[Developer Guide](extending.md)** | write your own family of ambiguity sets, and run an experiment on it |
| **[Troubleshooting](extending-faq.md)** | the traps, what they look like, and what to do |
| **[API](api.md)** | generated from the docstrings |
| **[Citation](citation.md)** | the software citation, and the references it implements |

If you only read one page and you intend to rely on the guarantee, make it
[Concepts](concepts.md): the guarantee is conditional, and the condition is a declaration you
write, not something the library can check for you.

## What "distributionally robust" changes

The objective, in one line: instead of trusting the nominal model, maximize the worst case over
the set.

```
    max over the predictive distribution p̂   of   min over P in the ambiguity set   of   E_P[ score(p̂) ]
```

The price is a little average performance: a more conservative predictive covariance. What it
buys is a floor that holds for every system in the set, and that floor is what the
[Concepts](concepts.md) page is about, including the part that surprises people: **a predictor
does not become better by having a worse model**, and the declaration matters more than the
model's accuracy. See also
[Two bounds bracket the answer](concepts.md#two-bounds-bracket-the-answer) for what that floor
costs, and [Extending](extending.md) for how another ambiguity set gets its own DRPP.

## Two ways to use it

- **You already have a gymnasium environment.** Declare a nominal model and an ambiguity set, and
  let a predictor run. This is the main path, and it does not require the environment layer at
  all.
- **You want to simulate your own ambiguity**, to build a benchmark or to reproduce a paper.
  Then you also use the environment layer, which supplies the ambiguity set and a choice of
  dynamics modes, including an adversarial one.

[Getting Started](getting-started.md) covers the first; [Concepts](concepts.md) and
[Examples](examples.md) cover both.
