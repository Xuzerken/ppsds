# Put the predictor on your own system

> **Put the predictor on your own system — and find out when it pays for itself and when it charges you.**

You already have something you can `step()`: a simulator, or a gymnasium environment you wrote. This
example is about adding probabilistic prediction to it, and about the two ways it goes wrong.

| | |
|---|---|
| **Who this is for** | You have an environment and you intend to use the result, not just read about it |
| **Prerequisites** | A gymnasium environment; the idea of a nominal model. No paper required |
| **Runs on** | `pip install ppsds` alone — nothing here needs the environment layer |

---

## Four steps, and the order matters

```bash
python examples/apply_drpp_to_your_own_env/quickstart.py      # about 1 second
```

`ppsds` never asks your environment for anything. The whole integration is a **declaration**, made
in a fixed order: **nominal model → family of ambiguity sets → that family's parameters →
predictors**. The order matters because a family describes uncertainty in its own language, so
picking the family decides what the next step must supply.

```
predictor     mean log score   note
----------------------------------------------------------------
nominal               -2.867   trusts the nominal model
Eig-DRPP              -3.366   made robust to the worst case
```

**How to read it:** `nominal` scores *higher* here, and that is the expected outcome rather than a
failure. DRPP optimizes the worst case, so on one ordinary rollout its conservatism is a cost with
nothing yet collected. Its return arrives in the second act.

## "My model is bad" is not by itself a reason DRPP wins

```bash
python examples/apply_drpp_to_your_own_env/with_standard_gym_env.py      # about 10 seconds
```

`Acrobot-v1` is six-dimensional, underactuated and chaotic, and nobody wrote it for this package —
so the nominal model and the ambiguity set have to be **identified from data**. The script prints
how bad the fit actually is (a residual of about 74%), then answers the question the first script left open:
does DRPP win now?

**It does not.** That is the point of the act: *"the model is bad" is by itself not enough to make
DRPP win.* What decides it is whether the ambiguity set you declared **contains** the true system.
Declare optimistically and DRPP helps you; declare conservatively and it charges you a premium.
The script's last section runs that comparison over two regimes of identification data.

## The covariance you computed may be unusable

```bash
python examples/apply_drpp_to_your_own_env/pendulum_singular_covariance.py      # about 5 seconds
```

`Pendulum-v1` produces a **structurally singular** residual covariance: the state is
`(cos θ, sin θ, θ̇)`, so two of its coordinates live on a circle. A log score needs a density, and a
degenerate distribution has none — so the score comes back `-inf`, not as a bug but as arithmetic.

The fix is to regularize, and the act's real content is that **the floor you choose is itself a
modelling claim**: the script sweeps it and shows the score is dominated by the width you declared,
not by prediction quality.

---

## Where to go next

- **Deeper into the guarantee**: [Concepts](https://xuzerken.github.io/ppsds/concepts/) — what the
  worst case may touch, and what the guarantee costs
- **Understand what you just declared**: `../understand_drpp_on_simulated_ambiguity/` — build an
  ambiguity set by hand and generate a system from it, so the set stops being a black box
- **Add a family of your own**: [Extending ppsds](https://xuzerken.github.io/ppsds/extending/)
- **The pitfalls in full**: [Troubleshooting](https://xuzerken.github.io/ppsds/extending-faq/)

## What this example uses

| Subpackage | For |
|---|---|
| `ppsds.ambiguity` | declaring the nominal model and the ambiguity set |
| `ppsds.probabilistic_predictor.drpp` | `NominalPredictor`, `NoiseDrpp`, `EigDrpp` |
| `ppsds.scoring_rule` | `LogarithmScore` |
| `ppsds.dynamics.identification` | fitting a nominal model from rollouts (Acts 2 and 3) |
