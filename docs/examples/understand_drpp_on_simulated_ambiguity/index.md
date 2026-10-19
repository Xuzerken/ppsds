# What “worst case” actually means

> **Declare an ambiguity set, let the system grow out of it, then watch three predictors face two kinds of truth.**

| | |
|---|---|
| **Who it is for** | You want the central semantics of DRPP, not just the API |
| **Prerequisites** | The idea of an ambiguity set; nothing from the papers |
| **Runs on** | `pip install ppsds` — this page uses the core only |

## One ambiguity set, two meanings of “worst”

```bash
python examples/understand_drpp_on_simulated_ambiguity/quickstart.py
```

A few seconds, single process, **core API only**: no cache, no command line, no model table. It
declares a conic-moment ambiguity set by hand, then runs the same rollout loop under two semantics:

- **`random`** — one SDS is sampled from the set and shared by every predictor, so only the noise
  differs.
- **`worst`** — each predictor faces the SDS chosen specifically against it, a best response. This
  is the paper's central semantics.

Eig-DRPP dominates the runtime because it solves a cone program at every step; lower `N_TRAJ` or
`T` in the file to make it faster.

```
Mean logarithmic score (higher is better), 4 trajectories x 50 steps

predictor       random     worst
--------------------------------
Nominal         -4.726   -23.365
Noise-DRPP      -3.974   -10.186
Eig-DRPP        -4.566    -5.541

Theoretical bounds (per step): upper -4.048, lower -5.833
```

**What to conclude:** under `random` the three predictors sit close together: the nominal model is
roughly right, so robustness buys little and costs a little. Under `worst` they separate sharply:
`Nominal` is driven down to `-23.4` because it is attacked on purpose, while `Eig-DRPP` shrinks its
covariance along the worst direction and holds near its lower bound of `-5.833`. That gap is the
guarantee, measured: average performance given up under `random`, worst-case floor collected under
`worst`. `Noise-DRPP` sits in between, which is what “robust to noise only” buys.

The two bounds printed at the end are the **within-one-set** pair
(`value_upper_bound_true_transition` / `value_lower_bound_eigvec_fixed`). They are not the
“predictability ceiling versus DRPP floor” pair; [Research](../../research/index.md) keeps the two
straight.

## The same set, at production size

The teaching script is deliberately small. Reproducing the paper's sweeps needs seven models, a
validated cache, and multiple processes — that is a **separate experiment**, kept out of this
example so the example stays readable:

[DRPP on simulated ambiguity →](../../experiments/drpp.md)

## Next

- **The batch runner, its flags and its cache**: [the DRPP experiment](../../experiments/drpp.md)
- **What the bounds mean**: [Research](../../research/index.md)
- **The other side of the question**: [the predictability example](../understand_predictability/index.md)
- **Write a family of your own**: [A new family and its DRPP](../../tutorials/developer-guide.md)
