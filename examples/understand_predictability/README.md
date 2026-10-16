# Understand predictability

> **How well could anything predict this system? The ceiling, and the figure from the paper.**

Everything else in this repository is about building a predictor. This story is about the number
such a predictor is trying to reach — and it needs **no predictor at all**. Given only the system,
the epsilon-logarithm score has a best possible expected value, and that value is the ceiling.

| | |
|---|---|
| **Who this is for** | You want to reproduce or check the paper's figure; or you want to know what "good" means before you measure anything |
| **Prerequisites** | The idea of a scoring rule. No paper required — the golden values are printed |
| **Runs on** | `pip install ppsds` alone |

---

## Act 1 — Six numbers you can check by hand

```bash
python examples/understand_predictability/quickstart.py      # about 2 seconds
```

Shear plus uniform noise, small enough to read. The script prints the differential entropy of the
true law, the cross-entropy against a reference predictor, and then a table:

| column | what it is |
|---|---|
| `predictability` | the ceiling: the best expected score any predictive law could attain |
| `approximation` | the paper's `O(epsilon)` approximation to it |
| `measured` | the ceiling estimated by sampling, which is what you would have without the theory |
| `\|err\| / eps` | how fast the approximation closes in as `epsilon` grows |

It ends with a "how to read this" section, including the one discontinuity worth knowing: the
`epsilon = 0` case is a different formula and does not sit on the same curve.

## Act 2 — The paper's figure, at the paper's scale

```bash
python examples/understand_predictability/batch_experiment/run.py                  # 100,000 trajectories
python examples/understand_predictability/batch_experiment/run.py --n_traj 2000    # quick look
python examples/understand_predictability/batch_experiment/run.py --save-figure    # write the PDF
```

Same system and same predictor, at the resolution the paper uses, plus the three-panel figure. It
takes seconds rather than minutes because the per-step work is vectorized over trajectories; the
script says so, and says where the vectorization lives, so you can check the claim rather than take
it.

The runtime is worth noticing in itself: the whole point of this side of the package is that it is
**cheap**, because it never solves an optimization problem.

---

## Where to go next

- **The other side of the same coin**: `../understand_drpp_on_simulated_ambiguity/` — the floor,
  which is what you get when the system is *not* known exactly
- **Make sure you have the two bound-pairs straight**: [Concepts](https://xuzerken.github.io/ppsds/concepts/)
  § *Two bounds one level up, and they are not these two*
- **Score your own predictor against the ceiling**: [Predictability](https://xuzerken.github.io/ppsds/predictability/)
- **The same idea as a game**: `../understand_ppsds_by_pmf_betting/` — you place the bet, and the
  ceiling is the number no bettor can beat

## What this story uses

| Subpackage | For |
|---|---|
| `ppsds.dynamics.known_sds` | a system known exactly: a drift plus a named noise law |
| `ppsds.numerics.random_variables` | the noise law (`multivariate_uniform`) |
| `ppsds.predictability` | the ceiling, the approximation, entropy, and trajectory aggregation |
| `ppsds.scoring_rule` | `EpsilonLogarithmScore` |
| `ppsds.visualization` | the three-panel figure (Act 2) |
