# Understand DRPP on a simulated ambiguity

> **Declare an ambiguity set, let the system grow out of it, then watch three predictors face two kinds of truth.**

The [previous story](../apply_drpp_to_your_own_env/) puts a predictor on a system you already have.
This one asks the reverse question: *what if I write down the uncertainty first?* You declare an
ambiguity set by hand, generate a stochastic dynamical system **from it**, and let the predictors
predict. Nothing is hidden, so you can see exactly what each predictor is being asked to do.

| | |
|---|---|
| **Who this is for** | You want the central semantics of DRPP, not just the API |
| **Prerequisites** | The idea of an ambiguity set; nothing from the papers |
| **Runs on** | `pip install "ppsds[experiments]"` for the batch runner; the teaching example needs only the core |

---

## Act 1 — The same set, two meanings of "worst"

```bash
python examples/understand_drpp_on_simulated_ambiguity/quickstart.py      # a few seconds
```

Single process, core API only: no cache, no command line, no model registry. It declares an
ambiguity set, samples rollouts from it, and scores three predictors under **two semantics**:

| | `random` | `worst` |
|---|---|---|
| Which system? | one shared sample from the set | a system chosen **against each predictor** |
| What it measures | average performance | the guaranteed floor |

The script also prints the two theoretical bounds for the same set, so the numbers on screen have
something to sit between. If you read one file in this repository to understand DRPP, read this one.

> The run time is dominated by `EigDrpp`, which solves a cone program at every step — that is the
> price of the guarantee, and it is worth watching happen once.

## Act 2 — Seven models, cached, in parallel

```bash
cd examples/understand_drpp_on_simulated_ambiguity/batch_experiment
python run.py -n 8 -s 3 -p 100 -m final_round -c none -t 100          # random, slowly-varying
python run.py -n 8 -s 3 -p 1   -m final_round -c none -t 100          # random, i.i.d.
python run.py -n 8 -s 3        -m NLink       -c none -t 100 -e worst # adversarial, 6-link arm
```

`run.sh` (Linux/WSL) and `run.ps1` (Windows) run all six combinations for you.

**This is a heavy tool, not an advanced version of Act 1**, and the difference is not an accident.
Three things cannot be removed from a batch runner and are all absent from the teaching example: a
table of seven model definitions, a cache whose shape is part of its contract, and a `fork`
requirement (the nonlinear ambiguity set holds lambdas, so it will not pickle). Making Act 2 look
like Act 1 would mean deleting one of those, and each of them is load-bearing. Take Act 1 to learn,
Act 2 to produce numbers.

Two flags worth knowing before you launch anything big:

- **`-c regulation`** (closed-loop LQR) **only works on the five linear models**. Non-linear models
  exit with an error, which is why the scripts default to `-c none`.
- **The cache is validated, not trusted.** Before skipping a run it checks a manifest (source
  fingerprint, parameter snapshot, per-file hashes) and the shapes of what is on disk; anything that
  does not match is recomputed, with a printed reason. `-f` forces a full recompute.

Results land under `data/gym/...`, which is **not** under version control.

---

## Where to go next

- **The concept behind it**: [Concepts](https://xuzerken.github.io/ppsds/concepts/) — especially
  *Two bounds bracket the answer*, and the warning there that it is **not** the same pair of bounds
  as predictability-versus-DRPP
- **The other side of the story**: `../understand_predictability/` — the ceiling, which needs no
  predictor at all
- **Write a family of your own**: [Extending ppsds](https://xuzerken.github.io/ppsds/extending/)
- **When the numbers look wrong**: [Troubleshooting](https://xuzerken.github.io/ppsds/extending-faq/)

## What this story uses

| Subpackage | For |
|---|---|
| `ppsds.ambiguity` | the ambiguity set — the thing you declare |
| `ppsds.environment` | building a gymnasium environment from an ambiguity set (scene 2 only) |
| `ppsds.environment.dynamics_mode` | `PreSampledDynamics` and the adversarial `EigvecFixedAdversarialDynamics` |
| `ppsds.probabilistic_predictor.drpp` | `NominalPredictor`, `NoiseDrpp`, `EigDrpp` |
| `ppsds.experiment` | the batch driver and its validated cache (Act 2) |
| `ppsds.visualization` | the ellipse and error-bar figures (Act 2) |
