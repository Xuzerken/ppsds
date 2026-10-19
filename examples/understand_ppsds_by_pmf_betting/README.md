# Understand ppsds by betting on a pmf

> **Be the predictor: place bets on where the system goes next, and find out what they were worth.**

This example is the **no-install front door**: a page you can play. Its rigorous half — the same
move over thousands of trajectories, scored against the ceiling — lives next to the other batch
experiments, in [`experiments/pmf_betting/`](../../experiments/pmf_betting/). The game is the fast
way in; the experiment is where the numbers become trustworthy.

| | |
|---|---|
| **Who this is for** | Anyone — the first act needs no background at all |
| **Prerequisites** | *Step 1: none.* Step 2 onward: knowing what a distribution is |
| **Runs on** | Nothing for the game; `pip install ppsds` for the experiment |

---

## Step 1 — Thirty seconds on a table

**Play it: <https://xuzerken.github.io/ppsds/examples/understand_ppsds_by_pmf_betting/>**

A puck moves around a table. You put stacks where you think it will land next, the dealer drops a
ball, and every stack within `epsilon` of the landing pays its share of your bet.

Three things are on screen, and they are the three things this package is made of:

| on screen | in the package |
|---|---|
| the moving puck | the **SDS** — a drift plus a known noise law |
| the stacks you place | the **probabilistic predictor** — a pmf over where the puck will be |
| how they pay | the **scoring rule** — the `epsilon`-logarithm score |

That is the whole idea with the abstraction removed. Nothing is downloaded, nothing is uploaded, and
the page carries no second copy of any system: its data is exported from
[`experiments/pmf_betting/presets.py`](../../experiments/pmf_betting/presets.py).

## Step 2 — The same move, training wheels off

```bash
python experiments/pmf_betting/quickstart.py      # about a second
```

Three hands — an even spread, an aimed one, and a **safe layout** — over thousands of trajectories,
against the one number the game can only hint at: the **predictability upper bound**, i.e. the best
any bettor could possibly do.

The lesson the table makes visible: **concentrating does not automatically help.** An aimed hand
wins big when it is right and scores `-inf` when it is wrong; the safe layout trades peak payout for
a floor. The gap to the bound column is where that trade-off shows up as a number.

## Step 3 — Bring your own hand back

```bash
# on the page: "Copy this round" or "Export this run", then
python experiments/pmf_betting/run.py --bets my_run.json
```

The exported file is a plain record of what you bet — the table, the radius, and the hand for each
round. `run.py` settles every round **again, for itself**, over as many trajectories as you like.
It is not replaying the page's arithmetic; it is recomputing it, which is what makes the agreement
between the two a check rather than a coincidence.

Other entry points:

```bash
python experiments/pmf_betting/run.py --list                                        # the tables
python experiments/pmf_betting/run.py --preset rotation --eps 0.25
python experiments/pmf_betting/run.py --covering --preset shear --eps 0.5           # the safe hand
python experiments/pmf_betting/run.py --bets my_bets.json --n_traj 10000
```

---

## Where to go next

- **What the ceiling is**: `../understand_predictability/` — the same number, derived rather than
  played
- **Why a pmf is a legitimate prediction at all**:
  [Research](https://xuzerken.github.io/ppsds/research/)
- **Put a predictor on a system of your own**: `../apply_drpp_to_your_own_env/`
- **The full CLI**: [the betting experiment](https://xuzerken.github.io/ppsds/experiments/pmf_betting/)

## What this example uses

| Subpackage | For |
|---|---|
| `ppsds.numerics.atom_pmf` | `AtomPmf` — a weighted set of atoms, i.e. a hand of stacks |
| `ppsds.probabilistic_predictor.pmf` | `DriftShiftedAtomPmfPredictor` — carrying a pmf forward |
| `ppsds.scoring_rule` | `EpsilonLogarithmScore` — how a pmf gets scored at all |
| `ppsds.predictability` | the upper bound the experiment measures against |

## For maintainers

`experiments/pmf_betting/presets.py` is the **single source of truth** for the tables, shared by
the page and by every experiment. Change it and you must re-run

```bash
python experiments/pmf_betting/export_game_assets.py
```

or the committed page data will disagree with the code, and the contract tests will say so. A
preset's `key` is frozen: it is what a saved bet file records, so renaming one invalidates bets
people have already exported.

The README animation is a **recording of the page**, not a drawing of it, and its tooling lives in
`tools/` rather than here — it needs a browser, so it cannot run in a user's install or in CI. See
`tools/make_demo_gif.py`.
