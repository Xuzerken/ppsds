# …then measure it properly

The [game](index.md) settles one hand at a time. This page is the same thing with the training
wheels off: the same tables, the same payout rule, but thousands of trajectories and a comparison
against the one number the game can only hint at.

The code lives in
[`examples/understand_ppsds_by_pmf_betting/experiment/`](https://github.com/Xuzerken/ppsds/tree/main/examples/understand_ppsds_by_pmf_betting/experiment).

## The same move, over thousands of trajectories

```bash
python examples/understand_ppsds_by_pmf_betting/experiment/quickstart.py      # about a second
```

Three hands — an even spread, an aimed one, and a **safe layout** — scored against the
**predictability upper bound**: the best any bettor could possibly do on that table at that radius.

The lesson the table makes visible: **concentrating does not automatically help.** An aimed hand
wins big when it is right and scores `-inf` when it is wrong; the safe layout trades peak payout
for a floor. The gap-to-the-bound column is where that trade-off becomes a number.

## Bring your own hand back

On the game page, **Copy this round** and **Export this run** write out what you bet — the table,
the radius, and the hand for each round. Feed that file back here:

```bash
python examples/understand_ppsds_by_pmf_betting/experiment/run.py --bets my_run.json
```

`run.py` settles every round **again, for itself**, over as many trajectories as you like. It is
not replaying the page's arithmetic; it is recomputing it. That is what makes the agreement between
the two a check rather than a coincidence, and the repository's test suite asserts it round by
round.

Other entry points:

```bash
python examples/understand_ppsds_by_pmf_betting/experiment/run.py --list                    # the tables
python examples/understand_ppsds_by_pmf_betting/experiment/run.py --preset rotation --eps 0.25
python examples/understand_ppsds_by_pmf_betting/experiment/run.py --covering --preset shear --eps 0.5
python examples/understand_ppsds_by_pmf_betting/experiment/run.py --bets my_bets.json --n_traj 10000
```

| flag | meaning |
|---|---|
| `--preset` | which table |
| `--eps` | the payout radius (defaults to the middle of that table's range) |
| `--covering` | place the safe layout instead of a hand |
| `--bets` | read a hand or a whole run exported from the page (this overrides `--preset` / `--eps`) |
| `--n_traj`, `--n_step`, `--seed` | how much to sample, and reproducibly |
| `--list` | print the available tables and exit |

## A pmf is a legitimate prediction

Worth saying plainly, because it is the reason this story exists: the stacks you place **are** a
probabilistic prediction — a weighted set of atoms, which is what `AtomPmf` is. The package does
not only score Gaussians it produced itself; it will score a prediction *you* supplied, by the
same rule, in the same environment. That is what makes the game and the experiment the same thing
rather than two demos.

## Where to go next

- **What the ceiling is, derived rather than played**: [Predictability](../predictability.md)
- **The same idea on a system of your own**: [Getting Started](../getting-started.md)
- **Every example, with its real output**: [Examples](../examples.md)
- **How the page and the experiment are kept in agreement**: [Concepts](../concepts.md)
