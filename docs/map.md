# The map

The sidebar is the shelf order — one reasonable route through the material, arranged so that each
page makes sense after the one above it. This page is the graph instead, for reading in whatever
order your own question suggests.

## The four stories

Each is a self-contained piece of the repository under
[`examples/`](https://github.com/Xuzerken/ppsds/tree/main/examples), with its own `README.md`, and
each prints the numbers it talks about.

```text
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  understand_ppsds_by_pmf_betting              the front door: no install     │
   │  ─────────────────────────────────────────────────────────────────────────   │
   │  act 1  play the game            →  what a probabilistic prediction IS       │
   │  act 2  same move, 10^3 traj     →  how close to the ceiling you got         │
   │  act 3  export your own hand     →  ppsds scores a prediction YOU supplied   │
   └───────────────────────────────────┬──────────────────────────────────────────┘
                                       │  you now know what the package is about
              ┌────────────────────────┴────────────────────────┐
              ▼                                                 ▼
   ┌──────────────────────────────┐              ┌──────────────────────────────┐
   │ understand_predictability    │              │ apply_drpp_to_your_own_env   │
   │ ──────────────────────────── │              │ ──────────────────────────── │
   │ the CEILING: the best any    │              │ the FLOOR, on a system of    │
   │ prediction could do on a     │              │ your own: declare a model,   │
   │ system you know exactly      │              │ declare a set, run           │
   │                              │              │                              │
   │ no predictor involved        │              │ act 2: "my model is bad" is  │
   │ → cheap, it never optimizes  │              │   NOT a reason it wins       │
   └──────────────┬───────────────┘              └──────────────┬───────────────┘
                  │                                             │
                  │            ┌────────────────────────────────┘
                  ▼            ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  understand_drpp_on_simulated_ambiguity      the semantics, nothing hidden   │
   │  ─────────────────────────────────────────────────────────────────────────   │
   │  declare an ambiguity set, grow a system out of it, watch three predictors   │
   │  face two meanings of "worst" (shared sample vs chosen against each one)     │
   └───────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
   ┌──────────────────────────────────────────────────────────────────────────────┐
   │  Developer Guide + Troubleshooting        write a family of your own         │
   │  ─────────────────────────────────────────────────────────────────────────   │
   │  a new ambiguity-set family = a new module at the same level in four places  │
   └──────────────────────────────────────────────────────────────────────────────┘
```

Two other routes exist and they cross the one above:

- **You have an environment and want to get on with it.** Skip straight to
  [Getting Started](getting-started.md) — the story above is what you read when the four-step
  declaration needs justifying.
- **You have read one of the papers.** Two entries, depending on which:
  [Predictability](predictability.md) for the *Automatica* paper on the ceiling, and the
  simulated-ambiguity story for the *TAC* paper on DRPP. Both meet the Developer Guide.

## Which part of the package does what

The stories use the same handful of subpackages in different combinations. This is the whole
public surface, one line each.

| Subpackage | It answers | You meet it in |
|---|---|---|
| `ppsds.ambiguity` | *What am I willing to admit about the system?* — the belief, written as a set | every story that names a DRPP |
| `ppsds.probabilistic_predictor` | *A prediction is a distribution, not a point.* Kinds sit side by side: `drpp/` derives one from an ambiguity set, `pmf/` takes one you supply | all four |
| `ppsds.scoring_rule` | *How well did it do?* — one number. The `epsilon`-neighbourhood score is what lets a **pmf** be scored at all | all four |
| `ppsds.predictability` | *How well could anything have done?* — the ceiling, which needs no predictor | predictability, and the betting game's act 2 |
| `ppsds.dynamics` | *The system itself* — known dynamics, mechanical models, and identifying a nominal model from rollouts | the two stories that use a real environment |
| `ppsds.environment` | *Run it as an environment* — for a system you declared an ambiguity set for | the simulated-ambiguity story |
| `ppsds.numerics` | Small numerical tools, and the weighted-atom pmf | the betting game |
| `ppsds.experiment`, `ppsds.visualization` | Running many runs reproducibly, and drawing the result (extra `pip install "ppsds[experiments]"`) | the batch halves |

## The one structural fact worth carrying away

```text
   ambiguity/                     additive_drift/conic_moment.py
   probabilistic_predictor/drpp/  additive_drift/conic_moment.py
   environment/dynamics_mode/adversarial/
                                  additive_drift/conic_moment.py
   experiment/drpp/               additive_drift/conic_moment.py
```

**Each family of ambiguity sets yields its own DRPP**, so a new family is a new module at the same
level in all four places — not a new branch inside the existing files. The path states the scope
layer by layer: *assumption* (additive drift) then *family* (conic moment). That is what
[the Developer Guide](extending.md) walks through.

## Where to go next

- **Start at the front door**: [play the betting game](understand_ppsds_by_pmf_betting/index.md)
- **Or skip to your case**: [Getting Started](getting-started.md) if you have an environment
- **The full list, with real output**: [Examples](examples.md)
- **What the guarantee costs**: [Concepts](concepts.md)
