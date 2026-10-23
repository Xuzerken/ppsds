# ppsds: probabilistic prediction for stochastic dynamical systems

**How probabilistically predictable is a system you only partly know?** `ppsds` answers from both sides: the **ceiling** any probabilistic prediction could reach on a system you know exactly, and a probabilistic predictor that comes with a **floor** over every system you are willing to admit.

**If that sounds abstract, that is expected.** The paragraph above is written for readers who already know what a *stochastic dynamical system* is and what *probabilistic prediction* means. Many people have never met either term, and some may already be working with such a system without knowing it. You cannot miss a capability you have never had a name for. The **one-minute game** below replaces the vocabulary with something you can play: place a few bets, watch the system move, and see what a probabilistic prediction is before reading a definition of it.

<p align="center">
<a href="https://xuzerken.github.io/ppsds/examples/understand_ppsds_by_pmf_betting/"><img src="https://raw.githubusercontent.com/Xuzerken/ppsds/main/docs/assets/media/game-demo.gif" width="720" alt="The game's six-step guided tour, looping: prediction turned on, the gold noise support covered with stacks, one stack raised, the deal, the next round, and then Auto deal playing three rounds on its own"></a><br>
<strong><a href="https://xuzerken.github.io/ppsds/assets/media/game-demo.mp4">▶ Watch the 45-second demo</a></strong> · <strong><a href="https://xuzerken.github.io/ppsds/examples/understand_ppsds_by_pmf_betting/">Play the game →</a></strong>
</p>

| on screen | in `ppsds` |
|---|---|
| the moving puck | the **system** — a drift plus a known noise law |
| the stacks you place | the **probabilistic predictor** — a probability mass function (pmf) over where the puck will be |
| how they pay | the **scoring rule** — the `epsilon`-logarithm score |

A puck moves around a table. You bet on where it will land next by placing stacks. Then the dealer drops a ball: every stack within `epsilon` of the landing pays its share of your bet. That is the whole idea with the abstraction removed; the table above maps the three things on screen onto the three things the package is made of. Thirty seconds on the table explains what the package is better than a paragraph can — no install, no server, nothing kept.

## Install

```
pip install ppsds
```

`0.0.3-alpha` is a pre-release: pin it with `pip install ppsds==0.0.3a0`, or pass `--pre`. Plain
`pip install ppsds` selects it while no stable version exists; once `0.1.0` is out, plain install
selects the stable release.

Core dependencies are `numpy`, `scipy`, `cvxpy` and `gymnasium`; Python 3.10 or newer. Only the experiment and plotting periphery needs more:

```
pip install "ppsds[experiments]"     # adds matplotlib, pandas, seaborn, tqdm
```

Each example and experiment directory also carries its own `environment.yml` if you would rather create a conda environment for exactly one page.

<details>
<summary>The game rules, and why they are the package</summary>

Bet by placing stacks; one stack is one atom of a pmf and its height is the weight. The dealer drops a ball, every stack within `epsilon` of the landing pays its share, and your round is scored by the `epsilon`-logarithm score — the log of the probability you gave the landing's neighbourhood.

**Cover the table** builds the safe layout: it tiles the noise support, so it pays the same share every round and cannot lose — at the price of never winning big. A run ends when a round pays nothing, or when you ask it to.

Playing it is **free**, needs **no install**, and **keeps nothing**: the page has no server behind it. When you want the same thing done rigorously, the bets you place export to a file that [the betting experiment](https://xuzerken.github.io/ppsds/experiments/pmf_betting/) reads back and recomputes round by round.

</details>

## Get started

Rows are **how far you want to go**. Columns are **what you start from**, ordered from nothing to data without a model.

| **From →** | **A · Just curious**<br>Nothing yet. | **B · Build it yourself**<br>No environment. | **C · Your system**<br>You can step it. | **D · Data only**<br>No model or env. |
|---|---|---|---|---|
| **L0 · See** | <img src="https://raw.githubusercontent.com/Xuzerken/ppsds/main/docs/assets/media/game-demo-poster.png" width="120" alt="the betting game demo"><br>[Just curious](https://xuzerken.github.io/ppsds/examples/understand_ppsds_by_pmf_betting/): see what a probabilistic prediction is in one minute. | — | — | — |
| **L1 · Use** | — | <img src="https://raw.githubusercontent.com/Xuzerken/ppsds/main/docs/assets/papers.png" width="120" alt="The two papers' figures at one scale"><br>[Reproduce a paper](https://xuzerken.github.io/ppsds/research/): build the system from the paper and check its figures. | <img src="https://raw.githubusercontent.com/Xuzerken/ppsds/main/docs/assets/acrobot_simulation.gif" width="120" alt="Acrobot in a rendered simulation"><br>[Use it on your system](https://xuzerken.github.io/ppsds/examples/apply_drpp_to_your_own_env/): put a predictor on a Gymnasium environment you already have. | [Only data](https://xuzerken.github.io/ppsds/research/preliminaries/#honest-limits): you need a system you can step; read the limits first. |
| **L2 · Extend** | — | [Add a predictor kind](https://xuzerken.github.io/ppsds/api/): try a different prediction law in a simulation you build. | <img src="https://raw.githubusercontent.com/Xuzerken/ppsds/main/docs/assets/drpp_ellipses.png" width="120" alt="Three 90% predictive ellipses"><br>[Add a family](https://xuzerken.github.io/ppsds/tutorials/developer-guide/): teach the library the uncertainty language your system needs. | — |
| **L3 · Research** | — | [Work on the ceiling](https://xuzerken.github.io/ppsds/research/probabilistic-predictability/): scoring rules, predictability, and what no predictor can beat. | [Do research on the floor](https://xuzerken.github.io/ppsds/research/preliminaries/): change the assumptions, not just the implementation. | — |
| **L4 · Contribute** | [Found a problem while playing](https://xuzerken.github.io/ppsds/contributing/reporting/): open an issue; the game is a real user of the library. | [Contribute what you built](https://xuzerken.github.io/ppsds/contributing/pull-requests/): file an L1 issue, or send an L2/L3 result (including a benchmark) as a pull request. | [Contribute what you built](https://xuzerken.github.io/ppsds/contributing/pull-requests/): file an L1 issue, or send your L2/L3 results as a pull request. | [Discuss a research collaboration](https://xuzerken.github.io/ppsds/contributing/research/): this column is future research; L0–L3 are blank here. |

## Recommended routes

The table above sells one story at a time. These are the **routes** — paths that cross the site's four sections, for a reader who knows what they are trying to get done.

**Need the panorama?** The map below shows every route at a glance; [the map page](https://xuzerken.github.io/ppsds/tutorials/map/) explains what each one means and how the sections connect.

<img src="https://raw.githubusercontent.com/Xuzerken/ppsds/main/docs/assets/map.png" width="760" alt="A map of the examples, experiments, papers and tutorials">

**1 · Play first.** [The game](https://xuzerken.github.io/ppsds/examples/understand_ppsds_by_pmf_betting/) needs nothing installed. Thirty seconds on the table, and you know what a probabilistic prediction is.

**2 · Measure the game properly.** [The game](https://xuzerken.github.io/ppsds/examples/understand_ppsds_by_pmf_betting/) → [the betting experiment](https://xuzerken.github.io/ppsds/experiments/pmf_betting/) → [Preliminaries](https://xuzerken.github.io/ppsds/research/preliminaries/). The same move over thousands of trajectories, against the ceiling.

**3 · The 10-minute evaluation.** [The apply quickstart](https://github.com/Xuzerken/ppsds/blob/main/examples/apply_drpp_to_your_own_env/quickstart.py) → [the honest limits](https://xuzerken.github.io/ppsds/research/preliminaries/#honest-limits). Run one script, read what the guarantee does not cover, and decide whether to try it.

**4 · Deploy on your own system.** [Apply DRPP](https://xuzerken.github.io/ppsds/examples/apply_drpp_to_your_own_env/) → [Getting Started](https://xuzerken.github.io/ppsds/tutorials/getting-started/) → [Troubleshooting](https://xuzerken.github.io/ppsds/tutorials/troubleshooting/). The four-step declaration, the pitfalls, and the two ways it surprises you.

**5 · Check or reproduce the conic DRPP paper.** [The paper](https://xuzerken.github.io/ppsds/research/drpp-conic/) → [the simulated-ambiguity example](https://xuzerken.github.io/ppsds/examples/understand_drpp_on_simulated_ambiguity/) → [the DRPP experiment](https://xuzerken.github.io/ppsds/experiments/drpp/) → [A new family and its DRPP](https://xuzerken.github.io/ppsds/tutorials/developer-guide/).

**6 · Check or reproduce the probabilistic predictability paper.** [The paper](https://xuzerken.github.io/ppsds/research/probabilistic-predictability/) → [the ceiling example](https://xuzerken.github.io/ppsds/examples/understand_predictability/) → [the predictability experiment](https://xuzerken.github.io/ppsds/experiments/predictability/).

**7 · Do research, or add a family.** [Preliminaries](https://xuzerken.github.io/ppsds/research/preliminaries/) → either paper → [A new family and its DRPP](https://xuzerken.github.io/ppsds/tutorials/developer-guide/) → [CONTRIBUTING](https://github.com/Xuzerken/ppsds/blob/main/CONTRIBUTING.md).

## Sections on this site

| Section | What it is |
|---|---|
| [Examples](https://xuzerken.github.io/ppsds/examples/) | Four stories you can run. Each prints every number it talks about; no batch infrastructure required |
| [Experiments](https://xuzerken.github.io/ppsds/experiments/) | The same questions at paper scale: cached, parallel, reproducible, with the figures the papers report |
| [Research](https://xuzerken.github.io/ppsds/research/) | The common preliminaries, and one page per paper with its result, its figure, its experiment links, and its citation |
| [Tutorials](https://xuzerken.github.io/ppsds/tutorials/) | Topics treated on their own terms — Getting Started, A new family and its DRPP, Troubleshooting, and the map |
| [API](https://xuzerken.github.io/ppsds/api/) | Generated from the docstrings, so it cannot drift from the code |

<details>
<summary>A little more about what is inside</summary>

`ppsds` predicts the next state of a stochastic dynamical system as a **distribution** rather than a point. It is for systems you can **step** (a simulator, or a Gymnasium environment) and can model, but not exactly: an underactuated robot, a vehicle or drone held near its operating point, a plant inside a process simulator.

An ordinary predictor trusts that model. That works until the real system drifts away from it, and then the prediction is quietly too narrow: **overconfident** exactly when you needed it to be honest.

`ppsds` ships one remedy today: **DRPP**, distributionally robust probabilistic prediction. You declare an **ambiguity set** — the systems you are willing to admit — and DRPP returns the predictive distribution whose *worst-case* expected score over that set is as high as it can be: a floor that holds for every system in the set, for the price of a little average performance.

That DRPP is the one for **conic-moment** ambiguity sets, on an **additive-drift** system. It is the first of a template rather than a single method: each family of ambiguity sets yields its own DRPP, and the `drpp` subpackage is built to hold more. Zoom out one level and DRPP is one *kind* of **probabilistic predictor**; [probabilistic predictability](https://xuzerken.github.io/ppsds/research/probabilistic-predictability/) answers the same question from the other side, for a system you *do* know exactly.

</details>

## Citation and licence

If you use `ppsds`, cite the software:

```bibtex
@software{ppsds,
  title   = {ppsds: probabilistic prediction for stochastic dynamical systems},
  author  = {Xu, Tao and He, Jianping},
  license = {MIT},
  url     = {https://github.com/Xuzerken/ppsds}
}
```

GitHub's **Cite this repository** button reads
[`CITATION.cff`](https://github.com/Xuzerken/ppsds/blob/main/CITATION.cff) and produces the same entry; an archived release with a DOI will be added once the first stable release is archived. Each paper's citation is at the end of its page: [conic DRPP paper](https://xuzerken.github.io/ppsds/research/drpp-conic/#citation) and [probabilistic predictability paper](https://xuzerken.github.io/ppsds/research/probabilistic-predictability/#citation).

`ppsds` is **MIT** licensed: use it freely, keep the copyright and licence notice. If you want to contribute, [CONTRIBUTING](https://github.com/Xuzerken/ppsds/blob/main/CONTRIBUTING.md) has the setup, the tests, and the rule that a change must not move a number.

## Next

- **Thirty seconds, no install**: [play the betting game](https://xuzerken.github.io/ppsds/examples/understand_ppsds_by_pmf_betting/)
- **You have an environment**: [Getting Started](https://xuzerken.github.io/ppsds/tutorials/getting-started/) puts a predictor on it
- **You want the theory**: [Research](https://xuzerken.github.io/ppsds/research/), then the two papers
- **You want the graph**: [the map](https://xuzerken.github.io/ppsds/tutorials/map/)
