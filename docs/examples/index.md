# Examples

Four self-contained ways into `ppsds`. Each lives in its own directory under
[`examples/`](https://github.com/Xuzerken/ppsds/tree/main/examples). Every number these pages
mention is printed by the script that produces it, so you can run the script and compare the
output instead of trusting the text.

| Example | The story | Who it is for | Time |
|---|---|---|---|
| [understand_ppsds_by_pmf_betting](understand_ppsds_by_pmf_betting/index.md) | Place bets on where a system goes next, then find out what they were worth | Anyone; no prerequisites | 30 seconds, no install |
| [apply_drpp_to_your_own_env](apply_drpp_to_your_own_env/index.md) | Put the predictor on a system you already step — and find out when it pays and when it charges you | You have a Gymnasium environment and intend to use the result | a few seconds per script |
| [understand_drpp_on_simulated_ambiguity](understand_drpp_on_simulated_ambiguity/index.md) | Declare an ambiguity set, grow a system out of it, and watch three predictors face two meanings of “worst” | You want the central semantics of DRPP, not just the API | seconds |
| [understand_predictability](understand_predictability/index.md) | The ceiling no predictor can beat, and the paper's figure | You want the number every predictor is measured against | about a second |

The first one needs no installation at all. The other three run on plain `pip install ppsds`; each
directory carries its own `environment.yml` if you would rather create a conda environment for it.

Several examples have a **heavier companion** under [Experiments](../experiments/index.md): the
same idea over thousands of trajectories, with a cache and a command line. The example is what you
read to understand it; the experiment is what you run to produce numbers.

## Next

- **The road through all of this**: [the map](../tutorials/map.md)
- **Put a predictor on a system of your own**: [Getting Started](../tutorials/getting-started.md)
- **The numbers behind the stories**: [Experiments](../experiments/index.md)
- **The theory the stories illustrate**: [Research](../research/index.md)
