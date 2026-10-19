# Experiments

An experiment is the rigorous half of an example: the same question, at a scale where the answer
is a number rather than an impression. Every experiment here is reproducible, states its runtime,
and is the code behind a figure or a table in a paper.

| Experiment | Deepens | Produces | Runtime | Paper |
|---|---|---|---|---|
| [drpp](drpp.md) | [what “worst case” means](../examples/understand_drpp_on_simulated_ambiguity/index.md) | cached pickles, error bars, covariance ellipses, seven models | minutes | conic DRPP paper |
| [pmf_betting](pmf_betting.md) | [the betting game](../examples/understand_ppsds_by_pmf_betting/index.md) | scored hands and exported runs, against the ceiling | seconds | uses the predictability bound |
| [predictability](predictability.md) | [the ceiling](../examples/understand_predictability/index.md) | the three-panel figure over 100,000 trajectories | seconds | probabilistic predictability paper |

Each experiment directory carries its own `environment.yml`, so you can create exactly what one
experiment needs. The command lines below assume the repository root as the working directory, with
`ppsds` installed editable (`python -m pip install -e ../..` from the experiment directory).

## Next

- **The examples these deepen**: [Examples](../examples/index.md)
- **The papers the figures belong to**: [Research](../research/index.md)
- **The commands, once**: [Getting Started](../tutorials/getting-started.md)
