## What changed, and why

<!-- The change and its reason. Link the issue if there is one. -->

## How it was verified

<!-- What you ran and what it produced. Numbers beat adjectives: a score, a maximum
     absolute difference, a runtime. If nothing numerical can move, say how you know. -->

## Does it touch numerics?

<!-- Anything numerical, the cache fingerprint, or the experiment pipeline goes through
     A/B comparison against the commit you branched from: -->

```
python tools/ab_compare.py --base <the commit you branched from>
```

- [ ] Every maximum absolute difference reported as `0.0`
- [ ] A difference is nonzero **and intended** — the measured magnitudes are stated above

<!-- The acceptance line here is bit-identical, not "close". The cone program keeps the best
     of several candidate worst directions and a 1e-14 perturbation can flip which one wins;
     the score then moves by about 1e-9, so the run still looks fine while the choice has
     changed. That is the change worth seeing rather than tolerating. -->

## Which side of the boundary

<!-- `experiment` and `visualization` are the periphery; the rest of the package is core.
     A core module must never import a periphery module: if it does, `pip install ppsds`
     drags matplotlib into every user's dependency closure. -->

- [ ] This change is confined to the periphery (`experiment`, `visualization`)
- [ ] This change touches core (`ambiguity`, `dynamics`, `probabilistic_predictor`,
      `environment`, `numerics`, `scoring_rule`, `control_policy`), and no core module now
      imports the periphery
