# The predictability experiment

The paper-scale version of [the ceiling example](../examples/understand_predictability/index.md):
100,000 trajectories, and the three-panel figure at the resolution the paper uses.

```bash
python experiments/predictability/run.py                 # 100,000 trajectories
python experiments/predictability/run.py --n_traj 2000   # a quick look
python experiments/predictability/run.py --save-figure   # write the three-panel PDF
```

## What it prints

```
   eps  predictability   approximation     measured      |err|  |err|/eps
------------------------------------------------------------------------------------
  0.25        -2.77259        -3.55752     -3.57131    0.01379    0.05516
   0.5        -1.38629        -2.17123     -2.22637    0.05514    0.11029
  0.75        -0.57536        -1.36030     -1.48344    0.12314    0.16419

At the horizon (step 30), running averages over 100000 trajectories:
   eps         mean         2.5%        97.5%
----------------------------------------------
  0.25     -3.57131     -3.64682     -3.49900
   0.5     -2.22637     -2.29730     -2.15844
  0.75     -1.48344     -1.54733     -1.42227

  h_d(noise)              = +1.38629
  H(noise || predictor)   = +2.17123
```

The `predictability` line is an upper bound on **any** predictor; the `approximation` line is
where the `O(epsilon)` result puts *this* one. The measured column tracks it. The gap between the
two lines is the price of the predictor being misspecified, not a statement that it is badly
implemented.

## Why it is cheap

The predictor predicts the **noise**, so the realised outcome it is scored on is the noise draw
itself — the drift cancels:

$$
L_\epsilon\bigl(\hat p_w(\cdot - f(x_k)),\, x_{k+1}\bigr)
  = L_\epsilon\bigl(\hat p_w,\, x_{k+1} - f(x_k)\bigr)
  = L_\epsilon(\hat p_w,\, w_k)
$$

The whole experiment is therefore one vectorised pass over `(n_traj * T, d_x)` noise draws, and
the predictor is a diagonal Gaussian, which the scoring rule has a closed form for. `10^5 x 30 x 3`
scores take seconds, not the hours a per-step cone program would. That contrast is the content:
the ceiling needs no predictor and no optimization, which is exactly why it can be computed at a
scale where the DRPP experiments cannot.

## Relation to the paper

This is the code behind the three-panel figure in *Probabilistic Predictability of Stochastic
Dynamical Systems*. The derivation of the bound, the `O(epsilon)` approximation, and the
single-trajectory convergence result are on the paper's page:

[The probabilistic predictability paper →](../research/probabilistic-predictability.md)

## Next

- **The derivation, not just the numbers**: [the predictability paper](../research/probabilistic-predictability.md)
- **The example that explains the table**: [the ceiling](../examples/understand_predictability/index.md)
- **The other side of the question**: [the DRPP experiment](drpp.md)
- **Score a predictor you wrote**: [the predictability paper](../research/probabilistic-predictability.md#evaluating-your-own-predictor)
