# DRPP on simulated ambiguity

The production-size version of [the simulated-ambiguity example](../examples/understand_drpp_on_simulated_ambiguity/index.md):
seven models, a validated cache, multiple processes, and the figures the paper reports.

```bash
python experiments/drpp/run.py -n 8 -s 3 -p 100 -m final_round -c none -t 100
```

## Seven models, one runner

The teaching script is deliberately flat. Reproducing the paper's sweeps needs more, and the
runner is honest about being a **batch runner** rather than a tutorial: a model table in
`models.py`, a cache whose shape is part of its contract, and a `fork` requirement. The seven
models are five linear systems (`final_round`, `original`, `larger_x0`, `different_Sigma_bar`,
`different_system_convergence`) and two nonlinear ones (`NLink`, `higher_dim`).

```bash
cd experiments/drpp

# randomly sampled SDS, slowly varying
python run.py -n 8 -s 3 -p 100 -m final_round -c none -t 100

# randomly sampled SDS, i.i.d. per step
python run.py -n 8 -s 3 -p 1 -m final_round -c none -t 100

# the nonlinear six-link arm, worst-case SDS
python run.py -n 8 -s 3 -m NLink -c none -t 100 -e worst
```

`run.sh` (Linux/WSL) and `run.ps1` (Windows) drive all six combinations in sequence.

The second command prints the following (progress bars and worker `FutureWarning`s about the
deprecated full-trajectory helper are omitted; the output paths are abbreviated — the runner
prints them in full):

```
======================================================================
DRPP Gymnasium [worst]: NLink, period=N/A, control=none
======================================================================
Evaluating Worst-Nominal...
  Saved Worst-Nominal to .../data/gym/model=NLink_n=6/control=none/.../seed=3/worst-case/nominal.pkl
Evaluating Worst-Eig-DRPP...
  Saved Worst-Eig-DRPP to .../data/gym/model=NLink_n=6/control=none/.../seed=3/worst-case/eig_drpp.pkl
Evaluating Worst-Noise-DRPP...
  Saved Worst-Noise-DRPP to .../data/gym/model=NLink_n=6/control=none/.../seed=3/worst-case/noise_drpp.pkl

cache: recomputed 3, reused 0
  - nominal.pkl: parameters or code changed (fingerprint mismatch)
  - eig_drpp.pkl: parameters or code changed (fingerprint mismatch)
  - noise_drpp.pkl: parameters or code changed (fingerprint mismatch)
Done! Results saved to .../data/gym/model=NLink_n=6/control=none/.../seed=3/worst-case/
```

Run the identical command again and the cache answers instead of the solver:

```
cache: recomputed 0, reused 3
```

Each model runs through the **same generic `Env`**; what differs between systems lives entirely in
the ambiguity-set subclass, which is why one table can cover all seven models.

## Command-line flags

| flag | meaning |
|---|---|
| `-n` | number of worker processes (default `8`) |
| `-s` | random seed (default `3`) |
| `-p` | SDS re-sampling period: `100` = slowly varying, `1` = i.i.d. (default `100`) |
| `-m` | model; linear: `final_round`, `original`, `larger_x0`, `different_Sigma_bar`, `different_system_convergence`; nonlinear: `NLink`, `higher_dim` |
| `-c` | control: `none` or `regulation` (default `none`); **`regulation` is valid for the linear models only** |
| `-e` | experiment: `random` (default) or `worst` |
| `-t` | number of trajectories (default `100`) |
| `-f`, `--force` | ignore the cache and recompute everything |

`-p` distinguishes the two mechanisms of the paper: a large period (slowly varying SDS) versus
`1` (i.i.d. per step). `higher_dim` is the four-dimensional linear system: “nonlinear” above means
“not one of the five paper linear models”, not “uses the nonlinear ambiguity set”.

## Where the output goes

Results land under `experiments/drpp/data/gym/...`, resolved from `run.py`'s own file location
rather than the working directory, so a run can be launched from anywhere. Three files are written
per run (`nominal.pkl`, `eig_drpp.pkl`, `noise_drpp.pkl`), plus `traj.pkl`, `optimal.pkl` and the
figures for `random` runs. The data tree is **not** under version control.

## Caching and safety

- The cache is **validating**: a step is reused only if its manifest still matches (source
  hashes, a parameter snapshot, each dependency's sha256, and the array shapes). If a parameter or
  the code changed it recomputes and says why. `-f` forces a full recompute; a directory with no
  manifest recomputes on first use.
- `-c regulation` on a nonlinear model **exits immediately with an error** rather than failing
  halfway through a run.
- Multiprocessing relies on the `fork` start method, because the nonlinear ambiguity set holds
  lambdas that cannot be pickled. The runner asserts this and fails loudly otherwise.
- A `-t` below the largest fixed plot index emits a `RuntimeWarning` and clamps instead of
  crashing; see [Troubleshooting](../tutorials/troubleshooting.md) for the message.

## Relation to the paper

This is the code behind the performance figures in *Distributionally Robust Probabilistic
Prediction for Stochastic Dynamical Systems*: the six-panel error-bar comparison and the
trajectory/covariance-ellipse figures. The paper's page explains what is being plotted and why the
two modes exist:

[The conic DRPP paper →](../research/drpp-conic.md)

## Next

- **The semantics first, without the cache**: [the simulated-ambiguity example](../examples/understand_drpp_on_simulated_ambiguity/index.md)
- **The paper the figures belong to**: [Conic-moment DRPP](../research/drpp-conic.md)
- **Put DRPP on your own environment**: [Getting Started](../tutorials/getting-started.md)
- **Add a family of your own**: [A new family and its DRPP](../tutorials/developer-guide.md)
