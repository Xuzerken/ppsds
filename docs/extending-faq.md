# Troubleshooting

For people **adding a new family of ambiguity sets**, and the DRPP it needs, or **running
experiments** on top of `ppsds`: the traps, what they look like when they bite, and what to do.

This page is deliberately separate from the [Extending](extending.md) guide so that the guide can
stay a straight line from idea to figure. It complements rather than repeats
[Getting Started](getting-started.md#common-pitfalls), which covers the pitfalls of *using* the
shipped predictors; read that one first if you have not.

Each entry has the same three parts: the **symptom** you will actually see, the **cause**, and the
**fix**. Symptoms are phrased the way they present, so the page is searchable by the text of an
error message rather than by topic.

## Installation and environment

Most installation problems are not specific to extending; [Getting
Started](getting-started.md#platform-notes) has the authoritative platform notes. The three below
are the ones that show up while developing against a checkout.

### `conda env create` aborts with `CondaVerificationError ... appears to be corrupted`

**Symptom**: building the environment from `environment.yml` fails with a message claiming a
package is corrupted. Retrying and clearing the cache change nothing.

**Cause**: the conda directory is on a **case-insensitive** filesystem. Packages contain paths that
differ only in case (`ncurses` alone has 25 such pairs under `share/terminfo/`), and on a
case-insensitive filesystem they overwrite each other. It is not a download problem, so no amount of
retrying helps.

**Fix**: put the conda directory on a case-sensitive filesystem. On macOS the default APFS is
case-insensitive, and a Docker **bind mount** inherits the host's semantics; a **named volume**
(ext4) does not. You can check a directory directly:

```bash
touch /path/to/check/P12-M /path/to/check/p12-m && ls /path/to/check/p12*
# one file listed -> case-insensitive (unsuitable); two -> case-sensitive
```

### The cone program fails to build, or `cvxpy` will not import, on Linux

**Symptom**: installing `cvxpy` from source fails, or importing it raises a compiler error.

**Cause**: `cvxpy` needs a C toolchain when pip finds no prebuilt wheel for your platform.

**Fix**: install `gcc` (and the usual build essentials), or use a Python version with published
wheels. `cvxpy` is a hard dependency of the package, but it is imported **lazily**, only when a
cone program is actually solved, so the nominal predictor and the environment layer work without a
working solver.

### Plots do not appear, or matplotlib aborts on a headless machine

**Symptom**: a script that calls `plt.show()` does nothing, or matplotlib raises about a missing
display.

**Cause**: there is no display.

**Fix**: set a non-interactive backend: `MPLBACKEND=Agg python your_script.py`. The test suite does
this, and it is what CI does. `matplotlib` is an optional dependency: install it with
`pip install "ppsds[experiments]"`, and code that uses it belongs in the periphery, not in a core
module.

## Numbers and dtypes

Two of these are version-dependent and two are conventions. All four produce a wrong number rather
than a crash.

### The cone program raises `The 'minimize' objective must be real valued`

**Symptom**: `eigvec_fixed_minmax` fails with that message, on a real symmetric input covariance.
The traceback points nowhere near the real cause.

**Cause**: `np.linalg.eig` does **not** return a consistent dtype across numpy versions. For the
same real symmetric matrix it returns `float64` on some versions and `complex128` on others (with
imaginary parts exactly zero and identical real parts). A complex array reaching the cone program
makes it complain about the objective.

**Fix**: never call `np.linalg.eig` directly. Use
[`ppsds.numerics.linalg.eig_real`](https://github.com/Xuzerken/ppsds/blob/main/src/ppsds/numerics/linalg.py),
which takes the real part of both the eigenvalues and the eigenvectors. That is lossless **for a
symmetric matrix**, which is the only case here. Do not reach for it as a general-purpose
`eig` wrapper.

### A tiny change flips which direction the adversary attacks

**Symptom**: you nudge a tolerance, or an unrelated numerical change lands, and the reported
worst-direction index changes. The scores move by about `1e-9`, so nothing looks broken.

**Cause**: the cone program keeps the best of several candidate directions. When two candidates are
within about `1e-14` of each other, the choice between them is decided by floating-point noise. The
solver's own parameters (`max_iters`, `eps`) participate in that.

**Fix**: treat the solver parameters as **load-bearing**, not as tuning knobs: changing them
changes a discrete output. If you must change them, expect the discrete index to move and justify it
with the score magnitudes. Do not assert on the index in a test unless you mean it; the shipped
suite has exactly one such assertion, and it is the most likely assertion to go red one day.

### The predictive covariance is symmetric only up to rounding

**Symptom**: a test asserting exact symmetry of a predictive covariance fails by about
`2.2e-16`, and passes on your machine but fails in CI.

**Cause**: the covariance is rebuilt as `v @ diag(lamb) @ v.T`. In exact arithmetic that is
symmetric; in floating point the two off-diagonal entries accumulate in different orders, so they can
differ in the last bit. Different BLAS builds round differently, which is why the failure moves
between machines.

**Fix**: compare with a small relative tolerance (about `1e-12` times the magnitude of the matrix),
not for equality. If you see a difference far larger than machine epsilon, that is a real bug, not
this.

### Numbers come out off by a square, or by `d/2`

**Symptom**: a score or a radius is wrong by a factor, a square, or a constant offset, with no
error anywhere.

**Cause**: two easy confusions. First, **radii that are squared and radii that are not**: the
shipped family mixes them on purpose, and [Getting
Started](getting-started.md#common-pitfalls) tabulates which is which. Second, an **expected** score
bound carries a `+d` term from the Gaussian differential entropy, while a `logpdf` evaluated at a
point does not; mixing the two shifts everything by `d/2`.

**Fix**: for the shipped family, take the table in Getting Started as the reference. For your own
family, the conventions are **yours to define**, and the trap is not writing them down: state, in the
docstring of the class, which quantity each parameter bounds, and whether it is squared. A radius
whose meaning is only in your head will be read the other way by the next person.

### You assumed the robust covariance dominates the inflated one, and it does not

**Symptom**: a test or a claim that the robust predictor's covariance is at least the noise-level
one in every direction fails on some inputs.

**Cause**: the cone constraint allows coordinates whose predictive eigenvalue falls **below** the
scaled nominal one; the robustification is not a componentwise inflation. Attempts to assert
dominance pass or fail depending on the draw.

**Fix**: if you need a dominance statement, state the one that is actually true for your family and
test it on a fixed seed. For the shipped family the meaningful statements are the two reference
bounds, and the inflation along the worst direction. See [Concepts](concepts.md#two-bounds-bracket-the-answer).

## Reproducibility

### Results drift when you add an unrelated random call

**Symptom**: after a change that should not affect numbers, every sampled trajectory is different.

**Cause**: the samplers draw from the **global** `numpy` random state. Any extra draw anywhere
before them shifts the whole consumption order, so all downstream samples change. This is a
deliberate design: reproducibility is the caller's, not the library's.

**Fix**: seed at the call site, once, immediately before sampling (`np.random.seed(...)`). Do not
insert a random call into a code path that others' existing results depend on. If you are adding a
mode that samples, follow the shipped random mode: it seeds the global state from the environment's
own generator and then **restores** it, so stepping an episode does not pollute the caller's state.

## Multiprocessing

### `cannot pickle 'function' object`, or the runner refuses to start

**Symptom**: worker processes die with a pickling error, or the batch runner raises a `RuntimeError`
mentioning `fork` and "not picklable".

**Cause**: ambiguity sets routinely hold lambdas (a drift callable, a radius callable). Lambdas
cannot be pickled, and worker processes receive their work by pickling unless the start method is
`fork`, which shares memory instead.

**Fix**: use `fork`. The runner asserts the start method and **fails loudly** rather than degrading
silently, because the silent version of this failure is worse: with `spawn` (the default on macOS and
Windows) it would error in a way that looks like a bug in your family. Two consequences when you
write your own driver: call the runner's `configure()` **before** creating any process pool, and keep
worker functions at module level so the pool can reference them.

## The cache

### A figure claims more trajectories than were ever computed

**Symptom**: the plotted result is smoother than it should be, or a re-run with a larger `-t`
finishes surprisingly fast.

**Cause**: this was a real bug, and it is worth knowing the shape of it: the old cache skipped work
when the output file merely **existed**, and its key did not encode the trajectory count. A directory
holding 5 trajectories was reused for a request of 100.

**Fix**: the cache is now **validating**: a step is reused only if its manifest still matches. Know
what invalidates it, because the answer surprises people:

| invalidates | does not invalidate |
|---|---|
| the parameters in the run's snapshot | the plotting code (excluded by module prefix) |
| the bytes of any loaded `ppsds.*` module | the working directory you launch from |
| a dependency file's sha256 | |
| array shapes that no longer match | |

The source fingerprint is computed **byte for byte** over the modules that are loaded. So renaming a
module, moving a file, or **editing a docstring or a warning message** invalidates the whole cache,
including the warning text, which is API. Expect a recompute after any such edit; it is the intended
price of the guarantee, not a bug.

### The cache never hits, or a manifest looks stale immediately

**Symptom**: every run recomputes, even though nothing changed.

**Cause**: the fingerprint is taken over `sys.modules`, so it only sees what has already been
imported. Building the cache context before all imports have finished produces a fingerprint of an
incomplete module set.

**Fix**: construct the cache context at the entry point, after imports. And remember that the
results directory is **not** under version control: the cached pickles are the only copy, so a forced
recompute (`-f`) is a real cost, not a refresh. See
[Examples](examples.md#caching-and-safety) for the runner's own account of this.

## The core and periphery boundary

### `import ppsds` got slower, or CI's core-only job fails

**Symptom**: a plain `pip install ppsds` starts pulling plotting libraries; or the `core-only`
continuous-integration job fails while everything passes locally.

**Cause**: a **core** module imported a **periphery** module. The core is `ambiguity`, `dynamics`,
`probabilistic_predictor`, `environment`, `numerics`, `scoring_rule`, `control_policy`, and it may
depend only on numpy, scipy, cvxpy and gymnasium. The periphery (`experiment`, `visualization`)
adds matplotlib, pandas, seaborn and tqdm, and is installed by an extra.

**Fix**: keep the import out of the core. This one is checked three ways, so it will not stay
broken: a static test, a test that loads each core entry point in a clean subprocess, and a CI job
that installs the package **without** the extra, which is the only way to prove the boundary rather
than merely assert it. If you add a core module, add it to that test's list of entry points.

### Nobody notices that the solver import crept to module level

**Symptom**: `import ppsds` becomes noticeably slower, or importing it fails where no solver is
installed.

**Cause**: `cvxpy` is a hard dependency, but it is imported **lazily**, inside the function that
solves the cone program. That is why the nominal predictor and the environment layer cost nothing
extra.

**Fix**: keep it lazy. If your family needs a solver, import it inside the function that uses it.

## Shapes and parameter formats

### An error deep inside `predict()` whose cause is invisible

**Symptom**: `predict()` raises something like `Array 'mean' must be a vector of length N²`, and
the code that produced the bad state is far away.

**Cause**: a mis-shaped input was **silently broadcast**. Passing a control of shape `(1, 1)`
turned a matrix-vector product into a `(d, 1)` array, and adding a `(d,)` bias broadcast the whole
thing to `(d, d)`. NumPy is happy to do this; the failure surfaces much later, in a different shape.

**Fix**: validate at the boundary. The predictors do this with a helper that checks rank and length
and raises naming the offending argument. Do the same in your own `update()`: an error raised where
the mistake happened is worth ten raised downstream.

### `ValueError` naming a key you thought was fine

**Symptom**: `Env.step()` raises, naming the offending key and the accepted ones.

**Cause**: the environment validates the parameter key set at the top of `step()`, against the set's
`param_keys`. Two families use different key sets, and the validation exists precisely because an
earlier version guessed the format by testing for a key's presence, which swallowed exactly this
mistake.

**Fix**: make sure your mode returns only keys your set declares. The common instance: the extra
key that `worst_case_params()` returns to report the worst direction. It is diagnostic, not a
parameter, and it must be removed before the parameters are handed to `step()`.

### An array changed under you

**Symptom**: a nominal parameter drifts a little after every sample; results differ between two runs
that should be identical.

**Cause**: the sets hold arrays **by reference** (`np.asarray` does not copy), so mutating
something you obtained from a set mutates the set.

**Fix**: copy before you write (`B = self.B_bar.copy()`), never hand a module-level array to a set,
and give every factory a fresh one. A related trap on the consumer side: arrays passed as
`sds_params` follow the same rule.

## Does your set contain its own generator?

**The invariant: every sample your generator produces must be a member of the set you declared.**
If it is not, your ground truth lies outside your own declaration, and the guarantee is vacuous: in
simulation it shows up as a bound that appears to fail, which looks like an algorithm bug and is
actually a declaration bug. [Concepts](concepts.md#honest-limits) states this as a design invariant;
here is what it cost when it was violated.

### The drift sampler put most of its samples outside the declared ball

**Symptom**: none, at run time. It surfaced only when someone measured it.

**Cause**: the sampler drew the drift perturbation from a Gaussian whose standard deviation was set
by a `3σ` heuristic, divided by three and applied **per component**. That ignores the dimension: the
expected squared norm grows with `d`, so at `d = 12` the draw was already past the bound on average.

**Fix**: the corrected sampler draws **uniformly inside the ball** of the declared radius. Measured
on the same test, the fraction of draws outside the set went from **70.1%** to zero. If your radius is
a norm bound, sample inside the ball; do not sample per component and hope.

### The noise-mean box was twice as wide as its own declaration

**Symptom**: again none until measured.

**Cause**: the noise mean is declared with a **squared** radius: the constraint is on the norm,
`|mu - mu_bar| <= sqrt(gamma)`. Sampling each component from a box of half-width `sqrt(gamma)`
instead of `sqrt(gamma / d)` makes each component's box `sqrt(d)` times too wide, and the
worst-case **squared** norm `d` times the declaration.

**Fix**: divide the half-width by `sqrt(d)`. More generally: when a radius bounds a **norm**, every
sampler for it needs the dimension in it somewhere, and the place it goes is easy to get wrong.

### The drift bound must be the sum, not the combined norm

**Symptom**: about 1% of samples fall outside the declared set, with everything else looking right.

**Cause**: when a perturbation splits into a state part and a control part, the tight bound on the
sum is the **sum of the two norms** (the triangle inequality is tight here). A single combined-norm
expression is always smaller, so it excludes samples the generator actually produces.

**Fix**: use the sum. This is worth internalising as the general pattern: the bound must be the one
your sampler can saturate, not the tightest-looking expression you can write down.

### An aliasing bug that made the nominal model walk away

**Symptom**: the set's nominal parameters drift permanently, a little per sample.

**Cause**: a sampler assigned `B = self.B_bar` instead of a copy and then wrote into `B`. Since the
trajectory sampler calls `sample()` once per period, the nominal value took a random walk, silently,
and permanently.

**Fix**: copy on the way in. There is a test for exactly this: sampling must not change the set.

!!! tip "When the paper and the code disagree, sample from it and ask which is self-consistent"
    That question has resolved every instance of this class of bug so far, in both directions: the
    paper's closed form was wrong once, and the code's sampler was wrong twice. Neither "the paper is
    authoritative" nor "the code is authoritative" is the right reflex. The useful question is
    **"which of these contains its own generator?"**

## Predictor state and reset

### Reusing a predictor for a second trajectory gives numbers from the first

**Symptom**: the second trajectory's first row carries the previous episode's last prediction
instead of an empty placeholder.

**Cause**: the full-trajectory helper rewrites `f_bar` / `mu_bar` / `Sigma_bar` **in place**, and
derived quantities (eigenvalues, eigenvectors) follow them. Clearing only the prediction outputs
leaves the first trajectory's parameters in place for the second.

**Fix**: `reset()` must restore the nominal quantities from a saved copy and recompute what is
derived from them, not just null the outputs. The interface's own docstring lists which quantities
this means. The shipped helper does it in a `finally`, so an exception mid-trajectory does not leave
the instance poisoned either.

### An interface that raises at construction time when you inherit it

**Symptom**: a class that satisfies an interface by inspection fails with `AttributeError` the
moment it is instantiated.

**Cause**: the interface declared its attributes as `@property`, which is a **data descriptor**,
while implementations assigned them as plain instance attributes. The two are incompatible. Worse, the
interface was registered virtually, so `isinstance()` returned `True` while nothing was enforced,
the incompatibility was hidden rather than surfaced.

**Fix**: this is why the predictor interface is a `typing.Protocol`: attributes are declared by
**annotation** (which a plain instance attribute satisfies), methods by `...`, there is zero runtime
overhead, and no inheritance is involved or wanted. When you define an interface, make sure the
implementations could actually satisfy it, because an interface you cannot implement is worse than none.

## Figures

### `RuntimeWarning` about a trajectory index being out of range

**Symptom**: running the batch runner with a small trajectory count prints a warning like:

```
RuntimeWarning: trajectory index in plot_traj 2 out of range (available 0..1); using 1
instead. Note that the figure content changes accordingly; to show a specific trajectory,
raise -t or use a smaller index.
```

**Cause**: which trajectories to plot is fixed in the model table, so a small `-t` cannot satisfy
it. This used to raise `IndexError` **after** the expensive computation had finished, throwing away
a completed run.

**Fix**: nothing breaks; the index is clamped to the last available trajectory. The warning is not
suppressed because clamping **changes what the figure shows**; that is the part worth reading. Raise
`-t`, or lower the index in the model table.

### A run "passed" but the figure is wrong

**Symptom**: the command exits `0` and the figure does not say what you think.

**Cause**: the exit code cannot see a clamped index, and it cannot see a number that moved either.

**Fix**: treat the **data**, not the exit status, as the acceptance criterion. That is exactly the
choice the A/B comparison tool makes: it compares the pickles element by element rather than trusting
whether a run completed.

## Known rough edges in this release

Honest limits, as of `0.0.1`, so you do not spend an afternoon rediscovering them.

### The full-trajectory prediction helper is deprecated, and its replacement is incomplete

**Symptom**: a `FutureWarning` telling you to migrate to the environment layer.

**Cause**: the helper runs a whole trajectory in one call. It is deprecated in favour of stepping an
`Env`, which is what the examples and the batch runner do.

**Fix**: use the environment layer for anything new. Note the honest caveat: the helper's
**linear time-varying path has no equivalent in the environment layer**, so if that is the path you
need, the migration target does not exist yet. Say so rather than silently relying on a deprecated
call.

### The shipped family cannot express general uncertainty in `A` and `B`

**Symptom**: none; this is a limit of what the shipped family can express, and a reason you may be
writing a new one.

**Cause**: the shipped linear parameterization perturbs only the superdiagonal of `A` and a single
entry of `B`. That is the structure of the paper's own example, not a general description of
uncertainty in `A` and `B`, and it cannot express it.

**Fix**: [Concepts](concepts.md#honest-limits) states this limit precisely; read it before concluding
that the shipped family is a general-purpose conic-moment implementation. Extending it is exactly what
the [Extending](extending.md) guide is for. Two smaller edges of the same kind: a few modules ship
with no callers and are therefore unexercised by the suite, and one assertion in the golden fixture
pins the discrete worst-direction index, which makes it the most likely test to go red on a
different numerical stack.
