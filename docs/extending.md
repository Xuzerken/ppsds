# Extending `ppsds`: a new ambiguity-set family, and the DRPP it needs

`ppsds` is built to hold **more than one family of ambiguity sets**, and each family brings its own
**DRPP**. This release ships the **conic-moment** family, on an **additive-drift** system: that is
the one DRPP you see today. This page is for the case where you want to add **your own family**
and the DRPP that goes with it, to study it, to compare it against what ships, or to put it in a
paper.

Adding a whole new *kind* of probabilistic predictor, rather than another family of ambiguity sets,
is a larger job, and it lands one level up in `ppsds.probabilistic_predictor`. The layers below
and the verification discipline are the same; what changes is the interface you have to satisfy,
which is the one in [Step 2](#step-2-the-predictor).

It is a **workflow**: what to decide before writing code, what each layer expects from you, in what
order to build it, how to run an experiment on it, and how to check a number before you believe it.
The traps and their symptoms are on a separate page, [Troubleshooting](extending-faq.md); this page
links to it rather than digressing.

If you only need to *use* the predictors that ship, you are on the wrong page; read
[Getting Started](getting-started.md) and [Concepts](concepts.md) instead.

## The shape of the work

Six phases, four of which are code. The point of the picture is that the **contracts flow one way**:
each layer consumes the one above it, so building them out of order means writing code against a
guess.

```
   a new family of ambiguity sets        (your idea, or your paper)
              │
              ▼
   P0  decide the mathematics            what the set is, what generates from it, what the
       no code yet                       adversary's best response is, and whether that
              │                          response is computable at all
              ▼
   P1  the ambiguity set                 ambiguity/additive_drift/<family>.py
       declares and generates            → its contract tests pass
              │
              ▼
   P2  the predictor                     probabilistic_predictor/drpp/additive_drift/<family>.py
       predicts, robustly                → its contract tests pass
              │
              ▼
   P3  the adversarial dynamics mode     environment/dynamics_mode/adversarial/
       attacks that predictor            additive_drift/<family>.py
              │                          → the env tests pass
              ▼
   P4  smoke test                        your own short script, one trajectory
       does the machinery turn?          → three scores, and they look sane
              │
              ▼
   P5  the experiment                    batch runner, or your own loop
       many trajectories, fairly         → pickles and figures
              │
              ▼
   P6  verification                      tools/ab_compare.py, pytest
       is the number trustworthy?        → the shipped family is untouched, bit for bit
              │
              ▼
        a figure you can defend
```

**P3 is optional.** The environment layer exists only for the *simulated ambiguity* job: declaring a
set, generating dynamics from it, and letting an adversary pick the worst member. If your family is
about prediction alone, stop after P2: you predict, and you score against states from your own
environment. [Concepts](concepts.md#the-two-ways-to-use-the-package) draws that line.

| phase | what it produces | where it is checked |
|---|---|---|
| P0 | decisions, written down | — |
| P1 | a set that declares *and* generates | the contract tests of [Step 1](#step-1-the-ambiguity-set) |
| P2 | a predictor that satisfies the interface | the contract tests of [Step 2](#step-2-the-predictor) |
| P3 | a mode whose "worst" is your set's rule | the env tests of [Step 3](#step-3-the-adversarial-dynamics-mode) |
| P4 | one trajectory's scores | eyeball them against [the reference bounds](#step-4-smoke-test-on-one-trajectory) |
| P5 | pickles and figures | [Examples](examples.md) explains the runner and its cache |
| P6 | a claim you can defend | `tools/ab_compare.py` and `pytest`: [Step 6](#step-6-before-you-believe-a-number) |

## The files you will touch

A family is a **new module beside the existing one**, never a branch inside it. That is the whole
convention, and it is why the paths spell out *assumption* / *family*:
`ambiguity/additive_drift/conic_moment` reads as *additive drift* (the assumption) + *conic moment*
(the family). Your family lands at `ambiguity/additive_drift/<family>.py`.

| | path | what happens |
|---|---|---|
| **add** | `src/ppsds/ambiguity/additive_drift/<family>.py` | your set: it declares the uncertainty and generates from it |
| **add** | `src/ppsds/probabilistic_predictor/drpp/additive_drift/<family>.py` | your predictor(s) |
| **add** | `src/ppsds/environment/dynamics_mode/adversarial/additive_drift/<family>.py` | your worst-case dynamics (P3 only) |
| **add** | `tests/test_<family>.py` | the contract tests, modelled on the existing ones |
| **edit** | `src/ppsds/environment/dynamics_mode/__init__.py` | re-export your mode, so the package surface stays consistent |
| **edit** | `tests/_systems.py` | add a factory building your set, and parametrize the contract tests over it |
| **never** | `src/ppsds/ambiguity/additive_drift/conic_moment.py` | the shipped family is a reference implementation, not a base class to grow |
| **never** | the dependency sets of core modules | see [the boundary](extending-faq.md#the-core-and-periphery-boundary) |

**Do not subclass the shipped set.** The radius representation and the worst-case rule differ
between families, so inheritance buys you nothing and leaves you with parent methods that silently do
the wrong thing. Only *part* of the shipped base class is family-agnostic; [Step 1](#step-1-the-ambiguity-set)
lists exactly which part to copy.

## Before you write code: three questions

Skip this and the code will be a guess. Each question has a wrong answer that only shows up much
later, in a figure or in a reviewer's comment.

**1. Does your set contain its own generator?** You are writing two things: a set, and a sampler
that draws members of it. The claim "the set is the ground truth" is only coherent if every sample
lands inside the set. This is a design invariant, not a detail: [Concepts](concepts.md#honest-limits)
explains what breaks when it fails, and [Troubleshooting](extending-faq.md#does-your-set-contain-its-own-generator)
has two measured examples of it failing. Write the check as a test in P1, before any experiment.

**2. What is the adversary's best response?** A robust predictor is built against a specific
worst case. Yours is: given the predictive covariance, which member of the set hurts that
prediction most? That single question *is* `worst_case_params()`, and it is also what the
adversarial mode turns into simulated dynamics. Being unable to answer it crisply means the
robustness you are about to claim has no object.

**3. Is that best response computable?** The shipped family solves its minmax as a cone program at
every step, which is what makes an eigenvalue-level robust predictor affordable. Yours needs *some*
route: a closed form, a convex program, or a relaxation you can state and defend. If there is none,
you can still build the nominal and noise-level baselines, but say so, rather than shipping a
predictor whose covariance was never derived.

!!! note "The set is a declaration, not a data structure"
    Nothing in `ppsds` checks your set against reality. The guarantee it offers is conditional:
    *for every system inside the declared set*, the expected score is at least such-and-such. If the
    true system falls outside the set, the statement is vacuous. That is the sense in which your
    declaration matters more than the accuracy of your nominal model. See
    [Concepts](concepts.md#honest-limits).

## Step 1: the ambiguity set

**The set's job is to declare how much uncertainty there is, and to generate SDS parameter
trajectories from that declaration.** Everything else in the package consumes those two abilities.

Part of the shipped base class is family-agnostic, so copy it. The rest is conic-moment specific
and has to be reimplemented from your own mathematics.

| family-agnostic: copy | family-specific: yours |
|---|---|
| `x_dim` / `u_dim` / `f_bar` / `mu_bar` / `Sigma_bar` | the radius or parameterization itself |
| the `param_keys` declaration and `validate()` | `sample()`: how a member is drawn |
| `transition()`, the additive-noise kernel | `worst_case_params()`: the worst-case rule |
| `sample_trajectory()`, the piecewise-constant skeleton | any closed-form bound you can prove |
| the `mean()` / `sample()` interface | |

`transition()` is where the **additive** assumption lives: `x' = mean(x, u; params) + w`. Keep it.
If your assumption is not additive (state-dependent or multiplicative noise, say), then you are
writing a different base class *and* a different predictor, and this page is only a shape to follow.

### Two shape contracts

These are the parts other layers actually rely on, so get them exactly right:

```
param_keys : frozenset            the sds_params keys this set accepts.
                                  A MISSING key means "use the nominal value";
                                  an UNKNOWN key must raise, naming the key and the class.

_traj_fields : tuple              ((key, per-step shape), ...) -- declares the output of
                                  sample_trajectory().

sample_trajectory(T, period=1)    -> dict keyed by _traj_fields' names; EVERY value has
                                     leading axis T+1, piecewise constant for `period` steps.
                                     period=None behaves as 1; period < 1 raises.

worst_case_params(x, u, lamb_pred) -> this family's parameters, PLUS one extra key naming the
                                     worst direction (the shipped family calls it 'j').
```

That last extra key is **load-bearing and easy to miss**: it is diagnostic, not part of
`sds_params`, and `Env.step()` validates the key set, so the mode must remove it before stepping.
Whatever you name yours, pop it in the mode ([Step 3](#step-3-the-adversarial-dynamics-mode)).

The missing-key rule is not laxity, it is a real case: on the first step of an adversarial episode
there is no prediction yet, so the mode returns `{}` and every term takes its nominal value.

### The four methods

| method | semantics |
|---|---|
| `mean(x, u, params)` | nominal drift **plus** this step's perturbation. Does **not** include the noise mean. |
| `sample()` | draw one member; return a dict shaped like `sds_params`. |
| `sample_trajectory(T, period=1)` | the piecewise-constant trajectory above. |
| `worst_case_params(x, u, lamb_pred)` | the adversary's best response, given the predictive covariance's eigenvalues. |

### The self-consistency check, as a test

Write this in P1, not at the end. It is the cheapest test you will ever write and it catches the
failure mode that is hardest to see in a figure:

```python
def test_samples_stay_inside_the_declared_set(my_set):
    """Every sample must be a genuine member -- the set is the ground truth."""
    np.random.seed(17)
    for _ in range(25):
        params = my_set.sample()
        # Assert YOUR set's definition, in YOUR terms. For a radius r on a quantity q:
        #     assert np.linalg.norm(params['q']) <= r + 1e-12
        # and remember which quantities are squared -- mixing them is the classic error
        # (see Troubleshooting).
        ...
```

Two more contract tests worth copying, both of which the shipped suite runs: `sample()` must not
mutate the set (arrays are held by reference, not copied, so alias carefully inside your sampler),
and `worst_case_params()`'s output must be accepted by `validate()` once the direction key is
removed. The suite that does all of this for the shipped family is the template:

- [`tests/test_ambiguity.py`](https://github.com/Xuzerken/ppsds/blob/main/tests/test_ambiguity.py)
- [`tests/_systems.py`](https://github.com/Xuzerken/ppsds/blob/main/tests/_systems.py): where the
  factories live; add yours and parametrize the contract tests over it

### A skeleton

Names are placeholders; the structure is the contract.

```python
import numpy as np


class AdditiveDriftFamilySet:
    """Additive drift + <your family>. Assumption is additive drift; the family is yours."""

    param_keys = frozenset({'delta_f', 'mu', 'Sigma'})   # whatever your family needs

    def __init__(self, f_bar, mu_bar, Sigma_bar, ...):
        self.f_bar = f_bar['function']
        self.x_dim, self.u_dim = int(f_bar['x_dim']), int(f_bar['u_dim'])
        self.mu_bar = np.asarray(mu_bar, dtype=np.float64)
        self.Sigma_bar = np.asarray(Sigma_bar, dtype=np.float64)
        self._traj_fields = (('delta_f', (self.x_dim,)),
                             ('mu', (self.x_dim,)),
                             ('Sigma', (self.x_dim, self.x_dim)))

    # -- environment side: these two are the "additive" assumption; copy them -------------
    def validate(self, params):        # unknown key -> ValueError naming key and class
        ...
    def transition(self, x, u, params, rng):   # x' ~ N(mean(x, u; params) + mu, Sigma)
        ...

    # -- generation: your mathematics ----------------------------------------------------
    def mean(self, x, u, params):
        return self.f_bar(x, u) + params.get('delta_f', np.zeros(self.x_dim))

    def sample(self): ...
    def sample_trajectory(self, T, period=1): ...     # the piecewise-constant skeleton
    def worst_case_params(self, x, u, lamb_pred): ...  # returns a dict, plus the direction key
```

## Step 2: the predictor

**A predictor is three attributes and three methods, and the interface is structural.** It is
declared as a `typing.Protocol` in
[`src/ppsds/probabilistic_predictor/base.py`](https://github.com/Xuzerken/ppsds/blob/main/src/ppsds/probabilistic_predictor/base.py):
attributes by annotation, methods by `...`, and **no inheritance**: the shipped predictors do not
inherit it either, and a test asserts that. Satisfy it by having the members, not by subclassing.

| member | why it exists |
|---|---|
| `x_pred` | the predicted mean; `None` before the first `update()` |
| `Sigma_pred` | the predictive covariance |
| `lamb_pred` | **the eigenvalues of `Sigma_pred`**: the adversarial mode picks the worst direction from exactly these |
| `predict()` | returns something with a `.logpdf`; that is what the scoring rule calls |
| `update(x_observe=None, u_observe=None)` | advance one step; `None` selects multi-step prediction |
| `reset()` | restore a reusable state; see the warning below |

!!! warning "`reset()` must restore, not just clear"
    The shipped full-trajectory helper rewrites the nominal `f_bar` / `mu_bar` / `Sigma_bar`
    **in place**, and derived quantities (eigenvalues, eigenvectors) follow. Clearing the prediction
    outputs alone leaves the first trajectory's parameters behind for the second. If you reuse
    predictor instances, restore from a saved copy of the nominal values: the contract's own
    docstring spells out which quantities.
    [Troubleshooting](extending-faq.md#predictor-state-and-reset) has the history.

`lamb_pred` is the one member people forget to fill. Filling `x_pred` and `Sigma_pred` makes the
predictor *score* correctly; leaving `lamb_pred` stale makes the adversarial mode attack the wrong
direction: a wrong number that looks entirely plausible.

!!! note "The mathematics on your set is your contribution"
    This page gives you the interface, not the derivation. Working out the functional minmax over
    *your* ambiguity set is the research; a page that handed you one would be handing you something
    it cannot check. [Concepts](concepts.md#what-drpp-optimizes) explains what the objective is and
    what the shipped relaxation gives up, which is the right thing to read before deriving yours.

One asymmetry is worth pinning down early, because it is a silent factor in every score: an
**expected** score bound (like `value_upper_bound_true_transition()`) carries a `+d` term from the
Gaussian differential entropy, while a `logpdf` **evaluated at a point** does not. Mixing the two
shifts everything by `d/2`.

## Step 3: the adversarial dynamics mode

The mode turns your set's worst-case rule into simulated dynamics. Implement three methods:

| method | what it does |
|---|---|
| `reset(rng, T)` | clear the injected information at the start of an episode |
| `get_params(step_idx)` | return the parameters for this step; delegate the worst case to `abg.worst_case_params()` |
| `set_info(lamb_pred, x, u)` | receive the predictor's information **before** each step |

Then re-export it from
[`src/ppsds/environment/dynamics_mode/__init__.py`](https://github.com/Xuzerken/ppsds/blob/main/src/ppsds/environment/dynamics_mode/__init__.py)
so it sits alongside the shipped modes.

Three details:

- **Return `{}` on the first step** if no prediction has been injected yet: nothing is known, so
  everything takes its nominal value.
- **Pop the direction key** before returning. `Env.step()` validates the parameter key set and would
  reject it.
- **Do not call it `AdversarialDynamics`.** The name would overclaim: this mode's worst case is
  defined entirely by your set's rule, so it is only adversarial *for your family*.

Injecting is a public, single-channel operation: `env.inject_eigvec_fixed_info(lamb_pred, u)`.
Note the argument order differs from the mode's own `set_info(lamb_pred, x, u)`; the environment
supplies `x` itself. For modes that do not consume the information it is a no-op, so a loop can call
it unconditionally.

## Step 4: smoke test on one trajectory

Before any experiment, run **one** trajectory. The step order is the semantics, not an
implementation detail:

```
    update the predictor with (x_k, u_k)     -> the prediction for k+1
    inject lamb_pred                          -> the adversary chooses from it
    step the environment                      -> the true system advances
    score the prediction against x_{k+1}
```

Getting this order wrong is not a crash; it is a plausible number computed from the wrong pairing.
The teaching example in
[`examples/understand_drpp_on_simulated_ambiguity/quickstart.py`](https://github.com/Xuzerken/ppsds/blob/main/examples/understand_drpp_on_simulated_ambiguity/quickstart.py)
is exactly this loop, ~40 lines, single process, core API only. Copy its shape:

```python
env = Env(my_set, MyAdversarialMode(my_set), x0=x0, max_steps=T)
predictor = MyPredictor(...)          # built from the SAME set -- never retype literals
x, _ = env.reset(seed=seed)
for k in range(T):
    u = policy(x)
    predictor.update(x, u)
    env.inject_eigvec_fixed_info(predictor.lamb_pred, u)
    x, *_ = env.step(u)
    scores[k] = scorer.score(predictor.predict(), x)
```

Two things that make the difference between a measurement and a coincidence:

- **Build the adversarial mode per predictor.** It holds that predictor's injected eigenvalues, so
  sharing one instance across predictors attacks all of them with the last one's information.
- **Seed once, at the call site.** The set's samplers draw from the global `numpy` random state, so
  reproducibility is the caller's job; see
  [Troubleshooting](extending-faq.md#reproducibility).

The shipped set exposes two reference values that bracket any score you can get, and they are the
fastest sanity check you have: `value_upper_bound_true_transition()`, the most a predictor could
score if it knew the true transition, and `value_lower_bound_eigvec_fixed()`, the least the
shipped robust predictor guarantees. **Your family will need its own pair, or an explicit statement
that it has none**: a curve with no reference line is a number without a scale.

## Step 5: the experiment, and what "a fair comparison" means

Two routes. Pick by whether you need a *measurement* or a *check*:

| | route A: your own script | route B: the batch runner |
|---|---|---|
| shape | one process, a loop, print or plot | model table, worker processes, validated cache |
| good for | checking the machinery, a quick sweep | a sweep you will re-run, and figures for a paper |
| cost | you write the loop | you fill in a model table and respect its cache contract |
| start from | [scene 2 quickstart](https://github.com/Xuzerken/ppsds/blob/main/examples/understand_drpp_on_simulated_ambiguity/quickstart.py) | [`batch_experiment/`](https://github.com/Xuzerken/ppsds/blob/main/examples/understand_drpp_on_simulated_ambiguity/batch_experiment/) |

Route B's command line, output tree, caching and its `fork` requirement are documented once, in
[Examples](examples.md#understand_drpp_on_simulated_ambiguitybatch_experimentrunpy): read it there rather than
here. What belongs *here* is the part that is about your family rather than about the runner:

- **The model table must declare everything that determines a number.** Each model entry carries a
  `signature` that goes into the cache key. Constants bound into a callable's default arguments are
  invisible to automatic extraction, so a parameter you forget to list becomes a parameter you can
  change without the cache noticing. List them by hand.
- **A comparison is only fair if four things are held fixed**: the same sampled SDS realisations, the
  same scoring rule, the same set of trajectories, and the same initial state. The shipped setup gets
  the first by sampling *one* parameter trajectory and replaying it for every predictor, so that what
  varies between trajectories is only the noise. Vary the SDS between predictors and you have
  compared the draws, not the predictors.
- **Plot cumulative means, with your reference lines.** The shipped figure code does this for the
  shipped family; if your family has no closed-form bound yet, say so in the caption instead of
  drawing a line you cannot justify.

## Step 6: before you believe a number

The bar in this repository is that **a change must not silently move a number**. Run these in order;
the first one is the one that matters.

1. **A/B on a clean baseline.** Check out the commit you branched from in a scratch worktree, run
   the same small configurations in both trees, and compare the pickles element by element:

   ```bash
   python tools/ab_compare.py --base <the commit you branched from>
   ```

   The acceptance line is **every maximum absolute difference equal to `0.0`**: bit-identical, not
   "close". `--base` is required, deliberately: a baseline is a statement about what a change is
   allowed to affect, so it belongs at the call site. See
   [CONTRIBUTING](https://github.com/Xuzerken/ppsds/blob/main/CONTRIBUTING.md) for why that bar is
   not paranoia.
2. **The test suite.** `pytest -m "not slow and not packaging"` is the inner loop (seconds);
   `pytest` runs everything.
3. **Your own contract tests**, from Step 1 and Step 2, including the "contains its own generator"
   one.
4. **Probes when A/B reports a difference.** Fix a seed, export the arrays at the layer you suspect
   (`sample_trajectory`, `worst_case_params`, a predictor's `lamb_pred` / `Sigma_pred`), and compare
   them bit for bit. This localises a difference in minutes instead of by bisecting the run.
5. **Regenerate the golden fixture only for an intended numerical change**, and read the diff; the
   diff *is* the review:
   `python tests/regenerate_fixtures.py`.
6. **Nothing new in the core's dependencies.** A core module must not import the plotting or
   experiment periphery; see
   [Troubleshooting](extending-faq.md#the-core-and-periphery-boundary).

## Your first session

Do these in order. Steps 1–5 exist to make every later failure *your* failure instead of an
environment problem, which is the single biggest time sink in a new codebase.

```bash
# 1. Install editable, plus the experiment periphery. Drop `experiments` to get exactly what a
#    plain `pip install ppsds` gives you -- that is the configuration that catches a core
#    module which has grown a dependency on the periphery.
pip install -e ".[dev,experiments]"

# 2. Prove the install works from any directory
python -c "import ppsds; print(ppsds.__version__)"

# 3. Run the two teaching examples: ~1 second, then a few seconds
python examples/apply_drpp_to_your_own_env/quickstart.py
python examples/understand_drpp_on_simulated_ambiguity/quickstart.py

# 4. Prove the toolchain is green BEFORE you change anything
pytest -m "not slow and not packaging"

# 5. Record your baseline, then prove the A/B harness passes on a clean tree
git rev-parse --short HEAD
python tools/ab_compare.py --base <the output of the previous command>
```

Step 5 is the one people skip. Do it on the **untouched** tree: from then on, any A/B failure is
caused by your change, and you never have to ask whether the harness was broken all along.

Then write a throwaway probe: twenty lines that touch the core API and print shapes. It proves you
can import, drive the set, and read the shapes; it changes nothing:

```python
import numpy as np
from ppsds.ambiguity.additive_drift.conic_moment import LinearAmbiguitySet
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (
    EigDrpp, NoiseDrpp, NominalPredictor,
)

abg = LinearAmbiguitySet(
    np.array([[1.0, 0.1], [0.0, 1.0]]), np.identity(2), np.zeros(2),
    np.array([[1.0, 0.5], [0.5, 1.5]]),
    {'A': 0.3, 'B': 0.3, 'mu': 0.5, 'Sigma': 3}, 5.0)

np.random.seed(0)
traj = abg.sample_trajectory(T=5, period=100)
print('sample_trajectory ->', {k: v.shape for k, v in traj.items()})

x, u = np.array([1.0, 0.5]), np.array([0.2, -0.1])
for name, pred in {
    'Nominal': NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar),
    'Noise-DRPP': NoiseDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma),
    'Eig-DRPP': EigDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma),
}.items():
    pred.update(x, u)
    print(f'{name:<11} x_pred {pred.x_pred.shape}  Sigma_pred {pred.Sigma_pred.shape}  '
          f'lamb_pred {np.round(pred.lamb_pred, 4)}')

print('value_upper_bound_true_transition() =', round(abg.value_upper_bound_true_transition(), 4))
print('value_lower_bound_eigvec_fixed()    =', round(abg.value_lower_bound_eigvec_fixed(), 4))
```

Its real output, so you can tell a working install from a broken one:

```
sample_trajectory -> {'A': (6, 2, 2), 'B': (6, 2, 2), 'mu': (6, 2), 'Sigma': (6, 2, 2)}
Nominal     x_pred (2,)   Sigma_pred (2, 2)   lamb_pred [0.691 1.809]
Noise-DRPP  x_pred (2,)   Sigma_pred (2, 2)   lamb_pred [2.0729 5.4271]
Eig-DRPP    x_pred (2,)   Sigma_pred (2, 2)   lamb_pred [2.7081 5.4271]
value_upper_bound_true_transition() = -4.0481
value_lower_bound_eigvec_fixed()    = -5.8328
```

Three things to read out of it: the trajectory arrays lead with `T+1`; the three predictors agree on
`x_pred` and differ only in the covariance, which is the entire point of the package; and
`Eig-DRPP`'s eigenvalues sit between the nominal ones and the noise-inflated ones, because it inflates
only along the worst direction.

Now start at [Step 1](#step-1-the-ambiguity-set).

## Where to go next

| I want to... | read |
|---|---|
| understand what the predictors optimize, and what the guarantee costs | [Concepts](concepts.md) |
| look up the protocol my predictor must satisfy | [API](api.md) |
| see the batch runner's flags, output tree and cache | [Examples](examples.md) |
| find out why a number came out wrong, or what a warning means | [Troubleshooting](extending-faq.md) |
| know the contribution rules and the A/B bar | [CONTRIBUTING](https://github.com/Xuzerken/ppsds/blob/main/CONTRIBUTING.md) |
