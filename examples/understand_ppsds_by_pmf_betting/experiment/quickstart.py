"""The game's move, done rigorously.

Run::

    python examples/understand_ppsds_by_pmf_betting/experiment/quickstart.py        # about a second

The interactive page lets you place bets and scores them live. This is the same thing with the
training wheels off: the same systems (from ``presets.py``, which the page's data is exported
from), the same settlement rule, but over thousands of trajectories and against the one number
the game can only hint at -- the predictability **upper bound**, i.e. the best any bettor could
possibly do.

Two things this makes concrete that the page cannot:

* **A hand becomes a rule.** Bets are offsets from the nominal next state, so "where I put my
  chips" is a pmf that means the same thing at every step of every trajectory.
* **The ordinary logarithm score is not an option.** An atom pmf has no density, so scoring it
  with :class:`~ppsds.scoring_rule.logarithm_score.LogarithmScore` gives ``-inf`` for every
  outcome. The neighbourhood is what makes a sparse prediction scoreable at all -- and it is why
  there is no ``O(eps)`` approximation line here, only the upper bound.

``run.py`` is the command-line version: pick a preset, hand it a bet file exported from the
page, and get the statistics.
"""

import numpy as np

from presets import build_sds, covering_bet, default_bets, get, sample_trajectories, support_box
from ppsds.numerics.atom_pmf import AtomPmf
from ppsds.predictability.bounds import predictability_upper_bound
from ppsds.probabilistic_predictor.pmf.atom import DriftShiftedAtomPmfPredictor
from ppsds.scoring_rule.epsilon_logarithm_score import EpsilonLogarithmScore

PRESET_KEY = 'shear'
EPS = 0.5
N_TRAJ = 2000
T = 30

preset = get(PRESET_KEY)

# =====================================================================
# 1. The environment
# =====================================================================
# x_{k+1} = f(x_k) + w_k with a known noise law. `build_sds` comes from presets.py, so this is
# literally the system the page is playing -- not a lookalike.

sds = build_sds(preset)
print(f'preset {preset.key!r}: {preset.title} -- {preset.blurb}')
print(f'  noise law:   {type(preset.noise()).__name__}')
print(f'  x_0:         {preset.x0.tolist()}')
print(f'  eps offered: {preset.epsilons}   (playing at {EPS})')

# =====================================================================
# 2. The bets -- a pmf over the noise
# =====================================================================
# Atoms are offsets from the nominal next state f(x_k), so the same pmf is meaningful at every
# step. Weights are normalised by AtomPmf, so raw stakes or counts are fine.

atoms, weights = default_bets(preset, EPS)
even_spread = AtomPmf(atoms, weights)

# A second, "aimed" bet: the same positions, but weighted towards the centre. Against a noise law
# with mass everywhere, concentrating is a real strategy question -- and exactly the kind of thing
# the page makes you feel and this file makes you measure.
#
# Weighted by distance rather than "the centre atom gets more": the lattice has no atom at the
# origin, so a rule that assumes one would quietly leave the bet identical to the even spread.
aimed_weights = np.exp(-0.5 * (np.linalg.norm(atoms, axis=1) / EPS) ** 2)
aimed = AtomPmf(atoms, aimed_weights)

# The third bet is the one the page builds for you: **"Cover the table"**. Its chips are placed so
# that their epsilon-neighbourhoods tile the noise's support, which means no landing is ever
# outside every chip -- the hand cannot score -inf. That is the page's safety badge, done
# rigorously; `run.py --covering` scores the same layout from the command line.
#
# Only possible while the noise has a support to cover: an unbounded (Gaussian) noise law has
# none, so there the same button covers a window and the page reports a probability instead of a
# guarantee.
low, high = support_box(preset)
safe_atoms, safe_weights = covering_bet(preset, EPS)
safe = AtomPmf(safe_atoms, safe_weights)

print()
print('bets (offsets from the nominal next state):')
print(f'  even spread: {even_spread}')
print(f'  aimed:       {aimed}   (centre weighted up)')
print(f'  safe layout: {safe}   (tiles the noise support, so a round cannot pay nothing)')
print(f'               its tiles span {np.round(low, 3).tolist()} to {np.round(high, 3).tolist()}')

# =====================================================================
# 3. One rollout through the predictor cycle
# =====================================================================
# The library object, used the way the rest of ppsds is used: predict() -> update().

predictor = DriftShiftedAtomPmfPredictor(even_spread, preset.drift)
predictor.update(preset.x0, np.zeros(0))
print()
print(f'predict() at x_0 returns {predictor.predict()} '
      f'(mean {np.round(predictor.x_pred, 4).tolist()})')

# =====================================================================
# 4. The same bets over thousands of trajectories
# =====================================================================
# The predictor predicts the noise, so the outcome it is scored on *is* the noise draw -- the
# drift cancels, and the sweep is one vectorised pass.

_, noise = sample_trajectories(preset, N_TRAJ, T, seed=9)
flat = noise.reshape(-1, 2)
scorer = EpsilonLogarithmScore(EPS)

upper = predictability_upper_bound(preset.noise(), EPS)
print()
print(f'predictability upper bound: {upper:+.5f}   (best any bettor could do)')
print()
print(f'{"bet":>12} {"mean score":>12} {"2.5%":>10} {"97.5%":>10} {"gap to bound":>14}')
print('-' * 62)
for name, pmf in (('even spread', even_spread), ('aimed', aimed), ('safe layout', safe)):
    scores = scorer.score(pmf, flat).reshape(N_TRAJ, T)
    running = np.cumsum(scores, axis=1) / np.arange(1, T + 1)
    at_horizon = running[:, -1]
    print(f'{name:>12} {at_horizon.mean():>12.5f} '
          f'{np.quantile(at_horizon, 0.025):>10.5f} {np.quantile(at_horizon, 0.975):>10.5f} '
          f'{at_horizon.mean() - upper:>14.5f}')

# =====================================================================
# 5. How to read this
# =====================================================================
print("""
How to read this
----------------
* The **upper bound** is what the best possible bettor scores on this system. It knows nothing
  about your bets -- it is a property of the system and the radius.
* The **gap** is therefore the price of betting the way you did. It is the honest version of the
  game's "par" line.
* There is no approximation line here, unlike the Gaussian case: an atom pmf has no density, so
  the O(eps) approximation does not exist for it. `eps` is what makes these bets scoreable at
  all -- at eps = 0 every one of them is -inf.
* **Do not assume concentrating helps.** Which bet wins depends on the noise law and on the
  radius. Here the centre-weighted bet edges ahead of the even spread -- weight parked on the
  outermost atoms sits outside the box too often -- but a tighter noise law moves that the other
  way quickly. This file *measures* that trade-off; the page only lets you feel it.
* The **safe layout** is the page's "Cover the table" button, and its row is worth reading twice:
  it pays the same share every round (spread 0.00000), because a landing always falls in exactly
  one of its tiles. Its weakness is the other column -- it cannot pay more than that share, ever.

Next: `python run.py --preset rotation --eps 0.25 --n_traj 10000`, `run.py --covering` for the
safe layout without the interactive page, and `run.py --bets my_bets.json` to score a hand
exported from the page.
""")
