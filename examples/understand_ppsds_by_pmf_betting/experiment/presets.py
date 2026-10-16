"""The systems the game and the experiments share -- **the single source of truth**.

The interactive page under ``public/docs/game/`` and the rigorous experiment in this directory
must be the *same world*, or the funnel's第三步 ("what I did in the game, done properly") is a
lookalike rather than a rerun. So the systems live here, in Python, and the page's data is
**exported** from this file by ``export_game_assets.py`` -- the page never carries a second copy
of a drift matrix.

What a preset is
----------------
An additive-drift system ``x_{k+1} = f(x_k) + w_k`` with a **known** noise law (see
:class:`~ppsds.dynamics.known_sds.AdditiveDriftSds`), plus the neighbourhood radii the game
offers for it.

Bets are in the **noise frame**
------------------------------
A player's bet is a set of atoms with weights, and where those atoms live matters. They are
defined as offsets from the **nominal next state** ``f(x_k)`` -- i.e. a pmf over the noise --
for two reasons:

* it is the reduction the predictability paper uses, so the experiment reuses
  :class:`~ppsds.probabilistic_predictor.pmf.atom.DriftShiftedAtomPmfPredictor` directly;
* it makes a bet **portable**: the same atoms mean the same thing at every step and on every
  trajectory, which is exactly what turns "one hand I played" into "a rule I can evaluate over
  10,000 trajectories".

The page draws the current state and its nominal next state, so betting around that marker is
the natural gesture.

What the page shows about a table
---------------------------------
The page has a "table card" that states, in words, what the drift does (:attr:`Preset.motion`)
next to an animated picture of it -- the nominal dynamics are *given* in this setting, so the
page shows them rather than making the player infer them. The card is also where
:func:`covering_bet` shows up as the "safe" layout: a bet whose neighbourhoods cover the noise's
whole support, so no landing can pay nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_normal

from ppsds.numerics.random_variables import multivariate_uniform

from noise_laws import BlockNoise


@dataclass(frozen=True)
class Preset:
    """One system, plus the radii the game offers for it.

    Attributes
    ----------
    key : str
        Stable identifier; it is what a bet file records, so renaming one invalidates saved bets.
    title : str
        Short human name, shown in the game's picker.
    blurb : str
        One sentence on what the system does, for the page.
    motion : str
        The drift in plain words, for the page's table card. Kept apart from ``blurb`` because
        they answer different questions: ``blurb`` says why you would pick this table,
        ``motion`` says what it does to a puck -- and a reader who wants to place chips well
        needs the second one.
    drift : callable
        ``f(x, u) -> (2,)``. Every preset is two-dimensional: the game is a 2-D canvas.
    noise : callable
        A factory returning a fresh noise law (fresh, because the law objects are mutable-ish
        and sampling from a shared one would couple the game and the experiment).
    x0 : ndarray
        Where a trajectory starts.
    epsilons : tuple of float
        The neighbourhood radii offered, smallest first. ``eps`` is the game's difficulty dial.
    hidden_noise : bool
        Whether the page withholds this table's noise law. The table is then played blind: no
        prose about the landing law, no safety verdict, and the safe layout is off (it would be
        built from the very support the player is not told). The *drift* stays public -- the puck's
        motion is visible on the table anyway -- and the scoring is untouched, so the mathematics
        is as pinned as ever; what is withheld is information, not correctness.
    """

    key: str
    title: str
    blurb: str
    motion: str
    drift: object
    noise: object
    x0: np.ndarray
    epsilons: tuple
    hidden_noise: bool = False


def _shear(x, u):
    """A shear: the paper's system. Stretches along one direction only."""
    return np.array([[1.0, 1.0], [0.0, 1.0]]) @ np.asarray(x, dtype=float)


def _rotation(x, u):
    """A rotation by 15 degrees: the state orbits the origin instead of running away."""
    angle = np.deg2rad(15.0)
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    return rot @ np.asarray(x, dtype=float)


def _damped(x, u):
    """A contraction: the state spirals into the origin, so prediction gets easier over time."""
    return 0.9 * np.array([[np.cos(0.3), -np.sin(0.3)],
                           [np.sin(0.3), np.cos(0.3)]]) @ np.asarray(x, dtype=float)


#: The registry. Order is the order the game shows them.
#:
#: The names are the game's, not the library's: the page is a casino, so a table is called what
#: it does to a puck. The ``key`` is the exception and is **frozen** -- it is what a bet file
#: records, so renaming one invalidates every saved hand.
PRESETS = {
    'shear': Preset(
        key='shear',
        title='Long Table',
        blurb='A puck sliding down a long table: each round it drifts further along the rail, so '
              'the rounds spread out. This is the system from the predictability paper.',
        motion='Each round the puck slides along the rail by its own distance from the rail -- '
               'the further off the rail it sits, the further along it slides.',
        drift=_shear,
        noise=lambda: multivariate_uniform(low=-np.ones(2), high=np.ones(2)),
        x0=np.zeros(2),
        epsilons=(0.25, 0.5, 0.75),
    ),
    'rotation': Preset(
        key='rotation',
        title='Roulette',
        blurb='The table turns 15 degrees every round and the landing wobbles, so the puck '
              'keeps to a circle around the centre.',
        motion='Each round the table turns another 15 degrees, so the puck goes round the '
               'centre at a fixed distance from it.',
        drift=_rotation,
        noise=lambda: multivariate_normal(mean=np.zeros(2), cov=0.25 * np.eye(2)),
        x0=np.array([1.0, 0.0]),
        epsilons=(0.25, 0.5, 0.75),
    ),
    'damped': Preset(
        key='damped',
        title='Whirlpool',
        blurb='The table spirals inward, so each round is a smaller step than the last and the '
              'action crowds into a shrinking region.',
        motion='Each round the table turns by about 17 degrees and shrinks to nine tenths of '
               'its size, so the puck spirals in and every round is a smaller one.',
        drift=_damped,
        noise=lambda: multivariate_uniform(low=-0.3 * np.ones(2), high=0.3 * np.ones(2)),
        x0=np.array([1.5, 0.0]),
        epsilons=(0.1, 0.2, 0.4),
    ),
    # The two biased tables. Everything below is the same machinery with a noise law that is not
    # uniform, which is the point: covering the table stays *safe* but stops being the best you
    # can do, so the player has something to work out.
    'stacked_deck': Preset(
        key='stacked_deck',
        title='Stacked Deck',
        blurb='The Long Table with a stacked deck: landings pile up in the middle band and thin '
              'out towards the rails.',
        motion='Each round the puck slides along the rail by its own distance from the rail -- '
               'the further off the rail it sits, the further along it slides.',
        drift=_shear,
        noise=lambda: BlockNoise(
            boxes=[((-1.0, -1.0), (1.0, -1.0 / 3.0)),
                   ((-1.0, -1.0 / 3.0), (1.0, 1.0 / 3.0)),
                   ((-1.0, 1.0 / 3.0), (1.0, 1.0))],
            weights=[0.10, 0.80, 0.10],
        ),
        x0=np.zeros(2),
        epsilons=(0.25, 0.5, 0.75),
    ),
    'house_rules': Preset(
        key='house_rules',
        title='House Rules',
        blurb='Four corners, four different odds, and the house is not saying which is which. '
              'The payout radius is the only thing you are told.',
        motion='Each round the table turns another 15 degrees, so the puck goes round the centre '
               'at a fixed distance from it.',
        drift=_rotation,
        noise=lambda: BlockNoise(
            boxes=[((-1.0, 0.0), (0.0, 1.0)),
                   ((0.0, 0.0), (1.0, 1.0)),
                   ((-1.0, -1.0), (0.0, 0.0)),
                   ((0.0, -1.0), (1.0, 0.0))],
            weights=[0.60, 0.20, 0.15, 0.05],
        ),
        x0=np.array([1.0, 0.0]),
        epsilons=(0.25, 0.5, 0.75),
        hidden_noise=True,
    ),
}

DEFAULT_PRESET = 'shear'


def get(key):
    """Look a preset up, failing loudly -- a typo in ``--preset`` should not silently pick one."""
    if key not in PRESETS:
        raise KeyError(f'unknown preset {key!r}; available: {sorted(PRESETS)}')
    return PRESETS[key]


def build_sds(preset):
    """An :class:`~ppsds.dynamics.known_sds.AdditiveDriftSds` for ``preset``."""
    from ppsds.dynamics.known_sds import AdditiveDriftSds

    return AdditiveDriftSds(preset.drift, preset.noise(), x0=preset.x0)


#: Atoms per side in the fallback bet. Four, giving a spacing of ``4 eps / 3``.
DEFAULT_BETS_PER_SIDE = 4


def default_bets(preset, eps):
    """A plain baseline bet: a uniform grid of atoms spanning the noise's plausible range.

    This is what the experiment falls back to when no bet file is given -- a "player" who spread
    effort evenly rather than aiming. It is deliberately not a good bet: the point of comparing
    against it is to see what aiming buys.

    The spacing matters more than it looks, and the number is chosen to avoid an integer ratio:
    a box is ``2 eps`` wide, so if the lattice spacing divides that exactly, the box catches the
    same number of atoms at *every* alignment, and the score comes out **constant** -- zero
    spread across trajectories, which is useless as a baseline and looks like a bug. At
    ``4 eps / 3`` the ratio is 1.5 and the count genuinely varies.
    """
    axis = np.linspace(-2.0 * eps, 2.0 * eps, DEFAULT_BETS_PER_SIDE)
    atoms = np.array([[a, b] for a in axis for b in axis])
    return atoms, np.full(len(atoms), 1.0 / len(atoms))


#: How far out the "safe" layout reaches when the noise is unbounded, in standard deviations.
#: A Gaussian can land anywhere, so there is no support to cover and no guarantee to be had;
#: this is the window the page's safety badge reports a catch probability over.
SAFE_WINDOW_SIGMA = 3.0


def support_box(preset):
    """``(low, high)`` of the noise's support box, or ``None`` when the noise is unbounded."""
    law = preset.noise()
    low, high = getattr(law, 'low', None), getattr(law, 'high', None)
    if low is None or high is None:
        return None
    return np.asarray(low, dtype=float).reshape(-1), np.asarray(high, dtype=float).reshape(-1)


def covering_bet(preset, eps, *, reach=None):
    """The **safe** layout: atoms whose neighbourhoods tile a box, so nothing there pays zero.

    Tiling is what makes it safe. With the boxes exactly ``2 eps`` wide and their centres no
    further than ``2 eps`` apart, neighbouring boxes touch, so every point of the box lies in
    at least one of them -- and a landing in that box always pays.

    What the box is depends on the noise, and so does what "safe" means:

    * **bounded** (uniform) noise: the box is the noise's own support, so the guarantee is real
      -- no landing can score ``-inf``. When the support's half-width is a whole multiple of
      ``eps`` the boxes tile it exactly, every landing pays the same share ``1 / n_atoms``, and
      having more chips no longer buys anything;
    * **unbounded** (Gaussian) noise: there is nothing to cover, so ``reach`` must be given and
      the tiling covers ``mean +- reach`` -- a chosen window, not a guarantee. The page shows the
      probability of catching a landing there instead of a worst case.

    The chips are one per tile centre with equal weights, so the layout is a pmf.
    """
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(
            f'a covering layout needs a positive radius, got {eps!r}: with eps = 0 the boxes '
            'have no area and no layout can guarantee a payout')

    box = support_box(preset)
    if box is None:
        if reach is None:
            raise ValueError(
                f'preset {preset.key!r} has unbounded noise, so a covering layout can only cover '
                'a chosen window -- pass reach=<half-width> to say which')
        mean = np.asarray(preset.noise().mean, dtype=float).reshape(-1)
        reach = np.broadcast_to(np.asarray(reach, dtype=float).reshape(-1), mean.shape)
        low, high = mean - reach, mean + reach
    else:
        low, high = box

    centre = 0.5 * (low + high)
    half = 0.5 * (high - low)
    # `ceil(half / eps)` tiles at a spacing of at most `2 eps`, which is what makes the boxes
    # touch. One tile is enough when the box is no wider than a box -- and then the centre is
    # the box's own centre, not `linspace`'s first point (which is what a one-element
    # `linspace` would hand back, and it would sit off to one side).
    counts = np.maximum(1, np.ceil(half / eps).astype(int))
    axes = [np.array([c]) if n == 1 else c + np.linspace(-(h - eps), h - eps, n)
            for c, h, n in zip(centre, half, counts)]
    atoms = np.array(np.meshgrid(*axes, indexing='ij')).reshape(2, -1).T
    return atoms, np.full(len(atoms), 1.0 / len(atoms))


def covering_meta(preset, eps, *, reach=None):
    """What the safe layout achieves on this table -- the numbers the page's badge shows."""
    from bet_coverage import uncovered_report

    atoms, weights = covering_bet(preset, eps, reach=reach)
    meta = {
        'key': preset.key,
        'eps': float(eps),
        'n_atoms': int(len(atoms)),
        'kind': 'gaussian' if support_box(preset) is None else 'bounded',
    }
    box = support_box(preset)
    if box is None:
        # Over the whole plane the worst case is a landing arbitrarily far away, which pays
        # nothing -- so it is zero for *every* hand. The useful number for unbounded noise is
        # `bet_coverage.gaussian_miss_probability`, which the page computes from the layout.
        meta['guaranteed'] = False
        meta['worst_case_roi'] = 0.0
        return meta
    report = uncovered_report(atoms, weights, box[0], box[1], eps)
    meta['guaranteed'] = not report['possible_neg_inf']
    meta['worst_case_roi'] = report['worst_case_roi']
    return meta


#: Seed for the exported fixtures. The page's rollouts and the self-check cases both come from
#: here, so "the trajectory the player is looking at" and "the trajectory ppsds scores" are the
#: same draw -- one number, owned by this module, is what makes them so.
EXPORT_SEED = 20261014

#: Steps in an exported fixture trajectory -- what the self-check vectors are built from.
FIXTURE_STEPS = 30

#: Steps in an exported **page** replay. Deliberately its own constant: a run's length is the
#: player's choice, so the replays ship long enough for the longest one, while the self-check
#: vectors keep their own length (lengthening those would rewrite every golden for nothing).
#:
#: It is shared with `run.py`, and it has to be: the draws a trajectory contains depend on how
#: many were asked for in one call (``scipy`` draws a whole axis at a time), so a 30-step
#: generation is **not** the first 30 steps of a 50-step one. Everything that reproduces a replay
#: therefore generates exactly this many and then slices.
EXPORT_STEPS = 50


def fixture_trajectory(preset, n_step=FIXTURE_STEPS):
    """One trajectory's ``(states, noise)``, as the exported fixtures use it.

    The seed is derived from the preset's position in the registry, so nothing else has to
    hard-code the mapping between a preset and "its" trajectory.
    """
    index = list(PRESETS).index(preset.key)
    states, noise = sample_trajectories(preset, 1, n_step, seed=EXPORT_SEED + index)
    return states[0], noise[0]


def sample_trajectories(preset, n_traj, n_step, seed):
    """``(n_traj, n_step + 1, 2)`` states and ``(n_traj, n_step, 2)`` noise draws.

    Reproducible from ``seed``: an explicit ``Generator`` per trajectory, so the same
    ``(preset, n_traj, n_step, seed)`` is the same world however it is generated.
    """
    sds = build_sds(preset)
    root = np.random.default_rng(seed)
    seeds = root.integers(0, 2 ** 31 - 1, size=n_traj)

    states = np.empty((n_traj, n_step + 1, 2))
    noise = np.empty((n_traj, n_step, 2))
    for index, trajectory_seed in enumerate(seeds):
        rollout = sds.sample_trajectory(n_step, rng=np.random.default_rng(int(trajectory_seed)))
        states[index] = rollout['x']
        noise[index] = rollout['w']
    return states, noise
