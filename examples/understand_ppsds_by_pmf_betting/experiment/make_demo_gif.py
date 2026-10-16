"""Render the README animation: the game, drawn from the same world the page plays.

Run::

    python examples/understand_ppsds_by_pmf_betting/experiment/make_demo_gif.py          # writes public/docs/assets/demo.gif

The animation is generated here rather than screen-recorded, which trades fidelity for
repeatability: it needs no browser, it is deterministic, and it cannot show a game that the
exported assets no longer produce. What it shows is the *idea* -- place chips, settle, watch the
running average climb towards par -- not the page's user interface.

Being a GIF rather than a PNG is now a **choice**, not a constraint. Until 2026-10-16 it was a
constraint -- ``.png`` was excluded from the export everywhere, ``public/`` included, so a still
image would silently never reach the public repository. The exporter now carves ``public/**`` out
of that exclusion (``PUBLIC_ALLOWED_SUFFIXES``), so static figures such as
``prediction_on_your_system.png`` do ship. This one stays a GIF because the point of it is the
**motion**: the score climbing towards par round after round is not visible in a still.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from presets import get, sample_trajectories
from ppsds.numerics.atom_pmf import AtomPmf
from ppsds.predictability.bounds import predictability_upper_bound
from ppsds.scoring_rule.epsilon_logarithm_score import EpsilonLogarithmScore

EXAMPLE_ROOT = Path(__file__).resolve().parent
#: ``parents[2]``: the repository root is two levels up from
#: ``examples/<story>/experiment/``. The index is load-bearing and fails silently -- see the
#: same note in ``export_game_assets.py``.
DEFAULT_OUT = EXAMPLE_ROOT.parents[2] / 'public' / 'docs' / 'assets' / 'demo.gif'

PRESET = 'shear'
EPS = 0.5
STEPS = 16
TRAJECTORY = 3


def chips_for(eps):
    """A hand of bets: a small lattice of offsets from the nominal next state."""
    axis = np.linspace(-1.2 * eps, 1.2 * eps, 3)
    atoms = np.array([[a, b] for a in axis for b in axis])
    return AtomPmf(atoms, np.full(len(atoms), 1.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--fps', type=int, default=5, help='frames per second (default: %(default)s)')
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        raise SystemExit(
            'matplotlib is required to render the animation: pip install ppsds[experiments]')

    preset = get(PRESET)
    states, noise = sample_trajectories(preset, TRAJECTORY + 1, STEPS, seed=1234)
    x = states[TRAJECTORY]
    w = noise[TRAJECTORY]

    pmf = chips_for(EPS)
    atoms = pmf.atoms
    weights = pmf.weights
    scorer = EpsilonLogarithmScore(EPS)
    scores = scorer.score(pmf, w)
    running = np.cumsum(scores) / np.arange(1, STEPS + 1)
    par = predictability_upper_bound(preset.noise(), EPS)

    matrix = np.array([[preset.drift(np.eye(2)[i], np.zeros(0))[j] for i in range(2)]
                       for j in range(2)])

    figure, (board, chart) = plt.subplots(1, 2, figsize=(8.6, 4.3), width_ratios=[1.25, 1])
    figure.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.11)

    def draw(step, settled):
        """One frame. ``step`` is the index; ``settled`` draws the outcome as well as the chips."""
        board.clear()
        chart.clear()
        here, outcome = x[step], matrix @ x[step]
        # The window follows the action rather than being fixed. A shear system drifts: by step
        # ten it is at x ~ 4.4 and its chips sit at x ~ 4.8, so a fixed window would draw the
        # chips -- the entire content of the animation -- off the edge of the picture.
        span = max(3.0, 0.5 * float(np.max(np.abs(outcome - here))) + 2 * EPS + 1.5)
        centre = 0.5 * (here + outcome)
        board.set_xlim(centre[0] - span, centre[0] + span)
        board.set_ylim(centre[1] - span, centre[1] + span)
        board.set_aspect('equal')
        board.set_title(f'{preset.title}   eps = {EPS}   step {step + 1}/{STEPS}', fontsize=10)

        # The path so far, so the system is visibly a trajectory and not a dot.
        board.plot(x[:step + 1, 0], x[:step + 1, 1], color='#c9d4e5', linewidth=1.0, zorder=1)
        board.plot(*here, 'o', color='#12151c', markersize=6, zorder=4)

        if EPS > 0:
            board.add_patch(plt.Rectangle(outcome - EPS, 2 * EPS, 2 * EPS,
                                          fill=False, edgecolor='#7dd3fc',
                                          linestyle='--', linewidth=1.0, zorder=2))

        landed = w[step]
        inside = np.max(np.abs(atoms - landed), axis=1) <= EPS
        colours = ['#4ade80' if hit else '#f87171' for hit in inside] if settled else ['#7dd3fc'] * len(atoms)
        placed = atoms + outcome
        board.scatter(placed[:, 0], placed[:, 1], s=40 + 260 * weights, c=colours, alpha=0.85,
                      edgecolors='white', linewidths=0.8, zorder=3)

        if settled:
            board.plot(*(outcome + landed), 'o', color='#fca5a5', markersize=8, zorder=5)

        board.set_xlabel('state', fontsize=9)
        board.tick_params(labelsize=8)

        chart.axhline(par, color='#fca5a5', linestyle=':', linewidth=1.6, label='par (upper bound)')
        if settled:
            shown = step + 1
            chart.plot(np.arange(1, shown + 1), running[:shown],
                       color='#2b6cb0', linewidth=2.0, label='your running average')
        chart.set_xlim(0, STEPS)
        chart.set_ylim(min(par, running.min()) - 0.6, max(running) + 0.6)
        chart.set_xlabel('step', fontsize=9)
        chart.set_ylabel('epsilon-logarithm score', fontsize=9)
        chart.tick_params(labelsize=8)
        chart.legend(fontsize=8, loc='lower right')
        chart.grid(alpha=0.25)

        figure.suptitle('ppsds: predict, then settle', fontsize=13)
        return []

    # Two frames per step -- chips placed, then settled -- so the payout is legible.
    def frame(index):
        step, settled = divmod(index, 2)
        return draw(step, settled)

    animation = FuncAnimation(figure, frame, frames=2 * STEPS, blit=False, interval=1000 / args.fps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # `writer='pillow'` rather than a writer *instance*: `save` has to set the writer up itself,
    # and an instance passed in is never initialised, which shows up as an empty frame list and a
    # bare IndexError from `finish()`.
    animation.save(args.out, writer='pillow', fps=args.fps)
    plt.close(figure)
    print(f'wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KiB, {2 * STEPS} frames)')


if __name__ == '__main__':
    main()
