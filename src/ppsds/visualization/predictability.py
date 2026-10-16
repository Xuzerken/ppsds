"""The predictability figure: per-step score curves against the two reference lines.

One panel shows, for a single neighbourhood radius:

- a few **individual trajectories'** running averages (thin) -- the point being that they
  converge to the expectation within a couple of dozen steps, so one trajectory is enough;
- the **expected score** over many trajectories (solid) with a confidence band;
- the **predictability upper bound** (dashed), and optionally the ``O(eps)``
  **approximation** for the given predictor (dashed).

Both reference values are **passed in**, never recomputed here. They come from
:mod:`ppsds.predictability.bounds`, and computing them inside a plotting function would
both hide that dependency and invite the figure to disagree with the number a test asserts.

The approximation line is optional on purpose: an atom/pmf predictor has no cross-entropy
against a continuous truth (see
:func:`~ppsds.predictability.entropy.cross_entropy`), so for those panels there is no
approximation to draw, only the upper bound and the score itself.

Which trajectories get drawn
----------------------------
By default, **three chosen at random per panel**, not "the first three". This is not a detail:
drawing the same three trajectories in every panel makes it look as though the individual
curves were a property of the radius, when they are a property of the sample. The choice is
seeded from ``eps`` so a figure is reproducible without being identical across panels.
"""

from __future__ import annotations

import numpy as np

from ..predictability.trajectory import across_trajectories, cumulative_average
from .bounds import clamp_index

#: Colours for the individual-trajectory curves. Seaborn's ``Set2``: muted enough to sit under
#: the reference lines without competing with the solid expected-score curve, which is the one
#: curve on the panel that is not a sample.
_TRAJECTORY_COLOURS = ('#fc8d62', '#8da0cb', '#66c2a5')

#: How many individual trajectories to draw when the caller does not say.
DEFAULT_N_CURVES = 3

#: The two reference lines and the expected-score curve, thick and dashed/solid exactly as the
#: published figure has them.
_REFERENCE_LINEWIDTH = 3.0
_EXPECTED_LINEWIDTH = 3.0


def choose_trajectories(n_traj, eps, count=DEFAULT_N_CURVES):
    """``count`` distinct trajectory indices, differing between radii but reproducible.

    Seeded from ``eps`` so that every panel draws its own sample and a re-run draws the same
    figure. Fewer trajectories than requested degrades to all of them, in order -- a short
    rollout should thin the figure, not break it.
    """
    if n_traj <= 0:
        raise ValueError('no trajectories to choose from')
    if count >= n_traj:
        return tuple(range(n_traj))
    seed = int(round(float(eps) * 1_000_000))
    return tuple(int(i) for i in np.random.default_rng(seed).choice(n_traj, count, replace=False))


def plot_predictability_panel(ax, scores, *, eps, upper_bound, approximation=None,
                              trajectory_indexes=None, draw_band=True, legend=True):
    """Draw one panel from a ``(n_traj, T)`` block of per-step scores.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    scores : array_like
        ``(n_traj, T)`` -- the *per-step* epsilon-logarithm scores, not yet averaged.
    eps : float
        The neighbourhood radius. It also seeds
        :func:`choose_trajectories` when ``trajectory_indexes`` is not given.
    upper_bound : float
        Predictability upper bound, drawn as the dashed red line.
    approximation : float, optional
        The ``O(eps)`` approximation for this predictor, drawn as the dashed blue line.
        ``None`` omits it, which is the pmf case.
    trajectory_indexes : sequence of int, optional
        Which individual trajectories to draw thin. Defaults to
        :func:`choose_trajectories` -- **three at random for this radius**, so panels do not
        share a sample. Indices are clamped into range, warning if changed, rather than
        raising: a short rollout should degrade the figure, not the run.
    draw_band : bool
        Whether to shade the confidence band between the two quantiles.
    legend : bool
        Whether to draw the legend, whose four entries are the published figure's.

    Returns
    -------
    dict
        ``{'mean', 'low', 'high'}``, each ``(T,)`` -- the aggregates actually drawn, so a
        caller can print the numbers it just plotted instead of recomputing them.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f'scores must be (n_traj, T), got shape {scores.shape}')
    n_traj, n_steps = scores.shape
    if n_traj == 0 or n_steps == 0:
        raise ValueError('scores is empty; there is nothing to plot')

    steps = np.arange(1, n_steps + 1)
    running = cumulative_average(scores)

    if trajectory_indexes is None:
        trajectory_indexes = choose_trajectories(n_traj, eps)
    for position, index in enumerate(trajectory_indexes):
        safe = clamp_index(index, n_traj)
        ax.plot(steps, running[safe],
                color=_TRAJECTORY_COLOURS[position % len(_TRAJECTORY_COLOURS)],
                linewidth=1.8, zorder=2)

    aggregate = across_trajectories(running)

    # Created in the published figure's legend order: predictability, approximation, expected
    # score, 95% CI. `zorder` decides the stacking independently of creation order.
    ax.axhline(upper_bound, color='tab:red', linestyle='--', linewidth=_REFERENCE_LINEWIDTH,
               label='predictability', zorder=4)
    if approximation is not None:
        ax.axhline(approximation, color='tab:blue', linestyle='--',
                   linewidth=_REFERENCE_LINEWIDTH, label='approximation', zorder=4)
    ax.plot(steps, aggregate['mean'], color='tab:blue', linewidth=_EXPECTED_LINEWIDTH,
            label='expected score', zorder=3)
    if draw_band:
        ax.fill_between(steps, aggregate['low'], aggregate['high'], color='tab:blue',
                        alpha=0.15, linewidth=0, label='95% CI', zorder=1)
    if legend:
        ax.legend(loc='upper right')

    ax.set_xlim(1, n_steps)
    ax.set_xlabel('time step')
    ax.set_ylabel(r'$\epsilon$-logarithm score')
    ax.grid(alpha=0.4)
    return aggregate
