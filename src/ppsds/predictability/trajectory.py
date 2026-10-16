"""Aggregating per-step scores into the curves the predictability figures plot.

A rollout produces a score per step per trajectory, so a ``(n_traj, T)`` array. Two shapes get
drawn from it:

- :func:`cumulative_average` -- the running mean along the step axis, giving one curve per
  trajectory. This is the curve that converges to the expected score, at rate
  ``O_p(T ** -0.5)`` when the process noise is i.i.d. -- which is why a *single* trajectory is
  enough to estimate the expectation, and why these figures do not need repeated samplings.
- :func:`across_trajectories` -- per-step mean and quantiles over trajectories, giving the
  expected-score curve and its confidence band.

Both are numpy-only and live in the core. :func:`cumulative_average` is the same operation as
``ppsds.visualization.data_transform.cum_average``, which now delegates here so that the core
and the plotting periphery cannot drift apart.
"""

import numpy as np


def cumulative_average(scores):
    """Running mean along the step axis of a ``(n_traj, T)`` score block.

    Entry ``[n, k]`` is the mean of ``scores[n, 0:k+1]``. The first step is taken as-is, which
    matters for a one-step prediction: there the score *is* the average.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(
            f'scores must be (n_traj, T), got shape {scores.shape}. A single trajectory should '
            'be passed as (1, T).'
        )
    averaged = np.cumsum(scores, axis=1)
    for step in range(averaged.shape[1]):
        averaged[:, step] /= step + 1
    return averaged


def across_trajectories(scores, quantiles=(0.025, 0.975)):
    """Per-step mean and quantiles over trajectories.

    Parameters
    ----------
    scores : array_like
        ``(n_traj, T)`` -- typically already passed through :func:`cumulative_average`.
    quantiles : sequence of float, optional
        Levels to compute per step. The default gives the paper's 95% band.

    Returns
    -------
    dict
        ``{'mean': (T,), 'quantile_levels': tuple, 'low': (T,), 'high': (T,)}``. With a single
        trajectory the quantiles equal the mean, so the band collapses rather than raising --
        the figures stay drawable while a rollout is still short.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f'scores must be (n_traj, T), got shape {scores.shape}')
    levels = tuple(float(level) for level in quantiles)
    if not levels:
        raise ValueError('quantiles must name at least one level')
    for level in levels:
        if not 0.0 <= level <= 1.0:
            raise ValueError(f'quantile levels must lie in [0, 1], got {level}')
    low, high = np.quantile(scores, [levels[0], levels[-1]], axis=0)
    return {
        'mean': scores.mean(axis=0),
        'quantile_levels': levels,
        'low': low,
        'high': high,
    }
