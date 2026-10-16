"""Drawing the results: score curves, covariance ellipses, trajectory overlays.

Part of the **experiments extra** — ``pip install "ppsds[experiments]"``. Nothing in the core
imports this subpackage, so a user who only wants predictions never pays for ``matplotlib``.

Everything here takes **paths to the pickles the experiments wrote**, not arrays, with one
exception: :func:`~ppsds.visualization.predictability.plot_predictability_panel` takes a score
array directly, which is what lets an example draw the paper's figure without running the batch
pipeline first.

Every entry point clamps its indices. An out-of-range trajectory index warns and draws the nearest
valid one rather than raising: the data has already been computed by then, and one figure should
not fail a whole experiment.
"""

from .bounds import clamp_horizon, clamp_index, clamp_len, clamp_time_indexes
from .covariance import (
    plot_ellipses_compare_predictors,
    plot_ellipses_compare_predictors_worst_case,
    plot_ellipses_over_traj,
)
from .curve import plot_error_bar, plot_traj
from .predictability import choose_trajectories, plot_predictability_panel
from .scatter import plot_target

__all__ = [
    'clamp_horizon',
    'clamp_index',
    'clamp_len',
    'clamp_time_indexes',
    'plot_ellipses_compare_predictors',
    'plot_ellipses_compare_predictors_worst_case',
    'plot_ellipses_over_traj',
    'plot_error_bar',
    'plot_traj',
    'choose_trajectories',
    'plot_predictability_panel',
    'plot_target',
]
