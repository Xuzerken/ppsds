"""Drawing the results: score curves, covariance ellipses, trajectory overlays.

Part of the **experiments extra** — ``pip install "ppsds[experiments]"``. Nothing in the core
imports this subpackage, so a user who only wants predictions never pays for ``matplotlib``.

⚠️ **This module deliberately re-exports nothing**, unlike every other subpackage of ``ppsds``.
Its submodules do not share a dependency floor — ``bounds``, ``scatter`` and ``graph_format`` are
numpy-only, while ``covariance``, ``curve`` and ``predictability`` import ``matplotlib`` at module
level. An eager ``__init__`` that imported all of them would make ``import
ppsds.visualization.bounds`` fail *without* the extra, which is a property that used to hold and
that ``tests/test_regressions.py`` relies on.

The rule this is an instance of, worth stating once: **an eager re-export raises the import floor
of every submodule to the highest one in the package.** That is fine where all the submodules cost
the same (``ambiguity``, ``dynamics``, ``numerics``, ``predictability``,
``probabilistic_predictor``, ``scoring_rule``, ``control_policy`` — all numpy/scipy), and wrong
here. Reach for the full path, or for one of these::

    from ppsds.visualization.bounds import clamp_index
    from ppsds.visualization.predictability import plot_predictability_panel

Every entry point clamps its indices. An out-of-range trajectory index warns and draws the nearest
valid one rather than raising: the data has already been computed by then, and one figure should
not fail a whole experiment.
"""
