"""Experiment driving and validated caching -- the toolkit for *running* experiments.

| Module | Contents |
|---|---|
| :mod:`~ppsds.experiment.drpp.additive_drift.conic_moment` | The batch runner: multiprocessing workers and drivers that evaluate the predictors and run the adversarial simulation |
| :mod:`~ppsds.experiment.cache` | :class:`~ppsds.experiment.cache.CacheContext`: the validated cache |

Serves the additive-drift / conic-moment DRPP experiments, which is what
``examples/scene2_simulated_ambiguity`` runs. The path says so: **family** (DRPP) /
**assumption** (additive drift) / **family** (conic moment).

.. note::
   This subpackage and :mod:`ppsds.visualization` need dependencies from outside the
   core set (``tqdm`` / ``pandas`` / ``matplotlib`` / ``seaborn``), which are **not**
   core requirements. Install them with ``pip install ppsds[experiments]``.

   Both were merged in from ``src/lab/`` (2026-10-12), because ``lab`` did not ship with
   the package and therefore ``examples/scene2`` could not run under a plain
   ``pip install ppsds``.
"""
