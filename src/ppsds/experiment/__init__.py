"""Running experiments at scale: the batch driver and its validated cache.

Part of the **experiments extra** — ``pip install "ppsds[experiments]"``. This is not the library
API: it is the machinery behind the batch halves of the examples, and the reason
``examples/understand_drpp_on_simulated_ambiguity`` can produce numbers at all.

| Module | Contents |
|---|---|
| :mod:`~ppsds.experiment.drpp.additive_drift.conic_moment` | The batch runner: multiprocessing workers and drivers that evaluate the predictors and run the adversarial simulation |
| :mod:`~ppsds.experiment.cache` | :class:`~ppsds.experiment.cache.CacheContext`: the validated cache |

Serves the additive-drift / conic-moment DRPP experiments, which is what
``examples/understand_drpp_on_simulated_ambiguity`` runs. The path says so: **family** (DRPP) /
**assumption** (additive drift) / **family** (conic moment).

.. note::
   This subpackage and :mod:`ppsds.visualization` need dependencies from outside the
   core set (``tqdm`` / ``pandas`` / ``matplotlib`` / ``seaborn``), which are **not**
   core requirements. Install them with ``pip install ppsds[experiments]``.

   Both were merged in from ``src/lab/`` (2026-10-12), because ``lab`` did not ship with
   the package and therefore the scene-2 example could not run under a plain
   ``pip install ppsds``.
"""
