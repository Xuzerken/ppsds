"""ppsds — probabilistic prediction for stochastic dynamical systems.

`ppsds` predicts the next state of a stochastic dynamical system (SDS) as a **predictive
distribution** rather than a point, and ships predictors that stay honest when your model of the
system is wrong. Predictors come in families, and the package is meant to hold more than one;
this release implements the **distributionally robust** family (DRPP) over conic-moment ambiguity
sets, which additionally guarantees that — as long as the true system lies within the ambiguity
set you declared — the expected score of the prediction under the worst case does not fall below
some lower bound.

Two usage scenarios
-------------------
**Scene 1: apply DRPP to your own gymnasium environment** ("can use")
    You only need to declare the nominal model and the ambiguity set, then let the predictor
    run. :mod:`ppsds.environment` is **not needed** — that belongs to Scene 2.

    .. code-block:: python

        from ppsds.ambiguity.additive_drift.conic_moment import LinearAmbiguitySet
        from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import EigDrpp
        from ppsds.scoring_rule.logarithm_score import LogarithmScore

**Scene 2: custom simulated ambiguity** ("can understand" / reproduce the paper)
    Additionally use :class:`ppsds.environment.Env` and the various modes in
    :mod:`ppsds.environment.dynamics_mode`.

Submodules
----------
========================================  ==============================================
:mod:`ppsds.ambiguity`                    ambiguity sets, layered by "assumption / family"
                                          (``additive_drift/conic_moment``)
:mod:`ppsds.dynamics`                     the object being predicted: robotic-arm models + system identification
:mod:`ppsds.probabilistic_predictor`      probabilistic predictors (the paper's core algorithmic contribution)
:mod:`ppsds.scoring_rule`                 logarithmic scoring rule
:mod:`ppsds.environment`                  Gymnasium environment layer (**serves Scene 2 only**)
:mod:`ppsds.numerics`                     small numerical utilities (``linalg`` / ``random_variables``)
:mod:`ppsds.experiment`                   experiment driver and validating cache (needs the ``experiments`` extra)
:mod:`ppsds.visualization`                plotting (needs the ``experiments`` extra)
:mod:`ppsds.control_policy`               small control utilities (``DLQR``)
========================================  ==============================================

Usage and further notes: see ``README.md`` and ``docs/tutorials/``.
"""

__version__ = "0.0.1"

# The minimal public surface for Scene 1. Scene 2 things (Env / the various DynamicsMode)
# are imported explicitly from their submodules, so that people who only want the predictor
# are not forced to pull in gymnasium.
from .probabilistic_predictor.drpp.additive_drift.conic_moment import (
    NominalPredictor,
    NoiseDrpp,
    EigDrpp,
    eigvec_fixed_minmax,
)
from .scoring_rule.logarithm_score import LogarithmScore
from .ambiguity.additive_drift.conic_moment import (
    AdditiveDriftConicMomentSet,
    LinearAmbiguitySet,
    NonlinearAmbiguitySet,
)

__all__ = [
    "__version__",
    "AdditiveDriftConicMomentSet",
    "LinearAmbiguitySet",
    "NonlinearAmbiguitySet",
    "NominalPredictor",
    "NoiseDrpp",
    "EigDrpp",
    "eigvec_fixed_minmax",
    "LogarithmScore",
]
