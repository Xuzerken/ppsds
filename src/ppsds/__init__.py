"""ppsds — probabilistic prediction for stochastic dynamical systems.

`ppsds` predicts the next state of a stochastic dynamical system (SDS) as a **predictive
distribution** rather than a point, and ships predictors that stay honest when your model of the
system is wrong. DRPP -- *distributionally robust probabilistic prediction* -- is one **kind** of
probabilistic predictor, not the definition of the library: **for each family of ambiguity sets
there is a DRPP of its own**. This release implements the one for **conic-moment** ambiguity sets,
on an **additive-drift** system; it additionally guarantees that -- as long as the true system
lies within the ambiguity set you declared -- the expected score under the worst case does not
fall below some lower bound.

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
:mod:`ppsds.dynamics`                     the object being predicted: robotic-arm models, system identification, and a known system (``known_sds``)
:mod:`ppsds.probabilistic_predictor`      probabilistic predictors, one subpackage per **kind**: ``drpp`` (one per family of ambiguity sets) and ``pmf`` (predicting with a pmf you supply)
:mod:`ppsds.scoring_rule`                 scoring rules: ``logarithm_score`` and the neighbourhood ``epsilon_logarithm_score``
:mod:`ppsds.environment`                  Gymnasium environment layer; ``Env`` serves Scene 2, ``KnownSdsEnv`` steps a system known exactly
:mod:`ppsds.numerics`                     small numerical utilities (``linalg`` / ``random_variables`` / ``atom_pmf``)
:mod:`ppsds.predictability`               scoring an arbitrary predictor, and the predictability upper bound of a known system
:mod:`ppsds.experiment`                   experiment driver and validating cache (needs the ``experiments`` extra)
:mod:`ppsds.visualization`                plotting (needs the ``experiments`` extra)
:mod:`ppsds.control_policy`               small control utilities (``DLQR``)
========================================  ==============================================

Usage and further notes: see ``README.md`` and ``docs/tutorials/``.
"""

__version__ = "0.0.2"

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
from .scoring_rule.epsilon_logarithm_score import EpsilonLogarithmScore
from .ambiguity.additive_drift.conic_moment import (
    AdditiveDriftConicMomentSet,
    LinearAmbiguitySet,
    NonlinearAmbiguitySet,
)
from .numerics.atom_pmf import AtomPmf
from .probabilistic_predictor.pmf.atom import (
    StaticAtomPmfPredictor,
    DriftShiftedAtomPmfPredictor,
)
from .predictability.bounds import approximation_value, predictability_upper_bound
from .predictability.entropy import cross_entropy, differential_entropy

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
    "EpsilonLogarithmScore",
    "AtomPmf",
    "StaticAtomPmfPredictor",
    "DriftShiftedAtomPmfPredictor",
    "predictability_upper_bound",
    "approximation_value",
    "differential_entropy",
    "cross_entropy",
]
