"""Gymnasium environment layer for DRPP -- **scene 2 only** (custom simulated ambiguity).

There is exactly one generic :class:`Env`, combining an ambiguity set with a
:class:`DynamicsMode`; it is not subclassed per system. Everything that differs between
systems lives in the **ambiguity set** (see
:mod:`ppsds.ambiguity.additive_drift.conic_moment`).

The name is just ``Env``, which already states its home -- it is not ``DRPPEnv``, because
"DRPP" carries no information (the whole package is DRPP) and invites the wrong assumption
that adversarialism is intrinsic to it: only one of the four modes
(:class:`~ppsds.environment.dynamics_mode.EigvecFixedAdversarialDynamics`) is adversarial,
while fixed / random / pre-sampled are not.

Scene 1 (applying DRPP to **your own** gymnasium environment) does not need this
subpackage -- only :mod:`ppsds.probabilistic_predictor` + :mod:`ppsds.scoring_rule`.
"""

from .base import Env
from .dynamics_mode import (
    DynamicsMode,
    FixedDynamics,
    RandomDynamics,
    PreSampledDynamics,
    EigvecFixedAdversarialDynamics,
)

__all__ = [
    "Env",
    "DynamicsMode",
    "FixedDynamics",
    "RandomDynamics",
    "PreSampledDynamics",
    "EigvecFixedAdversarialDynamics",
]
