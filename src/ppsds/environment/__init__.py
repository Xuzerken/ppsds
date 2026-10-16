"""The Gymnasium environment layer: simulate a system you declared an ambiguity set for.

You do not need this to *use* a predictor — putting one on your own environment is a declaration
and nothing else (see :mod:`ppsds.probabilistic_predictor` + :mod:`ppsds.scoring_rule`). This
subpackage is for the other direction: declaring an ambiguity set by hand and generating dynamics
from it.

There is exactly one generic :class:`Env`, combining an ambiguity set with a
:class:`DynamicsMode`; it is not subclassed per system. Everything that differs between
systems lives in the **ambiguity set** (see
:mod:`ppsds.ambiguity.additive_drift.conic_moment`).

The name is just ``Env``, which already states its home -- it is not ``DRPPEnv``, because
"DRPP" carries no information (the whole package is DRPP) and invites the wrong assumption
that adversarialism is intrinsic to it: only one of the four modes
(:class:`~ppsds.environment.dynamics_mode.EigvecFixedAdversarialDynamics`) is adversarial,
while fixed / random / pre-sampled are not.

The scope claim above — "simulate a system you declared a set for" — is about ``Env``, the
ambiguity-composing one. :class:`KnownSdsEnv` is the exception, and deliberately so: it steps a
system that is **known exactly**, with no ambiguity set at all, which is the setting the
predictability questions are asked in. So read the sentence as covering ``Env``, not the whole
subpackage.
"""

from .base import Env
from .dynamics_mode import (
    DynamicsMode,
    FixedDynamics,
    RandomDynamics,
    PreSampledDynamics,
    EigvecFixedAdversarialDynamics,
)
from .known_sds import KnownSdsEnv

__all__ = [
    "Env",
    "KnownSdsEnv",
    "DynamicsMode",
    "FixedDynamics",
    "RandomDynamics",
    "PreSampledDynamics",
    "EigvecFixedAdversarialDynamics",
]
