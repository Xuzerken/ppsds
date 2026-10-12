"""Dynamics-mode strategies: which set of SDS parameters each step uses.

Everything a mode needs from an ambiguity set is its **generation ability**:

- ``sample_trajectory`` (used by :class:`RandomDynamics`),
- ``worst_case_params`` (used by :class:`EigvecFixedAdversarialDynamics`).

:class:`SamplingAmbiguitySet` pins that contract down as a Protocol: a mode does not know
about "ambiguity sets" as a concept, only about objects that can hand out a parameter
trajectory and a worst-case parameter. :class:`PreSampledDynamics` does not even need
that -- its parameters were sampled outside.

Layout
------
Non-adversarial modes are flat in :mod:`.non_adversarial`; adversarial modes are filed
by **assumption / family**, mirroring :mod:`ppsds.ambiguity`::

    dynamics_mode/
    ├── base.py                     SamplingAmbiguitySet + DynamicsMode
    ├── non_adversarial.py          Fixed / Random / PreSampled
    └── adversarial/
        └── additive_drift/
            └── conic_moment.py     EigvecFixedAdversarialDynamics

An adversarial mode's notion of "worst case" is defined entirely by the ambiguity-set
family it targets, so a **new family** gets a new module here rather than a new flag on an
existing mode. That is also why the one adversarial mode is named narrowly: see
:class:`EigvecFixedAdversarialDynamics`.

Usage
-----
::

    mode = RandomDynamics(abg, period=100)
    mode.reset(rng, T=50)
    for step in range(T):
        params = mode.get_params(step)   # keys declared by abg.param_keys
"""

from .base import DynamicsMode, SamplingAmbiguitySet
from .non_adversarial import FixedDynamics, PreSampledDynamics, RandomDynamics
from .adversarial.additive_drift.conic_moment import EigvecFixedAdversarialDynamics

__all__ = [
    "SamplingAmbiguitySet",
    "DynamicsMode",
    "FixedDynamics",
    "RandomDynamics",
    "PreSampledDynamics",
    "EigvecFixedAdversarialDynamics",
]
