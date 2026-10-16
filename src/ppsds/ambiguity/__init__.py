"""Ambiguity sets: what you are willing to admit about the system, written down as a set.

This is the **belief** side of the package. A set supplies the nominal model every admitted system
is near, and it is what a predictor is made robust against — but it predicts nothing itself.
:mod:`ppsds.probabilistic_predictor` does that, and **each family of ambiguity sets yields its own
DRPP**.

The tree states the two things a set has to declare::

    ambiguity/<assumption>/<family>.py

``additive_drift`` is the assumption — ``x' = f(x, u) + delta_f + w``, with the drift error and the
noise both entering the state *added* to the nominal drift. ``conic_moment`` is the family, which
parameterizes how far ``delta_f``, the noise mean and the noise covariance may move. **A new family
is a new module at the same level**, not a new branch inside this one; see
:mod:`ppsds.ambiguity.additive_drift.conic_moment` for what a family must provide.
"""

from .additive_drift.conic_moment import (
    AdditiveDriftConicMomentSet,
    LinearAmbiguitySet,
    NonlinearAmbiguitySet,
)

__all__ = ['AdditiveDriftConicMomentSet', 'LinearAmbiguitySet', 'NonlinearAmbiguitySet']
