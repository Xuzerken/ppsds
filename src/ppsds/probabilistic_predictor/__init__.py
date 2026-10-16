"""Probabilistic predictors: a prediction is a **distribution**, not a point.

Predictors come in *kinds*, and the kinds sit side by side, one subpackage each:

- :mod:`~ppsds.probabilistic_predictor.drpp` — a predictor **derived** from an ambiguity set. This
  is where DRPP lives, and it is the kind that comes with a worst-case guarantee.
- :mod:`~ppsds.probabilistic_predictor.pmf` — a predictor whose law is a pmf **you** supply. No
  optimizer is involved; the point is that a prediction you made by hand is still a prediction, and
  gets scored by the same rules in the same environments.

A third kind — a data-driven, learning-based one — would be a third subpackage here, with no
change to anything above.

:mod:`~ppsds.probabilistic_predictor.base` holds the contract every kind satisfies.
"""

from .base import ProbabilisticPredictor
from .drpp.additive_drift.conic_moment import (
    EigDrpp,
    NoiseDrpp,
    NominalPredictor,
    eigvec_fixed_minmax,
)
from .pmf.atom import DriftShiftedAtomPmfPredictor, StaticAtomPmfPredictor

__all__ = [
    'ProbabilisticPredictor',
    'NominalPredictor',
    'NoiseDrpp',
    'EigDrpp',
    'eigvec_fixed_minmax',
    'StaticAtomPmfPredictor',
    'DriftShiftedAtomPmfPredictor',
]
