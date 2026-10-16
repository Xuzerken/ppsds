"""How predictable the system is at all: the **ceiling**, as a reference line.

Everything else in this package builds a predictor. This subpackage asks the question from the
other side and needs no predictor to answer it: given only the true one-step law, the
epsilon-logarithm score has a best possible expected value, and that value is what any prediction
is measured against.

Three pieces:

- :mod:`~ppsds.predictability.bounds` — the ceiling, and the paper's ``O(epsilon)`` approximation
  to it;
- :mod:`~ppsds.predictability.entropy` — the differential entropy of the true law, and its
  cross-entropy against a predictive law;
- :mod:`~ppsds.predictability.trajectory` — aggregating per-step scores into the curves a figure
  plots. Numpy-only, and it lives in the core because the aggregation is not a plotting concern.

This is an **upper** bound, and a property of the system. It is not the same pair of bounds as
``value_upper_bound_true_transition`` / ``value_lower_bound_eigvec_fixed``, which bracket the
relaxation gap *inside one* ambiguity set. Both pairs are easy to mix up and worth keeping apart.
"""

from .bounds import approximation_value, predictability_upper_bound
from .entropy import cross_entropy, differential_entropy
from .trajectory import across_trajectories, cumulative_average

__all__ = [
    'predictability_upper_bound',
    'approximation_value',
    'differential_entropy',
    'cross_entropy',
    'cumulative_average',
    'across_trajectories',
]
