"""Scoring rules: how well a prediction did, in one number.

A scoring rule is what makes "a good prediction" mean anything. Two are shipped, and the choice
between them is not cosmetic:

- :class:`~ppsds.scoring_rule.logarithm_score.LogarithmScore` — ``logpdf`` of the realised state.
  **Higher is better.** It needs the predictive law to have a density, so a degenerate covariance
  scores ``-inf`` rather than raising — arithmetic, not a bug.
- :class:`~ppsds.scoring_rule.epsilon_logarithm_score.EpsilonLogarithmScore` — the logarithm of the
  predictive probability *near* the outcome, within an ``epsilon`` box. This is the one that lets a
  **pmf** be scored at all, and it degrades to the ordinary log score as ``epsilon`` goes to zero.

The same rule is used for the theoretical bounds and for every comparison in the examples, so a
number means the same thing wherever you meet it.
"""

from .epsilon_logarithm_score import EpsilonLogarithmScore
from .logarithm_score import LogarithmScore

__all__ = ['LogarithmScore', 'EpsilonLogarithmScore']
