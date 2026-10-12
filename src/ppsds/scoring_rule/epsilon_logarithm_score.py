"""ε-neighborhood logarithmic scoring rule: take the logarithm of the CDF integral over an
ε-box around the true state.

**Currently has zero callers** (merged in on 2026-10-12 from
``src/lab/scoring_rule/epsilon_logarithm_score.py`` — lab was merged into ppsds wholesale and
this dead code was brought along, not deleted).
:class:`~ppsds.scoring_rule.logarithm_score.LogarithmScore` is the one actually used
everywhere.
"""

import numpy as np

def bin_to_list(x, d):
    return np.array([(x>>k)&1 for k in range(d-1, -1, -1)])

class EpsilonLogarithmScore:
    def __init__(self, eps):
          self.eps = eps
    
    def score(self, multivariate_rv, x):
        d_x = x.shape[0]
        x_full = x + self.eps * np.ones(d_x)
        m = (1 << d_x) - 1
        s = m
        v = bin_to_list(s, d_x)
        return_rate = multivariate_rv.cdf(x_full - 2 * self.eps * v) * ( 1-2*(np.sum(v)%2) )

        while (s > 0):
            s = (s-1) & m
            v = bin_to_list(s, d_x)
            return_rate += multivariate_rv.cdf(x_full - 2 * self.eps * v) * ( 1-2*(np.sum(v)%2) )

        with np.errstate(divide="ignore"):
            utility = np.log(return_rate)

        return utility
