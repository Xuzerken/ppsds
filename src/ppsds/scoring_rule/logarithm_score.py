"""Logarithmic scoring rule: score a predictive distribution against the true state.

``score(predictive_dist, x_true)`` returns ``logpdf(x_true)`` — **higher is better**.
DRPP's theoretical bounds (``value_upper_bound_true_transition`` /
``value_lower_bound_eigvec_fixed``) and the comparisons of predictors in the experiments all
use it.

For ``multivariate_rv`` only ``logpdf`` is required (duck typing); it need not be a scipy
distribution object.
"""


class LogarithmScore:
    """The standard logarithmic scoring rule."""

    def score(self, multivariate_rv, x):
        """Returns ``logpdf(x)``."""
        return multivariate_rv.logpdf(x)
