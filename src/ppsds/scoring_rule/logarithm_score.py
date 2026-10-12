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


class LogarithmScoreArgs(LogarithmScore):
    """Construct the distribution from positional arguments first, then score."""

    def __init__(self, multivariate_rv):
        super().__init__()
        self.multivariate_rv = multivariate_rv

    def score_args(self, args, x):
        """``args`` are the **positional** arguments used to construct the distribution."""
        return self.multivariate_rv(*args).logpdf(x)


class LogarithmScoreKargs(LogarithmScore):
    """Construct the distribution from keyword arguments first, then score."""

    def __init__(self, multivariate_rv):
        super().__init__()
        self.multivariate_rv = multivariate_rv

    def score_kargs(self, kargs, x):
        """``kargs`` are the **keyword** arguments used to construct the distribution."""
        return self.multivariate_rv(*kargs).logpdf(x)
