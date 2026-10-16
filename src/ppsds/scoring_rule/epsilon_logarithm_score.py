"""The epsilon-logarithm score: the logarithm of the predictive probability *near* the outcome.

Given a neighbourhood radius ``eps``, score a predictive law ``p`` on a realised outcome ``x`` by

.. math::
    \\mathcal L_\\epsilon(p, x) = \\log \\int_{\\|s - x\\|_\\infty \\le \\epsilon} p(s)\\, ds ,

which is the ordinary logarithm score when ``eps = 0``. Taking a neighbourhood is what lets the
score be evaluated on a **pmf**, whose support need not contain the truth's -- the ordinary
logarithm score is ``-inf`` for such a predictor almost surely, the epsilon version is not.
``\\|\\cdot\\|_\\infty`` is the norm the predictability paper uses, chosen for the convenience of
integrating over a box.

Which path is taken
-------------------
:meth:`EpsilonLogarithmScore.score` dispatches on the law it is handed, cheapest first:

1. ``eps == 0`` -- ``logpdf``, the ordinary logarithm score;
2. a law exposing ``box_integral(x, eps)`` -- the pmf path, e.g.
   :class:`~ppsds.numerics.random_variables.AtomPmf`;
3. a Gaussian with **diagonal** covariance -- the closed form
   ``prod_i [Phi((x_i + eps - mu_i) / sigma_i) - Phi((x_i - eps - mu_i) / sigma_i)]``, which is
   what makes a ``10**5``-trajectory sweep of a Gaussian predictor cheap;
4. anything else with a ``cdf`` -- inclusion-exclusion over the ``2 ** d_x`` box corners. This
   is the general fallback and costs exponentially in ``d_x``, so it suits small ``d_x``.

Input is a single outcome ``(d_x,)`` or a batch ``(n, d_x)``, returning a scalar or ``(n,)``.
A zero box probability gives ``-inf``, silently: that *is* the score for a predictor that puts
no mass anywhere near what happened.

.. warning::
   The score is **discontinuous at ``eps = 0``**, and this is a property of the definition
   rather than a numerical artefact. At exactly ``eps = 0`` the score is ``log p(x)``; for
   ``eps > 0`` the box integral is about ``(2 eps) ** d_x * p(x)``, so the score is about
   ``log p(x) + d_x log(2 eps)`` and rushes to ``-inf`` as ``eps -> 0+``. Said the other way
   round, ``L_eps - d_x log(2 eps) -> log p(x)``: the ordinary logarithm score is recovered
   only after subtracting the box's log volume, and the ``I_{eps > 0}`` indicator in the
   predictability formulas is what encodes this. Consequences worth knowing: a pmf predictor
   scores ``-inf`` at ``eps = 0`` for *any* finite ``eps`` too, only less so; and a slider
   sweeping ``eps`` down to zero shows a cliff, not a landing.
"""

import numpy as np


def bin_to_list(x, d):
    """Bit decomposition of ``x`` into ``d`` most-significant-first bits."""
    return np.array([(x>>k)&1 for k in range(d-1, -1, -1)])


class EpsilonLogarithmScore:
    """The epsilon-logarithm scoring rule.

    Parameters
    ----------
    eps : float
        Neighbourhood radius, ``>= 0``. Zero degenerates to the ordinary logarithm score.
    """

    def __init__(self, eps):
        if not np.isfinite(eps) or eps < 0:
            raise ValueError(f'eps must be a finite radius >= 0, got {eps}')
        self.eps = eps

    def score(self, multivariate_rv, x):
        """Score ``multivariate_rv`` on the realised outcome(s) ``x``; higher is better.

        Parameters
        ----------
        multivariate_rv : object
            The predictive law. Any of: one exposing ``box_integral``; a Gaussian exposing
            ``mean`` / ``cov``; or anything exposing ``cdf``. With ``eps == 0`` it must expose
            ``logpdf``.
        x : array_like
            A single outcome ``(d_x,)`` or a batch ``(n, d_x)``.

        Returns
        -------
        float or numpy.ndarray
            The score, or the vector of scores for a batch.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim not in (1, 2):
            raise ValueError(f'x must be (d_x,) or (n, d_x), got shape {x.shape}')
        if self.eps == 0:
            return self._log_density(multivariate_rv, x)
        probability = self._box_probability(multivariate_rv, x)
        with np.errstate(divide="ignore"):
            return np.log(probability)

    __call__ = score

    def _log_density(self, multivariate_rv, x):
        """``eps == 0``: the ordinary logarithm score."""
        logpdf = getattr(multivariate_rv, 'logpdf', None)
        if logpdf is None:
            raise TypeError(
                f'eps == 0 needs logpdf, but {type(multivariate_rv).__name__} has none. A pmf '
                'predictor has no density at all -- that is what the eps > 0 neighbourhood is '
                'for.'
            )
        return logpdf(x)

    def _box_probability(self, multivariate_rv, x):
        """``int_{||s-x||_inf <= eps} p(s) ds``, by the cheapest available route."""
        box_integral = getattr(multivariate_rv, 'box_integral', None)
        if box_integral is not None:
            return np.asarray(box_integral(x, self.eps), dtype=float)
        diagonal = self._diagonal_gaussian_probability(multivariate_rv, x)
        if diagonal is not None:
            return diagonal
        cdf = getattr(multivariate_rv, 'cdf', None)
        if cdf is None:
            raise TypeError(
                f'{type(multivariate_rv).__name__} exposes none of box_integral, a Gaussian '
                'mean/cov, or cdf, so the epsilon-box integral cannot be evaluated.'
            )
        return self._cdf_inclusion_exclusion(cdf, x)

    def _diagonal_gaussian_probability(self, multivariate_rv, x):
        """Closed form for a Gaussian with diagonal covariance; ``None`` when it does not apply.

        The box integral of a diagonal Gaussian factorises into a product of univariate normal
        differences, which avoids ``2 ** d_x`` general-CDF evaluations per point.
        """
        mean, cov = getattr(multivariate_rv, 'mean', None), getattr(multivariate_rv, 'cov', None)
        if mean is None or cov is None:
            return None
        mean, cov = np.asarray(mean, dtype=float), np.asarray(cov, dtype=float)
        if mean.ndim != 1 or cov.shape != (mean.size, mean.size):
            return None
        if not np.allclose(cov, np.diag(np.diag(cov))):
            return None
        sigma = np.sqrt(np.diag(cov))
        if np.any(sigma <= 0):
            return None
        from scipy.stats import norm
        upper = norm.cdf((x + self.eps - mean) / sigma)
        lower = norm.cdf((x - self.eps - mean) / sigma)
        return np.prod(upper - lower, axis=-1)

    def _cdf_inclusion_exclusion(self, cdf, x):
        """Box probability from the CDF, by inclusion-exclusion over the ``2 ** d_x`` corners."""
        d_x = x.shape[-1]
        upper = x + self.eps * np.ones(d_x)
        total = None
        for index in range(1 << d_x):
            corner = bin_to_list(index, d_x)
            sign = 1 - 2 * (np.sum(corner) % 2)
            term = sign * np.asarray(cdf(upper - 2 * self.eps * corner), dtype=float)
            total = term if total is None else total + term
        return total
