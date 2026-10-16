"""Small random-variable utilities.

:class:`multivariate_uniform` — a multivariate uniform distribution with **independent**
components. Used by the core: when sampling an ambiguity set, the noise mean is drawn from a box
(see :mod:`ppsds.ambiguity.additive_drift.conic_moment`).

.. note::
   ``rvs()`` defaults to the **global** ``np.random`` and has no seed of its own, so
   reproducibility is the caller's responsibility (e.g. ``np.random.seed(...)`` before sampling)
   -- or pass an explicit ``rng``, which is the form new code should use.

   ``pdf()`` / ``cdf()`` accept a single point ``(d,)`` and return a scalar, or many points
   ``(n, d)`` and return ``(n,)``.

A ``multivariate_gmm`` (Gaussian mixture) class lived here until 2026-10-16. It was merged in
with ``src/lab/`` on 2026-10-12, had **zero callers** for its whole life, and shipped in the wheel
as unexercised code. Removed rather than carried: a class nothing calls and nothing tests is a
claim the package cannot back up.
"""

import numpy as np
from scipy.stats import uniform


class multivariate_uniform:
    """A multivariate uniform distribution with independent components; the box is given by
    ``low`` / ``high``."""

    def __init__(self, low = -np.ones(2), high=np.ones(2)):
        """
        Initializes a multivariate uniform random variable.

        Parameters:
        low (numpy.ndarray): A 1D array representing the lower bounds of the uniform distribution for each dimension.
        high (numpy.ndarray): A 1D array representing the upper bounds of the uniform distribution for each dimension.
        """
        self.low = np.array(low)
        self.high = np.array(high)
        self.dimensions = len(self.low)
        
        # Create a list of independent univariate uniform distributions
        self.univariate_dists = [uniform(loc=self.low[i], scale=self.high[i] - self.low[i]) for i in range(self.dimensions)]
        
    def rvs(self, size=1, rng=None):
        """
        Generates a sample from the multivariate uniform distribution.

        Parameters:
        size (int): The number of samples to generate.
        rng (numpy.random.Generator, optional): Source of randomness. ``None`` (the default)
            uses the global ``np.random`` state, which is what the existing callers rely on;
            pass a ``Generator`` for a reproducible draw.

        Returns:
        numpy.ndarray: A 2D array of shape (size, dimensions) containing the samples.
        """
        samples = np.zeros((size, self.dimensions))
        for i in range(self.dimensions):
            samples[:, i] = self.univariate_dists[i].rvs(size=size, random_state=rng)
        return samples

    def pdf(self, x):
        """Density at a point ``(d,)``, or at many points ``(n, d)`` -> ``(n,)``."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            if (np.all(x <= self.high) & np.all(x >= self.low)):
                return 1 / (np.prod(self.high - self.low))
            else:
                return 0
        inside = np.all((x <= self.high) & (x >= self.low), axis=-1)
        return np.where(inside, 1.0 / np.prod(self.high - self.low), 0.0)

    def cdf(self, x):
        """CDF at a point ``(d,)`` (scalar), or at many points ``(n, d)`` -> ``(n,)``."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            v = x - self.low
            if (np.all(v > 0)):
                v_range = self.high - self.low
                v_effect = np.minimum(v, v_range)
                return np.prod(v_effect) / np.prod(v_range)
            else:
                return 0
        v_range = self.high - self.low
        v_effect = np.clip(x - self.low, 0.0, v_range)
        return np.prod(v_effect / v_range, axis=-1)
    
    def entropy(self):
        return np.log( np.prod(self.high - self.low) )
