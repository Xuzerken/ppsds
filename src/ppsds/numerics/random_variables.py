"""Small random-variable utilities.

Two classes:

- :class:`multivariate_uniform` — a multivariate uniform distribution with **independent**
  components. Used by the core: when sampling an ambiguity set, the noise mean is drawn from
  a box (see :mod:`ppsds.ambiguity.additive_drift.conic_moment`).
- :class:`multivariate_gmm` — a Gaussian mixture. **Currently has zero callers**; shipped with
  the package because it still has reference value, not deleted.

.. note::
   Both ``rvs()`` use the **global** ``np.random`` and have no seed of their own.
   Reproducibility is the caller's responsibility (e.g. ``np.random.seed(...)`` before
   sampling).
"""

import numpy as np
from scipy.stats import uniform, multivariate_normal


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
        
    def rvs(self, size=1):
        """
        Generates a sample from the multivariate uniform distribution.
        
        Parameters:
        size (int): The number of samples to generate.
        
        Returns:
        numpy.ndarray: A 2D array of shape (size, dimensions) containing the samples.
        """
        samples = np.zeros((size, self.dimensions))
        for i in range(self.dimensions):
            samples[:, i] = self.univariate_dists[i].rvs(size=size)
        return samples
    
    def pdf(self, x):
        if (np.all(x <= self.high) & np.all(x >= self.low)):
            return 1 / (np.prod(self.high - self.low))
        else:
            return 0
        
    def cdf(self, x):
        v = x - self.low
        if (np.all(v > 0)):
            v_range = self.high - self.low
            v_effect = np.minimum(v, v_range)
            return np.prod(v_effect) / np.prod(v_range)
        else:
            return 0
    
    def entropy(self):
        return np.log( np.prod(self.high - self.low) )


class multivariate_gmm:
    """A Gaussian mixture random variable. **Currently has zero callers**; kept for reference.

    (Merged in on 2026-10-12 from ``src/lab/random_variables/stats.py`` — lab was merged into
    ppsds wholesale and, being small and on a par with linalg, was placed in this module.)
    """

    def __init__(self, means, covariances, weights):
        """
        Initialize the Gaussian Mixture Multivariate Random Variable.

        Parameters:
        means (numpy.ndarray): A 2D array of means for each component, where each row represents the mean of a component.
        covariances (numpy.ndarray): A 3D array of covariance matrices for each component, where each 2D matrix represents the covariance of a component.
        weights (numpy.ndarray): A 1D array of weights for each component.
        """
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.n_components = len(weights)

    def rvs(self, size=1):
        """
        Sample from the Gaussian Mixture Multivariate Random Variable.

        Parameters:
        size (int): The number of samples to generate.

        Returns:
        numpy.ndarray: A 2D array of samples, where each row represents a sample.
        """
        # Sample the component index
        component_idx = np.random.choice(self.n_components, size=size, p=self.weights)

        # Sample from each component and combine the samples
        samples = np.zeros((size, self.means.shape[1]))
        for i in range(self.n_components):
            mask = component_idx == i
            samples[mask] = np.random.multivariate_normal(self.means[i], self.covariances[i], size=np.sum(mask))

        return samples

    def pdf(self, x):
        """
        Evaluate the probability density function (PDF) of the Gaussian Mixture Multivariate Random Variable.

        Parameters:
        x (numpy.ndarray): A 2D array of points at which to evaluate the PDF, where each row represents a point.

        Returns:
        numpy.ndarray: A 1D array of PDF values, where each element represents the PDF value at the corresponding point.
        """
        pdf_values = np.zeros(x.shape[0])
        for i in range(self.n_components):
            pdf_values += self.weights[i] * multivariate_normal.pdf(x, mean=self.means[i], cov=self.covariances[i])
        return pdf_values
