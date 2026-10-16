"""Numerical helpers that exist purely to keep behavior consistent across numpy builds."""

from __future__ import annotations

import numpy as np
import pytest

from ppsds.numerics.linalg import eig_real
from ppsds.numerics.random_variables import multivariate_uniform

#: A real symmetric matrix with repeated-ish structure, so the eigen-decomposition is not
#: trivially diagonal. Same shape as the ambiguity-set covariances in the suite.
SYMMETRIC = np.array([[1.0, 0.45], [0.45, 1.30]])


def test_eig_real_returns_real_arrays_for_a_symmetric_matrix():
    lamb, vec = eig_real(SYMMETRIC)
    assert lamb.dtype == np.float64, f'eigenvalues came back as {lamb.dtype}'
    assert vec.dtype == np.float64, f'eigenvectors came back as {vec.dtype}'


def test_eig_real_is_lossless():
    """Taking the real part must not change the numbers.

    ``np.linalg.eig`` may return ``complex128`` (with exactly zero imaginary parts) or
    ``float64`` depending on the numpy build. For a symmetric input the imaginary parts are
    necessarily zero, so dropping them is exact -- if this ever fails, ``eig_real`` is silently
    discarding information rather than normalizing a dtype.
    """
    raw_lamb, raw_vec = np.linalg.eig(SYMMETRIC)
    lamb, vec = eig_real(SYMMETRIC)
    np.testing.assert_array_equal(lamb, raw_lamb.real)
    np.testing.assert_array_equal(vec, raw_vec.real)


def test_eig_real_agrees_with_the_symmetric_solver():
    """Eigenvalues must be the real ones, in whatever order ``np.linalg.eig`` chose."""
    lamb, _ = eig_real(SYMMETRIC)
    np.testing.assert_allclose(np.sort(lamb), np.linalg.eigvalsh(SYMMETRIC),
                               rtol=1e-12, atol=1e-12)


def test_multivariate_uniform_has_the_documented_shape():
    dist = multivariate_uniform(low=np.zeros(3), high=np.ones(3))
    assert dist.rvs().shape == (1, 3), 'the default size is 1, giving a (1, d) block'
    assert dist.rvs(size=7).shape == (7, 3)
    assert dist.dimensions == 3


def test_multivariate_uniform_samples_stay_in_the_box():
    low, high = np.array([-2.0, 0.5]), np.array([3.0, 4.0])
    dist = multivariate_uniform(low=low, high=high)
    samples = dist.rvs(size=500)
    assert np.all(samples >= low) and np.all(samples <= high)


def test_multivariate_uniform_density_is_one_over_the_volume():
    low, high = np.array([0.0, 0.0]), np.array([2.0, 5.0])
    dist = multivariate_uniform(low=low, high=high)
    assert dist.pdf(np.array([1.0, 1.0])) == pytest.approx(1.0 / 10.0)
    assert dist.pdf(np.array([3.0, 1.0])) == 0        # outside the box in one coordinate
    assert dist.pdf(np.array([1.0, -1.0])) == 0

# NOTE: a `multivariate_gmm` class used to live beside `multivariate_uniform` and, having zero
# callers, deliberately got no tests. It was deleted on 2026-10-16 -- an untested class nothing
# calls is a claim the package cannot back up, and "kept for reference" is what the git history is
# for. See `docs/strategy/MARKETING.md` on why every shipped module needs a sentence saying what
# it is for.
