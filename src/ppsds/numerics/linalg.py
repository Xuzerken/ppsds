"""Small numerical utilities that keep behavior consistent across numpy versions.

Why :func:`eig_real` is needed
------------------------------
What dtype ``np.linalg.eig`` returns for a **real** matrix is not consistent across numpy
builds: for the same real symmetric covariance matrix, numpy 2.3.5 returns ``float64``
while 2.5.4 returns ``complex128`` (imaginary parts exactly 0, real parts **bit-for-bit
identical** to the former).

The downstream cone program requires real-valued input and raises
``The 'minimize' objective must be real valued`` on the latter — and the error point is very
far from the real cause.

A covariance matrix is necessarily symmetric, so its eigenvalues are necessarily real, which
makes taking the real part **lossless**; for return values that are already real arrays,
``.real`` returns them unchanged, bit for bit.
"""

import numpy as np


def eig_real(matrix):
    """A real-valued version of ``np.linalg.eig``: returns ``(eigenvalues, eigenvectors)``, both
    guaranteed to be real arrays.

    Should only be used when the input is a symmetric matrix (e.g. a covariance) — in that
    case the eigenvalues are necessarily real, so taking the real part is lossless.
    """
    lamb, vec = np.linalg.eig(matrix)
    return lamb.real, vec.real
