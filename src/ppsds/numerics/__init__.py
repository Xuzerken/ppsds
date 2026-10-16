"""Small numerical tools, and the weighted-atom pmf.

Two things that do not belong anywhere more specific:

- :mod:`~ppsds.numerics.linalg` — :func:`~ppsds.numerics.linalg.eig_real`. Use this rather than
  calling ``np.linalg.eig`` directly: the dtype it returns changed between numpy versions for the
  same real symmetric matrix, and the complex form makes a downstream cone program refuse to solve.
- :mod:`~ppsds.numerics.atom_pmf` — :class:`~ppsds.numerics.atom_pmf.AtomPmf`: a weighted set of
  atoms. It is the object behind "a pmf is a legitimate prediction", and
  :meth:`~ppsds.numerics.atom_pmf.AtomPmf.box_integral` is what lets the epsilon score evaluate it.
- :mod:`~ppsds.numerics.random_variables` — :class:`~ppsds.numerics.random_variables.multivariate_uniform`,
  the noise law the ambiguity sets sample from.
"""

from .atom_pmf import AtomPmf
from .linalg import eig_real
from .random_variables import multivariate_uniform

__all__ = ['AtomPmf', 'eig_real', 'multivariate_uniform']
