"""A discrete predictive law: probability atoms with weights.

:class:`AtomPmf` is the "predictive law may be a pmf" object. Its support need not contain the
support of the truth, and that is the point: under a neighbourhood scoring rule a small set of
well-placed atoms is a perfectly usable prediction, whereas under the ordinary logarithm score
it would be hopeless almost surely.

Which is what :meth:`AtomPmf.logpdf` is here to make concrete. It exists, and it returns
``-inf``, because an atom law has no density: scoring it with
:class:`~ppsds.scoring_rule.logarithm_score.LogarithmScore` gives ``-inf`` for every outcome,
correctly. Score it with
:class:`~ppsds.scoring_rule.epsilon_logarithm_score.EpsilonLogarithmScore` instead -- the box
integral :meth:`AtomPmf.box_integral` is what that score evaluates.
"""

from __future__ import annotations

import numpy as np


class AtomPmf:
    """A weighted set of atoms: ``sum_i weights[i] * delta(atoms[i])``.

    Parameters
    ----------
    atoms : array_like
        ``(n_atoms, d_x)``. ``(d_x,)`` for a single atom is promoted to one row.
    weights : array_like
        ``(n_atoms,)``, non-negative. **Normalised on construction**, so raw counts or
        stakes are accepted as they are; ``AtomPmf.weights`` is what the law actually uses.

    Raises
    ------
    ValueError
        If the shapes disagree, a weight is negative, or the weights sum to zero.
    """

    def __init__(self, atoms, weights):
        atoms = np.atleast_2d(np.asarray(atoms, dtype=float))
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if atoms.ndim != 2:
            raise ValueError(f'atoms must be (n_atoms, d_x), got shape {atoms.shape}')
        if weights.shape[0] != atoms.shape[0]:
            raise ValueError(
                f'got {atoms.shape[0]} atoms but {weights.shape[0]} weights; they must pair up'
            )
        if atoms.shape[0] == 0:
            raise ValueError('a pmf needs at least one atom')
        if np.any(weights < 0):
            raise ValueError('weights must be non-negative')
        total = weights.sum()
        if total <= 0:
            raise ValueError('weights must sum to something positive')
        self.atoms = atoms
        self.weights = weights / total

    @property
    def dimensions(self):
        """The state dimension ``d_x``, matching ``multivariate_uniform``'s attribute name."""
        return self.atoms.shape[1]

    @property
    def mean(self):
        """The weighted mean of the atoms."""
        return self.weights @ self.atoms

    @property
    def cov(self):
        """The weighted covariance of the atoms -- the moment summary of the law."""
        centred = self.atoms - self.mean
        return (centred * self.weights[:, None]).T @ centred

    def entropy(self):
        """The **Shannon** entropy of the weights, ``-sum w log w``.

        Not a differential entropy, and not interchangeable with one: this is non-negative,
        while a differential entropy is a log of a volume and may be negative. The
        predictability upper bound takes the entropy of the *continuous* truth, so do not pass
        an :class:`AtomPmf` as that truth.
        """
        positive = self.weights > 0
        return float(-np.sum(self.weights[positive] * np.log(self.weights[positive])))

    def logpdf(self, x):
        """``-inf`` everywhere -- an atom law has no density.

        Provided rather than omitted so that scoring a pmf with the ordinary logarithm score
        *runs* and reports ``-inf``, instead of failing with an attribute error that hides the
        reason. Use the epsilon-logarithm score.
        """
        x = np.asarray(x, dtype=float)
        shape = () if x.ndim == 1 else (x.shape[0],)
        return np.full(shape, -np.inf)

    def rvs(self, size=1, rng=None):
        """Draw ``size`` atoms according to the weights. ``(size, d_x)``.

        Parameters
        ----------
        size : int
            Number of draws.
        rng : numpy.random.Generator, optional
            Source of randomness; ``None`` uses the global ``np.random`` state (the same
            convention as :class:`~ppsds.numerics.random_variables.multivariate_uniform`).
        """
        chooser = np.random if rng is None else rng
        picked = chooser.choice(self.atoms.shape[0], size=size, p=self.weights)
        return self.atoms[picked]

    def box_integral(self, x, eps):
        """Total weight within the ``L-infinity`` box of half-width ``eps`` around each point.

        Returns ``sum_i weights[i]`` over atoms with ``||atom_i - x||_inf <= eps``. This is the
        quantity the epsilon-logarithm score takes the logarithm of, and the reason a pmf is
        scoreable at all.

        Parameters
        ----------
        x : array_like
            ``(d_x,)`` for one outcome, or ``(n, d_x)`` for a batch -- the shape a rollout
            produces.
        eps : float
            Box half-width, ``>= 0``.

        Returns
        -------
        float or numpy.ndarray
            A scalar, or ``(n,)``.

        Notes
        -----
        Exact, and vectorised over the outcome batch: every atom is tested against every
        outcome, in chunks that bound the intermediate boolean array. The cost is
        ``O(n_atoms * n_outcomes * d_x)``, which sounds worse than a spatial index but is not
        in practice -- measured at 0.9 s for the full ``10**5`` trajectories x ``30`` steps
        (3e6 outcomes) against 50 atoms, and it stays exact, which cell-summing over a lattice
        would not be: with cells two eps wide and a box one eps either side, a neighbouring
        cell only overlaps the box partially, so summing whole cells over-counts. If a pmf
        ever carries thousands of atoms, a ``cKDTree`` ball query with ``p = inf`` is the next
        step; at tens to hundreds it would be slower, not faster.
        """
        eps = float(eps)
        if not np.isfinite(eps) or eps < 0:
            raise ValueError(f'eps must be a finite radius >= 0, got {eps}')
        single = np.asarray(x, dtype=float).ndim == 1
        query = np.atleast_2d(np.asarray(x, dtype=float))
        if query.shape[1] != self.dimensions:
            raise ValueError(
                f'x must have {self.dimensions} coordinate(s), got {query.shape[1]}'
            )
        chunk = max(1, int(2e7 // self.atoms.shape[0]))
        out = np.empty(query.shape[0])
        for start in range(0, query.shape[0], chunk):
            block = query[start:start + chunk]
            hit = np.ones((block.shape[0], self.atoms.shape[0]), dtype=bool)
            for axis in range(self.dimensions):
                np.logical_and(
                    hit,
                    np.abs(block[:, axis, None] - self.atoms[None, :, axis]) <= eps,
                    out=hit,
                )
            out[start:start + chunk] = hit @ self.weights
        return float(out[0]) if single else out

    def __repr__(self):
        return f'AtomPmf(n_atoms={self.atoms.shape[0]}, d_x={self.dimensions})'
