"""Adversarial (worst-case) dynamics for the additive-drift / conic-moment family.

The file path states the scope: **assumption** (additive drift) / **family** (conic
moment). An adversarial mode's "worst case" rule comes entirely from
``abg.worst_case_params()``, so a different family gets its own module beside this one
rather than a branch inside it.
"""

from typing import Optional

import numpy as np

from ...base import DynamicsMode, SamplingAmbiguitySet


class EigvecFixedAdversarialDynamics(DynamicsMode):
    """Pick SDS parameters adversarially so the predictor's score is worst.

    **Why the name is so narrow**: it delivers the ambiguity set's
    :meth:`~ppsds.ambiguity.additive_drift.conic_moment.AdditiveDriftConicMomentSet.value_lower_bound_eigvec_fixed`
    -- the bound Eig-DRPP guarantees under the "eigenvectors fixed" relaxation. Calling it
    ``AdversarialDynamics`` would **overclaim**: this mode's "worst" is defined entirely by
    ``abg.worst_case_params()``, so swapping in another ambiguity-set family changes the
    worst-case rule, and that family needs a mode class of its own.

    It needs information from the predictor -- specifically ``lamb_pred`` (the eigenvalues
    of its current predictive covariance). ``set_info(lamb_pred, x, u)`` must be called
    **before** each ``env.step()``.

    The actual worst-case computation is delegated to
    ``abg.worst_case_params(x, u, lamb_pred)``, which returns a dict shaped like
    ``sample()``'s output.

    Parameters
    ----------
    abg : SamplingAmbiguitySet
        Must implement ``worst_case_params()``.
    """

    def __init__(self, abg: SamplingAmbiguitySet):
        self._abg = abg
        # Internal buffer -- set_info() fills it, get_params() reads it
        self._lamb_pred: Optional[np.ndarray] = None
        self._x: Optional[np.ndarray] = None
        self._u: Optional[np.ndarray] = None
        self.j: int = -1

    def set_info(self, lamb_pred: np.ndarray, x: np.ndarray,
                 u: np.ndarray) -> None:
        """Inject predictor information **before** the next ``step()``.

        In adversarial mode this must be called once per step.

        Parameters
        ----------
        lamb_pred : ndarray of shape (x_dim,)
        x : ndarray of shape (x_dim,)
        u : ndarray of shape (u_dim,)
        """
        self._lamb_pred = np.asarray(lamb_pred)
        self._x = np.asarray(x)
        self._u = np.asarray(u)

    def reset(self, rng: np.random.Generator, T: int) -> None:
        self.j = -1
        self._lamb_pred = None
        self._x = None
        self._u = None

    def get_params(self, step_idx: int) -> dict:
        if self._lamb_pred is None:
            # First step: the predictor has no output yet, so use the nominal parameters.
            return {}
        params = self._abg.worst_case_params(
            self._x, self._u, self._lamb_pred)
        # Record the worst-case direction index (it is diagnostic information, not part of sds_params)
        if 'j' in params:
            self.j = params.pop('j')
        return params
