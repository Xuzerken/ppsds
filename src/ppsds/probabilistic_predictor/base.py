"""Interface contract for probabilistic predictors (``typing.Protocol``).

Every DRPP predictor follows a **stateful** ``predict() -> update()`` cycle:

1. ``predict()`` returns the predictive distribution of :math:`x_{k+1}` given the
   current internal state;
2. ``update(x_observe, u_observe)`` advances the internal state once the true state and
   control have been observed.

The statefulness exists so that expensive computations (such as the cone program of
Eig-DRPP) can be cached from step to step.

Why ``Protocol`` and not ``ABC``
--------------------------------
This module **used to be** an ``ABC`` plus ``ABCMeta.register()`` virtual subclasses, but
that design had a trap:

- The contract declared ``lamb_pred`` / ``x_pred`` / ``Sigma_pred`` as ``@property``.
  ``@property`` is a **data descriptor**, whereas the concrete predictors assign them as
  **plain instance attributes** (``self.x_pred = ...``). The two are incompatible -- **to
  actually inherit it would raise ``AttributeError`` at construction time**.
- Registering virtually through ``register()`` is precisely what **hid** that
  incompatibility: ``isinstance()`` returned True, while not one clause of the contract was
  actually enforced.

With ``Protocol``: attributes are declared by **annotation** (a plain instance attribute
satisfies them), methods are declared with ``...``, there is zero runtime overhead, and
``register()`` is no longer needed. Use ``mypy`` when you want static checking.

``Protocol`` cannot provide default implementations, so implementers must supply their own
``reset()``.
"""

from typing import Any, Protocol

import numpy as np


class ProbabilisticPredictor(Protocol):
    """Interface for a probabilistic predictor of a stochastic dynamical system.

    Attributes
    ----------
    ``x_pred`` / ``Sigma_pred`` / ``lamb_pred`` are **instance attributes that must
    exist**: the environment layer's
    :class:`~ppsds.environment.dynamics_mode.EigvecFixedAdversarialDynamics` selects the
    worst-case perturbation direction from them.
    """

    #: Mean of the current predicted state; ``None`` before any prediction.
    x_pred: np.ndarray | None
    #: Current predictive covariance; ``None`` before any prediction.
    Sigma_pred: np.ndarray | None
    #: Eigenvalues of the current predictive covariance (``EigvecFixedAdversarialDynamics``
    #: uses them to select the worst-case perturbation direction).
    lamb_pred: np.ndarray | None

    def predict(self) -> Any:
        """Return the predictive distribution for the current step (usually a scipy distribution object)."""
        ...

    def update(self, x_observe: np.ndarray | None = None,
               u_observe: np.ndarray | None = None) -> None:
        """Advance the internal state after observing the true state and control.

        Parameters
        ----------
        x_observe : ndarray of shape (x_dim,), optional
            The observed state :math:`x_k`. Passing ``None`` selects **multi-step
            prediction** mode -- the previous step's prediction is taken as the state and
            extrapolation continues.
        u_observe : ndarray of shape (u_dim,), optional
            The control input :math:`u_k`. Passing ``None`` reuses the previous step's
            input.
        """
        ...

    def reset(self) -> None:
        """Restore the predictor to a reusable initial state.

        ⚠️ An implementation **must** restore the quantities that
        ``trajectory_prediction`` **rewrites in place** (the nominal drift ``f_bar``, the
        nominal noise ``mu_bar`` / ``Sigma_bar``, and the eigendecomposition
        ``lamb`` / ``v`` derived from ``Sigma_bar``). Clearing the prediction outputs alone
        is not enough -- reusing the same instance for a second trajectory would carry the
        leftover parameters of the first.
        """
        ...
