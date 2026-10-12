"""DRPP's Gymnasium environment (generic, not subclassed per system).

**This subpackage serves scene 2 only** -- "custom simulated ambiguity": you declare an
ambiguity set by hand, then generate dynamics from it and simulate. Scene 1 (applying DRPP
to **your own** gymnasium environment) does not need it, only
:mod:`ppsds.probabilistic_predictor` + :mod:`ppsds.scoring_rule`.

``Env`` is assembled from two things:

- an **ambiguity set** (:class:`~ppsds.ambiguity.additive_drift.conic_moment.AdditiveDriftConicMomentSet`
  and its subclasses) -- it supplies the nominal model, the state/control dimensions, and
  "how to advance one step given ``sds_params``";
- a :class:`~ppsds.environment.dynamics_mode.DynamicsMode` -- it decides which set of
  ``sds_params`` each step uses.

The key set of ``sds_params`` is **declared by the ambiguity set itself**
(``param_keys``) and checked at the ``step()`` entry point, which **fails loudly** on a
mismatch -- rather than the old behaviour of guessing the format via ``'A' in sds_params``
and silently taking a different branch when the guess was wrong.
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

from .dynamics_mode import DynamicsMode


class Env(gym.Env):
    """Gymnasium environment for an SDS, assembled from an ambiguity set + a DynamicsMode.

    **Why it is called ``Env`` and not ``DRPPEnv``**: the qualifier "DRPP" carries no
    information (the whole package is DRPP) but does suggest adversarialism is intrinsic to
    it. In fact adversarialism is only one of four modes -- ``FixedDynamics`` /
    ``RandomDynamics`` / ``PreSampledDynamics`` are not adversarial at all. And
    ``ppsds.environment.Env`` already states where the class belongs.

    **Spaces**

    - ``observation_space``: ``Box(shape=(x_dim,), dtype=float64)``
    - ``action_space``: ``Box(shape=(u_dim,), dtype=float64)``

    **step() returns**

    - ``observation``: next state :math:`x_{k+1}`
    - ``reward``: always 0.0 (DRPP prediction quality is assessed externally)
    - ``terminated``: ``True`` when state goes out of bounds
    - ``truncated``: ``True`` after ``max_episode_steps``
    - ``info``: ``{'sds_params': {...}, 'mode': str}``

    Parameters
    ----------
    ambiguity_set : AdditiveDriftConicMomentSet
        Supplies the nominal model (``f_bar`` / ``mu_bar`` / ``Sigma_bar``), the
        dimensions, :meth:`transition` and :attr:`param_keys`.
    mode : DynamicsMode
        Strategy selecting ``sds_params`` each step.
    x0 : ndarray of shape (x_dim,), optional
        Initial state.  If ``None``, defaults to zeros.
    max_steps : int, default=50
        Episode length (truncation horizon).
    bounds : tuple of (low, high) ndarrays, optional
        State bounds for early termination.  ``None`` means no bounds check.
    """

    metadata = {"render_modes": []}

    def __init__(self, ambiguity_set, mode: DynamicsMode,
                 x0: np.ndarray | None = None,
                 max_steps: int = 50,
                 bounds: tuple | None = None):
        super().__init__()

        self._abg = ambiguity_set
        self._mode = mode
        self._max_steps = max_steps
        self._bounds = bounds

        # Initial state
        if x0 is None:
            self._x0 = np.zeros(self._abg.x_dim)
        else:
            self._x0 = np.asarray(x0, dtype=np.float64)

        # Spaces
        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(self._abg.x_dim,), dtype=np.float64,
        )
        self.action_space = Box(
            low=-np.inf, high=np.inf,
            shape=(self._abg.u_dim,), dtype=np.float64,
        )

        # Runtime state (set in reset())
        self.x = self._x0.copy()
        self._step_count = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Start a new episode.

        Parameters
        ----------
        seed : int, optional
        options : dict, optional
            May contain ``'x0'`` to override the initial state.

        Returns
        -------
        observation, info
        """
        super().reset(seed=seed)

        if options is not None and 'x0' in options:
            self._x0 = np.asarray(options['x0'], dtype=np.float64)

        self._mode.reset(self.np_random, self._max_steps)
        self._step_count = 0
        self.x = self._x0.copy()

        return self.x, {}

    def step(self, action):
        """Advance the simulation by one step.

        Parameters
        ----------
        action : ndarray of shape (u_dim,)
            Control input.

        Returns
        -------
        observation, reward, terminated, truncated, info
        """
        action = np.asarray(action, dtype=np.float64)

        # 1. Determine SDS parameters for this step
        sds_params = self._mode.get_params(self._step_count)

        # 2. Validate the parameter format: the schema is declared by the ambiguity set,
        #    and a mismatch fails loudly.
        self._abg.validate(sds_params)

        # 3. Compute next state
        self.x = self._abg.transition(self.x, action, sds_params, self.np_random)

        # 3. Termination / truncation
        terminated = self._is_out_of_bounds(self.x)
        self._step_count += 1
        truncated = self._step_count >= self._max_steps

        return self.x, 0.0, terminated, truncated, {
            'sds_params': sds_params,
            'mode': type(self._mode).__name__,
        }

    def _is_out_of_bounds(self, x: np.ndarray) -> bool:
        """Check whether the state is outside the allowed bounds."""
        if self._bounds is None:
            return False
        low, high = self._bounds
        return bool(np.any(x < low) or np.any(x > high))

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def ambiguity_set(self):
        """The underlying ambiguity set object."""
        return self._abg

    @property
    def dynamics_mode(self) -> DynamicsMode:
        """The active dynamics mode strategy."""
        return self._mode

    def inject_eigvec_fixed_info(self, lamb_pred, u, x=None) -> None:
        """Feed the predictor's predicted eigenvalues into the current mode (for the
        "eigenvectors fixed" adversarial mode).

        **The ``eigvec_fixed`` in the name is deliberate**: the only consumer of this
        information is
        :class:`~ppsds.environment.dynamics_mode.EigvecFixedAdversarialDynamics`
        -- it delivers the Eig-DRPP bound (``value_lower_bound_eigvec_fixed``). Calling it
        ``inject_adversarial_info`` would present this **very narrow** use as a general one.

        This is the **only** public channel from the agent side to a
        :class:`DynamicsMode`. It is necessary because of the **paper's semantics**: the
        adversarial SDS must "pick the worst SDS from the predicted covariance **after**
        the predictor emits its pdf" -- i.e. a best response, not a fixed strategy
        independent of the predictor. So it is not an architectural defect.

        For modes that do not consume predictor information (``FixedDynamics`` /
        ``RandomDynamics`` / ``PreSampledDynamics``) the base ``set_info`` is a no-op, so
        this can be called unconditionally.

        Parameters
        ----------
        lamb_pred : ndarray of shape (x_dim,) or None
            Eigenvalues of the predictive covariance. ``None`` skips the injection -- on
            the first ``step()`` there is no prediction yet, and the mode falls back to
            nominal parameters internally.
        u : ndarray of shape (u_dim,)
            Control input for this step.
        x : ndarray of shape (x_dim,), optional
            State for this step; defaults to ``self.x``. Passing it explicitly makes the
            call site self-explanatory and avoids depending on the implementation detail
            of when the state advances.
        """
        if lamb_pred is None:
            return
        self._mode.set_info(
            np.asarray(lamb_pred),
            self.x if x is None else np.asarray(x),
            np.asarray(u),
        )
