"""A Gymnasium environment for a system that is known exactly.

Sibling of :mod:`~ppsds.environment.base`, and the third thing this subpackage serves: where
``Env`` composes an *ambiguity set* with a :class:`~ppsds.environment.dynamics_mode.DynamicsMode`,
:class:`KnownSdsEnv` steps an :class:`~ppsds.dynamics.known_sds.AdditiveDriftSds` directly. No
ambiguity set is involved, so a non-Gaussian noise law -- which the ambiguity sets' Gaussian
transition cannot express -- is exactly what this environment is for.

It exists so that "set up an environment, predict, score" is one story across the package: the
same ``reset(seed=...)`` / ``step(action)`` cycle the rest of the library uses, on a system whose
truth is known, which is what the predictability questions are asked about.
"""

import numpy as np
import gymnasium as gym

from ..dynamics.known_sds import AdditiveDriftSds


class KnownSdsEnv(gym.Env):
    """Steps a known :class:`~ppsds.dynamics.known_sds.AdditiveDriftSds`.

    Parameters
    ----------
    sds : AdditiveDriftSds
        The system. Its ``noise_law`` and ``f_bar`` stay reachable through this environment, so a
        predictor and the predictability reference lines can be built from the same object.
    x0 : array_like, optional
        Initial state; defaults to the system's own.
    max_steps : int
        Horizon after which ``step`` reports ``truncated``. As in
        :class:`~ppsds.environment.base.Env`, nothing is ever ``terminated``: these systems run
        forever, they are just being simulated for a while.

    Attributes
    ----------
    x : ndarray or None
        The current state, public as it is on :class:`~ppsds.environment.base.Env`. ``None``
        before the first ``reset``.
    """

    def __init__(self, sds, x0=None, max_steps=50):
        super().__init__()
        if not isinstance(sds, AdditiveDriftSds):
            raise TypeError(
                f'sds must be an AdditiveDriftSds, got {type(sds).__name__}. To step an '
                'ambiguity set instead, use ppsds.environment.Env.'
            )
        self.sds = sds
        start = sds.x0 if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        if start.shape != (sds.x_dim,):
            raise ValueError(f'x0 must be ({sds.x_dim},), got shape {start.shape}')
        self._x0 = start
        self._max_steps = int(max_steps)
        self.x = None
        self._step_count = 0

    @property
    def noise_law(self):
        """The process-noise law, for building scoring rules and predictability references."""
        return self.sds.noise_law

    @property
    def f_bar(self):
        """The known drift."""
        return self.sds.f_bar

    @property
    def x0(self):
        """The initial state this environment resets to."""
        return self._x0

    def reset(self, seed=None, options=None):
        """Start a fresh episode; ``options['x0']`` overrides the initial state."""
        super().reset(seed=seed)
        start = self._x0
        if options is not None and options.get('x0') is not None:
            start = np.asarray(options['x0'], dtype=float).reshape(-1)
            if start.shape != (self.sds.x_dim,):
                raise ValueError(
                    f"options['x0'] must be ({self.sds.x_dim},), got shape {start.shape}"
                )
        self.x = np.array(start, dtype=float)
        self._step_count = 0
        return self.x, {}

    def step(self, action):
        """Advance one step; returns ``(state, reward, terminated, truncated, info)``.

        The reward is always ``0.0``: this is a prediction environment, and what a prediction
        earns is measured by a scoring rule, not by a reward signal.
        """
        if self.x is None:
            raise RuntimeError('step() called before reset(); call reset(seed=...) first')
        self.x = self.sds.transition(self.x, action, rng=self.np_random)
        self._step_count += 1
        truncated = self._step_count >= self._max_steps
        return self.x, 0.0, False, truncated, {}
