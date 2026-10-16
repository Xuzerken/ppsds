"""A stochastic dynamical system that is **known exactly**: a drift plus a known noise law.

The counterpart to :mod:`~ppsds.ambiguity.additive_drift.conic_moment`, which describes a system
known only through an ambiguity set. Here there is no uncertainty about the model at all --
``x' = f(x, u) + w`` with ``f`` given and ``w`` drawn from a law you name -- so the noise can be
non-Gaussian, which the ambiguity sets' Gaussian transition cannot express.

This is the object behind the predictability questions ("how well could anything predict this
system?"), and the truthful half of a simulation: an ambiguity set is what a *predictor* believes,
while this is what actually generated the data.

It belongs in ``dynamics/`` rather than ``ambiguity/`` because it is the thing being predicted,
not a belief about it: ``ambiguity/`` is reserved for sets of possible models.
"""

import inspect

import numpy as np


class AdditiveDriftSds:
    """``x' = f(x, u) + w`` with ``f`` known and ``w`` from a named law.

    Parameters
    ----------
    f_bar : callable
        The drift ``f_bar(x, u) -> (x_dim,)``.
    noise_law : object
        The process noise. Must expose ``rvs``, taking ``size`` and a keyword for the random
        source -- either ``rng`` (as
        :class:`~ppsds.numerics.random_variables.multivariate_uniform` and
        :class:`~ppsds.numerics.atom_pmf.AtomPmf` do) or ``random_state`` (as ``scipy.stats``
        distributions do). Which one is detected once, at construction.
    x0 : array_like, optional
        The default initial state. Defaults to the origin of the noise's dimension.

    Raises
    ------
    ValueError
        If ``x_dim`` cannot be determined from ``x0`` or the noise law.
    """

    def __init__(self, f_bar, noise_law, x0=None):
        rvs = getattr(noise_law, 'rvs', None)
        if rvs is None:
            raise TypeError(f'noise_law must expose rvs(), got {type(noise_law).__name__}')
        self.f_bar = f_bar
        self.noise_law = noise_law
        self._noise_kwarg = _random_source_kwarg(rvs)

        if x0 is not None:
            x0 = np.asarray(x0, dtype=float).reshape(-1)
        self.x_dim = _dimension_of(noise_law, x0)
        if x0 is None:
            x0 = np.zeros(self.x_dim)
        elif x0.shape != (self.x_dim,):
            raise ValueError(
                f'x0 has {x0.shape[0]} coordinate(s) but the noise law has {self.x_dim}; an '
                'additive-drift system moves in the noise\'s own dimension'
            )
        self.x0 = x0

    def transition(self, x, u, rng=None):
        """One step: the drift, plus a fresh noise draw. ``(x_dim,)``.

        Parameters
        ----------
        x, u : array_like
            State and control.
        rng : numpy.random.Generator, optional
            Source of randomness for the noise draw; ``None`` uses the global ``np.random``
            state. Pass one for a reproducible rollout.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        drift = np.asarray(self.f_bar(x, np.asarray(u, dtype=float).reshape(-1)), dtype=float)
        if drift.shape != (self.x_dim,):
            raise ValueError(
                f'f_bar returned shape {drift.shape}, expected ({self.x_dim},); a mis-shaped '
                'drift would broadcast against the state instead of adding to it'
            )
        return drift + self._draw_noise(1, rng)[0]

    def sample_trajectory(self, T, rng=None, x0=None, u=None):
        """Roll out ``T`` steps, returning both the states and the noise that drove them.

        The noise is returned alongside because it is often the quantity of interest: when the
        dynamics are known to the predictor, the one-step prediction reduces to a prediction of
        the *noise*, so scoring happens on ``w`` rather than on ``x``.

        Parameters
        ----------
        T : int
            Number of steps. The returned states cover ``T + 1`` time points, the noise ``T``.
        rng : numpy.random.Generator, optional
            Source of randomness; ``None`` uses the global state.
        x0 : array_like, optional
            Initial state; defaults to the one given at construction.
        u : array_like, optional
            The control. Either constant, shaped ``(u_dim,)``, or per step, ``(T, u_dim)``.
            Defaults to the **empty** vector, which is what an uncontrolled system wants; a
            system whose drift does ``B @ u`` will then fail loudly on the shape rather than
            quietly simulate a control nobody asked for.

        Returns
        -------
        dict
            ``{'x': (T + 1, x_dim), 'w': (T, x_dim)}``.
        """
        T = int(T)
        if T < 0:
            raise ValueError(f'T must be >= 0, got {T}')
        state = self.x0 if x0 is None else np.asarray(x0, dtype=float).reshape(-1)
        if state.shape != (self.x_dim,):
            raise ValueError(f'x0 must be ({self.x_dim},), got shape {state.shape}')
        controls = _control_per_step(u, T)

        noise = self._draw_noise(T, rng)
        states = np.empty((T + 1, self.x_dim))
        states[0] = state
        for step in range(T):
            drift = np.asarray(self.f_bar(states[step], controls[step]), dtype=float)
            if drift.shape != (self.x_dim,):
                raise ValueError(
                    f'f_bar returned shape {drift.shape}, expected ({self.x_dim},); a mis-shaped '
                    'drift would broadcast against the state instead of adding to it'
                )
            states[step + 1] = drift + noise[step]
        return {'x': states, 'w': noise}

    def _draw_noise(self, size, rng):
        """Draw ``size`` noise samples as ``(size, x_dim)``, whatever the law's keyword is."""
        if size == 0:
            return np.empty((0, self.x_dim))
        return np.asarray(self.noise_law.rvs(size=size, **{self._noise_kwarg: rng}), dtype=float)

    def __repr__(self):
        return f'AdditiveDriftSds(x_dim={self.x_dim}, noise={type(self.noise_law).__name__})'


def _control_per_step(u, T):
    """``(T, u_dim)`` controls from either a constant ``(u_dim,)`` or a per-step ``(T, u_dim)``.

    ``None`` means no control, i.e. the empty vector.
    """
    if u is None:
        return np.zeros((T, 0))
    u = np.asarray(u, dtype=float)
    if u.ndim == 0:
        return np.full((T, 1), float(u))
    if u.ndim == 1:
        return np.tile(u, (T, 1))
    if u.ndim == 2 and u.shape[0] == T:
        return u
    raise ValueError(
        f'u must be constant (u_dim,) or per step (T, u_dim) with T={T}, got shape {u.shape}'
    )


def _random_source_kwarg(rvs):
    """Whether ``rvs`` takes its random source as ``rng`` (ours) or ``random_state`` (scipy's).

    Detected by signature rather than by trying one and catching ``TypeError``, so that a
    genuine error raised *inside* ``rvs`` is not mistaken for a signature mismatch.
    """
    try:
        parameters = inspect.signature(rvs).parameters
    except (TypeError, ValueError):  # builtins and C extensions have no signature
        return 'random_state'
    if 'rng' in parameters:
        return 'rng'
    return 'random_state'


def _dimension_of(noise_law, x0):
    """The state dimension: the noise law's, or ``x0``'s if the law does not declare one."""
    for attribute in ('dimensions', 'dim'):
        value = getattr(noise_law, attribute, None)
        if value is not None:
            return int(value)
    mean = getattr(noise_law, 'mean', None)
    if mean is not None:
        return int(np.size(mean))
    if x0 is not None:
        return int(x0.shape[0])
    raise ValueError(
        f'cannot determine the state dimension: {type(noise_law).__name__} declares no '
        'dimensions/dim/mean, and no x0 was given. Pass x0.'
    )
