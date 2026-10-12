"""**Identify a linear nominal model** from rollouts, and diagnose the residual.

This is the first step of "connecting DRPP to your own environment": you usually cannot
obtain the system's analytical model, but you can run a few trajectories and fit it
**approximately** to

.. math::

    x_{k+1} \\approx A x_k + B u_k + c + w_k

This module does only **least squares + diagnostics**, **not rigorous system identification**
— the goal is not to find the true values, but to give you a **reasonable** nominal model
plus a set of residual statistics **on which to base a tolerance declaration**.

On "what noise actually is"
---------------------------
A model is merely a human's understanding of some physical phenomenon, and **all models are
wrong**. For this repository's additive SDS:

* :math:`f` is the deterministic law you **believe you understand**;
* :math:`w` is **everything else you have not yet understood**.

Both can be wrong — that is just modeling. So the residual this module computes **need not**
"look like noise" to deserve being called noise; whether it is, and how much uncertainty to
assign it, **is yours to declare**.

.. warning::
   There is a **practical** hurdle (not a theoretical one): DRPP evaluates predictions with a
   **logarithmic score**, and the logarithmic score requires the predictive distribution to
   **have a density**. If you use the residual covariance **as is** for :math:`\\Sigma`, it is
   often **numerically singular** —

   * some state components may be **identically linear** (e.g. the exact Euler step
     ``x' = x + dt*xdot``);
   * observations may lie on a **low-dimensional manifold** (e.g. the robotic-arm observation
     ``(cos θ, sin θ, θdot)`` satisfies ``cos²+sin²=1``, a 2D cylinder in 3D space).

   The support of a degenerate distribution is a null set, so any value even slightly off it
   scores ``-inf`` (measured: at a relative distance of only **1.4e-6** from the support, all
   50 steps still gave ``-inf``).

   So it must be **regularized** to positive definite. That is not a stopgap — **even
   dimensions that look deterministic can be assigned uncertainty**, and that is precisely
   part of modeling. The magnitude is yours to choose, and it directly determines the
   magnitude of the score (see :func:`regularize`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------

def encode_action(space, a) -> np.ndarray:
    """Turn a policy's output into the **action features used for regression**.

    * ``Discrete(n)`` → **one-hot**. Note that the integer itself must **not** be used
      directly: the action indices are nominal (there is no ordering among 0/1/2), and
      treating them as ordinal numbers imposes a linear structure that does not exist,
      inflating the residual.
    * ``Box`` → flatten as is.

    What is returned is the ``u`` inside ``f_bar(x, u)`` — so the predictor's
    ``update(x, u)`` must use the same encoding.
    """
    import gymnasium as gym
    if isinstance(space, gym.spaces.Discrete):
        v = np.zeros(int(space.n), dtype=np.float64)
        v[int(np.asarray(a).reshape(-1)[0])] = 1.0
        return v
    return np.atleast_1d(np.asarray(a, dtype=np.float64)).reshape(-1)


def collect_rollouts(env, policy, n_episodes, T, seed0=0, reset_seed=None):
    """Run several trajectories with a fixed policy; returns ``(X, U, X_next)``.

    Parameters
    ----------
    env : gymnasium.Env
    policy : callable
        ``policy(obs) -> action``. The policy you use throughout.
        Its return value is passed **as is** to ``env.step()`` (a discrete action space
        needs an int, so do not convert it to an array first).
    n_episodes : int
        Number of trajectories.
    T : int
        Maximum number of steps per trajectory.
    seed0 : int
        ``reset(seed=seed0)`` for the first trajectory; subsequent ones use +1 each
        (for reproducibility).
    reset_seed : int, optional
        If given, every trajectory uses the same ``seed`` (for observing the variation
        from **the same initial condition**).

    Returns
    -------
    X, U, X_next : ndarray
        ``U`` is the action features encoded by :func:`encode_action` (one-hot for a
        discrete space), **not** the raw actions.
    """
    X, U, Xn = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=reset_seed if reset_seed is not None else seed0 + ep)
        for _ in range(T):
            a = policy(obs)                      # passed to env as is
            nxt, _r, term, trunc, _info = env.step(a)
            X.append(np.asarray(obs, dtype=np.float64).reshape(-1))
            U.append(encode_action(env.action_space, a))
            Xn.append(np.asarray(nxt, dtype=np.float64).reshape(-1))
            obs = nxt
            if term or trunc:
                break
    return (np.array(X), np.array(U), np.array(Xn))


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------

@dataclass
class LinearFit:
    """The least-squares solution of ``x' ≈ A x + B u + c``."""

    A: np.ndarray
    B: np.ndarray
    c: np.ndarray

    @property
    def x_dim(self) -> int:
        return self.A.shape[0]

    @property
    def u_dim(self) -> int:
        return self.B.shape[1]

    def f_bar(self, x, u) -> np.ndarray:
        """The nominal drift — usable directly as a DRPP predictor's ``f_bar``."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        u = np.asarray(u, dtype=np.float64).reshape(-1)
        return self.A @ x + self.B @ u + self.c

    def predict(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Batch prediction: ``X_next ≈ X @ A.T + U @ B.T + c``."""
        return X @ self.A.T + U @ self.B.T + self.c

    def residuals(self, X, U, X_next) -> np.ndarray:
        """The residual ``w = x_next - f_bar(x, u)``, with the same shape as ``X``."""
        return np.asarray(X_next) - self.predict(X, U)


def fit_linear(X, U, X_next) -> LinearFit:
    """Least-squares fit of ``x' ≈ A x + B u + c``.

    Uses ``lstsq`` rather than the normal equations: ``[X, U, 1]`` is often ill-conditioned
    (state components differ greatly in scale, and some components are identically linear),
    and the normal equations square that conditioning.
    """
    X = np.asarray(X, dtype=np.float64)
    U = np.atleast_2d(np.asarray(U, dtype=np.float64))
    if U.shape[0] != X.shape[0]:
        U = U.reshape(X.shape[0], -1)
    Phi = np.hstack([X, U, np.ones((len(X), 1))])
    Theta, *_ = np.linalg.lstsq(Phi, np.asarray(X_next, dtype=np.float64),
                                rcond=None)
    x_dim, u_dim = X.shape[1], U.shape[1]
    return LinearFit(A=Theta[:x_dim].T,
                     B=Theta[x_dim:x_dim + u_dim].T,
                     c=Theta[x_dim + u_dim])


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

#: Relative threshold (relative to the largest eigenvalue) for deciding "numerically zero eigenvalue"
ZERO_TOL = 1e-10


def diagnose(residuals: np.ndarray, increments: np.ndarray) -> dict:
    """Diagnose the residual — **to inform modeling decisions**, not as a pass/fail criterion.

    Returns
    -------
    dict
        ``residual_ratio``
            Mean residual norm / mean state-increment norm. How large counts as large
            depends on your application.
        ``lag1_autocorr``
            Average of the per-dimension lag-1 autocorrelation. **High** means the residual
            still contains structure the model can explain (i.e. "the model is not there
            yet"); low means it is more like i.i.d. This is only a diagnostic — see the
            module docstring: whether the residual looks like noise does not affect whether
            it can be declared as noise.
        ``zero_directions``
            Number of numerically zero eigenvalues. **> 0 means the covariance is singular
            and must be regularized.**
        ``eigenvalues`` / ``condition_number``
    """
    res = np.asarray(residuals, dtype=np.float64)
    inc = np.asarray(increments, dtype=np.float64)
    r_norm = np.linalg.norm(res, axis=1)
    i_norm = np.linalg.norm(inc, axis=1)
    ac = [np.corrcoef(res[:-1, i], res[1:, i])[0, 1]
          for i in range(res.shape[1]) if res[:, i].std() > 0]
    Sigma = np.cov(res.T)
    ev = np.linalg.eigvalsh(Sigma)
    return {
        'residual_ratio': float(r_norm.mean() / i_norm.mean()) if i_norm.mean() else float('nan'),
        'lag1_autocorr': float(np.mean(ac)) if ac else float('nan'),
        'eigenvalues': ev,
        'condition_number': float(np.linalg.cond(Sigma)),
        'zero_directions': int(np.sum(ev <= ZERO_TOL * max(ev.max(), 1e-300))),
        'mu_bar': res.mean(axis=0),
        'Sigma': Sigma,
    }


def regularize(Sigma, floor: float) -> np.ndarray:
    """Lift a singular covariance to positive definite: ``Sigma + floor * I``.

    **This step is modeling, not a stopgap.** See the module docstring: even dimensions that
    look deterministic can be assigned uncertainty. ``floor`` is exactly "how much uncertainty
    you are willing to admit in those directions".

    ⚠️ The magnitude of ``floor`` **directly determines the magnitude of the score** (because
    it determines the width of the predictive distribution), so it should not be an arbitrary
    number. Two reasonable sources:

    1. **Physics**: perturbations you did not observe, actuator/sensor noise, unmodeled
       high-frequency dynamics;
    2. **Consistency**: pick an initial value, run once, and see whether the resulting score
       magnitude matches your intuition about "how accurate the prediction is", then adjust.

    A negative ``floor`` or an overly large ``floor`` makes the prediction meaningless, but
    **there is no single correct answer** — that is precisely what "modeling" means.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    if floor < 0:
        raise ValueError(f'floor should be >= 0, got {floor}')
    return Sigma + floor * np.eye(Sigma.shape[0])


def format_report(diag: dict, fit: LinearFit | None = None) -> str:
    """Lay out the result of :func:`diagnose` as a human-readable report."""
    lines = []
    if fit is not None:
        lines.append(f'  nominal model: x\' ≈ A x + B u + c   '
                     f'(A {fit.A.shape}, B {fit.B.shape})')
    lines.append(f'  residual/increment     = {diag["residual_ratio"]:8.1%}')
    lines.append(f'  lag-1 autocorrelation  = {diag["lag1_autocorr"]:8.3f}'
                 f'   (high -> the residual still has explainable structure, i.e. the model is not there yet)')
    lines.append(f'  covariance condition number = {diag["condition_number"]:8.2e}')
    lines.append(f'  numerically zero eigenvalues = {diag["zero_directions"]:8d}'
                 f'   (>0 -> singular, **must** be regularized before feeding it to DRPP)')
    lines.append(f'  eigenvalues            = {np.array2string(diag["eigenvalues"], precision=3)}')
    return '\n'.join(lines)
