"""The small fixed systems the test suite runs on -- the single source of truth.

Two systems, both deliberately tiny (2 states, 1 input) so that one cone program costs
milliseconds rather than seconds:

- :func:`linear_ambiguity_set` -- the same parameters
  ``examples/scene1_own_environment/quickstart.py`` declares, so the tests and the first example
  talk about the same system rather than two lookalikes.
- :func:`nonlinear_ambiguity_set` -- a damped pendulum, standing in for the
  ``NonlinearAmbiguitySet`` path without dragging in the 12-state ``NLink``.

Every factory builds **fresh arrays**. The ambiguity sets keep references to what they are
handed (``np.asarray`` does not copy), so handing out a module-level singleton would let one
test's mutation leak into the next.
"""

from __future__ import annotations

import numpy as np

from ppsds.ambiguity.additive_drift.conic_moment import (
    LinearAmbiguitySet,
    NonlinearAmbiguitySet,
)

#: The uncertainty radii of the linear system: drift perturbation size, then the noise-mean
#: and covariance radii. Note that ``mu`` is a **squared** radius while ``A`` / ``B`` are not.
LINEAR_ALPHAS = {'A': 0.30, 'B': 0.30, 'mu': 0.25, 'Sigma': 3.0}

#: The hard, **un-squared** bound on the drift norm ``||delta_f||``.
LINEAR_F_ERR_MAX = 0.30

#: Where the linear system starts.
LINEAR_X0 = np.array([2.0, 1.0])

#: The true system behind the linear tests: deliberately offset from the declared nominal
#: model, so that robustification has something to buy. Matches ``quickstart.py``.
LINEAR_A_TRUE = np.array([[1.0, 0.14], [0.0, 1.0]])
LINEAR_B_TRUE = np.array([[1.0], [0.0]])
LINEAR_SIGMA_TRUE = np.array([[1.6, 0.7], [0.7, 2.0]])

#: The nonlinear system's integration step. Bound into the lambdas below as a default argument
#: rather than closed over, the same idiom the batch runner's model table uses.
NONLINEAR_DT = 0.1

#: Where the nonlinear system starts. Deliberately away from the pendulum's equilibrium:
#: at ``x = 0`` the drift radius ``gamma[0]`` is zero, which collapses the cone program into a
#: degenerate (and much less interesting) case.
NONLINEAR_X0 = np.array([0.3, 0.0])


def linear_ambiguity_set(f_err_max=LINEAR_F_ERR_MAX):
    """A fresh ``LinearAmbiguitySet`` on the two-state system from the scene-1 example.

    Parameters
    ----------
    f_err_max : float
        The hard, un-squared bound on the drift norm. ``0`` is a meaningful choice rather than a
        degenerate one: it switches the drift uncertainty off entirely, which turns Eig-DRPP into
        a case with a closed-form answer (see ``test_golden``).
    """
    return LinearAmbiguitySet(
        np.array([[1.0, 0.10], [0.0, 1.0]]),      # A_bar
        np.array([[1.0], [0.0]]),                 # B_bar
        np.zeros(2),                              # mu_bar
        np.array([[1.0, 0.45], [0.45, 1.30]]),    # Sigma_bar
        dict(LINEAR_ALPHAS),
        f_err_max,
    )


def nonlinear_drift(x, u, dt=NONLINEAR_DT):
    """A damped pendulum: ``x = [angle, angular rate]``, ``u = [torque]``."""
    theta, omega = x
    return np.array([
        theta + dt * omega,
        omega + dt * (-np.sin(theta) - 0.5 * omega + u[0]),
    ])


def nonlinear_ambiguity_set(legacy_sampler=False):
    """A fresh ``NonlinearAmbiguitySet`` on a damped pendulum.

    Parameters
    ----------
    legacy_sampler : bool
        Passed through to the ambiguity set; selects the old (bound-violating) drift sampler.
        Only the reproduction path should ask for it.
    """
    x_dim, u_dim = 2, 1
    f_err_max = 1.0
    gamma_0 = 0.05

    def f_err(x, u, g0=gamma_0, fm=f_err_max):
        return min(g0 * (np.linalg.norm(x) ** 2 + np.linalg.norm(u) ** 2), fm ** 2)

    gamma = [
        {'function': f_err, 'x_dim': x_dim, 'u_dim': u_dim, 'max': f_err_max},
        0.02,   # gamma[1]: squared radius of the noise mean
        2.0,    # gamma[2]: covariance scaling factor
    ]
    f_bar = {'function': nonlinear_drift, 'x_dim': x_dim, 'u_dim': u_dim}
    return NonlinearAmbiguitySet(
        f_bar,
        np.zeros(x_dim),
        np.array([[1.0, 0.3], [0.3, 1.2]]),
        gamma,
        legacy_sampler=legacy_sampler,
    )


def true_step(x, u, rng):
    """One step of the *true* linear system behind the golden scenario.

    The nominal model the tests declare is ``A_bar`` / ``Sigma_bar``; this is what actually
    generates the observations. Deterministic given ``rng``, which is what makes the golden
    fixture reproducible without touching global random state.
    """
    mean = LINEAR_A_TRUE @ x + LINEAR_B_TRUE @ u
    return rng.multivariate_normal(mean, LINEAR_SIGMA_TRUE)


def control_at(i):
    """A deterministic, non-trivial input sequence (zero input hides sign errors)."""
    return np.array([0.1 * np.cos(i)])
