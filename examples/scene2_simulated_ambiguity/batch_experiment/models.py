"""The model table for the batch experiment, and the figure configuration.

**This module is the single source of truth for the model parameters.** One table covers all
seven models -- five linear plus two nonlinear. What differs between systems is carried by
the **ambiguity-set subclass** (``LinearAmbiguitySet`` / ``NonlinearAmbiguitySet``), which is
why every model is run through the same generic ``Env``.

Two things live here rather than in ``run.py``, so that ``run.py`` reads as flow:

- :func:`build` -- the model registry (parameter literals + cache signature);
- :data:`PLOT_SPECS` / :data:`ERROR_BAR_SPECS` -- the figure configuration (trajectory index
  ``k``, axis limits, inset window) that used to be a nest of ``if/elif`` in the visualizer.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import block_diag

from ppsds.ambiguity.additive_drift.conic_moment import (
    LinearAmbiguitySet,
    NonlinearAmbiguitySet,
)
from ppsds.dynamics.mechanical import NLink


#: Models that get the paper's ellipse figures, and the only ones that support `regulation`.
LINEAR_MODELS = ('final_round', 'original', 'larger_x0',
                 'different_Sigma_bar', 'different_system_convergence')

#: The two models that do not get ellipse figures. Note that ``higher_dim`` still uses the
#: linear A/B ambiguity-set format -- "nonlinear" here means "not one of the five paper
#: linear models", not "uses NonlinearAmbiguitySet".
NONLINEAR_MODELS = ('NLink', 'higher_dim')

ALL_MODELS = LINEAR_MODELS + NONLINEAR_MODELS


# ``traj.pkl``'s key set differs by model family: NLink stores an additive drift, the
# linear-format models go through the A/B representation.
NLINK_TRAJ_KEYS = ['x', 'u', 'delta_f', 'mu', 'Sigma']
LINEAR_TRAJ_KEYS = ['x', 'u', 'A', 'B', 'mu', 'Sigma']


@dataclass(frozen=True, eq=False)
class ModelSpec:
    """Everything about one model that the runner needs.

    Attributes
    ----------
    ambig : object
        The constructed ambiguity set (``LinearAmbiguitySet`` or ``NonlinearAmbiguitySet``).
    x0 : ndarray
        Initial state.
    path : str
        Output-directory template under ``data/gym/``. Deliberately **not** unified across
        models, so that each model's existing ``data/`` directory still matches.
    traj_keys : list of str
        Keys stored in ``traj.pkl`` for this family.
    signature : dict
        Parameter snapshot fed to the cache fingerprint. It **must list explicitly** every
        constant that determines the output: NLink's ``dt`` / ``gamma_0`` / ``f_err_max`` are
        bound into lambda default arguments and are **not** attributes of ``abg``, so
        automatic extraction cannot see them -- omitting one means "changed the parameter
        but the cache did not invalidate".
    is_ellipse_model : bool
        Whether this model gets the ellipse figures (and supports ``regulation``).
    """

    ambig: object
    x0: np.ndarray
    path: str
    traj_keys: list
    signature: dict
    is_ellipse_model: bool


# ----------------------------------------------------------------------
# Linear-format model parameters
# ----------------------------------------------------------------------
# One entry per linear-format model. The expressions are kept in the exact form they were
# originally written in, because that also fixes the arrays' dtypes (``higher_dim``'s
# ``B_bar`` and ``mu_bar`` are integer arrays, its ``f_err_max`` is the int 5).

_LINEAR_PARAMS = {
    'higher_dim': dict(
        x0=np.array([2, 1, 3, 4]),
        alphas={'A': 0.3, 'B': 0.3, 'mu': 0.5, 'Sigma': 3},
        A_bar=np.array([[1, 0.1, 0, 0], [0, 1, 0, 0],
                        [0, 0, 1, 0.2], [0, 0, 0, 1]]),
        B_bar=np.array([[1, 0], [0, 1], [1, 0], [0, 1]]),
        mu_bar=np.array([0, 0, 0, 0]),
        Sigma_bar=np.array([[1, 0.5, 0, 0], [0.5, 1.5, 0, 0],
                            [0, 0, 2, 0.3], [0, 0, 0.3, 1]]),
        f_err_max=5,
    ),
    'original': dict(
        x0=np.array([2.0, 1.0]),
        alphas={'A': 0.3, 'B': 0.3, 'mu': 1.35, 'Sigma': 4},
        A_bar=0.9802 * np.identity(2) + np.diag([0.0196], 1),
        B_bar=np.identity(2),
        mu_bar=np.array([0.0, 0.0]),
        Sigma_bar=np.array([[1.9608, 0.0195], [0.0195, 1.9605]]),
        f_err_max=5.0,
    ),
    'larger_x0': dict(
        x0=np.array([0.0, 0.0]),
        alphas={'A': 0.3, 'B': 0.3, 'mu': 1.35, 'Sigma': 4},
        A_bar=0.9802 * np.identity(2) + np.diag([0.0196], 1),
        B_bar=np.identity(2),
        mu_bar=np.array([0.0, 0.0]),
        Sigma_bar=np.array([[1.9608, 0.0195], [0.0195, 1.9605]]),
        f_err_max=5.0,
    ),
    'different_Sigma_bar': dict(
        x0=np.array([2.0, 1.0]),
        alphas={'A': 0.1, 'B': 0.3, 'mu': 0.5, 'Sigma': 3},
        A_bar=0.9802 * np.identity(2) + np.diag([0.0196], 1),
        B_bar=np.identity(2),
        mu_bar=np.array([0.0, 0.0]),
        Sigma_bar=np.array([[1.0, 0.5], [0.5, 1.5]]),
        f_err_max=5.0,
    ),
    'different_system_convergence': dict(
        x0=np.array([2.0, 1.0]),
        alphas={'A': 0.3, 'B': 0.1, 'mu': 0.5, 'Sigma': 3},
        A_bar=np.array([[1.0, 0.1], [0.0, 1.0]]),
        B_bar=np.identity(2),
        mu_bar=np.array([0.0, 0.0]),
        Sigma_bar=np.array([[1.0, 0.5], [0.5, 1.5]]),
        f_err_max=5.0,
    ),
    'final_round': dict(
        x0=np.array([2.0, 1.0]),
        alphas={'A': 0.3, 'B': 0.3, 'mu': 0.5, 'Sigma': 3},
        A_bar=np.array([[1.0, 0.1], [0.0, 1.0]]),
        B_bar=np.identity(2),
        mu_bar=np.array([0.0, 0.0]),
        Sigma_bar=np.array([[1.0, 0.5], [0.5, 1.5]]),
        f_err_max=5.0,
    ),
}


def _build_linear(model: str, control_type: str) -> ModelSpec:
    p = _LINEAR_PARAMS[model]
    alphas = p['alphas']
    abg = LinearAmbiguitySet(p['A_bar'], p['B_bar'], p['mu_bar'], p['Sigma_bar'],
                             alphas, p['f_err_max'])
    return ModelSpec(
        ambig=abg,
        x0=p['x0'],
        path=(f"model={model}/"
              f"a={alphas['A']}_{alphas['B']}_{alphas['mu']}_{alphas['Sigma']}/"
              f"f_err_max={p['f_err_max']}/control={control_type}"),
        traj_keys=LINEAR_TRAJ_KEYS,
        signature={'kind': 'linear', 'A_bar': p['A_bar'], 'B_bar': p['B_bar'],
                   'mu_bar': p['mu_bar'], 'Sigma_bar': p['Sigma_bar'],
                   'alphas': alphas, 'f_err_max': p['f_err_max']},
        is_ellipse_model=model in LINEAR_MODELS,
    )


def _build_nlink(control_type: str) -> ModelSpec:
    """The 6-link arm. Its ambiguity set holds lambdas, which is why the runner must fork."""
    n_link = NLink(n=6)
    x0 = np.zeros(n_link.x_dim)
    dt = 0.1
    gamma_0 = 0.3 * (dt ** 2)
    gamma_1 = 0.5 * (dt ** 2)
    gamma_2 = 3.0
    f_err_max = 5 * dt
    gamma = [{
        'function': lambda x, u, g0=gamma_0, fm=f_err_max:
            np.min([g0 * (np.linalg.norm(x) ** 2 + np.linalg.norm(u) ** 2),
                    fm ** 2]),
        'x_dim': n_link.x_dim, 'u_dim': n_link.u_dim, 'max': f_err_max,
    }, gamma_1, gamma_2]
    f_bar = {
        'function': lambda x, u, mdl=n_link, dt=dt: mdl.discretize(x, u, dt),
        'x_dim': n_link.x_dim, 'u_dim': n_link.u_dim,
    }
    mu_bar = np.zeros(n_link.x_dim)
    Sigma_bar = block_diag(
        *[np.array([[1.0, 0.5], [0.5, 1.5]]) for _ in range(n_link.n)]
    ) * (dt ** 2)
    odd_indices = np.arange(1, 2 * n_link.n, 2)
    even_indices = np.arange(0, 2 * n_link.n, 2)
    new_order = np.concatenate([even_indices, odd_indices])
    Sigma_bar = Sigma_bar[new_order][:, new_order]
    return ModelSpec(
        ambig=NonlinearAmbiguitySet(f_bar, mu_bar, Sigma_bar, gamma),
        x0=x0,
        path=(f"model=NLink_n={n_link.n}/control={control_type}/"
              f"gamma={gamma_0}_{gamma_1}_{gamma_2}/dt={dt}"),
        traj_keys=NLINK_TRAJ_KEYS,
        signature={'kind': 'nlink', 'n': n_link.n, 'dt': dt,
                   'gamma_0': gamma_0, 'gamma_1': gamma_1,
                   'gamma_2': gamma_2, 'f_err_max': f_err_max,
                   'mu_bar': mu_bar, 'Sigma_bar': Sigma_bar},
        is_ellipse_model=False,
    )


def build(model: str, control_type: str) -> ModelSpec:
    """Construct the model bundle for ``model`` under the given control type."""
    if model == 'NLink':
        return _build_nlink(control_type)
    if model in _LINEAR_PARAMS:
        return _build_linear(model, control_type)
    raise ValueError(f"Unknown model_type: {model}")


# ----------------------------------------------------------------------
# Figure configuration
# ----------------------------------------------------------------------

#: kwargs for ``plot_error_bar``, keyed by ``(experiment, is_ellipse_model)``.
#: The random branches use the inset zoom; the worst-case ones do not.
ERROR_BAR_SPECS = {
    ('random', True):  dict(ylim=[-28, -1.5], show_local=True),
    ('random', False): dict(show_local=False),
    ('worst', True):   dict(ylim=[-28, -1.5], show_local=False),
    ('worst', False):  dict(show_local=False),
}

#: Spec for the ellipse panel, keyed by ``(experiment, control, period)``. ``period`` is
#: ``None`` where the figure does not depend on it (every ``regulation`` run, and every
#: ``worst`` run). The trajectory index ``k`` was hardcoded per case in the original script;
#: keeping it in a table means a new model cannot silently miss a configuration. Note that
#: the plotting functions clamp out-of-range indices with a ``RuntimeWarning``, so a small
#: ``-t`` no longer crashes the plotting stage.
PLOT_SPECS = {
    ('random', 'none', 100): dict(
        time_indexes=[1, 16, 31], k=2,
        xlim=[-50, 10], ylim=[-45, 95],
        local_ax={'x1': -7.5, 'x2': 6.5, 'y1': -10, 'y2': 7,
                  'ax': [0.1, 0.45, 0.85, 0.4]}),
    ('random', 'none', 1): dict(
        time_indexes=[1, 16, 31], k=82,
        xlim=[-50, 10], ylim=[-45, 95],
        local_ax={'x1': -13, 'x2': 7, 'y1': -16, 'y2': 7,
                  'ax': [0.1, 0.45, 0.85, 0.4]}),
    ('random', 'regulation', None): dict(
        time_indexes=[31], k=1, xlim=[-12, 10], ylim=[-15, 10]),
    ('worst', 'none', None): dict(
        time_indexes=[1, 16, 31], k=7, xlim=[-56, 15], ylim=[-40, 95]),
    ('worst', 'regulation', None): dict(
        time_indexes=[31], k=43, xlim=[-12, 10], ylim=[-15, 10]),
}


def plot_spec(experiment: str, control_type: str,
              period: int) -> Optional[dict]:
    """Ellipse-panel spec for this run, or ``None`` when there is no ellipse panel."""
    if experiment == 'random' and control_type == 'none':
        return PLOT_SPECS.get((experiment, control_type, period))
    return PLOT_SPECS.get((experiment, control_type, None))
