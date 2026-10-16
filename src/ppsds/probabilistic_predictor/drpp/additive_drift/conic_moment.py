"""DRPP predictors over an additive-drift + conic-moment ambiguity set.

The path ``probabilistic_predictor / drpp / additive_drift / conic_moment`` states the
scope layer by layer: **DRPP family** / **additive-drift assumption** / **conic-moment
ambiguity set**. Another ambiguity-set family gets a module on the same level --
:func:`eigvec_fixed_minmax` solves equation (19) of the paper and **holds only for the
conic-moment ambiguity set**; a different family needs a predictor of its own.

Three predictors
----------------
All three share the same private skeleton :class:`_ConicMomentPredictor`; their **only**
substantive difference is "where the predictive covariance comes from":

=================================  ==================================================
:class:`NominalPredictor`          Nominal: ``Sigma_pred = Sigma_bar``
:class:`NoiseDrpp`                 Noise-robust only: ``Sigma_pred = gamma[2] * Sigma_bar``
:class:`EigDrpp`                   The paper's Eig-DRPP: solves :func:`eigvec_fixed_minmax` each step
=================================  ==================================================

.. note::
   **Noise-DRPP now has a class of its own.** It used to have none and was stood in for by
   "handing the nominal predictor an inflated ``gamma[2] * Sigma_bar``" -- two roles mixed
   together, and the name gave nothing away. This class internally **computes that product
   first and then hands it to the skeleton** (the same expression as the old code), so the
   numbers are bit-for-bit unchanged.

Core optimization
-----------------
:func:`eigvec_fixed_minmax` -- solves equation (19) of the paper: it turns the functional
maximin into a Euclidean cone program (CVXPY + SCS) and returns
``(optimal eigenvalues, optimal value, worst direction j)``.

Relation to the predictor contract
----------------------------------
All three classes **satisfy**
:class:`~ppsds.probabilistic_predictor.base.ProbabilisticPredictor` (the three methods
``predict`` / ``update`` / ``reset``, and the three attributes ``x_pred`` / ``Sigma_pred`` /
``lamb_pred``) but do **not inherit from it** -- that is a ``typing.Protocol``, with zero
runtime overhead and recognition by structure. This repository has tripped over "providing
an interface" before: the earlier ABC wrote the attributes as ``@property`` (data
descriptors) while the implementing classes used plain attribute assignment, so to actually
inherit it would raise **``AttributeError`` at construction time**, and
``ABCMeta.register()`` hid that incompatibility as well. **If you give an interface you have
to be able to keep it**, so here we deliberately keep a structural contract and add no
inheritance.

Usage
-----
::

    from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (
        NominalPredictor, NoiseDrpp, EigDrpp,
    )

    pred = EigDrpp(f_bar, mu_bar, Sigma_bar, gamma)
    pred.update(x, u)      # solve one cone program
    pred.predict()         # this step's predictive distribution
    pred.lamb_pred         # predictive covariance eigenvalues -- the adversarial
                           # dynamics selects the worst direction from these
"""

import numpy as np
import warnings
from scipy.stats import multivariate_normal

from ....numerics.linalg import eig_real
from ..._inputs import as_vector


class _ConicMomentPredictor:
    """The skeleton shared by the three predictors (**private**, not exported).

    Subclasses supply only two things -- and those are the **only** substantive difference
    between the three:

    - :meth:`_drift`: this step's predictive drift;
    - :meth:`_set_covariance`: where the predictive covariance and its eigenvalues come
      from.

    Attributes
    ----------
    x_pred, Sigma_pred : ndarray
        The current predicted state and covariance.
    lamb_pred : ndarray
        The eigenvalues of ``Sigma_pred``. **The adversarial dynamics selects the worst
        direction from these**, so every subclass must fill them in correctly.
    """

    #: Whether the predictive drift contains an "additive bias" term ``self.bias``.
    #: ``True`` for nominal and Noise-DRPP (the time-varying-parameter path of
    #: ``trajectory_prediction`` writes it); ``False`` for Eig-DRPP (that path simply has
    #: no corresponding env-layer implementation).
    _has_bias = True

    def __init__(self, f_bar, mu_bar, Sigma_bar, multivariate_rv=multivariate_normal):
        # -- nominal model --
        self.f_bar = f_bar
        self.mu_bar = mu_bar
        self.Sigma_bar = Sigma_bar
        self.lamb, self.v = eig_real(self.Sigma_bar)
        # Restore point: trajectory_prediction **rewrites** the three quantities above in
        # place, and reset() uses this to put them back
        self._nominal = (f_bar, mu_bar, Sigma_bar)

        # -- predictive distribution family --
        self.multivariate_rv = multivariate_rv

        # -- internal state --
        self.x = None          # current observed state
        self.u = None          # current control input
        self.bias = None       # additive bias (used by the time-varying-parameter path; see _has_bias)

        # -- prediction output --
        self.nu_pred = None     # predictive drift
        self.mu_pred = None     # predictive noise mean
        self.x_pred = None      # predicted state = nu_pred + mu_pred
        self.Sigma_pred = None  # predictive covariance
        self.lamb_pred = None   # eigenvalues of the predictive covariance

    def predict(self):
        """Return the predictive distribution for the current step.

        Returns
        -------
        scipy distribution object
            Parameterized with ``x_pred`` as the mean and ``Sigma_pred`` as the
            covariance.
        """
        return self.multivariate_rv(self.x_pred, self.Sigma_pred)

    def update(self, x_observe=None, u_observe=None):
        """Advance one step after observing the state and control.

        The common part (drift / mean / predicted state) is computed here; the covariance
        is left to :meth:`_set_covariance` -- and that is where the three subclasses
        diverge.

        Parameters
        ----------
        x_observe : (x_dim,) ndarray, optional
            The observed state; ``None`` means reuse the previous step's prediction
            (multi-step prediction mode).
        u_observe : (u_dim,) ndarray, optional
            The control input; ``None`` means reuse the previous step's input.
        """
        if x_observe is None:  # support multi-step prediction
            x_observe = self.x_pred
        if u_observe is None:
            u_observe = self.u
        x_observe = as_vector(x_observe, len(self.mu_bar), 'x_observe')
        u_observe = as_vector(u_observe, None, 'u_observe')
        if self._has_bias and self.bias is None:
            self.bias = np.zeros_like(x_observe)
        self.x = x_observe
        self.u = u_observe

        # predicted state = predictive drift + nominal noise mean
        self.nu_pred = self._drift(x_observe, u_observe)
        self.mu_pred = self.mu_bar
        self.x_pred = self.nu_pred + self.mu_pred
        self._set_covariance()

    def reset(self):
        """Restore a reusable state (undo the in-place rewrites of the nominal quantities).

        ⚠️ Clearing the prediction outputs alone is not enough: ``trajectory_prediction``
        rewrites ``f_bar`` / ``mu_bar`` / ``Sigma_bar``, and ``lamb`` / ``v`` are derived
        from ``Sigma_bar`` -- without restoring them, reusing this instance for a second
        trajectory would carry the leftover parameters of the first. See the contract notes
        in :mod:`ppsds.probabilistic_predictor.base`.
        """
        self.f_bar, self.mu_bar, self.Sigma_bar = self._nominal
        self.lamb, self.v = eig_real(self.Sigma_bar)
        self.x = self.u = self.bias = None
        self.nu_pred = self.mu_pred = None
        self.x_pred = self.Sigma_pred = self.lamb_pred = None

    # -- subclass interface --------------------------------------------

    def _drift(self, x, u):
        """This step's predictive drift: the nominal drift (plus ``self.bias`` when ``_has_bias``)."""
        d = self.f_bar(x, u)
        return d + self.bias if self._has_bias else d

    def _set_covariance(self):
        """Fill ``Sigma_pred`` and ``lamb_pred`` (Eig-DRPP also fills ``j`` / ``val_wst``)."""
        raise NotImplementedError

    # -- whole trajectory (**deprecated**, please migrate to the environment layer) ----

    def trajectory_prediction(self, scoring_rule, x_traj, u_traj, sds_params=None):
        """Run the full predict-update cycle over one trajectory.

        .. deprecated::
           Deprecated; please migrate to the Gymnasium layer in
           :mod:`ppsds.environment`. Note that the **linear A/B time-varying path here has
           no corresponding env-layer implementation**, so the deprecation target does not
           fully exist.

        Parameters
        ----------
        scoring_rule : ScoringRule
            A scoring rule with a ``score(predictive_dist, true_state)`` method.
        x_traj : (T+1, x_dim) ndarray
            The true state trajectory.
        u_traj : (T+1, u_dim) ndarray
            The control input trajectory.
        sds_params : dict, optional
            The step-varying uncertain parameters, in one of two formats:

            - **linear format** (from ``LinearAmbiguitySet``):
              ``{'A': (T+1,dim_x,dim_x), 'B': (T+1,dim_x,dim_u),
              'mu': (T+1,dim_x), 'Sigma': (T+1,dim_x,dim_x)}``.
              Each step **replaces** the nominal dynamics with the true linear map
              ``A@x + B@u``.
            - **nonlinear format** (from ``NonlinearAmbiguitySet``):
              ``{'delta_f': (T+1,dim_x), 'mu': (T+1,dim_x),
              'Sigma': (T+1,dim_x,dim_x)}``.
              Adds the additive bias ``delta_f`` **on top of** the nominal dynamics.

            In both formats ``mu_bar`` / ``Sigma_bar`` are updated step by step.

        Returns
        -------
        dict
            ``{'x_pred': (T+1,x_dim), 'Sigma_pred': (T+1,x_dim,x_dim),
            'score': (T+1,)}``.
        """
        warnings.warn(
            "trajectory_prediction() is deprecated; please migrate to the Gymnasium "
            "layer in ppsds.environment.",
            FutureWarning, stacklevel=2,
        )
        T = x_traj.shape[0] - 1
        dim_x = x_traj.shape[1]
        x_pred_traj = np.zeros((T+1, dim_x))
        Sigma_pred_traj = np.zeros((T+1, dim_x, dim_x))
        score_traj = np.zeros(T+1)
        try:
            for i in range(T+1):
                x_pred_traj[i] = self.x_pred
                Sigma_pred_traj[i] = self.Sigma_pred
                if i:
                    score_traj[i] = scoring_rule.score(self.predict(), x_traj[i, :])
                if sds_params is not None:
                    # -- update the uncertain parameters step by step --
                    # dispatch on the format: linear (A/B) or nonlinear (delta_f)
                    if 'A' in sds_params and 'B' in sds_params:
                        # linear: replace the dynamics function outright with this step's
                        # true (A, B). The bias is zeroed -- the full dynamics are already
                        # carried by the new f_bar.
                        Ai = sds_params['A'][i]
                        Bi = sds_params['B'][i]
                        self.f_bar = lambda x, u, Ai=Ai, Bi=Bi: Ai @ x + Bi @ u
                        self.bias = np.zeros(dim_x)
                    else:
                        # nonlinear: keep the nominal f_bar and only add this step's bias.
                        self.bias = sds_params['delta_f'][i]
                    self.mu_bar = sds_params['mu'][i]
                    self.Sigma_bar = sds_params['Sigma'][i]
                self.update(x_traj[i, :], u_traj[i, :])
        finally:
            # above, f_bar / mu_bar / Sigma_bar were **rewritten in place** -- they must be
            # restored, or reusing the same instance for a second trajectory would carry the
            # leftover parameters of the first.
            self.reset()
        return {'x_pred': x_pred_traj, 'Sigma_pred': Sigma_pred_traj, 'score': score_traj}


class NominalPredictor(_ConicMomentPredictor):
    """The nominal predictor: no robustification at all.

    The predictive mean uses the nominal dynamics ``f_bar``, and the predictive covariance
    is simply the nominal covariance ``Sigma_bar``.

    Parameters
    ----------
    f_bar : callable
        The nominal dynamics ``f(x, u) -> x_next``.
    mu_bar : (x_dim,) ndarray
        Nominal noise mean.
    Sigma_bar : (x_dim, x_dim) ndarray
        Nominal noise covariance.
    multivariate_rv : scipy distribution class, optional
        The predictive distribution family (default ``multivariate_normal``).
    """

    def _set_covariance(self):
        # nominal: the predictive covariance is the nominal one, with Sigma_bar's eigenvalues
        self.Sigma_pred = self.Sigma_bar
        self.lamb_pred = self.lamb


class NoiseDrpp(_ConicMomentPredictor):
    """Noise-DRPP: robust to **noise** only -- inflates the nominal covariance by ``gamma[2]``.

    The prediction law is word-for-word the same as :class:`NominalPredictor`; the only
    difference is "which covariance it acts on".

    .. note::
       This class internally **computes ``gamma[2] * Sigma_bar`` first and then hands it to
       the skeleton**, the same expression as the formulation that preceded it ("handing the
       nominal predictor an inflated covariance"), so the numbers are bit-for-bit unchanged.
       That order is not arbitrary: ``eig_real(γ₂·Σ̄)`` and ``γ₂ · eig_real(Σ̄)`` may differ
       in their last bits, and ``lamb_pred`` is fed to the adversarial dynamics, where a
       difference in the last bits gets amplified.

    Parameters
    ----------
    f_bar : callable
        The nominal dynamics ``f(x, u) -> x_next``.
    mu_bar : (x_dim,) ndarray
        Nominal noise mean.
    Sigma_bar : (x_dim, x_dim) ndarray
        The **nominal** covariance -- the inflation happens inside this class, so callers
        pass the nominal value.
    gamma : list of length 3
        The ambiguity-set radius; only ``gamma[2]`` (the covariance scaling factor) is used.
    multivariate_rv : scipy distribution class, optional
        The predictive distribution family (default ``multivariate_normal``).
    """

    def __init__(self, f_bar, mu_bar, Sigma_bar, gamma,
                 multivariate_rv=multivariate_normal):
        self.gamma = gamma
        super().__init__(f_bar, mu_bar, gamma[2] * Sigma_bar,
                         multivariate_rv=multivariate_rv)

    def _set_covariance(self):
        # the inflation was already folded into self.Sigma_bar in __init__; same as nominal here
        self.Sigma_pred = self.Sigma_bar
        self.lamb_pred = self.lamb


class EigDrpp(_ConicMomentPredictor):
    """Eig-DRPP: the distributionally robust predictor with eigenvectors fixed.

    It solves :func:`eigvec_fixed_minmax` once per step; only the **eigenvalues** of the
    predictive covariance are optimized, while the **eigenvectors are fixed to** those of
    ``Sigma_bar`` -- that is what "restricted" refers to. The predictive mean is the same as
    the nominal predictor's.

    This is the **suboptimal** solution of DRPP; the paper bounds its gap to the global
    optimum.

    Parameters
    ----------
    f_bar : callable
        The nominal dynamics ``f(x, u) -> x_next``.
    mu_bar : (x_dim,) ndarray
        Nominal noise mean.
    Sigma_bar : (x_dim, x_dim) ndarray
        Nominal noise covariance (its eigenvectors are used directly as the fixed basis).
    gamma : list of length 3
        The ambiguity-set radius, in the same format as :func:`eigvec_fixed_minmax`.
    multivariate_rv : scipy distribution class, optional
        The predictive distribution family (default ``multivariate_normal``).

    Attributes
    ----------
    j : int or None
        The index of the worst eigen-direction selected at this step (diagnostic).
    val_wst : float or None
        The worst-case expected-log-score lower bound at this step (diagnostic).
    """

    #: Eig-DRPP does not take the time-varying-parameter path of trajectory_prediction, so it has no additive bias term.
    _has_bias = False

    def __init__(self, f_bar, mu_bar, Sigma_bar, gamma,
                 multivariate_rv=multivariate_normal):
        self.gamma = gamma
        self.j = None       # worst eigen-direction index
        self.val_wst = None  # worst-case expected-log-score lower bound
        super().__init__(f_bar, mu_bar, Sigma_bar, multivariate_rv=multivariate_rv)

    def _set_covariance(self):
        lamb_opt, val_opt, j_opt = eigvec_fixed_minmax(
            self.lamb, self.gamma, self.x, self.u)
        self.j = j_opt
        self.val_wst = val_opt
        self.lamb_pred = lamb_opt
        # rebuild the covariance in the fixed eigenbasis
        self.Sigma_pred = self.v @ np.diag(lamb_opt) @ self.v.T

    def reset(self):
        """Restore a reusable state (restore the nominal quantities and redo the eigendecomposition).

        ``gamma`` is never rewritten, so it needs no restoring. See the contract notes in
        :mod:`ppsds.probabilistic_predictor.base`.
        """
        super().reset()
        self.j = self.val_wst = None

    def trajectory_prediction(self, scoring_rule, x_traj, u_traj, abg_params=None):
        """Run the full predict-update cycle over one trajectory.

        .. deprecated::
           Deprecated; please migrate to the Gymnasium layer in :mod:`ppsds.environment`.

        Parameters
        ----------
        scoring_rule : ScoringRule
        x_traj : (T+1, dim_x) ndarray
        u_traj : (T+1, dim_u) ndarray
        abg_params : optional
            Reserved for a future adaptive DRPP (currently unused).

        Returns
        -------
        dict
            ``{'x_pred', 'j_pred', 'Sigma_pred', 'score'}``.
        """
        warnings.warn(
            "trajectory_prediction() is deprecated; please migrate to the Gymnasium "
            "layer in ppsds.environment.",
            FutureWarning, stacklevel=2,
        )
        T = x_traj.shape[0] - 1
        dim_x = x_traj.shape[1]
        x_pred_traj = np.zeros((T+1, dim_x))
        Sigma_pred_traj = np.zeros((T+1, dim_x, dim_x))
        j_pred_traj = np.zeros(T+1, dtype=int)
        score_traj = np.zeros(T+1)
        try:
            for i in range(T+1):
                x_pred_traj[i] = self.x_pred
                Sigma_pred_traj[i] = self.Sigma_pred
                j_pred_traj[i] = self.j if self.j is not None else -1
                if i:
                    score_traj[i] = scoring_rule.score(self.predict(), x_traj[i, :])
                if abg_params is not None:
                    # reserved for adaptive DRPP (future work)
                    pass
                self.update(x_traj[i, :], u_traj[i, :])
        finally:
            # This override does not rewrite ``f_bar`` / ``mu_bar`` / ``Sigma_bar`` the way the
            # base-class path does, but it *does* leave ``x_pred`` / ``Sigma_pred`` / ``j`` /
            # ``val_wst`` populated. Without this, a second trajectory on the same instance would
            # report the previous episode's last prediction in row 0, where a fresh instance
            # reports the NaN placeholder -- the same cross-episode leak the base class guards
            # against with its own ``finally``.
            self.reset()
        return {'x_pred': x_pred_traj, 'j_pred': j_pred_traj,
                'Sigma_pred': Sigma_pred_traj, 'score': score_traj}


def eigvec_fixed_minmax(lamb, gamma, x, u):
    """Solve the "eigenvectors fixed" DRPP cone program (equation 19 of the paper).

    This is the **central optimization** of the DRPP framework. Given the eigenvalues
    ``lamb`` of the nominal covariance and the uncertainty parameters ``gamma``, it finds
    the worst-case optimal eigenvalues of the predictive covariance.

    The functional maximin over probability measures can be converted equivalently into a
    Euclidean cone program, solved with CVXPY + SCS.

    Parameters
    ----------
    lamb : (dim_x,) ndarray
        The eigenvalues of the nominal covariance ``Sigma_bar``.
    gamma : list of length 3
        The uncertainty parameters:

        - ``gamma[0]``: a callable ``f(x, u)`` returning the **squared** upper bound of the
          drift perturbation;
        - ``gamma[1]``: float, the bound on the noise-mean uncertainty;
        - ``gamma[2]``: float, the covariance scaling factor.
    x : (dim_x,) ndarray
        The current state.
    u : (dim_u,) ndarray
        The current control input.

    Returns
    -------
    optimal_lamb : (dim_x,) ndarray
        The worst-case optimal eigenvalues of the predictive covariance.
    optimal_value : float
        The optimal objective value (the lower bound of the expected log score).
    optimal_j : int
        The index of the selected "worst" eigen-direction.

    Notes
    -----
    ⚠️ The SCS parameters (``max_iters=5000, eps=1e-9``) were tuned for cone-program
    numerical stability, and **changing them affects the discrete choice of ``j``**: a
    perturbation of 1e-14 between nearly tied candidates is enough to flip ``j`` (after the
    flip the score differs by only ~1e-9, but the result is then no longer bit-for-bit
    identical). Do not change them in passing.

    The optimization sweeps over ``j`` (each possible worst direction) and takes the minimum.
    """
    # Imported here rather than at module scope on purpose: `import ppsds` re-exports EigDrpp, so
    # a module-level import would make every user pay for cvxpy -- and its solvers (osqp,
    # clarabel, scs, ecos) -- even when they only ever touch NominalPredictor, NoiseDrpp or the
    # environment layer. cvxpy is still a hard dependency; it is simply not imported until a cone
    # program is actually solved. `tests/test_import_hygiene.py` pins this.
    import cvxpy as cp

    d_x = len(lamb)
    z_aux = gamma[0](x, u)  # squared upper bound of the perturbation at (x, u)

    # precompute the coefficient vector c_j (given inline in the paper's equations)
    coefficient = np.ones(d_x)
    for i in range(d_x):
        coefficient[i] = 2 * np.sqrt(z_aux * gamma[1] * lamb[i]) + z_aux

    optimal_value = np.inf
    optimal_lamb = lamb
    optimal_j = -1

    # search over the worst-direction index j
    for j in range(d_x):
        # decision variable: x = 1 / lambda_pred (the reciprocal of the eigenvalues)
        x_var = cp.Variable(d_x)

        # the modified eigenvalues for this j
        lamb_modified = gamma[2] * lamb
        lamb_modified[j] += coefficient[j]

        # objective: sum(-log(x_i) + lambda_modified_i * x_i)
        obj = cp.Minimize(cp.sum(-cp.log(x_var) + np.diag(lamb_modified) @ x_var))

        # linear constraints after the cone rearrangement
        Q_1 = np.zeros((d_x, d_x))
        Q_1[:, j] = coefficient[j] * np.ones(d_x)
        Q_2 = np.diag(coefficient)
        Q = Q_1 - Q_2
        b = np.zeros(d_x)
        cst = [Q @ x_var >= b, x_var >= b]

        prob = cp.Problem(obj, cst)

        # solve the cone program (SCS suits this class of problem better)
        prob.solve(solver=cp.SCS, max_iters=5000, eps=1e-9)

        if prob.value < optimal_value:
            optimal_value = prob.value
            optimal_lamb = 1 / np.array(x_var.value)
            optimal_j = j

    # convert back from the cone dual to the expected log score
    optimal_value = (-d_x * np.log(2 * np.pi) - optimal_value) / 2

    return optimal_lamb, optimal_value, optimal_j
