"""Ambiguity sets based on conic moments.

This module covers only **one** branch of the model assumptions::

    x_{k+1} = f(x_k, u_k) + delta_f + w_k

that is

- **additive drift**: the drift error ``delta_f`` enters the state **added** to the
  nominal drift ``f``;
- **additive noise**: the noise ``w`` likewise enters the state **added** to it, and does
  not change with the state (homogeneous);
- **conic moment uncertainty**: the norm of ``delta_f``, the noise mean ``mu`` and the
  noise covariance ``Sigma`` each lie in a set parameterized by ``gamma``.

.. note::
   **The ambiguity set itself does not assume Gaussianity.** The conic moment
   constraints use only the **first two moments** of the noise. Gaussianity appears only
   in two downstream places: (a) the generative kernel used for sampling
   (:meth:`~AdditiveDriftConicMomentSet.transition`), and (b) the logarithmic scoring
   rule. Do not read "moment constraints" as "distributional assumptions".

.. note::
   **This module does not cover**: state-dependent noise, multiplicative noise, ``w``
   correlated with ``x``, or drift errors that enter the state nonlinearly. Those require
   another base class -- and another predictor (the existing Eig-DRPP predictor requires
   the three things ``f`` / ``mu`` / ``Sigma``).

Class hierarchy
---------------
:class:`AdditiveDriftConicMomentSet` is the base class, carrying the parts **common** to
both families (the sampling-trajectory skeleton, the theoretical bounds, the selection of
the worst-case direction, ``transition``/``validate``); the two subclasses differ only in
**parameterization** -- that is, in what ``sds_params`` looks like:

=================================  =============================  ==============================
Subclass                           Dynamics                       ``sds_params`` keys
=================================  =============================  ==============================
:class:`LinearAmbiguitySet`        ``x' = A x + B u + w``         ``A`` / ``B`` / ``mu`` / ``Sigma``
:class:`NonlinearAmbiguitySet`     ``x' = f(x,u) + delta_f + w``  ``delta_f`` / ``mu`` / ``Sigma``
=================================  =============================  ==============================

The difference between the two is only one of **representation**: the linear family
writes the drift error as a matrix perturbation
(``(A_bar+dA)x + (B_bar+dB)u``, which expands to ``f_bar(x,u) + delta_f``) and folds it
into the noise mean in the worst case; the nonlinear family gives the ``delta_f`` vector
directly. Given the same ``gamma`` and the same prediction covariance, the two update to
the same state.

> **On the class names**: this module's path ``ambiguity/additive_drift/conic_moment``
> already says it all -- "assumption (additive drift) + family (conic moment)" -- so the
> subclass names need not repeat it: they say only their own business (linear /
> nonlinear). The base class name ``AdditiveDriftConicMomentSet`` keeps the full name,
> because it is the reference point when **adding a family** (see below).

Adding another family
---------------------
**The landing spot is a sibling of this module**:
``ambiguity/additive_drift/<family>.py``. The ``additive_drift`` level is the
**assumption**; ``conic_moment`` is one **family** under it, and a different family is a
new module beside this one rather than a branch inside it.

**Do not** subclass :class:`AdditiveDriftConicMomentSet` -- the representation of the
radius and the worst-case rule both differ. Only part of the base class is
**family-agnostic**:

===========================================  ==========================================
Family-agnostic                              conic moment specific
===========================================  ==========================================
``x_dim`` / ``u_dim`` / ``f_bar`` /            ``gamma`` (three-element radius)
``mu_bar`` / ``Sigma_bar``
``param_keys`` / ``validate``                 ``worst_case_params`` (worst direction + covariance shrink)
``transition`` (additive noise kernel)        ``value_upper_bound_true_transition`` /
``sample_trajectory`` (by ``_traj_fields``)   ``value_lower_bound_eigvec_fixed``
``mean()`` / ``sample()`` abstract interface
===========================================  ==========================================

When a second family actually lands, then extract the left column into a shared base
class; right now there is only one family, so extracting would be guessing.
"""

import numpy as np
from scipy.stats import multivariate_normal

from ...numerics.linalg import eig_real
from ...numerics.random_variables import multivariate_uniform
from ...probabilistic_predictor.drpp.additive_drift.conic_moment import (
    eigvec_fixed_minmax,
)


class AdditiveDriftConicMomentSet:
    """Base class for conic moment ambiguity sets: assumes [additive drift + additive noise].

    "Additive" is not just an adjective in the name -- it is **enforced by the base
    class**: :meth:`transition` implements ``x' = mean(x, u; params) + w`` -- the drift is
    supplied by the subclass's :meth:`mean`, and the noise enters the state **added** to
    it. Subclasses cannot get around this.

    Attributes
    ----------
    x_dim, u_dim : int
        State / control dimensions. ``dim_x`` / ``dim_u`` are compatible aliases for them.
    mu_bar : ndarray of shape (x_dim,)
        Nominal noise mean.
    Sigma_bar : ndarray of shape (x_dim, x_dim)
        Nominal noise covariance.
    gamma1 : float
        The **squared** radius of the noise mean: ``|mu - mu_bar| <= sqrt(gamma1)``.
    gamma2 : float
        Covariance scaling factor: ``Sigma = gamma2 * Sigma_bar - ...``.
    f_err_max : float
        The **un-squared** hard upper bound on the additive drift norm ``||delta_f||``.
    """

    #: The ``sds_params`` keys this ambiguity set accepts; a missing key means "take
    #: the nominal value for that term".
    param_keys: frozenset = frozenset()

    #: Fields of a sampled trajectory: a sequence of (key name, per-step shape). Determines
    #: the output shape and key order of :meth:`sample_trajectory` -- subclasses declare it
    #: in ``__init__``.
    _traj_fields: tuple = ()

    def __init__(self, x_dim, u_dim, mu_bar, Sigma_bar, *,
                 f_err_callable, gamma1, gamma2, f_err_max):
        self.x_dim = int(x_dim)
        self.u_dim = int(u_dim)
        self.mu_bar = np.asarray(mu_bar, dtype=np.float64)
        self.Sigma_bar = np.asarray(Sigma_bar, dtype=np.float64)
        #: ``gamma[0]``: a callable ``f(x, u)`` that returns an upper bound on the
        #: **squared norm** of the drift.
        self._f_err_callable = f_err_callable
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.f_err_max = f_err_max

    # -- Compatibility aliases -----------------------------------------

    @property
    def dim_x(self):
        """Alias for ``x_dim``."""
        return self.x_dim

    @property
    def dim_u(self):
        """Alias for ``u_dim``."""
        return self.u_dim

    @property
    def gamma(self):
        """The canonical format consumed by the predictor: ``[f_err, gamma1, gamma2]``.

        The meaning of the three elements (**all of them concern "squared", the place
        most easily filled in wrong**):

        - ``gamma[0]`` -- a callable ``f(x, u)`` returning an upper bound on the
          **squared norm** of the drift ``||delta_f||^2``. Note this differs from
          ``f_err_max``: the latter is the **un-squared** hard upper bound.
        - ``gamma[1]`` -- the **squared** radius of the noise mean.
        - ``gamma[2]`` -- the covariance scaling factor.

        This list is consumed directly by the DRPP predictor (``drpp_eigvec_fixed_minmax``)
        and by the adversarial dynamics, and it is the **only** shape that must be
        aligned with them.
        """
        return [self._f_err_callable, self.gamma1, self.gamma2]

    # -- Subclass interface --------------------------------------------

    def mean(self, x, u, params):
        """Nominal drift + per-parameter perturbation; does **not** include the noise
        mean ``mu``."""
        raise NotImplementedError

    def sample(self):
        """Sample one SDS realization from the ambiguity set, returning a dict shaped
        like ``sds_params``."""
        raise NotImplementedError

    def _assemble_worst(self, x, u, z_aux, lamb, v, v_j, j, Sigma):
        """Assemble the worst-case direction into this family's ``sds_params`` (families
        differ in "where the drift term lands")."""
        raise NotImplementedError

    # -- Environment-side interface ------------------------------------

    def validate(self, params):
        """Validate that every key of ``sds_params`` is accepted by this ambiguity set.

        A missing key is **allowed** -- it means that term takes its nominal value (on
        the first step of adversarial mode there is no prediction yet, so ``{}`` is
        returned). Only an **unknown key** raises: that is precisely where the old
        "guess the format via ``'A' in params``" used to swallow the error.
        """
        unknown = set(params) - self.param_keys
        if unknown:
            raise ValueError(
                f"{type(self).__name__} does not accept these sds_params keys: {sorted(unknown)}. "
                f"Accepted keys are {sorted(self.param_keys)} (a missing key means the nominal value).")

    def transition(self, x, u, params, rng):
        """Take one step: ``x' ~ N(mean(x,u;params) + mu, Sigma)``.

        **This line is the executable form of the "additive" assumption** -- the drift
        and the noise enter the state added together, rather than through a
        state-dependent transform. Choosing a Gaussian **kernel** for the noise is a
        sampling-side choice (and also the kernel of the logarithmic scoring rule), not a
        distributional assumption of the ambiguity set.

        ``rng`` is supplied by the caller (``gym.Env.np_random``) so that results are
        reproducible; when ``mu`` / ``Sigma`` are missing the nominal values are used.
        """
        mu = params.get('mu', self.mu_bar)
        Sigma = params.get('Sigma', self.Sigma_bar)
        return np.asarray(
            rng.multivariate_normal(self.mean(x, u, params) + mu, Sigma),
            dtype=np.float64)

    # -- Generation capabilities ---------------------------------------

    def sample_trajectory(self, T, period=1):
        """Sample one **piecewise-constant** trajectory of SDS parameters.

        Parameters
        ----------
        T : int
            Number of time steps.
        period : int
            How many steps each SDS realization persists before resampling: ``1`` means
            step-by-step i.i.d., ``100`` means slowly varying.

        Returns
        -------
        dict
            Keys are declared by :attr:`_traj_fields`; the leading dimension of each
            value is ``T+1``.
        """
        if period is None:          # compatible with the old call sample_trajectory(T, None)
            period = 1
        if period < 1:
            raise ValueError(f'period should be >= 1, got {period}')

        arrays = {k: np.zeros((T + 1,) + shape) for k, shape in self._traj_fields}
        for i in range(T + 1):
            if i % period == 0:
                d = self.sample()
            else:
                d = {k: arrays[k][i - 1] for k, _ in self._traj_fields}
            for k, _ in self._traj_fields:
                arrays[k][i] = d[k]
        return arrays

    # -- Theoretical bounds --------------------------------------------

    def value_upper_bound_true_transition(self):
        """Upper bound: the tight value of the expected log score when the **true
        transition is known**.

        It is the closed form of the Gaussian log score ``-(d·log2π + d + log det(Σ))/2``
        evaluated when the prediction covariance equals the noise covariance (the largest
        one in the set, ``gamma[2]*Sigma_bar``). It is a **tight** upper bound: no
        predictor can score higher than "true transition known". It is a pure property of
        the set and involves no relaxation.
        """
        return -(self.x_dim * np.log(2*np.pi) + self.x_dim
                 + np.log(np.linalg.det(self.gamma[2]*self.Sigma_bar))) / 2

    def value_lower_bound_eigvec_fixed(self):
        """Lower bound: the expected log score that Eig-DRPP **guarantees** at ``x=0, u=0``.

        Opposite to the upper bound: the upper bound assumes the true transition is
        known, whereas this lower bound assumes the worst case has already occurred and
        that the predictor may only use the **eigenvectors fixed** relaxation (fixed to
        the eigenvectors of ``Sigma_bar``) -- that is, the design space of Eig-DRPP. It
        takes the optimal value of the minmax under that relaxation, i.e. "how much one
        can at least obtain in the worst case".

        .. note::
           This method calls
           :func:`~ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment.eigvec_fixed_minmax`
           from ``probabilistic_predictor.drpp.additive_drift.conic_moment``, so
           ``ambiguity`` depends on ``probabilistic_predictor``. This is not a typo: the
           lower bound **is** defined by the Eig-DRPP minmax, and without the predictor it
           has no meaning.
        """
        lamb, v = eig_real(self.Sigma_bar)
        gamma = self.gamma.copy()
        gamma[0] = lambda x, u: self.f_err_max ** 2
        lamb_opt, val_opt, j_opt = eigvec_fixed_minmax(lamb, gamma, 0, 0)
        return val_opt

    # -- Adversarial (worst-case) generation ----------------------------

    def worst_case_params(self, x, u, lamb_pred):
        """For a given predictor, construct the SDS parameters that maximize its score loss.

        First the worst eigendirection *j* is selected, then the covariance is shrunk, and
        finally the subclass's :meth:`_assemble_worst` assembles this family's
        ``sds_params``.

        Parameters
        ----------
        x : ndarray of shape (x_dim,)
            Current state.
        u : ndarray of shape (u_dim,)
            Current control input.
        lamb_pred : ndarray of shape (x_dim,)
            Eigenvalues of the predictor's current prediction covariance.

        Returns
        -------
        dict
            This family's ``sds_params``, plus the key ``'j'`` (index of the worst
            direction).
        """
        z_aux = self.gamma[0](x, u)
        lamb, v = eig_real(self.Sigma_bar)

        # select the worst eigendirection
        j = int(np.argmax([
            (2 * np.sqrt(self.gamma[1] * z_aux * lamb[i]) + z_aux) / max(lp, 1e-12)
            for i, lp in enumerate(lamb_pred)
        ]))
        v_j = v[:, j:(j + 1)]

        # the shrunk covariance (same for both families)
        Sigma = self.gamma[2] * self.Sigma_bar - self.gamma[1] * lamb[j] * v_j @ v_j.T

        return self._assemble_worst(x, u, z_aux, lamb, v, v_j, j, Sigma)


class LinearAmbiguitySet(AdditiveDriftConicMomentSet):
    """Linear time-invariant SDS: ``x_{k+1} = A x_k + B u_k + w_k``.

    The uncertainty is on ``(A, B, mu, Sigma)``, parameterized by ``alphas`` and
    ``f_err_max``. The drift error still enters **additively** -- here it is written as a
    matrix perturbation ``delta_f = dA x + dB u``, which expands to
    ``f_bar(x, u) + delta_f``.

    Parameters
    ----------
    A_bar : ndarray of shape (x_dim, x_dim)
        Nominal dynamics matrix.
    B_bar : ndarray of shape (x_dim, u_dim)
        Nominal input matrix.
    mu_bar : ndarray of shape (x_dim,)
        Nominal noise mean.
    Sigma_bar : ndarray of shape (x_dim, x_dim)
        Nominal noise covariance.
    alphas : dict
        The uncertainty radius of each component, with keys ``'A'`` / ``'B'`` / ``'mu'`` /
        ``'Sigma'``:

        - ``alphas['A']``: the maximum perturbation of each **superdiagonal** element of
          ``A`` (**un-squared** magnitude; the sampler draws uniformly on
          ``[-alphas['A'], alphas['A']]``).
        - ``alphas['B']``: the maximum perturbation of ``B[0,-1]``, same convention.
        - ``alphas['mu']``: an upper bound on ``||mu_a||^2`` (the **squared** norm of the
          noise-mean deviation), i.e. ``sum_i (mu_i - mu_bar_i)^2 <= alphas['mu']``. Hence
          the per-component box half-width is ``sqrt(alphas['mu'] / x_dim)`` -- that is
          what :meth:`sample` uses, so that ``max||mu_a||^2`` exactly exhausts the declared
          bound.
        - ``alphas['Sigma']``: the scaling factor of the nominal covariance.
    f_err_max : float
        The hard upper bound on the perturbation norm ``||delta_f||``, **un-squared**.
        :meth:`f_err` compares it against the un-squared ``norm_max`` and returns the
        **square** of the smaller, so its return value is a squared quantity whose upper
        bound is ``f_err_max**2``.
    """

    param_keys = frozenset({'A', 'B', 'mu', 'Sigma'})

    def __init__(self, A_bar, B_bar, mu_bar, Sigma_bar, alphas, f_err_max):
        A_bar = np.asarray(A_bar, dtype=np.float64)
        B_bar = np.asarray(B_bar, dtype=np.float64)
        self.A_bar = A_bar
        self.B_bar = B_bar
        self.alphas = alphas
        super().__init__(A_bar.shape[0], B_bar.shape[1], mu_bar, Sigma_bar,
                         f_err_callable=self.f_err,
                         gamma1=alphas['mu'], gamma2=alphas['Sigma'],
                         f_err_max=f_err_max)
        self._traj_fields = (
            ('A', (self.x_dim, self.x_dim)),
            ('B', (self.x_dim, self.u_dim)),
            ('mu', (self.x_dim,)),
            ('Sigma', (self.x_dim, self.x_dim)),
        )

    # -- Nominal model --------------------------------------------------

    def f_bar(self, x, u):
        """Nominal dynamics: ``A_bar @ x + B_bar @ u``."""
        return self.A_bar @ x + self.B_bar @ u

    def f_err(self, x, u):
        """An upper bound on the squared additive drift norm, clipped by ``f_err_max``.

        Taking ``a_A||x|| + a_B||u||`` (a **sum**) rather than the combined norm
        ``a * sqrt(||x||^2 + ||u||^2)`` is deliberate: the drift produced by the sampler
        is ``delta_f = dA x + dB u``, and by the triangle inequality its **tight** upper
        bound is exactly this sum (``||dA x|| <= a_A||x||``, ``||dB u|| <= a_B||u||``, see
        :meth:`sample`). The combined-norm expression is always ``<=`` this one, so using
        it would place some samples **outside** the declared ambiguity set -- i.e. the set
        could not contain its own generator. "It contains it" is the criterion.
        """
        norm_max = self.alphas['A'] * np.linalg.norm(x) + self.alphas['B'] * np.linalg.norm(u)
        return (np.min([self.f_err_max, norm_max])) ** 2

    def mean(self, x, u, params):
        """When ``A`` / ``B`` are given, **replace** the nominal matrices wholesale --
        this is the existing semantics of the linear family's parameterization."""
        A = params.get('A', self.A_bar)
        B = params.get('B', self.B_bar)
        return A @ x + B @ u

    # -- Sampling -------------------------------------------------------

    def sample(self):
        """Sample one SDS realization from the ambiguity set.

        - ``A``: add a ``U[-alphas['A'], alphas['A']]`` perturbation to the superdiagonal;
        - ``B``: add a ``U[-alphas['B'], alphas['B']]`` perturbation to ``B[0,-1]``;
        - ``mu``: per-component box half-width ``sqrt(alphas['mu'] / x_dim)``, so that
          ``max||mu_a||^2`` exactly exhausts ``alphas['mu']`` (the paper takes
          ``mu = 0.5 * alpha_3`` with ``alpha_3 ~ U[-1,1]^d``, whose half-width is exactly
          this value);
        - ``Sigma``: ``alphas['Sigma'] * Sigma_bar - mu_a mu_a^T``.
        """
        A = self.A_bar + np.diag(self.alphas['A'] * (2*np.random.random(self.x_dim-1)-1), 1)
        # ⚠️ .copy() is required: writing `B = self.B_bar` directly would be an **alias**,
        # and the next line would rewrite the ambiguity set's own B_bar in place -- each
        # sample would permanently drift the nominal B once (sample_trajectory calls this
        # T+1 times, which amounts to a random walk).
        B = self.B_bar.copy()
        B[0,-1] += self.alphas['B'] * (2*np.random.random()-1)
        # per-component half-width = sqrt(gamma_1 / d), so that max ||mu_a||^2 = gamma_1 (exactly exhausts the bound).
        half = np.sqrt(self.alphas['mu'] / self.x_dim)
        mu = multivariate_uniform(low = self.mu_bar - half*np.ones(self.x_dim),
                                  high = self.mu_bar + half*np.ones(self.x_dim)).rvs()[0]
        mu_a = (mu-self.mu_bar).reshape([-1,1])
        Sigma = self.alphas['Sigma']*self.Sigma_bar - mu_a@mu_a.T
        return {'A':A, 'B':B, 'mu':mu, 'Sigma':Sigma}

    # -- Worst case -----------------------------------------------------

    def _assemble_worst(self, x, u, z_aux, lamb, v, v_j, j, Sigma):
        # Representation of the linear family: the drift error is folded into the **noise
        # mean**; A / B take their nominal values.
        mu = (np.sqrt(self.gamma1 * lamb[j]) + np.sqrt(z_aux)) * v[:, j] + self.mu_bar
        return {'A': self.A_bar, 'B': self.B_bar, 'mu': mu, 'Sigma': Sigma, 'j': j}


class NonlinearAmbiguitySet(AdditiveDriftConicMomentSet):
    """Nonlinear SDS: ``x_{k+1} = f(x_k, u_k) + delta_f + w_k``.

    This is the most direct form of "additive drift" -- ``delta_f`` is added straight to
    the nominal drift. The ``NLink`` in the name comes from this repository's original
    use case (an n-link manipulator), but the class itself requires only that ``f_bar`` be
    a callable and is not tied to any particular system.

    Parameters
    ----------
    f_bar : dict
        Must contain ``'function'`` (a callable ``f(x, u)`` returning the nominal next
        state), ``'x_dim'``, and ``'u_dim'``.
    mu_bar : ndarray of shape (x_dim,)
        Nominal noise mean.
    Sigma_bar : ndarray of shape (x_dim, x_dim)
        Nominal noise covariance.
    gamma : list of length 3
        - ``gamma[0]``: a dict containing ``'function'`` (a callable returning an upper
          bound on the **squared** norm of the drift at ``(x,u)``), ``'max'`` (the hard
          upper bound on the un-squared norm ``||delta_f||``), ``'x_dim'``, and
          ``'u_dim'``;
        - ``gamma[1]``: the **squared** radius of the noise mean
          (``|mu - mu_bar| <= sqrt(gamma[1])``);
        - ``gamma[2]``: the covariance scaling factor.
    legacy_sampler : bool, default False
        Reproduce the old generator -- ``delta_f ~ N(0, (f_err_max/3)^2 I)``. When
        ``x_dim`` is large that sampler **violates** its own declared bound (``/3`` is a
        per-component 3-sigma rule of thumb that ignores the dimension:
        ``E||delta_f||^2 = x_dim * (f_err_max/3)^2``, which already exceeds
        ``f_err_max^2`` once ``x_dim > 9``). Kept only to reproduce old results.
    """

    param_keys = frozenset({'delta_f', 'mu', 'Sigma'})

    def __init__(self, f_bar, mu_bar, Sigma_bar, gamma, legacy_sampler=False):
        self.f_bar = f_bar['function']
        self.legacy_sampler = legacy_sampler
        super().__init__(f_bar['x_dim'], f_bar['u_dim'], mu_bar, Sigma_bar,
                         f_err_callable=gamma[0]['function'],
                         gamma1=gamma[1], gamma2=gamma[2],
                         f_err_max=gamma[0]['max'])
        self._traj_fields = (
            ('delta_f', (self.x_dim,)),
            ('mu', (self.x_dim,)),
            ('Sigma', (self.x_dim, self.x_dim)),
        )

    # -- Nominal model --------------------------------------------------

    def mean(self, x, u, params):
        """The nominal drift ``f_bar(x, u)`` plus the additive ``delta_f`` (zero by
        default)."""
        return self.f_bar(x, u) + params.get('delta_f', np.zeros(self.x_dim))

    # -- Sampling -------------------------------------------------------

    def _sample_delta_f(self):
        """Draw the additive drift from the ambiguity set.

        The set is bounded by ``||delta_f|| <= f_err_max`` (the hard upper bound from
        ``gamma[0]``). Sampling is **uniform inside that ball**, so every sample is a
        genuine member of the set -- which matters for two things: the worst-case
        guarantee covers only SDSs inside the set, and the "the set is the ground truth"
        usage must not contradict itself.

        The radius must be ``f_err_max * U**(1/d)`` for the distribution to be uniform in
        the ball (``P(r <= t) = (t/f_err_max)**d``).
        """
        if self.legacy_sampler:
            return multivariate_normal(
                mean=np.zeros(self.x_dim),
                cov=(self.f_err_max/3)**2 * np.identity(self.x_dim)).rvs()

        direction = np.random.normal(size=self.x_dim)
        direction /= np.linalg.norm(direction)
        radius = self.f_err_max * np.random.random() ** (1.0 / self.x_dim)
        return direction * radius

    def sample(self):
        """Sample one SDS realization from the ambiguity set.

        ``mu`` uses rejection sampling: the construction
        ``Sigma = gamma[2]*Sigma_bar - mu_a mu_a^T`` may be non-positive-definite when
        ``mu`` approaches the box boundary, and the ambiguity set contains only valid
        covariances, so invalid ``(mu, Sigma)`` pairs are discarded and redrawn.
        """
        delta_f = self._sample_delta_f()
        while True:
            mu = multivariate_uniform(low = self.mu_bar-np.sqrt(self.gamma[1])*np.ones(self.x_dim), high = self.mu_bar + np.sqrt(self.gamma[1])*np.ones(self.x_dim)).rvs()[0]
            mu_a = (mu-self.mu_bar).reshape([-1,1])
            Sigma = self.gamma[2]*self.Sigma_bar - mu_a@mu_a.T
            Sigma = (Sigma + Sigma.T) / 2
            if np.linalg.eigvalsh(Sigma).min() > 1e-12:
                break
        return {'delta_f':delta_f, 'mu':mu, 'Sigma':Sigma}

    # -- Worst case -----------------------------------------------------

    def _assemble_worst(self, x, u, z_aux, lamb, v, v_j, j, Sigma):
        # Representation of the nonlinear family: the drift error is returned
        # **separately** as delta_f; mu does not include it.
        delta_f = np.sqrt(z_aux) * v[:, j]
        mu = np.sqrt(self.gamma1 * lamb[j]) * v[:, j] + self.mu_bar
        return {'delta_f': delta_f, 'mu': mu, 'Sigma': Sigma, 'j': j}
