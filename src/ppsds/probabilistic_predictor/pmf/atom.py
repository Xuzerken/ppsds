"""Predictors whose predictive law is a pmf **you** supply, over a set of atoms.

The second category of probabilistic predictor beside ``drpp/``. Where a DRPP predictor derives
a Gaussian from a nominal model and an ambiguity set, these take an
:class:`~ppsds.numerics.atom_pmf.AtomPmf` as given and predict with it.

Why this is useful even without an optimizer
--------------------------------------------
Because it is the piece that lets a prediction you chose *by hand* be evaluated as rigorously as
one a model produced: same environment, same scoring rule, same statistics. That makes the
interactive bet-placing picture and an experiment the same object, rather than two demos that
happen to look alike.

It is deliberately **not** a search over pmfs. Nothing here picks the atoms or the weights; it
evaluates the ones it is handed. Two ways to hand them over:

- :class:`StaticAtomPmfPredictor` -- the atoms are in the **state** frame and never move. This is
  the shape of "here is what I think the next state is", one step at a time.
- :class:`DriftShiftedAtomPmfPredictor` -- the atoms are in the **noise** frame, and each step
  shifts them by the nominal drift. This is the shape of a predictor that keeps up with a
  trajectory, and it is the same reduction the predictability paper uses for an i.i.d. process.
"""

import numpy as np

from ...numerics.atom_pmf import AtomPmf
from ...numerics.linalg import eig_real
from .._inputs import as_vector


class _AtomPmfPredictor:
    """The skeleton shared by the pmf predictors (**private**, not exported).

    Subclasses supply only :meth:`_law_for` -- which pmf this step predicts -- which is the only
    substantive difference between them. The ``predict`` / ``update`` / ``reset`` cycle, the
    multi-step fallback and the moment summaries all live here.

    Attributes
    ----------
    x_pred, Sigma_pred : ndarray
        The mean and covariance of the predictive pmf. These are **summaries of the law**, not
        the law: :meth:`predict` returns the pmf itself. They are filled in because the
        predictor contract asks for them and the environment layer reads them.
    lamb_pred : ndarray
        Eigenvalues of ``Sigma_pred``. An atom law is often degenerate, so these can include
        zeros.
    """

    def __init__(self, atom_pmf):
        if not isinstance(atom_pmf, AtomPmf):
            raise TypeError(f'atom_pmf must be an AtomPmf, got {type(atom_pmf).__name__}')
        self.atom_pmf = atom_pmf
        self.x_dim = atom_pmf.dimensions

        self.x = None
        self.u = None
        self._prediction = None

        self.x_pred = None
        self.Sigma_pred = None
        self.lamb_pred = None

    def predict(self):
        """The pmf for the current step, as an :class:`~ppsds.numerics.atom_pmf.AtomPmf`.

        Raises
        ------
        ValueError
            If called before :meth:`update` -- there is nothing to predict yet, and returning
            ``None`` would surface later as an obscure failure inside a scoring rule.
        """
        if self._prediction is None:
            raise ValueError(
                'predict() was called before update(); the predictor has no state yet. Call '
                'update(x_observe, u_observe) once before predicting.'
            )
        return self._prediction

    def update(self, x_observe=None, u_observe=None):
        """Advance to the next step.

        Parameters
        ----------
        x_observe : (x_dim,) ndarray, optional
            The observed state; ``None`` reuses the previous step's prediction (multi-step
            prediction mode).
        u_observe : (u_dim,) ndarray, optional
            The control input; ``None`` reuses the previous step's input.
        """
        if x_observe is None:  # support multi-step prediction
            x_observe = self.x_pred
        if x_observe is None:
            raise ValueError(
                'update() needs an observed state the first time; there is no previous '
                'prediction to carry forward.'
            )
        if u_observe is None:
            u_observe = self.u
        x_observe = as_vector(x_observe, self.x_dim, 'x_observe')
        if u_observe is not None:
            u_observe = as_vector(u_observe, None, 'u_observe')

        self.x = x_observe
        self.u = u_observe
        self._prediction = self._law_for(x_observe, u_observe)

        self.x_pred = self._prediction.mean
        self.Sigma_pred = self._prediction.cov
        self.lamb_pred = eig_real(self.Sigma_pred)[0]

    def reset(self):
        """Restore the predictor to its reusable initial state.

        The supplied pmf is **not** touched -- it is the caller's object, and it is the same
        law on every trajectory. Only the per-trajectory state is cleared.
        """
        self.x = None
        self.u = None
        self._prediction = None
        self.x_pred = None
        self.Sigma_pred = None
        self.lamb_pred = None

    def _law_for(self, x_observe, u_observe):
        """The pmf to predict at this step -- the subclass's one difference."""
        raise NotImplementedError


class StaticAtomPmfPredictor(_AtomPmfPredictor):
    """Predicts the same pmf every step: the atoms you place are where you say they are.

    This is the shape of a one-step prediction made by hand -- "the next state is near here,
    with these weights" -- repeated. It is also the shape a bet-placing game has, which is what
    makes the two comparable.
    """

    def _law_for(self, x_observe, u_observe):
        return self.atom_pmf


class DriftShiftedAtomPmfPredictor(_AtomPmfPredictor):
    """Shifts a noise-frame pmf by the nominal drift every step.

    Predicts ``p_w(x_k - f(x_{k-1}, u_{k-1}))``: the atoms describe the *noise*, and moving them
    by the drift is what turns them into a prediction about the state. A predictor of this shape
    is what the predictability paper's convergence result assumes.

    Parameters
    ----------
    atom_pmf : AtomPmf
        The pmf over the **noise**, in the drift's frame.
    f_bar : callable
        The nominal drift ``f_bar(x, u) -> (x_dim,)``.
    """

    def __init__(self, atom_pmf, f_bar):
        if f_bar is None:
            raise TypeError('f_bar is required: without a drift there is nothing to shift by')
        super().__init__(atom_pmf)
        self.f_bar = f_bar

    def _law_for(self, x_observe, u_observe):
        drift = np.asarray(self.f_bar(x_observe, u_observe), dtype=float)
        if drift.shape != (self.x_dim,):
            raise ValueError(
                f'f_bar returned shape {drift.shape}, expected ({self.x_dim},). A mis-shaped '
                'drift would broadcast against the atoms and quietly misplace every one of them.'
            )
        return AtomPmf(self.atom_pmf.atoms + drift, self.atom_pmf.weights)
