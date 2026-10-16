"""Two reference values for the expected epsilon-logarithm score.

Given a true one-step law ``q`` (written here in the state frame; for an additive-drift system
it is the noise law translated by the drift, so it is the *same shape* every step), the
``epsilon``-logarithm score of a predictive law ``p`` on a realised outcome ``x`` is

.. math::
    \\mathcal L_\\epsilon(p, x) = \\log \\int_{\\|s - x\\|_\\infty \\le \\epsilon} p(s)\\, ds ,

and this module provides the two horizontal lines the predictability literature plots
against it:

- :func:`predictability_upper_bound` -- :math:`d_x \\log(2\\epsilon) - h_d(q)`. An upper bound
  on the expected score of **any** predictor, so it is a property of the system alone: no
  predictor enters. This is the "probabilistic predictability" of the system.
- :func:`approximation_value` -- :math:`d_x \\log(2\\epsilon) - \\mathrm H(q \\| p)`. The
  ``O(epsilon)`` approximation of the expected score of the *given* predictor ``p``.

.. note::
   The upper bound is a bound, not a target: it is attained only by some systems, so a gap
   between a predictor's score and this line is not by itself evidence that the predictor is
   poor. Where the bound *is* attained, the gap closes from below.

   Both formulas carry the same indicator: the ``d_x log(2 epsilon)`` volume term is present
   only for ``epsilon > 0``. At ``epsilon = 0`` the score degenerates to the ordinary
   logarithm score, the volume term vanishes, and the two lines become ``-h_d(q)`` and
   ``-H(q || p)`` respectively.

Where the predictability upper bound sits relative to DRPP
----------------------------------------------------------
These two answer "how well could anything do, for a system known **exactly**" -- an upper
bound on prediction quality. DRPP answers "what can be **guaranteed** for a system known only
through an ambiguity set" -- a lower bound. They are complementary, and they are a different
pair of bounds from ``value_upper_bound_true_transition`` / ``value_lower_bound_eigvec_fixed``,
which bracket the *relaxation gap inside one* ambiguity set.
"""

import math

import numpy as np

from .entropy import cross_entropy, differential_entropy


def predictability_upper_bound(law_q, eps, d_x=None):
    """:math:`d_x \\log(2\\epsilon) - h_d(q)` -- the best expected score achievable, in the bound.

    Parameters
    ----------
    law_q : object
        The true continuous one-step law; needs ``entropy()``.
    eps : float
        Neighbourhood radius, ``>= 0``. Zero degenerates to the ordinary logarithm score.
    d_x : int, optional
        State dimension. Inferred from ``law_q`` when omitted, which is correct whenever the
        process noise spans the state -- true for every additive-drift system here. Pass it
        explicitly if the state is higher-dimensional than the noise.

    Returns
    -------
    float
        The bound, in nats.
    """
    eps = _validate_eps(eps)
    d_x = _resolve_dim(law_q, d_x)
    return _volume_term(d_x, eps) - differential_entropy(law_q)


def approximation_value(law_q, law_p, eps, d_x=None, *, bounds=None, n_quad=None):
    """:math:`d_x \\log(2\\epsilon) - \\mathrm H(q \\| p)` -- the ``O(epsilon)`` score approximation.

    The true expected score differs from this by ``O(epsilon)``, so it is the line a
    *continuous* predictor's score converges to. It is **not** available for an atom/pmf
    predictor: :func:`~ppsds.predictability.entropy.cross_entropy` raises, because such a
    predictor has ``log p = -inf`` almost everywhere.

    Parameters
    ----------
    law_q, law_p : object
        True and predictive laws. See
        :func:`~ppsds.predictability.entropy.cross_entropy`.
    eps : float
        Neighbourhood radius, ``>= 0``.
    d_x : int, optional
        State dimension; inferred from ``law_q`` when omitted.
    bounds, n_quad : optional
        Passed through to :func:`~ppsds.predictability.entropy.cross_entropy`.

    Returns
    -------
    float
        The approximation, in nats.
    """
    eps = _validate_eps(eps)
    d_x = _resolve_dim(law_q, d_x)
    return _volume_term(d_x, eps) - cross_entropy(law_q, law_p, bounds=bounds, n_quad=n_quad)


def _volume_term(d_x, eps):
    """``d_x log(2 eps)`` -- the log volume of the epsilon-box, and 0 when ``eps == 0``."""
    if eps <= 0:
        return 0.0
    return d_x * math.log(2 * eps)


def _validate_eps(eps):
    eps = float(eps)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError(f'eps must be a finite radius >= 0, got {eps}')
    return eps


def _resolve_dim(law_q, d_x):
    """``d_x`` if given, else the law's own dimension."""
    if d_x is not None:
        d_x = int(d_x)
        if d_x < 1:
            raise ValueError(f'd_x must be >= 1, got {d_x}')
        return d_x
    dimensions = getattr(law_q, 'dimensions', None)
    if dimensions is not None:
        return int(dimensions)
    mean = getattr(law_q, 'mean', None)
    if mean is not None:
        return int(np.size(mean))
    raise ValueError(
        f'cannot infer d_x from {type(law_q).__name__}; pass d_x explicitly. It is the '
        'dimension of the *state*, which equals the law\'s dimension when the noise spans '
        'the state.'
    )
