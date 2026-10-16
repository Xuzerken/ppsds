"""Differential entropy of the true law, and its cross-entropy against a predictive law.

Two per-step quantities, both expectations under the **true** one-step law ``q``:

- :func:`differential_entropy` -- :math:`h_d(q)`;
- :func:`cross_entropy` -- :math:`\\mathrm H(q \\,\\|\\, p) = -\\mathbb E_{x \\sim q} \\log p(x)`.

Why this module exists
----------------------
The predictability upper bound is ``d_x log(2 eps) - h_d(q)``, and the ``O(eps)``
approximation of a *given* predictor's expected score is ``d_x log(2 eps) - H(q || p)``.
Both need an entropy **of a law**, not of a sample, so both live here rather than being
re-derived at each call site.

A "law" is duck-typed: :func:`cross_entropy` needs ``.pdf``, :func:`differential_entropy`
needs ``.entropy()``. Both
:class:`~ppsds.numerics.random_variables.multivariate_uniform` and the ``scipy.stats``
multivariate distributions qualify.

.. warning::
   ``q`` here is the **true continuous** one-step law. The bound formula above does not apply
   when ``q`` is itself an atom/pmf law: ``differential_entropy`` would then return its Shannon
   entropy, which is a different quantity (non-negative, where a differential entropy may be
   negative). Predict *with* a pmf; take the entropy of the continuous truth.
"""

import math

import numpy as np


def differential_entropy(law_q):
    """Return :math:`h_d(q)`, the differential entropy of the true one-step law.

    Delegates to the law's own ``entropy()`` -- for
    :class:`~ppsds.numerics.random_variables.multivariate_uniform` that is the closed form
    ``log(prod(high - low))``; for a ``scipy.stats`` distribution it is scipy's.

    Parameters
    ----------
    law_q : object
        The true continuous one-step law; must expose ``entropy()``.

    Returns
    -------
    float
        The entropy. **May be negative** -- a differential entropy is a log of a volume, and
        the volume can be below 1.
    """
    entropy = getattr(law_q, 'entropy', None)
    if entropy is None:
        raise TypeError(
            f'differential_entropy needs a law with .entropy(), but {type(law_q).__name__} has '
            'none. ppsds.numerics.random_variables.multivariate_uniform and the scipy.stats '
            'multivariate distributions both provide it.'
        )
    value = float(entropy())
    if not math.isfinite(value):
        raise ValueError(f'the law reports a non-finite entropy ({value})')
    return value


def cross_entropy(law_q, law_p, *, bounds=None, n_quad=None):
    """Return :math:`\\mathrm H(q \\,\\|\\, p) = -\\mathbb E_{x \\sim q} \\log p(x)`.

    The expectation is integrated over ``q``'s support by tensor-product trapezoid rule --
    adequate for ``d_x <= 3``, which is what the predictability work needs. For a box law
    (``.low`` / ``.high``, as :class:`~ppsds.numerics.random_variables.multivariate_uniform`
    has) the support is taken from the law; otherwise pass ``bounds=(low, high)``.

    Parameters
    ----------
    law_q : object
        The true law; must expose ``pdf``, and ``low`` / ``high`` unless ``bounds`` is given.
    law_p : object
        The predictive law; must expose ``pdf``. A pmf/atom predictor does **not** -- see the
        ``TypeError`` below, which is deliberate.
    bounds : tuple of array_like, optional
        ``(low, high)``, each of length ``d_x``. Defaults to the box ``q`` declares.
    n_quad : int, optional
        Grid points per axis. Defaults to 2001 in 1-D, 201 in 2-D, 41 above -- roughly 4e4
        quadrature points either way.

    Returns
    -------
    float
        The cross-entropy, in nats.

    Raises
    ------
    TypeError
        If ``law_p`` has no ``pdf``. This is the pmf case, and it is worth being loud about:
        an atom law assigns zero density almost everywhere, so ``log p(x) = -inf`` and the
        cross-entropy is ``+inf``. That is exactly why the epsilon-logarithm score exists --
        score a pmf with
        :class:`~ppsds.scoring_rule.epsilon_logarithm_score.EpsilonLogarithmScore`, and use
        :func:`~ppsds.predictability.bounds.predictability_upper_bound` as the reference line
        rather than the ``O(eps)`` approximation.
    ValueError
        If ``q`` puts mass where ``p`` is zero (the cross-entropy is then genuinely ``+inf``).
    """
    p_pdf = getattr(law_p, 'pdf', None)
    if p_pdf is None:
        raise TypeError(
            f'cross_entropy needs a density for the predictive law, but '
            f'{type(law_p).__name__} has no .pdf. An atom/pmf predictor is zero almost '
            'everywhere, so its cross-entropy against a continuous truth is +inf -- there is '
            'no O(eps) approximation line to draw for it. Score it with '
            'EpsilonLogarithmScore and compare against predictability_upper_bound instead.'
        )
    q_pdf = getattr(law_q, 'pdf', None)
    if q_pdf is None:
        raise TypeError(f'the true law {type(law_q).__name__} has no .pdf to integrate against')

    low, high = _integration_box(law_q, bounds)
    d_x = low.shape[0]
    axes = [np.linspace(low[i], high[i], n_quad or _default_n_quad(d_x)) for i in range(d_x)]
    mesh = np.meshgrid(*axes, indexing='ij')
    points = np.stack([m.reshape(-1) for m in mesh], axis=-1)
    weights = _tensor_trapezoid_weights(axes)

    density_q = np.asarray(q_pdf(points), dtype=float)
    with np.errstate(divide='ignore'):
        log_density_p = np.log(np.asarray(p_pdf(points), dtype=float))
    undefined = (~np.isfinite(log_density_p)) & (density_q > 0)
    if np.any(undefined):
        raise ValueError(
            f'the predictive law has zero density at {int(undefined.sum())} quadrature point(s) '
            'where the true law has mass, so log p = -inf and the cross-entropy is +inf'
        )
    return float(-np.sum(weights * density_q * log_density_p))


def _integration_box(law_q, bounds):
    """The box to integrate over: explicit ``bounds``, else the one the law declares."""
    if bounds is not None:
        box = np.asarray(bounds, dtype=float)
        if box.ndim != 2 or box.shape[0] != 2:
            raise ValueError(f'bounds must be (low, high) of shape (2, d_x), got {box.shape}')
        return box[0], box[1]
    low, high = getattr(law_q, 'low', None), getattr(law_q, 'high', None)
    if low is None or high is None:
        raise ValueError(
            f'cannot infer an integration box from {type(law_q).__name__}: it declares no '
            'low/high. Pass bounds=(low, high) covering its support.'
        )
    return np.asarray(low, dtype=float), np.asarray(high, dtype=float)


def _default_n_quad(d_x):
    """Grid points per axis, so that the tensor grid stays around 4e4 points."""
    if d_x <= 1:
        return 2001
    if d_x == 2:
        return 201
    return 41


def _tensor_trapezoid_weights(axes):
    """Flattened tensor-product trapezoid weights, in the C order of ``meshgrid(..., 'ij')``."""
    per_axis = []
    for axis in axes:
        h = axis[1] - axis[0]
        w = np.full(axis.shape[0], h)
        w[0] = w[-1] = h / 2
        per_axis.append(w)
    out = per_axis[0]
    for w in per_axis[1:]:
        out = np.multiply.outer(out, w)
    return out.reshape(-1)
