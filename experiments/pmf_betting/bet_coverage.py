"""Exact answers to "can this hand of chips miss?" -- the arithmetic behind the safety badge.

A landing pays nothing (and the round scores ``-inf``) when it falls outside **every** chip's
neighbourhood. So the question "can this hand ever lose?" is a geometric one: does the union of
the chips' boxes cover the noise's support? This module answers it, and it answers it
**exactly** rather than by sampling:

* all the boxes are the same size and axis-aligned, so the plane can be cut at every box edge
  (:func:`compressed_grid`). Inside one cell of that grid every box is either wholly in or
  wholly out, so a cell carries one constant payout;
* the payout over a cell is the total weight of the chips covering it, and the worst case is the
  smallest such total over the support -- zero, exactly when some cell is uncovered.

The page recomputes both of these in JavaScript (``public/docs/game/coverage.js``): the badge has
to update on every drag, with no server. Two implementations of one rule drift apart, so the
exported self-check vectors pin them together, and this module is the authority.

The counterpart for unbounded noise is :func:`gaussian_miss_probability`: a Gaussian can land
anywhere, so no hand can *guarantee* a payout, and the honest number to show is the probability
the hand catches the landing, not a worst case that is always zero.
"""

from __future__ import annotations

import numpy as np

#: How far out the miss probability is evaluated for unbounded noise, in standard deviations.
#: Beyond this window the tail mass is below 1e-8, so clipping it changes nothing a badge shows.
MISS_WINDOW_SIGMA = 6.0


def compressed_grid(atoms, low, high, eps):
    """The grid of cells on which the payout is constant.

    Returns ``(xs, ys)``: the cell boundaries along each axis, spanning ``[low, high]`` and
    including every chip's box edges (clipped to the box, since only the support matters).
    """
    atoms = np.atleast_2d(np.asarray(atoms, dtype=float))
    low = np.asarray(low, dtype=float).reshape(-1)
    high = np.asarray(high, dtype=float).reshape(-1)
    _validate(atoms, low, high, eps)

    def edges(lo, hi, coordinates):
        candidates = np.concatenate([[lo, hi], coordinates - eps, coordinates + eps])
        # Clipping before `unique` matters: an edge outside the support would add a zero-area
        # or out-of-support strip, and those must not enter the areas or the worst case.
        return np.unique(np.clip(candidates, lo, hi))

    return edges(low[0], high[0], atoms[:, 0]), edges(low[1], high[1], atoms[:, 1])


def covered_mass(atoms, weights, xs, ys, eps):
    """``(n_x, n_y)`` total weight covering each cell of :func:`compressed_grid`."""
    atoms = np.atleast_2d(np.asarray(atoms, dtype=float))
    weights = np.asarray(weights, dtype=float).reshape(-1)
    middle_x = 0.5 * (xs[:-1] + xs[1:])
    middle_y = 0.5 * (ys[:-1] + ys[1:])
    # A cell's middle decides for the whole cell: no box edge crosses a cell interior.
    in_x = np.abs(middle_x[:, None] - atoms[None, :, 0]) <= eps
    in_y = np.abs(middle_y[:, None] - atoms[None, :, 1]) <= eps
    return np.einsum('im,jm,m->ij', in_x.astype(float), in_y.astype(float), weights)


def uncovered_report(atoms, weights, low, high, eps):
    """What a hand of chips can and cannot do, over the box ``[low, high]``.

    Returns a dict with

    ``uncovered_area_fraction``
        The share of the box no chip's neighbourhood reaches. Zero means the hand cannot miss
        anything that lands in the box.
    ``worst_case_roi``
        The smallest payout share over the box -- zero whenever any part of it is uncovered.
        A hand that tiles the box with ``N`` equal chips has ``1 / N``.
    ``possible_neg_inf``
        Whether some landing in the box pays nothing. The page turns this into the safety badge.
    """
    atoms = np.atleast_2d(np.asarray(atoms, dtype=float))
    weights = np.asarray(weights, dtype=float).reshape(-1)
    low = np.asarray(low, dtype=float).reshape(-1)
    high = np.asarray(high, dtype=float).reshape(-1)
    _validate(atoms, low, high, eps)
    if atoms.shape[0] != weights.shape[0]:
        raise ValueError(f'got {atoms.shape[0]} atoms but {weights.shape[0]} weights')
    if np.any(weights < 0):
        raise ValueError('weights must be non-negative')
    total = float(weights.sum())
    if total <= 0:
        raise ValueError('weights must sum to something positive')

    xs, ys = compressed_grid(atoms, low, high, eps)
    mass = covered_mass(atoms, weights, xs, ys, eps)
    area = np.outer(np.diff(xs), np.diff(ys))
    box_area = float(np.prod(high - low))
    covered_area = float(area[mass > 0].sum())
    return {
        'uncovered_area_fraction': 1.0 - covered_area / box_area,
        'worst_case_roi': float((mass / total).min()),
        'possible_neg_inf': bool(np.any(mass <= 0)),
    }


def gaussian_miss_probability(atoms, weights, mean, cov, eps):
    """The probability a Gaussian landing pays nothing, for this hand of chips.

    Exact up to the ``MISS_WINDOW_SIGMA`` window: the mass outside the window is added to the
    answer rather than dropped, so the number is a probability over the whole plane.
    """
    atoms = np.atleast_2d(np.asarray(atoms, dtype=float))
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    sigma = np.sqrt(np.diag(cov))
    low = mean - MISS_WINDOW_SIGMA * sigma
    high = mean + MISS_WINDOW_SIGMA * sigma

    xs, ys = compressed_grid(atoms, low, high, eps)
    mass = covered_mass(atoms, weights, xs, ys, eps)
    miss = 1.0 - _box_probability(low, high, mean, cov)
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            if mass[i, j] <= 0:
                miss += _box_probability(
                    np.array([xs[i], ys[j]]), np.array([xs[i + 1], ys[j + 1]]), mean, cov)
    return float(miss)


def _box_probability(low, high, mean, cov):
    """``P(low <= X <= high)`` for ``X ~ N(mean, cov)``."""
    diagonal = np.diag(np.diag(cov))
    if np.allclose(cov, diagonal):
        sigma = np.sqrt(np.diag(cov))
        upper = _normal_cdf((high - mean) / sigma)
        lower = _normal_cdf((low - mean) / sigma)
        return float(np.prod(upper - lower))

    from scipy.stats import multivariate_normal  # noqa: PLC0415  (only the non-diagonal path)

    total = 0.0
    for bits in np.ndindex(*(2,) * len(low)):
        corner = np.where(bits, high, low)
        sign = 1.0 - 2.0 * (sum(bits) % 2)
        total += sign * multivariate_normal.cdf(corner, mean=mean, cov=cov)
    return float(total)


def _normal_cdf(z):
    from scipy.special import ndtr  # noqa: PLC0415

    return ndtr(z)


def _validate(atoms, low, high, eps):
    if atoms.ndim != 2 or atoms.shape[1] != 2:
        raise ValueError(f'atoms must be (n_atoms, 2), got shape {atoms.shape}')
    if low.shape != (2,) or high.shape != (2,):
        raise ValueError(f'the box must be two-dimensional, got {low.shape} and {high.shape}')
    if not np.all(high > low):
        raise ValueError(f'the box is empty: low={low} high={high}')
    if not np.isfinite(eps) or eps < 0:
        raise ValueError(f'the radius must be finite and non-negative, got {eps!r}')
