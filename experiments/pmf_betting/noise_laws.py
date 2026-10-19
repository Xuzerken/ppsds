"""Noise laws for the tables the game offers, and the experiments behind them.

Why a block law and not a mixture
---------------------------------
The par line is the predictability upper bound, ``d log 2 eps - h_d(q)``, so it needs the truth's
**differential entropy**. A Gaussian mixture has no closed form for that, and the tools in
``ppsds.predictability`` deliberately refuse to guess (``differential_entropy`` calls the law's own
``entropy()`` and raises if there is none).

A law that is *constant on each of a few boxes* has an exact entropy instead:

    h_d(q) = -sum_i w_i log(w_i / V_i)        (V_i = the volume of box i)

and it buys the one thing the uniform tables cannot teach: **covering the support stops being the
best you can do**. A covering hand pays ``log(1/N)`` wherever the landing falls; a hand that aims
at the fat box pays more. Safety and optimality come apart, and the gap has a clean reading --
with the support fixed, it is exactly ``h_d(uniform on the support) - h_d(q)``.

The interface is dictated by what the rest of the example already asks of a noise law: ``low`` and
``high`` (the support box, for the covering layout), ``dimensions``/``mean``/``cov`` (for the
dimension and for the unbounded branches), ``rvs(size=, rng=)`` (for ``AdditiveDriftSds``),
``entropy()`` (for par) and ``pdf`` (so the entropy can be checked against an independent estimate).
"""

from __future__ import annotations

import numpy as np


class BlockNoise:
    """A bounded law that is constant on each of a few **disjoint** boxes.

    Parameters
    ----------
    boxes : sequence of (low, high)
        Axis-aligned boxes, each a pair of ``(d,)`` points. Their union is the support; the
        support **box** is the bounding box of that union (which is what the covering layout
        tiles, so blocks that do not fill the box cost a little safety margin and score worse --
        honest, and not what the shipped tables do).
    weights : sequence of float
        How much probability each box carries. Non-negative; normalised on construction, so raw
        counts are accepted.

    Raises
    ------
    ValueError
        If the boxes are ragged, a box is empty, a weight is negative or all of them are zero, or
        two boxes overlap. Overlap is refused rather than tolerated: ``entropy`` is a closed form
        only for disjoint boxes, and quietly returning the wrong number for the par line would be
        worse than refusing to build the law at all.
    """

    #: What the exporter records so the page can describe this law without guessing from
    #: attributes (a law with ``low``/``high`` used to be silently labelled "uniform").
    kind = 'blocks'

    def __init__(self, boxes, weights):
        self.boxes = [
            (np.asarray(low, dtype=float).reshape(-1), np.asarray(high, dtype=float).reshape(-1))
            for low, high in boxes
        ]
        if not self.boxes:
            raise ValueError('a block law needs at least one box')
        self.dimensions = self.boxes[0][0].shape[0]
        for low, high in self.boxes:
            if low.shape != (self.dimensions,) or high.shape != (self.dimensions,):
                raise ValueError(
                    f'every box must be {self.dimensions}-dimensional, got {low.shape} and {high.shape}')
            if not np.all(high > low):
                raise ValueError(f'the box {low}..{high} is empty')

        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != len(self.boxes):
            raise ValueError(
                f'got {len(self.boxes)} boxes but {weights.shape[0]} weights; they must pair up')
        if np.any(weights < 0):
            raise ValueError('weights must be non-negative')
        total = float(weights.sum())
        if total <= 0:
            raise ValueError('weights must sum to something positive')
        self.weights = weights / total

        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                if not self._disjoint(self.boxes[i], self.boxes[j]):
                    raise ValueError(
                        f'boxes {i} and {j} overlap; the closed-form entropy below is only valid '
                        'for disjoint boxes, so this is refused rather than guessed at')

        self.low = np.min(np.stack([low for low, _ in self.boxes]), axis=0)
        self.high = np.max(np.stack([high for _, high in self.boxes]), axis=0)
        self.volumes = np.array([float(np.prod(high - low)) for low, high in self.boxes])
        self._means = np.stack([0.5 * (low + high) for low, high in self.boxes])

    @staticmethod
    def _disjoint(first, second):
        """Two axis-aligned boxes are disjoint iff they are separated along at least one axis."""
        (low_a, high_a), (low_b, high_b) = first, second
        return bool(np.any((high_a <= low_b) | (high_b <= low_a)))

    # -- what the rest of the example asks of a noise law ------------------------------------

    @property
    def mean(self):
        """The exact mean, which is also what ``_resolve_dim`` reads the dimension from."""
        return self.weights @ self._means

    @property
    def cov(self):
        """The exact covariance: within-box spread plus the spread of the box centres."""
        centred = self._means - self.mean
        total = np.zeros((self.dimensions, self.dimensions))
        for weight, (low, high), offset in zip(self.weights, self.boxes, centred):
            inside = np.diag((high - low) ** 2) / 12.0  # a uniform box's own covariance
            total += weight * (inside + np.outer(offset, offset))
        return total

    def rvs(self, size=1, rng=None):
        """``(size, d)`` draws: a box by weight, then uniformly inside it."""
        source = np.random.default_rng() if rng is None else rng
        picks = source.choice(len(self.boxes), size=int(size), p=self.weights)
        draws = np.empty((int(size), self.dimensions))
        for index, (low, high) in enumerate(self.boxes):
            chosen = picks == index
            if not np.any(chosen):
                continue
            draws[chosen] = source.uniform(low, high, size=(int(chosen.sum()), self.dimensions))
        return draws

    def pdf(self, x):
        """The density: ``w_i / V_i`` on box ``i``, zero outside. Accepts ``(d,)`` or ``(n, d)``."""
        points = np.atleast_2d(np.asarray(x, dtype=float))
        if points.shape[1] != self.dimensions:
            raise ValueError(f'expected {self.dimensions}-dimensional points, got {points.shape}')
        density = np.zeros(points.shape[0])
        for weight, (low, high), volume in zip(self.weights, self.boxes, self.volumes):
            inside = np.all((points >= low) & (points <= high), axis=1)
            density[inside] += weight / volume
        return density[0] if np.ndim(x) == 1 else density

    def entropy(self):
        """The differential entropy in nats -- exact, because the density is piecewise constant."""
        return float(-np.sum(self.weights * np.log(self.weights / self.volumes)))

    def __repr__(self):
        parts = ', '.join(
            f'{weight:.3g} on {np.round(low, 3).tolist()}..{np.round(high, 3).tolist()}'
            for weight, (low, high) in zip(self.weights, self.boxes))
        return f'BlockNoise({parts})'
