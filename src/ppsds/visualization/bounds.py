"""Index/length **clamping** at the visualization entry points.

Background
----------
The experiment runner hardcoded trajectory indices (``k=2`` / ``k=7`` / ``k=43``), and the
library-level comparison functions carried their own defaults (``k=53`` / ``k=23``). As soon
as ``-t`` was smaller than that, the **plotting stage** raised ``IndexError`` -- even though
the data had long been computed and the pickle already written, so a single figure made the
whole experiment exit non-zero. (Measured: ``python run.py -t 2 ...`` crashed straight
inside ``covariance.py``.)

This module **clamps** indices and lengths into their legal range.

Why clamping **warns** instead of silently adjusting
----------------------------------------------------
Clamping **changes which trajectory is drawn** (or how many steps). That is a semantic
change and must be visible, otherwise you will believe the figure shows trajectory 43 when
it actually shows the last one. A ``RuntimeWarning`` is used because it is displayed once by
default -- it does not flood the output, but it is not swallowed either.

Clamping is **idempotent**: called with an in-range value it emits no warning and changes
nothing.
"""

from __future__ import annotations

import warnings


def clamp_index(k, n, *, what='trajectory index', stacklevel=3):
    """Clamp the index ``k`` into ``[0, n-1]``.

    Parameters
    ----------
    k : int
    n : int
        Number of available entries along that axis (``n <= 0`` raises -- that means the
        data itself is broken, which should not be clamped).
    what : str
        Name appearing in the warning, to make it easy to locate the out-of-range site.
    """
    if n <= 0:
        raise ValueError(f'{what}: no entries available, cannot index (the data itself is broken)')
    k = int(k)
    if 0 <= k < n:
        return k
    new = min(max(k, 0), n - 1)
    warnings.warn(
        f'{what} {k} out of range (available 0..{n - 1}); using {new} instead. '
        f'Note that the figure content changes accordingly; to show a specific trajectory, '
        f'raise -t or use a smaller index.',
        RuntimeWarning, stacklevel=stacklevel)
    return new


def clamp_time_indexes(time_indexes, n_steps, *, what='time index', stacklevel=3):
    """Clamp each time index; emit **only one** warning if any of them changed."""
    if n_steps <= 0:
        raise ValueError(f'{what}: nothing available (the data itself is broken)')
    original = list(time_indexes)
    out = [min(max(int(t), 0), n_steps - 1) for t in original]
    if out != original:
        warnings.warn(
            f'{what} {original} has out-of-range entries (available 0..{n_steps - 1}); '
            f'using {out} instead.',
            RuntimeWarning, stacklevel=stacklevel)
    return out


def clamp_len(length, n_steps, *, what='trajectory length', stacklevel=3):
    """Clamp the slice length into ``[1, n_steps]``."""
    if n_steps <= 0:
        raise ValueError(f'{what}: nothing available (the data itself is broken)')
    length = int(length)
    if 1 <= length <= n_steps:
        return length
    new = min(max(length, 1), n_steps)
    warnings.warn(
        f'{what} {length} out of range (available 1..{n_steps}); using {new} instead.',
        RuntimeWarning, stacklevel=stacklevel)
    return new


def clamp_horizon(horizon, n_available, *, what='time horizon', stacklevel=3):
    """Clamp "how many steps to draw" into ``[1, n_available]`` (same semantics as :func:`clamp_len`).

    The separate name makes the call site self-explanatory: a **scoring horizon** constant
    like ``T=31`` is not the same thing as "trajectory length" (the former is how far the
    score is plotted, the latter is the trajectory truncation).
    """
    return clamp_len(horizon, n_available, what=what, stacklevel=stacklevel + 1)
