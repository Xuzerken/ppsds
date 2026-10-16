"""Input validation shared by the predictor families.

Lives one level above ``drpp/`` and ``pmf/`` because both need it and neither owns it.
"""

import numpy as np


def as_vector(v, expected_len, name):
    """Coerce ``v`` into a 1-D array and validate its length.

    Previously ``update()`` **silently broadcast** inputs of the wrong shape: for example,
    passing ``u`` as ``(1, 1)`` turned ``B @ u`` into ``(dim_x, 1)``, and adding the
    ``(dim_x,)`` bias **broadcast it to ``(dim_x, dim_x)``** -- silent all the way, until
    ``predict()`` threw an error like "Array 'mean' must be a vector of length N²" whose
    root cause was nowhere to be seen.

    ``expected_len=None`` means only 1-D is required (the predictor does not know the
    dimension of ``u``).
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or (expected_len is not None and len(v) != expected_len):
        want = f'({expected_len},)' if expected_len is not None else '1-D'
        raise ValueError(
            f'{name} should be {want}, got shape={v.shape}. '
            f'(A mismatched shape used to be silently broadcast and only blew up '
            f'later in predict(), with an incomprehensible error.)')
    return v
