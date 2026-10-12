"""JSON helpers that keep float fixtures exact and reviewable.

Golden fixtures store floats as ``float.hex()`` strings rather than decimal reprs, for two
reasons:

- **Exactness.** ``float.hex()`` round-trips bit for bit through ``float.fromhex()``, whereas a
  decimal repr can round away a change in the last bits -- which is exactly the kind of change
  a numerical regression test exists to catch.
- **Reviewability.** A diff shows the hex digits that actually moved rather than a wall of
  17-digit decimals that all look alike.

Only ``float64`` is supported: these fixtures are all float arrays, and silently accepting
other dtypes would make "the dtype changed" invisible.
"""

from __future__ import annotations

import numpy as np


def encode_array(array):
    """Encode an ndarray as ``{'shape': [...], 'hex': [...]}``."""
    array = np.asarray(array)
    if array.dtype != np.float64:
        raise TypeError(f'only float64 fixtures are supported, got dtype={array.dtype}')
    return {
        'shape': list(array.shape),
        'hex': [float(v).hex() for v in array.reshape(-1)],
    }


def decode_array(obj):
    """Inverse of :func:`encode_array`."""
    values = [float.fromhex(h) for h in obj['hex']]
    return np.array(values, dtype=np.float64).reshape(obj['shape'])


def encode_float(value):
    """Encode a scalar float as its hex string (``None`` passes through)."""
    return None if value is None else float(value).hex()


def decode_float(text):
    """Inverse of :func:`encode_float`."""
    return None if text is None else float.fromhex(text)


def encode_arrays(arrays):
    return [encode_array(a) for a in arrays]


def decode_arrays(objs):
    return [decode_array(o) for o in objs]


def encode_floats(values):
    return [encode_float(v) for v in values]


def decode_floats(texts):
    return [decode_float(t) for t in texts]
