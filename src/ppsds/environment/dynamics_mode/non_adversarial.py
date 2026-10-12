"""Modes that never treat the predictor adversarially.

They differ only in *who* picks the SDS parameters and *how often* they change:
:class:`FixedDynamics` never perturbs anything, :class:`RandomDynamics` re-samples on a
period, and :class:`PreSampledDynamics` replays a trajectory the caller already sampled.
"""

from typing import Optional

import numpy as np

from .base import DynamicsMode, SamplingAmbiguitySet


# ----------------------------------------------------------------------
# Fixed (degenerate) dynamics
# ----------------------------------------------------------------------

class FixedDynamics(DynamicsMode):
    """Always use the nominal dynamics -- no perturbation at all.

    The degenerate case: when the ambiguity set shrinks to a single point, simulating is
    equivalent to running that deterministic system directly.
    """

    def reset(self, rng: np.random.Generator, T: int) -> None:
        pass  # nothing to precompute

    def get_params(self, step_idx: int) -> dict:
        # empty dict = take the nominal value for every field
        return {}


# ----------------------------------------------------------------------
# Random (piecewise-constant) dynamics
# ----------------------------------------------------------------------

class RandomDynamics(DynamicsMode):
    """Re-sample SDS parameters from the ambiguity set every ``period`` steps.

    ``reset()`` samples the whole parameter trajectory in one go via
    ``abg.sample_trajectory(T, period)``. Therefore:

    - **same seed** -> every episode sees the same ``sds_params`` ("sample once, run many");
    - **different seeds** -> episodes are independent.

    Parameters
    ----------
    abg : SamplingAmbiguitySet
    period : int, default=1
        Number of steps between two re-samples (1 = i.i.d., larger = closer to constant).
    """

    def __init__(self, abg: SamplingAmbiguitySet, period: int = 1):
        self._abg = abg
        self._period = period
        self._params: Optional[dict] = None
        self._T: int = 0

    def reset(self, rng: np.random.Generator, T: int) -> None:
        self._T = T
        # abg.sample_trajectory() uses the global np.random internally, so seed it from
        # rng first and restore afterwards, to avoid polluting the caller's global state.
        old_state = np.random.get_state()
        np.random.seed(rng.integers(0, 2**31 - 1))
        self._params = self._abg.sample_trajectory(T, self._period)
        np.random.set_state(old_state)

    def get_params(self, step_idx: int) -> dict:
        if self._params is None:
            raise RuntimeError("RandomDynamics.reset() must be called before get_params()")
        # Pull this step's slice out: the trajectory arrays have first axis T+1, the rest
        # is the single-step shape.
        out = {}
        for key, val in self._params.items():
            if isinstance(val, np.ndarray) and val.ndim >= 3:
                out[key] = val[step_idx]       # (T+1, d, d) -> (d, d)
            elif isinstance(val, np.ndarray) and val.ndim == 2:
                out[key] = val[step_idx]       # (T+1, d) -> (d,)
            else:
                out[key] = val
        return out


# ----------------------------------------------------------------------
# Pre-sampled dynamics
# ----------------------------------------------------------------------

class PreSampledDynamics(DynamicsMode):
    """Replay a parameter trajectory that was sampled outside.

    The only difference from :class:`RandomDynamics` is *who* samples: here the caller
    samples it and passes it in. This suits the reproducible setting where one parameter
    trajectory is shared by every episode.

    Parameters
    ----------
    sds_params : dict of ndarrays
        The output of ``abg.sample_trajectory(T, period)``. Every value has first axis ``T+1``.
    """

    def __init__(self, sds_params: dict):
        self._params = sds_params

    def reset(self, rng: np.random.Generator, T: int) -> None:
        pass  # parameters are already fixed, do not re-sample

    def get_params(self, step_idx: int) -> dict:
        out = {}
        for key, val in self._params.items():
            if isinstance(val, np.ndarray) and val.ndim >= 3:
                out[key] = val[step_idx]
            elif isinstance(val, np.ndarray) and val.ndim == 2:
                out[key] = val[step_idx]
            else:
                out[key] = val
        return out
