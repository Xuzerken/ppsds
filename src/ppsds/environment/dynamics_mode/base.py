"""The two contracts every dynamics mode is built on.

:class:`SamplingAmbiguitySet` is a **structural** contract (a :class:`typing.Protocol`)
describing what a mode needs from an ambiguity set. :class:`DynamicsMode` is the strategy
interface the environment drives.
"""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np


class SamplingAmbiguitySet(Protocol):
    """The minimal contract :class:`DynamicsMode` needs: it can generate SDS parameters.

    :class:`~ppsds.ambiguity.additive_drift.conic_moment.AdditiveDriftConicMomentSet`
    satisfies it naturally, but the two have no inheritance relation -- a mode uses this
    **structural** contract only.
    """

    x_dim: int
    u_dim: int

    def sample_trajectory(self, T: int, period: int = 1) -> dict:
        """Sample one piecewise-constant parameter trajectory; each value's first axis is ``T+1``."""
        ...

    def worst_case_params(self, x, u, lamb_pred) -> dict:
        """Worst-case parameters given the predicted covariance eigenvalues (includes key ``'j'``)."""
        ...


class DynamicsMode(ABC):
    """Strategy interface: pick one set of SDS parameters per step.

    A concrete mode implements ``reset()`` (called when an episode starts) and
    ``get_params(step_idx)`` (called before every ``env.step()``).
    """

    @abstractmethod
    def reset(self, rng: np.random.Generator, T: int) -> None:
        """(Re)initialize, preparing for a new episode.

        Parameters
        ----------
        rng : numpy.random.Generator
            Seeded generator obtained from ``env.np_random``.
        T : int
            Number of steps in the episode.
        """
        ...

    @abstractmethod
    def get_params(self, step_idx: int) -> dict:
        """Return the SDS parameters to use at this step.

        Parameters
        ----------
        step_idx : int
            Current step index (0-based).

        Returns
        -------
        dict
            The key set is declared by the ambiguity set's ``param_keys``; a missing key
            means "take the nominal value for that item" (so ``{}`` means "all nominal").
        """
        ...

    def set_info(self, lamb_pred, x, u) -> None:
        """Optional hook: inject predictor information before ``step()``.

        By default it does **nothing** -- only :class:`EigvecFixedAdversarialDynamics`
        consumes it. It is defined on the base class so that the caller
        (``Env.inject_eigvec_fixed_info``) does not have to sniff with
        ``hasattr(mode, 'set_info')``.
        """
        return None
