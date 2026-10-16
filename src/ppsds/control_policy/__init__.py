"""A control policy, for the experiments that need one.

Not part of the prediction story: this exists so that the linear experiments can be run under
closed-loop regulation as well as open loop. If you have your own controller, use it — nothing in
:mod:`ppsds.probabilistic_predictor` or :mod:`ppsds.ambiguity` needs this.

- :class:`~ppsds.control_policy.lqr.DLQR` — the infinite-horizon discrete LQR regulator.

⚠️ The batch experiment's ``-c regulation`` works on the **linear** models only; the non-linear
ones exit with an error, which is why the scripts default to ``-c none``.
"""

from .lqr import DLQR

__all__ = ['DLQR']
