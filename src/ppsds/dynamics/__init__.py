"""The system itself: known dynamics and mechanical models, plus identifying a nominal model from
rollouts.

The split from :mod:`ppsds.ambiguity` is the split between *the thing being predicted* and *what
you believe about it*. A system here generates the data; an ambiguity set states the uncertainty a
predictor must survive. They are kept in separate subpackages because they answer different
questions and change for different reasons.

- :mod:`~ppsds.dynamics.known_sds` — :class:`~ppsds.dynamics.known_sds.AdditiveDriftSds`: a drift
  plus a **named noise law**, known exactly. Because the law is given rather than described by
  moments, the noise need not be Gaussian.
- :mod:`~ppsds.dynamics.mechanical` — :class:`~ppsds.dynamics.mechanical.TwoLink` and
  :class:`~ppsds.dynamics.mechanical.NLink`, with :func:`~ppsds.dynamics.mechanical.rk4_step` for
  discretization. ``NLink`` is a nonlinear, strongly coupled nominal drift, not a validated
  robotic-arm model. Linearizing around an equilibrium is done through its own
  ``NLink.equilibrium_for_pose`` / ``NLink.linearize_at_discrete`` **methods**, not module-level
  functions.
- :mod:`~ppsds.dynamics.identification` — fitting a linear nominal model from rollouts, and
  diagnosing the residual. This is what you reach for when your environment has no analytic
  dynamics, and it is the entry point the "apply DRPP to your own environment" story uses.
"""

from .identification import (
    LinearFit,
    collect_rollouts,
    diagnose,
    encode_action,
    fit_linear,
    format_report,
    regularize,
)
from .known_sds import AdditiveDriftSds
from .mechanical import NLink, TwoLink, rk4_step

__all__ = [
    'AdditiveDriftSds',
    'TwoLink',
    'NLink',
    'rk4_step',
    'LinearFit',
    'collect_rollouts',
    'fit_linear',
    'diagnose',
    'regularize',
    'encode_action',
    'format_report',
]
