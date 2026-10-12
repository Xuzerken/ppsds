"""Properties that must hold for **any** parameters, not just the ones the fixture happens to use.

These are cheap, and they check the theory rather than the numbers. A failure here is a worse
class of problem than a number moving: it means the implementation contradicts the paper.
"""

from __future__ import annotations

import numpy as np
import pytest

from _systems import linear_ambiguity_set, nonlinear_ambiguity_set
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import EigDrpp

SET_FACTORIES = [linear_ambiguity_set, nonlinear_ambiguity_set]


@pytest.mark.parametrize('factory', SET_FACTORIES)
def test_upper_bound_dominates_the_lower_bound(factory):
    """The two theoretical bounds must be ordered.

    The upper bound assumes the true transition is known, so no predictor can beat it; the lower
    bound is what Eig-DRPP guarantees in the worst case. The guarantee cannot exceed the
    best-possible value -- if it did, the relaxation and the worst case would have been mixed up.
    """
    abg = factory()
    lower = abg.value_lower_bound_eigvec_fixed()
    upper = abg.value_upper_bound_true_transition()
    assert np.isfinite(lower), f'{type(abg).__name__}: lower bound is {lower}'
    assert np.isfinite(upper), f'{type(abg).__name__}: upper bound is {upper}'
    assert lower <= upper, (
        f'{type(abg).__name__}: lower bound {lower} exceeds upper bound {upper}')


@pytest.mark.parametrize('factory', SET_FACTORIES)
def test_upper_bound_is_the_closed_form_at_the_widest_covariance(factory):
    """``-(d*log(2*pi) + d + log det(gamma[2]*Sigma_bar)) / 2``.

    The ``+d`` is the expected-value term: this is the expected log score under the widest
    covariance in the set, not the log density at its mean (that one has no ``+d``). Pinning the
    closed form is what keeps the two apart.
    """
    abg = factory()
    d = abg.x_dim
    expected = -(d * np.log(2 * np.pi) + d
                 + np.log(np.linalg.det(abg.gamma[2] * abg.Sigma_bar))) / 2
    assert abg.value_upper_bound_true_transition() == pytest.approx(expected, rel=1e-12)


def test_lower_bound_uses_the_un_squared_hard_radius():
    """``value_lower_bound_eigvec_fixed`` substitutes ``f_err_max ** 2`` for ``gamma[0]``.

    ``f_err_max`` is the **un-squared** bound while ``gamma[0]`` must return a squared one, and
    that is the single easiest place in this parameterization to be off by a square. Equipping
    the set with a *different* ``f_err_max`` must therefore move the lower bound, and in the
    right direction: a larger drift radius means a worse guaranteed score.
    """
    small = linear_ambiguity_set(f_err_max=0.1)
    large = linear_ambiguity_set(f_err_max=0.5)
    assert large.value_lower_bound_eigvec_fixed() < small.value_lower_bound_eigvec_fixed()

    # f_err_max = 0 removes the drift uncertainty entirely, so the guarantee is the score of the
    # widest covariance in the set -- the upper bound, attained.
    none_at_all = linear_ambiguity_set(f_err_max=0.0)
    assert none_at_all.value_lower_bound_eigvec_fixed() == pytest.approx(
        none_at_all.value_upper_bound_true_transition(), rel=1e-6, abs=1e-6)


@pytest.mark.parametrize('x,u', [
    (np.array([2.0, 1.0]), np.array([0.1])),
    (np.array([10.0, -5.0]), np.array([1.0])),
    (np.array([-0.4, 0.9]), np.array([-2.0])),
    (np.zeros(2), np.zeros(1)),
])
def test_worst_case_value_never_beats_the_upper_bound(x, u):
    """At runtime, not just in the abstract: each step's worst-case value is ``<=`` the bound.

    The upper bound is "what you would score if you knew the true transition", so the worst case
    can only do worse. The ``x = u = 0`` case is the interesting one -- the drift radius vanishes
    there, so Eig-DRPP collapses onto the widest covariance and the bound is attained exactly.
    """
    abg = linear_ambiguity_set()
    pred = EigDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma)
    pred.update(x, u)
    assert pred.val_wst <= abg.value_upper_bound_true_transition() + 1e-9


def test_the_guarantee_is_attained_at_the_zero_state():
    """At ``x = u = 0`` the drift radius is 0, so the "worst case" is the nominal one and the
    lower bound is reached rather than merely approached."""
    abg = linear_ambiguity_set()
    upper = abg.value_upper_bound_true_transition()
    pred = EigDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma)
    pred.update(np.zeros(2), np.zeros(1))
    assert pred.val_wst == pytest.approx(upper, rel=1e-6, abs=1e-6)
