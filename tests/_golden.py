"""The golden numerical scenario: one fixed system, one fixed seed, a fixed number of steps.

Both :mod:`test_golden` (which compares against ``golden.json``) and ``regenerate_fixtures.py``
(which writes it) go through :func:`compute` and :func:`serialize`, so the two can never drift
apart.

Why the assertions are shaped the way they are
----------------------------------------------
:func:`~ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment.eigvec_fixed_minmax`
solves one cone program per candidate worst direction ``j`` and keeps the smallest objective.
When two candidates are nearly tied, a perturbation of ~1e-14 is enough to flip which ``j`` wins
-- and after the flip the resulting score differs by only ~1e-9. A bit-for-bit comparison would
therefore go red on another platform for no real reason, while a loose comparison would stop
catching real regressions.

So the suite asserts two different things, and it is :mod:`test_golden` that spells them out:
the **sorted** eigenvalues must match the fixture within a modest relative tolerance, and the
**invariants** (symmetry, positive definiteness, finiteness, and agreement with
``eigvec_fixed_minmax`` re-run on the fixture's own inputs) must hold exactly.
"""

from __future__ import annotations

import numpy as np

from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (
    EigDrpp,
    NominalPredictor,
    NoiseDrpp,
)
from ppsds.scoring_rule.logarithm_score import LogarithmScore

from _hexjson import (
    decode_arrays,
    decode_floats,
    encode_arrays,
    encode_floats,
)
from _systems import LINEAR_X0, control_at, linear_ambiguity_set, true_step

#: Seed and horizon of the scenario. The seed is the release date, so it is memorable enough
#: that nobody changes it by accident.
SEED = 20261020
STEPS = 8

#: Predictor order is fixed so that the fixture's structure is stable across regenerations.
PREDICTOR_NAMES = ('nominal', 'noise_drpp', 'eig_drpp')


def build_predictors(abg):
    """The three predictors, all taking their model off the same ambiguity set.

    Note that they come off ``abg`` rather than from freshly typed literals: that way "the model
    the predictor uses" and "the model the ambiguity set declares" agree by construction.
    """
    return {
        'nominal': NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar),
        'noise_drpp': NoiseDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma),
        'eig_drpp': EigDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma),
    }


def compute():
    """Run the scenario and return everything worth pinning down.

    Fully deterministic: the ambiguity sets' samplers are never called, and the observations
    come from a generator seeded here rather than from global random state.

    Returns
    -------
    dict
        ``{'seed', 'steps', 'x_obs', 'u_obs', 'predictors', 'eig_drpp_j',
        'eig_drpp_val_wst'}``, where ``predictors[name]`` holds the per-step ``lamb_pred`` /
        ``Sigma_pred`` / ``x_pred`` / ``score``. ``x_obs`` / ``u_obs`` are the inputs of each
        step, recorded (rather than recomputed) so that the fixture is self-contained.
    """
    abg = linear_ambiguity_set()
    predictors = build_predictors(abg)
    scorer = LogarithmScore()
    rng = np.random.default_rng(SEED)

    out = {
        name: {'lamb_pred': [], 'Sigma_pred': [], 'x_pred': [], 'score': []}
        for name in PREDICTOR_NAMES
    }
    eig_j, eig_val = [], []
    x_obs, u_obs = [], []

    x = np.array(LINEAR_X0, dtype=np.float64)
    for i in range(STEPS):
        u = control_at(i)
        for pred in predictors.values():
            pred.update(x, u)
        x_next = true_step(x, u, rng)
        for name, pred in predictors.items():
            out[name]['lamb_pred'].append(np.array(pred.lamb_pred, dtype=np.float64))
            out[name]['Sigma_pred'].append(np.array(pred.Sigma_pred, dtype=np.float64))
            out[name]['x_pred'].append(np.array(pred.x_pred, dtype=np.float64))
            out[name]['score'].append(float(scorer.score(pred.predict(), x_next)))
        eig_j.append(int(predictors['eig_drpp'].j))
        eig_val.append(float(predictors['eig_drpp'].val_wst))
        # The inputs of the step, recorded so the fixture is self-contained: the oracle check in
        # test_golden needs to re-run the cone program on exactly these.
        x_obs.append(np.array(x, dtype=np.float64))
        u_obs.append(np.asarray(u, dtype=np.float64))
        x = x_next

    return {
        'seed': SEED,
        'steps': STEPS,
        'x_obs': x_obs,
        'u_obs': u_obs,
        'predictors': out,
        'eig_drpp_j': eig_j,
        'eig_drpp_val_wst': eig_val,
    }


def serialize(result):
    """Turn :func:`compute`'s output into JSON-ready data with hex-encoded floats."""
    predictors = {
        name: {
            'lamb_pred': encode_arrays(d['lamb_pred']),
            'Sigma_pred': encode_arrays(d['Sigma_pred']),
            'x_pred': encode_arrays(d['x_pred']),
            'score': encode_floats(d['score']),
        }
        for name, d in result['predictors'].items()
    }
    return {
        '_comment': 'Generated by tests/regenerate_fixtures.py -- do not edit by hand.',
        'seed': result['seed'],
        'steps': result['steps'],
        'x_obs': encode_arrays(result['x_obs']),
        'u_obs': encode_arrays(result['u_obs']),
        'predictors': predictors,
        'eig_drpp_j': result['eig_drpp_j'],
        'eig_drpp_val_wst': encode_floats(result['eig_drpp_val_wst']),
    }


def deserialize(obj):
    """Inverse of :func:`serialize`."""
    predictors = {
        name: {
            'lamb_pred': decode_arrays(d['lamb_pred']),
            'Sigma_pred': decode_arrays(d['Sigma_pred']),
            'x_pred': decode_arrays(d['x_pred']),
            'score': decode_floats(d['score']),
        }
        for name, d in obj['predictors'].items()
    }
    return {
        'seed': obj['seed'],
        'steps': obj['steps'],
        'x_obs': decode_arrays(obj['x_obs']),
        'u_obs': decode_arrays(obj['u_obs']),
        'predictors': predictors,
        'eig_drpp_j': list(obj['eig_drpp_j']),
        'eig_drpp_val_wst': decode_floats(obj['eig_drpp_val_wst']),
    }
