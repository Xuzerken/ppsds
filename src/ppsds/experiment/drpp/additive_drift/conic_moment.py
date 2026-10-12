"""Batch experiment runner for the additive-drift / conic-moment DRPP experiments.

**This module drives an experiment; it is not a predictor.** (The predictor of the same
assumption and family lives in
:mod:`ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment`.) The path states the
scope -- **family** (DRPP) / **assumption** (additive drift) / **family** (conic moment) --
in parallel with :mod:`ppsds.ambiguity` and :mod:`ppsds.probabilistic_predictor`. A new
ambiguity-set family gets a sibling module here.

What it does: generate trajectories, evaluate the four predictors on them, run the
adversarial simulation, spread the work over a process pool, and read/write the validated
cache.

.. warning::
   **The multiprocessing constraint (``fork``) -- this is the easiest thing here to break.**

   The workers **depend on module-level context** (set by :func:`configure`) instead of
   receiving it as a Pool argument. The reason is that **NLink's ambiguity set holds
   lambdas in ``f_bar`` and ``gamma[0]``, which cannot be pickled**. It currently runs
   **only because multiprocessing's default start method is ``fork``** (child processes
   inherit memory directly, so the context never goes through serialization).

   Therefore :func:`configure` **explicitly asserts ``fork``**: if the platform or the
   caller switches to ``spawn``, this **fails loudly**, instead of letting the nonlinear
   experiment **fail silently**.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from tqdm import tqdm

from ...cache import CacheContext
from ppsds.numerics.linalg import eig_real
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (
    NominalPredictor,
    NoiseDrpp,
    EigDrpp,
)
from ppsds.scoring_rule.logarithm_score import LogarithmScore


def organize_traj_data(keys, results):
    """Collect a list of per-trajectory dicts into a dict of stacked arrays.

    Parameters
    ----------
    keys : list of str
        Keys of the resulting dict.
    results : list of dict
        Usually the output of a ``map`` over the worker functions.
    """
    out = {}
    for key in keys:
        out[key] = np.array([item[key] for item in results])
    return out


# ----------------------------------------------------------------------
# Context
# ----------------------------------------------------------------------

@dataclass
class ExperimentContext:
    """All shared state for one experiment run.

    ``abg`` / ``mode`` may contain lambdas and are therefore **not picklable** -- see the
    module docstring.
    """

    abg: object
    mode: object                 # used by the random branch; each worst-branch worker builds its own
    x0: np.ndarray
    T: int
    traj_num: int
    n_seed: int
    policy: Callable             # x -> u
    env_factory: Callable        # mode -> gym.Env
    file_traj: str
    cache: CacheContext
    sds_params_shared: Optional[dict] = None


_CTX: Optional[ExperimentContext] = None


def configure(ctx: ExperimentContext) -> None:
    """Set the module-level context. **Must be called before creating a Pool.**

    It also asserts ``fork`` -- see the module docstring.
    """
    global _CTX
    # allow_none=False: when unset this returns the platform **default** ('fork' on Linux,
    # 'spawn' on macOS/Windows). With allow_none=True we would get None, which means "not
    # set yet" rather than "not fork" -- reporting an error on that would be a false alarm.
    method = mp.get_start_method(allow_none=False)
    if method != 'fork':
        raise RuntimeError(
            f"this module needs fork to start subprocesses (got start method = {method!r}). "
            "Reason: NLink's ambiguity set holds f_bar / gamma[0] as lambdas, which are "
            "**not picklable**, so the context can only be inherited from the parent via "
            "fork. Switching to spawn would make the nonlinear experiment **fail "
            "silently**. Either use fork, or make the context picklable (a later phase).")
    _CTX = ctx


def ctx() -> ExperimentContext:
    if _CTX is None:
        raise RuntimeError('call configure(...) before using the workers')
    return _CTX


# ----------------------------------------------------------------------
# Workers (**must be module-level functions**: the Pool pickles them by reference)
# ----------------------------------------------------------------------

def gen_one_traj(k: int) -> dict:
    """Generate one trajectory through the env layer (randomly sampled SDS)."""
    c = ctx()
    env = c.env_factory(c.mode)
    obs, _ = env.reset(seed=c.n_seed + k)      # a different noise realization per trajectory
    xs = [obs.copy()]
    us = []
    for _step in range(c.T):
        u = c.policy(obs)
        obs, _reward, _term, _trunc, _info = env.step(u)
        xs.append(obs.copy())
        us.append(u.copy())
    us.append(np.zeros_like(us[-1]) if us else np.zeros(env.action_space.shape))
    return {'x': np.array(xs), 'u': np.array(us)}


def _load_traj_k(k: int):
    with open(ctx().file_traj, 'rb') as f:
        return pickle.load(f)


def eval_nominal(k: int) -> dict:
    """Evaluate the nominal predictor on trajectory k."""
    c = ctx()
    data = _load_traj_k(k)
    pred = NominalPredictor(c.abg.f_bar, c.abg.mu_bar, c.abg.Sigma_bar)
    return pred.trajectory_prediction(LogarithmScore(), data['x'][k], data['u'][k])


def eval_optimal(k: int) -> dict:
    """Evaluate the optimal (oracle) predictor on trajectory k."""
    c = ctx()
    data = _load_traj_k(k)
    pred = NominalPredictor(c.abg.f_bar, c.abg.mu_bar, c.abg.Sigma_bar)
    return pred.trajectory_prediction(LogarithmScore(), data['x'][k], data['u'][k],
                                      sds_params=c.sds_params_shared)


def eval_eig_drpp(k: int) -> dict:
    """Evaluate the Eig-DRPP predictor on trajectory k."""
    c = ctx()
    data = _load_traj_k(k)
    pred = EigDrpp(
        c.abg.f_bar, c.abg.mu_bar, c.abg.Sigma_bar, c.abg.gamma)
    return pred.trajectory_prediction(LogarithmScore(), data['x'][k], data['u'][k])


def eval_noise_drpp(k: int) -> dict:
    """Evaluate the Noise-DRPP predictor on trajectory k (inflated covariance)."""
    c = ctx()
    data = _load_traj_k(k)
    pred = NoiseDrpp(c.abg.f_bar, c.abg.mu_bar, c.abg.Sigma_bar, c.abg.gamma)
    return pred.trajectory_prediction(LogarithmScore(), data['x'][k], data['u'][k])


def worst_against(predictor, k: int) -> dict:
    """Let the adversarial SDS evolve against the given predictor.

    Ordering: each step first updates the predictor with ``(x_i, u_i)`` and **then**
    calls ``step()``, so the adversarial dynamics react to a fresh ``lamb_pred`` (the
    paper's best-response semantics).
    """
    from ppsds.environment.dynamics_mode.adversarial.additive_drift.conic_moment import (
        EigvecFixedAdversarialDynamics,
    )

    c = ctx()
    mode = EigvecFixedAdversarialDynamics(c.abg)
    env = c.env_factory(mode)
    sr = LogarithmScore()

    x, _ = env.reset(seed=c.n_seed + k)
    _, v = eig_real(c.abg.Sigma_bar)

    T = c.T
    d, ud = c.abg.x_dim, c.abg.u_dim
    xs = np.zeros((T + 1, d))
    us = np.zeros((T + 1, ud))
    js = np.zeros(T + 1, dtype=int)
    nus = np.zeros((T + 1, d))
    mus = np.zeros((T + 1, d))
    Sigs = np.zeros((T + 1, d, d))
    xps = np.zeros((T + 1, d))
    Sps = np.zeros((T + 1, d, d))
    scs = np.zeros(T + 1)

    # --- Step 0: initial state / predictor ---
    u = c.policy(x)
    xs[0] = x
    us[0] = u
    xps[0] = predictor.x_pred       # None during construction -> nan
    Sps[0] = predictor.Sigma_pred
    js[0] = -1
    nus[0] = c.abg.f_bar(x, u)
    mus[0] = c.abg.mu_bar
    Sigs[0] = c.abg.Sigma_bar

    for i in range(T):
        predictor.update(x, u)                       # update first -> LP_i
        env.inject_eigvec_fixed_info(predictor.lamb_pred, u)
        x_next, _r, _term, _trunc, info = env.step(u)
        p = info['sds_params']
        u_next = c.policy(x_next)
        j = mode.j                                   # the worst direction for this transition

        xs[i + 1] = x_next
        us[i + 1] = u_next
        xps[i + 1] = predictor.x_pred
        Sps[i + 1] = predictor.Sigma_pred
        js[i + 1] = j
        nus[i + 1] = c.abg.f_bar(x, u) + np.sqrt(c.abg.gamma[0](x, u)) * v[:, j]
        mus[i + 1] = p['mu']
        Sigs[i + 1] = p['Sigma']
        scs[i + 1] = sr.score(predictor.predict(), x_next)

        x, u = x_next, u_next

    return {'x': xs, 'u': us, 'j': js, 'nu': nus, 'mu': mus,
            'Sigma': Sigs, 'x_pred': xps, 'Sigma_pred': Sps, 'score': scs}


def worst_nominal(k: int) -> dict:
    c = ctx()
    return worst_against(
        NominalPredictor(c.abg.f_bar, c.abg.mu_bar, c.abg.Sigma_bar), k)


def worst_eig_drpp(k: int) -> dict:
    c = ctx()
    return worst_against(
        EigDrpp(c.abg.f_bar, c.abg.mu_bar, c.abg.Sigma_bar, c.abg.gamma), k)


def worst_noise_drpp(k: int) -> dict:
    c = ctx()
    return worst_against(
        NoiseDrpp(c.abg.f_bar, c.abg.mu_bar, c.abg.Sigma_bar, c.abg.gamma), k)


# ----------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------

def run_pool(worker_fn, total: int, n_cpu: int, label: str) -> list:
    """Run ``worker_fn(0..total-1)`` over a process pool, with a progress bar."""
    if n_cpu > 1 and total > 1:
        with mp.Pool(processes=min(n_cpu, total)) as pool:
            return list(tqdm(pool.imap(worker_fn, range(total)), total=total, desc=label))
    return [worker_fn(k) for k in tqdm(range(total), desc=label)]


def eval_and_save(worker_fn, name: str, label: str, keys, n_cpu: int, deps=()) -> None:
    """Evaluate and save -- skipped only if **validation** passes (see :mod:`ppsds.experiment.cache`)."""
    c = ctx()
    if c.cache.load(name, keys, c.traj_num, c.T, deps=deps) is not None:
        return
    print(f'Evaluating {label}...')
    results = run_pool(worker_fn, c.traj_num, n_cpu, label)
    c.cache.save(organize_traj_data(keys, results), name, deps=deps)
    print(f'  Saved {label} to {os.path.join(c.cache.dir_out, name)}')


def generate_and_save_trajectories(traj_keys, n_cpu: int) -> None:
    """(random branch) Generate trajectories and save them, skipping if already fresh."""
    c = ctx()
    if c.cache.load('traj.pkl', traj_keys, c.traj_num, c.T) is not None:
        return
    print(f'Generating {c.traj_num} trajectories via env layer...')
    results = run_pool(gen_one_traj, c.traj_num, n_cpu, 'Traj')
    c.cache.save(organize_traj_data(['x', 'u'], results) | c.sds_params_shared, 'traj.pkl')
    print(f'  Saved to {c.file_traj}')
