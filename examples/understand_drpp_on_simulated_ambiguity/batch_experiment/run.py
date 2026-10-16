"""Batch experiment runner (scene 2: custom simulated ambiguity), over Gymnasium.

**This is the multi-process / cached runner**, meant for large batch runs. If you want to
see how to drive ppsds by hand -- one trajectory, one predictor, no CLI -- read
``examples/understand_drpp_on_simulated_ambiguity/quickstart.py`` instead.

One model table (:mod:`models`) covers all seven models -- five linear plus two nonlinear.
What differs between systems is carried by the **ambiguity-set subclass**
(``LinearAmbiguitySet`` / ``NonlinearAmbiguitySet``), so every model runs through the same
generic ``Env``.

Usage
-----
::

    cd examples/understand_drpp_on_simulated_ambiguity/batch_experiment

    # randomly sampled SDS
    python run.py -n 8 -s 3 -p 100 -m final_round -c none -t 100

    # worst-case SDS
    python run.py -n 8 -s 3 -m NLink -c none -t 100 -e worst

``-c regulation`` is only valid for the linear models; nonlinear models **exit with an
error** (no control policy is implemented for them).

Results land in ``../data/gym/...`` -- the scene's own ``data/`` root, resolved from this
file's location rather than the working directory, so that both entry points share one
result tree and the run can be launched from anywhere.

.. warning::
   The multiprocessing workers rely on ``fork`` (NLink's ambiguity set holds lambdas that
   cannot be pickled). ``configure()`` in the runner module asserts this and fails loudly
   otherwise -- see :mod:`ppsds.experiment.drpp.additive_drift.conic_moment`.
"""

import argparse
import os
from pathlib import Path

import numpy as np

from ppsds.control_policy.lqr import DLQR
from ppsds.environment.base import Env
from ppsds.environment.dynamics_mode import PreSampledDynamics
from ppsds.experiment.cache import CacheContext
from ppsds.experiment.drpp.additive_drift import conic_moment as runner
from ppsds.visualization.covariance import (
    plot_ellipses_compare_predictors,
    plot_ellipses_compare_predictors_worst_case,
)
from ppsds.visualization.curve import plot_error_bar

import models


#: Episode length (truncation horizon).
T = 50

#: The scene root, i.e. ``examples/understand_drpp_on_simulated_ambiguity/``. Output goes to
#: ``<SCENE_ROOT>/data/gym/...`` -- see the module docstring.
SCENE_ROOT = Path(__file__).resolve().parents[1]

# Keys of the per-predictor result pickles.
PRED_KEYS = ['x_pred', 'Sigma_pred', 'score']
EIG_KEYS = ['x_pred', 'j_pred', 'Sigma_pred', 'score']
WORST_KEYS = ['x', 'u', 'j', 'nu', 'mu', 'Sigma', 'x_pred', 'Sigma_pred', 'score']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Run the DRPP batch experiment for one model.')
    parser.add_argument("--n_cpu", "-n", default=8, type=int,
                        help="number of worker processes")
    parser.add_argument("--n_seed", "-s", default=3, type=int)
    parser.add_argument("--period", "-p", default=100, type=int,
                        help="SDS re-sampling period (1=i.i.d., 100=slowly-varying)")
    parser.add_argument("--model_type", "-m", default='final_round', type=str,
                        choices=list(models.ALL_MODELS))
    parser.add_argument("--control_type", "-c", default='none', type=str,
                        choices=['none', 'regulation'])
    parser.add_argument("--experiment", "-e", default='random', type=str,
                        choices=['random', 'worst'],
                        help="random = randomly-sampled SDS; worst = worst-case adversarial SDS")
    parser.add_argument("--n_traj", "-t", default=100, type=int,
                        help="number of trajectories")
    parser.add_argument("--force", "-f", action="store_true",
                        help="ignore the cache and recompute everything "
                             "(overwrites existing pickles)")
    return parser.parse_args(argv)


def build_policy(model_type, spec, control_type):
    """Regulation policy for the linear models, zero control otherwise."""
    if control_type == 'regulation':
        if not spec.is_ellipse_model:
            raise SystemExit(
                f"{model_type} does not support -c regulation "
                f"(its control policy is not implemented)")
        Q = np.identity(spec.ambig.x_dim)
        R = np.identity(spec.ambig.u_dim)
        x_d = np.zeros(spec.ambig.x_dim)
        controller = DLQR(Q, R, spec.ambig.A_bar, spec.ambig.B_bar)
        return lambda x: controller(x, x_d)
    return lambda x: np.zeros(spec.ambig.u_dim, dtype=np.float64)


def visualize(experiment, control_type, period, spec, file_traj, preds,
              value_upper_bound, value_lower_bound):
    """Draw the error bar always, and the ellipse panel when the model has one."""
    plot_error_bar(preds, value_upper_bound, value_lower_bound,
                   **models.ERROR_BAR_SPECS[(experiment, spec.is_ellipse_model)])
    if not spec.is_ellipse_model:
        return

    pspec = models.plot_spec(experiment, control_type, period)
    if pspec is None:
        return
    if experiment == 'random':
        plot_ellipses_compare_predictors(
            file_traj, preds, pspec['time_indexes'], k=pspec['k'],
            xlim=pspec['xlim'], ylim=pspec['ylim'],
            local_ax=pspec.get('local_ax'))
    else:
        plot_ellipses_compare_predictors_worst_case(
            preds, pspec['time_indexes'], k=pspec['k'],
            xlim=pspec['xlim'], ylim=pspec['ylim'])


def main(argv=None):
    args = parse_args(argv)
    model_type, control_type = args.model_type, args.control_type
    period, traj_num, n_cpu = args.period, args.n_traj, args.n_cpu
    experiment = args.experiment

    # ---- model, control policy, env factory ----
    spec = models.build(model_type, control_type)
    abg, x0 = spec.ambig, spec.x0
    policy = build_policy(model_type, spec, control_type)
    value_upper_bound = abg.value_upper_bound_true_transition()
    value_lower_bound = abg.value_lower_bound_eigvec_fixed()

    def make_env(mode):
        return Env(abg, mode=mode, x0=x0, max_steps=T)

    # ---- output paths ----
    dir_out = SCENE_ROOT / 'data' / 'gym' / spec.path / f'seed={args.n_seed}' / (
        f'period={period}' if experiment == 'random' else 'worst-case')
    os.makedirs(dir_out, exist_ok=True)
    files = {name: str(dir_out / f'{name}.pkl')
             for name in ('traj', 'nominal', 'optimal', 'eig_drpp', 'noise_drpp')}

    # ---- shared SDS parameters ----
    np.random.seed(args.n_seed)
    sds_params_shared = None
    mode = None            # the worst branch builds its own mode per worker
    if experiment == 'random':
        sds_params_shared = abg.sample_trajectory(T, period)
        mode = PreSampledDynamics(sds_params_shared)

    # The snapshot must list every parameter that determines the output. The cache key used
    # to be just the directory path, **without** traj_num / T / x0, so a directory written by
    # `-t 5` was silently reused for `-t 100`. See ppsds/experiment/cache.py.
    cache = CacheContext(str(dir_out), snapshot={
        'model_type': model_type, 'control_type': control_type,
        'experiment': experiment, 'n_seed': args.n_seed, 'period': period,
        'traj_num': traj_num, 'T': T, 'x0': x0,
        **spec.signature,
    }, force=args.force)

    # ⚠️ configure() must come **before** creating any Pool -- it also asserts fork (NLink's
    # ambiguity set holds unpicklable lambdas, and switching to spawn would fail silently).
    runner.configure(runner.ExperimentContext(
        abg=abg, mode=mode, x0=x0, T=T, traj_num=traj_num, n_seed=args.n_seed,
        policy=policy, env_factory=make_env, file_traj=files['traj'],
        cache=cache, sds_params_shared=sds_params_shared))

    print('=' * 70)
    print(f'DRPP Gymnasium [{experiment}]: {model_type}, '
          f'period={period if experiment == "random" else "N/A"}, '
          f'control={control_type}')
    print('=' * 70)

    if experiment == 'random':
        runner.generate_and_save_trajectories(spec.traj_keys, n_cpu)
        # Evaluate the predictors (all of them depend on the shared sds_params in traj.pkl).
        deps = ['traj.pkl']
        runner.eval_and_save(runner.eval_nominal, 'nominal.pkl', 'Nominal', PRED_KEYS, n_cpu, deps)
        runner.eval_and_save(runner.eval_optimal, 'optimal.pkl', 'Optimal', PRED_KEYS, n_cpu, deps)
        runner.eval_and_save(runner.eval_eig_drpp, 'eig_drpp.pkl', 'Eig-DRPP', EIG_KEYS, n_cpu, deps)
        runner.eval_and_save(runner.eval_noise_drpp, 'noise_drpp.pkl', 'Noise-DRPP', PRED_KEYS, n_cpu, deps)
        preds = {'Nominal': files['nominal'], 'Eig-DRPP': files['eig_drpp'],
                 'Noise-DRPP': files['noise_drpp'], 'Optimal': files['optimal']}
    else:
        # Worst-case SDS: there is no traj.pkl; the three files are independent.
        runner.eval_and_save(runner.worst_nominal, 'nominal.pkl', 'Worst-Nominal', WORST_KEYS, n_cpu)
        runner.eval_and_save(runner.worst_eig_drpp, 'eig_drpp.pkl', 'Worst-Eig-DRPP', WORST_KEYS, n_cpu)
        runner.eval_and_save(runner.worst_noise_drpp, 'noise_drpp.pkl', 'Worst-Noise-DRPP', WORST_KEYS, n_cpu)
        preds = {'Nominal': files['nominal'], 'Eig-DRPP': files['eig_drpp'],
                 'Noise-DRPP': files['noise_drpp']}

    visualize(experiment, control_type, period, spec, files['traj'], preds,
              value_upper_bound, value_lower_bound)

    print(f'\n{cache.summary()}')
    print(f'Done! Results saved to {dir_out}/')


if __name__ == '__main__':
    main()
