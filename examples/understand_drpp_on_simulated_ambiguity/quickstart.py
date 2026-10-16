"""Minimal example: **build your own simulated ambiguity**, then compare DRPP predictors on it.

Scene 1 asks "how do I put **my own** environment under DRPP?". Scene 2 asks the reverse:

    I declare an ambiguity set by hand, **generate** an SDS (stochastic dynamical system)
    from it, and let the predictors predict -- comparing them on the same ambiguity set.

Two semantics are the whole point of this example (sections 3 and 4 below):

- ``random`` -- sample **one** SDS parameter trajectory from the ambiguity set. Every
  predictor faces the **same** true SDS (only the noise realizations differ), so this
  compares average performance.
- ``worst``  -- adversarially **pick** the SDS that makes **that** predictor's score worst
  (a best response). Each predictor faces an SDS chosen specifically against itself. This is
  the paper's central semantics.

The ambiguity set only does two things: it declares "how much uncertainty there is"
(section 1) and it generates SDSs. ``Env`` merely consumes that generation ability, in
whatever way its ``DynamicsMode`` specifies.

Run (a few seconds)::

    python examples/understand_drpp_on_simulated_ambiguity/quickstart.py

The runtime is dominated by Eig-DRPP, which solves a cone program at every step. Lower
``N_TRAJ`` or ``T`` below to make it faster.

This script deliberately uses **only the core ppsds surface** -- no ``ppsds.experiment``, no
cache, no command line. For the multi-process batch runner see
``examples/understand_drpp_on_simulated_ambiguity/batch_experiment/run.py``.
"""

import numpy as np

from ppsds.ambiguity.additive_drift.conic_moment import LinearAmbiguitySet
from ppsds.environment.base import Env
from ppsds.environment.dynamics_mode import (
    EigvecFixedAdversarialDynamics,
    PreSampledDynamics,
)
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (
    EigDrpp,
    NoiseDrpp,
    NominalPredictor,
)
from ppsds.scoring_rule.logarithm_score import LogarithmScore

# =====================================================================
# 1. Declare the ambiguity set -- the ONLY source of the model parameters
# =====================================================================
# x_{k+1} = A x_k + B u_k + w_k. The semantics match scene 1:
#   alphas['mu']    -- squared radius of the noise-mean deviation
#   alphas['Sigma'] -- covariance scaling (true covariance is dominated by alphas['Sigma'] * Sigma_bar)
#   f_err_max       -- un-squared bound on the drift norm ||delta_f||
A_bar = np.array([[1.0, 0.1], [0.0, 1.0]])
B_bar = np.identity(2)
mu_bar = np.zeros(2)
Sigma_bar = np.array([[1.0, 0.5], [0.5, 1.5]])
alphas = {'A': 0.3, 'B': 0.3, 'mu': 0.5, 'Sigma': 3}
f_err_max = 5.0

x0 = np.array([2.0, 1.0])
T = 50
PERIOD = 100      # SDS re-sampling period: 100 = slowly varying, 1 = i.i.d. per step
N_TRAJ = 4        # keep it small: this is a demonstration, not a measurement


# =====================================================================
# 2. One rollout, shared by both semantics
# =====================================================================
# random and worst case run the SAME loop; see the docstring below for why no ``if`` is
# needed. Sections 3 and 4 differ only in which mode the loop is handed.


def rollout(make_predictor, make_mode, seed, abg, policy, scorer):
    """Run one trajectory and return the per-step logarithmic scores, shape (T,).

    The order of each step is the paper's best-response timing:

        update the predictor with (x_k, u_k)
        -> inject the predicted eigenvalues (the adversarial mode picks the worst SDS from them)
        -> env.step, i.e. the true system advances
        -> score the prediction that was just made against x_{k+1}

    ``inject_eigvec_fixed_info`` is a **no-op** for the non-adversarial modes (the base
    ``set_info`` does nothing), which is why random and worst case can share this one loop
    without an ``if``.
    """
    env = Env(abg, make_mode(), x0=x0, max_steps=T)
    predictor = make_predictor()
    x, _ = env.reset(seed=seed)
    scores = np.zeros(T)
    for k in range(T):
        u = policy(x)
        predictor.update(x, u)
        env.inject_eigvec_fixed_info(predictor.lamb_pred, u)
        x, _reward, _term, _trunc, _info = env.step(u)
        scores[k] = scorer.score(predictor.predict(), x)
    return scores


if __name__ == '__main__':
    abg = LinearAmbiguitySet(A_bar, B_bar, mu_bar, Sigma_bar, alphas, f_err_max)
    scorer = LogarithmScore()

    def policy(x):
        """Zero control; see run.py for a DLQR policy."""
        return np.zeros(abg.u_dim)

    # All three predictors take their model from the SAME ambiguity set -- do not retype the
    # literals above. The three are peers, and the only real difference is where the
    # predictive covariance comes from:
    #   Nominal    -- Sigma_bar              (trust the nominal model)
    #   Noise-DRPP -- gamma[2] * Sigma_bar   (robust to noise only)
    #   Eig-DRPP   -- one cone program per step (robust to the worst case; the paper's method)
    predictor_factories = {
        'Nominal':    lambda: NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar),
        'Noise-DRPP': lambda: NoiseDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma),
        'Eig-DRPP':   lambda: EigDrpp(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma),
    }

    # =================================================================
    # 3. random -- one shared SDS, faced by every predictor
    # =================================================================
    # Sample a single SDS parameter trajectory (first axis T+1) and let every predictor
    # replay it; the only variation between trajectories is the noise seed. That way we
    # compare predictors, without mixing in the variance of SDS sampling.
    #
    # Seeding is needed because sample_trajectory() draws from the global numpy RNG.
    np.random.seed(0)
    sds = abg.sample_trajectory(T, PERIOD)
    # PreSampledDynamics replays what we sampled. RandomDynamics(abg, period) is the other
    # option: it re-samples per episode from env.np_random, so each trajectory would see a
    # different SDS.
    random_curves = {
        name: np.mean([rollout(pf, lambda: PreSampledDynamics(sds), k, abg, policy, scorer)
                       for k in range(N_TRAJ)], axis=0)
        for name, pf in predictor_factories.items()
    }

    # =================================================================
    # 4. worst -- each predictor faces the SDS chosen against itself
    # =================================================================
    # Note that the mode must be built **per predictor**: it holds the predicted eigenvalues
    # and picks the worst direction from them.
    worst_curves = {
        name: np.mean([rollout(pf, lambda: EigvecFixedAdversarialDynamics(abg), k,
                               abg, policy, scorer)
                       for k in range(N_TRAJ)], axis=0)
        for name, pf in predictor_factories.items()
    }

    # =================================================================
    # 5. Print the comparison
    # =================================================================
    upper = abg.value_upper_bound_true_transition()   # reachable value when the true transition is known
    lower = abg.value_lower_bound_eigvec_fixed()      # value Eig-DRPP guarantees at x = u = 0
    print(f'Mean logarithmic score (higher is better), '
          f'{N_TRAJ} trajectories x {T} steps\n')
    print(f'{"predictor":<12}{"random":>10}{"worst":>10}')
    print('-' * 32)
    for name in predictor_factories:
        print(f'{name:<12}{random_curves[name].mean():>10.3f}'
              f'{worst_curves[name].mean():>10.3f}')
    print(f'\nTheoretical bounds (per step): upper {upper:.3f}, lower {lower:.3f}')
    print('(The lower bound is evaluated at x = u = 0; treat it as a reference magnitude.)')
    print('\nHow to read this: under random SDS the three predictors sit close together -- the')
    print('nominal model is roughly right, so robustness buys little and costs a little. Under')
    print('worst-case SDS they separate sharply: Nominal is driven far down (it is attacked on')
    print('purpose), while Eig-DRPP shrinks its covariance along the worst direction and keeps')
    print('its score near the lower bound.')

    # =================================================================
    # 6. (Optional) plot -- matplotlib is an optional dependency
    # =================================================================
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if plt is None:
        print('\nmatplotlib is not installed; skipping the plot '
              '(install it with: pip install "ppsds[experiments]")')
    else:
        colors = {'Nominal': '#699ECA', 'Noise-DRPP': '#8D73BA', 'Eig-DRPP': '#ED8828'}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, curves, title in (
            (axes[0], random_curves, 'random SDS (one shared SDS)'),
            (axes[1], worst_curves, 'worst-case SDS (best response)'),
        ):
            for name, curve in curves.items():
                cumulative = np.cumsum(curve) / np.arange(1, len(curve) + 1)
                ax.plot(np.arange(1, len(curve) + 1), cumulative, label=name,
                        color=colors[name], linewidth=2)
            ax.axhline(upper, color='#999999', linestyle=':', linewidth=1.5, label='upper bound')
            ax.axhline(lower, color='#999999', linestyle='--', linewidth=1.5, label='lower bound')
            ax.set_title(title)
            ax.set_xlabel('step')
            ax.grid(True, color='#DDDDDD', linewidth=0.8)
        axes[0].set_ylabel('cumulative mean log score')
        axes[0].legend(fontsize=9)
        fig.tight_layout()
        plt.show()
