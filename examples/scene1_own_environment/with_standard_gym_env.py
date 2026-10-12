"""Use DRPP on a **standard gym environment** (a more realistic step than quickstart).

Run::

    python examples/scene1_own_environment/with_standard_gym_env.py      # about 10 seconds

The setting: you are using an off-the-shelf gym environment and have already fixed a control
policy, and now you want to add **probabilistic prediction with a lower-bound guarantee** to it.

There are only two differences from ``quickstart.py``, but they force the real pitfalls out
into the open:

* the environment is not one you built -- you **do not know** its analytic dynamics;
* the nominal model and the ambiguity set therefore have to be **identified from data and
  declared by you**.

This example uses **Acrobot-v1** (6-dimensional state, underactuated, chaotic) -- a linear
nominal model is **clearly not enough** here (the residual is 74% of the state increment),
which makes it a good place to answer a key question:

    **If the model is bad, does DRPP win?**

The answer is no (see section 3). What actually decides it is **whether the bounds you declare
are optimistic or conservative** (see section 4), and **too little identification data** is the
most common source of optimism.
"""

import numpy as np
import gymnasium as gym

from ppsds.dynamics.identification import (       # noqa: E402
    collect_rollouts, fit_linear, diagnose, encode_action, format_report,
)
from ppsds.ambiguity.additive_drift.conic_moment import LinearAmbiguitySet   # noqa: E402
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (     # noqa: E402
    NominalPredictor,
    EigDrpp,
)
from ppsds.scoring_rule.logarithm_score import LogarithmScore    # noqa: E402


# =====================================================================
# 0. A standard gym environment + a policy you have fixed
# =====================================================================

ENV_ID = 'Acrobot-v1'
#: Identification and evaluation use **different** trajectories (different seeds) -- otherwise
#: you are grading yourself on the answers.
EVAL_SEEDS = (601, 602, 603, 604, 605)


def fixed_policy(obs):
    """A fixed "pump energy" heuristic: apply torque in the direction of the angular velocity.

    (The action is discrete, {0,1,2}. Note that everything goes through ``encode_action`` to
    become one-hot before being fed to the predictor -- an action number must **not** be used
    as an ordered numeric value.)
    """
    return 2 if obs[4] + obs[5] > 0 else 0


def build(env, n_episodes, seed0=0):
    """Identify -> declare the ambiguity set. Returns everything the evaluation needs."""
    X, U, X_next = collect_rollouts(env, fixed_policy, n_episodes, 200, seed0=seed0)
    fit = fit_linear(X, U, X_next)
    res = fit.residuals(X, U, X_next)
    diag = diagnose(res, X_next - X)

    r_norm = np.linalg.norm(res, axis=1)
    nx = np.linalg.norm(X, axis=1)
    f_err_max = float(np.percentile(r_norm, 99))
    alphas = {'A': float(np.median(r_norm) / max(np.median(nx), 1e-9)),
              'B': float(2 * np.median(r_norm)),
              'mu': 1e-6, 'Sigma': 1.0}
    abg = LinearAmbiguitySet(fit.A, fit.B, diag['mu_bar'], diag['Sigma'],
                               alphas, f_err_max)
    return fit, diag, abg


def evaluate(env, fit, diag, abg, T=80):
    """Evaluate both predictors on a **separate batch of trajectories**; return mean scores."""
    sr = LogarithmScore()
    pn = NominalPredictor(fit.f_bar, diag['mu_bar'], diag['Sigma'])
    pd = EigDrpp(fit.f_bar, diag['mu_bar'], diag['Sigma'],
                                       abg.gamma)
    sn, sd = [], []
    for seed in EVAL_SEEDS:
        o, _ = env.reset(seed=seed)
        for _ in range(T):
            a = fixed_policy(o)
            u = encode_action(env.action_space, a)    # one-hot, matching what f_bar expects
            pn.update(o, u)
            pd.update(o, u)
            o, _r, term, trunc, _i = env.step(a)
            sn.append(sr.score(pn.predict(), o))
            sd.append(sr.score(pd.predict(), o))
            if term or trunc:
                break
    return float(np.mean(sn)), float(np.mean(sd))


# =====================================================================
if __name__ == '__main__':
    env = gym.make(ENV_ID)

    print('=' * 74)
    print('1. System identification: run a few trajectories and fit a **linear** nominal model')
    print('=' * 74)
    fit, diag, abg = build(env, n_episodes=20)
    print(format_report(diag, fit))
    print()
    print('  How to read this: a residual of **74%** says the linear model is clearly not')
    print('        enough here; while the lag-1 autocorrelation is only **0.03** -- the residual')
    print('        is almost unpredictable, it really does look **like noise** (Acrobot is')
    print('        chaotic).')
    print('        This also settles a common question: whether a residual *looks* like noise')
    print('        has nothing to do with whether it can *be used* as noise.')

    print()
    print('=' * 74)
    print('2. Declare the ambiguity set (same order: pick the family, then fill in its language)')
    print('=' * 74)
    print(f'  f_err_max = {abg.f_err_max:.3f}  (99th percentile of the residual norm)')
    print(f'  alphas    = A={abg.alphas["A"]:.3f} (scales with ||x||), '
          f'B={abg.alphas["B"]:.3f} (scales with ||u||), mu=1e-6, Sigma=1.0')

    # =================================================================
    print()
    print('=' * 74)
    print('3. First question: the model is this bad, so did DRPP win?')
    print('=' * 74)
    n0, d0 = evaluate(env, fit, diag, abg)
    print('  identification data is plentiful (20 episodes), evaluation uses **another 5** '
          'trajectories (different seeds):')
    print(f'    nominal = {n0:8.3f}      Eig-DRPP = {d0:8.3f}      '
          f'-> {"nominal" if n0 > d0 else "Eig-DRPP"} wins')
    print()
    print('  **It did not.** The reason is not "the model is good", but that `Sigma_bar` was')
    print('  **estimated from your own residuals**, so nominal\'s predictive distribution is on')
    print('  the **same scale** as the errors it will meet -- it is well calibrated.')
    print('  => **"The model is bad" is by itself not enough to make DRPP win.**')

    # =================================================================
    print()
    print('=' * 74)
    print('4. What actually decides it: whether your declared bounds are optimistic or conservative')
    print('=' * 74)
    print('  Shrink the identification data (very common in real work: you do not get many')
    print('  chances to run trajectories). The evaluation trajectories never change. Note that')
    print('  **no** artificial perturbation is injected here.')
    print()
    print(f'  {"ident. data":>12} {"est. resid std":>14} {"nominal":>10} {"Eig-DRPP":>10}   winner')
    print('  ' + '-' * 62)
    rows = []
    for tag, nep in (('1 episode', 1), ('2 episodes', 2), ('4 episodes', 4), ('20 episodes', 20)):
        f2, d2, a2 = build(env, nep)
        n, d = evaluate(env, f2, d2, a2)
        rows.append((tag, float(np.sqrt(np.diag(d2['Sigma'])).mean()), n, d))
        print(f'  {tag:>12} {rows[-1][1]:>14.4f} {n:>10.3f} {d:>10.3f}   '
              f'{"nominal" if n > d else "**Eig-DRPP**"}')

    thin, thick = rows[0], rows[-1]
    print()
    print('  How to read this:')
    print(f'  - Little data ({thin[0]}, residual std estimated at {thin[1]:.3f}) -> `Sigma_bar` is')
    print(f'    **underestimated by about {thick[1]/thin[1]:.1f}x** -> nominal\'s predictive')
    print('    distribution is **too narrow** ->')
    print(f'    it scores very low on the true errors ({thin[2]:.2f}), while Eig-DRPP holds at '
          f'{thin[3]:.2f}.')
    print(f'  - Plenty of data ({thick[0]}, residual std ~ {thick[1]:.3f}) -> `Sigma_bar` is')
    print(f'    calibrated -> nominal wins ({thick[2]:.2f} vs {thick[3]:.2f}) -- and then DRPP\'s')
    print('    robustness is pure insurance premium.')
    print()
    print('  => In one line: **DRPP helps you when you declared optimistically, and charges you a')
    print('     premium when you declared conservatively.**')
    print('     Too little identification data is the most common source of optimism.')
    print()
    print('  Whether to use DRPP depends on your confidence in your own declaration -- not on')
    print('  how accurate the model is.')
    print()
    print('  Companion piece (a different pitfall that will stop you cold: a **structurally')
    print('  singular** residual covariance, and "regularization is modeling"):')
    print('      python examples/scene1_own_environment/pendulum_singular_covariance.py')
