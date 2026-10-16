"""**Another pitfall** with standard gym environments: a structurally singular residual
covariance (Pendulum).

Run::

    python examples/apply_drpp_to_your_own_env/pendulum_singular_covariance.py      # about 5 seconds

Its companion ``with_standard_gym_env.py`` (Acrobot) covers "**when** does DRPP actually win".
This one covers a different pitfall that stops you cold when wiring up a real environment --
**the `Sigma_bar` you computed cannot be used as is**.

The pitfall looks like this: DRPP evaluates predictions with the **logarithmic score**, and
that score requires the predictive distribution to **have a density**. But the residual
covariance you estimate from data **is often numerically singular**, and the support of a
degenerate distribution is a null set -- so almost every value scores ``-inf``. This example
runs into it head-on, and explains why "regularization" is not a stopgap but a **step of
modeling**.
"""

from functools import partial

import numpy as np
import gymnasium as gym
from scipy.stats import multivariate_normal

from ppsds.dynamics.identification import (       # noqa: E402
    collect_rollouts, fit_linear, diagnose, regularize, format_report,
)
from ppsds.ambiguity.additive_drift.conic_moment import LinearAmbiguitySet   # noqa: E402
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (     # noqa: E402
    NominalPredictor,
    EigDrpp,
)
from ppsds.scoring_rule.logarithm_score import LogarithmScore    # noqa: E402


ENV_ID = 'Pendulum-v1'


def fixed_policy(obs, k_p=2.0, k_d=1.0):
    """A fixed PD controller (the policy you already have)."""
    theta = np.arctan2(obs[1], obs[0])
    return np.clip(-k_p * theta - k_d * obs[2], -2.0, 2.0).reshape(1)


if __name__ == '__main__':
    env = gym.make(ENV_ID)

    # =================================================================
    print('=' * 74)
    print('1. System identification (run a few trajectories, fit a linear nominal model)')
    print('=' * 74)
    X, U, X_next = collect_rollouts(env, fixed_policy, n_episodes=20, T=200)
    fit = fit_linear(X, U, X_next)
    res = fit.residuals(X, U, X_next)
    diag = diagnose(res, X_next - X)
    print(format_report(diag, fit))
    print()
    print('  How to read this: a residual/increment of ~20% says the linear model cannot capture')
    print('        Pendulum\'s nonlinearity; it is clearer component by component (the third')
    print('        residual component is exactly 0 -- see section 2):')
    inc_std = (X_next - X).std(0)
    print(f'          residual std / increment std = {np.round(res.std(0)/np.where(inc_std>0, inc_std, 1), 3)}')

    Sigma_raw = diag['Sigma']
    mu_bar = diag['mu_bar']

    # =================================================================
    print()
    print('=' * 74)
    print('2. The pitfall: a **structurally singular** residual covariance -- and not because')
    print('   the residuals are small')
    print('=' * 74)
    print(f'  covariance eigenvalues = {np.array2string(diag["eigenvalues"], precision=3)}')
    print(f'  condition number       = {diag["condition_number"]:.2e}')
    print()
    print('  The cause is **structural**, unrelated to how accurate the model is. Pendulum\'s')
    print('  observation is (cos(theta), sin(theta), theta_dot):')
    print('    - the observation lies on the **2-dimensional cylinder** cos^2+sin^2=1 (in 3-D)')
    print(f'      measured max|cos^2+sin^2-1| = {np.abs(X[:,0]**2+X[:,1]**2-1).max():.1e}')
    print('    - and theta_dot updates are **exactly linear** in the observation:')
    print('      theta_dot\' = theta_dot + (3g/2l*sin(theta) + 3/(ml^2)*u)*dt -- and sin(theta)')
    print('      is right there in the observation')
    print('      -> that component\'s residual is identically 0')
    print()
    print('  Each environment has its own structural source: for Pendulum it is a **manifold')
    print('  constraint**, for CartPole it is an **exact Euler step** like x\' = x + dt*x_dot')
    print('  (residual ~1e-9).')

    # ---- feed the unregularized covariance straight in ----
    print()
    print('  Consequence: the logarithmic score requires the predictive distribution to **have a')
    print('  density**, whereas a degenerate distribution\'s support is a null set --')
    p0 = NominalPredictor(
        fit.f_bar, mu_bar, Sigma_raw,
        multivariate_rv=partial(multivariate_normal, allow_singular=True))
    o, _ = env.reset(seed=7)
    p0.update(o, fixed_policy(o))
    nxt, *_ = env.step(fixed_policy(o))
    d_vec = nxt - p0.x_pred
    print(f'    measured: the realized value is {np.linalg.norm(d_vec):.2e} away from the')
    print(f'              **predictive mean**, and the score is '
          f'{LogarithmScore().score(p0.predict(), nxt)}')
    print('    -- a slight departure from the support is already -inf. And DRPP\'s selling point')
    print('       is precisely a **lower bound on the score**; a lower bound of -inf means')
    print('       nothing.')

    # =================================================================
    print()
    print('=' * 74)
    print('3. Regularization: this step **is itself a modeling statement**')
    print('=' * 74)
    print('  You must lift Sigma to be positive definite. This is not a stopgap -- **even a')
    print('  dimension that looks deterministic can be given uncertainty**. The floor is "how')
    print('  much uncertainty you are willing to admit in those directions".')
    print()
    print('  WARNING: but the magnitude of the floor **directly sets the magnitude of the score**')
    print('  (it sets the width of the predictive distribution):')
    print()
    print(f'  {"floor":>9} {"min eigenvalue":>15} {"nominal":>10} {"Eig-DRPP":>10}')
    print('  ' + '-' * 46)
    sr = LogarithmScore()
    results = []
    for floor in (1e-6, 1e-4, 1e-3, 1e-2):
        S = regularize(Sigma_raw, floor)
        r_norm = np.linalg.norm(res, axis=1)
        nx = np.linalg.norm(X, axis=1)
        alphas = {'A': float(np.median(r_norm) / max(np.median(nx), 1e-9)),
                  'B': float(2 * np.median(r_norm)), 'mu': 1e-6, 'Sigma': 1.0}
        abg = LinearAmbiguitySet(fit.A, fit.B, mu_bar, S, alphas,
                                   float(np.percentile(r_norm, 99)))
        pn = NominalPredictor(fit.f_bar, mu_bar, S)
        pd = EigDrpp(fit.f_bar, mu_bar, S, abg.gamma)
        sn, sd = [], []
        o, _ = env.reset(seed=123)
        for _ in range(200):
            a = fixed_policy(o)
            pn.update(o, a)
            pd.update(o, a)
            o, _r, term, trunc, _i = env.step(a)
            sn.append(sr.score(pn.predict(), o))
            sd.append(sr.score(pd.predict(), o))
            if term or trunc:
                break
        results.append((floor, np.min(np.linalg.eigvalsh(S)), np.mean(sn), np.mean(sd)))
        print(f'  {floor:>9.0e} {results[-1][1]:>15.2e} {results[-1][2]:>10.3f} '
              f'{results[-1][3]:>10.3f}')

    lo, hi = results[0], results[-1]
    print()
    print(f'  As the floor goes from {lo[0]:.0e} to {hi[0]:.0e}, nominal\'s score moves from '
          f'{lo[2]:.2f} to {hi[2]:.2f}')
    print(f'  -- a span of {abs(hi[2]-lo[2]):.1f} points. **It is dominated by the width you')
    print('  declared, not by prediction quality.**')
    print()
    print('  So the floor should not be a number you invent. Two defensible sources:')
    print('    1. physics: perturbations you do not observe, actuator/sensor noise, unmodeled')
    print('       high-frequency dynamics;')
    print('    2. consistency: pick a starting value, run it once, and see whether the resulting')
    print('       magnitude matches your intuition about "how accurate the prediction is".')

    # =================================================================
    print()
    print('=' * 74)
    print('4. So when does DRPP actually win?')
    print('=' * 74)
    print('  At the same floor above, Eig-DRPP sits consistently **below** nominal -- the true')
    print('  system is near the nominal model, so robustness here is pure cost.')
    print()
    print('  That is not a defect but how it is priced: **average performance traded for a')
    print('  worst-case floor**. For when that insurance pays off, see the companion piece:')
    print('      python examples/apply_drpp_to_your_own_env/with_standard_gym_env.py  # Acrobot, with a two-regime comparison')
