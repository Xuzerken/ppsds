"""Reproduce the figure of "Probabilistic predictability of stochastic dynamical systems".

Run::

    python examples/understand_predictability/quickstart.py        # about 2 seconds

The paper (Xu, Li, He, *Automatica* 174 (2025) 112160) proposes an ``epsilon``-logarithm score
and characterises the **predictability** of a system as the best expected score any predictor
could reach. This script reproduces its only figure in miniature, and prints the numbers the
figure's two dotted lines are.

Three things are being demonstrated, in the order the paper argues them:

1. **What the score is.** A predictive law is scored on how much of its probability sits within
   ``eps`` of what actually happened -- the neighbourhood is what makes the score finite for
   predictions that would be hopeless under the ordinary logarithm score.
2. **The two reference lines.** The red one is the predictability upper bound
   ``d_x log(2 eps) - h_d(q)``: a property of the *system*, with no predictor in it. The blue one
   is ``d_x log(2 eps) - H(q || p)``, the ``O(eps)`` approximation for the predictor actually in
   use -- here a **deliberately misspecified** Gaussian, because the paper's point is that the
   approximation is good even for a predictor that is not the best one.
3. **Why one trajectory is enough.** Each thin curve is a *single* trajectory's running average,
   and they land on the expected score within a couple of dozen steps. That is the paper's
   ``O(T ** -0.5)`` convergence, and it is why its figure does not need repeated samplings.

The system, its noise and the predictor are declared below and nowhere else.
"""

import numpy as np
from scipy.stats import multivariate_normal

from ppsds.dynamics.known_sds import AdditiveDriftSds
from ppsds.numerics.random_variables import multivariate_uniform
from ppsds.predictability.bounds import approximation_value, predictability_upper_bound
from ppsds.predictability.entropy import cross_entropy, differential_entropy
from ppsds.predictability.trajectory import across_trajectories, cumulative_average
from ppsds.scoring_rule.epsilon_logarithm_score import EpsilonLogarithmScore

# =====================================================================
# 1. The paper's system -- a shear, driven by uniform noise
# =====================================================================
# x_{k+1} = A x_k + w_k, with w i.i.d. uniform on the cube [-1, 1]^2. Note the noise is *not*
# Gaussian: that is the whole point of this example, and it is what an ambiguity set could not
# have expressed (its transition is Gaussian by construction). A system known exactly, with a
# noise law you name, is ppsds.dynamics.known_sds.

A_TRUE = np.array([[1.0, 1.0],
                   [0.0, 1.0]])

X0 = np.array([0.0, 0.0])
T = 30
N_TRAJ = 2000
EPSILONS = (0.25, 0.5, 0.75)


def drift(x, u):
    """The paper's drift. It takes no control input; ``u`` is accepted and ignored."""
    return A_TRUE @ np.asarray(x, dtype=float)


def paper_noise():
    """``U([-1, 1]^2)``, the paper's process noise."""
    return multivariate_uniform(low=-np.ones(2), high=np.ones(2))


def paper_predictor():
    """The paper's predictor: a standard Gaussian on the noise, deliberately misspecified.

    It predicts the *noise*, so the prediction for the next state is this law shifted by the
    drift -- the reduction the paper uses, and the reason the scores below are computed on the
    noise rather than on the state.
    """
    return multivariate_normal(mean=np.zeros(2), cov=np.eye(2))


# =====================================================================
# 2. One rollout, to show what is being predicted
# =====================================================================
# The sweep below never needs the states, and the comment there says why. One real trajectory
# first, so "predicting this system" is not an abstraction.

sds = AdditiveDriftSds(drift, paper_noise(), x0=X0)
rollout = sds.sample_trajectory(T, rng=np.random.default_rng(0))

print('The system: x_{k+1} = A x_k + w_k, w ~ U([-1,1]^2), T = %d' % T)
print(f'  A = {A_TRUE.tolist()}')
print('  x_0 = %s' % X0.tolist())
print('  first three states of one rollout:')
for step in range(3):
    print('    x_%-2d = [% .4f, % .4f]   w_%-2d = [% .4f, % .4f]'
          % (step + 1, rollout['x'][step + 1][0], rollout['x'][step + 1][1],
             step, rollout['w'][step][0], rollout['w'][step][1]))

# =====================================================================
# 3. The two reference lines
# =====================================================================
# Both are expectations under the true law, so both are computed from the laws themselves --
# no sampling. The upper bound needs only the truth's differential entropy; the approximation
# additionally needs the predictor's cross-entropy against it.

q = paper_noise()
p = paper_predictor()

print()
print('Reference values, from the laws (not from samples):')
print('  differential entropy of the noise   h_d(q)    = %+.5f' % differential_entropy(q))
print('  cross-entropy against the predictor H(q || p) = %+.5f' % cross_entropy(q, p))

# =====================================================================
# 4. The sweep -- one vectorised pass, no state simulation
# =====================================================================
# The predictor predicts the noise, so the realised outcome it is scored on is the noise draw:
#     L_eps(p_hat_w(.- f(x_k)), x_{k+1})  ==  L_eps(p_hat_w, x_{k+1} - f(x_k))  ==  L_eps(p_hat_w, w_k)
# The drift cancels. That identity is why the paper's experiment is a single pass over noise
# draws rather than a simulation, and it is why this stays fast at 100,000 trajectories.

noise = paper_noise().rvs(size=N_TRAJ * T, rng=np.random.default_rng(9))

print()
print('%-6s %12s %12s %12s %10s %10s' %
      ('eps', 'predictability', 'approximation', 'measured', '|err|', '|err|/eps'))
print('-' * 76)

panels = []
for eps in EPSILONS:
    scores = EpsilonLogarithmScore(eps).score(p, noise).reshape(N_TRAJ, T)
    measured = scores.mean()
    upper = predictability_upper_bound(q, eps)
    approximation = approximation_value(q, p, eps)
    error = abs(measured - approximation)
    panels.append({'eps': eps, 'scores': scores, 'upper': upper, 'approximation': approximation})
    print('%-6s %12.5f %12.5f %12.5f %10.5f %10.5f'
          % (eps, upper, approximation, measured, error, error / eps))

# =====================================================================
# 5. What the figure's curves look like at the horizon
# =====================================================================

print()
print('At the horizon (step %d), the running averages:' % T)
print('%-6s %12s %12s %12s' % ('eps', 'mean', '2.5%', '97.5%'))
print('-' * 46)
for panel in panels:
    aggregate = across_trajectories(cumulative_average(panel['scores']))
    print('%-6s %12.5f %12.5f %12.5f'
          % (panel['eps'], aggregate['mean'][-1], aggregate['low'][-1], aggregate['high'][-1]))

# =====================================================================
# 6. How to read this
# =====================================================================
print("""
How to read this
----------------
* **predictability** (the red dotted line) is an upper bound on what *any* predictor could
  score on this system. No predictor is involved in computing it -- it is a property of the
  system, and it is what "probabilistic predictability" means.
* **approximation** (the blue dotted line) is what the paper says the *given* predictor should
  score, up to an error of order eps. It is close: the measured column tracks it, and the
  error grows with eps exactly as the paper reports (about 0.1 at the largest radius).
* The gap between the two lines is the price of the predictor being misspecified -- it is not
  a statement that the predictor is badly implemented.
* The two lines do **not** meet at eps = 0. The score is discontinuous there: at exactly
  eps = 0 it is log p(x), while for any eps > 0 it carries a d_x * log(2 eps) volume term that
  rushes to -infinity. Neither line is a limit of the other.
* Run ``batch_experiment/run.py --save-figure`` for the paper's three-panel figure at 100,000
  trajectories.
""")

# =====================================================================
# 7. (Optional) the figure itself
# =====================================================================
# Matplotlib is an optional extra (`pip install ppsds[experiments]`), so its absence degrades
# the plot rather than the run -- the same convention the scene-2 example uses.

try:
    import matplotlib.pyplot as plt

    from ppsds.visualization.predictability import plot_predictability_panel
except ImportError:
    print('(matplotlib not installed -- skipping the figure; `pip install ppsds[experiments]`)')
else:
    # Same panel size as batch_experiment/run.py, and as the published figure (see the note there).
    figure, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.8), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        plot_predictability_panel(ax, panel['scores'], eps=panel['eps'],
                                  upper_bound=panel['upper'],
                                  approximation=panel['approximation'])
        ax.set_title(rf'$\epsilon = {panel["eps"]}$', fontsize=11)
    figure.tight_layout()
    out = 'examples/understand_predictability/predictability.png'
    figure.savefig(out, dpi=120, bbox_inches='tight')
    print(f'(figure written to {out})')
