"""Minimal example: put DRPP on **your own** gymnasium environment (scene 1).

Run::

    python examples/scene1_own_environment/quickstart.py      # about 1 second, 50 steps

This example answers: "I have my own stochastic dynamical system -- how do I use DRPP on it?"

The point: **you do not need to modify the environment.** What you do is **declare three
things**, and the order matters:

    1. the nominal model        -- what I believe my system is (independent of which
                                   ambiguity-set family you pick)
    2. the **family** of the ambiguity set -- which description of uncertainty?
                                   (**decide this first**)
    3. the **parameterization** within that family -- describe "how much uncertainty there is"
                                   in that family's language

Do not skip step 2 before step 3. Different families describe uncertainty in completely
different ways, so you must fix the family first and then fill in parameters in that family's
language -- a habit worth forming from the start.

(The DRPP predictors can currently also be fed the four objects ``f_bar, mu_bar, Sigma_bar,
gamma`` directly, without going through an ambiguity set. But that is a historical accident
of "only the conic-moment family is implemented so far" and is **not portable** -- it breaks
the moment you change family. Do not use it that way.)
"""

import numpy as np
import gymnasium as gym

from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (      # noqa: E402
    NominalPredictor,
    EigDrpp,
)
from ppsds.scoring_rule.logarithm_score import LogarithmScore   # noqa: E402


# =====================================================================
# 0. Your own environment
# =====================================================================
# This section stands in for "what the user already has" -- any gymnasium environment will
# do. A two-state linear system plays that role here; in a real setting it might be your
# robot arm, a battery model, a chemical process... DRPP does not care how it is implemented.

class MySystem(gym.Env):
    """x_{k+1} = A x_k + B u_k + w_k, with w_k ~ N(0, Sigma).

    The true system (the one DRPP has to cope with) -- note that it is **wilder than the
    model the user believes in**: A is deliberately offset from the declared nominal value,
    so that DRPP's conservatism has something to buy.
    """

    def __init__(self, seed=0):
        self.A = np.array([[1.0, 0.14], [0.0, 1.0]])     # truth
        self.B = np.array([[1.0], [0.0]])
        self.Sigma = np.array([[1.6, 0.7], [0.7, 2.0]])  # true noise
        self._rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,))
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,))
        self.x = np.zeros(2)

    def reset(self, seed=None, options=None):
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        self.x = np.array([2.0, 1.0])
        return self.x.copy(), {}

    def step(self, u):
        u = np.asarray(u, dtype=float).reshape(-1)
        self.x = self._rng.multivariate_normal(self.A @ self.x + self.B @ u,
                                               self.Sigma)
        return self.x.copy(), 0.0, False, False, {}


# =====================================================================
# 1. The nominal model -- independent of which ambiguity-set family you pick
# =====================================================================
# This is "what you think your system is". Changing family does **not** change this section.

A_bar = np.array([[1.0, 0.10], [0.0, 1.0]])
B_bar = np.array([[1.0], [0.0]])
mu_bar = np.zeros(2)
Sigma_bar = np.array([[1.0, 0.45], [0.45, 1.30]])


# =====================================================================
# 2. First fix the **family** of the ambiguity set
# =====================================================================
# Different families describe "how much uncertainty there is" in **completely different**
# ways:
#
#     conic moment  -- a few "radii": drift perturbation size, noise-mean deviation,
#                      covariance scaling (the one this repo implements)
#     other kinds   -- each family describes the same thing in its own language, so the family
#                      you pick decides what step 3 asks you to fill in
#
# So the order is: **choose the family here first**. Only then do you know what parameters
# the next step has to supply.

from ppsds.ambiguity.additive_drift.conic_moment import LinearAmbiguitySet   # noqa: E402


# =====================================================================
# 3. Within that family, declare **its** parameterization
# =====================================================================
# The conic-moment family's language is the set of radii below. Note that several involve a
# **square**, which is the easiest place to make a mistake:
#
#   f_err_max   -- bound on the drift **norm** (**un-squared**): ||true drift - f_bar(x,u)|| <= f_err_max
#   alphas['mu'] -- **squared** radius of the noise mean: |true mean - mu_bar| <= sqrt(alphas['mu'])
#   alphas['Sigma'] -- covariance scaling: true covariance is dominated by alphas['Sigma'] * Sigma_bar
#   alphas['A'] / alphas['B'] -- largest perturbation of each entry of A and B (un-squared)

alphas = {'A': 0.30, 'B': 0.30, 'mu': 0.25, 'Sigma': 3.0}
f_err_max = 0.30

abg = LinearAmbiguitySet(A_bar, B_bar, mu_bar, Sigma_bar, alphas, f_err_max)


# =====================================================================
# 4. The predictor takes the model it needs from the **ambiguity set**
# =====================================================================
# Note that it comes off `abg`, rather than retyping the literals above -- that way "the model
# the predictor uses" and "the model the ambiguity set declares" agree by construction, and
# nothing gets missed when you change family.

if __name__ == '__main__':
    T = 50
    env = MySystem()
    scorer = LogarithmScore()

    # Two predictors, both taking their model from the same ambiguity set. The only difference
    # is "does it account for the ambiguity":
    #   nominal  -- trust the nominal model (the ordinary approach)
    #   Eig-DRPP -- optimal in the worst case (DRPP's contribution)
    predictors = {
        'nominal': NominalPredictor(abg.f_bar, abg.mu_bar, abg.Sigma_bar),
        'Eig-DRPP': EigDrpp(
            abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma),
    }
    scores = {name: [] for name in predictors}

    x, _ = env.reset(seed=42)
    for _ in range(T):
        u = np.zeros(1)                       # zero control; swap in your own policy
        for pred in predictors.values():
            pred.update(x, u)                 # update the prediction from the current observation
        x_next, _r, _term, _trunc, _info = env.step(u)
        for name, pred in predictors.items():
            scores[name].append(scorer.score(pred.predict(), x_next))
        x = x_next

    print(f'Running {T} steps on **one ordinary trajectory** (your own environment, zero control)\n')
    print(f'{"predictor":<12}{"mean log score":>16}   note')
    print('-' * 64)
    print(f'{"nominal":<12}{np.mean(scores["nominal"]):>16.3f}   trusts the nominal model')
    print(f'{"Eig-DRPP":<12}{np.mean(scores["Eig-DRPP"]):>16.3f}   made robust to the worst case')
    print()
    print("How to read this: on this trajectory nominal usually scores **higher** -- it fits the")
    print("nominal model better. DRPP trades a little average performance for a **guaranteed")
    print("worst-case lower bound**: once the true system departs from your nominal model,")
    print("nominal's score can collapse, whereas DRPP's predictive covariance has already")
    print("accounted for that departure. To see the size of that difference in practice, look at")
    print("the adversarial experiment (-e worst) under examples/scene2_simulated_ambiguity/:")
    print("there each predictor faces the worst SDS chosen **specifically against itself**.")
