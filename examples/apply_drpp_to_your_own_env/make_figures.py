"""Draw the two figures the README uses for this story.

Run::

    python examples/apply_drpp_to_your_own_env/make_figures.py      # about 15 seconds

Writes into ``public/docs/assets/``:

* ``prediction_on_your_system.png`` -- **what you actually get**: a real off-the-shelf gymnasium
  environment, the true path through one plane of its state space, and the predictive
  distribution the package produced at two steps. This is the answer to "run it and show me what
  comes out" rather than a description of it. It deliberately carries **no score comparison**: on
  this system DRPP should *not* win (that is Act 2's lesson), so putting the two scores side by
  side here would turn the output into a horse race it is not meant to be.
* ``when_drpp_pays.png`` -- that comparison, in the one place where it means something: the same
  environment, scored with the nominal model identified from different amounts of data.

Both are **deterministic**: fixed seeds throughout, no interactive backend, no network. Re-running
produces the same bytes, which is what lets ``tests/test_figures_reproduce.py`` compare the
committed files against a fresh generation.

Why Acrobot: it is the *real* system this repository can honestly demonstrate on -- six
dimensional, underactuated, chaotic, and not written for this package. The package's other
demonstrated system is a six-link arm (``ppsds.dynamics.mechanical``). There is **no** aircraft or
process-plant model here, and no figure should imply one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym

from ppsds.dynamics.identification import encode_action
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import EigDrpp, NominalPredictor
from ppsds.scoring_rule.logarithm_score import LogarithmScore

#: The story's own script supplies the identification and the evaluation protocol. Importing them
#: rather than copying them is the point: a figure that re-derives its own numbers is a figure
#: that can quietly disagree with the table printed next to it, and on a chaotic system it will --
#: a single trajectory is not an estimate of anything.
from with_standard_gym_env import (  # noqa: E402  (same directory, run as a script)
    ENV_ID, EVAL_SEEDS, build, evaluate, fixed_policy,
)

EXAMPLE_ROOT = Path(__file__).resolve().parent
#: ``parents[1]``: this file sits at ``examples/<story>/make_figures.py``, so the repository root
#: is **one** level up. Count the levels in the path you are editing rather than copying the index
#: from a sibling script -- the betting story's exporter is one directory deeper and therefore
#: needs ``parents[2]``, and getting this wrong raises nothing at all. It writes to a directory
#: that looks plausible and the page silently keeps serving the old file.
DEFAULT_OUT = EXAMPLE_ROOT.parents[1] / 'public' / 'docs' / 'assets'

#: The state is ``(cos t1, sin t1, cos t2, sin t2, dt1, dt2)``. The last two are the angular
#: velocities: a plane with physical meaning and no wrap-around, which is what makes it a fair
#: place to draw a covariance ellipse.
PLANE = (4, 5)
PLANE_LABELS = (r'$\dot\theta_1$', r'$\dot\theta_2$')

#: Ellipse radius for a 2-D Gaussian at 90%: ``sqrt(chi2.ppf(0.9, 2))``, hard-coded so the figure
#: does not depend on a scipy version's quantile function.
RADIUS_90 = 2.1460

IDENT_EPISODES = 20
#: One of the story's own evaluation seeds, so the trajectory drawn is one the script also scores.
PLOT_SEED = EVAL_SEEDS[0]
T = 80


def rollout(env, fit, diag, ambiguity, seed: int, T: int = T):
    """One trajectory, recording the true states and both predictors' distributions each step."""
    nominal = NominalPredictor(fit.f_bar, diag['mu_bar'], diag['Sigma'])
    robust = EigDrpp(fit.f_bar, diag['mu_bar'], diag['Sigma'], ambiguity.gamma)
    scorer = LogarithmScore()

    obs, _ = env.reset(seed=seed)
    truth, draws = [np.asarray(obs, dtype=float)], []
    nominal_scores, robust_scores = [], []
    for _ in range(T):
        action = fixed_policy(obs)
        u = encode_action(env.action_space, action)
        nominal.update(obs, u)
        robust.update(obs, u)
        draws.append((nominal.predict(), robust.predict()))
        obs, _r, terminated, truncated, _i = env.step(action)
        truth.append(np.asarray(obs, dtype=float))
        nominal_scores.append(scorer.score(draws[-1][0], obs))
        robust_scores.append(scorer.score(draws[-1][1], obs))
        if terminated or truncated:
            break
    return np.array(truth), draws, np.array(nominal_scores), np.array(robust_scores)


def ellipse(cov2x2, mean2):
    """``(width, height, angle_degrees)`` of the 90% ellipse of a 2-D Gaussian."""
    values, vectors = np.linalg.eigh(cov2x2)
    values = np.clip(values, 0.0, None)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    return 2 * RADIUS_90 * np.sqrt(values), (float(mean2[0]), float(mean2[1])), angle


def figure_prediction(out_dir: Path, env) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    fit, diag, ambiguity = build(env, IDENT_EPISODES)
    truth, draws, _sn, _sd = rollout(env, fit, diag, ambiguity, PLOT_SEED)

    figure, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(truth[:, PLANE[0]], truth[:, PLANE[1]], color='#1f2937', lw=2.0, zorder=3,
            label='what the system did')

    # **Both predictors at the same step.** Drawing them at different steps would look like a
    # comparison of the two predictors while actually being a comparison of two moments -- the
    # reader would take away the wrong number. Colour encodes *which predictor*, line style
    # encodes *which step*, so the two encodings never collide: every solid ellipse is comparable
    # with every other solid one, and solid-with-dashed is the robustness being visible.
    for step, style in ((12, '-'), (40, '--')):
        if step >= len(draws):
            continue
        for index, colour in ((0, '#2563eb'), (1, '#dc2626')):
            draw = draws[step][index]
            (width, height), (cx, cy), angle = ellipse(
                draw.cov[np.ix_(PLANE, PLANE)], draw.mean[list(PLANE)])
            ax.add_patch(Ellipse((cx, cy), width, height, angle=angle, facecolor='none',
                                 edgecolor=colour, lw=2.0, alpha=0.9, ls=style, zorder=4))

    handles = [plt.Line2D([], [], color='#2563eb', lw=2.0, label='nominal, 90%'),
               plt.Line2D([], [], color='#dc2626', lw=2.0, label='Eig-DRPP, 90%'),
               plt.Line2D([], [], color='#4b5563', lw=1.5, ls='-', label='step 12'),
               plt.Line2D([], [], color='#4b5563', lw=1.5, ls='--', label='step 40')]
    ax.set_xlabel(PLANE_LABELS[0])
    ax.set_ylabel(PLANE_LABELS[1])
    ax.set_title('Acrobot-v1: what the predictor hands you at each step\n'
                 'is a distribution, not a number', fontsize=11)
    ax.legend(handles=handles + [plt.Line2D([], [], color='#1f2937', lw=2.0,
                                            label='what the system did')],
              loc='upper left', fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.25)

    figure.tight_layout()
    path = out_dir / 'prediction_on_your_system.png'
    figure.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(figure)
    return path


def figure_when_drpp_pays(out_dir: Path, env) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    episodes = (1, 2, 4, 20)
    nominal, robust, resid_std = [], [], []
    for count in episodes:
        fit, diag, ambiguity = build(env, count)
        n, d = evaluate(env, fit, diag, ambiguity)      # the script's own protocol: 5 seeds x 80
        nominal.append(n)
        robust.append(d)
        resid_std.append(float(np.sqrt(np.diag(diag['Sigma'])).mean()))

    figure, ax = plt.subplots(figsize=(7.0, 4.6))
    x = np.arange(len(episodes))
    ax.plot(x, nominal, 'o-', color='#2563eb', lw=2.0, label='nominal')
    ax.plot(x, robust, 'o-', color='#dc2626', lw=2.0, label='Eig-DRPP')
    # The estimated residual standard deviation on a second axis: it is what *decides* the
    # outcome, so leaving it out would leave the figure's claim unexplained.
    twin = ax.twinx()
    twin.plot(x, resid_std, 's:', color='#6b7280', lw=1.5, ms=5,
              label=r'estimated $\sigma$ of the residual')
    twin.set_ylabel(r'estimated residual $\sigma$', color='#6b7280')
    twin.tick_params(axis='y', colors='#6b7280')
    twin.set_ylim(0, max(resid_std) * 1.6)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{e} episode' if e == 1 else f'{e} episodes' for e in episodes])
    ax.set_xlabel('how much data the nominal model was identified from')
    ax.set_ylabel('mean log score')
    ax.set_title('The model is bad in every column. What decides the score\n'
                 'is whether the noise you declared was too narrow.', fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc='lower right', fontsize=9)
    ax.grid(alpha=0.25)
    figure.tight_layout()
    path = out_dir / 'when_drpp_pays.png'
    figure.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(figure)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT,
                        help='where to write the figures (default: %(default)s)')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_ID)
    try:
        for path in (figure_prediction(args.out_dir, env),
                     figure_when_drpp_pays(args.out_dir, env)):
            print(f'wrote {path}  ({path.stat().st_size / 1024:.0f} KiB)')
    finally:
        env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
