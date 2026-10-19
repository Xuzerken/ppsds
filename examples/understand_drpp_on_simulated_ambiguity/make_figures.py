"""Draw the documentation site's covariance figure for this example.

Run::

    python examples/understand_drpp_on_simulated_ambiguity/make_figures.py      # a few seconds

Writes ``public/docs/assets/drpp_ellipses.png``: the predicted covariance ellipses along one
trajectory, at two steps. It sits on the Concepts page, immediately after the maximin formula,
where it is the picture of where that width comes from.

**What this file used to also draw, and why it does not any more.** It used to write a second
asset, ``drpp_error_bars.png``: the mean score of the three predictors under both semantics, with
the set's two theoretical bounds. The home page wanted the paper's figure there rather than a
figure of our own devising, and the paper's figure is available as a vector PDF, so that panel is
now a rasterisation of the paper itself -- see ``tools/make_paper_figures.py``. Drawing it twice,
once faithfully and once approximately, was the alternative, and the approximate one is what the
review rejected. What is left here is the figure the paper does *not* have in this form: one
state, three predictive distributions, at a scale you can read.

The declaration below is deliberately the one ``quickstart.py`` uses -- the same system, the same
three predictors. Re-deriving it here would let the figure and the numbers printed beside it drift
apart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import Ellipse

from ppsds.ambiguity.additive_drift.conic_moment import LinearAmbiguitySet
from ppsds.environment.base import Env
from ppsds.environment.dynamics_mode import RandomDynamics
from ppsds.probabilistic_predictor.drpp.additive_drift.conic_moment import (
    EigDrpp,
    NoiseDrpp,
    NominalPredictor,
)

EXAMPLE_ROOT = Path(__file__).resolve().parent
#: ``parents[1]``: this file sits at ``examples/<example>/make_figures.py``. A script one
#: directory deeper needs ``parents[2]`` -- count the levels, because a wrong index raises nothing
#: and writes somewhere plausible instead.
DEFAULT_OUT = EXAMPLE_ROOT.parents[1] / 'public' / 'docs' / 'assets'

#: The same declaration ``quickstart.py`` makes. See that file for what each name means.
A_BAR = np.array([[1.0, 0.1], [0.0, 1.0]])
B_BAR = np.identity(2)
MU_BAR = np.zeros(2)
SIGMA_BAR = np.array([[1.0, 0.5], [0.5, 1.5]])
ALPHAS = {'A': 0.3, 'B': 0.3, 'mu': 0.5, 'Sigma': 3}
F_ERR_MAX = 5.0
X0 = np.array([2.0, 1.0])
T = 40
PERIOD = 100
SEED = 3

#: ``sqrt(chi2.ppf(0.9, 2))``, hard-coded so the figure does not track a scipy version.
RADIUS_90 = 2.1460

#: The three predictors, in the order the tables use everywhere else.
PREDICTORS = (('Nominal', NominalPredictor), ('Noise-DRPP', NoiseDrpp), ('Eig-DRPP', EigDrpp))
PREDICTOR_COLOURS = {'Nominal': '#699ECA', 'Noise-DRPP': '#8D73BA', 'Eig-DRPP': '#ED8828'}


def build() -> LinearAmbiguitySet:
    return LinearAmbiguitySet(A_BAR, B_BAR, MU_BAR, SIGMA_BAR, ALPHAS, F_ERR_MAX)


def make_predictor(name: str, abg: LinearAmbiguitySet):
    for label, factory in PREDICTORS:
        if label == name:
            if factory is EigDrpp:
                return factory(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma)
            if factory is NoiseDrpp:
                return factory(abg.f_bar, abg.mu_bar, abg.Sigma_bar, abg.gamma)
            return factory(abg.f_bar, abg.mu_bar, abg.Sigma_bar)
    raise KeyError(name)


def figure_ellipses(out_dir: Path) -> Path:
    """The true trajectory, with each predictor's 90% ellipse at two steps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    abg = build()
    fig, ax = plt.subplots(figsize=(6.6, 5.0))

    # One shared trajectory: the ellipses are only comparable if they are drawn around the same
    # state, so this is a single rollout and every predictor rides it.
    states, example_predictions = run_trajectory_shared(abg, seed=SEED)
    ax.plot(states[:, 0], states[:, 1], color='#1f2937', lw=2.0, zorder=3,
            label='what the system did')

    for step, style in ((4, '-'), (18, '--')):
        if step >= len(example_predictions):
            continue
        for name in ('Nominal', 'Noise-DRPP', 'Eig-DRPP'):
            mean, cov = example_predictions[step][name]
            values, vectors = np.linalg.eigh(cov)
            values = np.clip(values, 0.0, None)
            order = np.argsort(values)[::-1]
            values, vectors = values[order], vectors[:, order]
            angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
            ax.add_patch(Ellipse((mean[0], mean[1]), *(2 * RADIUS_90 * np.sqrt(values)),
                                 angle=angle, facecolor='none', ls=style, lw=1.8, alpha=0.95,
                                 edgecolor=PREDICTOR_COLOURS[name], zorder=4))

    handles = [plt.Line2D([], [], color=PREDICTOR_COLOURS[n], lw=2.0, label=f'{n}, 90%')
               for n, _ in PREDICTORS]
    handles += [plt.Line2D([], [], color='#4b5563', lw=1.5, ls='-', label='step 4'),
                plt.Line2D([], [], color='#4b5563', lw=1.5, ls='--', label='step 18'),
                plt.Line2D([], [], color='#1f2937', lw=2.0, label='what the system did')]
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('The same state, three predictive distributions', fontsize=11)
    ax.legend(handles=handles, loc='best', fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / 'drpp_ellipses.png'
    fig.savefig(path, dpi=140, bbox_inches=None)
    plt.close(fig)
    return path


def run_trajectory_shared(abg, seed: int):
    """One rollout of the true system, plus **all three** predictors' distributions each step."""
    mode = RandomDynamics(abg, PERIOD)
    env = Env(abg, mode, x0=X0, max_steps=T)
    predictors = {name: make_predictor(name, abg) for name, _ in PREDICTORS}

    x, _ = env.reset(seed=seed)
    states = [np.array(x, dtype=float)]
    per_step = []
    for _ in range(T):
        u = np.zeros(2)
        for predictor in predictors.values():
            predictor.update(x, u)
        env.inject_eigvec_fixed_info(predictors['Eig-DRPP'].lamb_pred, u)
        per_step.append({name: (p.x_pred.copy(), p.Sigma_pred.copy())
                         for name, p in predictors.items()})
        x, _r, _t, _tr, _i = env.step(u)
        states.append(np.array(x, dtype=float))
    return np.array(states), per_step


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT,
                        help='where to write the figure (default: %(default)s)')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    path = figure_ellipses(args.out_dir)
    print(f'wrote {path}  ({path.stat().st_size / 1024:.0f} KiB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
