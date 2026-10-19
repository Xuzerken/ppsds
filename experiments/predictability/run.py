"""The paper's figure at full scale.

Run::

    python experiments/predictability/run.py                     # 100,000 trajectories
    python experiments/predictability/run.py --n_traj 2000       # quick look
    python experiments/predictability/run.py --save-figure

This is the heavy companion to ``quickstart.py``: same system, same predictor, same reference
lines, but at the resolution the paper uses (10^5 trajectories of length 30) and writing the
three-panel figure. ``quickstart.py`` is written flat and standalone for teaching; this file
carries the command line, the figure and the full-scale default.

Why this stays cheap
--------------------
The predictor predicts the *noise*, so the realised outcome it is scored on is the noise draw
itself -- the drift cancels:

    L_eps(p_hat_w(.- f(x_k)), x_{k+1})  ==  L_eps(p_hat_w, x_{k+1} - f(x_k))  ==  L_eps(p_hat_w, w_k)

So the whole experiment is one vectorised pass over ``(n_traj * T, d_x)`` noise draws, and the
predictor is a diagonal Gaussian, which the scoring rule has a closed form for. 10^5 x 30 x 3
scores take seconds, not the hours a per-step cone program would.

The system, the predictor and the radii are declared here and nowhere else in this file's
directory; ``quickstart.py`` declares its own copy because a teaching example has to be
readable top to bottom.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

from ppsds.numerics.random_variables import multivariate_uniform
from ppsds.predictability.bounds import approximation_value, predictability_upper_bound
from ppsds.predictability.entropy import cross_entropy, differential_entropy
from ppsds.predictability.trajectory import across_trajectories, cumulative_average
from ppsds.scoring_rule.epsilon_logarithm_score import EpsilonLogarithmScore

#: The paper's shear: ``x_{k+1} = A x_k + w_k``.
A_TRUE = np.array([[1.0, 1.0],
                   [0.0, 1.0]])

#: The paper's process noise: i.i.d. uniform on ``[-1, 1] ** 2``.
NOISE_LOW, NOISE_HIGH = -np.ones(2), np.ones(2)

#: The neighbourhood radii the paper sweeps.
EPSILONS = (0.25, 0.5, 0.75)

#: Trajectory length and start, as in the paper.
T = 30
X0 = np.zeros(2)

#: Output lands in this experiment's directory, resolved from this file rather than the working
#: directory (the same convention the DRPP experiment uses).
EXPERIMENT_ROOT = Path(__file__).resolve().parent

#: The published figure's panel size, in inches, as taken from `papers/.../pic/*.pdf`.
PANEL_WIDTH, PANEL_HEIGHT = 6.4, 4.8


def paper_noise():
    return multivariate_uniform(low=NOISE_LOW, high=NOISE_HIGH)


def paper_predictor():
    """The paper's deliberately misspecified predictor: a standard Gaussian on the noise."""
    return multivariate_normal(mean=np.zeros(2), cov=np.eye(2))


def run(n_traj, seed, epsilons=EPSILONS):
    """Score every trajectory at every radius; returns one panel's worth of data per radius."""
    if n_traj < 1:
        raise ValueError(f'--n_traj must be >= 1, got {n_traj}')
    q, p = paper_noise(), paper_predictor()
    noise = q.rvs(size=n_traj * T, rng=np.random.default_rng(seed))

    panels = []
    for eps in epsilons:
        scores = EpsilonLogarithmScore(eps).score(p, noise).reshape(n_traj, T)
        panels.append({
            'eps': eps,
            'scores': scores,
            'upper_bound': predictability_upper_bound(q, eps),
            'approximation': approximation_value(q, p, eps),
            'measured': scores.mean(),
        })
    return panels


def report(panels, marginals, n_traj):
    print(f'{"eps":>6} {"predictability":>15} {"approximation":>15} {"measured":>12}'
          f' {"|err|":>10} {"|err|/eps":>10}')
    print('-' * 84)
    for panel in panels:
        error = abs(panel['measured'] - panel['approximation'])
        print(f'{panel["eps"]:>6} {panel["upper_bound"]:>15.5f} {panel["approximation"]:>15.5f}'
              f' {panel["measured"]:>12.5f} {error:>10.5f} {error / panel["eps"]:>10.5f}')

    print()
    print(f'At the horizon (step {panels[0]["scores"].shape[1]}), running averages over '
          f'{n_traj} trajectories:')
    print(f'{"eps":>6} {"mean":>12} {"2.5%":>12} {"97.5%":>12}')
    print('-' * 46)
    for panel in panels:
        aggregate = across_trajectories(cumulative_average(panel['scores']))
        print(f'{panel["eps"]:>6} {aggregate["mean"][-1]:>12.5f} {aggregate["low"][-1]:>12.5f}'
              f' {aggregate["high"][-1]:>12.5f}')

    print()
    print(f'  h_d(noise)              = {marginals["entropy"]:+.5f}')
    print(f'  H(noise || predictor)   = {marginals["cross_entropy"]:+.5f}')
    print('  The predictability line is an upper bound on *any* predictor; the approximation')
    print('  line is where the O(eps) result puts *this* one. The measured column tracks it.')


def visualize(panels, out_path):
    """The paper's three-panel figure. Matplotlib is an optional extra."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        from ppsds.visualization.predictability import plot_predictability_panel
    except ImportError:
        print()
        print('(matplotlib is not installed -- skipping the figure; '
              '`pip install ppsds[experiments]`)')
        return None

    # One panel per radius, each at the published figure's own size: the `pic/*.pdf` files are
    # 6.33 x 4.53 in, i.e. matplotlib's 6.4 x 4.8 default trimmed by a tight bbox. Panel aspect
    # decides the font-to-panel ratio, so matching it is what makes the reproduction *look* like
    # the original rather than merely contain the same curves.
    figure, axes = plt.subplots(1, len(panels), figsize=(PANEL_WIDTH * len(panels), PANEL_HEIGHT),
                                squeeze=False)
    for ax, panel in zip(axes[0], panels):
        # The panel draws its own legend, with the published figure's four entries. Each panel
        # picks its own three trajectories: one shared sample across panels would be a property
        # of the sample, not of the radius.
        plot_predictability_panel(ax, panel['scores'], eps=panel['eps'],
                                  upper_bound=panel['upper_bound'],
                                  approximation=panel['approximation'])
        # The paper's subcaptions. The standalone panels in papers/.../pic/ carry no title --
        # LaTeX adds these -- so the caption belongs here, at the assembly site, and not inside
        # the plotting helper.
        ax.set_title(rf'$\epsilon = {panel["eps"]}$', fontsize=11)
    figure.tight_layout()
    figure.savefig(out_path, dpi=140, bbox_inches='tight')
    print()
    print(f'(figure written to {out_path})')
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--n_traj', type=int, default=100_000,
                        help='number of trajectories (default: 100000, as in the paper)')
    parser.add_argument('--seed', type=int, default=9, help='noise seed (default: 9)')
    parser.add_argument('--save-figure', action='store_true',
                        help='write the three-panel figure')
    parser.add_argument('--out', type=Path, default=EXPERIMENT_ROOT / 'predictability.pdf',
                        help='where to write the figure (default: %(default)s)')
    args = parser.parse_args()

    q, p = paper_noise(), paper_predictor()
    panels = run(args.n_traj, args.seed)
    report(panels,
           {'entropy': differential_entropy(q), 'cross_entropy': cross_entropy(q, p)},
           args.n_traj)
    if args.save_figure:
        visualize(panels, args.out)


if __name__ == '__main__':
    main()
