"""Draw the paper's figure as a PNG for the documentation site.

Run::

    python examples/understand_predictability/make_figures.py      # about 3 seconds

The experiment is **not** re-implemented here: ``run()`` is imported from the story's own batch
runner, so this figure and the numbers printed beside it come from one code path. Only the
assembly differs -- the runner writes a paper-shaped PDF (`--save-figure`), and this writes a
narrower PNG for a web page.

Deterministic: fixed seed throughout, no interactive backend. Re-running produces the same bytes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from batch_experiment.run import run      # noqa: E402  (run as a script)
from ppsds.visualization.predictability import plot_predictability_panel

EXAMPLE_ROOT = Path(__file__).resolve().parent
#: ``parents[1]``: this file sits at ``examples/<story>/make_figures.py``. Note that a script one
#: directory deeper needs ``parents[2]`` -- count the levels rather than copying the index from a
#: sibling, because a wrong index raises nothing and writes somewhere plausible instead.
DEFAULT_OUT = EXAMPLE_ROOT.parents[1] / 'public' / 'docs' / 'assets' / 'predictability.png'

#: Enough trajectories that the curves are smooth; the paper uses 10^5 and the difference is not
#: visible at web resolution.
N_TRAJ = 20_000
SEED = 9


def main() -> int:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='where to write the figure (default: %(default)s)')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    panels = run(N_TRAJ, SEED)

    width = 5.0
    figure, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), width * 0.78),
                                squeeze=False)
    for ax, panel in zip(axes[0], panels):
        plot_predictability_panel(ax, panel['scores'], eps=panel['eps'],
                                  upper_bound=panel['upper_bound'],
                                  approximation=panel['approximation'])
        ax.set_title(rf'$\epsilon = {panel["eps"]}$', fontsize=11)
    figure.tight_layout()
    figure.savefig(args.out, dpi=110, bbox_inches=None)
    plt.close(figure)

    print(f'wrote {args.out}  ({args.out.stat().st_size / 1024:.0f} KiB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
