"""Score a hand of bets rigorously, over as many trajectories as you like.

Run::

    python experiments/pmf_betting/run.py                                  # the default bet
    python experiments/pmf_betting/run.py --preset rotation --eps 0.25
    python experiments/pmf_betting/run.py --covering --preset shear --eps 0.5
    python experiments/pmf_betting/run.py --bets my_bets.json --n_traj 10000
    python experiments/pmf_betting/run.py --bets my_run.json               # a whole run
    python experiments/pmf_betting/run.py --list

This is the third step of the funnel: pick a system and a radius (the same ones the interactive
page offers), hand over a pmf -- a file exported from the page, the safe layout the page's button
builds, or the built-in even spread -- and get the statistics the page can only show one
trajectory at a time: the expected score, its spread, and the gap to the predictability upper
bound.

Two kinds of file come out of the page, and they answer different questions
---------------------------------------------------------------------------
**A single hand** (``Copy this round``) is a *rule*: the same atoms mean the same thing on every
round and every trajectory, which is exactly what lets it be scored against thousands of draws.
That is what ``--bets`` with a hand does.

**A whole run** (``Export this run``) is a *transcript*: which hand was played at which step of
which trajectory. Replaying it here reproduces the numbers the player watched -- the same walk,
the same landings, the same scores -- and that is the point: it is the page's own claim, checked
by the library rather than by the page. A run is not a sample of anything, so the statistics are
reported for the distinct hands it used, not for the run as a whole.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from presets import (
    EXPORT_SEED,
    EXPORT_STEPS,
    PRESETS,
    SAFE_WINDOW_SIGMA,
    covering_bet,
    covering_meta,
    default_bets,
    get,
    sample_trajectories,
    support_box,
)
from ppsds.numerics.atom_pmf import AtomPmf
from ppsds.predictability.bounds import predictability_upper_bound
from ppsds.predictability.trajectory import across_trajectories, cumulative_average
from ppsds.scoring_rule.epsilon_logarithm_score import EpsilonLogarithmScore


def read_bets(path):
    """Load a hand exported from the page: ``{preset, eps, atoms, weights}``.

    The preset and radius in the file are **authoritative** -- they are what the hand was played
    against, and re-scoring it against a different system would silently produce a number with no
    relation to what the player saw.
    """
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    missing = {'preset', 'eps', 'atoms', 'weights'} - set(payload)
    if missing:
        raise ValueError(f'{path} is missing {sorted(missing)}; expected '
                         "{'preset', 'eps', 'atoms', 'weights'}")
    atoms = np.asarray(payload['atoms'], dtype=float)
    weights = np.asarray(payload['weights'], dtype=float)
    return payload['preset'], float(payload['eps']), AtomPmf(atoms, weights)


def read_run(path):
    """Load a run exported from the page: ``{preset, eps, trajectory, rounds}``."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    missing = {'preset', 'eps', 'trajectory', 'rounds'} - set(payload)
    if missing:
        raise ValueError(f'{path} is missing {sorted(missing)}; expected '
                         "{'preset', 'eps', 'trajectory', 'rounds'}")
    if not payload['rounds']:
        raise ValueError(f'{path} has no settled rounds, so there is nothing to replay')
    for index, item in enumerate(payload['rounds']):
        absent = {'step', 'atoms', 'weights'} - set(item)
        if absent:
            raise ValueError(f'round {index} of {path} is missing {sorted(absent)}')
    return payload


def load_bets(path):
    """Either kind of file the page writes, told apart by the one key the other never has."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return ('run' if 'rounds' in payload else 'hand'), path


def trajectory_of(preset, index, n_step):
    """The page's trajectory ``index``, reproduced exactly -- **as its first ``n_step`` steps**.

    The page replays ``trajectories.json``, which the exporter generated from
    ``sample_trajectories(preset, 12, EXPORT_STEPS, seed=EXPORT_SEED + preset_index)``: entry ``i``
    of that file is the ``i``-th of ``i + 1`` trajectories drawn from the same root seed. So the
    wake the player watched can be rebuilt here from a seed rather than recorded twice.

    ⚠️ The draw is generated at :data:`~presets.EXPORT_STEPS` and then sliced, and that is not
    belt-and-braces: the noise a trajectory contains **depends on how many steps were asked for in
    one call** (scipy draws a whole axis at a time, so a 30-step generation is not the first 30 of a
    50-step one). Asking for ``n_step`` directly would replay a *different* wake from the one the
    page showed whenever a run is shorter than the export -- the closed loop would break silently.
    """
    seed = EXPORT_SEED + list(PRESETS).index(preset.key)
    states, noise = sample_trajectories(preset, index + 1, EXPORT_STEPS, seed=seed)
    return states[index][:n_step + 1], noise[index][:n_step]


def replay(preset, run):
    """Recompute a recorded run on its own trajectory: what the player saw, from the library.

    Returns the ``(states, noise, rounds)`` where each round carries the payout share and the
    score for the landing that actually happened at that step.
    """
    steps = [int(item['step']) for item in run['rounds']]
    n_step = int(run.get('n_step') or (max(steps) + 1))
    states, noise = trajectory_of(preset, int(run.get('trajectory', 0)), n_step)
    eps = float(run['eps'])
    scorer = EpsilonLogarithmScore(eps)
    rows = []
    for item in run['rounds']:
        step = int(item['step'])
        pmf = AtomPmf(np.asarray(item['atoms'], dtype=float), np.asarray(item['weights'], dtype=float))
        landing = noise[step]
        rows.append({
            'step': step,
            'pmf': pmf,
            'roi': float(pmf.box_integral(landing, eps)),
            'utility': float(scorer.score(pmf, landing)),
        })
    return states, noise, rows


def score(pmf, noise, eps, n_traj, n_step):
    """Per-trajectory running averages and their spread, over ``(n_traj, n_step)``."""
    scores = EpsilonLogarithmScore(eps).score(pmf, noise.reshape(-1, 2)).reshape(n_traj, n_step)
    running = cumulative_average(scores)
    aggregate = across_trajectories(running)
    return running, aggregate


def safe_hand(preset, eps):
    """The layout the page's "cover the table" button builds, and what it achieves."""
    reach = None if support_box(preset) is not None else \
        SAFE_WINDOW_SIGMA * np.sqrt(np.diag(preset.noise().cov))
    atoms, weights = covering_bet(preset, eps, reach=reach)
    return AtomPmf(atoms, weights), covering_meta(preset, eps, reach=reach)


def report_statistics(preset, eps, pmf, n_traj, n_step, seed):
    """The hand over many trajectories: the expected score, its spread, and the gap to the bound.

    A hand that can miss has no expected score -- one round of ``-inf`` makes the average ``-inf``
    for good -- so the honest report changes shape rather than printing NaNs: what is worth
    knowing about a hand that can ruin you is how often it does, and what it pays while it lasts.
    """
    _, noise = sample_trajectories(preset, n_traj, n_step, seed=seed)
    scores = EpsilonLogarithmScore(eps).score(pmf, noise.reshape(-1, 2)).reshape(n_traj, n_step)
    upper = predictability_upper_bound(preset.noise(), eps)
    solvent = np.isfinite(scores).all(axis=1)

    print()
    print(f'predictability upper bound (no bettor can beat this): {upper:+.5f}')
    if not solvent.all():
        report_ruin(scores, solvent, eps, upper)
        return

    running, aggregate = score(pmf, noise, eps, n_traj, n_step)
    print(f'{"step":>6} {"mean":>12} {"2.5%":>12} {"97.5%":>12}')
    print('-' * 46)
    # The running mean is what the page draws live; these are its quantiles at a few horizons,
    # so the convergence the page animates can be read off as numbers too.
    for step in (1, 5, 10, n_step):
        index = min(step, n_step) - 1
        print(f'{index + 1:>6} {aggregate["mean"][index]:>12.5f} {aggregate["low"][index]:>12.5f}'
              f' {aggregate["high"][index]:>12.5f}')

    at_horizon = running[:, -1]
    print()
    print(f'at the horizon:        {at_horizon.mean():+.5f}  '
          f'(gap to the bound: {_gap_text(at_horizon.mean() - upper)})')
    print(f'across trajectories:   sd {at_horizon.std():.5f}, '
          f'2.5% {np.quantile(at_horizon, 0.025):+.5f}, '
          f'97.5% {np.quantile(at_horizon, 0.975):+.5f}')
    print()
    print('The gap is the price of betting the way you did; the bound is a property of the')
    print('system, not of this hand. There is no O(eps) approximation line for a pmf bet --')
    print('an atom law has no density, so the bound is the only reference it has.')


def report_ruin(scores, solvent, eps, upper):
    """A hand that can miss: how often it does, and what it pays until it does."""
    n_traj, n_step = scores.shape
    first_miss = np.where(~np.isfinite(scores).all(axis=1),
                          np.argmin(np.isfinite(scores), axis=1), n_step)
    print(f'This hand can miss: {int((~solvent).sum())} of {n_traj} runs scored −inf, and a run '
          'that does is over -- one round that pays nothing ends it.')
    print()
    print(f'{"step":>6} {"still solvent":>15} {"mean score so far":>18}')
    print('-' * 42)
    for step in (1, 5, 10, n_step):
        index = min(step, n_step) - 1
        alive = first_miss > index
        share = alive.mean() * 100
        if alive.any():
            partial = np.cumsum(np.where(np.isfinite(scores), scores, 0.0), axis=1)[alive, index]
            mean = partial.mean() / (index + 1)
            print(f'{index + 1:>6} {share:>14.1f}% {mean:>+18.5f}')
        else:
            print(f'{index + 1:>6} {share:>14.1f}% {"—":>18}')
    print()
    print(f'The bound for comparison is {upper:+.5f}; a hand that can miss trades that bound for a')
    print('chance at more. "Cover the table" on the page builds the layout that cannot miss.')


def _score_text(value):
    """A score as it should be read: ``-inf`` is a settlement, not a small number."""
    return '−inf' if not np.isfinite(value) else f'{value:+.5f}'


def _gap_text(value):
    """A gap, with the last bits of the arithmetic rounded away: +0.00000, never -0.00000.

    A hand that lands exactly on the bound is the interesting case, and reporting it as a negative
    zero would read as "just short of par" when the point is that it *is* par.
    """
    if not np.isfinite(value):
        return '—'
    return f'{0.0 if abs(value) < 5e-6 else value:+.5f}'


def report_run(preset, eps, run, n_traj, n_step, seed):
    """A recorded run, replayed and then measured: did the page's own numbers survive?

    No stake column. The running product of the payout shares is `exp(rounds x average score)`,
    which is the score written a second way -- and because a payout share is always below one it
    only ever decays, so the number the page used to show at the end of a run was always near zero
    and never told anyone anything. The averages below say what it was trying to say.
    """
    states, noise, rows = replay(preset, run)
    upper = predictability_upper_bound(preset.noise(), eps)

    print()
    print(f'trajectory {run.get("trajectory", 0)}, {len(rows)} settled rounds, '
          f'{len(noise)} steps available')
    print(f'{"round":>5} {"step":>5} {"stacks":>7} {"paid":>8} {"score":>10} {"average":>10}')
    print('-' * 52)
    total = 0.0
    for index, row in enumerate(rows):
        total += row['utility']
        print(f'{index + 1:>5} {row["step"]:>5} {len(row["pmf"].atoms):>7} '
              f'{row["roi"] * 100:>7.1f}% {_score_text(row["utility"]):>10} '
              f'{total / (index + 1):>+10.5f}')

    average = total / len(rows)
    print()
    if np.isfinite(average):
        print(f'the run averaged {average:+.5f} a round '
              f'(gap to the bound: {_gap_text(average - upper)})')
        print(f'a round gave back {np.mean([row["roi"] for row in rows]) * 100:.1f}% of the bet on '
              f'average; the score compounds, that average does not')
    else:
        print('the run averaged −inf a round: a round paid nothing, and a round that pays nothing '
              'ends the run')

    # A run is a transcript, not a sample: the same hand may have been played every round (the
    # safe layout is), so what can be measured over many trajectories is the distinct hands.
    distinct = []
    seen = {}
    for row in rows:
        key = (tuple(np.round(row['pmf'].atoms.ravel(), 9)), tuple(np.round(row['pmf'].weights, 9)))
        if key in seen:
            distinct[seen[key]] = (row, distinct[seen[key]][1] + 1)
        else:
            seen[key] = len(distinct)
            distinct.append((row, 1))

    _, hand_noise = sample_trajectories(preset, n_traj, n_step, seed=seed)
    flat = hand_noise.reshape(-1, 2)
    print()
    print(f'{"hand":>12} {"rounds":>7} {"expected score":>16} {"gap to bound":>14}')
    print('-' * 52)
    for row, count in distinct:
        mean = EpsilonLogarithmScore(eps).score(row['pmf'], flat).mean()
        print(f'{len(row["pmf"].atoms):>9} stacks {count:>7} {_score_text(mean):>16} '
              f'{_gap_text(mean - upper):>14}')
    if len(distinct) == 1:
        print()
        print('One hand, played every round -- so the statistics above are the whole run\'s.')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--preset', default='shear', help='which system (default: %(default)s)')
    parser.add_argument('--eps', type=float, default=None,
                        help='neighbourhood radius (default: the preset\'s middle one)')
    parser.add_argument('--bets', type=Path, default=None,
                        help='a file exported from the game, a hand or a whole run; overrides '
                             '--preset/--eps')
    parser.add_argument('--covering', action='store_true',
                        help='score the safe layout (the page\'s "cover the table" button)')
    parser.add_argument('--n_traj', type=int, default=2000, help='trajectories (default: %(default)s)')
    parser.add_argument('--n_step', type=int, default=30, help='steps per trajectory (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=9, help='noise seed (default: %(default)s)')
    parser.add_argument('--list', action='store_true', help='list the presets and exit')
    args = parser.parse_args()

    if args.list:
        print(f'{"key":>10}  {"eps offered":<24} title')
        print('-' * 60)
        for preset in PRESETS.values():
            print(f'{preset.key:>10}  {str(preset.epsilons):<24} {preset.title}')
        return

    kind = load_bets(args.bets)[0] if args.bets is not None else 'hand'

    if kind == 'run':
        run = read_run(args.bets)
        preset = get(run['preset'])
        eps = float(run['eps'])
        print(f'run from {args.bets}: preset {preset.key!r} ({preset.title}), eps={eps}')
        print(f'  {"trajectory":<10} {run.get("trajectory", 0)}')
        report_run(preset, eps, run, args.n_traj, args.n_step, args.seed)
        return

    if args.bets is not None:
        preset_key, eps, pmf = read_bets(args.bets)
        preset = get(preset_key)
        print(f'bets from {args.bets}: preset {preset_key!r}, eps={eps}, {pmf}')
    else:
        preset = get(args.preset)
        eps = args.eps if args.eps is not None else preset.epsilons[len(preset.epsilons) // 2]
        if args.covering:
            pmf, meta = safe_hand(preset, eps)
            print(f'preset {preset.key!r} ({preset.title}), eps={eps}, safe layout: '
                  f'{meta["n_atoms"]} stacks')
            if meta['guaranteed']:
                print('  every landing pays: worst case '
                      f'{meta["worst_case_roi"] * 100:.1f}% of the bet')
            else:
                print('  no guarantee is possible here (the noise is unbounded); this layout '
                      f'covers +-{SAFE_WINDOW_SIGMA:g} sigma')
        else:
            atoms, weights = default_bets(preset, eps)
            pmf = AtomPmf(atoms, weights)
            print(f'preset {preset.key!r} ({preset.title}), eps={eps}, built-in even spread '
                  f'({pmf})')

    for label, value in (('preset', preset.key), ('eps', eps), ('n_traj', args.n_traj),
                         ('n_step', args.n_step), ('seed', args.seed)):
        print(f'  {label:<8} {value}')

    report_statistics(preset, eps, pmf, args.n_traj, args.n_step, args.seed)


if __name__ == '__main__':
    main()
