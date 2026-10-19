#!/usr/bin/env python3
"""A/B numerical-invariance comparison: this project's acceptance procedure for numerical changes, scripted.

A refactor must leave the simulated numbers **bit-identical**. This script checks out the
"before" commit into a git worktree, runs the same small set of configurations in both trees,
and compares ``data/gym/**/*.pkl`` element by element.

Usage::

    python tools/ab_compare.py                 # default base 8c9aea4 (before the 2nd refactor round)
    python tools/ab_compare.py --base <ref>
    python tools/ab_compare.py -v              # also print the result for every file
    python tools/ab_compare.py --keep          # keep the worktree (removed by default)

Exit code: 0 = everything bit-identical; 1 = differences or missing files.

Two key design choices
----------------------
1. **``--force`` is mandatory.** The cache would skip configurations already computed, so both
   sides would "produce no new files" and the comparison would spin on nothing. Every
   configuration therefore carries ``-f`` to force a recomputation.
2. **The criterion is the pkl files, not the exit code.** The plotting stage is allowed to be
   imperfect (it clamps out-of-range trajectory indices with a ``RuntimeWarning``), so a
   non-zero exit is tolerated: the script only requires that the expected pkl files appear and
   are bit-identical.

Why ``PYTHONPATH`` can select between two trees inside one environment
---------------------------------------------------------------------
An editable install is just a ``.pth`` that adds ``src/`` to ``sys.path``, and ``PYTHONPATH``
entries are ordered **before** site-packages, so which tree is in effect is decided by
``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The scene-2 directory, **newest layout first**. The batch runner moved out of `examples/`
#: into `experiments/drpp/` on 2026-10-18, and `--base` checks out a tree that predates the move,
#: so the directory is *probed* in each tree rather than assumed, for the same reason
#: ``RUNNER_SCRIPTS`` is. Getting this wrong does not fail loudly: the baseline tree would
#: simply report no pkl, and that surfaces as "file sets differ" rather than as a comparison of
#: numbers.
SCENE2_CANDIDATES = (
    os.path.join('experiments', 'drpp'),
    os.path.join('examples', 'understand_drpp_on_simulated_ambiguity'),
    os.path.join('examples', 'scene2_simulated_ambiguity'),
)

#: ``--base`` is **required**, deliberately. It used to default to the commit this harness was
#: first written against (``8c9aea4``, the baseline of the second structural review) -- a
#: convenient default for exactly one person, and for anyone else either a stale comparison or
#: an error about a commit that does not exist in their clone. A baseline is a judgement about
#: what a change is allowed to affect, so it should be stated at the call site.
#: The usage in ``CONTRIBUTING.md`` is ``--base <the commit you branched from>``.

#: Where the batch runner may live **inside the scene directory**, newest layout first. The
#: runner moved from ``exp_gym.py`` to ``batch_experiment/run.py``, and then to ``run.py``
#: directly in ``experiments/drpp/``; a baseline ref may predate either move, so each tree is
#: probed rather than assumed.
RUNNER_SCRIPTS = (
    ('', 'run.py'),
    ('batch_experiment', 'run.py'),
    ('', 'exp_gym.py'),
)

#: A/B configurations: all ``-t 5`` (small), all ``-c none`` (safe for every model).
#: The third one uses ``-p 1``, whose figure uses a large trajectory index -- that is a
#: deliberate coverage case for the plotting clamps (a ``RuntimeWarning``, not a crash).
CONFIGS = [
    ['-n', '1', '-s', '3', '-p', '100', '-m', 'final_round', '-c', 'none', '-t', '5'],
    ['-n', '1', '-s', '3', '-m', 'NLink', '-c', 'none', '-e', 'worst', '-t', '5'],
    ['-n', '1', '-s', '3', '-p', '1', '-m', 'final_round', '-c', 'none', '-t', '5'],
]


# ----------------------------------------------------------------------
# Comparing arrays
# ----------------------------------------------------------------------

def compare(a, b, path=''):
    """Recursively compare two structures (as unpickled).

    Returns
    -------
    list of (path, max abs diff or None, explanation)
        ``max abs diff`` of ``0.0`` means bit-identical; ``None`` means the structure or
        type differs.
    """
    import numpy as np

    diffs = []
    here = path or '<root>'

    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            sub = f'{path}.{k}' if path else str(k)
            if k not in a:
                diffs.append((sub, None, 'only present on side B'))
            elif k not in b:
                diffs.append((sub, None, 'only present on side A'))
            else:
                diffs += compare(a[k], b[k], sub)
        return diffs

    if a is None or b is None:
        if a is not b:
            diffs.append((here, None, f'None mismatch ({a!r} vs {b!r})'))
        return diffs

    try:
        x = np.asarray(a, dtype=float)
        y = np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        if not (a == b if not isinstance(a, type) else a is b):
            diffs.append((here, None, f'non-numeric and unequal ({a!r} vs {b!r})'))
        return diffs

    if x.shape != y.shape:
        diffs.append((here, None, f'shape differs ({x.shape} vs {y.shape})'))
        return diffs

    if not np.array_equal(x, y, equal_nan=True):
        with np.errstate(invalid='ignore'):
            d = np.nanmax(np.abs(x - y)) if x.size else 0.0
        diffs.append((here, float(d), 'values differ'))
    return diffs


# ----------------------------------------------------------------------
# Running one side
# ----------------------------------------------------------------------

def scene2_dir(root):
    """The scene-2 directory inside the tree at ``root``, newest name first."""
    for rel in SCENE2_CANDIDATES:
        if os.path.isdir(os.path.join(root, rel)):
            return rel
    tried = ', '.join(SCENE2_CANDIDATES)
    raise SystemExit(f'no scene-2 directory found under {root}; tried: {tried}')


def find_runner(root):
    """Locate the batch runner inside ``root``, tolerating the rename and the layout change."""
    scene = scene2_dir(root)
    for rel_dir, script in RUNNER_SCRIPTS:
        if os.path.isfile(os.path.join(root, scene, rel_dir, script)):
            return os.path.join(scene, rel_dir), script
    tried = ', '.join(os.path.join(scene, d, s) for d, s in RUNNER_SCRIPTS)
    raise SystemExit(f'no batch runner found under {root}; tried: {tried}')


def pkl_mtimes(root):
    """``{relative path: mtime_ns}`` for the pkl files under ``<root>/<scene-2>/data``.

    **mtime**, rather than the set of files, decides what "this run produced": the working
    tree may already hold a ``data/`` from an earlier run, and ``-f`` **overwrites** those
    files rather than creating new ones, so a set difference would reveal nothing. An mtime
    change treats "created" and "overwritten" alike.
    """
    base = os.path.join(root, scene2_dir(root), 'data')
    out = {}
    for dirpath, _dirnames, filenames in os.walk(base):
        for fn in filenames:
            if fn.endswith('.pkl'):
                p = os.path.join(dirpath, fn)
                out[os.path.relpath(p, base)] = os.stat(p).st_mtime_ns
    return out


def run_config(root, args, python, label):
    """Run one configuration in the tree at ``root``; return the pkl paths written/overwritten."""
    runner_dir, script = find_runner(root)
    before = pkl_mtimes(root)
    cmd = [python, script, *args, '-f']
    env = dict(os.environ, PYTHONPATH=os.path.join(root, 'src'))
    proc = subprocess.run(cmd, cwd=os.path.join(root, runner_dir), env=env,
                          capture_output=True, text=True)
    after = pkl_mtimes(root)
    touched = {rel for rel, m in after.items() if before.get(rel) != m}
    tag = ' '.join(args)
    print(f'  [{label}] {tag}\n'
          f'      exit code {proc.returncode}, pkl written/overwritten: {len(touched)}')
    if not touched:
        tail = (proc.stderr or proc.stdout or '').strip().splitlines()[-5:]
        print('      WARNING: no pkl produced! tail of the output:')
        for line in tail:
            print('        ' + line)
    return touched, proc.returncode


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='A/B numerical-invariance comparison')
    ap.add_argument('--base', required=True,
                    help='baseline commit/ref to compare against -- normally the commit you '
                         'branched from, since a baseline is a statement about what this change '
                         'is allowed to affect')
    ap.add_argument('--worktree', default=None,
                    help='directory to check the baseline out into (default /tmp/ab-base-<ref>)')
    ap.add_argument('--python', default=sys.executable,
                    help='interpreter used to run the simulation (default: the current one)')
    ap.add_argument('-v', '--verbose', action='store_true',
                    help='print the comparison result for every file')
    ap.add_argument('--keep', action='store_true',
                    help='keep the worktree (removed when done by default)')
    args = ap.parse_args()

    wt = args.worktree or f'/tmp/ab-base-{args.base[:10]}'
    created = False
    if not os.path.isdir(wt):
        print(f'Checking out baseline {args.base} into {wt} ...')
        subprocess.run(['git', 'worktree', 'add', '--detach', wt, args.base],
                       cwd=REPO_ROOT, check=True)
        created = True
    else:
        print(f'Reusing the existing worktree: {wt}')

    try:
        print(f'\nInterpreter: {args.python}\n')
        print(f'=== Running {len(CONFIGS)} configurations (once per side, with -f to force recompute) ===')
        all_diffs = []
        for cfg in CONFIGS:
            tag = ' '.join(cfg)
            print(f'\n>>> {tag}')
            a_new, _ = run_config(wt, cfg, args.python, 'base')
            b_new, _ = run_config(REPO_ROOT, cfg, args.python, 'cur')

            only_a = a_new - b_new
            only_b = b_new - a_new
            common = sorted(a_new & b_new)
            base_data = os.path.join(wt, scene2_dir(wt), 'data')
            cur_data = os.path.join(REPO_ROOT, scene2_dir(REPO_ROOT), 'data')

            if only_a or only_b:
                all_diffs.append((tag, 'file sets differ',
                                  f'only base has {sorted(only_a)}; only cur has {sorted(only_b)}'))
            if not common:
                all_diffs.append((tag, 'no pkl to compare', ''))
                continue

            for rel in common:
                with open(os.path.join(base_data, rel), 'rb') as f:
                    da = pickle.load(f)
                with open(os.path.join(cur_data, rel), 'rb') as f:
                    db = pickle.load(f)
                diffs = compare(da, db)
                if diffs:
                    for sub, d, why in diffs:
                        all_diffs.append((tag, f'{rel}::{sub}',
                                          why if d is None else f'max abs diff = {d!r}'))
                elif args.verbose:
                    print(f'      OK  {rel}')

        print('\n' + '=' * 70)
        if all_diffs:
            print(f'FAIL: {len(all_diffs)} inconsistencies (the acceptance bar is bit-identical):')
            for tag, what, why in all_diffs:
                print(f'   [{tag}] {what}\n       {why}')
            return 1
        print('PASS: everything is bit-identical (max abs diff == 0.0, equal_nan=True)')
        return 0
    finally:
        if created and not args.keep:
            subprocess.run(['git', 'worktree', 'remove', '--force', wt],
                           cwd=REPO_ROOT, check=False)
            print(f'\n(Removed the worktree {wt}; pass --keep to retain it.)')


if __name__ == '__main__':
    sys.exit(main())
