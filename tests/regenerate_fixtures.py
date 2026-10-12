#!/usr/bin/env python3
"""Regenerate ``tests/golden.json`` from the scenario in :mod:`_golden`.

Run this **only** when a numerical change is intended::

    python tests/regenerate_fixtures.py

Then read the diff of ``golden.json`` -- that diff *is* the review. Nothing regenerates the
fixture automatically, and CI deliberately does not check "the fixture is up to date" either:
cross-platform floating-point differences would make such a gate permanently red, and a fixture
that rewrites itself asserts nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _golden  # noqa: E402  (needs the sys.path line above when run as a script)

FIXTURE = Path(__file__).resolve().parent / 'golden.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true',
                        help='do not write; report whether the fixture is current')
    args = parser.parse_args(argv)

    payload = _golden.serialize(_golden.compute())
    text = json.dumps(payload, indent=1, ensure_ascii=False) + '\n'

    if args.check:
        current = FIXTURE.read_text(encoding='utf-8') if FIXTURE.is_file() else None
        if current == text:
            print(f'{FIXTURE.name} is current')
            return 0
        print(f'{FIXTURE.name} differs (run without --check to regenerate)')
        return 1

    FIXTURE.write_text(text, encoding='utf-8')
    print(f'wrote {FIXTURE}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
