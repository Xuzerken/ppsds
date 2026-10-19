# Reporting a problem

Open an issue at <https://github.com/Xuzerken/ppsds/issues>. The game is a real user of the
library, so "I played a round and the safe layout paid less than it promised" is a valid report,
not a lesser one than a failing test.

## What to include

- **What you did**, in the smallest form that still shows it: a command, a few lines of code, or
  the round you played and exported.
- **What you expected**, and **what happened** instead. If a number is involved, include it.
- **The versions.** This package has had two failures that were specific to a numpy or cvxpy
  version, so the environment is part of the report:

```bash
python -c "import numpy, scipy, cvxpy, gymnasium, sys; print(sys.version); print(numpy.__version__, scipy.__version__, cvxpy.__version__, gymnasium.__version__)"
```

- **The output**, including warnings. A warning about a clamped trajectory index or a cache
  fingerprint is often the whole explanation.

## If it is a numerical disagreement

Say what you compared against, and include the measured magnitude. "The score moved by 1e-9" and
"the score moved by 0.4" are different reports. If you can, say whether the run was the same
commit, the same seed, and the same `-f` setting.

## If it is a security issue

Do not open a public issue. Follow the private advisory link in
[SECURITY.md](https://github.com/Xuzerken/ppsds/blob/main/SECURITY.md).

## Next

- **The rules a fix will be held to**: [Pull requests](pull-requests.md)
- **The vocabulary behind the numbers**: [Preliminaries](../research/preliminaries.md)
- **The map**: [The map](../tutorials/map.md)
