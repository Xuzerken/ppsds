# Security policy

`ppsds` is a numerical library. It has no network service, no authentication, no database and
no privileged operations — nothing that listens on a port or parses a request from a stranger.
The honest version of this policy is therefore short: there is no security process here to
speak of, and inventing one would only obscure that.

## Supported versions

| version | supported |
|---|---|
| `0.1.x` | yes |

Every version before `0.1.0` was published only to collaborators and never to PyPI, so no
supported artifact predates the `0.1.x` line. Fixes go there; the project is pre-1.0 and does
not maintain parallel release branches.

## What the attack surface actually is

`ppsds` consumes arrays you supply and, for one of the three predictors, hands them to a cone
program. That is the whole of it:

- **Malformed input.** `sds_params` is validated on the way into the environment layer and an
  unknown key raises and names the accepted keys. Shapes are a different story: an array whose
  dimension is not the one expected may fail deep inside `numpy`, `scipy` or `cvxpy` rather than
  at the boundary, so you get an exception from somewhere unexpected — or, conceivably, a number
  you did not intend. The wrong number is the more interesting outcome of the two for this
  package.
- **Resource exhaustion.** Every `EigDrpp` step solves a cone program through CVXPY/SCS. A large
  state dimension, or a long rollout, is real work — real work on the calling process's own
  machine, which is the caller's cost to bear, not a third party's.
- **Instance construction.** `LinearAmbiguitySet` and `NonlinearAmbiguitySet` take radii and
  bounds. A value that makes a set inconsistent is a modelling error and is not guaranteed to be
  rejected up front.

There is no remote-compromise story that goes with any of these. If you run `ppsds` on data you
did not produce, the threat that matters is the ordinary one — the input arrays — and the
defence is the ordinary one too: validate them at your own boundary before they reach the
predictor.

## Reporting a vulnerability

For anything sensitive, use GitHub's private security advisory form:

**https://github.com/Xuzerken/ppsds/security/advisories/new**

That channel is visible only to you and the maintainer until an advisory is published, which is
what makes it the right one for a report you would not want to file in public.

For everything else — a crash, a wrong number, a shape that was accepted and should not have
been — open an issue instead: **https://github.com/Xuzerken/ppsds/issues**. The two failure
modes this package has hit were both specific to a numpy or cvxpy version, so state your
versions; the issue form asks for them.

There is no bug bounty and no response-time commitment. This is research software maintained
alongside the research it implements. Reports are read, and a reproducible one is usually quick to
assess — it either moves a number or it does not.
