# Research collaborations

The last column of [Get started](../index.md#get-started) is **only data**: you have rollouts but
no dynamics model, or you work on a system this library does not cover yet. L0–L3 are deliberately
empty there. That column is future research, and a discussion is the right first contribution.

## What is worth discussing

- **A system or a domain the conic-moment family cannot express.** State-dependent or
  multiplicative noise, correlated noise, or a constraint that is not additive are all outside
  the current family. A small, honest description of the gap is more useful than a general offer.
- **A new family of ambiguity sets, and the DRPP it implies.** The library is built for this:
  each family is a new module at the same level in four places. The
  [developer guide](../tutorials/developer-guide.md) is the end-to-end contract.
- **A new kind of probabilistic predictor.** The predictor contract is the
  [`ProbabilisticPredictor`](../api.md) protocol; a predictor that predicts a pmf you supply is
  already the second kind in the package.
- **A scoring rule or a predictability question** that changes what "good" means. Start from the
  [probabilistic predictability paper](../research/probabilistic-predictability.md).

## How to start

Open an issue at <https://github.com/Xuzerken/ppsds/issues> describing the system, the data you
have, and the question you want the prediction to answer. If you already have a derivation or a
prototype, say so; the useful next step is usually a contract, not a full implementation.

The published surface deliberately does not describe unpublished methods. If your work depends on
one, say that you have it and discuss the interface, not its internals.

## Next

- **The extension contract**: [Extending `ppsds`](../tutorials/developer-guide.md)
- **The shared vocabulary**: [Preliminaries](../research/preliminaries.md)
- **Pull requests and the no-number rule**: [Pull requests](pull-requests.md)
