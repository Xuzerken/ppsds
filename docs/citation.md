# Citation

## The software

If you use `ppsds`, cite the software. GitHub's **Cite this repository** button reads
[`CITATION.cff`](https://github.com/Xuzerken/ppsds/blob/main/CITATION.cff) and produces this:

```bibtex
@software{ppsds,
  title   = {ppsds: probabilistic prediction for stochastic dynamical systems},
  author  = {Xu, Tao and He, Jianping},
  license = {MIT},
  url     = {https://github.com/Xuzerken/ppsds}
}
```

An archived release with a DOI will be added here once the first version is tagged, so that a
citation can point at exactly the code that produced a result.

## References

`ppsds` is not the artifact of a single paper. It is a library for probabilistic prediction that
implements predictors from research, and it is intended to carry predictors from more than one
paper. Each predictor's implementation records which reference it follows.

### Distributionally robust prediction over conic-moment ambiguity sets

The DRPP implemented in this release: the predictors `EigDrpp` and `NoiseDrpp`, the
conic-moment ambiguity sets, and the worst-case bounds in `ppsds.ambiguity`.

```bibtex
@article{xuDRPP,
  title   = {Distributionally Robust Probabilistic Prediction for Stochastic Dynamical Systems},
  author  = {Xu, Tao and He, Jianping},
  journal = {IEEE Transactions on Automatic Control}
}
```

See [Concepts](concepts.md) for what this family optimizes and what its guarantee costs.

## Where to go next

- **Reproduce the second paper's figure**: the predictability story in
  [Examples](examples.md) — six golden values you can check by hand, then the figure
- **Cite the software rather than the paper**: the BibTeX entry above is what `CITATION.cff`
  produces from GitHub's *Cite this repository* button
- **What is implemented, and what is not**: [Concepts](concepts.md) § *Honest limits*
- **Extend it into a paper of your own**: [the Developer Guide](extending.md)
