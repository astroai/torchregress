# SAGE-Reg Representation Sensitivity

## Claim

SAGE-Reg should be presented as a **representation-aware** semi-supervised principle, not as a
claim that every predictive head benefits equally from the same agreement signal.

The method trusts an unlabeled sample when its predictive **law** is stable across perturbations.
That immediately makes the predictive representation part of the method:

- Gaussian heads produce a smooth law with a low-dimensional, well-behaved perturbation response.
- Quantile heads produce a piecewise density reconstructed from sparse quantile levels.
- Bar / binned PDF heads produce a coarse categorical density over fixed bins.

All three can be mapped into a common law, but they do not expose the same agreement geometry.

## Why This Matters

If SAGE-Reg shows stronger gains for Gaussian heads than for quantile or bar heads, that does not
invalidate the method. It supports a narrower and more precise statement:

- SAGE-Reg is useful when perturbation-stable predictive laws can be measured reliably.
- Some predictive families make that stability easier to estimate than others.
- Head choice changes how disagreement behaves and therefore changes the effective trust signal.

This is a substantive difference from generic uncertainty-guided pseudo-labeling. Scalar confidence
methods mostly ask whether a prediction is sharp or low-variance. SAGE-Reg instead asks whether the
**full predictive law** is stable under stochastic views, and that question depends on the chosen
predictive representation.

## Reading The Backbone Benchmark

`examples/benchmarks/self_agreement_backbone_comparison.py` should therefore be read as a
**representation-sensitivity benchmark**.

The benchmark now reports two diagnostics for each backbone/fraction pair:

- `MeanWeight`: the final average sample trust weight applied to the unlabeled objective.
- `MeanDisagreement`: the final average pairwise predictive disagreement before weighting.

These diagnostics help distinguish three regimes:

1. `MeanDisagreement` is low and performance improves.
   This is the intended SAGE-Reg regime: the head exposes stable predictive laws and the resulting
   trust signal is useful.
2. `MeanDisagreement` is low but performance does not improve.
   The head may be too coarse or too smooth, so agreement is not selective enough to filter harmful
   unlabeled samples.
3. `MeanDisagreement` is high and performance degrades.
   The head may be exposing unstable or hard-to-compare predictive laws under perturbation, making
   the agreement signal noisy.

## Current Prototype Interpretation

In the current prototype, Gaussian is the most compatible backbone. Quantile and bar heads are not
failures of the core idea; they show that the current agreement functional and density conversion
are more naturally aligned with some predictive families than others.

That is a publishable result if stated carefully:

- SAGE-Reg is not only a semi-supervised objective.
- SAGE-Reg is also a question about which predictive representations make self-agreement informative.

## Paper Framing

Recommended framing for the paper and docs:

- Do not claim universal gains across all regression heads.
- Claim that SAGE-Reg is a head-agnostic principle in implementation, but not necessarily
  head-invariant in empirical effect.
- Treat representation sensitivity as a first-class experiment, not as a post-hoc caveat.
