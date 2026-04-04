# SIMEX, SIMFEX, and Input-Noise EIV

## Summary

`torchregress` should treat `SIMEX`, `SIMFEX`, and input-noise marginalization as related but distinct tools.

- `SIMEX` is the generic simulation-and-extrapolation baseline for continuous measurement error.
- `SIMFEX` is a specialized simulation-free pseudo-SIMEX method built around transition / misclassification structure.
- `InputNoiseMarginalizationLoss` is the most natural fit for neural tabular regression in `torchregress` today.

## SIMEX

Classical `SIMEX`:

1. Adds synthetic noise at several `lambda` levels.
2. Refits the model for each `lambda`.
3. Extrapolates parameter or prediction behavior back to `lambda = -1`.

Implications for `torchregress`:

- It is broadly applicable to continuous-input tabular models.
- It is expensive because it requires repeated refits.
- It should use multiple Monte Carlo replicates per `lambda`; a single replicate is too noisy.
- For neural nets, prediction-level SIMEX is easier to wire than exact parameter-level SIMEX, but parameter-level summaries may still be useful in linear or generalized-linear settings.

## SIMFEX

The current `SIMFEX` paper/repo is not a generic drop-in replacement for `SIMEX`.

What it does:

- works in a misspecified logistic setting
- discretizes the noisy covariate into categories
- estimates a misclassification / transition matrix
- applies matrix-power extrapolation instead of simulating new noisy datasets

Practical interpretation:

- `SIMFEX` is closer to a matrix-based pseudo-SIMEX for categorized inputs
- it is most relevant for discrete, ordinal, or binned predictors
- it is not the right first implementation target for continuous multifeature photo-z regression

## Recommendation For Torchregress

Near-term:

1. Keep improving `SIMEX`.
2. Keep `InputNoiseMarginalizationLoss` and its MDN / binned-PDF variants as the main neural EIV path.
3. Do not add the current paper-specific `SIMFEX` implementation directly as a headline generic method.

Later, if needed:

- add a generalized `PseudoSIMEX` / `MatrixSIMEX` abstraction
- target ordinal or binned models first
- let users provide either a known transition matrix or an estimator for it

## Photo-z Guidance

For `torchz`-style tabular photo-z:

- prefer `InputNoiseGaussianCRPS`
- prefer `InputNoiseMDN`
- prefer `InputNoiseBinnedPDF`

These are much closer to the real problem than the current paper’s categorical `SIMFEX` setup.

If a `SIMFEX`-like idea is pursued for photo-z, it should likely appear as:

- bin-transition correction for `binned_pdf`
- ordinal transition correction for cumulative-link models
- post-hoc pseudo-SIMEX over predicted redshift-bin distributions rather than raw continuous fluxes
