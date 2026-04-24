# Conformal API

Conformal predictors and wrappers are implemented in `torchregress.losses.conformal`.

See the [Losses API](losses.md#torchregress.losses.conformal) for generated signatures and source links.
# API Reference — Conformal Prediction

Conformal prediction classes are exported from `torchregress.losses` and documented in the [Losses API](losses.md).

For usage guides and mathematical background, see:

- [Conformal Prediction Overview](../methods/conformal/index.md)
- [Predictors Reference](../methods/conformal/predictors.md)
- [Distributional Conformal](../methods/conformal/distributional.md)

## Quick Reference

| Class | Purpose |
|:------|:--------|
| `ConformalPredictor` | Base class for all conformal methods |
| `SplitConformal` | Residual-based conformal |
| `CQR` | Conformalized Quantile Regression |
| `DensityConformal` | Density-weighted residuals |
| `PrevalenceAdjustedCP` | Group-prevalence-adjusted |
| `MonteCarloConformal` | MC-sample normalised |
| `MultiTargetConformal` | Multi-output |
| `DistributionalConformal` | PIT-based distributional |
| `CTI` | Density level-set (smallest intervals) |
| `R2CConformal` | Regression-to-classification |
| `ConformalLoss` | Legacy training+calibration wrapper |
| `MultiDimensionalConformalLoss` | Multi-dimensional legacy wrapper |
