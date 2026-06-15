# Propensity + Tail Regression Comparison

Script: `examples/propensity_tail_regression_comparison.py`

→ API: [`PropensityWeightedLoss`](../api/losses.md#propensityweightedloss), [`DensityWeightedLoss`](../api/losses.md#densityweightedloss), [`GaussianNLLLoss`](../api/losses.md#gaussiannllloss), [`MultiQuantileLoss`](../api/losses.md#multiquantileloss).

Compares five approaches under covariate-dependent label observation:

- `MSE` (observed-only baseline)
- `DensityWeighted`
- `PropensityWeighted`
- `GaussianNLL`
- `Quantile90`

Reported metrics:

- `MAE`
- `TailMAE90`
- `TailRMSE90`
- `NativeCov90`
- `NativeWidth90`
- `TailCov90`
- `ObservedRate`
- `train_s`, `eval_s`

## When This Fails

- Propensity weighting can become unstable if observation propensities are very small.
- Density weighting can over-amplify noisy sparse tails.
- Gaussian intervals can under-cover multimodal tails; quantile intervals can become too wide in low-signal regions.

## Run

```bash
uv run python examples/propensity_tail_regression_comparison.py
```

## Summary Artifact

```bash
uv run python examples/propensity_tail_regression_comparison.py \
  --summary-json-path reports/example_summaries/propensity_tail_regression_comparison_full.json
```
