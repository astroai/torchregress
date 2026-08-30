# Noisy Label Comparison

→ Guide: [Noisy labels](../losses/noisy_labels.md), [Robust losses](../losses/robust.md). API: [Robust loss section](../api/losses.md), [interval metrics](../api/metrics.md).

This example provides a dedicated runnable comparison for label-corrupted regression with both robustness and calibration metrics.

Script: `examples/noisy_label_comparison.py`

## What It Compares

- [`WeightedMSELoss`](../api/losses.md) baseline
- [`WeightedHuberLoss`](../api/losses.md)
- [`CauchyLoss`](../api/losses.md)
- [`TukeyBiweightLoss`](../api/losses.md)
- [`GaussianNLLLoss`](../api/losses.md)
- [`MultiQuantileLoss`](../api/losses.md) (90% interval head)

## Metrics Reported

- Clean-label test metrics (`CleanMSE`, `CleanMAE`)
- Observed-label test MSE (`ObsMSE`)
- Shared split-conformal interval calibration metrics:
  - `ConformalCov90`
  - `ConformalWidth90`
  - `ConformalIS90`
- Native interval metrics for Gaussian / quantile models (`NativeCov90`, `NativeWidth90`)
- Runtime (`train_s`, `eval_s`)

## Fairness / Comparability Notes

- Fixed seeds and shared corrupted train/cal/test splits
- Shared architecture scale and optimizer budget
- Same noisy calibration split used to build split-conformal intervals for all methods
- Coverage evaluated against clean test labels to reveal robustness under label corruption

## When This Fails

- This is synthetic corruption (outlier-style label noise), not a full taxonomy of real annotation noise.
- Split-conformal intervals inherit noise in the calibration labels and can become conservative.
- The example compares losses and simple probabilistic heads, not co-teaching / noise-adaptive meta-learning methods (which are roadmap/concept coverage in docs, not implemented APIs).

## Run

```bash
pixi run python examples/noisy_label_comparison.py
```
