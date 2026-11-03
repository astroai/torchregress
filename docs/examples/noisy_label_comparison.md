# Noisy Label Comparison

→ Guide: [Noisy labels](../losses/noisy_labels.md), [Robust losses](../losses/robust.md). API: [Robust loss section](../api/losses.md), [interval metrics](../api/metrics.md).

This example provides a dedicated runnable comparison for label-corrupted regression with both robustness and calibration metrics.
One script covers two datasets, selected with `--dataset`:

| `--dataset` | Data | Corruption | Seed / splits (train / cal / test) | Budget |
|:------------|:-----|:-----------|:-----------------------------------|:-------|
| `synthetic` (default) | 1-D curve $y = \sin(1.6x) + 0.4x + 0.15x^2$ on $x \in [-3, 3]$ | 20% of labels get $\mathcal{N}(0, 2.5^2)$ shifts, plus base noise $0.15$ | seed 7; 512 / 128 / 256 | 50 epochs |
| `diabetes` | [`sklearn.datasets.load_diabetes`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) (10 features, targets standardized on clean train labels) | same mechanism, shift scale $1.75$ | seed 17; 256 / 96 / 90 | 40 epochs |

Script: `examples/noisy_label_comparison.py`

## What It Compares

- [`WeightedMSELoss`](../api/losses.md) baseline
- [`WeightedHuberLoss`](../api/losses.md)
- [`CauchyLoss`](../api/losses.md)
- [`TukeyBiweightLoss`](../api/losses.md) (`synthetic` only)
- [`GaussianNLLLoss`](../api/losses.md)
- [`MultiQuantileLoss`](../api/losses.md) (90% interval head, `Quantile90`)

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

- Fixed seeds and shared corrupted train/cal/test splits (per dataset)
- Shared architecture scale and optimizer budget
- Same noisy calibration split used to build split-conformal intervals for all methods
- Coverage evaluated against clean test labels to reveal robustness under label corruption

## When This Fails

- Both datasets use synthetic corruption (outlier-style label noise), not a full taxonomy of real annotation noise. On `diabetes` the features and targets are real, but the corruption is not naturally occurring.
- Split-conformal intervals inherit noise in the calibration labels and can become conservative; coverage can degrade when calibration labels are heavily contaminated.
- Results can vary with corruption ratio/scale and train/cal split size.
- The example compares losses and simple probabilistic heads, not co-teaching / noise-adaptive meta-learning methods (which are roadmap/concept coverage in docs, not implemented APIs).

## Run

```bash
pixi run python examples/noisy_label_comparison.py                      # synthetic
pixi run python examples/noisy_label_comparison.py --dataset diabetes   # real data
```

## Summary Artifact

```bash
pixi run python examples/noisy_label_comparison.py --dataset diabetes \
  --summary-json-path reports/example_summaries/noisy_label_comparison_diabetes.json
```
