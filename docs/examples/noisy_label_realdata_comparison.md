# Noisy Label Real-Data Comparison

→ Guide: [Noisy labels](../losses/noisy_labels.md). API: [Robust and probabilistic losses](../api/losses.md#weightedhuberloss), [interval metrics](../api/metrics.md#prediction_interval_coverage_probability).

This example runs a shared-budget comparison of robust and probabilistic methods on a **real tabular dataset**
([`sklearn.datasets.load_diabetes`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html))
with **synthetic label corruption**.

Script: `examples/noisy_label_realdata_comparison.py`

## What It Compares

- [`WeightedMSELoss`](../api/losses.md#weightedmseloss) baseline
- [`WeightedHuberLoss`](../api/losses.md#weightedhuberloss)
- [`CauchyLoss`](../api/losses.md#cauchyloss)
- [`GaussianNLLLoss`](../api/losses.md#gaussiannllloss)
- [`MultiQuantileLoss`](../api/losses.md#multiquantileloss) (`Quantile90`)

## Metrics Reported

- `CleanMSE`, `CleanMAE` (against clean test labels)
- `ObsMSE` (against corrupted test labels)
- `ConformalCov90`, `ConformalWidth90`, `ConformalIS90`
- `NativeCov90`, `NativeWidth90` (Gaussian / quantile heads)
- `train_s`, `eval_s`

## Fairness Notes

- Fixed seed and shared train/cal/test split
- Shared corruption mask policy and training budget
- Coverage is evaluated on **clean** test labels, while conformal calibration uses **noisy** calibration labels

## When This Fails / Caveats

- This is still **synthetic corruption on real data**, not naturally noisy-label ground truth.
- Results can vary with corruption ratio/scale and train/cal split size.
- Coverage quality can degrade under severe corruption when calibration labels are heavily contaminated.

## Run

```bash
uv run python examples/noisy_label_realdata_comparison.py
```
