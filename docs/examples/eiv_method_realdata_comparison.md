# EIV Method Comparison (Real Data)

This example compares EIV losses on a **real tabular dataset** (`sklearn.datasets.load_diabetes`) with **synthetic feature/label measurement error injection**.

→ API: [`FunctionalEIVLoss`](../api/losses.md), [`OrthogonalDistanceRegressionLoss`](../api/losses.md).

Script: `examples/eiv_method_realdata_comparison.py`

## What It Compares

- Baseline observed-data MSE training
- `FunctionalEIVLoss` (analytic)
- `FunctionalEIVLoss` (Monte Carlo)
- `StructuralEIVLoss`
- `OrthogonalDistanceRegressionLoss`
- `EnsembleEIVLoss`

## Metrics Reported

- Clean test MSE (clean inputs / clean targets)
- Observed test MSE (noisy inputs / noisy targets)
- Stress MSE (noisy inputs / clean targets)
- Runtime (`train_s`, `eval_s`)

## Fairness Notes

- Fixed seed and shared Diabetes split
- Shared synthetic measurement-error injection policy
- Shared architecture/optimizer budget and runtime tracking
- ODR uses reduced inner iterations for tractable runtime

## When This Fails / Caveats

- This is real-data plus **synthetic** measurement error, not a domain benchmark.
- Diabetes is a small tabular dataset; rankings may not transfer to high-dimensional settings.
- ODR runtime can dominate as feature dimension / dataset size grows.

## Run

```bash
pixi run python examples/eiv_method_realdata_comparison.py
```
