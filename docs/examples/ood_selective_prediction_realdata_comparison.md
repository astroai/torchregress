# OOD / Selective Prediction Comparison (Real Data)

This example compares uncertainty methods on a **real tabular regression dataset**
(`sklearn.datasets.load_diabetes`) using a **deterministic covariate-shift OOD split**
and shared budgets.

Script: `examples/ood_selective_prediction_realdata_comparison.py`

## What It Compares

- `DeepEnsemble`
- `HeteroscedasticEnsemble`
- `MCDropoutWrapper` (proxy implementation in the example)
- `SWAG`
- `BayesianNeuralNetwork`

## Metrics Reported

- `MSE_ID`, `MSE_OOD`
- `AURC` (risk-coverage)
- `rej20_risk`, `rej20_cov`
- `ood_unc_gap`
- `train_s`, `eval_s`

## OOD Split Definition

- The example builds an OOD pool from samples with **extreme values of one selected feature**
  (absolute magnitude on a standardized feature axis).
- Remaining samples are shuffled and split into train and ID test sets.

## Fairness Notes

- Fixed seed and shared train/ID/OOD split
- Shared model budgets and uncertainty sample counts
- Common metrics and runtime reporting across methods

## When This Fails / Caveats

- This is **real data**, but not a labeled OOD benchmark suite.
- OOD is a **feature-shift proxy**; it may not match your deployment failure modes.
- Validate on domain-specific shifts and use multiple signals (not uncertainty-gap alone).

## Run

```bash
uv run python examples/ood_selective_prediction_realdata_comparison.py
```
