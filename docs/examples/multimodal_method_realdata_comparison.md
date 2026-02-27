# Multimodal Method Comparison (Real Data Features)

This example compares shared-budget probabilistic methods on **real tabular features**
(`sklearn.datasets.load_diabetes`) with a **synthetic conditional multimodal multi-target target construction**.

Script: `examples/multimodal_method_realdata_comparison.py`

## What It Compares

- Diagonal Gaussian NLL baseline (`GaussianNLLLoss`)
- Mixture Density Network (`MDNLoss`)
- Conditional normalizing flow (`NormalizingFlowLoss`, optional if `zuko` is installed)

## Metrics Reported

- Point metrics (`MSE`, `MAE`, `R2`)
- `NLL`
- `Energy` (multivariate energy score)
- `MCE` (marginal calibration error)
- Runtime (`train_s`, `eval_s`)

## Fairness Notes

- Fixed seed and shared feature split
- Shared synthetic multimodal target construction on the same real covariates
- Shared model budgets and runtime tracking
- Common sample-based calibration and distribution-fit metrics

## When This Fails / Caveats

- Real covariates are used, but multimodality is still **synthetically induced**.
- Flow row is optional and may be skipped without `zuko` (`torchregress[flows]`).
- This reduces synthetic-only evidence risk but is not a domain benchmark for multimodal regression.

## Run

```bash
uv run python examples/multimodal_method_realdata_comparison.py
```
