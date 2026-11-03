# Multimodal Method Comparison

→ Guide: [Multi-target regression](../guide/multi-target-regression.md). API: [`GaussianNLLLoss`](../api/losses.md), [`MixtureDensityLoss`](../api/losses.md), [`NormalizingFlowLoss`](../api/losses.md), [`marginal_calibration_error`](../api/metrics.md).

This example compares shared-budget probabilistic methods on a multimodal, multi-target regression task, either fully synthetic or built on real tabular features.

Script: `examples/multimodal_method_comparison.py`

## Datasets

Select the dataset with `--dataset`. Each dataset keeps its own fixed seed, split sizes, and training budget.

| `--dataset` | Inputs | Targets | Defaults |
|:--|:--|:--|:--|
| `synthetic` (default) | 1D grid on $[-3, 3]$ | 2D targets with an input-dependent bimodal region ($\lvert x \rvert < 1.4$); train/test generated with seeds `seed` and `seed + 1` | `seed=42`, `n_train=384`, `n_test=256`, `epochs=40`, `eval_samples=128` |
| `diabetes` | Real `sklearn.datasets.load_diabetes` features (10-D) | **Synthetic** conditional multimodal 2D target built from the standardized Diabetes target and covariates; one shared permutation split | `seed=1234`, `n_train=256`, `n_test=128`, `epochs=30`, `eval_samples=96` |

## What It Compares

- Diagonal Gaussian NLL baseline ([`GaussianNLLLoss`](../api/losses.md))
- Mixture Density Network ([`MixtureDensityLoss`](../api/losses.md))
- Conditional normalizing flow ([`NormalizingFlowLoss`](../api/losses.md), optional if `zuko` is installed)

## Metrics Reported

- Point metrics (`MSE`, `MAE`, `R2`) using predictive mean / sample mean
- Test negative log-likelihood (`NLL`)
- Multivariate Energy Score (`Energy`)
- Marginal calibration error from predictive samples (`MCE`) — [`marginal_calibration_error`](../api/metrics.md)
- Runtime (`train_s`, `eval_s`)

## Fairness / Comparability Notes

- Fixed seeds and shared train/test splits within each dataset
- Shared MLP backbone scale and optimizer budget
- Shared summary table and runtime tracking
- Common sample-based calibration and distribution-fit metrics

## When This Fails

- Synthetic 2D multimodal data is useful for method behavior, not external performance claims.
- With `--dataset diabetes` the covariates are real, but multimodality is still **synthetically induced**; this reduces synthetic-only evidence risk but is not a domain benchmark for multimodal regression.
- Flow row is optional and may be skipped if `zuko` is not installed (`torchregress[flows]`).
- Gaussian baseline is intentionally unimodal and can look artificially strong on point metrics while missing modes.

## Run

```bash
pixi run python examples/multimodal_method_comparison.py
pixi run python examples/multimodal_method_comparison.py --dataset diabetes
```

## Summary Artifact

```bash
pixi run python examples/multimodal_method_comparison.py --dataset diabetes \
  --summary-json-path reports/example_summaries/multimodal_method_comparison_diabetes.json
```
