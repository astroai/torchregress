# Multimodal Method Comparison

→ Guide: [Multi-target regression](../guide/multi-target-regression.md). API: [`GaussianNLLLoss`](../api/losses.md), [`MDNLoss`](../api/losses.md), [`NormalizingFlowLoss`](../api/losses.md), [`marginal_calibration_error`](../api/metrics.md).

This example compares shared-budget probabilistic methods on a multimodal, multi-target synthetic regression task.

Script: `examples/multimodal_method_comparison.py`

## What It Compares

- Diagonal Gaussian NLL baseline ([`GaussianNLLLoss`](../api/losses.md))
- Mixture Density Network ([`MDNLoss`](../api/losses.md))
- Conditional normalizing flow ([`NormalizingFlowLoss`](../api/losses.md), optional if `zuko` is installed)

## Metrics Reported

- Point metrics (`MSE`, `MAE`, `R2`) using predictive mean / sample mean
- Test negative log-likelihood (`NLL`)
- Multivariate Energy Score (`Energy`)
- Marginal calibration error from predictive samples (`MCE`) — [`marginal_calibration_error`](../api/metrics.md)
- Runtime (`train_s`, `eval_s`)

## Fairness / Comparability Notes

- Fixed seeds and shared synthetic train/test splits
- Shared MLP backbone scale and optimizer budget
- Shared summary table and runtime tracking
- Common sample-based calibration and distribution-fit metrics

## When This Fails

- Synthetic 2D multimodal data is useful for method behavior, not external performance claims.
- Flow row is optional and may be skipped if `zuko` is not installed (`torchregress[flows]`).
- Gaussian baseline is intentionally unimodal and can look artificially strong on point metrics while missing modes.

## Run

```bash
uv run python examples/multimodal_method_comparison.py
```
