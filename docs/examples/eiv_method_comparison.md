# EIV Method Comparison

This example compares multiple Error-in-Variables (EIV) loss variants on a shared synthetic measurement-error regression task.

→ API: [`FunctionalEIVLoss`](../api/losses.md#functionaleivloss), [`OrthogonalDistanceRegressionLoss`](../api/losses.md#orthogonaldistanceregressionloss). Guide: [EIV losses](../losses/eiv.md).

Script: `examples/eiv_method_comparison.py`

## What It Compares

- Baseline observed-data MSE training (context baseline)
- `FunctionalEIVLoss` (analytic)
- `FunctionalEIVLoss` (Monte Carlo)
- `StructuralEIVLoss`
- `OrthogonalDistanceRegressionLoss`
- `EnsembleEIVLoss`

## Metrics Reported

- Clean test MSE (true inputs / true targets)
- Observed test MSE (noisy inputs / noisy targets)
- Stress MSE (noisy inputs / clean targets)
- Runtime (`train_s`, `eval_s`)

## Fairness / Comparability Notes

- Fixed seed and shared synthetic measurement-error splits
- Shared model architecture and optimizer budget
- Shared summary table formatting and runtime tracking
- ODR uses a reduced inner-optimization budget for tractable example runtime

## When This Fails

- Synthetic low-dimensional setup is not a substitute for a real benchmark.
- ODR runtime can dominate on larger problems; compare quality/runtime tradeoffs on your target data.
- Structural EIV with zero cross-covariance in this example does not exercise all structural-error regimes.

## Run

```bash
uv run python examples/eiv_method_comparison.py
```
