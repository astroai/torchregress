# Transformed-Target Regression Comparison

→ Guide: [Target transforms](../losses/transforms.md). API: [`LogTransformLoss`](../api/losses.md#logtransformloss), [`BoxCoxTransformLoss`](../api/losses.md#boxcoxtransformloss), [`SqrtTransformLoss`](../api/losses.md#sqrttransformloss).

Script: `examples/transformed_target_regression_comparison.py`

Compares shared-budget positive-target models using:

- [`WeightedMSELoss`](../api/losses.md#weightedmseloss)
- [`LogTransformLoss`](../api/losses.md#logtransformloss)
- [`BoxCoxTransformLoss`](../api/losses.md#boxcoxtransformloss)
- [`SqrtTransformLoss`](../api/losses.md#sqrttransformloss)

Reported metrics:

- `MSE`
- `MAE`
- `R2`
- `MAPE`
- `TailMAE80`
- `train_s`, `eval_s`

## Run

```bash
uv run python examples/transformed_target_regression_comparison.py
```

## Summary Artifact

```bash
uv run python examples/transformed_target_regression_comparison.py \
  --summary-json-path reports/example_summaries/transformed_target_regression_comparison_full.json
```
