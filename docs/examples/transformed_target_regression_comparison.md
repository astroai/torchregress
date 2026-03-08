# Transformed-Target Regression Comparison

Script: `examples/transformed_target_regression_comparison.py`

Compares shared-budget positive-target models using:

- `WeightedMSELoss`
- `LogTransformLoss`
- `BoxCoxTransformLoss`
- `SqrtTransformLoss`

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
