# Propensity + Tail Regression Comparison

Script: `examples/propensity_tail_regression_comparison.py`

Compares three approaches under covariate-dependent label observation:

- `MSE` (observed-only baseline)
- `DensityWeighted`
- `PropensityWeighted`

Reported metrics:

- `MAE`
- `TailMAE90`
- `TailRMSE90`
- `ObservedRate`
- `train_s`, `eval_s`

## Run

```bash
uv run python examples/propensity_tail_regression_comparison.py
```

## Summary Artifact

```bash
uv run python examples/propensity_tail_regression_comparison.py \
  --summary-json-path reports/example_summaries/propensity_tail_regression_comparison_full.json
```
