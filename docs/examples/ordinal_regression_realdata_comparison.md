# Ordinal Regression Comparison (Real Data)

Script: `examples/ordinal_regression_realdata_comparison.py`

Compares three ordered-target methods on real covariates/targets (`sklearn` Diabetes)
with train-quantile binning:

- `OrdinalCrossEntropy`
- `CumulativeLink`
- `CORAL`

Reported metrics:

- `Accuracy`
- `OrdinalMAE` (class-index MAE)
- `QWK` (quadratic weighted kappa)
- `train_s`, `eval_s`

## Run

```bash
uv run python examples/ordinal_regression_realdata_comparison.py
```

## Summary Artifact

```bash
uv run python examples/ordinal_regression_realdata_comparison.py \
  --summary-json-path reports/example_summaries/ordinal_regression_realdata_comparison_full.json
```
