# Ordinal Regression Comparison

Script: `examples/ordinal_regression_comparison.py`

Compares three ordered-target methods under a shared budget:

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
uv run python examples/ordinal_regression_comparison.py
```

## Summary Artifact

```bash
uv run python examples/ordinal_regression_comparison.py \
  --summary-json-path reports/example_summaries/ordinal_regression_comparison_full.json
```
