# Censored Regression Comparison

Script: `examples/censored_regression_comparison.py`

Compares three censored-loss approaches under a shared budget:

- `CensoredGaussianNLL`
- `CensoredQuantile`
- `AFT`

Censor code convention in the example and APIs:

- `0` observed
- `1` right-censored
- `-1` left-censored

The example also includes a small interval-censored subset using explicit bounds.

## Run

```bash
uv run python examples/censored_regression_comparison.py
```

## Summary Artifact

```bash
uv run python examples/censored_regression_comparison.py \
  --summary-json-path reports/example_summaries/censored_regression_comparison_full.json
```
