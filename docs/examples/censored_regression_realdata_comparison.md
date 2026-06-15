# Censored Regression Comparison (Real Data)

Script: `examples/censored_regression_realdata_comparison.py`

→ API: [`CensoredGaussianNLLLoss`](../api/losses.md#censoredgaussiannllloss), [`CensoredQuantileLoss`](../api/losses.md#censoredquantileloss), [`AFTLoss`](../api/losses.md#aftloss).

Compares three censored-loss approaches on real covariates/targets (`sklearn` Diabetes)
with shared synthetic censoring overlays:

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
uv run python examples/censored_regression_realdata_comparison.py
```

## Summary Artifact

```bash
uv run python examples/censored_regression_realdata_comparison.py \
  --summary-json-path reports/example_summaries/censored_regression_realdata_comparison_full.json
```
