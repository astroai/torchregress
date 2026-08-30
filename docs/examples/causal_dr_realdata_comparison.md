# Causal DR Comparison (Real Covariates)

Script: `examples/causal_dr_realdata_comparison.py`

→ API: [`dr_ate`](../api/causal.md), [`dr_cate`](../api/causal.md).

Compares:

- `NaiveDiff`
- `dr_ate` (cross-fitted doubly-robust ATE)
- `dr_cate` (cross-fitted DR pseudo-outcome CATE)

across two real-covariate scenarios based on `sklearn` Diabetes features.

Reported metrics:

- `ATE_true`, `ATE_hat`, `ATE_abs_error`
- `CI_contains_true`, `CI_width`
- `OverlapRate`, `MinESS`
- `train_s`

## Run

```bash
pixi run python examples/causal_dr_realdata_comparison.py
```

## Summary Artifact

```bash
pixi run python examples/causal_dr_realdata_comparison.py \
  --summary-json-path reports/example_summaries/causal_dr_realdata_comparison_full.json
```
