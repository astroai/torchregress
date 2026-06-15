# Uncertain-GT + Density Conformal Comparison (Real Data)

Script: `examples/uncertain_gt_density_conformal_realdata_comparison.py`

→ API: [`SplitConformal`](../api/losses.md#splitconformal), [`DensityConformal`](../api/conformal.md#densityconformal), [`NoisyTargetGaussianNLL`](../api/losses.md#noisytargetgaussiannll).

Compares:

- `SplitConformal`
- `DensityConformal`
- `PrevalenceAdjustedCP`
- `MonteCarloConformal`

and reports uncertain-ground-truth objectives:

- `NoisyTargetGaussianNLL`
- `ConsistencyRegLoss`
- `PseudoLabelNLL`

Reported metrics:

- `Coverage90`
- `Width90`
- `NoisyTargetNLL`
- `ConsistencyLoss`
- `PseudoLabelNLL`
- `train_s`, `eval_s`

## Run

```bash
uv run python examples/uncertain_gt_density_conformal_realdata_comparison.py
```

## Summary Artifact

```bash
uv run python examples/uncertain_gt_density_conformal_realdata_comparison.py \
  --summary-json-path reports/example_summaries/uncertain_gt_density_conformal_realdata_comparison_full.json
```
