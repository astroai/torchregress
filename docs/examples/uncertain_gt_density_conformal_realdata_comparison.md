# Uncertain-GT + Density Conformal Comparison (Real Data)

Script: `examples/uncertain_gt_density_conformal_realdata_comparison.py`

→ API: [`SplitConformal`](../api/losses.md), [`DensityConformal`](../api/conformal.md), [`NoisyTargetGaussianNLL`](../api/losses.md).

Compares:

- `SplitConformal`
- `DensityConformal`
- `PrevalenceAdjustedCP`
- `MonteCarloConformal`

and reports uncertain-ground-truth objectives:

- `NoisyTargetGaussianNLL`
- `ConsistencyRegLoss`
- `PseudoLabelNLL`

Reported metrics (summary JSON column names — class names in parentheses):

- `Coverage90`
- `Width90`
- `NoisyTargetNLL` (`NoisyTargetGaussianNLL`)
- `ConsistencyLoss` (`ConsistencyRegLoss`)
- `PseudoLabelNLL`
- `train_s`, `eval_s`

## Run

```bash
pixi run python examples/uncertain_gt_density_conformal_realdata_comparison.py
```

## Summary Artifact

```bash
pixi run python examples/uncertain_gt_density_conformal_realdata_comparison.py \
  --summary-json-path reports/example_summaries/uncertain_gt_density_conformal_realdata_comparison_full.json
```
