# Uncertain-GT + Density Conformal Comparison

Script: `examples/uncertain_gt_density_conformal_comparison.py`

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
uv run python examples/uncertain_gt_density_conformal_comparison.py
```

## Summary Artifact

```bash
uv run python examples/uncertain_gt_density_conformal_comparison.py \
  --summary-json-path reports/example_summaries/uncertain_gt_density_conformal_comparison_full.json
```
