# Uncertain-GT + Density Conformal Comparison

Script: `examples/uncertain_gt_density_conformal_comparison.py`

→ API: [`SplitConformal`](../api/losses.md#splitconformal), [`DensityConformal`](../api/conformal.md#densityconformal), [`NoisyTargetGaussianNLL`](../api/losses.md#noisytargetgaussiannll), [`PseudoLabelNLL`](../api/losses.md#pseudolabelnll).

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

See also: [Uncertain-GT + Density Conformal Comparison (Real Data)](uncertain_gt_density_conformal_realdata_comparison.md)
