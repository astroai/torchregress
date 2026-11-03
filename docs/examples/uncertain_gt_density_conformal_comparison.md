# Uncertain-GT + Density Conformal Comparison

Script: `examples/uncertain_gt_density_conformal_comparison.py`

→ API: [`SplitConformal`](../api/losses.md), [`DensityConformal`](../api/conformal.md), [`NoisyTargetGaussianNLL`](../api/losses.md), [`PseudoLabelNLL`](../api/losses.md).

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

## Datasets

Select the dataset with `--dataset`. Calibration always uses the noisy observed labels; coverage is measured against the clean targets.

| `--dataset` | Clean targets | Label noise | Predictor, pseudo labels, MC draws | Seed / splits |
|:------------|:--------------|:------------|:-----------------------------------|:--------------|
| `synthetic` (default) | simulated long-tailed function of 5 Gaussian features | heteroscedastic, $\sigma_{\text{obs}} = 0.08 + 0.25\,\mathrm{sigmoid}(1.2\,x_0)$ | simulated: predictor = clean target + noise; teacher/pseudo labels and 24 MC draws simulated around it | seed 260227; cal 400 / test 300 |
| `diabetes` | [`sklearn.datasets.load_diabetes`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) targets, standardized on train labels | feature-dependent, $\sigma_{\text{obs}} = 0.08 + 0.22\,\mathrm{sigmoid}(2\,x_0)$ | a 2-layer MLP fitted on noisy train labels; predictive variance from train residuals; 24 MC draws from that variance | seed 260305; train 240 / cal 120 / test 80 |

On `diabetes`, the time spent fitting the shared predictor is included in every method's `train_s`.

## Run

```bash
pixi run python examples/uncertain_gt_density_conformal_comparison.py                      # synthetic
pixi run python examples/uncertain_gt_density_conformal_comparison.py --dataset diabetes   # real data
```

## Summary Artifact

```bash
pixi run python examples/uncertain_gt_density_conformal_comparison.py \
  --summary-json-path reports/example_summaries/uncertain_gt_density_conformal_comparison_full.json
pixi run python examples/uncertain_gt_density_conformal_comparison.py --dataset diabetes \
  --summary-json-path reports/example_summaries/uncertain_gt_density_conformal_comparison_diabetes.json
```
