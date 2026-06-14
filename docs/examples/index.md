# Examples

This section contains practical examples of using torchregress for various regression tasks. The examples are designed to demonstrate key features and use cases of the library.

## Start Here (Task-First)

If you are choosing a method for a hard regression problem (outliers, multimodal targets,
calibration, OOD robustness, noisy features, imbalance), start with the
[Task-First Method Selection Matrix](../guide/method-selection.md) and then come
back to this page for runnable examples. For budget/latency/coverage constraints, use
[Choosing Methods by Constraint](../guide/choosing-by-constraint.md) before selecting an example.
Use the [Comparative Evidence Matrix](../reports/comparative_evidence_matrix.md) to see which hard
tasks currently have decision-grade comparisons vs demo-only examples.

## Getting Started

**New to torchregress?** Start with the [Concepts Guide](../getting-started/concepts.md) to learn key concepts, then use the [Task-First Method Selection Matrix](../guide/method-selection.md) to pick a method family.

## Example Reading Guide (Audit Priority)

Use the comparison examples first when making implementation decisions:

- `examples/comprehensive_comparison.py`: robust losses + uncertainty + ensemble behavior on multiple scenarios
- `examples/comprehensive_loss_comparison.py`: controlled robust-loss comparisons
- `examples/imbalanced_regression.py`: tail performance vs calibration tradeoffs
- `examples/evaluate_conformal_methods.py`: coverage vs interval width tradeoffs
- `examples/ood_selective_prediction_comparison.py`: OOD vs selective-prediction tradeoffs (AURC, rejection policy, OOD gap)
  now includes DeepEnsemble, heteroscedastic ensemble, MC dropout, `SWAG`, and `BNN`
- `examples/ood_selective_prediction_realdata_comparison.py`: real-data OOD/selective comparison on Diabetes with covariate-shift split
- `examples/eiv_method_comparison.py`: EIV method tradeoffs (analytic/MC/ODR/ensemble variants)
- `examples/eiv_method_realdata_comparison.py`: real-data EIV tradeoffs on Diabetes with synthetic measurement-error injection
- `examples/multimodal_method_comparison.py`: Gaussian vs MDN vs optional flow on multimodal multi-target data
- `examples/multimodal_method_realdata_comparison.py`: real-data features + synthetic multimodal targets for Gaussian/MDN/flow comparison
- `examples/contrastive_flow_parameter_estimation_comparison.py`: shared-budget synthetic comparison for Gaussian summary density vs plain flow vs contrastive flow
- `examples/noisy_label_comparison.py`: robust + probabilistic methods under label corruption with calibration metrics
- `examples/ordinal_regression_comparison.py`: ordered-target comparison (`OrdinalCrossEntropy`, `CumulativeLink`, `CORAL`)
- `examples/ordinal_regression_realdata_comparison.py`: real-data ordinal comparison on Diabetes with train-quantile binning
- `examples/ordinal_uncertain_ground_truth_comparison.py`: ordered-bin ambiguous-label comparison with soft plausibility targets and soft pseudo labels
- `examples/censored_regression_comparison.py`: censored/interval-censored comparison (`CensoredGaussianNLL`, `CensoredQuantile`, `AFT`)
- `examples/censored_regression_realdata_comparison.py`: real-data censored comparison on Diabetes with shared censoring overlays
- `examples/propensity_tail_regression_comparison.py`: selection-bias and long-tail comparison (`MSE`, `DensityWeighted`, `PropensityWeighted`, `GaussianNLL`, `Quantile90`)
- `examples/constraints_calibration_comparison.py`: constrained heads + post-hoc calibration transforms comparison
- `examples/transformed_target_regression_comparison.py`: skewed-target transform comparison (`MSE`, `LogTransform`, `BoxCox`, `Sqrt`)
- `examples/semi_supervised_regression_comparison.py`: masked-label semi-supervised regression with pseudo labels and teacher consistency
- `examples/ppi_calibrated_mean.py`: raw vs affinely calibrated prediction-powered mean inference (arXiv:2604.21260)
- `examples/ppi_mean_plus_split_conformal.py`: PPI-style mean inference plus `SplitConformal` predictive bands with a labeled split (see `docs/methods/inference.md`)
- `examples/uncertain_gt_density_conformal_comparison.py`: uncertain-label losses + density/prevalence/MC conformal comparison
- `examples/uncertain_gt_density_conformal_realdata_comparison.py`: real-data uncertain-label + conformal comparison on Diabetes
- `examples/causal_dr_uplift_comparison.py`: doubly-robust ATE/CATE comparison with overlap diagnostics
- `examples/causal_dr_realdata_comparison.py`: real-covariate DR ATE/CATE comparison with overlap diagnostics
- `examples/external_comparison_conformal_vs_mapie.py`: conformal intervals (split + CQR) vs MAPIE; JSON → `reports/external_comparison_conformal_vs_mapie_latest.json`
- `examples/external_comparison_bayesian_linear_vs_botorch.py`: low-shot Bayesian linear head vs BoTorch `SingleTaskGP`; JSON → `reports/external_comparison_bayesian_linear_vs_botorch_latest.json`
- `examples/external_comparison_tweedie_vs_sklego.py`: Tweedie / compound-Poisson regression vs scikit-lego `GLMRegressor`; JSON → `reports/external_comparison_tweedie_vs_sklego_latest.json`
- [External comparison guide](external-comparison-vs-mapie-botorch-sklego.md): consolidated page for the three external benchmarks.
- `examples/normalizing_flows_multitarget.py`: multi-target multimodal modeling (single-method deep dive)
- `examples/contrastive_flow_parameter_estimation.py`: nuisance-aware parameter scans with `ContrastiveFlowLoss`

When comparing methods, prefer examples that document:

- fixed seeds / shared splits
- comparable training budgets
- common metrics (accuracy, calibration, runtime)
- explicit failure modes / caveats

## Basic Examples

### [Basic Usage](basic_usage.md)

This tutorial covers the fundamental usage patterns of torchregress:

- Setting up simple regression models
- Choosing and applying loss functions
- Evaluating models with appropriate metrics
- Basic visualization of results

```python
import torch
import torchregress as tr

# Example of basic torchregress usage
X_train, y_train = load_data()
model = MyRegressionModel()
loss_fn = tr.losses.WeightedHuberLoss()

# Train model
# ...

# Evaluate
predictions = model(X_test)
rmse = tr.metrics.rmse(predictions, y_test)
```

### [Loss Comparison](loss_comparison.md)

This example compares different loss functions on the same dataset:

- Comparing traditional vs. robust losses
- Analyzing how different losses handle outliers
- Visualizing the effect of loss choice on predictions
- Determining which loss is best for different data characteristics

```python
import torchregress as tr

# Dictionary of loss functions to compare
losses = {
    "MSE": tr.losses.WeightedMSELoss(),
    "MAE": tr.losses.WeightedL1Loss(),
    "Huber": tr.losses.WeightedHuberLoss(),
    "Barron": tr.losses.BarronLoss(alpha=1.0, scale=1.0),
    "LogCosh": tr.losses.LogCoshLoss(),
}

# Compare losses and visualize results
# ...
```

### [Comprehensive Comparison](https://github.com/sfabbro/torchregress/blob/main/examples/comprehensive_comparison.py) 🆕

**All-in-one comparison** demonstrating the three main capabilities of torchregress:

- **Robust Regression** - Handling outliers (MSE vs. Huber vs. Cauchy)
- **Uncertainty Estimation** - Quantifying confidence (Gaussian NLL, ensembles)
- **Ensemble Methods** - Combining models (Deep Ensemble, Heteroscedastic Ensemble)

Three challenging scenarios:
1. Clean data (baseline comparison)
2. Data with outliers (robust losses)
3. Heteroscedastic data (uncertainty decomposition)

```python
# Compare robust losses on outlier data
losses = {
    "MSE": WeightedMSELoss(),      # Sensitive to outliers
    "Huber": WeightedHuberLoss(delta=1.0),  # Balanced
    "Barron": BarronLoss(alpha=1.0, scale=1.0),  # Smooth adaptive family
    "Cauchy": CauchyLoss(scale=0.5) # Very robust
}
```

## Advanced Examples

### [OT shift conformal demo](ot_shift_conformal_demo.md)

- Synthetic calibration vs shifted target scores; prints CDF-gap diagnostic, weighted split-conformal threshold, and mean prediction-set size on random candidate score matrices.
- Script: [`examples/ot_shift_conformal_demo.py`](https://github.com/sfabbro/torchregress/blob/main/examples/ot_shift_conformal_demo.py).

### [Bayesian linear head (test-time) demo](test_time_bayesian_linear_head.md)

- Synthetic linear targets on fixed features; compares batch `BayesianLinearHead.fit` to two-step `RecursiveBayesianHead.partial_fit` with `forgetting_factor=1`, reports posterior error vs a known weight vector, and prints rough held-out Gaussian interval coverage.
- Script: [`examples/test_time_bayesian_linear_head_demo.py`](https://github.com/sfabbro/torchregress/blob/main/examples/test_time_bayesian_linear_head_demo.py).

### [BLR predictive adapter (SupportsPredictiveBatch)](test_time_blr_predictive_adapter.md)

- Thin wrapper exposing `predict_distribution(...)` around `BayesianLinearHead.predictive_batch`.
- Runtime protocol-check against `SupportsPredictiveBatch` and prints `PredictiveBatch` diagnostics.
- Script: [`examples/test_time_blr_predictive_adapter_demo.py`](https://github.com/sfabbro/torchregress/blob/main/examples/test_time_blr_predictive_adapter_demo.py).

### [Gaussian Wasserstein bound demo](gaussian_wasserstein_bound.md)

- One-step mean + full-covariance supervision with [`GaussianWassersteinBoundLoss`](../losses/gaussian_wasserstein.md); prints loss and gradient norms.
- Script: [`examples/gaussian_wasserstein_bound_demo.py`](https://github.com/sfabbro/torchregress/blob/main/examples/gaussian_wasserstein_bound_demo.py).

### [Wasserstein-bound hybrid pretrain](wasserstein_bound_hybrid_pretrain.md)

- Tiny scalar head: neighbourhood covariance pseudo-labels + diagonal Wasserstein-bound pretrain, then Gaussian NLL fine-tune; prints NLL before/after the second phase.
- Script: [`examples/wasserstein_bound_hybrid_pretrain_demo.py`](https://github.com/sfabbro/torchregress/blob/main/examples/wasserstein_bound_hybrid_pretrain_demo.py).

### [Beta-NLL heteroscedastic demo](heteroscedastic_beta_nll.md)

- Small Gaussian-head MLP on synthetic heteroscedastic noise (scale grows with $|x|$).
- Trains with [`GaussianNLLLoss`](../losses/gaussian.md) vs [`BetaNLLLoss`](../losses/beta_nll.md) from the **same** initial weights; reports validation RMSE and validation Gaussian NLL.
- Script: [`examples/heteroscedastic_beta_nll_demo.py`](https://github.com/sfabbro/torchregress/blob/main/examples/heteroscedastic_beta_nll_demo.py).

### [Ensemble Methods](ensemble_methods.md) 🆕

**Complete guide to uncertainty quantification with ensembles:**

- **Deep Ensemble** - Epistemic uncertainty from model disagreement
- **Heteroscedastic Ensemble** - Both epistemic and aleatoric uncertainty
- **Batch Ensemble** - Efficient alternative for limited compute
- **Uncertainty Decomposition** - Separating model vs. data uncertainty

Includes decision trees, comparison tables, and complete working example ([`ensemble_tutorial.py`](https://github.com/sfabbro/torchregress/blob/main/examples/ensemble_tutorial.py)).

```python
# Train heteroscedastic ensemble with uncertainty decomposition
ensemble_models = train_heteroscedastic_ensemble(n_models=5, ...)
epistemic, aleatoric = ensemble_variance_decomposition(means, log_vars)
```

### [Ordinal Regression Comparison](ordinal_regression_comparison.md)

- Shared-budget comparison for ordered targets.
- Compares class-logit cross-entropy against cumulative-link and CORAL objectives.
- Reports ordinal accuracy, class-index MAE, QWK, and runtime.

### [Ordinal Regression Comparison (Real Data)](ordinal_regression_realdata_comparison.md)

- Shared-budget comparison on real Diabetes covariates/targets with train-quantile binning.
- Compares class-logit cross-entropy against cumulative-link and CORAL objectives.
- Reports ordinal accuracy, class-index MAE, QWK, and runtime.

### [Ordinal Uncertain Ground Truth Comparison](ordinal_uncertain_ground_truth_comparison.md)

- Ordered-bin regression-as-classification comparison for ambiguous labels represented as soft class probabilities.
- Compares hard sampled labels, soft plausibility targets, soft pseudo labels, and a cumulative-link soft-target variant.
- Reports ordinal accuracy, class-index MAE, QWK, true-class NLL, plausibility cross-entropy, pseudo-label acceptance, and runtime.

### [Censored Regression Comparison](censored_regression_comparison.md)

- Shared-budget comparison for right/left and interval-censored targets.
- Compares Gaussian NLL, quantile, and AFT censored losses.
- Reports true-target MAE, observed MAE, concordance index, censoring rate, and runtime.

### [Censored Regression Comparison (Real Data)](censored_regression_realdata_comparison.md)

- Shared-budget comparison on real Diabetes covariates/targets with synthetic censoring overlays.
- Compares Gaussian NLL, quantile, and AFT censored losses.
- Reports true-target MAE, observed MAE, concordance index, censoring rate, and runtime.

### [Propensity + Tail Regression Comparison](propensity_tail_regression_comparison.md)

- Shared-budget comparison under covariate-dependent label observation.
- Compares reweighting methods plus distributional/quantile families.
- Reports overall MAE, upper-tail MAE/RMSE, native interval coverage/width, and runtime.

### [Constraints + Calibration Comparison](constraints_calibration_comparison.md)

- Demonstrates output-head constraints and post-hoc calibration transforms in one workflow.
- Compares raw outputs against calibrated + constrained outputs.
- Reports MAE, NLL, PIT chi-square, crossing-rate, bound-violation, and runtime.

### [Transformed-Target Regression Comparison](transformed_target_regression_comparison.md)

- Shared-budget benchmark for skewed positive targets with multiplicative noise.
- Compares `MSE` against `LogTransformLoss`, `BoxCoxTransformLoss`, and `SqrtTransformLoss`.
- Reports MSE, MAE, R2, MAPE, upper-tail MAE, and runtime.

### [Semi-Supervised Regression Comparison](semi_supervised_regression_comparison.md)

- Real-data proxy benchmark on Diabetes with train-label masking.
- Compares `SupervisedMSE`, `PseudoLabelConsistencyLoss`, and `PseudoLabelNLL`.
- Reports MSE, MAE, R2, pseudo-label acceptance/confidence, and runtime.

### [Uncertain-GT + Density Conformal Comparison](uncertain_gt_density_conformal_comparison.md)

- Shared-budget comparison for uncertain ground-truth supervision and density-aware conformal intervals.
- Compares `SplitConformal`, `DensityConformal`, `PrevalenceAdjustedCP`, and `MonteCarloConformal`.
- Reports coverage/width plus uncertain-GT objective values (`NoisyTargetNLL`, `ConsistencyLoss`, `PseudoLabelNLL`).

### [Uncertain-GT + Density Conformal Comparison (Real Data)](uncertain_gt_density_conformal_realdata_comparison.md)

- Shared-budget comparison on real Diabetes covariates/targets with feature-dependent annotation noise.
- Compares `SplitConformal`, `DensityConformal`, `PrevalenceAdjustedCP`, and `MonteCarloConformal`.
- Reports coverage/width plus uncertain-GT objective values (`NoisyTargetNLL`, `ConsistencyLoss`, `PseudoLabelNLL`).

### [Causal DR Uplift Comparison](causal_dr_uplift_comparison.md)

- Shared-budget comparison for doubly-robust causal regression (`dr_ate`, `dr_cate`).
- Includes a synthetic uplift scenario.
- Reports ATE error, CI quality, overlap-rate, and effective sample size diagnostics.

### [Causal DR Comparison (Real Covariates)](causal_dr_realdata_comparison.md)

- Shared-budget comparison for doubly-robust causal regression (`dr_ate`, `dr_cate`) on real covariates.
- Uses Diabetes features/baseline outcomes with confounded treatment assignment and known effect construction.
- Reports ATE error, CI quality, overlap-rate, and effective sample size diagnostics.

### [Conformal Regression](conformal_regression_example.md)

- Using conformal prediction to obtain prediction intervals with guaranteed coverage.
- Compare against `evaluate_conformal_methods.py` for coverage-vs-width tradeoffs across methods.

### [Evidential Regression](evidential_regression.md)

- Decomposing uncertainty into aleatoric and epistemic components.

### [Imbalanced Regression](imbalanced_regression.md)

- Handling imbalanced datasets in regression tasks.

### [Balanced MSE / BMC](balanced_mse.md)

- Skewed targets: [`BalancedMSELoss`](../losses/imbalanced.md) and [`BMCLoss`](../losses/imbalanced.md) vs plain MSE; script [`examples/balanced_mse_demo.py`](https://github.com/sfabbro/torchregress/blob/main/examples/balanced_mse_demo.py).
- Includes calibration-validation guidance because aggressive reweighting can distort uncertainty.

### [Noisy Labels Regression](noisy_labels_regression.md)

- Training models on data with noisy labels.

### [Noisy Label Comparison](noisy_label_comparison.md)

- Shared-budget comparison of robust/probabilistic methods under label corruption.
- Includes split-conformal coverage/width metrics plus native interval metrics for Gaussian/quantile heads.

### [Noisy Label Real-Data Comparison](noisy_label_realdata_comparison.md)

- Shared-budget comparison on the Diabetes dataset with synthetic label corruption.
- Adds real-data evidence breadth for noisy-label robustness/calibration tradeoffs.

### [Normalizing Flows](normalizing_flows_multitarget.md)

- Using normalizing flows for multi-target regression.
- Multi-target/multimodal demo; compare with Gaussian and MDN baselines separately for decision-grade selection.

### [Multimodal Method Comparison](multimodal_method_comparison.md)

- Shared-budget comparison of `GaussianNLL`, `MDN`, and optional flow models on multimodal multi-target data.
- Includes common point metrics, NLL, calibration error, energy score, and runtime.

### [Multimodal Method Comparison (Real Data Features)](multimodal_method_realdata_comparison.md)

- Shared-budget comparison on Diabetes features with synthetic multimodal multi-target outputs.
- Adds real-covariate evidence breadth for Gaussian/MDN/flow tradeoffs.

### [OOD / Selective Prediction Comparison](ood_selective_prediction_comparison.md)

- Shared-budget comparison of uncertainty methods for risk-coverage and OOD signal behavior.
- Includes AURC, rejection-policy risk/coverage, split-conformal interval metrics, and OOD uncertainty-gap summaries.

### [OOD / Selective Prediction Comparison (Real Data)](ood_selective_prediction_realdata_comparison.md)

- Shared-budget comparison on Diabetes with deterministic covariate-shift OOD split.
- Adds real-data evidence breadth for OOD/selective tradeoffs across DeepEnsemble/MCDropout/`SWAG`/`BNN` with split-conformal interval diagnostics.

### [EIV Method Comparison](eiv_method_comparison.md)

- Shared-budget comparison of EIV variants (`FunctionalEIV`, `StructuralEIV`, `ODR`, `EnsembleEIV`).
- Includes clean-vs-observed test MSE and runtime tradeoffs.

### [EIV Method Comparison (Real Data)](eiv_method_realdata_comparison.md)

- Shared-budget EIV comparison on Diabetes with synthetic feature/label measurement error.
- Adds real-data evidence breadth for EIV quality/runtime tradeoffs.

## Running the Examples

All examples can be run directly from the repository:

```bash
# Clone the repository
git clone https://github.com/sfabbro/torchregress.git
cd torchregress

# Install dependencies
pip install -e '.[all]'

# Run a specific example
python examples/basic_usage.py
```

For questions or issues with the examples, please [open an issue](https://github.com/sfabbro/torchregress/issues) in the repository.
