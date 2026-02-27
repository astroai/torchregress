# Examples

This section contains practical examples of using torchregress for various regression tasks. The examples are designed to demonstrate key features and use cases of the library.

## Start Here (Task-First)

If you are choosing a method for a hard regression problem (outliers, multimodal targets,
calibration, OOD robustness, noisy features, imbalance), start with the
[Task-First Method Selection Matrix](../guides/method_selection_matrix.md) and then come
back to this page for runnable examples. For budget/latency/coverage constraints, use
[Choosing Methods by Constraint](../guides/choosing_by_constraint.md) before selecting an example.
Use the [Comparative Evidence Matrix](../guides/comparative_evidence_matrix.md) to see which hard
tasks currently have decision-grade comparisons vs demo-only examples.

## Getting Started

**New to torchregress?** Start with the [Concepts Guide](../guides/concepts.md) to learn key concepts, then use the [Task-First Method Selection Matrix](../guides/method_selection_matrix.md) to pick a method family.

## Example Reading Guide (Audit Priority)

Use the comparison examples first when making implementation decisions:

- `examples/comprehensive_comparison.py`: robust losses + UQ + ensemble behavior on multiple scenarios
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
- `examples/noisy_label_comparison.py`: robust + probabilistic methods under label corruption with calibration metrics
- `examples/noisy_label_realdata_comparison.py`: real-data (Diabetes) label-corruption comparison with shared calibration metrics
- `examples/photoz_benchmark_comparison.py`: SDSS-style photo-z benchmark for robust/probabilistic/EIV tradeoffs with photo-z metrics
- `examples/photoz_nnc_crps_rail_comparison.py`: ordered-bin NNC-CRPS-style photo-z comparison (examples-only) with calibration + PDF metrics
- `examples/ppi_photoz_inference_comparison.py`: prediction-powered inference (mean/quantile CI) under limited labels
- `examples/ordinal_regression_comparison.py`: ordered-target comparison (`OrdinalCrossEntropy`, `CumulativeLink`, `CORAL`)
- `examples/censored_regression_comparison.py`: censored/interval-censored comparison (`CensoredGaussianNLL`, `CensoredQuantile`, `AFT`)
- `examples/propensity_tail_regression_comparison.py`: selection-bias and long-tail comparison (`MSE`, `DensityWeighted`, `PropensityWeighted`)
- `examples/constraints_calibration_comparison.py`: constrained heads + post-hoc calibration transforms comparison
- `examples/uncertain_gt_density_conformal_comparison.py`: uncertain-label losses + density/prevalence/MC conformal comparison
- `examples/causal_dr_uplift_comparison.py`: doubly-robust ATE/CATE comparison with overlap diagnostics
- `examples/normalizing_flows_multitarget.py`: multi-target multimodal modeling (single-method deep dive)

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
loss_fn = tr.losses.HuberLoss()

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
    "MSE": tr.losses.MSELoss(),
    "MAE": tr.losses.L1Loss(),
    "Huber": tr.losses.HuberLoss(),
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
    "Huber": HuberLoss(delta=1.0),  # Balanced
    "Cauchy": CauchyLoss(scale=0.5) # Very robust
}
```

## Advanced Examples

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

### [Photometric Redshift Estimation](photoz.md)

A real-world application for astronomy:

- Implementing uncertainty-aware regression for photometric redshift estimation
- Using specialized loss functions for astronomical data
- Creating calibrated prediction intervals
- Evaluating results with domain-specific metrics

### [Photo-z Benchmark Comparison (SDSS-style)](photoz_benchmark_comparison.md)

- Shared-budget comparison benchmark for photo-z on SDSS-style features and measurement errors.
- Includes robust/probabilistic/EIV methods with NMAD, catastrophic-outlier rate, high-z MAE, interval coverage/width, and runtime.
- Uses cached real SDSS data if available, otherwise deterministic simulated SDSS-style fallback (CI/offline-friendly).

### [Photo-z NNC-CRPS + RAIL-Ready Comparison](photoz_nnc_crps_rail_comparison.md)

- Ordered-bin regression-as-classification comparison (`BinnedCE` vs `OrderedBinCRPS`) with optional post-hoc temperature scaling.
- Kept intentionally in `examples/` (specialized setup; no new core public API).
- Emits machine-readable summary rows with photo-z domain metrics plus PDF metrics (`CRPS`, `PDF_NLL`, `PITChi2`).

### [Photo-z RAIL Baseline Comparison](photoz_rail_baseline_comparison.md)

- Adapter workflow to merge torchregress photo-z summary artifacts with RAIL tabular baseline outputs.
- Core baseline set: `flexzboost`, `pzflow`, `delight`, `bpz` (optional `lephare`).
- Manifest-based parity mode enforces dataset/split/method consistency with the intended paper-comparison setup.

### [PPI Photo-z Inference Comparison](ppi_photoz_inference_comparison.md)

- Compares labeled-only vs prediction-powered confidence intervals for mean and high-quantile targets.
- Designed for low-label settings where a larger prediction-only pool is available.
- Emits summary artifacts with estimate error, CI width, and truth-coverage diagnostics.

### [Ordinal Regression Comparison](ordinal_regression_comparison.md)

- Shared-budget comparison for ordered targets.
- Compares class-logit cross-entropy against cumulative-link and CORAL objectives.
- Reports ordinal accuracy, class-index MAE, QWK, and runtime.

### [Censored Regression Comparison](censored_regression_comparison.md)

- Shared-budget comparison for right/left and interval-censored targets.
- Compares Gaussian NLL, quantile, and AFT censored losses.
- Reports true-target MAE, observed MAE, concordance index, censoring rate, and runtime.

### [Propensity + Tail Regression Comparison](propensity_tail_regression_comparison.md)

- Shared-budget comparison under covariate-dependent label observation.
- Compares naive observed-only MSE, density weighting, and propensity weighting.
- Reports overall MAE plus upper-tail MAE/RMSE and runtime.

### [Constraints + Calibration Comparison](constraints_calibration_comparison.md)

- Demonstrates output-head constraints and post-hoc calibration transforms in one workflow.
- Compares raw outputs against calibrated + constrained outputs.
- Reports MAE, NLL, PIT chi-square, crossing-rate, bound-violation, and runtime.

### [Uncertain-GT + Density Conformal Comparison](uncertain_gt_density_conformal_comparison.md)

- Shared-budget comparison for uncertain ground-truth supervision and density-aware conformal intervals.
- Compares `SplitConformal`, `DensityConformal`, `PrevalenceAdjustedCP`, and `MonteCarloConformal`.
- Reports coverage/width plus uncertain-GT objective values (`NoisyTargetNLL`, `ConsistencyLoss`, `PseudoLabelNLL`).

### [Causal DR Uplift Comparison](causal_dr_uplift_comparison.md)

- Shared-budget comparison for doubly-robust causal regression (`dr_ate`, `dr_cate`).
- Includes synthetic uplift and astronomy-style selection-bias scenarios.
- Reports ATE error, CI quality, overlap-rate, and effective sample size diagnostics.

### [Conformal Regression](conformal_regression_example.md)

- Using conformal prediction to obtain prediction intervals with guaranteed coverage.
- Compare against `evaluate_conformal_methods.py` for coverage-vs-width tradeoffs across methods.

### [Evidential Regression](evidential_regression.md)

- Decomposing uncertainty into aleatoric and epistemic components.

### [Imbalanced Regression](imbalanced_regression.md)

- Handling imbalanced datasets in regression tasks.
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
- Includes AURC, rejection-policy risk/coverage, and OOD uncertainty-gap summaries.

### [OOD / Selective Prediction Comparison (Real Data)](ood_selective_prediction_realdata_comparison.md)

- Shared-budget comparison on Diabetes with deterministic covariate-shift OOD split.
- Adds real-data evidence breadth for OOD/selective tradeoffs across DeepEnsemble/MCDropout/`SWAG`/`BNN`.

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
git clone https://github.com/username/torchregress.git
cd torchregress

# Install dependencies
pip install -e '.[examples]'

# Run a specific example
python examples/basic_usage.py
```

For questions or issues with the examples, please [open an issue](https://github.com/username/torchregress/issues) in the repository.
