# Metrics API

Complete reference for `torchregress.metrics`. Every metric, class, and report function is listed here. For the conceptual guides, see the [Metrics overview](../metrics/index.md) and the per-category pages.

> **Imports.** `MeanSquaredError`, `MeanAbsoluteError`, and `R2Score` are no longer re-exported from torchregress. Import them directly from `torchmetrics` if you want stateful, epoch-accumulating metrics. The torchregress `mse`, `mae`, `r2_score`, etc. are **functional** wrappers with optional **`sample_weight`** (not `mask`). For masked reductions, apply masks before calling or use loss utilities in `torchregress.utils.reduction`.

---

---

## Point metrics (`metrics.point`)

→ **Guide:** [Point metrics](../metrics/point.md)

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `mse` | `mse(y_pred, y, sample_weight=None)` | Functional MSE |
| `rmse` | `rmse(y_pred, y, sample_weight=None)` | Functional RMSE |
| `mae` | `mae(y_pred, y, sample_weight=None)` | Functional MAE |
| `r2_score` | `r2_score(y_pred, y, as_numpy=False)` | Functional R² |
| `huber_loss` | `huber_loss(y_pred, y, delta=1.0)` | Functional Huber loss |
| `median_absolute_error` | `median_absolute_error(y_pred, y)` | MedAE |
| `median_absolute_deviation` | `median_absolute_deviation(y_pred, y)` | MAD |
| `mean_absolute_percentage_error` | `mean_absolute_percentage_error(y_pred, y)` | MAPE |
| `mean_squared_log_error` | `mean_squared_log_error(y_pred, y)` | MSLE |
| `normalized_rmse` | `normalized_rmse(y_pred, y, normalization='std')` | NRMSE |
| `trimmed_mean_squared_error` | `trimmed_mean_squared_error(y_pred, y, proportion=0.1)` | Trimmed MSE |
| `tail_rmse` | `tail_rmse(y_pred, y, q=0.1, tail='upper')` | Upper-tail RMSE |
| `tail_mae` | `tail_mae(y_pred, y, q=0.1, tail='upper')` | Upper-tail MAE |
| `outlier_fraction` | `outlier_fraction(y_pred, y, threshold=0.15)` | Fraction beyond threshold |
| `mean_squared_error` | `mean_squared_error(y_pred, y, ...)` | Alias for `mse` |
| `mean_absolute_error` | `mean_absolute_error(y_pred, y, ...)` | Alias for `mae` |
| `attenuation_factor` | `attenuation_factor(y_pred, y)` | Regression attenuation factor |
| `regression_metrics_report` | `regression_metrics_report(y_pred, y)` | Aggregate report dict |

**Stateful metric classes** — accumulate over epochs via `update()` / `compute()`:

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `HuberMetric` | `HuberMetric(delta=1.0)` | Stateful Huber loss accumulator |
| `MedianAbsoluteError` | `MedianAbsoluteError()` | Stateful MedAE accumulator |
| `MedianAbsoluteDeviation` | `MedianAbsoluteDeviation()` | Stateful MAD accumulator |
| `NormalizedMedianAbsoluteDeviation` | `NormalizedMedianAbsoluteDeviation()` | Stateful NMAD accumulator |
| `NormalizedRMSE` | `NormalizedRMSE(normalization="std")` | Stateful NRMSE accumulator |
| `OutlierFraction` | `OutlierFraction(threshold=0.15)` | Stateful outlier-fraction accumulator |
| `TrimmedMeanSquaredError` | `TrimmedMeanSquaredError(proportion=0.1)` | Stateful trimmed MSE accumulator |

---

## Distributional metrics (`metrics.distribution`)

→ **Guide:** [Distribution metrics](../metrics/distribution.md)

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `crps_gaussian` | `crps_gaussian(mean, y_true, std, reduction="mean")` | Analytic CRPS for Gaussian (μ, σ) |
| `continuous_ranked_probability_score` | `continuous_ranked_probability_score(quantiles, y_true)` | CRPS from quantile forecasts |
| `crps_from_samples` | `crps_from_samples(samples, y)` | Empirical CRPS from MC samples |
| `energy_score` | `energy_score(y_samples, y_true, beta=1.0)` | Multivariate energy score |
| `gaussian_nll` | `gaussian_nll(mean, y_true, var, reduction="mean")` | Diagonal Gaussian NLL (variance, not log-var) |
| `probability_integral_transform` | `probability_integral_transform(cdf_fn, y_true)` | PIT values via CDF callable |
| `kolmogorov_smirnov_uniform_statistic` | `kolmogorov_smirnov_uniform_statistic(pit)` | KS-uniform on PIT |
| `distribution_metrics_report` | `distribution_metrics_report(dist, y_true)` | Aggregate report |
| `conditional_density_estimation_loss` | `conditional_density_estimation_loss(...)` | Conditional density estimation loss |
| `highest_posterior_density_coverage` | `highest_posterior_density_coverage(...)` | HPD coverage metric |
| `highest_posterior_density_level` | `highest_posterior_density_level(...)` | HPD level computation |

**Stateful distribution metric classes:**

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `ContinuousRankedProbabilityScore` | `ContinuousRankedProbabilityScore()` | Stateful CRPS accumulator (quantile-based) |
| `EnergyScore` | `EnergyScore(beta=1.0)` | Stateful multivariate energy score accumulator |

---

## Interval metrics (`metrics.interval`)

→ **Guide:** [Interval metrics](../metrics/interval.md)

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `interval_score` | `interval_score(lower, upper, target, alpha=0.1)` | Functional Winkler interval score |
| `IntervalScore` | `IntervalScore(alpha=0.1)` | Stateful Winkler score |
| `prediction_interval_coverage` | `prediction_interval_coverage(lower, upper, target)` | Alias for `prediction_interval_coverage_probability` |
| `prediction_interval_coverage_probability` | `prediction_interval_coverage_probability(lower, upper, target)` | Functional PICP |
| `PredictionIntervalCoverageProbability` | `PredictionIntervalCoverageProbability()` | Stateful PICP accumulator |
| `MeanPredictionIntervalWidth` | `MeanPredictionIntervalWidth()` | Stateful MPIW accumulator |
| `interval_metrics_report` | `interval_metrics_report(lower, upper, y, alpha=0.1)` | Aggregate report |

---

## Calibration metrics (`metrics.calibration`)

→ **Guide:** [Calibration metrics](../metrics/calibration.md)

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `expected_calibration_error` | `expected_calibration_error(y_pred_quantiles, target)` | ECE for quantile predictions |
| `ExpectedCalibrationError` | `ExpectedCalibrationError()` | Stateful ECE accumulator |
| `marginal_calibration_error` | `marginal_calibration_error(y_pred_samples, target, n_bins=20)` | Marginal calibration error |
| `MarginalCalibrationError` | `MarginalCalibrationError()` | Stateful MCE accumulator |
| `bias` | `bias(y_pred, target)` | Mean signed error |
| `calibration_score` | `calibration_score(y_true, pred_mean, pred_std)` | Combined calibration score |
| `calibration_metrics_report` | `calibration_metrics_report(y_pred, y_pred_std, y)` | Aggregate report |

---

## Ensemble metrics (`metrics.ensemble`)

→ **Guide:** [Ensemble metrics](../metrics/ensemble.md)

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `ensemble_statistics` | `ensemble_statistics(predictions, dim=0)` | Mean and variance across members |
| `uncertainty_decomposition` | `uncertainty_decomposition(means, variances, dim=0)` | Epistemic/aleatoric/total split |
| `gaussian_nll_ensemble` | `gaussian_nll_ensemble(means, variances, target)` | Ensemble NLL |
| `ensemble_interval_bounds` | `ensemble_interval_bounds(means, variances, alpha=0.1)` | Gaussian prediction interval |
| `ensemble_interval_metrics` | `ensemble_interval_metrics(means, variances, target, alpha=0.1)` | Ensemble PICP + interval score |

**Stateful ensemble metric classes:**

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `GaussianNLLEnsemble` | `GaussianNLLEnsemble()` | Stateful ensemble NLL accumulator |
| `EnsembleIntervalMetrics` | `EnsembleIntervalMetrics(alpha=0.1)` | Stateful ensemble interval metric accumulator |

**Additional ensemble helpers:**

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `ensemble_mean` | `ensemble_mean(predictions, dim=0)` | Mean across ensemble members |
| `ensemble_std` | `ensemble_std(predictions, dim=0)` | Std across ensemble members |
| `ensemble_variance_decomposition` | `ensemble_variance_decomposition(means, variances)` | Epistemic/aleatoric/total split |

---

## OOD metrics (`metrics.ood`)

→ **Guide:** [OOD metrics](../metrics/ood.md)

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `mahalanobis_distance` | `mahalanobis_distance(x, mean, cov)` | Distance to feature centroid |
| `typicality_score` | `typicality_score(model_output, target_x)` | Typicality test |
| `entropy_score` | `entropy_score(samples, n_bins=50)` | Predictive entropy |
| `kernel_density_score` | `kernel_density_score(x_test, x_reference, bandwidth=0.5)` | KDE density score |
| `ood_metrics_report` | `ood_metrics_report(...)` | Aggregate report |

**Stateful OOD metric classes:**

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `EntropyScore` | `EntropyScore(n_bins=50)` | Stateful predictive entropy accumulator |
| `KernelDensityScore` | `KernelDensityScore(bandwidth=0.5)` | Stateful KDE score accumulator |
| `MahalanobisDistance` | `MahalanobisDistance()` | Stateful Mahalanobis distance accumulator |
| `TypicalityScore` | `TypicalityScore()` | Stateful typicality test accumulator |

---

## Decision metrics (`metrics.decision`)

→ **Guide:** [Decision metrics](../metrics/decision.md)

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `risk_coverage_curve` | `risk_coverage_curve(y_pred, y_true, uncertainty, ...)` | Functional risk-coverage curve |
| `RiskCoverageCurve` | `RiskCoverageCurve(n_points=100)` | Stateful risk-coverage curve |
| `RejectionPolicy` | `RejectionPolicy(fraction=0.1)` or `RejectionPolicy(threshold=0.5)` | Stateful rejection policy |

---

## Censored, ordinal, multivariate & other metrics

→ **Guides:** [Censored](../metrics/censored.md) · [Ordinal](../metrics/ordinal.md) · [Multivariate](../metrics/multivariate.md)

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `censoring_rate` | `censoring_rate(censoring_indicator)` | Fraction censored |
| `observed_mae` | `observed_mae(y_pred, y_true, censoring_indicator)` | MAE on uncensored only |
| `concordance_index` | `concordance_index(y_pred, y_true, censoring_indicator)` | Harrell's C-index |
| `interval_overlap_rate` | `interval_overlap_rate(...)` | Predicted interval vs censored interval overlap |
| `ordinal_accuracy` | `ordinal_accuracy(logits, labels, encoding="class_logits")` | Exact class match rate |
| `mean_absolute_class_error` | `mean_absolute_class_error(logits, labels, encoding="class_logits")` | Mean class-index distance |
| `quadratic_weighted_kappa` | `quadratic_weighted_kappa(logits, labels, encoding="class_logits")` | QWK agreement |
| `MultivariateRMSE` | `MultivariateRMSE()` | Stateful vector RMSE |
| `MultivariateMAE` | `MultivariateMAE()` | Stateful vector MAE |
| `TaskAgnosticCorrelations` | `TaskAgnosticCorrelations()` | Stateful TAC metric |
| `task_agnostic_correlations` | `task_agnostic_correlations(y_pred, y_true, cov)` | Functional TAC |

---

## Uncertain GT metrics

→ **Guide:** [Uncertain ground truth](../losses/uncertain_ground_truth.md)

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `noisy_target_gaussian_nll` | `noisy_target_gaussian_nll(...)` | NLL with known target noise |
| `consistency_error` | `consistency_error(...)` | Student-teacher agreement error |
| `pseudo_label_acceptance_rate` | `pseudo_label_acceptance_rate(...)` | Pseudo-label acceptance fraction |
| `uncertain_gt_metrics_report` | `uncertain_gt_metrics_report(...)` | Aggregate report |
