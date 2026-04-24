# Evaluation Metrics

torchregress provides a comprehensive set of metrics for evaluating regression models, with special focus on uncertainty quantification.

## Metric Categories

### Point Prediction Metrics

Metrics for evaluating point predictions (without uncertainty):

- **Mean Squared Error (MSE)**: Average squared difference
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **Mean Absolute Error (MAE)**: Average absolute difference
- **R²**: Coefficient of determination
- **Robust diagnostics**: median absolute error, Huber loss, trimmed MSE, tail MAE/RMSE

[Learn more about point metrics →](point.md)

### Ordinal Metrics

Metrics for ordered-target predictions:

- **Ordinal Accuracy**: exact class-index match rate
- **Mean Absolute Class Error**: average class-index distance
- **Quadratic Weighted Kappa (QWK)**: agreement metric with stronger penalty for larger ordinal mistakes

[Learn more about ordinal metrics →](ordinal.md)

### Censored Metrics

Metrics for censored and interval-censored targets:

- **Censoring Rate**
- **Observed MAE**
- **Concordance Index**
- **Interval Overlap Rate**

[Learn more about censored metrics →](censored.md)

### Distribution Metrics

Metrics for evaluating full predictive distributions:

- **Negative Log-Likelihood (NLL)**: How well the predicted distribution matches the data
- **Continuous Ranked Probability Score (CRPS)**: Distribution-based generalization of MAE
- **Energy Score**: Multivariate generalization of CRPS
- **Proper Scoring Rules**: Various proper scoring rules for probabilistic predictions

[Learn more about distribution metrics →](distribution.md)

### Interval Metrics

Metrics for evaluating prediction intervals:

- **Prediction Interval Coverage Probability (PICP)**: Fraction of true values within the interval
- **Mean Prediction Interval Width (MPIW)**: Average width of prediction intervals
- **Interval Score**: Combined measure of interval width and coverage

[Learn more about interval metrics →](interval.md)

### Calibration Metrics

Metrics for evaluating model calibration:

- **Expected Calibration Error (ECE)**: Average calibration error across bins
- **Marginal Calibration Error (MCE)**: Predictive marginal CDF mismatch
- **Calibration Score**: Gaussian predictive quantile calibration summary
- **Bias**: Mean prediction bias

[Learn more about calibration metrics →](calibration.md)

### Out-of-Distribution Detection Metrics

Scores for flagging potential OOD or shifted samples:

- **Mahalanobis Distance**: Distance from reference feature statistics
- **Typicality Score**: Predictive-distribution typicality
- **Entropy Score**: Predictive-sample entropy
- **Kernel Density Score**: Similarity to a reference sample

[Learn more about OOD metrics →](ood.md)

### Ensemble Metrics

Metrics for evaluating ensemble models:

- **Ensemble Statistics**: Mean, standard deviation, and variance across members.
- **Uncertainty Decomposition**: Epistemic, aleatoric, and total uncertainty for heteroscedastic ensembles.

[Learn more about ensemble metrics →](ensemble.md)

### Multivariate Metrics

Metrics for evaluating multivariate regression models:

- **Multivariate RMSE**: Vector-output root mean squared error.
- **Multivariate MAE**: Vector-output mean absolute error.

[Learn more about multivariate metrics →](multivariate.md)

## Metric Selection Guide

| If you need to evaluate... | Consider using... |
|---------------------------|-------------------|
| Point prediction accuracy | `rmse`, `mae`, `r2_score` |
| Prediction intervals | `prediction_interval_coverage_probability`, `MeanPredictionIntervalWidth`, `interval_score` |
| Full predictive distributions | `gaussian_nll`, `crps_gaussian`, `energy_score` |
| Model calibration | `expected_calibration_error`, `marginal_calibration_error`, `calibration_score` |
| OOD scoring | `mahalanobis_distance`, `typicality_score`, `entropy_score`, `kernel_density_score` |

For detailed guidance on metric selection and interpretation, see the [practical usage guide](../guide/practical-usage.md).
