# Evaluation Metrics

TorchRegression provides a comprehensive set of metrics for evaluating regression models, with special focus on uncertainty quantification.

## Metric Categories

### Point Prediction Metrics

Metrics for evaluating point predictions (without uncertainty):

- **Mean Squared Error (MSE)**: Average squared difference
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **Mean Absolute Error (MAE)**: Average absolute difference
- **Mean Absolute Percentage Error (MAPE)**: Percentage-based error
- **R²**: Coefficient of determination
- **Adjusted R²**: R² adjusted for model complexity

[Learn more about point metrics →](point.md)

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
- **Normalized Interval Score**: Interval score normalized for comparison

[Learn more about interval metrics →](interval.md)

### Calibration Metrics

Metrics for evaluating model calibration:

- **Expected Calibration Error (ECE)**: Average calibration error across bins
- **Maximum Calibration Error (MCE)**: Maximum calibration error across bins
- **Sharpness**: Measure of prediction confidence
- **Calibration Curve Statistics**: Metrics derived from reliability diagrams

[Learn more about calibration metrics →](calibration.md)

### Out-of-Distribution Detection Metrics

Metrics for evaluating OOD detection performance:

- **AUROC**: Area under the ROC curve for OOD detection
- **AUPR**: Area under the precision-recall curve
- **FPR at 95% TPR**: False positive rate at 95% true positive rate
- **Detection Error**: Minimum misclassification probability

[Learn more about OOD metrics →](ood.md)

## Metric Selection Guide

| If you need to evaluate... | Consider using... |
|---------------------------|-------------------|
| Point prediction accuracy | `rmse`, `mae`, `r2_score` |
| Prediction intervals | `picp`, `mpiw`, `interval_score` |
| Full predictive distributions | `nll`, `crps` |
| Model calibration | `expected_calibration_error`, `calibration_curve` |
| OOD detection capability | `auroc_ood`, `fpr_at_tpr` |

For detailed guidance on metric selection and interpretation, see the [practical usage guide](../usage/practical_usage.md).
