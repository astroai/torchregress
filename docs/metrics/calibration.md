# Calibration Metrics

Calibration metrics evaluate whether the predicted distributions accurately reflect the true uncertainty in the data.

## Expected Calibration Error (ECE)

Measures the average discrepancy between predicted quantiles and observed proportions.

```python
from torchregress.metrics.calibration import expected_calibration_error

# Using a dictionary of quantile predictions
quantiles = {0.1: q10_pred, 0.5: q50_pred, 0.9: q90_pred}

# Calculate basic ECE metrics
ece_metrics = expected_calibration_error(quantiles, y_true)
print(f"Mean Absolute Calibration Error: {ece_metrics['mean_absolute_calibration_error']}")
print(f"Maximum Calibration Error: {ece_metrics['maximum_calibration_error']}")

# Get detailed diagnostics
ece_detailed = expected_calibration_error(quantiles, y_true, n_bins=10, return_diagnostics=True)
print(f"Expected proportions: {ece_detailed['expected_proportions']}")
print(f"Actual proportions: {ece_detailed['actual_proportions']}")
```

## Marginal Calibration Error (MCE)

Evaluates how well the predictive distribution's marginals match the empirical distribution of observations.

```python
from torchregress.metrics.calibration import marginal_calibration_error

# y_pred_samples has shape [n_samples, batch_size]
mce_metrics = marginal_calibration_error(y_pred_samples, y_true)
print(f"Marginal Calibration Error: {mce_metrics['marginal_calibration_error']}")
print(f"Maximum MCE: {mce_metrics['maximum_marginal_calibration_error']}")

# Get detailed diagnostics
mce_detailed = marginal_calibration_error(y_pred_samples, y_true, n_bins=20, return_diagnostics=True)
```

## Comprehensive Reporting

### Calibration Metrics Report

Generate a comprehensive report of calibration metrics.

```python
from torchregress.metrics.calibration import calibration_metrics_report
from torch.distributions import Normal

# Using a PyTorch distribution
dist = Normal(mean_pred, std_pred)
report = calibration_metrics_report(dist, y_true)

# Using samples from predictive distribution
report = calibration_metrics_report(y_pred_samples, y_true)

# Using quantiles with custom levels
quantiles = {0.1: q10_pred, 0.5: q50_pred, 0.9: q90_pred}
report = calibration_metrics_report(
    {}, y_true, y_pred_quantiles=quantiles,
    quantile_levels=[0.1, 0.5, 0.9]
)

print(f"Mean Absolute Calibration Error: {report['mean_absolute_calibration_error']}")
print(f"Marginal Calibration Error: {report.get('marginal_calibration_error', 'Not computed')}")
```

## Visual Calibration Assessment

While not built into the metrics directly, the data from these functions can be used for visual assessment:

```python
import matplotlib.pyplot as plt

# Get calibration data
ece_detailed = expected_calibration_error(quantiles, y_true, n_bins=10, return_diagnostics=True)

# Plot reliability diagram
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
plt.plot(ece_detailed['expected_proportions'], ece_detailed['actual_proportions'], 'o-', label='Model')
plt.xlabel('Expected proportion')
plt.ylabel('Observed proportion')
plt.title('Reliability Diagram')
plt.legend()
plt.grid(True)
```
