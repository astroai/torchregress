# Point Prediction Metrics

Point prediction metrics evaluate the accuracy of single-point predictions without considering uncertainty information.

## Basic Metrics

### Mean Squared Error (MSE)

The average of squared differences between predictions and targets.

```python
from torchregress.metrics.point import mean_squared_error

mse = mean_squared_error(y_pred, y_true)
```

### Mean Absolute Error (MAE)

The average of absolute differences between predictions and targets.

```python
from torchregress.metrics.point import mean_absolute_error

mae = mean_absolute_error(y_pred, y_true)
```

### Root Mean Squared Error (RMSE)

The square root of the mean squared error.

```python
from torchregress.metrics.point import mean_squared_error
import torch

mse = mean_squared_error(y_pred, y_true)
rmse = torch.sqrt(torch.tensor(mse))
```

### Mean Absolute Percentage Error (MAPE)

The average percentage difference between predictions and targets.

```python
from torchregress.metrics.point import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_pred, y_true)
```

### R² (Coefficient of Determination)

Measures the proportion of variance in the target that is predictable from the model.

```python
from torchregress.metrics.point import r2_score

r2 = r2_score(y_pred, y_true)
```

### Explained Variance Score

Measures the proportion of variance explained by the model.

```python
from torchregress.metrics.point import explained_variance_score

explained_var = explained_variance_score(y_pred, y_true)
```

### Mean Squared Log Error (MSLE)

Mean squared error after log transformation, useful for data with exponential trends.

```python
from torchregress.metrics.point import mean_squared_log_error

# Note: Only works with strictly positive values
msle = mean_squared_log_error(y_pred, y_true)
```

## Robust Metrics

### Median Absolute Error

Median of absolute differences, robust to outliers.

```python
from torchregress.metrics.point import median_absolute_error

median_ae = median_absolute_error(y_pred, y_true)
```

### Huber Loss

Combines MSE and MAE to balance sensitivity to outliers.

```python
from torchregress.metrics.point import huber_loss

# delta controls the transition point from MSE to MAE
hl = huber_loss(y_pred, y_true, delta=1.0)
```

### Trimmed Mean Squared Error

MSE after removing extreme values, robust to outliers.

```python
from torchregress.metrics.point import trimmed_mean_squared_error

# trim 10% of data from both ends
tmse = trimmed_mean_squared_error(y_pred, y_true, proportion=0.1)
```

### Median Absolute Deviation (MAD)

Median of absolute deviations from the median error, highly robust to outliers.

```python
from torchregress.metrics.point import median_absolute_deviation

mad = median_absolute_deviation(y_pred, y_true)
```

### Normalized RMSE

RMSE normalized by a scale factor for comparability across datasets.

```python
from torchregress.metrics.point import normalized_rmse

# Normalize by the standard deviation of the true values
nrmse = normalized_rmse(y_pred, y_true, normalization='std')

# Other normalization options include 'range', 'mean', and 'iqr'
nrmse_range = normalized_rmse(y_pred, y_true, normalization='range')
```

### Normalized Median Absolute Deviation

MAD normalized by the median or by a shifted-relative error scheme for non-negative targets.

```python
from torchregress.metrics.point import normalized_median_absolute_deviation

nmad = normalized_median_absolute_deviation(y_pred, y_true)
nmad_relative = normalized_median_absolute_deviation(y_pred, y_true, normalization="relative")
```

## Application-Specific Metrics

### Outlier Fraction

Fraction of predictions with errors exceeding a threshold.

```python
from torchregress.metrics.point import outlier_fraction

# Standard outlier detection (scaled by std of y_true)
of = outlier_fraction(y_pred, y_true, threshold=0.15)

# Shifted-relative outlier detection (scaled by 1 + y_true, useful for non-negative targets)
of_rel = outlier_fraction(y_pred, y_true, threshold=0.15, mode="relative")
```

### Tail Metrics

Evaluate performance specifically on rare/long-tail target regions:

```python
from torchregress.metrics import tail_mae, tail_rmse

# Top 10% target values
mae_tail = tail_mae(y_pred, y_true, quantile=0.9, tail="upper")
rmse_tail = tail_rmse(y_pred, y_true, quantile=0.9, tail="upper")
```

## Comprehensive Reporting

### Regression Metrics Report

Generate a comprehensive report of multiple metrics.

```python
from torchregress.metrics.point import regression_metrics_report

report = regression_metrics_report(y_pred, y_true)
print(f"MSE: {report['mse']}, MAE: {report['mae']}, R²: {report['r2']}")

# Include robust metrics
report = regression_metrics_report(y_pred, y_true, include_robust=True)

# Include outlier metrics
report = regression_metrics_report(y_pred, y_true, include_outliers=True)
```
