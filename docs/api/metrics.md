# Metrics API Reference

This page documents the evaluation metrics available in TorchRegression.

## Point Prediction Metrics

### `mse`

```python
def mse(y_pred, y_true, mask=None)
```

Compute Mean Squared Error.

**Parameters:**

- `y_pred` (torch.Tensor): The predicted values
- `y_true` (torch.Tensor): The target values
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None

**Returns:**

- torch.Tensor: The computed MSE value

**Example:**

```python
mse_value = tr.metrics.mse(y_pred, y_true)
```

### `rmse`

```python
def rmse(y_pred, y_true, mask=None)
```

Compute Root Mean Squared Error.

**Parameters:**

- `y_pred` (torch.Tensor): The predicted values
- `y_true` (torch.Tensor): The target values
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None

**Returns:**

- torch.Tensor: The computed RMSE value

**Example:**

```python
rmse_value = tr.metrics.rmse(y_pred, y_true)
```

### `mae`

```python
def mae(y_pred, y_true, mask=None)
```

Compute Mean Absolute Error.

**Parameters:**

- `y_pred` (torch.Tensor): The predicted values
- `y_true` (torch.Tensor): The target values
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None

**Returns:**

- torch.Tensor: The computed MAE value

**Example:**

```python
mae_value = tr.metrics.mae(y_pred, y_true)
```

### `mape`

```python
def mape(y_pred, y_true, mask=None, epsilon=1e-8)
```

Compute Mean Absolute Percentage Error.

**Parameters:**

- `y_pred` (torch.Tensor): The predicted values
- `y_true` (torch.Tensor): The target values
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None
- `epsilon` (float, optional): Small constant to avoid division by zero. Default: 1e-8

**Returns:**

- torch.Tensor: The computed MAPE value

**Example:**

```python
mape_value = tr.metrics.mape(y_pred, y_true)
```

### `r2_score`

```python
def r2_score(y_pred, y_true, mask=None)
```

Compute R² (coefficient of determination).

**Parameters:**

- `y_pred` (torch.Tensor): The predicted values
- `y_true` (torch.Tensor): The target values
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None

**Returns:**

- torch.Tensor: The computed R² value

**Example:**

```python
r2 = tr.metrics.r2_score(y_pred, y_true)
```

### `adjusted_r2_score`

```python
def adjusted_r2_score(y_pred, y_true, n_features, mask=None)
```

Compute adjusted R², which accounts for the number of features.

**Parameters:**

- `y_pred` (torch.Tensor): The predicted values
- `y_true` (torch.Tensor): The target values
- `n_features` (int): Number of features/predictors used in the model
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None

**Returns:**

- torch.Tensor: The computed adjusted R² value

**Example:**

```python
adj_r2 = tr.metrics.adjusted_r2_score(y_pred, y_true, n_features=10)
```

## Distribution Metrics

### `gaussian_nll`

```python
def gaussian_nll(y_pred, y_true, y_var, mask=None, full=False, eps=1e-6)
```

Compute Gaussian Negative Log-Likelihood.

**Parameters:**

- `y_pred` (torch.Tensor): The predicted mean values
- `y_true` (torch.Tensor): The target values
- `y_var` (torch.Tensor): The predicted variance values
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None
- `full` (bool, optional): If True, include the constant term. Default: False
- `eps` (float, optional): Small constant for numerical stability. Default: 1e-6

**Returns:**

- torch.Tensor: The computed NLL value

**Example:**

```python
nll = tr.metrics.gaussian_nll(mean, y_true, variance)
```

### `crps_gaussian`

```python
def crps_gaussian(y_pred, y_true, y_std, mask=None)
```

Compute Continuous Ranked Probability Score for Gaussian distributions.

**Parameters:**

- `y_pred` (torch.Tensor): The predicted mean values
- `y_true` (torch.Tensor): The target values
- `y_std` (torch.Tensor): The predicted standard deviation values
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None

**Returns:**

- torch.Tensor: The computed CRPS value

**Example:**

```python
crps = tr.metrics.crps_gaussian(mean, y_true, std)
```

### `energy_score`

```python
def energy_score(samples, y_true, mask=None)
```

Compute Energy Score for multivariate distributions.

**Parameters:**

- `samples` (torch.Tensor): Samples from the predictive distribution of shape (n_samples, batch_size, dim)
- `y_true` (torch.Tensor): The target values of shape (batch_size, dim)
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None

**Returns:**

- torch.Tensor: The computed Energy Score

**Example:**

```python
energy = tr.metrics.energy_score(samples, y_true)
```

## Interval Metrics

### `picp`

```python
def picp(y_true, y_lower, y_upper, mask=None)
```

Compute Prediction Interval Coverage Probability.

**Parameters:**

- `y_true` (torch.Tensor): The target values
- `y_lower` (torch.Tensor): The lower bound of the prediction interval
- `y_upper` (torch.Tensor): The upper bound of the prediction interval
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None

**Returns:**

- torch.Tensor: The computed PICP value

**Example:**

```python
picp_value = tr.metrics.picp(y_true, lower, upper)
```

### `mpiw`

```python
def mpiw(y_lower, y_upper, mask=None, normalized=False, y_true=None)
```

Compute Mean Prediction Interval Width.

**Parameters:**

- `y_lower` (torch.Tensor): The lower bound of the prediction interval
- `y_upper` (torch.Tensor): The upper bound of the prediction interval
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None
- `normalized` (bool, optional): If True, normalize by the range of y_true. Default: False
- `y_true` (torch.Tensor, optional): The target values, required if normalized=True. Default: None

**Returns:**

- torch.Tensor: The computed MPIW value

**Example:**

```python
mpiw_value = tr.metrics.mpiw(lower, upper)
```

### `interval_score`

```python
def interval_score(y_true, y_lower, y_upper, alpha=0.1, mask=None)
```

Compute Interval Score, a proper scoring rule for interval forecasts.

**Parameters:**

- `y_true` (torch.Tensor): The target values
- `y_lower` (torch.Tensor): The lower bound of the prediction interval
- `y_upper` (torch.Tensor): The upper bound of the prediction interval
- `alpha` (float, optional): The significance level (e.g., 0.1 for 90% intervals). Default: 0.1
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None

**Returns:**

- torch.Tensor: The computed interval score

**Example:**

```python
int_score = tr.metrics.interval_score(y_true, lower, upper, alpha=0.1)
```

## Calibration Metrics

### `expected_calibration_error`

```python
def expected_calibration_error(y_pred, y_true, y_std, n_bins=10, mask=None)
```

Compute Expected Calibration Error for regression.

**Parameters:**

- `y_pred` (torch.Tensor): The predicted mean values
- `y_true` (torch.Tensor): The target values
- `y_std` (torch.Tensor): The predicted standard deviation values
- `n_bins` (int, optional): Number of bins for calibration. Default: 10
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None

**Returns:**

- torch.Tensor: The computed ECE value
- torch.Tensor: Confidence levels for each bin
- torch.Tensor: Accuracy levels for each bin
- torch.Tensor: Count of samples in each bin

**Example:**

```python
ece, conf, acc, counts = tr.metrics.expected_calibration_error(mean, y_true, std)
```

### `calibration_curve`

```python
def calibration_curve(y_pred, y_true, y_std, n_bins=10, mask=None)
```

Compute the calibration curve for regression.

**Parameters:**

- `y_pred` (torch.Tensor): The predicted mean values
- `y_true` (torch.Tensor): The target values
- `y_std` (torch.Tensor): The predicted standard deviation values
- `n_bins` (int, optional): Number of bins for calibration. Default: 10
- `mask` (torch.Tensor, optional): A mask for valid values. Default: None

**Returns:**

- torch.Tensor: The confidence levels for each bin
- torch.Tensor: The observed fraction of points in each bin
- torch.Tensor: Count of samples in each bin

**Example:**

```python
conf, obs_frac, counts = tr.metrics.calibration_curve(mean, y_true, std)
```

## OOD Detection Metrics

### `auroc_ood`

```python
def auroc_ood(uncertainty, is_ood)
```

Compute Area Under the ROC Curve for OOD detection.

**Parameters:**

- `uncertainty` (torch.Tensor): Model's uncertainty estimates
- `is_ood` (torch.Tensor): Binary labels indicating whether samples are OOD (1) or in-distribution (0)

**Returns:**

- torch.Tensor: The computed AUROC value

**Example:**

```python
auroc = tr.metrics.auroc_ood(uncertainty, is_ood)
```

### `fpr_at_tpr`

```python
def fpr_at_tpr(uncertainty, is_ood, tpr_target=0.95)
```

Compute False Positive Rate at a target True Positive Rate.

**Parameters:**

- `uncertainty` (torch.Tensor): Model's uncertainty estimates
- `is_ood` (torch.Tensor): Binary labels indicating whether samples are OOD (1) or in-distribution (0)
- `tpr_target` (float, optional): Target true positive rate. Default: 0.95

**Returns:**

- torch.Tensor: The computed FPR at target TPR

**Example:**

```python
fpr = tr.metrics.fpr_at_tpr(uncertainty, is_ood, tpr_target=0.95)
```

### `detection_error`

```python
def detection_error(uncertainty, is_ood)
```

Compute minimum detection error for OOD detection.

**Parameters:**

- `uncertainty` (torch.Tensor): Model's uncertainty estimates
- `is_ood` (torch.Tensor): Binary labels indicating whether samples are OOD (1) or in-distribution (0)

**Returns:**

- torch.Tensor: The minimum detection error
- torch.Tensor: The threshold that achieves this error

**Example:**

```python
min_error, threshold = tr.metrics.detection_error(uncertainty, is_ood)
```
