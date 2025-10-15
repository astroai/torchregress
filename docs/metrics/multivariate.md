# Multivariate Metrics

This page covers metrics for evaluating multivariate regression models.

## `multivariate_rmse`

Computes the root mean squared error over vector outputs.

**Arguments:**

- `y_pred` (torch.Tensor or np.ndarray): The predicted values.
- `y_true` (torch.Tensor or np.ndarray): The ground truth values.
- `reduction` (str, optional): The reduction to apply to the output. Can be one of `'none'`, `'mean'`, or `'sum'`. Defaults to `'mean'`.

**Returns:**

The root mean squared error of the predictions.

**Example:**

```python
import torch
from torchregress.metrics.multivariate import multivariate_rmse

y_pred = torch.randn(100, 10)
y_true = torch.randn(100, 10)

rmse = multivariate_rmse(y_pred, y_true)
```

## `multivariate_mae`

Computes the mean absolute error over vector outputs.

**Arguments:**

- `y_pred` (torch.Tensor or np.ndarray): The predicted values.
- `y_true` (torch.Tensor or np.ndarray): The ground truth values.
- `reduction` (str, optional): The reduction to apply to the output. Can be one of `'none'`, `'mean'`, or `'sum'`. Defaults to `'mean'`.

**Returns:**

The mean absolute error of the predictions.

**Example:**

```python
import torch
from torchregress.metrics.multivariate import multivariate_mae

y_pred = torch.randn(100, 10)
y_true = torch.randn(100, 10)

mae = multivariate_mae(y_pred, y_true)
```
