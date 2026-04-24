# Ensemble Metrics

This page covers metrics for evaluating ensemble models.

## `ensemble_statistics`

Computes the mean and variance of an ensemble's predictions.

**Arguments:**

- `predictions` (torch.Tensor or np.ndarray): The predictions of the ensemble members.
- `dim` (int, optional): The dimension to aggregate over. Defaults to `0`.

**Returns:**

A tuple containing the mean and variance of the ensemble's predictions.

**Example:**

```python
import torch
from torchregress.metrics.ensemble import ensemble_statistics

predictions = torch.randn(5, 100)  # 5 ensemble members, 100 predictions each
mean, variance = ensemble_statistics(predictions)
```

## `uncertainty_decomposition`

Decomposes the uncertainty of an ensemble into epistemic and aleatoric uncertainty.

**Arguments:**

- `means` (torch.Tensor or np.ndarray): The predicted means of the ensemble members.
- `variances` (torch.Tensor or np.ndarray): The predicted aleatoric variances of the ensemble members.
- `dim` (int, optional): The dimension to aggregate over. Defaults to `0`.

**Returns:**

A dictionary with the following keys:

- `mean`: The mean of the ensemble's predictions.
- `epistemic_uncertainty`: The epistemic uncertainty of the ensemble.
- `aleatoric_uncertainty`: The aleatoric uncertainty of the ensemble.
- `total_uncertainty`: The total uncertainty of the ensemble.

**Example:**

```python
import torch
from torchregress.metrics.ensemble import uncertainty_decomposition

means = torch.randn(5, 100)
variances = torch.rand(5, 100)

uncertainty = uncertainty_decomposition(means, variances)
```

## `gaussian_nll_ensemble`

Computes the Gaussian negative log-likelihood for an ensemble's predictions.

**Arguments:**

- `means` (torch.Tensor or np.ndarray): The predicted means of the ensemble members.
- `variances` (torch.Tensor or np.ndarray): The predicted aleatoric variances of the ensemble members.
- `y_true` (torch.Tensor or np.ndarray): The ground truth values.

**Returns:**

The Gaussian negative log-likelihood of the ensemble's predictions.

**Example:**

```python
import torch
from torchregress.metrics.ensemble import gaussian_nll_ensemble

means = torch.randn(5, 100)
variances = torch.rand(5, 100)
y_true = torch.randn(100)

nll = gaussian_nll_ensemble(means, variances, y_true)
```

## `ensemble_interval_bounds`

Computes the symmetric Gaussian prediction intervals for an ensemble's predictions.

**Arguments:**

- `means` (torch.Tensor or np.ndarray): The predicted means of the ensemble members.
- `variances` (torch.Tensor or np.ndarray): The predicted aleatoric variances of the ensemble members.
- `alpha` (float, optional): The significance level for the prediction intervals. Defaults to `0.1`.
- `dim` (int, optional): The dimension to aggregate over. Defaults to `0`.

**Returns:**

A tuple containing the lower and upper bounds of the prediction intervals.

**Example:**

```python
import torch
from torchregress.metrics.ensemble import ensemble_interval_bounds

means = torch.randn(5, 100)
variances = torch.rand(5, 100)

lower, upper = ensemble_interval_bounds(means, variances)
```

## `ensemble_interval_metrics`

Computes the interval score and coverage for an ensemble's predictions.

**Arguments:**

- `means` (torch.Tensor or np.ndarray): The predicted means of the ensemble members.
- `variances` (torch.Tensor or np.ndarray): The predicted aleatoric variances of the ensemble members.
- `y_true` (torch.Tensor or np.ndarray): The ground truth values.
- `alpha` (float, optional): The significance level for the prediction intervals. Defaults to `0.1`.

**Returns:**

A dictionary with the following keys:

- `interval_score`: The interval score of the ensemble's predictions.
- `picp`: The prediction interval coverage probability of the ensemble's predictions.

**Example:**

```python
import torch
from torchregress.metrics.ensemble import ensemble_interval_metrics

means = torch.randn(5, 100)
variances = torch.rand(5, 100)
y_true = torch.randn(100)

metrics = ensemble_interval_metrics(means, variances, y_true)
```
