# Multivariate Metrics

> ← [Decision Metrics](decision.md) | [Censored Metrics](censored.md) →

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
from torchregress.metrics import MultivariateMAE

y_pred = torch.randn(100, 10)
y_true = torch.randn(100, 10)

metric = MultivariateMAE()
mae = metric(y_pred, y_true)
```

---

## `TaskAgnosticCorrelations` (TAC)

Task-Agnostic Correlations (TAC) is a metric designed to evaluate the accuracy of predicted covariance matrices. It evaluates how well the predicted covariance captures target-level correlations by masking one dimension of the target variable $y_i$ at a time and predicting it from the remaining observed dimensions $y_{-i}$ using conditional Gaussian updates.

### Mathematical Definition

For a predicted mean $\hat{y}$ and predicted covariance $\Sigma \in \mathbb{R}^{D \times D}$ at a given sample, let $i \in \{1, \dots, D\}$ be the masked dimension, and let $-i$ denote all other dimensions. The conditional mean update for the masked target $y_i$ given the observed target variables $y_{-i}$ is:

$$\tilde{y}_i = \hat{y}_i + \Sigma_{i, -i} \Sigma_{-i, -i}^{-1} (y_{-i} - \hat{y}_{-i})$$

The TAC error measures the average absolute deviation of these conditionally updated predictions from the true target values:

$$\text{TAC} = \frac{1}{N \cdot D} \sum_{n=1}^N \sum_{i=1}^D |\tilde{y}_{n, i} - y_{n, i}|$$

A lower TAC score indicates that the predicted covariance matrix $\Sigma$ more accurately captures the correlation structure between targets, allowing for better conditional updates.

### Reference Table

| # | Reference |
|:-:|:----------|
| 1 | Shukla et al., ["TIC-TAC: A Framework For Improved Covariance Estimation In Deep Heteroscedastic Regression"](https://arxiv.org/abs/2407.00296) (ICML 2024). |

---

## Comparison Table: When to use which

| Metric | Target Type | Evaluates | Key Benefit |
| :--- | :--- | :--- | :--- |
| `MultivariateRMSE` | Vector/Multivariate | Point prediction error | Standard MSE baseline for vector targets |
| `MultivariateMAE` | Vector/Multivariate | Point prediction error | Outlier-robust baseline for vector targets |
| `TaskAgnosticCorrelations` | Vector + Covariance | Covariance/correlation structure | Measures how well covariance informs conditional predictions |

### Example

Here is a complete, runnable example using `TaskAgnosticCorrelations` and the functional wrapper `task_agnostic_correlations`:

```python
import torch
from torchregress.metrics import TaskAgnosticCorrelations, task_agnostic_correlations

# Batch of 5 samples, target dimension 3
y_pred = torch.randn(5, 3)
y_true = torch.randn(5, 3)

# Symmetric positive-definite covariance matrices [B, D, D]
cov = torch.eye(3).unsqueeze(0).expand(5, -1, -1) + 0.1 * torch.randn(5, 3, 3)
cov = torch.bmm(cov, cov.transpose(1, 2))  # Ensure SPD

# Class-based metric usage
metric = TaskAgnosticCorrelations()
tac_val = metric(y_pred, y_true, cov)
print(f"TAC Metric value: {tac_val.item():.4f}")

# Functional usage
tac_func_val = task_agnostic_correlations(y_pred, y_true, cov)
print(f"TAC Functional value: {tac_func_val.item():.4f}")
```

!!! tip
    To ensure numerical stability when computing $\Sigma_{-i, -i}^{-1}$, a small jitter stabilizer (e.g. $10^{-6} \cdot I$) is automatically added to the diagonal of the sub-covariance matrix.

!!! warning
    TAC is defined only for multivariate targets ($D \ge 2$) because it measures conditional updates across dimensions. Applying it to scalar targets ($D = 1$) will result in errors.

---

## Next steps

- [Point metrics](point.md) — standard univariate baselines to decompose multivariate error per dimension
- [Ensemble metrics](ensemble.md) — uncertainty decomposition for multivariate ensemble predictions
- [TIC-TAC](../methods/algorithms/tictac.md) — Taylor-Induced Covariance for improved multivariate covariance estimation
- [Distributional metrics](distribution.md) — Energy Score as the multivariate generalisation of CRPS
