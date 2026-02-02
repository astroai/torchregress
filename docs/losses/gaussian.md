# Gaussian Loss Functions

Gaussian loss functions are based on the normal distribution and form the foundation of many regression approaches. These loss functions are effective for problems where the error distribution is expected to be symmetric and well-behaved.

## Mathematical Background

The Gaussian (normal) distribution is defined by its probability density function:

$$p(y|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right)$$

Where:
- $\mu$ is the mean parameter
- $\sigma^2$ is the variance parameter

The negative log-likelihood (NLL) of this distribution is:

$$\mathcal{L}_{\text{NLL}}(y, \mu, \sigma^2) = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y-\mu)^2}{2\sigma^2}$$

For a fixed $\sigma^2$, minimizing this NLL is equivalent to minimizing the mean squared error $(y-\mu)^2$.

## Available Gaussian Losses

### Quick Reference: Which Loss to Use?

| Use Case | Recommended Loss | Model Output |
|----------|-----------------|--------------|
| **Homoscedastic** (constant variance) | `MSELoss` | Single mean value |
| **Heteroscedastic** (learned variance per sample) | `GaussianNLLLoss` | Tuple of (mean, logvar) |
| **Multivariate** (correlated features) | `MultivariateGaussianLoss` | Mean + covariance matrix |

### MSELoss (Homoscedastic)

```python
# Standard weighted MSE Loss
MSELoss = WeightedMSELoss
```

For simple Gaussian losses with **constant variance**, use MSE loss. This is equivalent to Gaussian NLL with fixed variance σ²=1.

**Example:**

```python
import torch
import torchregress as tr

# Mean squared error (fixed variance)
loss_fn = tr.losses.MSELoss()

# Predictions and y_trues
y_pred = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([0.0, 2.0, 4.0])

# Basic MSE loss
basic_loss = loss_fn(y_pred, y_true)  # tensor(0.6667)

# With mask (ignore the 2nd sample)
mask = torch.tensor([True, False, True])
masked_loss = loss_fn(y_pred, y_true, mask=mask)
```

### GaussianNLLLoss (Heteroscedastic)

```python
class GaussianNLLLoss(DistributionLoss)
```

Negative Log-Likelihood loss for diagonal Gaussian distributions where the model predicts both mean and variance (heteroscedastic regression).

**Parameters:**

- `fixed_variance` (float, optional): Fixed variance value when model only predicts mean. Default: `None`
- `min_variance` (float, optional): Minimum variance for numerical stability. Default: `1e-6`
- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `_extract_distribution_parameters(y_pred)`: Extracts mean and variance from predictions
- `_calculate_nll(y_pred, y_true, mask)`: Calculates the negative log-likelihood
- `forward(y_pred, y_true, mask=None, weights=None)`: Computes the Gaussian NLL loss

The full NLL formula used in this implementation is:

$$\mathcal{L}_{\text{DiagGaussNLL}}(y, \mu, \sigma^2) = \frac{1}{2N}\sum_{i=1}^{N}\left(\log(2\pi\sigma_i^2) + \frac{(y_i-\mu_i)^2}{\sigma_i^2}\right)$$

**Example:**

```python
import torch
import torch.nn as nn
import torchregress as tr

# Model that outputs both mean and log-variance
class HeteroscedasticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(5, 32), nn.ReLU())
        self.mean_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)

    def forward(self, x):
        h = self.shared(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return (mean, logvar)  # Return as tuple

# Create model and loss
model = HeteroscedasticModel()
loss_fn = tr.losses.GaussianNLLLoss()

# Training
X = torch.randn(100, 5)
y = torch.randn(100, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(10):
    y_pred = model(X)  # Returns (mean, logvar) tuple
    loss = loss_fn(y_pred, y)  # Clean API: just (predictions, y_trues)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# At inference, extract mean and uncertainty
model.eval()
with torch.no_grad():
    mean, logvar = model(X)
    std = torch.exp(0.5 * logvar)  # Convert log-variance to std

    # 95% prediction intervals
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
```

**Why GaussianNLLLoss?**

- Clean, intuitive API: `loss(y_pred, y)` where `y_pred` is a tuple
- Handles tuple unpacking internally
- Perfect for heteroscedastic regression where uncertainty varies across inputs
- Consistent with torchregress design patterns

**Input Formats:**

The loss accepts multiple input formats:

1. **Tuple** (recommended): `(mean, log_variance)`
2. **Concatenated tensor**: `[..., 2*n_features]` containing `[mean, log_var]`
3. **Mean only**: Just mean predictions (requires `fixed_variance` parameter)

### MultivariateGaussianLoss

```python
class MultivariateGaussianLoss(DistributionLoss)
```

Negative Log-Likelihood loss for multivariate Gaussian with full covariance matrices, which can capture correlations between output dimensions.

**Parameters:**

- `n_features` (int, optional): Number of features (required for `learnable_adjustment`)
- `learnable_adjustment` (bool, optional): Whether to learn feature-specific variance adjustments. Default: `False`
- `jitter` (float, optional): Small value added to diagonal for numerical stability. Default: `1e-6`
- `eps` (float, optional): Small constant for numerical stability in log calculations. Default: `1e-8`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `_extract_distribution_parameters(y_pred, covariance_matrices)`: Extracts mean and covariance
- `_calculate_nll(params, y_true, mask)`: Calculates multivariate Gaussian NLL
- `forward(y_pred, y_true, covariance_matrices, mask=None, weights=None)`: Computes the loss

The multivariate Gaussian NLL is defined as:

$$\mathcal{L}_{\text{MVGaussNLL}}(y, \mu, \Sigma) = \frac{1}{2}\left(\log|\Sigma| + (y-\mu)^T\Sigma^{-1}(y-\mu) + k\log(2\pi)\right)$$

where $k$ is the dimension of the output, $\Sigma$ is the covariance matrix, and $|\Sigma|$ is its determinant.

**Example:**

```python
import torch
import torchregress as tr

# Create the loss function
loss_fn = tr.losses.MultivariateGaussianLoss()

# Predicted means
y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y_true = torch.tensor([[0.0, 2.0], [3.0, 5.0]])

# Case 1: Shared covariance matrix for all samples
cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])  # 2x2 covariance matrix with correlation
loss = loss_fn(y_pred, y_true, cov)

# Case 2: Sample-specific covariance matrices
batch_size = y_pred.shape[0]
n_features = y_pred.shape[1]
# Create a batch of covariance matrices
batch_cov = torch.zeros(batch_size, n_features, n_features)
batch_cov[0] = torch.tensor([[1.0, 0.2], [0.2, 1.0]])
batch_cov[1] = torch.tensor([[2.0, 0.6], [0.6, 1.5]])
loss = loss_fn(y_pred, y_true, batch_cov)
```

## Factory Function

### create_gaussian_nll

```python
create_gaussian_nll(n_features, covariance_type='diagonal', model_predicts_variance=True,
                   fixed_variance=None, use_mse_for_unit_variance=False,
                   jitter=1e-6, reduction='mean', **kwargs)
```

A factory function that creates an appropriate Gaussian NLL loss based on the specified parameters.

**Parameters:**

- `n_features` (int): Number of features
- `covariance_type` (str, optional): One of 'diagonal' or 'full'. Default: 'diagonal'
- `model_predicts_variance` (bool, optional): Whether model predicts variance. Default: `True`
- `fixed_variance` (float, optional): Fixed variance value when model doesn't predict variance. Default: `None`
- `use_mse_for_unit_variance` (bool, optional): If `True` and `fixed_variance == 1.0`, return
  `WeightedMSELoss` instead of Gaussian NLL (changes the optimization objective). Default: `False`
- `jitter` (float, optional): Regularization strength for numerical stability. Default: `1e-6`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `**kwargs`: Additional arguments for specific loss types

**Returns:**

An appropriate Gaussian NLL loss object (`GaussianNLLLoss`, `MultivariateGaussianLoss`,
`LowRankGaussianLoss`, or `WeightedMSELoss` when `use_mse_for_unit_variance=True`)

**Example:**

```python
import torch
import torchregress as tr

# Model predicts (mean, log_var) - default case
loss_fn = tr.losses.create_gaussian_nll(n_features=3)

# Model predicts only mean with fixed variance
loss_fn = tr.losses.create_gaussian_nll(
    n_features=3,
    model_predicts_variance=False,
    fixed_variance=0.5
)

# Optional: preserve the MSE shortcut for unit variance (changes the objective)
loss_fn = tr.losses.create_gaussian_nll(
    n_features=3,
    model_predicts_variance=False,
    fixed_variance=1.0,
    use_mse_for_unit_variance=True
)

# Full covariance case
loss_fn = tr.losses.create_gaussian_nll(
    n_features=3,
    covariance_type='full'
)
```

## Choosing the Right Gaussian Loss

1. **MSELoss**: Simple squared error loss with masking and weighting support. Use when variance is constant
   and known (homoscedastic). `create_gaussian_nll` only returns `WeightedMSELoss` when you set
   `use_mse_for_unit_variance=True` with `fixed_variance=1.0` (which changes the objective).

2. **GaussianNLLLoss**:
   - Use when you want to model uncertainty for each output dimension independently
   - Perfect for heteroscedastic regression where uncertainty varies across the input space
   - Model should predict both mean and log-variance
   - Can use `fixed_variance` if your model only predicts mean but you want NLL semantics

3. **MultivariateGaussianLoss**:
   - Use when output dimensions are correlated (e.g., multivariate time series, spatial data)
   - More complex but can capture richer uncertainty relationships
   - Requires more data to estimate reliably compared to diagonal variants

## Comparison: GaussianNLLLoss vs WeightedGaussianNLLLoss

**GaussianNLLLoss (Recommended):**
- Clean API: `loss((mean, logvar), y_true)`
- Tuple input matches how models naturally output predictions
- Automatic unpacking and processing

**WeightedGaussianNLLLoss:**
- PyTorch wrapper: `loss(mean, y_true, var)` (3 separate arguments)
- Use only for PyTorch API compatibility
- Less convenient for torchregress patterns

## Mathematical Insights

1. **Connection to MSE**: When using fixed variance, minimizing GaussianNLL is equivalent to minimizing MSE.

2. **Uncertainty Calibration**: GaussianNLL naturally balances predictive accuracy with uncertainty estimation - it penalizes both errors in the mean prediction and miscalibrated variances.

3. **The Bias-Variance Tradeoff**:
   - Larger predicted variances reduce the penalty for errors in the mean but increase the log-variance term
   - The loss forces the model to predict larger variances only when there is genuine uncertainty

4. **Implementation Notes**:
   - When using `fixed_variance`, only mean values are predicted by your model
   - `create_gaussian_nll` defaults to `GaussianNLLLoss` even with `fixed_variance` unless you set
     `use_mse_for_unit_variance=True`
   - When `fixed_variance=None` (default), your model should output both mean and log-variance
   - For multivariate cases, eigendecomposition is used as a fallback for numerical stability
