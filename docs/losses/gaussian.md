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

### MSELoss

```python
class MSELoss(RegressionLoss)
```

Mean Squared Error loss with support for masking and weighting. This is a wrapper around PyTorch's F.mse_loss with additional functionality.

**Parameters:**

- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the MSE loss

**Example:**

```python
import torch
import torchregression as tr

loss_fn = tr.losses.MSELoss()

# Predictions and targets
y_pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([0.0, 2.0, 4.0])

# Basic MSE loss
basic_loss = loss_fn(y_pred, target)  # tensor(0.6667)

# With mask (ignore the 2nd sample)
mask = torch.tensor([True, False, True])
masked_loss = loss_fn(y_pred, target, mask=mask)
```

### MAELoss

```python
class MAELoss(RegressionLoss)
```

Mean Absolute Error loss with support for masking and weighting. This is a wrapper around PyTorch's F.l1_loss with additional functionality.

**Parameters:**

- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the MAE loss

**Example:**

```python
import torch
import torchregression as tr

loss_fn = tr.losses.MAELoss()

# Predictions and targets
y_pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([0.0, 2.0, 4.0])

# Basic MAE loss
loss = loss_fn(y_pred, target)  # tensor(0.6667)
```

### WeightedMSELoss

```python
class WeightedMSELoss(RegressionLoss)
```

An extension of the standard Mean Squared Error loss with support for sample/feature weighting and masking.

**Parameters:**

- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the weighted MSE loss

MSE is defined mathematically as:

$$\mathcal{L}_{\text{MSE}}(y, \hat{y}) = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

With weights, this becomes:

$$\mathcal{L}_{\text{WeightedMSE}}(y, \hat{y}, w) = \frac{\sum_{i=1}^{N}w_i(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}w_i}$$

**Example:**

```python
import torch
import torchregression as tr

loss_fn = tr.losses.WeightedMSELoss()

# Predictions and targets
y_pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([0.0, 2.0, 4.0])

# Basic MSE loss
basic_loss = loss_fn(y_pred, target)  # tensor(0.6667)

# With sample weights
weights = torch.tensor([0.5, 1.0, 2.0])  # Emphasize the 3rd sample
weighted_loss = loss_fn(y_pred, target, weights=weights)  # Gives more weight to the 3rd error

# With mask (ignore the 2nd sample)
mask = torch.tensor([True, False, True])
masked_loss = loss_fn(y_pred, target, mask=mask)
```

### GaussianNLLLoss

```python
class GaussianNLLLoss(DistributionLoss)
```

Gaussian Negative Log-Likelihood loss with explicit variance input. This simpler interface is useful when you have a separate model or estimate for variance.

**Parameters:**

- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, var, mask=None, weights=None)`: Computes the Gaussian NLL loss

**Example:**

```python
import torch
import torchregression as tr

# Create loss function
loss_fn = tr.losses.GaussianNLLLoss()

# Predictions, targets and variance
y_pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([0.0, 2.0, 4.0])
var = torch.tensor([0.5, 1.0, 2.0])  # Different uncertainty for each prediction

# Calculate NLL loss
loss = loss_fn(y_pred, target, var)
```

### DiagonalGaussianNLL

```python
class DiagonalGaussianNLL(DistributionLoss)
```

Negative Log-Likelihood loss for diagonal Gaussian distributions, which models each output dimension with an independent Gaussian distribution.

**Parameters:**

- `n_features` (int, optional): Number of output features (required when `learnable_variance=True`)
- `learnable_variance` (bool, optional): Whether to use learnable variance parameters. Default: `True`
- `fixed_variance` (float, optional): Fixed variance value when `learnable_variance=False`. Default: `1.0`
- `min_variance` (float, optional): Minimum variance for numerical stability. Default: `1e-6`
- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `_extract_distribution_parameters(y_pred)`: Extracts mean and variance from predictions
- `_calculate_nll(y_pred, target, mask)`: Calculates the negative log-likelihood
- `forward(y_pred, target, mask=None, weights=None)`: Computes the Gaussian NLL loss

The full NLL formula used in this implementation is:

$$\mathcal{L}_{\text{DiagGaussNLL}}(y, \mu, \sigma^2) = \frac{1}{2N}\sum_{i=1}^{N}\left(\log(2\pi\sigma_i^2) + \frac{(y_i-\mu_i)^2}{\sigma_i^2}\right)$$

**Example:**

```python
import torch
import torchregression as tr

# Case 1: Learnable variance parameters
loss_fn = tr.losses.DiagonalGaussianNLL(n_features=2, learnable_variance=True)
y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # Just mean predictions
target = torch.tensor([[0.0, 2.0], [3.0, 5.0]])
loss = loss_fn(y_pred, target)

# Case 2: Model predicts both mean and variance
loss_fn = tr.losses.DiagonalGaussianNLL(learnable_variance=False)
mean = torch.tensor([[1.0, 2.0]])
log_var = torch.tensor([[-1.0, 0.0]])  # log(0.368), log(1.0)
loss = loss_fn((mean, log_var), target[0:1])
```

### GaussianNLLWithCovariance

```python
class GaussianNLLWithCovariance(DistributionLoss)
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
- `_calculate_nll(params, target, mask)`: Calculates multivariate Gaussian NLL 
- `forward(y_pred, target, covariance_matrices, mask=None, weights=None)`: Computes the loss

The multivariate Gaussian NLL is defined as:

$$\mathcal{L}_{\text{MVGaussNLL}}(y, \mu, \Sigma) = \frac{1}{2}\left(\log|\Sigma| + (y-\mu)^T\Sigma^{-1}(y-\mu) + k\log(2\pi)\right)$$

where $k$ is the dimension of the output, $\Sigma$ is the covariance matrix, and $|\Sigma|$ is its determinant.

**Example:**

```python
import torch
import torchregression as tr

# Create the loss function
loss_fn = tr.losses.GaussianNLLWithCovariance()

# Predicted means
y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
target = torch.tensor([[0.0, 2.0], [3.0, 5.0]])

# Case 1: Shared covariance matrix for all samples
cov = torch.tensor([[1.0, 0.5], [0.5, 2.0]])  # 2x2 covariance matrix with correlation
loss = loss_fn(y_pred, target, cov)

# Case 2: Sample-specific covariance matrices
batch_size = y_pred.shape[0]
n_features = y_pred.shape[1]
# Create a batch of covariance matrices
batch_cov = torch.zeros(batch_size, n_features, n_features)
batch_cov[0] = torch.tensor([[1.0, 0.2], [0.2, 1.0]])
batch_cov[1] = torch.tensor([[2.0, 0.6], [0.6, 1.5]])
loss = loss_fn(y_pred, target, batch_cov)
```

## Factory Function

### create_gaussian_nll

```python
create_gaussian_nll(n_features, covariance_type='diagonal', learnable_variance=True, 
                   fixed_variance=1.0, jitter=1e-6, reduction='mean', **kwargs)
```

A factory function that creates an appropriate Gaussian NLL loss based on the specified parameters.

**Parameters:**

- `n_features` (int): Number of features
- `covariance_type` (str, optional): One of 'diagonal' or 'full'. Default: 'diagonal'
- `learnable_variance` (bool, optional): Whether to learn variance parameters. Default: `True`
- `fixed_variance` (float, optional): Fixed variance value when not learning. Default: `1.0`
- `jitter` (float, optional): Regularization strength for numerical stability. Default: `1e-6`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `**kwargs`: Additional arguments for specific loss types

**Returns:**

An appropriate Gaussian NLL loss object (`DiagonalGaussianNLL` or `GaussianNLLWithCovariance`)

**Example:**

```python
import torch
import torchregression as tr

# Create a diagonal Gaussian NLL with fixed variance
loss_fn = tr.losses.create_gaussian_nll(
    n_features=3, 
    covariance_type='diagonal',
    learnable_variance=False, 
    fixed_variance=0.5
)

# Create a full-covariance Gaussian NLL with learnable adjustment
loss_fn = tr.losses.create_gaussian_nll(
    n_features=3, 
    covariance_type='full',
    learnable_variance=True, 
    jitter=1e-5
)
```

## Choosing the Right Gaussian Loss

1. **MSELoss**: Simple squared error loss with masking and weighting support.

2. **MAELoss**: Mean absolute error loss when outliers should have less influence.

3. **WeightedMSELoss**: Use when you need to weight samples or features differently.

4. **GaussianNLLLoss**: Use when you have separate models for mean and variance.

5. **DiagonalGaussianNLL**:
   - Use when you want to model uncertainty for each output dimension independently
   - Good for heteroscedastic regression where uncertainty varies across the input space
   - Choose `learnable_variance=True` if your dataset has a constant but unknown noise level
   - Choose `learnable_variance=False` if your model should predict both mean and variance

6. **GaussianNLLWithCovariance**:
   - Use when output dimensions are correlated (e.g., multivariate time series, spatial data)
   - More complex but can capture richer uncertainty relationships
   - Requires more data to estimate reliably compared to diagonal variants

## Mathematical Insights

1. **Connection to MSE**: When using fixed variance, minimizing GaussianNLL is equivalent to minimizing MSE.

2. **Uncertainty Calibration**: GaussianNLL naturally balances predictive accuracy with uncertainty estimation - it penalizes both errors in the mean prediction and miscalibrated variances.

3. **The Bias-Variance Tradeoff**:
   - Larger predicted variances reduce the penalty for errors in the mean but increase the log-variance term
   - The loss forces the model to predict larger variances only when there is genuine uncertainty
