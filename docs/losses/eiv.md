# Error-in-Variables Losses

Error-in-variables (EiV) losses are designed for regression problems where both the input features (X) and the target variable (y) contain measurement errors or uncertainties. Traditional regression methods assume that only the target variable contains errors, which can lead to biased parameter estimates when the input features are also noisy.

## Mathematical Background

In a standard regression problem, we model:

$$y = f(X) + \epsilon_y$$

where $\epsilon_y$ represents the error in the target variable.

In an error-in-variables model, we acknowledge that our observed inputs $X$ may also contain measurement errors:

$$X = X^* + \epsilon_X$$
$$y = f(X^*) + \epsilon_y$$

where $X^*$ represents the true (unobserved) input values and $\epsilon_X$ represents the error in the input features.

## Available Error-in-Variables Losses

### BaseEIVLoss

```python
class BaseEIVLoss(RegressionLoss)
```

Base class for all Error-in-Variables loss functions, providing common functionality.

**Parameters:**

- `model` (Callable): Model function f(x) that predicts y
- `sigma_x` (float or torch.Tensor): Standard deviation or covariance of feature noise
- `sigma_y` (float or torch.Tensor, optional): Standard deviation or covariance of target noise
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `eps` (float, optional): Small value for numerical stability. Default: 1e-8

### FunctionalEIVLoss

```python
class FunctionalEIVLoss(BaseEIVLoss)
```

Implements the functional approach to errors-in-variables modeling, where the true values are treated as fixed but unknown parameters. It propagates uncertainty from inputs to outputs using a first-order Taylor approximation through model gradients.

**Parameters:**

- `model` (Callable): Model function f(x) that predicts y
- `sigma_x` (float or torch.Tensor): Standard deviation or covariance of feature noise
- `sigma_y` (float or torch.Tensor, optional): Standard deviation or covariance of target noise
- `monte_carlo` (bool, optional): Whether to use Monte Carlo sampling for gradient estimation. Default: False
- `n_samples` (int, optional): Number of MC samples if monte_carlo=True. Default: 20
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `eps` (float, optional): Small value for numerical stability. Default: 1e-8

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the Functional EIV loss

**Example:**

```python
import torch
from torchregress.losses import FunctionalEIVLoss

# Define a simple model
model = lambda x: x[:, 0:1] * 2 + x[:, 1:2]

# Create loss with diagonal covariance
loss_fn = FunctionalEIVLoss(model, sigma_x=torch.tensor([0.2, 0.1]), sigma_y=0.1)

# Generate some data
y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # x_obs in EIV terminology
target = torch.tensor([[4.0], [10.0]])           # y_true in EIV terminology

# Compute loss
loss_value = loss_fn(y_pred, target)
print(f"Loss: {loss_value.item():.4f}")
```

### StructuralEIVLoss

```python
class StructuralEIVLoss(BaseEIVLoss)
```

Implements the structural approach to errors-in-variables modeling, which accounts for correlations between errors in x and y through a cross-covariance matrix.

**Parameters:**

- `model` (Callable): Model function f(x) that predicts y
- `sigma_x` (float or torch.Tensor): Covariance of feature noise
- `sigma_y` (float or torch.Tensor): Covariance of target noise
- `sigma_xy` (torch.Tensor): Cross-covariance between feature and target noise
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `eps` (float, optional): Small value for numerical stability. Default: 1e-8

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the Structural EIV loss

**Example:**

```python
import torch
from torchregress.losses import StructuralEIVLoss

# Define a simple model
model = lambda x: x[:, 0:1] * 2 + x[:, 1:2]

# Create loss with cross-covariance
sigma_x = torch.tensor([[0.04, 0.01], [0.01, 0.01]])  # 2x2 covariance
sigma_y = torch.tensor([0.01])  # 1x1 covariance
sigma_xy = torch.tensor([[0.005, 0.002]])  # 1x2 cross-covariance
loss_fn = StructuralEIVLoss(model, sigma_x, sigma_y, sigma_xy)

# Generate some data
y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # x_obs in EIV terminology
target = torch.tensor([[4.0], [10.0]])           # y_true in EIV terminology

# Compute loss
loss_value = loss_fn(y_pred, target)
print(f"Loss: {loss_value.item():.4f}")
```

### OrthogonalDistanceRegressionLoss

```python
class OrthogonalDistanceRegressionLoss(BaseEIVLoss)
```

Orthogonal Distance Regression (ODR) loss minimizes the orthogonal (perpendicular) distances from data points to the model curve by optimizing latent true x values during the forward pass.

**Parameters:**

- `model` (Callable): Model function f(x) that predicts y
- `sigma_x` (float or torch.Tensor): Standard deviation or covariance of feature noise
- `sigma_y` (float or torch.Tensor): Standard deviation or covariance of target noise
- `learning_rate` (float, optional): Learning rate for the latent x optimization. Default: 0.01
- `max_iterations` (int, optional): Maximum iterations for latent x optimization. Default: 10
- `tolerance` (float, optional): Convergence criterion for optimization. Default: 1e-6
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `eps` (float, optional): Small value for numerical stability. Default: 1e-8

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the ODR loss

**Mathematical Formulation:**

ODR minimizes the weighted sum of squared distances between:
1. The observed inputs and the latent (optimized) inputs
2. The observed outputs and the model predictions using latent inputs

$$\mathcal{L}_{\text{ODR}}(X, y, \hat{X}, f, \Sigma_X, \Sigma_y) = (X - \hat{X})^T \Sigma_X^{-1} (X - \hat{X}) + (y - f(\hat{X}))^T \Sigma_y^{-1} (y - f(\hat{X}))$$

where:
- $\hat{X}$ are the latent true input values (optimized during loss computation)
- $f$ is the regression model
- $\Sigma_X$ and $\Sigma_y$ are the covariance matrices for input and output errors

**Example:**

```python
import torch
from torchregress.losses import OrthogonalDistanceRegressionLoss

# Define a simple model
model = lambda x: x[:, 0:1] * 2 + x[:, 1:2]

# Create loss with equal weighting of x and y errors
sigma_x = torch.tensor([1.0, 1.0])  # Equal uncertainty in both inputs
sigma_y = torch.tensor([1.0])       # Unit uncertainty in output
loss_fn = OrthogonalDistanceRegressionLoss(model, sigma_x, sigma_y)

# Generate some data
y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # x_obs in EIV terminology
target = torch.tensor([[4.0], [10.0]])           # y_true in EIV terminology

# Compute loss
loss_value = loss_fn(y_pred, target)
print(f"Loss: {loss_value.item():.4f}")
```

### EnsembleEIVLoss

```python
class EnsembleEIVLoss(BaseEIVLoss)
```

A simple ensemble approach for handling errors-in-variables by generating multiple perturbed versions of the input, running the model on each, and averaging the predictions before calculating the loss.

**Parameters:**

- `model` (Callable): Model function f(x) that predicts y
- `sigma_x` (float or torch.Tensor): Standard deviation or covariance of feature noise
- `n_samples` (int, optional): Number of perturbed samples to generate. Default: 20
- `perturb_method` (str, optional): Method for perturbing inputs ('gaussian', 'uniform'). Default: 'gaussian'
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `eps` (float, optional): Small value for numerical stability. Default: 1e-8

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the Ensemble EIV loss

**Example:**

```python
import torch
from torchregress.losses import EnsembleEIVLoss

# Define a simple model
model = lambda x: x[:, 0:1] * 2 + x[:, 1:2]

# Create ensemble EIV loss
loss_fn = EnsembleEIVLoss(model, sigma_x=torch.tensor([0.2, 0.1]), n_samples=30)

# Generate some data
y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # x_obs in EIV terminology
target = torch.tensor([[4.0], [10.0]])           # y_true in EIV terminology

# Compute loss
loss_value = loss_fn(y_pred, target)
print(f"Loss: {loss_value.item():.4f}")
```

## Factory Function

### create_eiv_loss

```python
create_eiv_loss(model, loss_type='functional', **kwargs)
```

Factory function to create an error-in-variables loss with the specified parameters.

**Parameters:**

- `model` (Callable): Model function f(x) that predicts y
- `loss_type` (str): Type of EIV loss: 'functional' | 'structural' | 'odr' | 'ensemble'
- `**kwargs`: Additional parameters specific to the chosen loss type

**Example:**

```python
import torch
import torchregress as tr

# Define a simple model
model = lambda x: x[:, 0:1] * 2 + x[:, 1:2]

# Create Functional EIV loss
functional_loss = tr.losses.create_eiv_loss(
    model,
    loss_type='functional',
    sigma_x=torch.tensor([0.2, 0.1]),
    sigma_y=0.1
)

# Create ODR loss
odr_loss = tr.losses.create_eiv_loss(
    model,
    loss_type='odr',
    sigma_x=torch.tensor([1.0, 1.0]),
    sigma_y=torch.tensor([1.0])
)
```

## Applications of EIV Regression

1. **Calibration Problems**:
   - Instrument calibration where both reference standards and measurements contain errors
   - Sensor cross-calibration

2. **Scientific Measurements**:
   - Astronomical observations with measurement uncertainties in multiple variables
   - Physical parameter estimation from experimental data

3. **Medical Research**:
   - Analysis of diagnostic tests where both the gold standard and new test have errors
   - Comparing different measurement methods

4. **Econometrics**:
   - Variables measured with error (e.g., reported income vs. actual income)
   - Panel data with measurement error

## Practical Considerations

1. **Error Variance Specification**:
   - If known, use the actual covariance matrices for `sigma_x` and `sigma_y`
   - If unknown, consider:
     - Using identity matrices for equal error weighting
     - Estimating from repeated measurements
     - Treating variance parameters as hyperparameters to tune

2. **Non-Linear Functions**:
   - For non-linear relationships, use OrthogonalDistanceRegressionLoss or FunctionalEIVLoss
   - Consider computational complexity when choosing between methods

3. **Computational Considerations**:
   - EIV methods are generally more computationally intensive than standard regression
   - For large datasets, consider mini-batch training with appropriate batch sizes
   - Monte Carlo approaches (EnsembleEIVLoss) may require more samples for stable results

4. **Covariance Structures**:
   - For multivariate problems, consider the full covariance structure of errors
   - StructuralEIVLoss can handle correlations between input and output errors

5. **Regularization**:
   - EIV methods can be more sensitive to overfitting
   - Consider adding appropriate regularization to your model
