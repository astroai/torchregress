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

- `forward(y_pred, y_true, mask=None, weights=None)`: Computes the Functional EIV loss

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
y_true = torch.tensor([[4.0], [10.0]])           # y_true in EIV terminology

# Compute loss
loss_value = loss_fn(y_pred, y_true)
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

- `forward(y_pred, y_true, mask=None, weights=None)`: Computes the Structural EIV loss

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
y_true = torch.tensor([[4.0], [10.0]])           # y_true in EIV terminology

# Compute loss
loss_value = loss_fn(y_pred, y_true)
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

- `forward(y_pred, y_true, mask=None, weights=None)`: Computes the ODR loss

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
y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # x_obs in EIV terminology
y_true = torch.tensor([[4.0], [10.0]])           # y_true in EIV terminology

# Compute loss
loss_value = loss_fn(y_pred, y_true)
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

- `forward(y_pred, y_true, mask=None, weights=None)`: Computes the Ensemble EIV loss

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
y_true = torch.tensor([[4.0], [10.0]])           # y_true in EIV terminology

# Compute loss
loss_value = loss_fn(y_pred, y_true)
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

## Decision Guide: Which EIV Method?

```
┌─ Do you have measurement errors in X? ──────────────────┐
│                                                          │
│  Only errors in Y (traditional regression)?             │
│  └─ No → Use standard regression losses (MSE, etc.)    │
│                                                          │
│  Errors in both X and Y?                                 │
│  └─ Yes → Continue below                                 │
│                                                          │
│  Do you know error covariances precisely?                │
│  ├─ Yes → Continue below                                 │
│  └─ No → Start with FunctionalEIVLoss (robust)         │
│                                                          │
│  Do X and Y errors correlate?                            │
│  ├─ Yes → StructuralEIVLoss (accounts for correlation) │
│  └─ No → Continue below                                  │
│                                                          │
│  Is your model differentiable?                           │
│  ├─ Yes → FunctionalEIVLoss (gradient-based)           │
│  └─ No → EnsembleEIVLoss (sampling-based)              │
│                                                          │
│  Need classical ODR solution?                            │
│  └─ Yes → OrthogonalDistanceRegressionLoss             │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Method Comparison

| Method | Best For | Computational Cost | Requires Gradients | Handles Correlation |
|--------|----------|-------------------|-------------------|-------------------|
| **FunctionalEIVLoss** | General use, differentiable models | Medium | Yes | No |
| **StructuralEIVLoss** | Correlated X-Y errors | Medium | Yes | Yes |
| **OrthogonalDistanceRegressionLoss** | Classical ODR, iterative optimization | High | Yes | Yes |
| **EnsembleEIVLoss** | Non-differentiable models, quick baseline | High (sampling) | No | No |

## Complete Example: Calibration Problem

```python
import torch
import torch.nn as nn
from torchregress.losses import FunctionalEIVLoss, EnsembleEIVLoss
import matplotlib.pyplot as plt

# Scenario: Calibrating a sensor against a reference
# Both sensor and reference have measurement errors

# Generate true relationship: y_true = 2.0 * x_true + 1.0
torch.manual_seed(42)
n_samples = 200

# True values
x_true = torch.linspace(0, 10, n_samples).unsqueeze(1)
y_true = 2.0 * x_true + 1.0

# Add measurement noise to both X and Y
sigma_x = 0.5  # Sensor noise std
sigma_y = 0.3  # Reference noise std

x_observed = x_true + sigma_x * torch.randn_like(x_true)
y_observed = y_true + sigma_y * torch.randn_like(y_true)

# Split data
train_size = 150
X_train = x_observed[:train_size]
y_train = y_observed[:train_size]
X_test = x_observed[train_size:]
y_test = y_observed[train_size:]

# Define calibration model
class CalibrationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# Method 1: Standard regression (ignores X errors)
print("=== Standard Regression (Baseline) ===")
model_standard = CalibrationModel()
optimizer_std = torch.optim.Adam(model_standard.parameters(), lr=0.01)
mse_loss = nn.MSELoss()

for epoch in range(100):
    optimizer_std.zero_grad()
    y_pred = model_standard(X_train)
    loss = mse_loss(y_pred, y_train)
    loss.backward()
    optimizer_std.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Method 2: EIV regression (accounts for X errors)
print("\n=== EIV Regression ===")
model_eiv = CalibrationModel()
optimizer_eiv = torch.optim.Adam(model_eiv.parameters(), lr=0.01)

# Create EIV loss with known error variances
eiv_loss = FunctionalEIVLoss(
    model=model_eiv,
    sigma_x=sigma_x,
    sigma_y=sigma_y,
    monte_carlo=False  # Use gradient-based estimation
)

for epoch in range(100):
    optimizer_eiv.zero_grad()
    # Note: In EIV, y_pred is actually x_observed
    loss = eiv_loss(X_train, y_train)
    loss.backward()
    optimizer_eiv.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Evaluate both methods
print("\n=== Evaluation on Test Set ===")

model_standard.eval()
model_eiv.eval()

with torch.no_grad():
    # Standard method
    y_pred_std = model_standard(X_test)
    mae_std = torch.abs(y_pred_std - y_test).mean()

    # EIV method
    y_pred_eiv = model_eiv(X_test)
    mae_eiv = torch.abs(y_pred_eiv - y_test).mean()

    print(f"Standard Regression MAE: {mae_std:.4f}")
    print(f"EIV Regression MAE:      {mae_eiv:.4f}")
    print(f"Improvement:             {(mae_std - mae_eiv)/mae_std * 100:.1f}%")

# Visualize results
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(x_observed[:train_size], y_observed[:train_size],
           alpha=0.3, label='Observed (train)')
plt.scatter(x_true[:train_size], y_true[:train_size],
           alpha=0.3, marker='x', label='True values')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Training Data with Errors')

plt.subplot(132)
x_plot = torch.linspace(0, 10, 100).unsqueeze(1)
with torch.no_grad():
    y_plot_std = model_standard(x_plot)
    y_plot_eiv = model_eiv(x_plot)

plt.plot(x_plot, y_plot_std, 'r-', label='Standard', linewidth=2)
plt.plot(x_plot, y_plot_eiv, 'b-', label='EIV', linewidth=2)
plt.plot(x_plot, 2.0 * x_plot + 1.0, 'k--', label='True', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Fitted Models')

plt.subplot(133)
errors_std = torch.abs(y_pred_std - y_test).numpy()
errors_eiv = torch.abs(y_pred_eiv - y_test).numpy()
plt.hist(errors_std, bins=20, alpha=0.5, label='Standard', color='red')
plt.hist(errors_eiv, bins=20, alpha=0.5, label='EIV', color='blue')
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.legend()
plt.title('Error Distribution')

plt.tight_layout()
plt.savefig('eiv_comparison.png')
print("\nPlot saved as eiv_comparison.png")
```

## When to Use EIV vs Standard Regression

### Use EIV Methods When:

✅ **Measurement errors in predictors**: X contains noise comparable to Y noise
✅ **Calibration problems**: Comparing two imperfect measurement methods
✅ **Scientific measurements**: Both variables measured with known/estimated uncertainties
✅ **Ratio of X error to Y error > 0.2**: When $\sigma_X / \sigma_Y > 0.2$, EIV provides meaningful improvement

### Use Standard Regression When:

❌ **Negligible X errors**: Predictors measured very precisely ($\sigma_X / \sigma_Y < 0.1$)
❌ **Unknown error structure**: Can't estimate or bound measurement errors
❌ **Computational constraints**: EIV methods are 2-5× slower than standard losses
❌ **Large sample sizes**: With N > 10,000, standard methods often sufficient

## Common Pitfalls

### ❌ Pitfall 1: Wrong Error Variance

```python
# Using arbitrary values
eiv_loss = FunctionalEIVLoss(model, sigma_x=1.0, sigma_y=1.0)
# → Results depend heavily on relative scales
```

**Solution**: Estimate errors from repeated measurements or domain knowledge. If unknown, tune as hyperparameters.

### ❌ Pitfall 2: Ignoring Computational Cost

```python
# Using expensive EIV when not needed
# X noise: 0.01, Y noise: 1.0 → ratio 0.01 (negligible)
eiv_loss = FunctionalEIVLoss(model, sigma_x=0.01, sigma_y=1.0)
# → 3× slower training with minimal benefit
```

**Solution**: Calculate error ratio first. If $\sigma_X / \sigma_Y < 0.1$, use standard regression.

### ❌ Pitfall 3: Not Testing on Clean Data

```python
# Training with EIV, testing on noisy data only
# → Can't separate model improvement from lucky noise cancellation
```

**Solution**: If possible, evaluate on independently measured "clean" reference data.

## References

- Fuller, W. A. (1987). "Measurement Error Models". Wiley.
- Carroll, R. J., et al. (2006). "Measurement Error in Nonlinear Models". Chapman & Hall.
- Cheng, C. L., & Van Ness, J. W. (1999). "Statistical Regression with Measurement Error". Arnold.
- Boggs, P. T., & Rogers, J. E. (1990). "Orthogonal Distance Regression". NIST.

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
