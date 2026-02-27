# Quantile & Expectile Losses

Quantile and expectile regression provide a more complete picture of the conditional distribution than standard mean regression, allowing for robust uncertainty estimation and prediction intervals without making distributional assumptions.

## Comparing Quantile and Expectile Regression

Both quantile and expectile regression are distribution-free approaches for estimating different parts of a conditional distribution:

| Feature | Quantile Regression | Expectile Regression |
|---------|---------------------|----------------------|
| **Minimizes** | Asymmetric absolute error | Asymmetric squared error |
| **Special case** | Median (τ=0.5) | Mean (τ=0.5) |
| **Interpretation** | τ% of observations are below | Weighted mean, more sensitive to extreme values |
| **Robustness** | More robust to outliers | Less robust to outliers |
| **Use case** | When exact probability levels are needed | When smoothness and efficiency matter |
| **Sensitivity** | Less sensitive to data in tails | More sensitive to data in tails |

## Mathematical Background

### Quantile Loss

The quantile loss function for a quantile level τ ∈ (0,1) is defined as:

$$\mathcal{L}_{\tau}(y, \hat{q}_{\tau}) = (y - \hat{q}_{\tau}) \cdot (\tau - \mathbb{1}_{y < \hat{q}_{\tau}})$$

Where:
- $y$ is the actual value
- $\hat{q}_{\tau}$ is the predicted τ-quantile
- $\mathbb{1}_{y < \hat{q}_{\tau}}$ is an indicator function (1 when y < $\hat{q}_{\tau}$, 0 otherwise)

This can also be written as:

$$\mathcal{L}_{\tau}(y, \hat{q}_{\tau}) = \begin{cases}
\tau \cdot (y - \hat{q}_{\tau}), & \text{if } y \geq \hat{q}_{\tau} \\
(1 - \tau) \cdot (\hat{q}_{\tau} - y), & \text{if } y < \hat{q}_{\tau}
\end{cases}$$

### Expectile Loss

The expectile loss function for an expectile level τ ∈ (0,1) is defined as:

$$\mathcal{L}_{\tau}(y, \hat{e}_{\tau}) = 2 \cdot |y - \hat{e}_{\tau}|^2 \cdot (\tau \cdot \mathbb{1}_{y \geq \hat{e}_{\tau}} + (1-\tau) \cdot \mathbb{1}_{y < \hat{e}_{\tau}})$$

Which can also be written as:

$$\mathcal{L}_{\tau}(y, \hat{e}_{\tau}) = \begin{cases}
2\tau \cdot (y - \hat{e}_{\tau})^2, & \text{if } y \geq \hat{e}_{\tau} \\
2(1 - \tau) \cdot (y - \hat{e}_{\tau})^2, & \text{if } y < \hat{e}_{\tau}
\end{cases}$$

**Note**: The factor of 2 ensures that when τ=0.5, the expectile loss equals standard MSE loss.

## Quantile Loss Functions

### QuantileLoss

```python
class QuantileLoss(RegressionLoss)
```

**Methods:**

- `forward(y_pred, y_true, mask=None, weights=None)`: Computes the quantile loss

**Example:**

```python
import torch
import torchregress as tr

# For median regression (q=0.5)
median_loss = tr.losses.QuantileLoss(quantile=0.5)
y_pred = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([0.0, 2.0, 4.0])
loss = median_loss(y_pred, y_true)

# For 90th percentile regression (q=0.9)
p90_loss = tr.losses.QuantileLoss(quantile=0.9)
loss = p90_loss(y_pred, y_true)  # Underestimation heavily penalized
```

### MultiQuantileLoss

```python
class MultiQuantileLoss(RegressionLoss)
```

Loss for simultaneously estimating multiple quantile levels, useful for generating prediction intervals.

**Parameters:**

- `quantiles` (list[float] or torch.Tensor): List of quantile levels in ascending order. Default: [0.1, 0.5, 0.9]
- `joint_prediction` (bool, optional): Whether predictions are passed as a joint tensor. Default: `True`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, y_true, mask=None, weights=None)`: Computes the combined quantile loss

**Example:**

```python
import torch
import torchregress as tr

# For 90% prediction intervals (5th and 95th percentiles) plus median
loss_fn = tr.losses.MultiQuantileLoss(quantiles=[0.05, 0.5, 0.95])

# Model predictions: [batch_size, num_quantiles, n_features]
# Here: 1 batch, 3 quantiles, 2 features
y_pred = torch.tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]])
y_true = torch.tensor([[2.0, 3.0]])

# Calculate combined loss across all quantiles
loss = loss_fn(y_pred, y_true)
```

### QuantileCrossover

```python
class QuantileCrossover(RegressionLoss)
```

Loss that encourages proper ordering of quantile predictions, ensuring lower quantiles predict smaller values than higher quantiles.

**Parameters:**

- `quantiles` (list[float] or torch.Tensor): List of quantile levels in ascending order
- `base_loss` (float, optional): Weight for standard quantile loss term. Default: `1.0`
- `crossover_penalty` (float, optional): Weight for crossover penalty term. Default: `10.0`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, y_true, mask=None, weights=None)`: Computes the quantile loss with crossover penalty

**Example:**

```python
import torch
import torchregress as tr

# Create loss for 10th, 50th, 90th percentiles
loss_fn = tr.losses.QuantileCrossover(
    quantiles=[0.1, 0.5, 0.9], 
    crossover_penalty=5.0
)

# Properly ordered predictions
good_pred = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])

# Predictions with crossover (q₁ > q₂)
bad_pred = torch.tensor([[[2.0, 3.0], [1.0, 2.0], [3.0, 1.0]]])

y_true = torch.tensor([[2.0, 2.0]])

# Normal loss for properly ordered predictions
good_loss = loss_fn(good_pred, y_true)

# Higher loss for predictions with crossover
bad_loss = loss_fn(bad_pred, y_true)
```

### SQRLoss

```python
class SQRLoss(RegressionLoss)
```

Simultaneous Quantile Regression (SQR) loss encourages distribution-free uncertainty estimation via twin quantiles.

**Parameters:**

- `lower_quantile` (float, optional): The lower quantile to predict. Default: `0.1`
- `upper_quantile` (float, optional): The upper quantile to predict. Default: `0.9`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, y_true, mask=None, weights=None)`: Computes the SQR loss

**Example:**

```python
import torch
import torchregress as tr

# Create SQR loss for 10th and 90th percentiles
loss_fn = SQRLoss(lower_quantile=0.1, upper_quantile=0.9)  # concept example

# Model predictions: [batch_size, 2*n_features]
# Here: 1 batch, 2 features
y_pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
y_true = torch.tensor([[2.0, 3.0]])

# Calculate combined loss
loss = loss_fn(y_pred, y_true)
```

## Expectile Loss Functions

### ExpectileLoss

```python
class ExpectileLoss(RegressionLoss)
```

Basic expectile regression loss function for estimating a single expectile level.

**Parameters:**

- `expectile` (float, optional): Expectile level to estimate (0 < τ < 1). Default: `0.5` (mean)
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, y_true, mask=None, weights=None)`: Computes the expectile loss

**Example:**

```python
import torch
import torchregress as tr

# For mean regression (τ=0.5)
mean_loss = tr.losses.ExpectileLoss(expectile=0.5)
y_pred = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([0.0, 2.0, 4.0])
loss = mean_loss(y_pred, y_true)  # Standard MSE at τ=0.5

# For 80th expectile regression (τ=0.8)
e80_loss = tr.losses.ExpectileLoss(expectile=0.8)
loss = e80_loss(y_pred, y_true)  # Underestimation penalized 4x more
```

### MultiExpectileLoss

```python
class MultiExpectileLoss(RegressionLoss)
```

Loss for simultaneously estimating multiple expectile levels, useful for characterizing the conditional distribution.

**Parameters:**

- `expectiles` (list[float] or torch.Tensor): List of expectile levels
- `joint_prediction` (bool, optional): Whether predictions are passed as a joint tensor. Default: `True`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, y_true, mask=None, weights=None)`: Computes the combined expectile loss

**Example:**

```python
import torch
import torchregress as tr

# For predicting 10th, 50th and 90th expectiles together
loss_fn = tr.losses.MultiExpectileLoss(expectiles=[0.1, 0.5, 0.9])

# Model predictions: [batch_size, num_expectiles, features]
y_pred = torch.tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]])
y_true = torch.tensor([[2.0, 3.0]])

# Calculate combined loss across all expectiles
loss = loss_fn(y_pred, y_true)
```

### ExpectileCrossover

```python
class ExpectileCrossover(RegressionLoss)
```

Loss that encourages proper ordering of expectile predictions, ensuring lower expectiles predict smaller values than higher expectiles.

**Parameters:**

- `expectiles` (list[float] or torch.Tensor): List of expectile levels in ascending order
- `base_loss` (float, optional): Weight for standard expectile loss term. Default: `1.0`
- `crossover_penalty` (float, optional): Weight for crossover penalty term. Default: `10.0`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, y_true, mask=None, weights=None)`: Computes the expectile loss with crossover penalty

**Example:**

```python
import torch
import torchregress as tr

# Create loss for 20th, 50th, 80th expectiles
loss_fn = tr.losses.ExpectileCrossover(
    expectiles=[0.2, 0.5, 0.8], 
    crossover_penalty=5.0
)

# Properly ordered predictions
good_pred = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])

# Predictions with crossover (e₁ > e₂)
bad_pred = torch.tensor([[[2.0, 3.0], [1.0, 2.0], [3.0, 1.0]]])

y_true = torch.tensor([[2.0, 2.0]])

# Compare losses
good_loss = loss_fn(good_pred, y_true)
bad_loss = loss_fn(bad_pred, y_true)  # Higher due to penalty
```

### AsymmetricLeastSquaresLoss

```python
class AsymmetricLeastSquaresLoss(ExpectileLoss)
```

Alias for ExpectileLoss, provided for compatibility with different naming conventions.

**Example:**

```python
import torch
import torchregress as tr

# This is equivalent to ExpectileLoss(expectile=0.75)
loss_fn = tr.losses.AsymmetricLeastSquaresLoss(tau=0.75)
y_pred = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([0.0, 2.0, 4.0])
loss = loss_fn(y_pred, y_true)
```

## Real-World Applications

### Prediction Intervals

The most common application is creating prediction intervals that capture the uncertainty in predictions:

```python
import torch
import torchregress as tr
import matplotlib.pyplot as plt

# Define model that outputs 3 values per input (q0.05, q0.5, q0.95)
class QuantileModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU()
        )
        self.q_heads = torch.nn.ModuleList([
            torch.nn.Linear(64, 1) for _ in range(3)
        ])
        
    def forward(self, x):
        features = self.shared(x)
        quantiles = torch.stack([head(features) for head in self.q_heads], dim=1)
        return quantiles  # Shape: [batch_size, 3, 1]

# Loss function with crossover penalty to ensure proper ordering
loss_fn = tr.losses.QuantileCrossover(
    quantiles=[0.05, 0.5, 0.95],
    crossover_penalty=10.0
)

# After training, visualize results
with torch.no_grad():
    predictions = model(X_test)
    lower = predictions[:, 0]  # 5th percentile
    median = predictions[:, 1]  # Median
    upper = predictions[:, 2]  # 95th percentile
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, alpha=0.5, label="Observations")
    plt.plot(X_test, median, 'r-', label="Median")
    plt.fill_between(X_test.squeeze(), 
                    lower.squeeze(), 
                    upper.squeeze(), 
                    alpha=0.3, label="90% Prediction Interval")
    plt.legend()
    plt.title("Quantile Regression: 90% Prediction Intervals")
```

### Comparing Multiple Quantile Levels

For a richer view of the distribution, we can estimate many quantile levels:

```python
# Define model that outputs 9 quantile levels (from 0.1 to 0.9)
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Use MultiQuantileLoss with CrossoverLoss
loss_fn = tr.losses.QuantileCrossover(
    quantiles=quantiles,
    crossover_penalty=10.0
)

# After training, visualize the quantile curves
with torch.no_grad():
    predictions = model(X_test)  # [batch_size, 9, 1]
    
    plt.figure(figsize=(12, 8))
    plt.scatter(X_test, y_test, alpha=0.3, label="Observations")
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(quantiles)))
    for i, q in enumerate(quantiles):
        plt.plot(X_test, predictions[:, i], color=colors[i], label=f"Q{q}")
    
    plt.legend()
    plt.title("Multiple Quantile Regression")
```

### Financial Risk Modeling

Quantile regression is particularly useful for risk measures like Value-at-Risk (VaR) and Expected Shortfall (ES):

```python
# For Value-at-Risk (q=0.05) and Expected Shortfall
var_loss = tr.losses.QuantileLoss(quantile=0.05)
es_loss = tr.losses.ExpectileLoss(expectile=0.02)  # Expectile can approximate ES

# Joint risk model
class RiskModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Sequential(
            # ...feature extraction layers...
        )
        self.var_head = torch.nn.Linear(64, 1)  # VaR prediction
        self.es_head = torch.nn.Linear(64, 1)   # ES prediction
        
    def forward(self, x):
        features = self.shared(x)
        var = self.var_head(features)
        es = self.es_head(features)
        return var, es
```

## Choosing Between Quantile and Expectile Regression

**Use Quantile Regression when:**
- You need direct probabilistic interpretation (e.g., 95th percentile)
- You're working with data that has outliers
- You need prediction intervals with specific coverage probabilities
- Computing risk measures like Value-at-Risk (VaR)

**Use Expectile Regression when:**
- You want smoother estimates that are more statistically efficient
- You're dealing with tails of the distribution
- You need more sensitivity to the magnitudes of extreme values
- Computing risk measures like Expected Shortfall (ES)

## Advanced Implementation Tips

1. **Monotonic Networks**:
   For strictly monotonic quantile predictions, consider using monotonic constraints in your network architecture.

2. **Ensemble Methods**:
   Quantile regression forests or bootstrapped ensembles can improve quantile estimates.

3. **Conformal Prediction**:
   For guaranteed coverage properties, combine quantile regression with conformal prediction:

   ```python
   from torchregress.calibration import ConformalQuantileRegression
   
   conformal = ConformalQuantileRegression(
       model,
       quantiles=[0.05, 0.95],
       calibration_ratio=0.2  # Use 20% of data for calibration
   )
   conformal.calibrate(X_calib, y_calib)
   lower, upper = conformal.predict(X_test)
   ```

4. **Spline-Based Approaches**:
   Consider non-parametric methods like quantile regression splines for flexible, smooth quantile functions.
