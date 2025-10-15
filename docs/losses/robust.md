# Robust Loss Functions

Robust loss functions are designed to be less sensitive to outliers than standard losses like MSE. They're particularly valuable when working with noisy data, datasets with potential measurement errors, or when you want your model to focus more on the majority of the data points rather than being heavily influenced by extreme values.

## Mathematical Background

The key idea behind robust losses is to reduce the influence of large errors by growing more slowly than squared error as the error magnitude increases. 

For a standard MSE loss, the gradient grows linearly with error magnitude:
$\frac{\partial}{\partial \hat{y}} (\hat{y} - y)^2 = 2(\hat{y} - y)$

In contrast, robust losses have gradients that are bounded or grow more slowly for large errors, making them less sensitive to outliers.

## Available Robust Losses

### HuberLoss

```python
class HuberLoss(RegressionLoss)
```

The Huber loss combines the best properties of MSE and MAE: it behaves like MSE for small errors and like MAE for large errors, controlled by the `delta` parameter.

**Parameters:**

- `delta` (float, optional): Threshold at which the loss changes from quadratic to linear. Default: `1.0`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the Huber loss

**Mathematical Formulation:**

$$\mathcal{L}_{\text{Huber}}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2, & \text{if } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2, & \text{otherwise}
\end{cases}$$

**Example:**

```python
import torch
import torchregress as tr

# Create Huber loss with default delta=1.0
loss_fn = tr.losses.HuberLoss()

# Predictions and targets
y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
target = torch.tensor([1.2, 1.9, 5.0, 4.1])  # Note the large error for index 2

# Calculate loss
loss = loss_fn(y_pred, target)

# Compare with MSE and MAE
mse_loss = tr.losses.MSELoss()(y_pred, target)
mae_loss = tr.losses.L1Loss()(y_pred, target)
print(f"Huber: {loss.item():.4f}, MSE: {mse_loss.item():.4f}, MAE: {mae_loss.item():.4f}")
# Huber will be somewhere between MSE and MAE, but closer to MAE due to the outlier
```

### L1Loss

```python
class L1Loss(RegressionLoss)
```

The L1 Loss (Mean Absolute Error) is less sensitive to outliers than MSE as it uses absolute differences instead of squared differences.

**Parameters:**

- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the L1 loss

**Mathematical Formulation:**

$$\mathcal{L}_{\text{L1}}(y, \hat{y}) = |y - \hat{y}|$$

**Example:**

```python
import torch
import torchregress as tr

# Create L1 loss
loss_fn = tr.losses.L1Loss()

# Predictions and targets
y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
target = torch.tensor([1.2, 1.9, 5.0, 4.1])

# Calculate loss
loss = loss_fn(y_pred, target)
```

### PseudoHuberLoss

```python
class PseudoHuberLoss(RegressionLoss)
```

The Pseudo-Huber loss is a smooth approximation of the Huber loss that ensures continuous derivatives everywhere.

**Parameters:**

- `delta` (float, optional): Controls the point where the loss transitions from quadratic to linear behavior. Default: `1.0`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the Pseudo-Huber loss

**Mathematical Formulation:**

$$\mathcal{L}_{\text{PseudoHuber}}(y, \hat{y}) = \delta^2 \left( \sqrt{1 + \left(\frac{y - \hat{y}}{\delta}\right)^2} - 1 \right)$$

**Example:**

```python
import torch
import torchregress as tr

# Create Pseudo-Huber loss with delta=1.0
loss_fn = tr.losses.PseudoHuberLoss(delta=1.0)

# Predictions and targets
y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
target = torch.tensor([1.2, 1.9, 5.0, 4.1])

# Calculate loss
loss = loss_fn(y_pred, target)
```

### LogCoshLoss

```python
class LogCoshLoss(RegressionLoss)
```

The Log-Cosh loss computes the logarithm of the hyperbolic cosine of the prediction error. It behaves like MSE for small errors and like MAE for large errors but is twice differentiable everywhere.

**Parameters:**

- `scale` (float, optional): Scaling factor that controls the transition from MSE to MAE. Default: `1.0`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the Log-Cosh loss

**Mathematical Formulation:**

$$\mathcal{L}_{\text{LogCosh}}(y, \hat{y}) = \text{log}(\text{cosh}(\text{scale} \cdot (y - \hat{y})))$$

**Example:**

```python
import torch
import torchregress as tr

# Create Log-Cosh loss
loss_fn = tr.losses.LogCoshLoss(scale=1.0)

# Predictions and targets
y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
target = torch.tensor([1.2, 1.9, 5.0, 4.1])

# Calculate loss
loss = loss_fn(y_pred, target)
```

### CharbonnierLoss

```python
class CharbonnierLoss(RegressionLoss)
```

The Charbonnier loss is a smooth alternative to L1 loss, often used in computer vision tasks.

**Parameters:**

- `eps` (float, optional): Small constant for numerical stability. Default: `1e-3`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the Charbonnier loss

**Mathematical Formulation:**

$$\mathcal{L}_{\text{Charbonnier}}(y, \hat{y}) = \sqrt{(y - \hat{y})^2 + \epsilon^2}$$

**Example:**

```python
import torch
import torchregress as tr

# Create Charbonnier loss
loss_fn = tr.losses.CharbonnierLoss(eps=1e-3)

# Predictions and targets
y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
target = torch.tensor([1.2, 1.9, 5.0, 4.1])

# Calculate loss
loss = loss_fn(y_pred, target)
```

### LqLoss

```python
class LqLoss(RegressionLoss)
```

The Lq Loss is a generalization of L1 (q=1) and L2 (q=2) losses, allowing for more flexible error penalization.

**Parameters:**

- `q` (float, optional): Order of the norm. Default: `1.5`
- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the Lq loss

**Mathematical Formulation:**

$$\mathcal{L}_{\text{Lq}}(y, \hat{y}) = |y - \hat{y}|^q$$

**Example:**

```python
import torch
import torchregress as tr

# Create Lq loss with q=1.5 (between L1 and L2)
loss_fn = tr.losses.LqLoss(q=1.5)

# Predictions and targets
y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
target = torch.tensor([1.2, 1.9, 5.0, 4.1])

# Calculate loss
loss = loss_fn(y_pred, target)
```

### TukeyBiweightLoss

```python
class TukeyBiweightLoss(RegressionLoss)
```

Tukey's biweight loss (or bisquare loss) completely ignores errors beyond a certain threshold, making it extremely robust to outliers.

**Parameters:**

- `c` (float, optional): Tuning constant (typical value 4.685). Default: `4.685`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the Tukey biweight loss

**Mathematical Formulation:**

$$\mathcal{L}_{\text{Tukey}}(y, \hat{y}) = \begin{cases}
\frac{c^2}{6} \left[1 - \left(1 - \left(\frac{y - \hat{y}}{c}\right)^2\right)^3\right], & \text{if } |y - \hat{y}| \leq c \\
\frac{c^2}{6}, & \text{otherwise}
\end{cases}$$

**Example:**

```python
import torch
import torchregress as tr

# Create Tukey biweight loss
loss_fn = tr.losses.TukeyBiweightLoss(c=4.685)

# Predictions and targets
y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
target = torch.tensor([1.2, 1.9, 15.0, 4.1])  # Extreme outlier at index 2

# Calculate loss - the outlier will have minimal impact
loss = loss_fn(y_pred, target)
```

### WinsorizedLoss

```python
class WinsorizedLoss(RegressionLoss)
```

Winsorized loss replaces extreme residuals with more moderate values, truncating the influence of outliers.

**Parameters:**

- `quantile_low` (float, optional): Lower quantile for winsorization (0-1). Default: `0.05`
- `quantile_high` (float, optional): Upper quantile for winsorization (0-1). Default: `0.95`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the winsorized loss

**Example:**

```python
import torch
import torchregress as tr

# Create Winsorized loss
loss_fn = tr.losses.WinsorizedLoss(quantile_low=0.25, quantile_high=0.75)

# Predictions and targets
y_pred = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
target = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])

# Calculate loss
loss = loss_fn(y_pred, target)
```

### LogBarrierLoss

```python
class LogBarrierLoss(RegressionLoss)
```

Log Barrier loss implements a logarithmic barrier function that gracefully limits the influence of large errors.

**Parameters:**

- `rho` (float, optional): Scale parameter defining the error threshold. Default: `1.0`
- `eps` (float, optional): Small constant to ensure loss remains finite. Default: `1e-8`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the log barrier loss

**Mathematical Formulation:**

$$\mathcal{L}_{\text{LogBarrier}}(y, \hat{y}) = -\log\left(1 - \min\left(\frac{|y - \hat{y}|}{\rho}, 1-\epsilon\right)^2\right)$$

**Example:**

```python
import torch
import torchregress as tr

# Create Log Barrier loss
loss_fn = tr.losses.LogBarrierLoss(rho=2.0)

# Predictions and targets
y_pred = torch.tensor([0.0, 1.0, 3.0])
target = torch.tensor([0.0, 2.0, 0.0])

# Calculate loss
loss = loss_fn(y_pred, target)
```

### AdaptiveHuberLoss

```python
class AdaptiveHuberLoss(RegressionLoss)
```

Adaptive Huber loss with automatic delta estimation based on data quantiles.

**Parameters:**

- `quantile` (float, optional): Quantile of absolute errors to use for delta estimation. Default: `0.8`
- `scale_factor` (float, optional): Additional scaling factor for delta. Default: `1.0`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the adaptive Huber loss

**Example:**

```python
import torch
import torchregress as tr

# Create Adaptive Huber loss
loss_fn = tr.losses.AdaptiveHuberLoss(quantile=0.8)

# Predictions and targets
y_pred = torch.tensor([0.0, 1.0, 2.0, 10.0])
target = torch.tensor([0.0, 2.0, 1.0, 0.0])

# Calculate loss
loss = loss_fn(y_pred, target)
```

### ClippedLoss

```python
class ClippedLoss(RegressionLoss)
```

Clipped loss for robust regression that clips errors above a threshold.

**Parameters:**

- `threshold` (float, optional): Error threshold beyond which the loss is clipped. Default: `1.0`
- `base_loss` (str, optional): Base loss function ('l1', 'mse'). Default: `'mse'`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the clipped loss

**Example:**

```python
import torch
import torchregress as tr

# Create Clipped loss
loss_fn = tr.losses.ClippedLoss(threshold=1.0, base_loss='mse')

# Predictions and targets
y_pred = torch.tensor([0.0, 1.0, 3.0])
target = torch.tensor([0.0, 2.0, 0.0])

# Calculate loss
loss = loss_fn(y_pred, target)
```

### FairLoss

```python
class FairLoss(RegressionLoss)
```

Fair loss grows less than linearly with the absolute error, making it less sensitive to outliers than MSE or MAE.

**Parameters:**

- `c` (float, optional): Scale parameter. Default: `1.0`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the fair loss

**Mathematical Formulation:**

$$\mathcal{L}_{\text{Fair}}(y, \hat{y}) = c^2 \left(\frac{|y - \hat{y}|}{c} - \log\left(1 + \frac{|y - \hat{y}|}{c}\right)\right)$$

**Example:**

```python
import torch
import torchregress as tr

# Create Fair loss
loss_fn = tr.losses.FairLoss(c=1.0)

# Predictions and targets
y_pred = torch.tensor([0.0, 1.0, 3.0])
target = torch.tensor([0.0, 2.0, 0.0])

# Calculate loss
loss = loss_fn(y_pred, target)
```

### CauchyLoss

```python
class CauchyLoss(RegressionLoss)
```

Cauchy loss uses the negative log of the Cauchy distribution density, making it very robust to outliers.

**Parameters:**

- `c` (float, optional): Scale parameter. Default: `1.0`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the Cauchy loss

**Mathematical Formulation:**

$$\mathcal{L}_{\text{Cauchy}}(y, \hat{y}) = \log\left(1 + \left(\frac{y - \hat{y}}{c}\right)^2\right)$$

**Example:**

```python
import torch
import torchregress as tr

# Create Cauchy loss with c=1.0
loss_fn = tr.losses.CauchyLoss(c=1.0)

# Predictions and targets
y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
target = torch.tensor([1.2, 1.9, 5.0, 4.1])  # Large error for index 2

# Calculate loss
loss = loss_fn(y_pred, target)

# Compare with other losses
huber_loss = tr.losses.HuberLoss()(y_pred, target)
mse_loss = tr.losses.MSELoss()(y_pred, target)
print(f"Cauchy: {loss.item():.4f}, Huber: {huber_loss.item():.4f}, MSE: {mse_loss.item():.4f}")
```

### BarronLoss

```python
class BarronLoss(RegressionLoss)
```

The Barron loss is a generalization of L1 and L2 losses, tunable via the `alpha` parameter. It can behave like L2 (`alpha=2`), Cauchy (`alpha=0`), or something in between.

**Parameters:**

- `alpha` (float, optional): Shape parameter controlling the robustness. Default: `1.0`
- `scale` (float, optional): Scale parameter. Default: `1.0`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the Barron loss

**Mathematical Formulation:**

$$\mathcal{L}_{\text{Barron}}(x, \alpha, c) = \begin{cases}
\frac{1}{2} (x/c)^2 & \alpha = 2 \\
\log(\frac{1}{2}(x/c)^2 + 1) & \alpha = 0 \\
\frac{|\alpha-2|}{\alpha} ((\frac{(x/c)^2}{|\alpha-2|} + 1)^{\alpha/2} - 1) & \text{otherwise}
\end{cases}$$

**Example:**

```python
import torch
import torchregress as tr

# Create Barron loss with alpha=1.0 (between L1 and L2)
loss_fn = tr.losses.BarronLoss(alpha=1.0)

# Predictions and targets
y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
target = torch.tensor([1.2, 1.9, 5.0, 4.1])

# Calculate loss
loss = loss_fn(y_pred, target)
```

## Choosing the Right Robust Loss

| Loss Function | Robustness Level | Twice Differentiable | Special Features |
|---------------|------------------|----------------------|-----------------|
| L1 (MAE) | Low | No | Simple absolute error |
| Huber | Moderate | No | Balance between MSE and MAE |
| Pseudo-Huber | Moderate | Yes | Smooth approximation of Huber |
| Log-Cosh | Moderate | Yes | Natural scaling of errors |
| Charbonnier | Moderate | Yes | Smooth L1 alternative |
| Lq | Variable | Depends on q | Flexible norm parameter |
| Barron | Variable | Yes | Generalization of L1/L2 and Cauchy |
| Cauchy | High | Yes | Very robust to extreme outliers |
| Fair | High | Yes | Gradual transition |
| TukeyBiweight | Highest | No | Completely ignores large outliers |
| Winsorized | High | No | Based on quantile thresholds |
| LogBarrier | High | No | Logarithmic barrier approach |
| AdaptiveHuber | Moderate | No | Auto-adjusting threshold |
| Clipped | Variable | No | Simple error capping |

## Practical Considerations

1. **MSE vs. Robust Losses**:
   - Use MSE when data has few or no outliers and errors are normally distributed
   - Use robust losses when data may contain outliers or non-Gaussian error distributions

2. **Delta/Scale Parameters**:
   - Smaller values make the loss more robust but may lead to slower convergence
   - Larger values make the loss behave more like MSE
   - A common approach is to start with a larger value and decrease it during training

3. **Combining Losses**:
   - Consider using a weighted combination of robust and standard losses
   - For example: `total_loss = 0.8 * huber_loss + 0.2 * mse_loss`

4. **Adaptive Robust Losses**:
   - For advanced use cases, consider losses like AdaptiveHuberLoss that automatically adjust robustness

5. **Visualization**:
   - When selecting a robust loss, visualize its behavior for different error magnitudes:

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchregress as tr

# Create different loss functions
mse_loss = tr.losses.MSELoss()
l1_loss = tr.losses.L1Loss()
huber_loss = tr.losses.HuberLoss(delta=1.0)
cauchy_loss = tr.losses.CauchyLoss(c=1.0)
tukey_loss = tr.losses.TukeyBiweightLoss(c=4.685)

# Generate errors
errors = torch.linspace(-10, 10, 1000)
target = torch.zeros_like(errors)
y_pred = errors  # error = pred - target = pred - 0 = pred

# Calculate losses
mse_values = mse_loss(y_pred, target, reduction='none')
l1_values = l1_loss(y_pred, target, reduction='none')
huber_values = huber_loss(y_pred, target, reduction='none')
cauchy_values = cauchy_loss(y_pred, target, reduction='none')
tukey_values = tukey_loss(y_pred, target, reduction='none')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(errors.numpy(), mse_values.numpy(), label='MSE')
plt.plot(errors.numpy(), l1_values.numpy(), label='MAE')
plt.plot(errors.numpy(), huber_values.numpy(), label='Huber')
plt.plot(errors.numpy(), cauchy_values.numpy(), label='Cauchy')
plt.plot(errors.numpy(), tukey_values.numpy(), label='Tukey')
plt.legend()
plt.title('Comparison of Loss Functions')
plt.xlabel('Error')
plt.ylabel('Loss')
plt.grid(True)
plt.xlim(-5, 5)
plt.ylim(0, 10)
plt.show()
```
