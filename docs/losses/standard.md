# Standard Loss Functions

This page documents the standard regression loss functions available in TorchRegression, including direct implementations and wrapped PyTorch losses with added support for masking and weights.

## Direct Implementations

### MSELoss

```python
class MSELoss(RegressionLoss)
```

Mean Squared Error loss with support for masking and per-sample weights.

**Parameters:**

- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the weighted MSE loss

**Mathematical Formulation:**

Mean Squared Error calculates the average of the squared differences between predicted and actual values:

$$\mathcal{L}_{\text{MSE}}(y, \hat{y}) = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

With weights, this becomes:

$$\mathcal{L}_{\text{WeightedMSE}}(y, \hat{y}, w) = \frac{\sum_{i=1}^{N}w_i(y_i - \hat{y}_i)^2}{\sum_{i=1}^{N}w_i}$$

**Example:**

```python
import torch
import torchregression as tr

# Create MSE loss
loss_fn = tr.losses.MSELoss()

# Simple case
y_pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.0, 2.5])
loss = loss_fn(y_pred, target)  # tensor(0.1667)

# With sample weights (emphasize the last sample)
weights = torch.tensor([1.0, 1.0, 3.0])
weighted_loss = loss_fn(y_pred, target, weights=weights)

# With masking (ignore the middle value)
mask = torch.tensor([True, False, True])
masked_loss = loss_fn(y_pred, target, mask=mask)
```

### BCELoss

```python
class BCELoss(MaskedLoss)
```

Binary Cross Entropy loss with support for masking, weighting, and positive class weighting.

**Parameters:**

- `pos_weight` (float or tensor, optional): Weight for the positive class. Default: None
- `weight` (float or tensor, optional): Global weighting factor. Default: None
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `eps` (float, optional): Small constant for numerical stability. Default: 1e-8

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the BCE loss

**Mathematical Formulation:**

Binary Cross Entropy measures the performance of a classification model whose output is a probability value between 0 and 1:

$$\mathcal{L}_{\text{BCE}}(y, \hat{y}) = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

With positive class weights:

$$\mathcal{L}_{\text{WeightedBCE}}(y, \hat{y}) = -\frac{1}{N}\sum_{i=1}^{N} \left[ w_p \cdot y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

**Example:**

```python
import torch
import torchregression as tr

# Create BCE loss with positive class weighting
loss_fn = tr.losses.BCELoss(pos_weight=2.0)

# Predicted probabilities
y_pred = torch.tensor([0.2, 0.7, 0.9])
# Binary targets
target = torch.tensor([0.0, 1.0, 1.0])

# Calculate loss
loss = loss_fn(y_pred, target)

# With masking (ignore the middle value)
mask = torch.tensor([True, False, True])
masked_loss = loss_fn(y_pred, target, mask=mask)
```

## PyTorch Loss Wrappers

TorchRegression provides wrapped versions of standard PyTorch losses with added support for masking and weights:

### MaskedMSELoss

```python
MaskedMSELoss = TorchLossWrapper(nn.MSELoss)
```

Wrapped version of PyTorch's MSELoss with masking support.

**Example:**

```python
loss_fn = tr.losses.MaskedMSELoss()
loss = loss_fn(y_pred, target, mask=mask)
```

### MaskedL1Loss

```python
MaskedL1Loss = TorchLossWrapper(nn.L1Loss)
```

Wrapped version of PyTorch's L1Loss (Mean Absolute Error) with masking support.

**Example:**

```python
loss_fn = tr.losses.MaskedL1Loss()
loss = loss_fn(y_pred, target, mask=mask)
```

### MaskedHuberLoss

```python
MaskedHuberLoss = TorchLossWrapper(nn.HuberLoss)
```

Wrapped version of PyTorch's HuberLoss with masking support.

**Example:**

```python
loss_fn = tr.losses.MaskedHuberLoss(delta=1.0)  # Use delta parameter from PyTorch
loss = loss_fn(y_pred, target, mask=mask)
```

### Other Wrapped Losses

TorchRegression provides masked versions of the following PyTorch losses:

- `MaskedCrossEntropyLoss`: For multi-class classification
- `MaskedBCEWithLogitsLoss`: For binary classification with logits output
- `MaskedKLDivLoss`: For KL divergence
- `MaskedNLLLoss`: For negative log-likelihood
- `MaskedSmoothL1Loss`: For smooth L1 (Huber with delta=1.0)
- `MaskedPoissonNLLLoss`: For Poisson negative log-likelihood

## Using the TorchLossWrapper

You can wrap any PyTorch loss to add masking and weighting support:

```python
import torch.nn as nn
import torchregression as tr

# Create a custom PyTorch loss
class CustomLoss(nn.Module):
    def forward(self, y_pred, target):
        return torch.abs(y_pred - target)**1.5

# Wrap it with TorchRegression's wrapper
wrapped_loss = tr.losses.TorchLossWrapper(CustomLoss())

# Now you can use it with masks and weights
y_pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.0, 2.5])
mask = torch.tensor([True, False, True])
loss = wrapped_loss(y_pred, target, mask=mask)
```

## When to Use Different Standard Losses

| Loss Function | Best For | Properties |
|---------------|----------|------------|
| MSELoss | General regression | Penalizes large errors more heavily, simple gradient |
| L1Loss (MAE) | Robust regression | Less sensitive to outliers, constant gradient |
| HuberLoss | Balancing MSE/MAE | MSE for small errors, MAE for large errors |
| BCELoss | Binary targets (0-1) | Classification with probability outputs |
| SmoothL1Loss | Regression with outliers | Smoother gradients than Huber |

## Considerations for Loss Selection

1. **Outlier Sensitivity**:
   - If your data has outliers, consider L1Loss or HuberLoss instead of MSELoss
   - MSELoss squares errors, so it heavily penalizes outliers

2. **Regression vs Classification**:
   - For real-valued outputs, use regression losses (MSE, MAE, Huber)
   - For binary outputs, use BCELoss or BCEWithLogitsLoss
   - For multi-class outputs, use CrossEntropyLoss

3. **Gradient Behavior**:
   - MSELoss has larger gradients for larger errors
   - L1Loss has constant gradients (potentially better for deep networks)
   - HuberLoss and SmoothL1Loss combine advantages of both

4. **Masked Data**:
   - All TorchRegression losses support masking for handling missing values
   - Use the `mask` parameter to specify which elements to include in the loss

5. **Weighted Samples**:
   - Use the `weights` parameter to emphasize certain samples or features
   - Useful for imbalanced datasets or when certain samples are more important
