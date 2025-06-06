# Base Loss Functions

This page documents the foundational loss function classes that serve as building blocks for all loss functions in TorchRegression.

## Class Hierarchy

TorchRegression implements a hierarchical structure of loss functions with the following inheritance tree:

```
BaseLoss
├── RegressionLoss
└── DistributionLoss
```

## BaseLoss

```python
class BaseLoss(torch.nn.Module)
```

`BaseLoss` is the root class for all loss functions in TorchRegression. It extends PyTorch's `nn.Module` and provides common functionality for reduction operations.

**Parameters:**

- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, **kwargs)`: Abstract method that subclasses must implement
- `_reduce(loss, mask=None, weights=None)`: Applies specified reduction to the loss tensor

**Example:**

```python
# Custom loss implementation
class CustomLoss(BaseLoss):
    def forward(self, y_pred, target, **kwargs):
        # Calculate point-wise loss
        loss = (y_pred - target)**2
        # Apply reduction
        return self._reduce(loss)
```

## RegressionLoss

```python
class RegressionLoss(BaseLoss)
```

`RegressionLoss` is a base class specifically designed for standard regression loss functions that operate on point predictions.

**Parameters:**

- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Abstract method for computing regression loss

**Example:**

```python
# Using a regression loss with weights
loss_fn = tr.losses.L1Loss()  # Inherits from RegressionLoss

y_pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([0.0, 2.0, 4.0])
weights = torch.tensor([0.5, 1.0, 2.0])  # Emphasize the importance of the 3rd sample

loss = loss_fn(y_pred, target, weights=weights)
```

## DistributionLoss

```python
class DistributionLoss(BaseLoss)
```

`DistributionLoss` serves as a base class for losses that model full probability distributions rather than just point predictions. These losses take distribution parameters as inputs and calculate proper scoring rules.

**Parameters:**

- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'

**Methods:**

- `_extract_distribution_parameters(y_pred)`: Extract distribution parameters from model outputs
- `_calculate_nll(y_pred, target, mask)`: Calculate negative log-likelihood
- `forward(y_pred, target, mask=None, weights=None)`: Abstract method for computing distributional loss

**Example:**

```python
# Using a distributional loss for uncertainty modeling
loss_fn = tr.losses.GaussianNLLLoss()  # Inherits from DistributionLoss

# Model outputs mean and log-variance
mean = torch.tensor([1.0, 2.0, 3.0])
log_var = torch.tensor([-1.0, 0.0, 1.0])
target = torch.tensor([0.8, 2.2, 2.7])

# Calculate NLL loss using both mean and variance predictions
loss = loss_fn((mean, log_var), target)
```

## TorchLossWrapper

```python
class TorchLossWrapper(BaseLoss)
```

`TorchLossWrapper` adapts standard PyTorch loss functions to TorchRegression's interface, adding support for masks and weights.

**Parameters:**

- `loss_fn` (Callable or nn.Module): PyTorch loss function class or instance
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `**kwargs`: Additional arguments to pass to the loss function

**Example:**

```python
import torch.nn as nn
import torchregress as tr

# Wrap a standard PyTorch loss
torch_mse = nn.MSELoss()
wrapped_mse = tr.losses.TorchLossWrapper(torch_mse)

# Now we can use it with masks and weights
y_pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.0, 2.5])
mask = torch.tensor([True, False, True])

loss = wrapped_mse(y_pred, target, mask=mask)
```

## WeightedLossWrapper

```python
class WeightedLossWrapper(BaseLoss)
```

`WeightedLossWrapper` adapts standard PyTorch loss functions to TorchRegression's interface, adding support for masks and weights.

**Parameters:**

- `loss_fn` (Callable or nn.Module): PyTorch loss function class or instance
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `**kwargs`: Additional arguments to pass to the loss function

**Example:**

```python
import torch.nn as nn
import torchregress as tr

# Wrap a standard PyTorch loss
torch_mse = nn.MSELoss()
wrapped_mse = tr.losses.WeightedLossWrapper(torch_mse)

# Now we can use it with masks and weights
y_pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.0, 2.5])
mask = torch.tensor([True, False, True])

loss = wrapped_mse(y_pred, target, mask=mask)
```

## Pre-defined Weighted Loss Functions

TorchRegression provides weighted versions of all standard PyTorch loss functions:

```python
# Available weighted versions of PyTorch losses
WeightedMSELoss = WeightedLossWrapper(nn.MSELoss)
WeightedL1Loss = WeightedLossWrapper(nn.L1Loss)
WeightedCrossEntropyLoss = WeightedLossWrapper(nn.CrossEntropyLoss)
WeightedBCELoss = WeightedLossWrapper(nn.BCELoss)
WeightedBCEWithLogitsLoss = WeightedLossWrapper(nn.BCEWithLogitsLoss)
WeightedKLDivLoss = WeightedLossWrapper(nn.KLDivLoss)
WeightedNLLLoss = WeightedLossWrapper(nn.NLLLoss)
WeightedSmoothL1Loss = WeightedLossWrapper(nn.SmoothL1Loss)
WeightedHuberLoss = WeightedLossWrapper(nn.HuberLoss)
WeightedPoissonNLLLoss = WeightedLossWrapper(nn.PoissonNLLLoss)
WeightedGaussianNLLLoss = WeightedLossWrapper(nn.GaussianNLLLoss)
# ...and many more
```

These can be used directly with mask and weight arguments:

```python
import torch
import torchregress as tr

# Use weighted version of standard PyTorch loss
loss_fn = tr.losses.WeightedMSELoss()

# Calculate with masks and weights
y_pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([0.0, 2.0, 4.0])
mask = torch.tensor([True, False, True])
weights = torch.tensor([0.5, 1.0, 2.0])

loss = loss_fn(y_pred, target, mask=mask, weights=weights)
```

## Best Practices for Custom Loss Functions

When implementing custom loss functions using the TorchRegression framework:

1. **Inherit from the right base class**:
   - For standard regression losses: `RegressionLoss`
   - For distribution-based losses: `DistributionLoss`
   
2. **Always validate inputs** by calling `self._validate_inputs(y_pred, target, mask)` in your `forward` method

3. **Handle reduction properly**:
   - For RegressionLoss, use `return self._reduce_with_mask(loss, mask, weights)`
   - For DistributionLoss, calculate NLL first, then use `self._reduce_with_mask()`
   
4. **Support masking and weighting** to make your loss function compatible with missing data scenarios

5. **Document the mathematical formulation** clearly in docstrings
