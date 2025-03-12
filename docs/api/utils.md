# Utility Functions

TorchRegression provides various utility functions to support regression tasks, from data validation to augmentation techniques. This page documents the most commonly used utilities.

## Validation Utilities

These functions help validate inputs and parameters to ensure they meet the requirements of regression models.

### `validate_tensors`

```python
def validate_tensors(*tensors, same_shape=True, allow_empty=False)
```

Validate that all input tensors are valid PyTorch tensors with appropriate shapes.

**Parameters:**

- `*tensors`: Variable number of tensors to validate
- `same_shape` (bool, optional): Whether all tensors should have the same shape. Default: True
- `allow_empty` (bool, optional): Whether to allow empty tensors. Default: False

**Returns:**

- list: List of validated PyTorch tensors

**Raises:**

- ValueError: If any tensor is invalid

**Example:**

```python
y_pred, y_true = tr.utils.validate_tensors(y_pred, y_true, same_shape=True)
```

### `validate_shape`

```python
def validate_shape(tensor, expected_shape=None, min_dim=None, max_dim=None)
```

Validate that a tensor has the expected shape or dimensions.

**Parameters:**

- `tensor` (torch.Tensor): Tensor to validate
- `expected_shape` (tuple, optional): Expected exact shape. Default: None
- `min_dim` (int, optional): Minimum number of dimensions. Default: None
- `max_dim` (int, optional): Maximum number of dimensions. Default: None

**Returns:**

- torch.Tensor: The input tensor if valid

**Raises:**

- ValueError: If the tensor shape is invalid

**Example:**

```python
y_pred = tr.utils.validate_shape(y_pred, min_dim=2, max_dim=3)
```

### `validate_sample_weights`

```python
def validate_sample_weights(sample_weight, n_samples)
```

Validate and normalize sample weights.

**Parameters:**

- `sample_weight` (torch.Tensor or None): Sample weights
- `n_samples` (int): Number of samples

**Returns:**

- torch.Tensor or None: Validated and normalized sample weights

**Example:**

```python
weights = tr.utils.validate_sample_weights(weights, len(y_true))
```

### `validate_reduction`

```python
def validate_reduction(reduction)
```

Validate reduction parameter.

**Parameters:**

- `reduction` (str): Reduction method ('none', 'mean', or 'sum')

**Returns:**

- str: The validated reduction method

**Raises:**

- ValueError: If reduction is invalid

**Example:**

```python
reduction = tr.utils.validate_reduction(reduction)
```

### `validate_range`

Validates that a value or tensor is within a specified range.

```python
from torchregression.utils.validation import validate_range

# Validate probability values
prob = torch.tensor([0.1, 0.3, 0.95])
validate_range(prob, 0.0, 1.0, "probability")  # Valid
 
# Will raise ValueError: "parameter must be between 0.0 and 1.0, got 1.5"
validate_range(1.5, 0.0, 1.0, "parameter")
```

### `validate_quantile`

Validates quantile values (must be between 0 and 1) and converts them to tensor format.

```python
from torchregression.utils.validation import validate_quantile

q = validate_quantile(0.5)  # Single quantile
q_multiple = validate_quantile(torch.tensor([0.1, 0.5, 0.9]))  # Multiple quantiles
```

## Tensor Operations

These utilities help with common tensor operations for regression tasks.

### `masked_mean`

```python
def masked_mean(tensor, mask=None, dim=None, keepdim=False)
```

Compute mean of tensor with optional masking of invalid values.

**Parameters:**

- `tensor` (torch.Tensor): Input tensor
- `mask` (torch.Tensor, optional): Boolean mask for valid values. Default: None
- `dim` (int or tuple, optional): Dimension to reduce. Default: None (all dimensions)
- `keepdim` (bool, optional): Whether to keep reduced dimensions. Default: False

**Returns:**

- torch.Tensor: Mean value with masked elements excluded

**Example:**

```python
mean_value = tr.utils.masked_mean(tensor, mask)
```

### `masked_sum`

```python
def masked_sum(tensor, mask=None, dim=None, keepdim=False)
```

Compute sum of tensor with optional masking of invalid values.

**Parameters:**

- `tensor` (torch.Tensor): Input tensor
- `mask` (torch.Tensor, optional): Boolean mask for valid values. Default: None
- `dim` (int or tuple, optional): Dimension to reduce. Default: None (all dimensions)
- `keepdim` (bool, optional): Whether to keep reduced dimensions. Default: False

**Returns:**

- torch.Tensor: Sum value with masked elements excluded

**Example:**

```python
sum_value = tr.utils.masked_sum(tensor, mask)
```

### `reduce_tensor`

```python
def reduce_tensor(tensor, reduction='mean', mask=None, dim=None, keepdim=False)
```

Apply reduction to tensor with optional masking.

**Parameters:**

- `tensor` (torch.Tensor): Input tensor
- `reduction` (str, optional): Reduction method ('none', 'mean', or 'sum'). Default: 'mean'
- `mask` (torch.Tensor, optional): Boolean mask for valid values. Default: None
- `dim` (int or tuple, optional): Dimension to reduce. Default: None (all dimensions)
- `keepdim` (bool, optional): Whether to keep reduced dimensions. Default: False

**Returns:**

- torch.Tensor: Reduced tensor

**Example:**

```python
result = tr.utils.reduce_tensor(loss_values, reduction='mean', mask=valid_mask)
```

### `broadcast_to_shape`

```python
def broadcast_to_shape(tensor, target_shape)
```

Broadcast a tensor to a target shape.

**Parameters:**

- `tensor` (torch.Tensor): Input tensor
- `target_shape` (tuple): Target shape to broadcast to

**Returns:**

- torch.Tensor: Broadcasted tensor

**Example:**

```python
broadcasted = tr.utils.broadcast_to_shape(tensor, (100, 5))
```

### `to_tensor`

```python
def to_tensor(x, dtype=torch.float32, device=None)
```

Convert input to PyTorch tensor.

**Parameters:**

- `x`: Input object (tensor, ndarray, list, etc.)
- `dtype` (torch.dtype, optional): Desired data type. Default: torch.float32
- `device` (torch.device, optional): Device to put tensor on. Default: None

**Returns:**

- torch.Tensor: PyTorch tensor version of the input

**Example:**

```python
tensor = tr.utils.to_tensor(numpy_array, dtype=torch.float32)
```

### `ensure_2d`

```python
def ensure_2d(tensor)
```

Ensure tensor has at least 2 dimensions.

**Parameters:**

- `tensor` (torch.Tensor): Input tensor

**Returns:**

- torch.Tensor: Tensor with at least 2 dimensions

**Example:**

```python
tensor_2d = tr.utils.ensure_2d(tensor)
```

### `standardize_tensor`

```python
def standardize_tensor(tensor, dim=0, eps=1e-8)
```

Standardize tensor to have zero mean and unit variance.

**Parameters:**

- `tensor` (torch.Tensor): Input tensor
- `dim` (int or tuple, optional): Dimension to standardize along. Default: 0
- `eps` (float, optional): Small constant for numerical stability. Default: 1e-8

**Returns:**

- torch.Tensor: Standardized tensor
- torch.Tensor: Mean of original tensor
- torch.Tensor: Standard deviation of original tensor

**Example:**

```python
std_tensor, mean, std = tr.utils.standardize_tensor(tensor)
```

### `create_mask`

```python
def create_mask(tensor, condition_fn=lambda x: ~torch.isnan(x))
```

Create a boolean mask for a tensor based on a condition function.

**Parameters:**

- `tensor` (torch.Tensor): Input tensor
- `condition_fn` (callable, optional): Function that returns True for valid values. Default: lambda x: ~torch.isnan(x)

**Returns:**

- torch.Tensor: Boolean mask with True for valid values

**Example:**

```python
# Create mask for non-NaN values
mask = tr.utils.create_mask(tensor)

# Create mask for positive values
mask = tr.utils.create_mask(tensor, lambda x: x > 0)
```

### `masked_reduction`

Apply reduction operations to a tensor with optional masking for handling missing values.

```python
from torchregression.utils.tensor_ops import masked_reduction

# Create data with some missing values (represented by mask)
data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
mask = torch.tensor([True, True, False, True, True])

# Compute mean of valid values only
mean = masked_reduction(data, mask, reduction='mean')  # Result: 3.0 (mean of [1,2,4,5])

# Compute sum of valid values
total = masked_reduction(data, mask, reduction='sum')  # Result: 12.0
```

### `standardize` / `unstandardize`

Standardize data to zero mean and unit variance, and convert it back to original scale.

```python
from torchregression.utils.tensor_ops import standardize, unstandardize

# Sample data
X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Standardize
X_std, mean, std = standardize(X)

# Use standardized values for model
# ...

# Convert predictions back to original scale
predictions_std = model(X_std)
predictions = unstandardize(predictions_std, mean, std)
```

### `calculate_propagated_variance`

Calculate variance propagation through a model, essential for uncertainty quantification.

```python
from torchregression.utils.tensor_ops import calculate_propagated_variance, prepare_model_input_for_gradients

# Prepare input with gradient tracking
X = prepare_model_input_for_gradients(X_test)

# Get model predictions and compute gradients
y_pred = model(X)
grads = torch.autograd.grad(
    outputs=y_pred,
    inputs=X,
    grad_outputs=torch.ones_like(y_pred),
    create_graph=True
)[0].view(-1, y_pred.shape[1], X.shape[1])

# Input uncertainty (could be measurement error)
sigma_x = torch.ones(X.shape[1]) * 0.1  

# Calculate propagated uncertainty
propagated_variance = calculate_propagated_variance(grads, sigma_x)
```

## Label Handling

Functions for encoding, decoding, and combining labels from multiple sources.

### `combine_dawid_skene`

Implements the Dawid-Skene model for aggregating annotations from multiple annotators, leveraging an EM algorithm to estimate true labels.

```python
from torchregression.utils.labels import combine_dawid_skene

# Multiple annotator labels (num_samples x num_annotators)
# -1 indicates missing annotations
annotations = torch.tensor([
    [0, 1, 0, -1],  # Sample 1: three annotators voted [0,1,0], one missing
    [1, 1, 1, 0],   # Sample 2: four annotators voted [1,1,1,0]
    [0, -1, 0, 0]   # Sample 3: three annotators voted [0,0,0], one missing
])

# Number of classes (binary in this case)
num_classes = 2

# Estimate true labels
pi, confusion_matrices, q_z = combine_dawid_skene(annotations, num_classes)

# q_z contains the probability of each true class
# Take argmax for hard labels
estimated_labels = torch.argmax(q_z, dim=1)
```

### `combine_continuous_blue_with_scaling`

Combines continuous estimates using Best Linear Unbiased Estimator (BLUE), scaling uncertainty when estimators disagree.

```python
from torchregression.utils.labels import combine_continuous_blue_with_scaling

# Different estimates for the same quantities (num_samples x num_estimators)
estimates = torch.tensor([
    [10.2, 9.8, 10.0, 10.1],  # Four estimates for sample 1
    [5.1, 4.9, 5.2, 5.0]      # Four estimates for sample 2
])

# Known variances of each estimator
variances = torch.tensor([0.2, 0.1, 0.15, 0.3])

# Combine estimates
combined_estimate, scaled_variance, scale_factor = combine_continuous_blue_with_scaling(
    estimates, variances=variances
)

print(f"Combined estimates: {combined_estimate}")
print(f"Uncertainties: {torch.sqrt(scaled_variance)}")
print(f"Scaling factor: {scale_factor}")  # >1 indicates inconsistent estimators
```

## Data Augmentation

Techniques for augmenting regression datasets to improve model generalization.

### `GaussianNoiseAugmentation`

Add Gaussian noise to input features for data augmentation.

```python
from torchregression.utils.augment import GaussianNoiseAugmentation

# Create augmenter with 50% probability and noise std=0.1
augmenter = GaussianNoiseAugmentation(std=0.1, probability=0.5)

# Apply to batch of data
X_batch = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y_batch = torch.tensor([10.0, 20.0])

X_augmented, y_augmented = augmenter(X_batch, y_batch)
```

### `MixUp`

Implement MixUp augmentation for regression tasks.

```python
from torchregression.utils.augment import MixUp

# Create MixUp augmenter
mixup = MixUp(alpha=0.2, probability=0.7)

# Apply to batch of data
X_batch = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y_batch = torch.tensor([10.0, 20.0, 30.0])

X_mixed, y_mixed = mixup(X_batch, y_batch)
```

### `EnsemblePerturbationAugmenter`

Generate multiple perturbed versions of inputs for ensemble prediction methods, useful for uncertainty estimation.

```python
from torchregression.utils.augment import EnsemblePerturbationAugmenter

# Create ensemble perturbation generator
perturbator = EnsemblePerturbationAugmenter(
    n_samples=20,  # Generate 20 perturbations
    perturb_method='gaussian',  # Use Gaussian noise
    sigma=0.05,  # With standard deviation 0.05
    feature_wise=True  # Apply noise to each feature independently
)

# Input features
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Generate perturbed samples
perturbed_samples = perturbator(X)  # Returns list of 20 perturbed tensors
# OR
stacked_samples = perturbator.generate_and_stack(X)  # Shape: [20, 2, 3]
```

## PyTorch Compatibility

Utilities to ensure compatibility across PyTorch versions.

### `get_device`

Automatically select the appropriate device for computation.

```python
from torchregression.utils.pytorch_compat import get_device

# Auto-select device (CUDA if available, else CPU)
device = get_device()

# Force CPU usage even if CUDA is available
device = get_device('cpu')

# Use specific GPU
device = get_device('cuda:1')

# Move model and data to device
model = model.to(device)
data = data.to(device)
```

### `set_all_seeds`

Set all random seeds for reproducible results.

```python
from torchregression.utils.pytorch_compat import set_all_seeds

# Set seed for reproducible results
set_all_seeds(42)

# Now all random operations will be deterministic
```

## Best Practices

- **Validation**: Always validate input parameters before passing them to models
- **Missing Data**: Use masked operations when working with incomplete data
- **Annotator Agreement**: For datasets with multiple annotators, use `combine_dawid_skene` to estimate true labels
- **Uncertainty**: Employ `calculate_propagated_variance` to quantify predictive uncertainty
- **Augmentation**: Use data augmentation techniques to improve model robustness, especially for small datasets
- **Reproducibility**: Set random seeds with `set_all_seeds` to ensure reproducible experiments
