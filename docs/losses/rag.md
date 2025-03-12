# Regression-as-Classification Losses

Regression-as-Classification (RAG) losses are designed for converting continuous regression problems into classification problems through binning. This approach offers several advantages over traditional regression:

- Better uncertainty quantification
- Handling multi-modal target distributions
- More flexible loss functions
- Greater robustness to noisy labels
- Ability to capture complex output distributions

## Mathematical Background

The regression-as-classification approach discretizes the continuous output space into bins and frames prediction as estimating a probability distribution over these bins:

$$p(y \in \text{bin}_i | x) = \hat{p}_i(x)$$

Where:
- $\hat{p}_i(x)$ is the predicted probability for bin $i$ given input $x$
- The final prediction can be computed as the expected value: $\hat{y} = \sum_i \text{center}_i \cdot \hat{p}_i(x)$

The target values are mapped to either:
1. Hard targets (one-hot encoding based on bin membership)
2. Soft targets (probability distribution across bins, typically using a Gaussian kernel)

For soft targets with a Gaussian kernel:

$$p(\text{bin}_i|y) = \frac{\exp(-\frac{1}{2}\frac{(y - \text{center}_i)^2}{\sigma^2})}{\sum_j \exp(-\frac{1}{2}\frac{(y - \text{center}_j)^2}{\sigma^2})}$$

Where $\sigma$ controls the smoothness of the distribution.

## Available Regression-as-Classification Losses

### StandardClassificationRegressionLoss

```python
class StandardClassificationRegressionLoss(BinnedRegressionLoss)
```

Standard classification-based regression loss that treats regression as a classification problem by binning continuous values and applying classification loss functions.

**Parameters:**

- `bins` (Union[int, torch.Tensor]): Number of bins or array of bin edges. Default: `10`
- `min_value` (float, optional): Minimum value for auto-generated bins. Default: `0.0`
- `max_value` (float, optional): Maximum value for auto-generated bins. Default: `1.0`
- `soft_targets` (bool): Whether to use soft targets (probability distributions). Default: `True`
- `sigma` (float): Standard deviation for soft targets. Default: `0.1`
- `reduction` (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `label_smoothing` (float): Smoothing factor in [0, 1] for hard targets. Default: `0.0`
- `loss_type` (str): Type of loss ('cross_entropy', 'kl_div', 'focal', 'nll'). Default: 'cross_entropy'
- `focal_gamma` (float): Gamma parameter for focal loss. Default: `2.0`

**Methods:**

- `forward(y_pred, target, mask=None, weights=None, uncertainty=None)`: Computes the loss
- `_extract_distribution_parameters(y_pred)`: Extract distribution parameters from predicted logits
- `_calculate_nll(y_true, params, mask=None)`: Calculate negative log-likelihood

Mathematically, with soft targets and cross-entropy loss:

$$\mathcal{L}_{\text{CE}}(y, \hat{y}) = -\sum_i p(\text{bin}_i|y) \log(q(\text{bin}_i|x))$$

where $p(\text{bin}_i|y)$ is the target distribution and $q(\text{bin}_i|x)$ is the predicted distribution.

**Example:**

```python
import torch
import torchregression as tr

# Create standard classification regression loss
loss_fn = tr.losses.StandardClassificationRegressionLoss(
    bins=15,
    min_value=0.0,
    max_value=10.0,
    soft_targets=True,
    sigma=0.1,
    loss_type='cross_entropy'
)

# Model output should be logits of shape [batch_size, n_bins]
y_pred = torch.randn(32, 15)  # logits
target = torch.randn(32, 1)   # continuous values

# Calculate loss
loss = loss_fn(y_pred, target)

# With mask (ignore some measurements)
mask = torch.ones(32).bool()
mask[5:10] = False
masked_loss = loss_fn(y_pred, target, mask=mask)
```

### OrdinalRegressionLoss

```python
class OrdinalRegressionLoss(BinnedRegressionLoss)
```

Ordinal regression loss for handling ordered categories, which uses binary encoding to represent the ordinal relationship between bins, potentially improving performance for regression tasks.

**Parameters:**

- `bins` (Union[int, torch.Tensor]): Number of bins or array of bin edges. Default: `10`
- `min_value` (float, optional): Minimum value for auto-generated bins. Default: `0.0`
- `max_value` (float, optional): Maximum value for auto-generated bins. Default: `1.0`
- `soft_targets` (bool): Whether to use soft targets (probability distributions). Default: `True`
- `sigma` (float): Standard deviation for soft targets. Default: `0.1`
- `reduction` (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `loss_type` (str): Type of loss to use ('bce', 'focal'). Default: 'bce'
- `focal_gamma` (float): Gamma parameter for focal loss. Default: `2.0`

**Methods:**

- `forward(y_pred, target, mask=None, weights=None, uncertainty=None)`: Computes the loss
- `_extract_distribution_parameters(y_pred)`: Extract distribution parameters from binary logits
- `_calculate_nll(y_true, params, mask=None)`: Calculate ordinal regression loss

Ordinal regression converts the problem into a series of binary classifications. For $n$ bins, we have $n-1$ thresholds, where for a value in bin $i$:
- Thresholds $k < i$ are assigned value 1
- Thresholds $k \geq i$ are assigned value 0

**Example:**

```python
import torch
import torchregression as tr

# Create ordinal regression loss
loss_fn = tr.losses.OrdinalRegressionLoss(
    bins=10,
    min_value=0.0,
    max_value=5.0,
    soft_targets=True,
    sigma=0.1,
    loss_type='bce'
)

# Model output should be binary logits of shape [batch_size, n_bins-1]
# Each output corresponds to P(value > threshold_k)
y_pred = torch.randn(32, 9)  # binary logits for 9 thresholds (10 bins)
target = torch.randn(32, 1)  # continuous values

# Calculate loss
loss = loss_fn(y_pred, target)
```

### HistogramRegressionLoss

```python
class HistogramRegressionLoss(BinnedRegressionLoss)
```

Histogram regression loss for flexibility in capturing output distributions, which treats the output as a histogram (probability distribution) over bins. This is particularly useful for multi-modal distributions and uncertainty estimation.

**Parameters:**

- `bins` (Union[int, torch.Tensor]): Number of bins or array of bin edges. Default: `10`
- `min_value` (float, optional): Minimum value for auto-generated bins. Default: `0.0`
- `max_value` (float, optional): Maximum value for auto-generated bins. Default: `1.0`
- `soft_targets` (bool): Whether to use soft targets (probability distributions). Default: `True`
- `sigma` (float): Standard deviation for soft targets. Default: `0.1`
- `reduction` (str): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `loss_type` (str): Type of loss ('kl_div', 'cross_entropy', 'wasserstein'). Default: 'kl_div'
- `normalize_targets` (bool): Whether to normalize target distributions. Default: `True`
- `wasserstein_p` (int): P parameter for Wasserstein distance. Default: `1`

**Methods:**

- `forward(y_pred, target, mask=None, weights=None, uncertainty=None)`: Computes the loss
- `_extract_distribution_parameters(y_pred)`: Extract distribution parameters from predicted logits or probabilities
- `_calculate_nll(y_true, params, mask=None)`: Calculate histogram loss

For a histogram regression with KL-divergence loss:

$$\mathcal{L}_{\text{KL}}(p, q) = \sum_i p(\text{bin}_i|y) \log\frac{p(\text{bin}_i|y)}{q(\text{bin}_i|x)}$$

For Wasserstein distance with p=1 (W1):

$$\mathcal{L}_{\text{W1}}(p, q) = \sum_i |CDF_p(\text{bin}_i) - CDF_q(\text{bin}_i)|$$

**Example:**

```python
import torch
import torchregression as tr

# Create histogram regression loss
loss_fn = tr.losses.HistogramRegressionLoss(
    bins=20,
    min_value=-5.0,
    max_value=5.0,
    soft_targets=True,
    sigma=0.1,
    loss_type='wasserstein',
    wasserstein_p=1
)

# Model output can be either logits or probabilities [batch_size, n_bins]
y_pred = torch.softmax(torch.randn(32, 20), dim=1)  # probabilities
target = torch.randn(32, 1)  # continuous values

# Calculate loss
loss = loss_fn(y_pred, target)
```

### RegressionAsClassificationLoss

```python
class RegressionAsClassificationLoss(BinnedRegressionLoss)
```

Unified regression-as-classification loss that combines the benefits of histogram binning, ordinal regression, and soft targets to create a powerful and flexible approach for regression problems.

**Parameters:**

- `bins` (Union[int, torch.Tensor]): Number of bins or array of bin edges. Default: `15`
- `min_value` (float, optional): Minimum value for auto-generated bins. Default: `0.0`
- `max_value` (float, optional): Maximum value for auto-generated bins. Default: `1.0`
- `order_aware` (bool): Whether to use ordinal encoding to encode bin ordering. Default: `True`
- `smooth_targets` (bool): Whether to use smooth target distributions. Default: `True`
- `sigma` (float): Standard deviation for soft targets. Default: `0.1`
- `reduction` (str): Method for reducing the loss. Default: 'mean'
- `loss_type` (str): Type of loss function ('cross_entropy', 'kl_div', 'focal', 'wasserstein'). Default: 'cross_entropy'
- `adaptive_sigma` (bool): Whether to adjust sigma based on bin widths. Default: `True`
- `focal_gamma` (float): Gamma parameter for focal loss. Default: `2.0`

**Methods:**

- `forward(y_pred, target, mask=None, weights=None, uncertainty=None)`: Computes the loss
- `_extract_distribution_parameters(y_pred)`: Extract distribution parameters from model outputs

This unified approach supports both classification-style and ordinal-style regression:
- For standard mode (n_bins outputs): Uses classification-style losses
- For ordinal mode (n_bins-1 outputs): Uses ordinal binary encoding

**Example:**

```python
import torch
import torchregression as tr

# Create unified regression-as-classification loss
loss_fn = tr.losses.RegressionAsClassificationLoss(
    bins=15,
    min_value=0.0,
    max_value=10.0,
    order_aware=True,
    smooth_targets=True,
    sigma=0.1,
    loss_type='cross_entropy',
    adaptive_sigma=True
)

# For standard classification mode [batch_size, n_bins]
y_pred_standard = torch.randn(32, 15)  # logits
loss_standard = loss_fn(y_pred_standard, target)

# For ordinal regression mode [batch_size, n_bins-1]
# (if order_aware=True)
y_pred_ordinal = torch.randn(32, 14)  # binary logits
loss_ordinal = loss_fn(y_pred_ordinal, target)
```

## Factory Functions

TorchRegression provides several factory functions to simplify the creation of regression-as-classification losses:

```python
# Create a binned regression loss with automatic method selection
loss_fn = tr.losses.create_binned_regression_loss(
    method='auto',  # 'auto', 'classification', 'ordinal', or 'histogram'
    bins=20,
    min_value=0.0,
    max_value=10.0,
    data=training_data,  # optional data for auto bin range detection
    noise_aware=True
)

# High-level simplified interface
loss_fn = tr.losses.regression_as_classification(
    bins=15,
    min_value=None,  # auto-detect from data
    max_value=None,  # auto-detect from data
    smooth_targets=True,
    robust_to_noise=False,
    auto_adapt=True
)

# Specialized for uncertainty quantification
loss_fn = tr.losses.uncertainty_regression(
    bins=20,
    min_value=None,
    max_value=None
)
```

## When to Use Regression-as-Classification Losses

1. **Multi-Modal Targets**: When the conditional distribution of the target given the input is multi-modal

2. **Uncertainty Quantification**: When you need to capture complex uncertainty estimates beyond simple Gaussian assumptions

3. **Noisy Labels**: When dealing with datasets that have noisy target values

4. **Non-Gaussian Distributions**: When the target distribution is skewed, heavy-tailed, or has other non-Gaussian characteristics

5. **Bounded Outputs**: When your output has natural boundaries (e.g., percentages, ratings on a fixed scale)

6. **Ordinal Regression**: When the ordering relationship between outputs is important

## Mathematical Insights

1. **Distribution Flexibility**: Unlike parametric distributional approaches (e.g., Gaussian NLL), binned regression can represent arbitrary distributions, including skewed, multi-modal, and bounded distributions.

2. **Smoothness Control**: The `sigma` parameter controls the smoothness of target distributions and implicitly regularizes the model. Smaller values create sharper targets, while larger values create smoother distributions.

3. **Uncertainty Representation**: The output distribution directly provides an uncertainty estimate. The entropy of the predicted distribution indicates prediction uncertainty.

4. **Ordinal vs. Standard**: Ordinal encoding captures the natural ordering of bins and typically works better for truly continuous outputs, while standard classification is more flexible for multi-modal distributions.

5. **Wasserstein Distance**: When using the Wasserstein loss type, the model is penalized based on the "earth-moving distance" between distributions, which respects the natural ordering of bins.

## Tips for Implementation

1. **Bin Selection**: The number and placement of bins significantly impacts performance. Use more bins for higher precision, but beware of overfitting with too many bins.

2. **Model Architecture**: For standard classification mode, end your model with a linear layer of size `n_bins`. For ordinal mode, use `n_bins-1` outputs.

3. **Output Activation**: No activation is needed on the model outputs as the loss functions handle the necessary transformations.

4. **Adaptive Sigma**: Enable `adaptive_sigma=True` to automatically adjust the smoothness based on bin widths.

5. **Inference**: Use the provided `decode_prediction` method to convert the predicted distribution back to a continuous value:

```python
# Get continuous prediction from distribution
y_pred_logits = model(x)
continuous_pred = loss_fn.decode_prediction(y_pred_logits)
```

6. **Uncertainty Extraction**: Extract uncertainty information from the predicted distribution:

```python
# Get full distribution
distribution = loss_fn.get_distribution(y_pred_logits)

# Calculate entropy as uncertainty measure
probs = distribution['bin_probs']
entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
```

7. **Data-driven Bin Range**: Set `min_value=None` and `max_value=None` and provide training data to automatically determine appropriate bin ranges.
