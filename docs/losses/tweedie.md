# Tweedie Loss Functions

Tweedie loss functions are derived from the Tweedie family of distributions, which include many common distributions as special cases. These losses are particularly valuable for modeling data with complex variance-mean relationships.

## Mathematical Background

The Tweedie distribution is defined by its variance function:

$$\text{Var}(Y) = \phi \cdot \mu^p$$

Where:
- $\mu$ is the mean
- $\phi$ is the dispersion parameter
- $p$ is the power parameter

This power parameter $p$ determines the specific distribution:
- $p = 0$: Normal distribution
- $p = 1$: Poisson distribution
- $p = 2$: Gamma distribution
- $p = 3$: Inverse Gaussian distribution
- $1 < p < 2$: Compound Poisson-Gamma (handles continuous data with exact zeros)

## Distribution Selection Guide

| Data Characteristics | Recommended Distribution | Power Parameter | Recommended Loss |
|----------------------|--------------------------|----------------|------------------|
| Symmetric, no constraints | Normal | p=0 | `TweedieLoss(p=0)` |
| Count data | Poisson | p=1 | `TweedieLoss(p=1)` |
| Positive continuous | Gamma | p=2 | `GammaLoss` or `TweedieLoss(p=2)` |
| Highly skewed positive | Inverse Gaussian | p=3 | `InverseGaussianLoss` or `TweedieLoss(p=3)` |
| Continuous with exact zeros | Compound Poisson | 1<p<2 | `CompoundPoissonLoss` or `TweedieLoss(p=1.5)` |

## Available Tweedie Losses

### TweedieLoss

```python
class TweedieLoss(RegressionLoss)
```

General-purpose Tweedie loss function for regression with configurable power parameter.

**Parameters:**

- `p` (float, optional): Power parameter defining the variance function V(μ) = μ^p. Default: `1.5`
- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `link` (str, optional): Link function, 'log' or 'identity'. Default is 'log' for p>=1, 'identity' for p=0

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the Tweedie loss

**Mathematical Formulation:**

The Tweedie deviance (loss) depends on the power parameter $p$:

For $p = 0$ (Normal):
$$\mathcal{L}(y, \mu) = \frac{1}{2}(y - \mu)^2$$

For $p = 1$ (Poisson):
$$\mathcal{L}(y, \mu) = \mu - y\log(\mu) + \log(y!)$$

For $p = 2$ (Gamma):
$$\mathcal{L}(y, \mu) = \log\left(\frac{\mu}{y}\right) + \frac{y}{\mu} - 1$$

For $1 < p < 2$ (Compound Poisson-Gamma):
$$\mathcal{L}(y, \mu) = \frac{2}{(2-p)(1-p)}\left[y^{2-p} - (2-p)y\mu^{1-p} + (1-p)\mu^{2-p}\right]$$

**Example:**

```python
import torch
import torchregress as tr

# For regression with compound Poisson-Gamma distribution (p=1.5)
loss_fn = tr.losses.TweedieLoss(p=1.5, link='log')

# Model predicts log of mean parameter
y_pred = torch.tensor([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]])  # log(μ) values
target = torch.tensor([[0.0, 2.0, 8.0], [1.0, 4.0, 12.0]])  # response values

# Calculate loss
loss = loss_fn(y_pred, target)

# With masking (ignore some values)
mask = torch.tensor([[True, True, False], [True, False, True]])
masked_loss = loss_fn(y_pred, target, mask=mask)
```

### GammaLoss

```python
class GammaLoss(TweedieLoss)
```

Specialized loss for gamma regression (p=2), suitable for positive continuous response variables with constant coefficient of variation.

**Parameters:**

- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `link` (str, optional): Link function, 'log' or 'identity'. Default: 'log'

**Mathematical Formulation:**

$$\mathcal{L}(y, \mu) = \log\left(\frac{\mu}{y}\right) + \frac{y}{\mu} - 1$$

**Example:**

```python
import torch
import torchregress as tr

# For modeling positive continuous data like prices or durations
loss_fn = tr.losses.GammaLoss(link='log')

# Model predicts log(μ)
y_pred = torch.tensor([[1.0, 2.0, 3.0]])  # log(μ) values
target = torch.tensor([[3.0, 7.5, 20.0]])  # positive response values

# Calculate loss
loss = loss_fn(y_pred, target)
```

### InverseGaussianLoss

```python
class InverseGaussianLoss(TweedieLoss)
```

Loss function for inverse Gaussian regression (p=3), suitable for positive data with variance that increases with the cube of the mean.

**Parameters:**

- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `link` (str, optional): Link function, 'log' or 'identity'. Default: 'log'

**Mathematical Formulation:**

$$\mathcal{L}(y, \mu) = \frac{(y - \mu)^2}{y \mu^2}$$

**Example:**

```python
import torch
import torchregress as tr

# For highly skewed positive data
loss_fn = tr.losses.InverseGaussianLoss(link='log')

# Model predicts log(μ)
y_pred = torch.log(torch.tensor([[2.0, 5.0, 10.0]]))
target = torch.tensor([[2.0, 5.0, 10.0]])

# Calculate loss
loss = loss_fn(y_pred, target)
```

### CompoundPoissonLoss

```python
class CompoundPoissonLoss(TweedieLoss)
```

Specialized loss for compound Poisson-Gamma regression (1<p<2), suitable for modeling continuous data with exact zeros, such as insurance claims or rainfall amounts.

**Parameters:**

- `p` (float, optional): Power parameter between 1 and 2. Default: `1.5`
- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `reduction` (str, optional): Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'
- `link` (str, optional): Link function, 'log' or 'identity'. Default: 'log'

**Example:**

```python
import torch
import torchregress as tr

# For insurance claims or rainfall data with many zeros
loss_fn = tr.losses.CompoundPoissonLoss(p=1.6)

# Model predicts log(μ)
y_pred = torch.tensor([[0.0, 1.0, 2.0, 3.0]])  # log(μ) values
target = torch.tensor([[0.0, 0.0, 5.0, 15.0]])  # many zeros, some positive values

# Calculate loss
loss = loss_fn(y_pred, target)
```

## When to Use Tweedie Loss

Tweedie distributions are particularly useful in these scenarios:

1. **Insurance Modeling**: 
   - Claim frequency and severity (typically p=1.5 to 1.8)
   - Premium calculation and risk modeling

2. **Environmental Science**:
   - Rainfall amounts (zeros for dry days, continuous for rainy days)
   - Pollution levels with detection thresholds

3. **Economics**:
   - Household expenditures (zeros for non-participants)
   - Asset returns with concentration at zero

4. **Ecology**:
   - Species abundance data
   - Biomass measurements

## Practical Considerations

1. **Link Function**: 
   - Use 'log' link for p>=1 to ensure positivity constraints
   - Use 'identity' link for p=0 (Normal distribution)

2. **Model Output**:
   - With log link, your model outputs log(μ)
   - With identity link, your model directly outputs μ

3. **Power Parameter Selection**:
   - For known distributions, use the corresponding p
   - For unknown distributions, p can be estimated using profile likelihood
   - Typical values for mixed discrete-continuous data range from 1.1 to 1.7

4. **Numerical Stability**:
   - Tweedie distributions with 1<p<2 may have numerical issues for very small values
   - The eps parameter helps maintain stability

## Related Loss Functions

For pure count data (without the continuous component), consider using Poisson losses:

[Learn more about Poisson losses →](poisson.md)

For scientific imaging applications with mixed noise sources, consider:

[Learn more about Poisson-Gaussian mixture losses →](poisson_gaussian.md)
