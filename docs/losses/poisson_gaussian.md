# Poisson-Gaussian Mixture Losses

Poisson-Gaussian mixture losses are specialized functions that model signals as a combination of Poisson (counting) noise and Gaussian (readout) noise. These losses are particularly valuable for:

- Scientific imaging (astronomy, microscopy)
- Medical imaging (CT scans, PET, low-dose X-ray)
- Low-light photography 
- Any signal processing application where both shot noise and electronic noise are present

## Mathematical Background

Many scientific measurements, particularly in imaging and signal detection, involve both discretized counting processes (Poisson) and additive electronic noise (Gaussian). The resulting signal can be modeled as:

$$Y = \alpha \cdot P(\lambda) + N(0, \sigma^2)$$

Where:
- $P(\lambda)$ is a Poisson random variable with rate $\lambda$
- $N(0, \sigma^2)$ is a Gaussian random variable with mean 0 and variance $\sigma^2$
- $\alpha$ is a scaling factor (gain)

The negative log-likelihood for this mixed model doesn't have a simple closed form, but there are several effective approximations implemented in TorchRegression.

## Available Poisson-Gaussian Losses

### PoissonGaussianMixtureLoss

```python
class PoissonGaussianMixtureLoss(RegressionLoss)
```

Negative log-likelihood loss for a mixture of Gaussian (readout noise) and Poisson (count) noise, common in imaging and signal processing applications.

**Parameters:**

- `eps` (float, optional): Small constant for numerical stability. Default: `1e-8`
- `learn_variance` (bool, optional): Whether to learn the Gaussian variance parameter. Default: `False`
- `initial_variance` (float, optional): Initial value for Gaussian variance. Default: `1.0`
- `min_variance` (float, optional): Minimum variance value for numerical stability. Default: `1e-6`
- `log_input` (bool, optional): Whether y_pred is provided as log(lambda). Default: `False`
- `mixture_weights` (optional): How to weight the mixture components:
  - If None: Equal weighting (0.5, 0.5)
  - If float: Fixed weighting (mixture_weights, 1-mixture_weights)
  - If 'learn': Learn the mixture weights
- `extra_variance_model` (bool, optional): Whether to include a separate learned variance term. Default: `False`
- `reduction` (str, optional): Method for loss reduction. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None, extra_var=None)`: Computes the mixture loss

The loss combines Poisson and Gaussian negative log-likelihoods using specified or learned weights:

$$\mathcal{L}_{\text{mixture}} = w_P \cdot \mathcal{L}_{\text{Poisson}}(y, \lambda) + w_G \cdot \mathcal{L}_{\text{Gaussian}}(y, \lambda, \sigma^2)$$

Where:
- $w_P$ and $w_G$ are the weights for Poisson and Gaussian components
- $\lambda$ is the predicted mean/rate parameter
- $\sigma^2$ is the Gaussian variance (fixed or learned)

**Example:**

```python
import torch
import torchregress as tr

# For imaging data with fixed noise variance
loss_fn = tr.losses.PoissonGaussianMixtureLoss(
    initial_variance=0.1,  # Set based on estimated readout noise
    log_input=True,        # Model outputs log(lambda)
    mixture_weights=0.7    # 70% Poisson, 30% Gaussian weighting
)

# Predicted signals (log-intensity)
y_pred = torch.log(torch.tensor([[10.0, 20.0], [5.0, 15.0]]))
# Measured signals
target = torch.tensor([[11.2, 19.5], [4.8, 16.3]])

# Calculate loss
loss = loss_fn(y_pred, target)

# With learned mixture weights
loss_fn_learned = tr.losses.PoissonGaussianMixtureLoss(
    learn_variance=True,
    mixture_weights='learn'
)
loss = loss_fn_learned(y_pred, target)
```

### EnhancedPoissonGaussianMixtureLoss

```python
class EnhancedPoissonGaussianMixtureLoss(RegressionLoss)
```

Advanced Poisson-Gaussian mixture loss with additional features for scientific applications, including gain/scaling factor, offset/bias term, and multiple variance components.

**Parameters:**

- `gain` (float or 'learn', optional): Fixed gain/scaling factor or 'learn' to make it learnable. Default: `1.0`
- `offset` (float or 'learn', optional): Fixed offset/bias or 'learn' to make it learnable. Default: `0.0`
- `read_noise` (float or 'learn', optional): Constant variance component (σ₁²). Default: `1.0`
- `shot_noise` (float or 'learn', optional): Signal-dependent variance component (σ₂²). Default: `0.0`
- `log_input` (bool, optional): Whether inputs are in log space. Default: `False`
- `calibration` (bool, optional): Whether to include calibration parameters. Default: `False`
- `reduction` (str, optional): Method for loss reduction. Default: 'mean'

**Methods:**

- `forward(y_pred, target, mask=None, weights=None)`: Computes the enhanced mixture loss

This loss implements the physically motivated model:

$$y \sim \text{Poisson}(g \cdot \lambda + b) + N(0, \sigma_r^2 + \sigma_s^2 \cdot \lambda)$$

Where:
- $g$ is the gain factor
- $b$ is the offset/bias
- $\sigma_r^2$ is the read noise (constant variance)
- $\sigma_s^2$ is the shot noise coefficient (signal-dependent variance)
- $\lambda$ is the predicted intensity

**Example:**

```python
import torch
import torchregress as tr

# For scientific imaging with signal-dependent noise
loss_fn = tr.losses.EnhancedPoissonGaussianMixtureLoss(
    gain='learn',         # Learn the gain factor
    read_noise=0.2,       # Fixed read noise variance
    shot_noise='learn',   # Learn the shot noise coefficient
    log_input=True        # Model outputs log(lambda)
)

# Predicted log-intensities
y_pred = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
# Measured values
target = torch.tensor([[2.8, 7.3, 19.5], [7.4, 20.1, 55.2]])

# Calculate loss
loss = loss_fn(y_pred, target)

# With calibration parameters
loss_fn_calib = tr.losses.EnhancedPoissonGaussianMixtureLoss(
    gain=1.0,
    read_noise=0.5,
    shot_noise=0.1,
    calibration=True  # Add learnable calibration parameters
)
loss = loss_fn_calib(y_pred, target)
```

## Factory Functions

### poisson_gaussian_mixture_loss

```python
poisson_gaussian_mixture_loss(learn_variance=False, initial_variance=1.0,
                             log_input=False, mixture_weights=None,
                             extra_variance_model=False, **kwargs)
```

Create a Poisson-Gaussian mixture loss function with the specified parameters.

**Example:**

```python
# Create a loss with learnable mixture weights
loss_fn = tr.losses.poisson_gaussian_mixture_loss(
    learn_variance=True,
    mixture_weights='learn',
    log_input=True
)
```

### enhanced_poisson_gaussian_loss

```python
enhanced_poisson_gaussian_loss(**kwargs)
```

Create an enhanced Poisson-Gaussian mixture loss with the specified parameters.

**Example:**

```python
# Create a loss with learnable gain and signal-dependent noise
loss_fn = tr.losses.enhanced_poisson_gaussian_loss(
    gain='learn',
    read_noise=0.1,
    shot_noise='learn',
    log_input=True
)
```

## When to Use Poisson-Gaussian Losses

1. **Scientific Imaging**: When dealing with photon-counting devices (CCDs, PMTs) with electronic readout noise

2. **Low-Count Scenarios with Noise**: When modeling count data contaminated by additive measurement noise

3. **Calibrated Systems**: When you have knowledge about the gain, offset, and noise characteristics of your measurement system

4. **Data with Signal-Dependent Noise**: When the noise variance increases with signal intensity

## Mathematical Insights

1. **Noise Regimes**: 
   - At high counts, the Poisson component dominates (shot noise)
   - At low counts, the Gaussian component can dominate (read noise)

2. **Variance Structure**: The total variance in a Poisson-Gaussian model is:
   
   $$\text{Var}(Y) = g^2 \cdot \lambda + \sigma^2$$

   which naturally models heteroscedasticity with both constant and signal-dependent terms.

3. **SNR Consideration**: Signal-to-noise ratio improves with square root of intensity in the Poisson-dominated regime, and linearly with intensity in the Gaussian-dominated regime.

4. **Well-Posedness**: Including gain and offset terms can make the inverse problem better conditioned, especially in low-light imaging.

## Practical Applications

### Astronomy

```python
# For astronomical image processing
astro_loss = tr.losses.EnhancedPoissonGaussianMixtureLoss(
    gain=2.5,            # e-/ADU conversion
    read_noise=10.0,     # Read noise in e-
    shot_noise=1.0,      # Shot noise coefficient 
    offset=100.0,        # Bias level
    log_input=True       # Model predicts log(photon counts)
)
```

### Microscopy

```python
# For fluorescence microscopy
microscopy_loss = tr.losses.EnhancedPoissonGaussianMixtureLoss(
    gain='learn',        # Camera gain will be learned
    read_noise='learn',  # Camera read noise will be learned
    shot_noise=1.0,      # Pure Poisson shot noise
    calibration=True     # Add calibration parameters
)
```

### Medical Imaging

```python
# For low-dose X-ray imaging
xray_loss = tr.losses.PoissonGaussianMixtureLoss(
    learn_variance=True,
    initial_variance=0.5,
    mixture_weights=0.8  # Emphasize Poisson component
)
```
