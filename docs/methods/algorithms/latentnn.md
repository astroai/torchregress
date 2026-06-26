# LatentNN (Latent-Input Neural Regression)

> ← [RC](rc.md) | [Error-Aware Encoding](error_aware.md) →

LatentNN is a lightweight algorithm for regression with **noisy input features** (errors-in-variables). Unlike [SIMEX](simex.md) (which trains multiple models at different noise levels) or [RC](rc.md) (which analytically debiases), LatentNN jointly optimizes **network parameters** and **per-sample latent clean inputs** in a single training loop.

!!! abstract "The idea"
    Keep a learned latent representation $x_{\text{latent}}$ for each training sample that is pulled toward the observed noisy input $x_{\text{obs}}$ via a Gaussian penalty, while the model $f_\theta$ is trained on the cleaner latent inputs.

---

## Mathematical Background

### Joint Objective

LatentNN minimizes a composite loss over model parameters $\theta$ and per-sample latent inputs $X_{\text{latent}}$:

$$\boxed{\;\mathcal{L} = \mathcal{L}_{\text{model}}(f_\theta(x_{\text{latent}}), y) + \lambda_x \cdot \frac{1}{N} \sum_{i=1}^N \left\| \frac{x_{\text{latent}}^{(i)} - x_{\text{obs}}^{(i)}}{\sigma_x^{(i)}} \right\|_2^2\;}$$

where:
- $\mathcal{L}_{\text{model}}$ is any regression loss (MSE, Gaussian NLL, etc.)
- $\lambda_x$ is the `latent_penalty_weight` controlling the tradeoff
- $\sigma_x$ is the known input measurement noise (scalar, per-feature, or per-sample)

### What the Penalty Does

The quadratic penalty acts as a **Gaussian prior** on the latent inputs: $x_{\text{latent}} \sim \mathcal{N}(x_{\text{obs}}, \sigma_x^2)$. When $\sigma_x$ is small (precise measurement), the penalty is strong and $x_{\text{latent}}$ stays close to $x_{\text{obs}}$. When $\sigma_x$ is large (noisy measurement), the penalty relaxes and the model can "denoise" the input.

---

## Usage

### Basic Setup

```python
import torch
import torch.nn as nn
from torchregress.algorithms import LatentNN

# Model factory — returns a fresh model for each fit
def make_model():
    return nn.Sequential(
        nn.Linear(5, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )

latent = LatentNN(
    model_factory=make_model,
    loss_fn=nn.MSELoss(),
    sigma_x=0.5,            # Known measurement noise std
    epochs=500,
    lr=1e-3,
    latent_lr=1e-3,         # Learning rate for latent inputs
    latent_penalty_weight=1.0,
)

# Fit jointly learns latent inputs + model parameters
latent.fit(X_observed, y_observed)

# Predict on new (potentially noisy) data
y_pred = latent.predict(X_test)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `model_factory` | `callable` | — | Returns a freshly initialized `nn.Module` |
| `loss_fn` | `callable` | `MSELoss()` | Loss applied to model predictions |
| `sigma_x` | `float` or `Tensor` | `1.0` | Known input noise. Scalar, per-feature vector `(D,)`, or per-sample matrix `(N, D)`. |
| `sigma_y` | `float` or `Tensor` or `None` | `None` | Known target noise for loss rescaling |
| `epochs` | `int` | `500` | Number of training epochs |
| `lr` | `float` | `1e-3` | Learning rate for model parameters |
| `latent_lr` | `float` or `None` | `None` | Learning rate for latent inputs (defaults to `lr`) |
| `batch_size` | `int` or `None` | `None` | Mini-batch size (defaults to full-batch) |
| `weight_decay` | `float` | `0.0` | Weight decay on model parameters |
| `latent_weight_decay` | `float` | `0.0` | Weight decay on latent inputs |
| `latent_penalty_weight` | `float` | `1.0` | Multiplier $\lambda_x$ on the quadratic penalty |
| `max_grad_norm` | `float` or `None` | `None` | Optional gradient clipping |

### Per-Sample Noise

When different samples have different measurement quality, pass a per-sample $\sigma_x$:

```python
# Each of 1000 samples has 5 features with different noise levels
sigma_x = 0.1 + 0.5 * torch.rand(1000, 5)  # (N, D) per-sample stds

latent = LatentNN(
    model_factory=make_model,
    sigma_x=sigma_x,
    epochs=300,
)
latent.fit(X_observed, y_observed)
```

### Target Noise Rescaling

When targets have known uncertainty $\sigma_y$, the model loss is rescaled. For `MSELoss`, residuals are divided by $\sigma_y$ per sample: $\frac{(\hat{y} - y)^2}{\sigma_y^2}$. For other loss functions, the behavior depends on the loss implementation — verify that your chosen loss function handles per-sample target uncertainty correctly.

```python
sigma_y = 0.05 * torch.ones(1000, 1)  # Known target noise

latent = LatentNN(
    model_factory=make_model,
    loss_fn=nn.MSELoss(),
    sigma_x=0.5,
    sigma_y=sigma_y,    # Rescales MSE by 1/σ_y² per sample
)
```

---

## Comparison with Other EIV Methods

| Method | Training Cost | Handles Nonlinear $f$ | Per-Sample Noise | Latent Recovery |
|:-------|:------------:|:--------------------:|:----------------:|:--------------:|
| **LatentNN** | $1\times$ (single joint training) | ✅ | ✅ | ✅ (per-sample) |
| **RC** | $O(D^3)$ (matrix inversion) | ❌ (linear correction) | ❌ (single $\Sigma_u$) | ❌ |
| **SIMEX** | $(|\lambda| + 1)\times$ | ✅ | ❌ (single $\Sigma_u$) | ❌ |
| **InputNoiseMarginalizationLoss** | $N_{\text{samples}}\times$ | ✅ | ✅ | ❌ |

---

## When to Use

!!! success "LatentNN is a good fit when"
    - Input measurement noise is **known** and potentially different per sample
    - The noise structure is simple (additive Gaussian)
    - You want a **single training run** (unlike SIMEX's multiple models)
    - You need access to the **denoised latent inputs** after training (`latent.x_latent_`)

!!! warning "LatentNN is not ideal when"
    - Input noise is **unknown** — the algorithm requires $\sigma_x$ as input
    - The data has complex noise (non-Gaussian, correlated across features) — consider `ErrorAwareFeatureEncoder`
    - The model is extremely large — storing per-sample latent inputs adds $N \times D$ parameters

---

## Limitations

!!! warning "Practical constraints"
    - **Noise must be known**: `sigma_x` is a required input. If you don't know the measurement noise, LatentNN can't help — the latent inputs will drift toward whatever values minimize the model loss without meaningful denoising.
    - **Per-sample storage**: LatentNN stores one latent vector per training sample ($N \times D$ parameters), which can exceed the model parameter count for large $N$.
    - **Gaussian noise assumption**: The quadratic penalty corresponds to a Gaussian likelihood for $x_{\text{obs}} \mid x_{\text{latent}}$. For non-Gaussian noise (e.g., quantization error, Poisson counts), the penalty form is suboptimal.
    - **Validation-aware fit**: The `.fit()` method tracks validation loss and restores the best model state, but the latent inputs are not validated — the best model checkpoint may be paired with suboptimal latent inputs if the validation set has different noise characteristics.

---

## Complete Example

```python
import torch
import torch.nn as nn
from torchregress.algorithms import LatentNN

torch.manual_seed(42)

# True clean data
X_true = torch.randn(200, 3)
y_true = X_true @ torch.tensor([1.5, -0.8, 0.3]) + 0.1 * torch.randn(200)

# Add measurement noise to inputs
X_obs = X_true + 0.4 * torch.randn(200, 3)

def make_model():
    return nn.Sequential(
        nn.Linear(3, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 1),
    )

latent = LatentNN(
    model_factory=make_model,
    sigma_x=0.4,
    epochs=300,
    lr=1e-3,
    latent_penalty_weight=1.0,
)

# Fit — jointly learns model and latent clean inputs
latent.fit(X_obs, y_true.unsqueeze(1))

# Access the learned latent inputs
print(f"Latent input shape: {latent.x_latent_.shape}")

# Denoising quality: how close are latent inputs to true clean inputs?
denoising_mae = (latent.x_latent_ - X_true).abs().mean()
print(f"Denoising MAE: {denoising_mae:.4f} (obs MAE: {(X_obs - X_true).abs().mean():.4f})")

# Predict on test data
X_test = torch.randn(50, 3) + 0.3 * torch.randn(50, 3)
y_pred = latent.predict(X_test)
```

---

## Next steps

- [SIMEX](simex.md) — multi-model extrapolation baseline; compare single-run LatentNN vs multi-model cost
- [Regression Calibration](rc.md) — fast matrix-based correction when a linear model suffices
- [Error-Aware Encoding](error_aware.md) — lightweight alternative with no per-sample parameter storage
- [EIV losses](../../losses/eiv.md) — loss-level handling when you prefer not to learn latent inputs

---

## References

| # | Reference |
|:-:|:----------|
| 1 | R.J. Carroll, D. Ruppert, L.A. Stefanski, C.M. Crainiceanu. *Measurement Error in Nonlinear Models*. 2nd ed., Chapman & Hall/CRC, **2006**. |
| 2 | S. Fabbro et al. "Latent-input neural regression for tabular errors-in-variables." *torchregress documentation*, **2025**. |
