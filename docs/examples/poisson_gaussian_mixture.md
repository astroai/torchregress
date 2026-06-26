# Poisson-Gaussian Mixture Examples

This guide demonstrates how to model data affected by both count (Poisson/shot) noise and continuous (Gaussian/readout) noise using `PoissonGaussianMixtureLoss` and `EnhancedPoissonGaussianMixtureLoss`.

→ API: [`PoissonGaussianMixtureLoss`](../api/losses.md).

| # | Reference |
|:-:|:----------|
| 1 | Foi, A., Trimeche, M., Katkovnik, V., & Egiazarian, K. (2008). [**Practical Poisson-Gaussian noise modeling and sensor calibration in single-image photometry**](https://doi.org/10.1109/TIP.2008.2001397). *IEEE Transactions on Image Processing*. |
| 2 | Luisier, F., Blu, T., & Unser, M. (2011). [**Image denoising in mixed Poisson–Gaussian noise**](https://doi.org/10.1109/TIP.2010.2052277). *IEEE Transactions on Image Processing*. |

---

## Mathematical Formulations

Poisson-Gaussian mixtures model a physical measurement $Y$ as:
$$Y = Z + E$$
where:
*   $Z \sim \text{Poisson}(\lambda)$ represents the photon counting or count-based shot noise.
*   $E \sim \mathcal{N}(0, \sigma^2)$ represents the electronic readout noise.

### Poisson-Gaussian Mixture Loss

The basic `PoissonGaussianMixtureLoss` computes a weighted combination of the individual negative log-likelihoods:
$$L(y, \lambda) = w_{\text{Poisson}} \cdot \text{NLL}_{\text{Poisson}}(y; \lambda) + w_{\text{Gaussian}} \cdot \text{NLL}_{\text{Gaussian}}(y; \lambda, \sigma^2)$$

### Enhanced Poisson-Gaussian Mixture Loss

The `EnhancedPoissonGaussianMixtureLoss` incorporates physical calibration parameters:
$$y \sim \text{Poisson}(g \cdot \lambda + b) + \mathcal{N}(0, \sigma_1^2 + \sigma_2^2 \cdot \lambda)$$
where:
*   $g$ is the sensor gain (scaling factor).
*   $b$ is the sensor offset (bias term).
*   $\sigma_1^2$ is the read noise variance (constant).
*   $\sigma_2^2$ is the shot noise variance coefficient (signal-dependent).

---

## Code Example

Below is a complete example showing how to train a model predicting a latent rate parameter under mixed noise.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchregress.losses import PoissonGaussianMixtureLoss, EnhancedPoissonGaussianMixtureLoss

# Generate synthetic Poisson-Gaussian noise data
np.random.seed(42)
n_samples = 1000
sigma = 0.6

x = np.random.uniform(-1.5, 1.5, size=(n_samples, 1)).astype(np.float32)
true_rate = np.exp(0.8 * x + 1.2).astype(np.float32)

# Count shot noise + Gaussian readout noise
shot = np.random.poisson(true_rate).astype(np.float32)
readout = np.random.normal(0, sigma, size=shot.shape).astype(np.float32)
y = shot + readout

x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)

# Neural network predicting latent rate
class SignalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# Train with learnable Gaussian variance
model = SignalModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = PoissonGaussianMixtureLoss(learn_variance=True, log_input=False)

for epoch in range(150):
    rate_pred = torch.exp(model(x_tensor))
    loss = loss_fn(rate_pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Poisson-Gaussian mixture model trained successfully.")
```
