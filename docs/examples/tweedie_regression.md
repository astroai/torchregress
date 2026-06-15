# Tweedie Regression and Compound Poisson-Gamma Examples

This guide demonstrates how to perform Tweedie regression for modeling zero-inflated continuous responses (e.g., rainfall, insurance claims) using Tweedie and Compound Poisson losses.

→ API: [`TweedieLoss`](../api/losses.md#tweedieloss).

| # | Reference |
|:-:|:----------|
| 1 | Jorgensen, B. (1987). [**Exponential Dispersion Models**](https://www.jstor.org/stable/2345173). *Journal of the Royal Statistical Society: Series B (Methodological)*. |
| 2 | Ohlsson, E., & Johansson, B. (2010). [**Non-Life Insurance Pricing with Generalized Linear Models**](https://doi.org/10.1007/978-3-642-10791-7). *Springer*. |

---

## Mathematical Formulations

The Tweedie distribution is a family of exponential dispersion models where the variance $V(\mu)$ is proportional to a power $p$ of the mean $\mu$:
$$\text{Var}(Y) = \phi \mu^p$$
where $\phi > 0$ is the dispersion parameter and $p$ is the Tweedie power parameter.

For the compound Poisson-Gamma case ($1 < p < 2$), the target variable $Y$ can be represented as:
$$Y = \sum_{j=1}^N X_j$$
where $N \sim \text{Poisson}(\lambda)$ is the number of events, and $X_j \sim \text{Gamma}(k, \theta)$ are independent and identically distributed event sizes.

The deviance loss $d(y, \mu)$ for $1 < p < 2$ is:
$$d(y, \mu) = 2 \left( \frac{y^{2-p}}{(1-p)(2-p)} - \frac{y \mu^{1-p}}{1-p} + \frac{\mu^{2-p}}{2-p} \right)$$
For $y = 0$, the deviance simplifies to:
$$d(0, \mu) = 2 \frac{\mu^{2-p}}{2-p}$$

---

## Code Example

Below is a complete example comparing `TweedieLoss` and `CompoundPoissonLoss` on synthetic compound Poisson-Gamma data.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchregress.losses import TweedieLoss, CompoundPoissonLoss

# Generate synthetic compound Poisson-Gamma data
np.random.seed(42)
n_samples = 1000
p_power = 1.6
phi = 0.8

x = np.random.uniform(-1.5, 1.5, size=(n_samples, 1)).astype(np.float32)
mu = np.exp(0.8 * x + 0.5).astype(np.float32)

# Translate to Poisson-Gamma parameters
lambda_p = (mu ** (2.0 - p_power)) / (phi * (2.0 - p_power))
shape_k = (2.0 - p_power) / (p_power - 1.0)
scale_theta = phi * (p_power - 1.0) * (mu ** (p_power - 1.0))

y = np.zeros((n_samples, 1), dtype=np.float32)
for i in range(n_samples):
    n_events = np.random.poisson(lambda_p[i])
    if n_events > 0:
        y[i, 0] = np.random.gamma(shape_k, scale_theta[i], size=n_events).sum()

x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)

# Simple MLP predicting positive mean
class TweedieModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# Train with Tweedie Loss
model = TweedieModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = TweedieLoss(p=p_power, link="log")

for epoch in range(150):
    y_pred_log = model(x_tensor)
    loss = loss_fn(y_pred_log, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Tweedie model trained successfully.")
```
