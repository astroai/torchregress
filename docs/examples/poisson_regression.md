# Poisson and Zero-Inflated Regression Examples

This guide demonstrates how to model count data using Poisson and Zero-Inflated Poisson regression loss functions.

| # | Reference |
|:-:|:----------|
| 1 | Lambert, D. (1992). **Zero-Inflated Poisson Regression, With an Application to Defects on Silicon Wafers**. *Technometrics*. |
| 2 | Cameron, A. C., & Trivedi, P. K. (2013). **Regression Analysis of Count Data**. *Cambridge University Press*. |

---

## Mathematical Formulations

*   **Poisson Deviance Loss**:
    $$L_{\text{Deviance}}(y, \lambda) = 2 \left( y \log \left( \frac{y}{\lambda} \right) - (y - \lambda) \right)$$
    where $\lambda > 0$ is the predicted Poisson rate parameter.

*   **Zero-Inflated Poisson (ZIP)**:
    A mixed model where a sample is zero with probability $p$ (zero inflation), and Poisson-distributed with mean $\lambda$ with probability $1-p$:
    $$P(Y = y) = \begin{cases} p + (1-p)e^{-\lambda} & \text{for } y = 0 \\ (1-p) \frac{\lambda^y e^{-\lambda}}{y!} & \text{for } y > 0 \end{cases}$$
    The Negative Log-Likelihood is used as the loss function.

*   **Negative Binomial NLL**:
    Models overdispersed count data where the variance exceeds the mean:
    $$\text{Var}(Y) = \mu + \alpha \mu^2$$
    where $\alpha > 0$ is the dispersion parameter.

---

## Code Example

Below is a complete example comparing Poisson Deviance, Zero-Inflated Poisson, and Negative Binomial losses on synthetic count data.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchregress.losses import (
    PoissonDevianceLoss,
    ZeroInflatedPoissonNLLLoss,
    NegativeBinomialNLLLoss,
)

# Generate synthetic count data following Lambert's ZIP model
np.random.seed(42)
n_samples = 1000
x = np.random.uniform(-2, 2, size=(n_samples, 1)).astype(np.float32)
rate = np.exp(0.5 * x + 1.2).astype(np.float32)
logit_p = -0.8 * x + 0.3
p = 1.0 / (1.0 + np.exp(-logit_p))

poisson_counts = np.random.poisson(rate)
zero_inflation = (np.random.rand(n_samples, 1) < p).astype(np.int32)
y = np.where(zero_inflation, 0, poisson_counts).astype(np.float32)

x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)

# Define count prediction model
class CountModel(nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# Train with Poisson Deviance
model_poisson = CountModel(output_dim=1)
opt_poisson = optim.Adam(model_poisson.parameters(), lr=0.01)
loss_poisson = PoissonDevianceLoss()

for epoch in range(100):
    rate_pred = torch.exp(model_poisson(x_tensor))
    loss = loss_poisson(rate_pred, y_tensor)
    opt_poisson.zero_grad()
    loss.backward()
    opt_poisson.step()

# Train with ZIP
model_zip = CountModel(output_dim=2)
opt_zip = optim.Adam(model_zip.parameters(), lr=0.01)
loss_zip = ZeroInflatedPoissonNLLLoss()

for epoch in range(100):
    out = model_zip(x_tensor)
    rate_logits = out[:, 0:1]
    pi_logits = out[:, 1:2]
    loss = loss_zip(rate_logits, y_tensor, pi_logits=pi_logits)
    opt_zip.zero_grad()
    loss.backward()
    opt_zip.step()

print("Poisson and ZIP models trained successfully.")
```
