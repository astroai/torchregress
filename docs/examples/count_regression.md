# Count and Zero-Inflated Regression Examples

This guide demonstrates two families of losses for non-negative targets with many zeros:
Poisson-family losses for **counts** (Poisson deviance, likelihood ratio, zero-inflated
Poisson, negative binomial) and Tweedie / compound Poisson-Gamma losses for
**zero-inflated continuous** responses (rainfall, insurance claim sizes).

→ API: [`PoissonDevianceLoss`](../api/losses.md), [`ZeroInflatedPoissonNLLLoss`](../api/losses.md), [`NegativeBinomialNLLLoss`](../api/losses.md), [`TweedieLoss`](../api/losses.md), [`CompoundPoissonLoss`](../api/losses.md). Guide: [Poisson and Tweedie losses](../losses/poisson_tweedie.md).

**Run the script** [`examples/count_regression_demo.py`](https://github.com/astroai/torchregress/blob/main/examples/count_regression_demo.py) (`--family all` by default runs both):

```bash
pixi run python examples/count_regression_demo.py --family poisson  # ZIP count data, four count losses
pixi run python examples/count_regression_demo.py --family tweedie  # compound Poisson-Gamma data, two Tweedie losses
```

| Family | Data | Losses | Reported metric |
|:--|:--|:--|:--|
| `poisson` | 1,200 draws from Lambert's ZIP model, $\log\lambda = 0.5x + 1.2$ | `PoissonDevianceLoss`, `PoissonLikelihoodRatioLoss`, `ZeroInflatedPoissonNLLLoss`, `NegativeBinomialNLLLoss` | MAE of predicted rate vs true $\lambda$ |
| `tweedie` | 1,000 compound Poisson-Gamma draws, $p = 1.6$, $\phi = 0.8$ | `TweedieLoss`, `CompoundPoissonLoss` | MAE of predicted mean vs true $\mu$ |

## When To Use Which

| Target | Excess zeros? | Variance vs mean | Use |
|:--|:--|:--|:--|
| Integer counts | No | $\text{Var} = \mu$ | `PoissonDevianceLoss` |
| Integer counts | Yes (structural zeros) | $\text{Var} \approx \mu$ on the non-zero part | `ZeroInflatedPoissonNLLLoss` |
| Integer counts | Maybe | $\text{Var} > \mu$ (overdispersed) | `NegativeBinomialNLLLoss` |
| Non-negative continuous | Yes (exact zeros) | $\text{Var} = \phi\mu^p$, $1 < p < 2$ | `TweedieLoss` / `CompoundPoissonLoss` |

---

## Count Data (`--family poisson`)

| # | Reference |
|:-:|:----------|
| 1 | Lambert, D. (1992). [**Zero-Inflated Poisson Regression, With an Application to Defects on Silicon Wafers**](https://www.jstor.org/stable/1269547). *Technometrics*. |
| 2 | Cameron, A. C., & Trivedi, P. K. (2013). [**Regression Analysis of Count Data**](https://doi.org/10.1017/CBO9781139013567). *Cambridge University Press*. |

### Mathematical Formulations

The count losses compared in this example are:

- **Poisson Deviance Loss**:
    $$L_{\text{Deviance}}(y, \lambda) = 2 \left( y \log \left( \frac{y}{\lambda} \right) - (y - \lambda) \right)$$
    where $\lambda > 0$ is the predicted Poisson rate parameter.

- **Zero-Inflated Poisson (ZIP)**:
    A mixed model where a sample is zero with probability $p$ (zero inflation), and Poisson-distributed with mean $\lambda$ with probability $1-p$:
    $$P(Y = y) = \begin{cases} p + (1-p)e^{-\lambda} & \text{for } y = 0 \\ (1-p) \frac{\lambda^y e^{-\lambda}}{y!} & \text{for } y > 0 \end{cases}$$
    The Negative Log-Likelihood is used as the loss function.

- **Negative Binomial NLL**:
    Models overdispersed count data where the variance exceeds the mean:
    $$\text{Var}(Y) = \mu + \alpha \mu^2$$
    where $\alpha > 0$ is the dispersion parameter.

### Code Example

Below is a condensed, self-contained example comparing Poisson Deviance, Zero-Inflated Poisson, and Negative Binomial losses on synthetic count data.

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

---

## Zero-Inflated Continuous Data (`--family tweedie`)

| # | Reference |
|:-:|:----------|
| 1 | Jorgensen, B. (1987). [**Exponential Dispersion Models**](https://www.jstor.org/stable/2345173). *Journal of the Royal Statistical Society: Series B (Methodological)*. |
| 2 | Ohlsson, E., & Johansson, B. (2010). [**Non-Life Insurance Pricing with Generalized Linear Models**](https://doi.org/10.1007/978-3-642-10791-7). *Springer*. |

### Mathematical Formulations

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

### Code Example

Below is a condensed, self-contained example comparing `TweedieLoss` and `CompoundPoissonLoss` on synthetic compound Poisson-Gamma data.

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
