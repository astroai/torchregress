# Expectile and Quantile Regression Examples

This guide demonstrates expectile regression and compares it with quantile regression on heteroscedastic datasets using `ExpectileCrossoverLoss` and `QuantileCrossoverLoss`.

| # | Reference |
|:-:|:----------|
| 1 | Newey, W. K., & Powell, J. L. (1987). [**Asymmetric Least Squares Estimation and Testing**](https://www.jstor.org/stable/1913610). *Econometrica*. |
| 2 | Koenker, R., & Bassett Jr, G. (1978). [**Regression Quantiles**](https://www.jstor.org/stable/1913643). *Econometrica*. |

---

## Mathematical Formulations

### Quantile Regression (Pinball Loss)

Quantile regression estimates the conditional quantiles of a response variable. The pinball loss (quantile loss) for a level $q \in (0, 1)$ is defined as:
$$L_{q}(y, \hat{y}) = \begin{cases} q(y - \hat{y}) & \text{if } y > \hat{y} \\ (1-q)(\hat{y} - y) & \text{if } y \le \hat{y} \end{cases}$$

### Expectile Regression (Asymmetric Least Squares)

Expectile regression generalizes mean regression using asymmetric least squares. For an expectile level $\tau \in (0, 1)$, the loss is:
$$L_{\tau}(y, \hat{y}) = \begin{cases} 2 \tau (y - \hat{y})^2 & \text{if } y > \hat{y} \\ 2 (1-\tau)(y - \hat{y})^2 & \text{if } y \le \hat{y} \end{cases}$$

### Crossover Penalties

When predicting multiple curves simultaneously, crossing curves violate the property that lower levels must be strictly smaller than higher levels. The crossover loss penalizes violations:
$$L_{\text{crossover}}(Y, \{\hat{y}_i\}) = L_{\text{base}}(Y, \{\hat{y}_i\}) + \gamma \sum_{i} \max(\hat{y}_i - \hat{y}_{i+1}, 0)$$
where $\gamma > 0$ is the penalty weight.

---

## Code Example

Below is a complete, runnable example comparing expectile and quantile models on heteroscedastic data.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchregress.losses import ExpectileCrossoverLoss, QuantileCrossoverLoss

# Generate synthetic heteroscedastic data
np.random.seed(42)
n_samples = 1000
x = np.random.uniform(-2.0, 2.0, size=(n_samples, 1)).astype(np.float32)
mean = 0.5 * (x**3)
noise_std = 0.5 + 0.5 * (x**2)
y = mean + np.random.normal(0.0, noise_std).astype(np.float32)

x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)

# Model outputting multiple levels
class MultiLevelModel(nn.Module):
    def __init__(self, num_levels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, num_levels)
        )
    def forward(self, x):
        return self.net(x).unsqueeze(2)

# Train Expectile Model (levels 0.1, 0.5, 0.9)
model_exp = MultiLevelModel(num_levels=3)
opt_exp = optim.Adam(model_exp.parameters(), lr=0.01)
loss_exp = ExpectileCrossoverLoss(expectiles=[0.1, 0.5, 0.9], crossover_penalty=10.0)

for epoch in range(150):
    preds = model_exp(x_tensor)
    loss = loss_exp(preds, y_tensor)
    opt_exp.zero_grad()
    loss.backward()
    opt_exp.step()

print("Expectile crossover regression model trained successfully.")
```
