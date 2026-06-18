# Mondrian Conformal Prediction Examples

This guide explains how to perform Mondrian (group-conditional) conformal prediction to guarantee valid coverage independently within subgroups of data using `SplitConformal` with group indicators.

→ API: [`SplitConformal`](../api/losses.md#splitconformal). Guide: [Conformal predictors](../methods/conformal/predictors.md).

| # | Reference |
|:-:|:----------|
| 1 | Vovk, V. (2002). [**Conditional Conformal Prediction**](https://link.springer.com/chapter/10.1007/3-540-36169-3_29). *International Conference on Algorithmic Learning Theory*. |
| 2 | Vovk, V., Gammerman, A., & Shafer, G. (2005). [**Algorithmic Learning in a Random World**](https://doi.org/10.1007/b106715). *Springer*. |

---

## Mathematical Formulations

Standard conformal prediction guarantees marginal coverage:
$$P(Y \in \hat{C}(X)) \ge 1 - \alpha$$
However, this overall average does not guarantee coverage conditional on group features $G$ (e.g., demographics, noise regimes). Mondrian conformal prediction guarantees:
$$P(Y \in \hat{C}(X) \mid G = g) \ge 1 - \alpha \quad \forall g$$

### Mondrian Calibration

Let the calibration set $C_{\text{cal}}$ be partitioned into group-wise calibration sets:
$$C_g = \{i \in C_{\text{cal}} \mid G_i = g\}$$

For each group $g$:
1.  Compute nonconformity scores for $i \in C_g$:
    $$S_i = |y_i - \hat{y}_i|$$
2.  Compute the conformal threshold as the $(1-\alpha)\left(1 + \frac{1}{|C_g|}\right)$ quantile of $\{S_i\}_{i \in C_g}$:
    $$\hat{q}_g = \text{Quantile}\left(\{S_i\}_{i \in C_g}, (1-\alpha)\left(1 + \frac{1}{|C_g|}\right)\right)$$

### Mondrian Prediction

For a new test point with prediction $\hat{y}$ and group label $G_{\text{test}} = g$, the prediction interval is:
$$[\hat{y} - \hat{q}_g, \hat{y} + \hat{q}_g]$$

---

## Code Example

Below is a complete, runnable example demonstrating group-conditional calibration.

```python
import numpy as np
import torch
import torch.nn as nn
from torchregress.losses.conformal import SplitConformal

# Generate synthetic grouped data with heteroscedastic noise
np.random.seed(42)
n_samples = 1000

x = np.random.uniform(-1.5, 1.5, size=(n_samples, 1)).astype(np.float32)
groups = np.random.choice([0, 1], size=(n_samples, 1)).astype(np.int64)
y_mean = 0.8 * x

# Group 0: low noise (0.2), Group 1: high noise (1.2)
noise_std = np.where(groups == 0, 0.2, 1.2).astype(np.float32)
y = y_mean + np.random.normal(0, noise_std).astype(np.float32)

x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)
groups_tensor = torch.from_numpy(groups).squeeze(1)

# Fit simple regression model
class MeanRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MeanRegressionModel()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    preds = model(x_tensor)
    loss = torch.nn.functional.mse_loss(preds, y_tensor)
    opt.zero_grad()
    loss.backward()
    opt.step()

# Calibrate Mondrian Conformal Predictor
model.eval()
with torch.no_grad():
    preds = model(x_tensor)

mondrian_cp = SplitConformal(alpha=0.1)
mondrian_cp.calibrate(preds, y_tensor, groups=groups_tensor)

print("Mondrian Conformal Predictor calibrated successfully.")
```
