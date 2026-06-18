# Censored Regression Losses

Censored regression handles data where the true target value is only **partially observed**.  The model knows that the true value exceeds (or falls below) a threshold, but not the exact value.

---

## When Does Censoring Arise?

!!! example "Common scenarios"

    - **Survival analysis** — time-to-event is right-censored when a subject is still alive at study end
    - **Sensor saturation** — detector readings clamped at hardware limits
    - **Detection limits** — flux below sensitivity → only an upper bound is known
    - **Administrative** — study ends before all events are observed

---

## Censoring Convention

torchregress uses an integer encoding for the censoring indicator $c_i$:

| $c_i$ | Meaning | Interpretation |
|:------:|:--------|:---------------|
| $0$ | **Observed** | Target $y_i$ is exact |
| $+1$ | **Right-censored** | True value $y_i^* \geq y_i$ |
| $-1$ | **Left-censored** | True value $y_i^* \leq y_i$ |

---

## Mathematical Background

### Gaussian Censored NLL

For a Gaussian with predicted mean $\mu$ and variance $\sigma^2$, the censored log-likelihood uses the **normal CDF** $\Phi$ to model the probability of the true value lying beyond the observed threshold:

$$\ell(y_i, c_i \mid \mu, \sigma^2) = \begin{cases}
\displaystyle\log\,\phi\!\!\left(\frac{y_i - \mu}{\sigma}\right) & c_i = 0 \;\text{(observed)} \\[10pt]
\displaystyle\log\,\Phi\!\!\left(\frac{\mu - y_i}{\sigma}\right) & c_i = +1 \;\text{(right-censored)} \\[10pt]
\displaystyle\log\,\Phi\!\!\left(\frac{y_i - \mu}{\sigma}\right) & c_i = -1 \;\text{(left-censored)}
\end{cases}$$

where $\phi(\cdot)$ is the standard normal **PDF** and $\Phi(\cdot)$ is the standard normal **CDF**.

!!! info "Intuition"
    For observed points, the loss is the usual Gaussian NLL.  For censored points, the loss is the log-probability of the true value being **beyond** the observed threshold — a softer constraint that prevents the model from fitting to boundary artefacts.

### Accelerated Failure Time (AFT)

The AFT model assumes the log of survival time is linear:

$$\log T = \mu(x) + \sigma \cdot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, 1)$$

The loss is the negative log-likelihood under this log-normal model, with censoring handled through $\Phi$.

---

## Available Losses

### CensoredGaussianNLLLoss

Full Gaussian NLL with right/left censoring support. See [CensoredGaussianNLLLoss API](../api/losses.md#censoredgaussiannllloss).

```python
import torch
from torchregress.losses import CensoredGaussianNLLLoss

loss_fn = CensoredGaussianNLLLoss()

# Model predicts (mean, log_var d)
mean = torch.randn(64, 1)
log_var = torch.randn(64, 1)
target = torch.rand(64, 1) * 5.0

# Censoring: 0 = observed, +1 = right, -1 = left
censoring = torch.randint(-1, 2, (64, 1))

loss = loss_fn((mean, log_var), target, censoring=censoring)
```

### CensoredQuantileLoss

Quantile regression adapted for censored data — provides non-parametric intervals without assuming Gaussian noise. See [CensoredQuantileLoss API](../api/losses.md#censoredquantileloss).

```python
from torchregress.losses import CensoredQuantileLoss

loss_fn = CensoredQuantileLoss(quantile=0.5)
loss = loss_fn(y_pred, target, censoring=censoring)
```

### AFTLoss

Log-normal Accelerated Failure Time model for survival analysis. See [AFTLoss API](../api/losses.md#aftloss).

```python
from torchregress.losses import AFTLoss

loss_fn = AFTLoss()
# y_pred: (loc, log_scale) for log-normal AFT (log_scale = log σ, not log variance)
# censoring: 0 = observed, +1 = right-censored, -1 = left-censored
# (not sklearn's event_indicator where 1 = event)
loss = loss_fn((loc, log_scale), survival_time, censoring=censoring)
```

---

## Complete Example: Survival Analysis

```python
import torch
import torch.nn as nn
from torchregress.losses import CensoredGaussianNLLLoss

# Simulate survival data with right censoring
torch.manual_seed(42)
n = 500
x = torch.randn(n, 5)
true_time = torch.exp(x @ torch.randn(5, 1) + 0.5 * torch.randn(n, 1))

# 30% of subjects are right-censored (study ended before event)
censor_time = torch.rand(n, 1) * true_time.median() * 3
observed_time = torch.min(true_time, censor_time)
censoring = (censor_time < true_time).long()  # 1 = right-censored, 0 = observed

# Model predicts (mean, log_variance)
class SurvivalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.mean_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.mean_head(h), self.logvar_head(h)

model = SurvivalModel()
loss_fn = CensoredGaussianNLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
    mean, logvar = model(x)
    loss = loss_fn((mean, logvar), observed_time, censoring=censoring)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

---

## Practical Tips

!!! tip "Creating the censoring indicator"
    Use `0` for fully observed, `+1` for right-censored, `-1` for left-censored. For survival analysis, `censoring = (event_not_observed).long()`.

!!! tip "Combining with conformal prediction"
    Wrap censored quantile predictions with [CQR](../methods/conformal/predictors.md) for **coverage-guaranteed** survival intervals.

---

## Related

- [Censored Metrics](../metrics/censored.md) — concordance index, Brier score, etc.
- [Censored Regression Comparison](../examples/censored_regression_comparison.md) — end-to-end benchmark
- [Censored Regression Comparison (Real Data)](../examples/censored_regression_realdata_comparison.md) — real-data benchmark

---

## References

| # | Reference |
|:-:|:----------|
| 1 | J. Tobin. ["Estimation of Relationships for Limited Dependent Variables."](https://www.jstor.org/stable/1907382) *Econometrica*, 26(1):24–36, **1958**. |
| 2 | D.R. Cox. ["Regression Models and Life-Tables."](https://www.jstor.org/stable/2985181) *JRSS B*, 34(2):187–220, **1972**. |
| 3 | J.L. Buckley, I.R. James. ["Linear Regression with Censored Data."](https://www.jstor.org/stable/2335161) *Biometrika*, 66(3):429–436, **1979**. |
