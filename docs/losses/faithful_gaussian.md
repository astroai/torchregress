# Faithful Heteroscedastic & CRPS Losses

This page covers two specialized Gaussian loss variants:
- **`FaithfulGaussianLoss`** — decouples mean and variance training to prevent the variance head from distorting point predictions
- **`GaussianCRPSLoss`** — a proper scoring rule alternative to NLL that is less sensitive to tail miscalibration

---

## FaithfulGaussianLoss

### The Problem with Joint NLL

Standard [`GaussianNLLLoss`](../api/losses.md) couples gradients from the variance head into the mean through the residual term:

$$\mathcal{L}_{\text{NLL}}(y, \mu, \sigma^2) = \frac{1}{2} \log(2\pi\sigma^2) + \frac{(y - \mu)^2}{2\sigma^2}$$

The gradient with respect to $\mu$ is $\frac{\mu - y}{\sigma^2}$, which means the variance head's outputs directly scale the gradients flowing into the mean head. If the model learns an inflated $\sigma^2$ in high-error regions, the mean head receives **weaker** gradients there — effectively "explaining away" prediction errors as aleatoric noise.

### The Faithful Solution

`FaithfulGaussianLoss` decouples the two objectives (see [API: FaithfulGaussianLoss](../api/losses.md)):

$$\boxed{\;\mathcal{L}_{\text{Faithful}} = \lambda_{\mu} (\mu - y)^2 + \lambda_{v} \cdot \frac{1}{2} \left( \log(2\pi\sigma^2) + \frac{(y - \operatorname{sg}(\mu))^2}{\sigma^2} \right)\;}$$

where $\operatorname{sg}$ is **stop-gradient**. The mean receives gradients only from the MSE term; the variance head receives gradients from the NLL term with a **detached mean**. This mirrors the intent of *faithful* heteroscedastic training: preserve point prediction quality while learning a noise model.

### Usage

```python
from torchregress.losses import FaithfulGaussianLoss

loss_fn = FaithfulGaussianLoss(mean_weight=1.0, variance_weight=1.0)

# Model outputs (mean, log_variance) tuple
mean, logvar = model(x)
loss = loss_fn((mean, logvar), y_true)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `mean_weight` | `float` | `1.0` | Multiplier on the MSE term. Set to `0` to train variance only. |
| `variance_weight` | `float` | `1.0` | Multiplier on the Gaussian NLL terms. Set to `0` to train mean only. |
| `min_variance` | `float` | `1e-6` | Minimum variance for numerical stability |
| `eps` | `float` | `1e-8` | Epsilon for log-stability |
| `reduction` | `str` | `"mean"` | `"mean"`, `"sum"`, or `"none"` |
| `split_dim` | `int` | `-1` | Dimension to split concatenated `[mean, log_var]` output |

### Comparison: Faithful vs Beta-NLL vs Joint NLL

| Feature | [`GaussianNLLLoss`](../api/losses.md) | [`BetaNLLLoss`](../api/losses.md) | [`FaithfulGaussianLoss`](../api/losses.md) |
|:--------|:-----------------:|:-------------:|:----------------------:|
| Mean/variance coupling | Coupled | Variance-detached reweighting | Explicitly split |
| Mean objective | NLL residual | NLL residual (detached variance) | Pure MSE |
| Variance objective | Joint NLL | Reweighted NLL | Separate NLL term |
| Gradient path | $\mu$ sees $\sigma^2$ | $\mu$ sees detached $\sigma^{2\beta}$ | $\mu$ sees only MSE |
| Best for | Standard heteroscedastic | Stabilized variance learning | Preserving point accuracy |

!!! tip "When to use FaithfulGaussianLoss"
    Use when you care more about **point prediction accuracy** than perfectly calibrated variance, but still want a meaningful noise model. Common in scientific applications where the mean prediction is the primary deliverable.

!!! warning "Detach tradeoff"
    Because the mean head only sees MSE gradients, it does **not** receive curvature information from the NLL about heteroscedasticity. On datasets where mean and variance are strongly coupled (e.g., Poisson-like count data), this decoupling can slow convergence of the mean head.

!!! warning "Edge cases"
    - **`mean_weight=0`**: The mean head receives zero gradients, but the (detached) mean still feeds the variance head's residual. If the mean is poorly initialized and never trained, the variance head receives a bad signal.
    - **`variance_weight=0`**: The variance head receives zero gradients. The loss reduces to plain MSE.

---

## GaussianCRPSLoss

### Why CRPS Instead of NLL?

The Continuous Ranked Probability Score (CRPS) is a **proper scoring rule** that evaluates the full predictive distribution. See the [GaussianCRPSLoss API](../api/losses.md) for parameters and the closed-form definition.

Unlike NLL, CRPS:

- Has the **same units** as the target variable (interpretable)
- Is **less sensitive** to tail events than NLL (a single very-low-probability point doesn't dominate)
- Penalizes both **calibration** and **sharpness** simultaneously
- Reduces to **MAE** when the predictive distribution is a point mass

### Analytic Gaussian CRPS

For a Gaussian predictive distribution $\mathcal{N}(\mu, \sigma^2)$, the CRPS has a closed form:

$$\boxed{\;\text{CRPS}(F, y) = \sigma \left[ z \cdot (2\Phi(z) - 1) + 2\phi(z) - \frac{1}{\sqrt{\pi}} \right]\;}$$

where $z = \frac{y - \mu}{\sigma}$, $\Phi$ is the standard normal CDF, and $\phi$ is the standard normal PDF.

### Usage

```python
from torchregress.losses import GaussianCRPSLoss

loss_fn = GaussianCRPSLoss()

# Same interface as GaussianNLLLoss
mean, logvar = model(x)
loss = loss_fn((mean, logvar), y_true)
```

Supports the same prediction formats as `GaussianNLLLoss`:
- Tuple `(mean, log_variance)`
- Concatenated tensor `[mean, log_variance]`
- Mean-only when `fixed_variance` is set

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `min_variance` | `float` | `1e-6` | Minimum variance for numerical stability |
| `eps` | `float` | `1e-8` | Epsilon for division stability |
| `fixed_variance` | `float` or `None` | `None` | Fixed variance for homoscedastic models |
| `reduction` | `str` | `"mean"` | `"mean"`, `"sum"`, or `"none"` |

### When to Prefer CRPS over NLL

!!! success "Use CRPS when"
    - You want **interpretable** loss values in target units
    - Your data has **heavy tails** where NLL is dominated by a few extreme points
    - You're comparing models across different distributional families (CRPS is a universal scoring rule)
    - You're evaluating ensemble or Bayesian model predictions via the **Energy Score** (multivariate CRPS)

!!! warning "Performance note"
    CRPS is approximately **2-3× slower** to compute per iteration than NLL, due to the additional `ndtr` (normal CDF) and `exp` operations per element. For large-scale training where NLL is adequate, the speed difference may matter.

!!! info "Use NLL when"
    - You need the tightest possible likelihood-based training signal
    - Your model is well-specified and residuals are approximately Gaussian
    - You're computing information criteria (AIC, BIC) or Bayes factors

---

## Complete Example

```python
import torch
import torch.nn as nn
from torchregress.losses import FaithfulGaussianLoss, GaussianCRPSLoss

# Heteroscedastic model
class HeteroModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.mean_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.mean_head(h), self.logvar_head(h)

torch.manual_seed(42)
x = torch.randn(200, 5)
y = 2.0 * x[:, [0]] + 0.3 * torch.randn(200, 1) * torch.exp(0.5 * x[:, [1]])

model = HeteroModel(5)
faithful_loss = FaithfulGaussianLoss(mean_weight=1.0, variance_weight=1.0)
crps_loss = GaussianCRPSLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    mean, logvar = model(x)
    # Train with faithful loss
    loss = faithful_loss((mean, logvar), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Monitor CRPS
    with torch.no_grad():
        crps = crps_loss((mean, logvar), y)
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Faithful Loss = {loss.item():.4f}, CRPS = {crps.item():.4f}")
```

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Stirn et al. ["Faithful Heteroscedastic Regression with Deep Neural Networks."](https://arxiv.org/abs/2212.00873) *AISTATS*, **2023**. |
| 2 | Hersbach, H. ["Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems."](https://doi.org/10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2) *Weather and Forecasting*, 15(5):559–570, **2000**. |
| 3 | Gneiting & Raftery. ["Strictly Proper Scoring Rules, Prediction, and Estimation."](https://www.tandfonline.com/doi/abs/10.1198/016214506000001437) *JASA*, 102(477):359–378, **2007**. |
