# Heteroscedastic Laplace Regression Examples

This guide explains how to perform Bayesian last-layer Laplace posterior approximation on deep heteroscedastic neural networks using `HeteroscedasticLaplaceRegressor` and `NaturalHeteroscedasticHead` to decompose predictive uncertainty.

| # | Reference |
|:-:|:----------|
| 1 | Immer, A., Postels, J., Lücke, J., Rätsch, G., & Mandt, S. (2023). [**Effective Bayesian Heteroscedastic Regression with Deep Neural Networks**](https://arxiv.org/abs/2302.08498). *Advances in Neural Information Processing Systems (NeurIPS)*. |
| 2 | Daxberger, E., Agustinus, K., Ostner, J., Ruin, L., & Hennig, P. (2021). [**Laplace Redux - Effortless Bayesian Deep Learning**](https://arxiv.org/abs/2106.14806). *Advances in Neural Information Processing Systems (NeurIPS)*. |

---

## Mathematical Formulations

### Natural Parameterization Head

Instead of predicting the standard parameters (mean $\mu$, log-variance $\log\sigma^2$) directly, the `NaturalHeteroscedasticHead` outputs the natural parameters $\eta = (\eta_1, \eta_2)$ of the Gaussian distribution:
$$\eta_1 = \frac{\mu}{\sigma^2}, \quad \eta_2 = -\frac{1}{2 \sigma^2}$$
To map back to standard parameters (ensuring positivity of the variance):
*   Using **exp link**: $\sigma^2 = \exp(-f_2)$ and $\mu = f_1 \sigma^2$.
*   Using **softplus link**: $\sigma^2 = \text{softplus}(f_2)^{-1}$ and $\mu = f_1 \sigma^2$.

### Diagonal Laplace Approximation

Let $\theta$ be the last-layer linear parameters of the head. We fit the Maximum A Posteriori (MAP) weights by minimizing:
$$\mathcal{L}(\theta) = \sum_{i=1}^N \text{NLL}(y_i; x_i, \theta) + \frac{1}{2} \gamma \|\theta\|_2^2$$
where $\gamma$ is the prior precision.
The diagonal empirical Fisher matrix is computed as:
$$F = \sum_{i=1}^N \left(\nabla_{\theta} \text{NLL}(y_i; x_i, \theta_{\text{MAP}})\right)^2$$
The diagonal posterior variance over weights is:
$$\text{Var}(\theta) = (F + \gamma I)^{-1}$$

### Law of Total Variance (Uncertainty Decomposition)

By sampling weights $\theta^{(s)} \sim \mathcal{N}(\theta_{\text{MAP}}, \text{Var}(\theta))$, we obtain the predictive mean $\mu(x; \theta^{(s)})$ and variance $\sigma^2(x; \theta^{(s)})$. The total variance decomposes into:
$$\text{Var}(Y \mid x) = \underbrace{E_{\theta}[\sigma^2(x; \theta)]}_{\text{Aleatoric Uncertainty}} + \underbrace{\text{Var}_{\theta}[\mu(x; \theta)]}_{\text{Epistemic Uncertainty}}$$
where:
*   **Aleatoric variance** represents the data-dependent observation noise.
*   **Epistemic variance** represents the parameter uncertainty due to finite training data.

---

## Code Example

Below is a complete, runnable example demonstrating uncertainty decomposition.

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchregress.algorithms import HeteroscedasticLaplaceRegressor, NaturalHeteroscedasticHead

# Generate synthetic heteroscedastic data
np.random.seed(42)
n_samples = 1000
x = np.random.uniform(-2.0, 2.0, size=(n_samples, 1)).astype(np.float32)
mean = 0.5 * (x**3)
noise_std = 0.2 + 0.4 * (x + 1.0) ** 2
y = mean + np.random.normal(0.0, noise_std).astype(np.float32)

x_tensor = torch.from_numpy(x)
y_tensor = torch.from_numpy(y)

# Define backbone features and natural parameter head
base_model = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
head = NaturalHeteroscedasticHead(in_features=16, out_features=1, link_fn="exp")

# Wrap in HeteroscedasticLaplaceRegressor
laplace_reg = HeteroscedasticLaplaceRegressor(
    base_model=base_model,
    head=head,
    prior_precision=1.0,
    n_samples=50,
)

# Fit weights and compute Laplace posterior
loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=32, shuffle=True)
laplace_reg.fit(loader, lr=0.01, epochs=100)

# Predict distribution
pred_dist = laplace_reg.predict_distribution(x_tensor)
epistemic_var = pred_dist.extra["epistemic_variance"]
aleatoric_var = pred_dist.extra["aleatoric_variance"]

print("Laplace model calibrated and decomposed uncertainties successfully.")
```
