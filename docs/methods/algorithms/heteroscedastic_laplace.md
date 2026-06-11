# Effective Bayesian Heteroscedastic Laplace

This algorithm performs a last-layer Laplace posterior approximation over a natural-parameterized heteroscedastic regression head. It allows for fast, post-hoc uncertainty decomposition into epistemic (model) and aleatoric (data) components.

---

## Mathematical Background

Standard deep heteroscedastic regression predicts mean $\mu$ and log-variance $\log\sigma^2$ directly. However, the resulting negative log-likelihood (NLL) loss is non-convex w.r.t these parameters, making the Gaussian approximation of the parameter posterior (Laplace approximation) inaccurate.

### 1. Natural Parameterization

Following Immer et al., we parameterize the Gaussian distribution using its exponential family natural parameters:

$$\eta_1 = \frac{\mu}{\sigma^2}, \quad \eta_2 = -\frac{1}{2 \sigma^2}$$

To ensure $\sigma^2 > 0$, we predict $f_1, f_2 \in \mathbb{R}$ and define the natural parameters using a link function. For the exponential link function (`link_fn="exp"`):

$$\eta_2 = -\frac{1}{2} \exp(f_2), \quad \eta_1 = f_1 \exp(f_2)$$

This maps back to standard parameters as:

$$\sigma^2 = \exp(-f_2), \quad \mu = f_1 \exp(-f_2)$$

This parameterization makes the NLL loss convex w.r.t the network outputs $f$, ensuring optimization stability and making the posterior well-suited for a Laplace approximation.

### 2. Last-Layer Laplace Approximation

We place a Gaussian prior over the natural head weights $\theta \sim \mathcal{N}(0, \sigma_0^2 I)$. After finding the MAP estimate $\theta_{MAP}$ on the training set, we approximate the posterior distribution of $\theta$:

$$q(\theta) = \mathcal{N}(\theta_{MAP}, \Lambda^{-1})$$

where the diagonal precision matrix $\Lambda$ is computed using the empirical Fisher information matrix:

$$\Lambda = \text{diag}\left(\frac{1}{\sigma_0^2} I + \sum_{n=1}^N (\nabla_\theta \mathcal{L}_n(\theta_{MAP}))^2\right)$$

where $\mathcal{L}_n$ is the NLL loss for sample $n$. This gradient w.r.t head weights is computed efficiently using `torch.func.grad` and vectorized via `torch.vmap`.

### 3. Uncertainty Decomposition

At test time, we draw $S$ weight samples $\theta^{(s)} \sim q(\theta)$ to compute $S$ forward passes. For each test point $x$, this yields $S$ mean predictions $\mu^{(s)}(x)$ and variance predictions $\sigma^{2(s)}(x)$. We decompose the predictive uncertainty into:

- **Epistemic Uncertainty** (model parameter uncertainty):
  $$\sigma^2_{\text{epistemic}}(x) = \text{Var}\left(\{\mu^{(s)}(x)\}_{s=1}^S\right)$$
- **Aleatoric Uncertainty** (inherent data noise):
  $$\sigma^2_{\text{aleatoric}}(x) = \frac{1}{S} \sum_{s=1}^S \sigma^{2(s)}(x)$$
- **Total Uncertainty**: $\sigma^2_{\text{total}}(x) = \sigma^2_{\text{epistemic}}(x) + \sigma^2_{\text{aleatoric}}(x)$.

---

## High-Level API: `HeteroscedasticLaplaceRegressor`

```python
from torchregress.algorithms import (
    NaturalHeteroscedasticHead,
    HeteroscedasticLaplaceRegressor,
)

# 1. Instantiate the natural parameter head
head = NaturalHeteroscedasticHead(in_features=64, out_features=1, link_fn="exp")

# 2. Wrap the backbone and head
model = HeteroscedasticLaplaceRegressor(
    base_model=my_backbone,
    head=head,
    prior_precision=1.0,  # Prior precision 1 / sigma_0^2
    n_samples=30,         # MC samples for inference
)

# 3. Fit the model and compute the Laplace posterior
model.fit(train_loader, lr=1e-3, epochs=10)

# 4. Predict a PredictiveBatch containing mean, std, samples, and epistemic/aleatoric metrics
pred = model.predict_distribution(x_test)
epistemic = pred.extra["epistemic_variance"]
aleatoric = pred.extra["aleatoric_variance"]
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `base_model` | `nn.Module` | — | Backbone feature extractor outputting `(B, in_features)` |
| `head` | `nn.Module` | — | `NaturalHeteroscedasticHead` or a custom head outputting natural parameters |
| `prior_precision` | `float` | `1.0` | Prior precision $\lambda_0 = 1/\sigma_0^2$ for L2 regularization |
| `n_samples` | `int` | `30` | Number of MC samples drawn from posterior for test-time prediction |
| `jitter` | `float` | `1e-6` | Stabilizer added to diagonal variance before division / inversion |

---

## Complete Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchregress.algorithms import (
    NaturalHeteroscedasticHead,
    HeteroscedasticLaplaceRegressor,
)

# Generate synthetic data
torch.manual_seed(42)
x = torch.randn(200, 5)
# Target has input-dependent noise
y = 2.0 * x[:, [0]] + torch.randn(200, 1) * torch.exp(0.5 * x[:, [1]])

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define backbone and head
backbone = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 8))
head = NaturalHeteroscedasticHead(in_features=8, out_features=1)

# Fit regressor
regressor = HeteroscedasticLaplaceRegressor(
    base_model=backbone,
    head=head,
    prior_precision=0.1,
)
regressor.fit(loader, lr=0.01, epochs=5)

# Predict distribution
x_test = torch.randn(10, 5)
pred_batch = regressor.predict_distribution(x_test)

print("Mean Predictions:\n", pred_batch.mean.squeeze(-1))
print("Epistemic Variance:\n", pred_batch.extra["epistemic_variance"].squeeze(-1))
print("Aleatoric Variance:\n", pred_batch.extra["aleatoric_variance"].squeeze(-1))
```

---

## When to use this method

| Scenario / Goal | Recommended Choice | Rationale |
|:---|:---:|:---|
| **Separating model error (epistemic) from data noise (aleatoric)** | **Yes (Recommended)** | This is one of the fastest ways to decompose uncertainty without training expensive ensembles. |
| **Out-of-Distribution (OOD) robustness** | **Yes** | Epistemic uncertainty increases on test inputs far from the training data distribution. |
| **Ultra-fast online updates** | **Yes** | The posterior covariance can be updated post-hoc by accumulating Fisher updates without retraining the backbone. |
| **Large-scale model training** | **Warning** | Last-layer Laplace is efficient, but full-parameter Laplace is too heavy. This regressor restricts posterior approximation to the final head layer only. |

!!! tip "Natural Head Stabilization"
    Using the `NaturalHeteroscedasticHead` is crucial for stable optimization. Standard heads outputting mean and log-variance directly can result in highly non-Gaussian parameter posteriors where Laplace updates are unstable.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Immer et al., "Effective Bayesian Heteroscedastic Regression with Deep Neural Networks" (NeurIPS 2023). |
