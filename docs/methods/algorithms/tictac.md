# TIC-TAC (Taylor-Induced Covariance)

TIC-TAC binds the predicted covariance matrix to the local derivatives (gradient and curvature) of the mean prediction model with respect to input features.

---

## Mathematical Background

Standard deep heteroscedastic regression predicts the mean $\hat{y} = \mu(x)$ and covariance $\Sigma(x)$ using separate output heads. This can lead to optimization instability and poor covariance alignment.

TIC-TAC links the covariance $\Sigma(x)$ directly to the local geometry of the mean model using a Taylor expansion representation:

$$\Sigma(x) = k_1(x) J(x) J(x)^T + k_2(x) H(x) + \text{diag}(k_3(x)) + \epsilon I$$

where:

- $J(x) \in \mathbb{R}^{D \times P}$ is the Jacobian of the mean prediction $f(x)$ w.r.t input features $x \in \mathbb{R}^P$:
  $$J(x)_{i, j} = \frac{\partial f_i(x)}{\partial x_j}$$
- $H(x) \in \mathbb{R}^{D \times D}$ is the curvature matrix, computed as the inner product of the target-dimension Hessians:
  $$H(x)_{i, j} = \text{Tr}\left(\frac{\partial^2 f_i(x)}{\partial x^2} \frac{\partial^2 f_j(x)}{\partial x^2}\right)$$
- $k_1(x), k_2(x)$ are positive scaling coefficients (either global scalar parameters or input-dependent MLP heads).
- $k_3(x) \in \mathbb{R}^D$ is a residual diagonal variance parameter (allowing individual target dimensions to have unique noise floors).
- $\epsilon I$ is a stabilizer jitter (default: $10^{-6}$) to ensure positive-definiteness.

Vectorized batch computation of $J(x)$ and $H(x)$ is performed efficiently using PyTorch's functional autograd API (`torch.func.jacrev` and `torch.func.hessian` vectorized via `torch.vmap`).

---

## High-Level API: `TaylorInducedCovarianceHead`

`TaylorInducedCovarianceHead` is a PyTorch `nn.Module` wrapping a base mean backbone.

```python
from torchregress.algorithms import TaylorInducedCovarianceHead

# Wrap a mean backbone
head = TaylorInducedCovarianceHead(
    base_model=my_backbone,
    target_dim=3,
    input_dim=5,  # Enable input-dependent coefficients k1, k2, k3
    jitter=1e-6,
)

# Returns mean (B, D) and covariance (B, D, D)
mean, cov = head(x)
```

### Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `base_model` | `nn.Module` | — | PyTorch module predicting the mean vector of shape `(B, target_dim)` |
| `target_dim` | `int` | — | Dimension of the target regression space $D$ |
| `input_dim` | `int \| None` | `None` | If provided, instantiates input-dependent networks for $k_1, k_2, k_3$. Otherwise, uses global learnable scalars. |
| `k1_init` | `float` | `1.0` | Initial value for $k_1$ |
| `k2_init` | `float` | `1.0` | Initial value for $k_2$ |
| `k3_init` | `float` | `1.0` | Initial value for $k_3$ |
| `jitter` | `float` | `1e-6` | Stabilizer value $\epsilon$ added to the diagonal |

---

## Complete Example

```python
import torch
import torch.nn as nn
from torchregress.algorithms import TaylorInducedCovarianceHead
from torchregress.metrics import TaskAgnosticCorrelations

# Setup synthetic features and target
torch.manual_seed(42)
x = torch.randn(10, 5)
y = torch.randn(10, 3)

# Simple mean network
backbone = nn.Sequential(
    nn.Linear(5, 10),
    nn.Tanh(),  # Tanh or gelu/silu is recommended to ensure non-zero second derivatives
    nn.Linear(10, 3)
)

# TIC-TAC head
model = TaylorInducedCovarianceHead(
    base_model=backbone,
    target_dim=3,
    input_dim=5,
)

# Forward pass
mean, cov = model(x)

# Evaluate using Task-Agnostic Correlations
metric = TaskAgnosticCorrelations()
tac_error = metric(mean, y, cov)
print(f"Task-Agnostic Correlation Error: {tac_error.item():.4f}")
```

---

## When to use TIC-TAC

| Scenario / Goal | Recommended Choice | Rationale |
|:---|:---:|:---|
| **Scalar regression ($D=1$)** | Avoid | TIC-TAC is designed for capturing multivariate correlation structures ($D \ge 2$). For $D=1$, standard Gaussian NLL or $\beta$-NLL is simpler. |
| **Multivariate targets ($D \ge 2$) with correlated noise** | **Yes (Recommended)** | TIC-TAC ties covariance to mean-model sensitivity, preventing arbitrary covariance collapse. |
| **Very high target dimensions ($D > 20$) or high input features** | **Warning** | Autograd w.r.t inputs (Jacobians/Hessians) can become memory-intensive. |

!!! tip "Activation Functions"
    For TIC-TAC, use smooth activation functions like `nn.Tanh()`, `nn.GELU()`, or `nn.SiLU()` in the backbone. Avoid standard `nn.ReLU()` since its second derivative is zero everywhere (which zero-out the Hessian term $H(x)$).

!!! info "Optimization"
    Because the covariance is parameterized via the mean model's derivatives w.r.t the inputs, gradients propagate through the autograd graphs back to the mean model parameters. This provides strong structural regularization during training.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Shukla et al., "TIC-TAC: A Framework For Improved Covariance Estimation In Deep Heteroscedastic Regression" (ICML 2024). |
