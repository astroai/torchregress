# SIMEX (Simulation Extrapolation)

SIMEX is a simulation-based method for correcting **measurement error in inputs**.  Unlike [RC](rc.md) which analytically debiases, SIMEX works by **adding** progressively more noise then **extrapolating back** to zero noise.

!!! abstract "The idea"
    1. Simulate additional noise at increasing levels $\lambda$  
    2. Train a separate model at each noise level  
    3. Fit a polynomial through predictions vs. $\lambda$  
    4. Extrapolate to $\lambda = -1$ (hypothetical zero-error case)

---

## Mathematical Background

### Adding Simulated Noise

Given observed data with measurement error covariance $\Sigma_u$, SIMEX adds **additional** simulated noise:

$$W_\lambda = W + \sqrt{\lambda}\,U', \qquad U' \sim \mathcal{N}(0,\, \Sigma_u)$$

At level $\lambda$, the total measurement error variance is $(1 + \lambda)\,\Sigma_u$.

### Extrapolation to $\lambda = -1$

The key insight: at $\lambda = -1$, the total noise variance would be:

$$(1 + (-1))\,\Sigma_u = \mathbf{0}$$

SIMEX fits a **polynomial** through predictions at $\lambda \in \{0, 0.5, 1.0, 1.5, 2.0\}$ and extrapolates:

$$\boxed{\;\hat{f}_{\text{SIMEX}}(x) = \sum_{j=0}^{p} \hat\beta_j \cdot (-1)^j\;}$$

where $\hat\beta_j$ are polynomial coefficients from fitting predictions vs. $\lambda$.

---

## Usage

```python
import torch
import torch.nn as nn
from torchregress.algorithms import SIMEX

# Model factory — returns a fresh model for each noise level
def model_factory():
    return nn.Sequential(
        nn.Linear(5, 32), nn.ReLU(),
        nn.Linear(32, 1),
    )

# Training function
def train_func(model, X, y):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(200):
        loss = nn.functional.mse_loss(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

# Create SIMEX corrector
simex = SIMEX(
    model_factory=model_factory,
    train_func=train_func,
    sigma_u=0.5,
    lambdas=[0.5, 1.0, 1.5, 2.0],
    extrapolation_order=2,   # quadratic
)

# Fit trains M+1 models internally
simex.fit(X_train, y_train)

# Predict with measurement-error correction
y_pred = simex.predict(X_test)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `model_factory` | callable | — | Returns a fresh model instance |
| `train_func` | callable | — | `(model, X, y) → trained_model` |
| `sigma_u` | float / Tensor | — | Measurement error (scalar, vector, or cov matrix) |
| `lambdas` | list of float | `[0.5, 1.0, 1.5, 2.0]` | Noise multiplier levels |
| `extrapolation_order` | int | `2` | Polynomial degree ($1$ = linear, $2$ = quadratic) |

---

## RC vs SIMEX

| | Regression Calibration | SIMEX |
|:-|:----------------------|:------|
| **Approach** | Analytical debiasing | Simulation + extrapolation |
| **Model assumptions** | Gaussian noise + linear correction | **Any** model, any loss |
| **Compute cost** | Very low (matrix ops) | High ($M$ model trainings) |
| **Flexibility** | Limited to linear effect | Handles **nonlinear** models |
| **Best for** | Quick correction, linear models | Neural nets, complex models |

---

## When to Use

!!! tip "Good fit for SIMEX"
    - Known measurement uncertainties on inputs
    - Model is **nonlinear** (neural network, tree ensemble)
    - RC's linear correction is **insufficient**

!!! warning "Computational cost"
    SIMEX trains $\lvert\text{lambdas}\rvert + 1$ complete models.  For large networks, consider BatchEnsemble-style weight sharing or use RC as a first approximation.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | J.R. Cook, L.A. Stefanski. "Simulation-Extrapolation Estimation in Parametric Measurement Error Models." *JASA*, 89(428):1314–1328, **1994**. |
| 2 | R.J. Carroll, D. Ruppert, L.A. Stefanski, C.M. Crainiceanu. *Measurement Error in Nonlinear Models*. 2nd ed., Chapman & Hall/CRC, **2006**. |
