# Errors-in-Variables Correction Examples

This guide explains how to use input error correction algorithms, namely **Regression Calibration (RC)**, **Simulation Extrapolation (SIMEX)**, and **Latent-Input Neural Networks (LatentNN)**, to handle measurement noise in feature variables.

| # | Reference |
|:-:|:----------|
| 1 | Carroll, R. J., Ruppert, D., Crainiceanu, C. M., & Stefanski, L. A. (2006). [**Measurement Error in Nonlinear Models: A Modern Perspective**](https://doi.org/10.1201/9781420010138). *Chapman & Hall/CRC*. |
| 2 | Cook, J. R., & Stefanski, L. A. (1994). [**Simulation-Extrapolation Estimation in Parametric Measurement Error Models**](https://doi.org/10.1080/01621459.1994.10476871). *Journal of the American Statistical Association*. |

---

## Mathematical Formulations

Let $X$ be the unobserved true inputs and $W$ be the observed noisy inputs:
$$W = X + U$$
where $U \sim \mathcal{N}(0, \Sigma_u)$ is measurement noise with known covariance.

### Regression Calibration (RC)

Assuming both the true signal $X$ and noise $U$ are Gaussian, the conditional expectation of $X$ given the observed data $W$ is:
$$E[X \mid W] = \mu_w + \Sigma_x (\Sigma_x + \Sigma_u)^{-1} (W - \mu_w)$$
where:
*   $\mu_w$ is the mean of $W$
*   $\Sigma_w$ is the covariance of $W$
*   $\Sigma_x = \Sigma_w - \Sigma_u$ is the estimated true signal covariance

We replace the noisy inputs $W$ with their calibrated estimates $\hat{X} = E[X \mid W]$ during model training and inference.

### Simulation Extrapolation (SIMEX)

SIMEX corrects measurement error by adding simulated noise of varying levels $\lambda > 0$, measuring the trend of performance degradation, and extrapolating back to the noise-free case ($\lambda = -1$):

1.  **Simulation**: For each $\lambda \in \{\lambda_1, \dots, \lambda_M\}$ and replicate $b = 1, \dots, B$, generate:
    $$W_{b}(\lambda) = W + \sqrt{\lambda} U_b, \quad U_b \sim \mathcal{N}(0, \Sigma_u)$$
2.  **Estimation**: Train models on each simulated dataset to predict $\hat{y}_b(\lambda)$.
3.  **Extrapolation**: Fit a polynomial model of order $d \in \{1, 2\}$ to the mean predictions $\bar{y}(\lambda) = \frac{1}{B} \sum_{b=1}^B \hat{y}_b(\lambda)$ and extrapolate to $\lambda = -1$.

### Latent-Input Neural Networks (LatentNN)

LatentNN treats clean inputs $X$ as learnable latent parameters $X_{\text{latent}}$. During training, it jointly optimizes model parameters $\theta$ and latent clean inputs $X_{\text{latent}}$ by minimizing a composite loss function:
$$\mathcal{L}(\theta, X_{\text{latent}}) = L_{\text{model}}(f_{\theta}(X_{\text{latent}}), Y) + \lambda_x \left\| \frac{X_{\text{latent}} - W}{\sigma_x} \right\|_2^2$$
where the second term acts as a Gaussian quadratic penalty to keep the latent inputs close to the observed values $W$.

---

## Code Example

Below is a complete example demonstrating the calibration algorithms on noisy input data.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchregress.algorithms import RegressionCalibration, SIMEX, LatentNN

# Generate synthetic EIV data
np.random.seed(42)
n_samples = 1000
sigma_u = 0.4

x_true = np.random.uniform(-2.0, 2.0, size=(n_samples, 1)).astype(np.float32)
y = (0.5 * x_true**2 + 0.8 * x_true + np.random.normal(0, 0.1, size=x_true.shape)).astype(np.float32)
w_obs = x_true + np.random.normal(0, sigma_u, size=x_true.shape).astype(np.float32)

w_tensor = torch.from_numpy(w_obs)
y_tensor = torch.from_numpy(y)

# 1. Regression Calibration
rc = RegressionCalibration(sigma_u=sigma_u)
w_calibrated = rc.fit_transform(w_tensor)

# 2. SIMEX Extrapolation
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x)

def train_func(model, x, y):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(50):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(bx), by)
            loss.backward()
            optimizer.step()
    return model

simex = SIMEX(
    model_factory=Regressor,
    train_func=train_func,
    sigma_u=sigma_u,
    lambdas=[0.5, 1.0, 1.5, 2.0],
    n_simulations=3,
)
simex.fit(w_tensor, y_tensor)
pred_simex = simex.predict(w_tensor)

# 3. Latent Input Neural Network (LatentNN)
latent_nn = LatentNN(
    model_factory=Regressor,
    sigma_x=sigma_u,
    epochs=100,
)
latent_nn.fit(w_tensor, y_tensor)
pred_latent = latent_nn.predict(w_tensor)

print("RC, SIMEX, and LatentNN models trained and executed successfully.")
```
