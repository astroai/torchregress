# Bayesian Linear Head (Test-Time) Example

This guide demonstrates how to use the conjugate `BayesianLinearHead` and its online recursive counter-part `RecursiveBayesianHead` for rapid test-time adaptation of deep network representation layers.

→ API: [Test-time](../api/test_time.md) (`BayesianLinearHead`, `RecursiveBayesianHead`).

| # | Reference |
|:-:|:----------|
| 1 | Bishop, C. M. (2006). [**Pattern Recognition and Machine Learning**](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/). *Springer*. (Chapter 3: Linear Models for Regression). |

---

## Mathematical Formulation

In feature-based regression, a deep network maps inputs to a latent representation $\phi(x) \in \mathbb{R}^D$, and a linear head predicts the target $y = \phi(x)^\top w$.

Bayesian Linear Regression (BLR) treats the weights $w$ as a random variable. Under a conjugate Gaussian prior and Gaussian likelihood, the posterior distribution can be computed in closed form.

### 1. Prior
We assume a Gaussian prior over the weights:

$$p(w) = \mathcal{N}(m_0, S_0) = \mathcal{N}(0, \alpha^{-1} I)$$

where $\alpha$ is the `prior_precision`.

### 2. Likelihood
We assume a Gaussian noise model with variance $\sigma_{\text{noise}}^2$:

$$p(y \mid \phi(x), w) = \mathcal{N}(\phi(x)^\top w, \sigma_{\text{noise}}^2)$$

### 3. Batch Posterior Update
Given a batch of features $\Phi \in \mathbb{R}^{N \times D}$ and targets $y \in \mathbb{R}^N$, the posterior distribution $p(w \mid \Phi, y) = \mathcal{N}(m_N, S_N)$ has:

$$S_N^{-1} = S_0^{-1} + \sigma_{\text{noise}}^{-2} \Phi^\top \Phi$$

$$m_N = S_N \left(S_0^{-1} m_0 + \sigma_{\text{noise}}^{-2} \Phi^\top y\right)$$

### 4. Recursive Update with Forgetting Factor
In non-stationary environments, we can decay past information using a forgetting factor $\gamma \in (0, 1]$:

$$S_t^{-1} = \gamma S_{t-1}^{-1} + \sigma_{\text{noise}}^{-2} \Phi_t^\top \Phi_t$$

$$m_t = S_t \left(\gamma S_{t-1}^{-1} m_{t-1} + \sigma_{\text{noise}}^{-2} \Phi_t^\top y_t\right)$$

When $\gamma = 1.0$, the recursive update matches the batch posterior exactly.

### 5. Predictive Distribution
For a new input feature $\phi(x_*)$, the predictive distribution is:

$$p(y_* \mid \phi(x_*)) = \mathcal{N}(\mu_*, \sigma_*^2)$$

$$\mu_* = \phi(x_*)^\top m_N$$

$$\sigma_*^2 = \sigma_{\text{noise}}^2 + \phi(x_*)^\top S_N \phi(x_*)$$

---

## Task-First Context

- **When to Use**: Use these heads when you need **fast adaptation** at test time (e.g. streaming data, domain shift) and want exact, closed-form updates instead of gradient descent.
- **Comparison Notes**: Ensure that recursive updates (using `partial_fit`) match the batch updates when the forgetting factor is set to 1.0.

---

## Code Example

Below is the complete, self-contained code comparing batch and recursive Bayesian linear head updates on synthetic linear data.

```python
import argparse
import torch
from torchregress.test_time import BayesianLinearHead, RecursiveBayesianHead

def main() -> None:
    # Setup data parameters
    n_train, n_test, d, noise = 200, 500, 5, 0.35
    torch.manual_seed(0)
    w_true = torch.randn(d)

    # Generate synthetic features and targets
    x_train = torch.randn(n_train, d)
    y_train = (x_train @ w_true).unsqueeze(-1) + noise * torch.randn(n_train, 1)
    x_test = torch.randn(n_test, d)
    y_test = (x_test @ w_true).unsqueeze(-1) + noise * torch.randn(n_test, 1)

    noise_var = noise**2
    cfg = dict(
        in_features=d,
        fit_intercept=False,
        prior_precision=1e-2,
        noise_variance=noise_var,
    )

    # 1. Batch adaptation
    batch_model = BayesianLinearHead(**cfg).fit(x_train, y_train)

    # 2. Recursive adaptation (two steps)
    rec_model = RecursiveBayesianHead(**cfg, forgetting_factor=1.0)
    mid = n_train // 2
    rec_model.partial_fit(x_train[:mid], y_train[:mid])
    rec_model.partial_fit(x_train[mid:], y_train[mid:])

    # 3. Check mathematical exactness and coverage
    max_post_diff = (batch_model.posterior_mean - rec_model.posterior_mean).abs().max().item()
    w_err = (batch_model.posterior_mean[0] - w_true).norm().item()
    print("max |posterior_mean_batch - posterior_mean_recursive|:", round(max_post_diff, 8))
    print("||posterior_mean - w_true||_2:", round(w_err, 4))

    # Evaluate predictive intervals on held-out test points
    pred = batch_model.predict(x_test, return_std=True, include_noise=True)
    mean, std = pred["mean"], pred["std"]
    z = (y_test - mean) / std.clamp(min=1e-8)
    coverage_95 = ((z.abs() <= 1.96).float().mean()).item()
    print("empirical 95% Gaussian interval coverage (held-out):", round(coverage_95, 3))

if __name__ == "__main__":
    main()
```
