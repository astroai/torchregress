# Bayesian Linear Head (Test-Time) Example

This guide demonstrates how to use the conjugate `BayesianLinearHead` and its online recursive counter-part `RecursiveBayesianHead` for rapid test-time adaptation of deep network representation layers.

→ API: [Test-time](../api/test_time.md) (`BayesianLinearHead`, `RecursiveBayesianHead`, `SupportsPredictiveBatch`).

**Run the script** [`examples/test_time_bayesian_linear_head_demo.py`](https://github.com/astroai/torchregress/blob/main/examples/test_time_bayesian_linear_head_demo.py) (`--mode all` by default runs both):

```bash
pixi run python examples/test_time_bayesian_linear_head_demo.py --mode exactness  # batch vs recursive posterior, coverage
pixi run python examples/test_time_bayesian_linear_head_demo.py --mode adapter    # SupportsPredictiveBatch adapter
```

| Mode | What it shows | Default sizes |
|:--|:--|:--|
| `exactness` | Batch `fit` vs two-step `RecursiveBayesianHead` (`forgetting_factor=1`), weight error, 95% interval coverage | `--n-train 200 --n-test 500 --dim 5 --noise 0.35` |
| `adapter` | [Predictive adapter](#predictive-adapter-supportspredictivebatch): wraps the head as a `SupportsPredictiveBatch` returning `PredictiveBatch` | `--n-train 96 --n-test 24 --dim 4` |

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

## Code Example (`--mode exactness`)

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

---

## Predictive Adapter (SupportsPredictiveBatch)

This guide explains how to wrap a `BayesianLinearHead` with a lightweight adapter class to satisfy the `SupportsPredictiveBatch` protocol, enabling seamless integration with downstream adaptation pipelines.

### The SupportsPredictiveBatch Protocol

Downstream test-time adaptation and uncertainty evaluation algorithms in `torchregress` often interact with models using a standardized protocol. Rather than requiring models to inherit from a specific base class, the library defines runtime protocols.

A key protocol is `SupportsPredictiveBatch`, which requires the model to implement a `predict_distribution` method:

```python
class SupportsPredictiveBatch(Protocol):
    def predict_distribution(self, X: np.ndarray, **kwargs: Any) -> PredictiveBatch:
        ...
```

The returned `PredictiveBatch` is a container that bundles:
1.  **`mean`**: The predictive mean tensor (point predictions).
2.  **`std`**: The predictive standard deviation (total uncertainty).
3.  **`extra`**: A dictionary containing auxiliary predictive outputs (e.g. epistemic and aleatoric variances, log-variances, or quantiles).

#### Adapter Mathematical Mapping

The `BLRPredictiveAdapter` wraps a trained `BayesianLinearHead` and routes calls to `predictive_batch`, returning a `PredictiveBatch` computed using closed-form BLR predictive moments:

$$\mu_* = \phi(X_{\text{test}})^\top m_N$$

$$\sigma_*^2 = \sigma_{\text{noise}}^2 + \phi(X_{\text{test}})^\top S_N \phi(X_{\text{test}})$$

---

### Code Example (`--mode adapter`)

Below is the complete, self-contained code demonstrating how to construct and use this adapter wrapper.

```python
import argparse
import numpy as np
from torchregress.prediction import PredictiveBatch
from torchregress.test_time import BayesianLinearHead, SupportsPredictiveBatch

class BLRPredictiveAdapter:
    """Minimal adapter implementing SupportsPredictiveBatch."""

    def __init__(self, head: BayesianLinearHead) -> None:
        self.head = head

    def predict_distribution(self, X: np.ndarray, **kwargs: object) -> PredictiveBatch:
        include_noise = bool(kwargs.get("include_noise", True))
        return self.head.predictive_batch(X, include_noise=include_noise)

def _make_toy(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=(d, 1)).astype(np.float32)
    y = x @ w + 0.15 * rng.normal(size=(n, 1)).astype(np.float32)
    return x, y

def main() -> None:
    # Setup data
    n_train, n_test, d, seed = 96, 24, 4, 0
    x_train, y_train = _make_toy(n_train, d, seed)
    x_test, y_test = _make_toy(n_test, d, seed + 1)

    # Train Bayesian Linear Head
    head = BayesianLinearHead(
        in_features=d,
        out_features=1,
        fit_intercept=True,
        prior_precision=1.0,
        noise_variance=0.2**2,
    ).fit(x_train, y_train)

    # Wrap in adapter
    adapter = BLRPredictiveAdapter(head)

    # Assert at runtime that the adapter conforms to the protocol
    assert isinstance(adapter, SupportsPredictiveBatch)

    # Generate predictions via the adapter
    pb = adapter.predict_distribution(x_test, include_noise=True)
    rmse = float(np.sqrt(np.mean((pb.mean.detach().cpu().numpy() - y_test) ** 2)))
    mean_std = float(pb.std.detach().mean().item())

    print(f"BLR adapter RMSE={rmse:.4f}, mean predictive std={mean_std:.4f}")
    print("PredictiveBatch extra keys:", sorted(pb.extra.keys()))

if __name__ == "__main__":
    main()
```
