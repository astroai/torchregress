# BLR Predictive Adapter (SupportsPredictiveBatch) Example

This guide explains how to wrap a `BayesianLinearHead` with a lightweight adapter class to satisfy the `SupportsPredictiveBatch` protocol, enabling seamless integration with downstream adaptation pipelines.

---

## The SupportsPredictiveBatch Protocol

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

### Adapter Mathematical Mapping

The `BLRPredictiveAdapter` wraps a trained `BayesianLinearHead` and routes calls to `predictive_batch`, returning a `PredictiveBatch` computed using closed-form BLR predictive moments:

$$\mu_* = \phi(X_{\text{test}})^\top m_N$$

$$\sigma_*^2 = \sigma_{\text{noise}}^2 + \phi(X_{\text{test}})^\top S_N \phi(X_{\text{test}})$$

---

## Code Example

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
