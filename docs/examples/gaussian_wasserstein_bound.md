# Gaussian Wasserstein Bound Example

This guide explains how to use `GaussianWassersteinBoundLoss` for joint supervision of predicted means and covariances.

→ API: [`GaussianWassersteinBoundLoss`](../api/losses.md). Guide: [Wasserstein bound](../losses/gaussian_wasserstein.md).

| # | Reference |
|:-:|:----------|
| 1 | Givens, C. R., & Shortt, R. M. (1984). [**A class of Wasserstein metrics for probability distributions**](https://doi.org/10.1307/mmj/1029003026). *Michigan Mathematical Journal*. |

---

## Mathematical Formulation

The 2-Wasserstein distance between two multivariate Gaussians $\mathcal{N}(\mu_1, \Sigma_1)$ and $\mathcal{N}(\mu_2, \Sigma_2)$ is:

$$\mathcal{D}_{W_2}(\mathcal{N}_1, \mathcal{N}_2)^2 = \|\mu_1 - \mu_2\|_2^2 + \text{Tr}\left(\Sigma_1 + \Sigma_2 - 2\left(\Sigma_1^{1/2} \Sigma_2 \Sigma_1^{1/2}\right)^{1/2}\right)$$

Evaluating the exact cross-covariance square root $\left(\Sigma_1^{1/2} \Sigma_2 \Sigma_1^{1/2}\right)^{1/2}$ is computationally expensive and numerically unstable for backpropagation.

`GaussianWassersteinBoundLoss` implements a surrogate upper bound by separating the covariance terms:

$$\mathcal{L}_{W_2\text{-bound}} = \lambda_{\mu} \|\hat{\mu} - \mu\|_2^2 + \lambda_{\Sigma} \|\hat{S} - S\|_F^2$$

where $\|\cdot\|_F$ is the Frobenius norm, and $\hat{S}, S$ are principal matrix square roots of the predicted and target covariances:

$$\hat{S} = \hat{\Sigma}^{1/2}, \quad S = \Sigma^{1/2}$$

### Covariance Parameterization Modes

1.  **`covariance`**: Full symmetric positive semi-definite (SPD) covariance matrices are provided. The principal square root is computed via eigen-decomposition (via `torch.linalg.eigh`):
    $$Q \Lambda Q^\top \implies S = Q \Lambda^{1/2} Q^\top$$
2.  **`cholesky`**: The model outputs lower-triangular Cholesky factors $L$. The loss computes $\Sigma = L L^\top$ and then calculates the principal square roots.
3.  **`sqrt`**: The inputs are treated directly as the symmetric PSD square root matrices $\hat{S}$ and $S$, bypassing eigen-decomposition.
4.  **`diagonal`**: The covariances are diagonal (variances $\hat{v}_i, v_i$), simplifies to:
    $$\mathcal{L}_{\text{diagonal}} = \lambda_{\mu} \|\hat{\mu} - \mu\|_2^2 + \lambda_{\Sigma} \sum_i (\sqrt{\hat{v}_i} - \sqrt{v_i})^2$$

---

## Task-First Context

- **When to Use**: Use this loss when you want to supervise a model to predict both a multivariate **mean** and **covariance matrix** directly using ground truth values (e.g. from physical simulations, ensemble outputs, or neighbor-based covariance pseudo-labels).
- **Alternative**: Unlike standard Gaussian NLL (which only supervises the target point $y$), the Wasserstein bound loss supervises the entire distribution parameters directly, preventing the variance from collapsing or exploding during early training phases.

---

## Code Example

Below is the complete, self-contained code showing a single forward-backward step with `GaussianWassersteinBoundLoss` in full covariance mode.

```python
import argparse
import torch
from torchregress.losses import GaussianWassersteinBoundLoss

def main() -> None:
    # Setup batch dimensions
    b, d = 16, 3

    # Generate mock predicted and target means
    mu_pred = torch.randn(b, d, requires_grad=True)
    mu_tgt = torch.randn(b, d)

    # Generate mock SPD predicted and target covariances
    raw = torch.randn(b, d, d, requires_grad=True)
    sig_pred = raw @ raw.transpose(-1, -2) + 0.25 * torch.eye(d).expand(b, d, d)
    sig_tgt = torch.eye(d).expand(b, d, d) * 0.5

    # Initialize loss function
    fn = GaussianWassersteinBoundLoss(
        covariance_parameterization="covariance",
        mean_weight=1.0,
        covariance_weight=1.0,
        jitter=1e-5,
        reduction="mean",
    )

    # Compute loss and gradients
    loss = fn(mu_pred, mu_tgt, sig_pred, sig_tgt)
    loss.backward()

    print("mean+covariance surrogate loss:", float(loss.item()))
    print("grad norm (mu):", float(mu_pred.grad.norm().item()))
    print("grad norm (raw factor):", float(raw.grad.norm().item()))

if __name__ == "__main__":
    main()
```
