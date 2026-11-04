# Gaussian Wasserstein Bound Example

This guide explains how to use `GaussianWassersteinBoundLoss` for joint supervision of predicted means and covariances, and how to use it as a pretraining phase before Gaussian NLL fine-tuning.

**Run the script** [`examples/gaussian_wasserstein_bound_demo.py`](https://github.com/astroai/torchregress/blob/main/examples/gaussian_wasserstein_bound_demo.py) (`--mode all` by default runs both):

```bash
pixi run python examples/gaussian_wasserstein_bound_demo.py --mode loss    # one full-covariance loss step (--batch, --dim)
pixi run python examples/gaussian_wasserstein_bound_demo.py --mode hybrid  # W2-bound pretrain, then Gaussian NLL fine-tune
```

| Mode | What it shows | Flags |
|:--|:--|:--|
| `loss` | One forward/backward step in `covariance` parameterization on random SPD covariances | `--batch`, `--dim`, `--seed` |
| `hybrid` | [Hybrid pretraining](#hybrid-pretraining-w2-bound-then-gaussian-nll): diagonal W2-bound on neighbourhood pseudo-covariances, then `GaussianNLLLoss` | `--pretrain-steps`, `--finetune-steps`, `--n`, `--lr`, `--seed` |

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

## Code Example (`--mode loss`)

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

---

## Hybrid Pretraining: W2-Bound then Gaussian NLL

→ Guide: [Gaussian Wasserstein loss](../losses/gaussian_wasserstein.md). API: [`GaussianWassersteinBoundLoss`](../api/losses.md), [`GaussianNLLLoss`](../api/losses.md), [`mahalanobis_covariance_pseudo_labels`](../api/algorithms.md).

This guide explains how to use the hybrid pretraining recipe to stabilize heteroscedastic regression models. It pretrains a mean-variance model using [`GaussianWassersteinBoundLoss`](../api/losses.md) against neighborhood covariance pseudo-labels before fine-tuning with [`GaussianNLLLoss`](../api/losses.md).

---

### The Hybrid Pretraining Recipe

Training heteroscedastic neural networks directly with Gaussian Negative Log-Likelihood (NLL) can be highly unstable:

$$\mathcal{L}_{\text{NLL}}(y, \mu, \sigma^2) = \frac{(y - \mu)^2}{2\sigma^2} + \frac{1}{2}\log\sigma^2$$

During early training, predictions $\mu(x)$ can be far from $y$. The network can minimize the loss by simply predicting extremely large variances $\sigma^2$, leading to a collapsed variance head that fails to recover.

To prevent this, we use a two-phase hybrid training recipe:

#### Phase 1: Neighborhood Covariance Pseudo-Labeling

For each training point $(x_i, y_i)$, we identify its $k$-nearest neighbors in input space, $\mathcal{N}(x_i)$. We estimate the local target variance $\sigma_i^2$ as the empirical variance of these neighbors:

$$\sigma_i^2 = \frac{1}{|\mathcal{N}(x_i)|} \sum_{j \in \mathcal{N}(x_i)} (y_j - \bar{y}_i)^2$$

#### Phase 2: Wasserstein-Bound Pretraining

We pretrain the network using the `GaussianWassersteinBoundLoss` (in diagonal mode) using the target $y_i$ for the mean and the pseudo-label $\sigma_i^2$ for the variance:

$$\mathcal{L}_{\text{pretrain}} = (\mu(x_i) - y_i)^2 + \left(\sqrt{\sigma^2(x_i)} - \sqrt{\sigma_i^2}\right)^2$$

Because this loss is linear/quadratic in the standard deviations rather than containing $1/\sigma^2$, it provides stable, well-behaved gradients even if the variance predictions are far from the target.

#### Phase 3: Fine-Tuning

Once the variance predictions are initialized to the correct scale, we fine-tune the model using [`GaussianNLLLoss`](../api/losses.md) to align the predictions with the true conditional distribution $p(y \mid x)$.

---

### Code Example (`--mode hybrid`)

Below is the complete, self-contained code showing how to set up and run this hybrid pretraining schedule.

```python
import argparse
import torch
import torch.nn as nn
from torchregress.algorithms import (
    NeighborhoodCovarianceConfig,
    NeighborhoodCovariancePseudoLabeler,
)
from torchregress.losses import GaussianNLLLoss, GaussianWassersteinBoundLoss

class ScalarGaussianHead(nn.Module):
    """Maps x [*, 1] to mean and log-variance."""
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.lin(x)
        return o[..., :1], o[..., 1:2]

def main() -> None:
    # Set seed and generate synthetic data with heteroscedastic noise
    torch.manual_seed(0)
    x = torch.linspace(-2.0, 2.0, 256).unsqueeze(-1)
    noise_scale = 0.15 * (1.0 + torch.abs(x))
    y = 0.8 * x + 0.3 * (x**2) + noise_scale * torch.randn_like(x)

    # 1. Generate neighborhood covariance pseudo-labels
    head = ScalarGaussianHead()
    labeler = NeighborhoodCovariancePseudoLabeler(
        NeighborhoodCovarianceConfig(n_neighbors=16, metric="euclidean", temperature=0.5)
    )
    cov_pseudo = labeler.fit_predict(x, y)
    target_var = cov_pseudo.squeeze(-1).squeeze(-1).clamp(min=1e-4).unsqueeze(-1)

    # 2. Phase 1: Wasserstein pretraining
    gw = GaussianWassersteinBoundLoss(
        covariance_parameterization="diagonal",
        mean_weight=1.0,
        covariance_weight=1.0,
        reduction="mean",
    )
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        mu, logv = head(x)
        v = torch.exp(logv).clamp(min=1e-4)
        loss = gw(mu, y, v, target_var)
        loss.backward()
        opt.step()

    with torch.no_grad():
        mu0, logv0 = head(x)
        nll_loss = GaussianNLLLoss(reduction="mean")
        nll0 = nll_loss(torch.cat([mu0, logv0], dim=-1), y)

    # 3. Phase 2: Gaussian NLL fine-tuning
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        mu, logv = head(x)
        loss = nll_loss(torch.cat([mu, logv], dim=-1), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        mu1, logv1 = head(x)
        nll1 = nll_loss(torch.cat([mu1, logv1], dim=-1), y)

    print("NLL after W2-bound pretrain:", round(float(nll0.item()), 4))
    print("NLL after GaussianNLL fine-tune:", round(float(nll1.item()), 4))

if __name__ == "__main__":
    main()
```
