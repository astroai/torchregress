# Wasserstein-Bound Hybrid Pretraining Example

This guide explains how to use the hybrid pretraining recipe to stabilize heteroscedastic regression models. It pretrains a mean-variance model using `GaussianWassersteinBoundLoss` against neighborhood covariance pseudo-labels before fine-tuning with `GaussianNLLLoss`.

---

## The Hybrid Pretraining Recipe

Training heteroscedastic neural networks directly with Gaussian Negative Log-Likelihood (NLL) can be highly unstable:

$$\mathcal{L}_{\text{NLL}}(y, \mu, \sigma^2) = \frac{(y - \mu)^2}{2\sigma^2} + \frac{1}{2}\log\sigma^2$$

During early training, predictions $\mu(x)$ can be far from $y$. The network can minimize the loss by simply predicting extremely large variances $\sigma^2$, leading to a collapsed variance head that fails to recover.

To prevent this, we use a two-phase hybrid training recipe:

### Phase 1: Neighborhood Covariance Pseudo-Labeling

For each training point $(x_i, y_i)$, we identify its $k$-nearest neighbors in input space, $\mathcal{N}(x_i)$. We estimate the local target variance $\sigma_i^2$ as the empirical variance of these neighbors:

$$\sigma_i^2 = \frac{1}{|\mathcal{N}(x_i)|} \sum_{j \in \mathcal{N}(x_i)} (y_j - \bar{y}_i)^2$$

### Phase 2: Wasserstein-Bound Pretraining

We pretrain the network using the `GaussianWassersteinBoundLoss` (in diagonal mode) using the target $y_i$ for the mean and the pseudo-label $\sigma_i^2$ for the variance:

$$\mathcal{L}_{\text{pretrain}} = (\mu(x_i) - y_i)^2 + \left(\sqrt{\sigma^2(x_i)} - \sqrt{\sigma_i^2}\right)^2$$

Because this loss is linear/quadratic in the standard deviations rather than containing $1/\sigma^2$, it provides stable, well-behaved gradients even if the variance predictions are far from the target.

### Phase 3: Fine-Tuning

Once the variance predictions are initialized to the correct scale, we fine-tune the model using the standard `GaussianNLLLoss` to align the predictions with the true conditional distribution $p(y \mid x)$.

---

## Code Example

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
