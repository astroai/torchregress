# Balanced MSE & BMC Examples

This guide demonstrates how to use the bin-based reweighting losses `BalancedMSELoss` and `BMCLoss` to tackle target imbalance in regression tasks.

→ Guide: [Imbalanced losses](../losses/imbalanced.md). API: [`BalancedMSELoss`](../api/losses.md), [`BMCLoss`](../api/losses.md), [imbalanced loss section](../api/losses.md).

| # | Reference |
|:-:|:----------|
| 1 | Ren, J., Zhang, C., Liu, S., Yang, H., & Yang, M. H. (2022). [**Balanced MSE for Long-Tailed Visual Recognition**](https://arxiv.org/abs/2203.16427). *CVPR*. |

---

## Mathematical Formulations

In imbalanced regression tasks (e.g. long-tailed target distributions), standard Mean Squared Error (MSE) tends to over-focus on dense target regions and ignore the rare tails. This is because MSE implicitly assumes a uniform distribution of targets:

$$\mathcal{L}_{\text{MSE}}(y, \hat{y}) = (y - \hat{y})^2$$

### Balanced MSE

Balanced MSE addresses this by incorporating the marginal label distribution $p(y)$ into the likelihood. Assuming a Gaussian predictive distribution $p(y \mid x) = \mathcal{N}(\hat{y}, \sigma^2)$, the balanced likelihood is defined as:

$$p_{\text{bal}}(y \mid x) = \frac{p(y \mid x) / p(y)}{\int p(y' \mid x) / p(y') dy'}$$

For bin-based discretization, this simplifies to a weighted MSE:

$$\mathcal{L}_{\text{BalancedMSE}}(y, \hat{y}) = w(y) (y - \hat{y})^2$$

where the per-sample weight $w(y)$ is inversely proportional to the frequency of the bin containing target $y$:

$$w(y) = \frac{1}{n(y) + \epsilon}$$

Here, $n(y)$ is the empirical sample count in the bin containing $y$, and $\epsilon \ge 0$ is a Laplace-style smoothing parameter (`count_smoothing`).

### BMCLoss (Balanced Mean Squared Error)

`BMCLoss` automatically constructs the bin edges (using either equal-width or quantile bin splits) and applies a noise parameter $\sigma_{\text{noise}}$ as a Laplace smoothing constant:

$$w(y) = \frac{1}{\text{count}(y) + \sigma_{\text{noise}}}$$

Larger values of $\sigma_{\text{noise}}$ regularize the weights towards uniform reweighting, which is useful to prevent noise amplification in extremely sparse tail bins.

---

## Task-First Context

- **When to Use**: Use these losses when you have **rare extreme values** or a **highly skewed/long-tailed target distribution**, and you want your model's predictions to not ignore the tails.
- **Comparison Notes**: Always compare imbalanced losses against standard MSE using a shared test set. Report both overall MSE and tail-region MSE.
- **Calibration Warning**: Aggressive reweighting can distort the model's uncertainty estimates. Always validate calibration post-hoc or apply calibration constraints.

---

## Code Example

Below is the complete, self-contained code demonstrating how to train models using `BalancedMSELoss` and `BMCLoss` on a skewed target distribution.

```python
import argparse
import torch
import torch.nn as nn
from torchregress.losses import BalancedMSELoss, BMCLoss, WeightedMSELoss

def make_skewed_targets(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(-1.0, 1.0, n).unsqueeze(1)
    # 90% of mass near 0, 10% large positive shift
    y = 0.5 * x + 0.1 * torch.randn(n, 1, generator=g)
    big = torch.rand(n, generator=g) < 0.1
    y = y + big.float().unsqueeze(1) * 2.5
    return x, y

def train_linear(
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    *,
    steps: int,
    lr: float,
) -> float:
    model = nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(x)
        return float(torch.mean((pred - y) ** 2).item())

def main() -> None:
    # Synthetic long-tailed data
    x, y = make_skewed_targets(256, 0)
    lo, hi = float(y.min()), float(y.max())
    edges = torch.linspace(lo, hi, 11)

    # Initialize losses
    bmc = BMCLoss(num_bins=10, noise_sigma=1.0, binning="equal").fit(y)
    bal = BalancedMSELoss(bin_edges=edges, count_smoothing=0.5).fit(y)
    mse = WeightedMSELoss()

    # Train linear models
    rmse_mse = train_linear(x, y, mse, steps=80, lr=0.05)
    rmse_bal = train_linear(x, y, bal, steps=80, lr=0.05)
    rmse_bmc = train_linear(x, y, bmc, steps=80, lr=0.05)

    print("Mean squared error on training set (lower is better):")
    print(f"  MSE loss:              {rmse_mse:.6f}")
    print(f"  BalancedMSELoss:       {rmse_bal:.6f}")
    print(f"  BMCLoss:               {rmse_bmc:.6f}")

if __name__ == "__main__":
    main()
```
