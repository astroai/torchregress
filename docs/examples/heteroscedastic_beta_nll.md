# Beta-NLL Heteroscedastic Regression Example

→ Guide: [Beta-NLL loss](../losses/beta_nll.md). API: [`BetaNLLLoss`](../api/losses.md#betanllloss), [`GaussianNLLLoss`](../api/losses.md#gaussiannllloss).

This example demonstrates how to use the [`BetaNLLLoss`](../api/losses.md#betanllloss) to train heteroscedastic neural networks that are robust to variance overestimation and outliers. Compare with [`GaussianNLLLoss`](../api/losses.md#gaussiannllloss) in the [loss guide](../losses/beta_nll.md).

| # | Reference |
|:-:|:----------|
| 1 | Seitzer, M., Tavakoli, A., Brown, D., & Peters, J. (2022). [**On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks**](https://openreview.net/forum?id=aPlHsXlwbd). *ICLR*. |

---

## Mathematical Formulation

In heteroscedastic regression, a model predicts both a mean $\mu(x)$ and variance $\sigma^2(x)$ for each input $x$. The standard loss is the Gaussian Negative Log-Likelihood (NLL):

$$\mathcal{L}_{\text{NLL}}(y, \mu, \sigma^2) = \frac{(y - \mu)^2}{2\sigma^2} + \frac{1}{2}\log\sigma^2$$

A well-known issue with standard NLL is its sensitivity to outliers or training inaccuracies. For a large prediction error $(y - \mu)^2$, the gradient with respect to the variance $\sigma^2$ is heavily biased towards increasing the variance:

$$\frac{\partial \mathcal{L}_{\text{NLL}}}{\partial \sigma^2} = \frac{1}{2\sigma^2} \left(1 - \frac{(y - \mu)^2}{\sigma^2}\right)$$

If an outlier is present, the model will drastically inflate $\sigma^2(x)$ to minimize the loss, leading to poorly calibrated variance estimates.

### Beta-NLL Loss

`BetaNLLLoss` solves this issue by weighting the loss with a power of the predicted variance:

$$\mathcal{L}_{\text{BetaNLL}}(y, \mu, \sigma^2) = \left( \sigma^2 \right)^\beta \left( \frac{(y - \mu)^2}{2\sigma^2} + \frac{1}{2}\log\sigma^2 \right)$$

where $\beta \in \[0, 1\]$ is a hyperparameter:

- **$\beta = 0$**: Recovers standard Gaussian NLL.
- **$\beta = 1$**: Under $\beta = 1$, the gradient of the loss with respect to the variance becomes independent of the magnitude of the absolute error, rendering variance estimation highly robust to outliers.
- **$\beta = 0.5$**: Typically offers the best trade-off between calibration quality and outlier robustness.

---

## Task-First Context

- **When to Use**: Use [`BetaNLLLoss`](../api/losses.md#betanllloss) for **heteroscedastic regression** tasks when the training data contains **heavy-tailed noise** or **outliers** that cause standard Gaussian NLL to overestimate predictive variance.
- **Comparison Notes**: Evaluate models trained with standard NLL vs. Beta-NLL on a clean, held-out validation set. Report both RMSE (for point prediction accuracy) and standard Gaussian NLL (for calibration quality).

---

## Code Example

Below is the complete, self-contained code comparing Gaussian NLL against Beta-NLL on a synthetic heteroscedastic dataset.

```python
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchregress.losses import BetaNLLLoss, GaussianNLLLoss

class GaussHeadMLP(nn.Module):
    """MLP predicting mean and log-variance."""
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean, log_var = out[:, :1], out[:, 1:2]
        return mean, log_var

def make_data(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n, 1).uniform_(-2.0, 2.0, generator=g)
    noise_std = 0.15 + 0.45 * torch.abs(x[:, 0:1])
    eps = torch.randn(n, 1, generator=g) * noise_std
    y = 2.0 * x + eps
    return x, y

def train(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            mean, log_var = model(xb)
            loss = loss_fn((mean, log_var), yb)
            loss.backward()
            opt.step()

def run_once(
    loss_name: str,
    *,
    epochs: int,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_va: torch.Tensor,
    y_va: torch.Tensor,
    init_state: dict,
    lr: float,
) -> dict[str, float]:
    ds = TensorDataset(x_tr, y_tr)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = GaussHeadMLP()
    model.load_state_dict(init_state)

    crit = GaussianNLLLoss() if loss_name == "nll" else BetaNLLLoss(beta=0.5)
    train(model, crit, loader, epochs=epochs, lr=lr)

    model.eval()
    with torch.no_grad():
        mean, log_var = model(x_va)
        nll_val = float(GaussianNLLLoss(reduction="mean")((mean, log_var), y_va).item())
        rmse_val = float(torch.sqrt(torch.mean((mean - y_va) ** 2)).item())

    return {"rmse": rmse_val, "val_nll": nll_val}

def main() -> None:
    # Setup data
    x, y = make_data(2000, 0)
    x_tr, y_tr = x[:-500], y[:-500]
    x_va, y_va = x[-500:], y[-500:]

    # Initialize shared model weights
    init = GaussHeadMLP().state_dict()

    # Run comparison
    n_metrics = run_once("nll", epochs=80, x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, init_state=init, lr=1e-3)
    b_metrics = run_once("beta_nll", epochs=80, x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, init_state=init, lr=1e-3)

    print("Gaussian NLL — val RMSE:", round(n_metrics["rmse"], 5), "  val NLL:", round(n_metrics["val_nll"], 5))
    print("Beta-NLL     — val RMSE:", round(b_metrics["rmse"], 5), "  val NLL:", round(b_metrics["val_nll"], 5))

if __name__ == "__main__":
    main()
```
