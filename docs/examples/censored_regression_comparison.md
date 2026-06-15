# Censored Regression Comparison

This guide compares multiple regression loss functions designed to train models on censored data (where some targets are only partially observed).

→ Loss API: [`CensoredGaussianNLLLoss`](../api/losses.md#censoredgaussiannllloss), [`CensoredQuantileLoss`](../api/losses.md#censoredquantileloss), [`AFTLoss`](../api/losses.md#aftloss). Guide: [Censored losses](../losses/censored.md).

| # | Reference |
|:-:|:----------|
| 1 | Tobin, J. (1958). [**Estimation of Relationships for Limited Dependent Variables**](https://www.jstor.org/stable/1907382). *Econometrica*. (The Tobit model). |
| 2 | Buckley, J., & James, I. (1979). [**Linear Regression with Censored Data**](https://www.jstor.org/stable/2335161). *Biometrika*. |

---

## Mathematical Formulations

Censoring occurs when the exact value of a target variable $T$ is not known. Instead, we observe a threshold value and the direction of the censoring. We classify censoring using the following codes:
- **$C = 0$ (Observed)**: The exact value is observed ($T_i = y_i$).
- **$C = 1$ (Right-Censored)**: The target is only known to be at least $y_i$ ($T_i \ge y_i$). For example, a patient survived *at least* $y_i$ days.
- **$C = -1$ (Left-Censored)**: The target is only known to be at most $y_i$ ($T_i \le y_i$). For example, a chemical concentration is below the detection limit $y_i$.
- **Interval-Censored**: The target is only known to lie within a range $[L_i, U_i]$ ($T_i \in [L_i, U_i]$).

### 1. Censored Gaussian NLL (Tobit Model)

Under a Gaussian predictive model $T_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$, we define the normalized prediction error $z_i = \frac{y_i - \mu_i}{\sigma_i}$. The censored negative log-likelihood is:

$$\mathcal{L} = -\sum_{i=1}^N \ell_i$$

where:

$$\ell_i = \begin{cases}
-\frac{1}{2}\log(2\pi\sigma_i^2) - \frac{(y_i - \mu_i)^2}{2\sigma_i^2} & \text{if } C_i = 0 \\
\log(1 - \Phi(z_i)) & \text{if } C_i = 1 \\
\log(\Phi(z_i)) & \text{if } C_i = -1 \\
\log\left(\Phi\left(\frac{U_i - \mu_i}{\sigma_i}\right) - \Phi\left(\frac{L_i - \mu_i}{\sigma_i}\right)\right) & \text{if interval-censored}
\end{cases}$$

Here, $\Phi$ is the standard normal cumulative distribution function (CDF).

### 2. Accelerated Failure Time (AFT) Loss

AFT models model the logarithm of the survival time $T$ log-linearly:

$$\log(T_i) = \mu_i + \sigma_i \cdot \epsilon_i$$

where $\epsilon_i$ represents the residual noise following a specific distribution (e.g., standard normal or extreme-value Gumbel). Like Censored Gaussian NLL, the AFT loss maximizes the survival likelihood in log-space, making it ideal for positive skewed targets (like failure times).

---

## Task-First Context

- **When to Use**: Use censored losses in clinical survival analysis, equipment predictive maintenance, or when target measurements hit assay ceiling/floor detection limits.
- **Comparison Metrics**: Report point accuracy on the unobserved true values (`MAE_true`), point accuracy on observed boundaries (`ObsMAE`), and the **Concordance Index** (C-index) which measures how well the model predicts the correct ordering of events/failure times.

---

## Code Example

Below is the complete, self-contained code comparing Censored Gaussian NLL, Censored Quantile, and AFT losses on simulated right/left and interval-censored data.

```python
import argparse
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import AFTLoss, CensoredGaussianNLLLoss, CensoredQuantileLoss
from torchregress.metrics import censoring_rate, concordance_index, observed_mae

@dataclass(frozen=True)
class CensoredComparisonConfig:
    seed: int = 260227
    n_train: int = 768
    n_test: int = 256
    n_features: int = 6
    hidden: int = 32
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-2

class MLP(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

def simulate_data(cfg: CensoredComparisonConfig) -> dict[str, Tensor]:
    torch.manual_seed(cfg.seed)
    n = cfg.n_train + cfg.n_test
    x = torch.randn(n, cfg.n_features)

    # Latent failure time
    w = torch.tensor([0.9, -0.6, 0.35, 0.2, -0.25, 0.15])[: cfg.n_features]
    latent = x @ w + 0.3 * x[:, 0] * x[:, 1] + 0.4 * torch.randn(n)
    true_t = torch.exp(latent)

    # Set up censoring limits
    right_limit = torch.exp(0.6 * torch.randn(n) + 0.6)
    left_limit = torch.exp(0.6 * torch.randn(n) - 0.8)

    right_mask = true_t > right_limit
    left_mask = (~right_mask) & (true_t < left_limit)
    observed_mask = ~(right_mask | left_mask)

    observed_t = true_t.clone()
    observed_t[right_mask] = right_limit[right_mask]
    observed_t[left_mask] = left_limit[left_mask]

    censoring = torch.zeros(n, dtype=torch.int64)
    censoring[right_mask] = 1
    censoring[left_mask] = -1

    # Add a small subset of interval censoring
    interval_mask = observed_mask & (torch.rand(n) < 0.12)
    lower_bound = torch.full_like(true_t, float("nan"))
    upper_bound = torch.full_like(true_t, float("nan"))
    lower_bound[interval_mask] = true_t[interval_mask] * 0.85
    upper_bound[interval_mask] = true_t[interval_mask] * 1.15

    split = cfg.n_train
    return {
        "x_train": x[:split], "x_test": x[split:],
        "y_true_train": true_t[:split], "y_true_test": true_t[split:],
        "y_obs_train": observed_t[:split], "y_obs_test": observed_t[split:],
        "c_train": censoring[:split], "c_test": censoring[split:],
        "lb_train": lower_bound[:split], "ub_train": upper_bound[:split],
        "lb_test": lower_bound[split:], "ub_test": upper_bound[split:],
    }

def train_model(model: torch.nn.Module, loss_name: str, data: dict, cfg: CensoredComparisonConfig) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ds = TensorDataset(data["x_train"], data["y_obs_train"], data["c_train"], data["lb_train"], data["ub_train"])
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    loss_fn = CensoredGaussianNLLLoss() if loss_name == "gaussian" else (CensoredQuantileLoss(quantile=0.5) if loss_name == "quantile" else AFTLoss())
    model.train()
    for _ in range(cfg.epochs):
        for xb, yb, cb, lb, ub in loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            if loss_name == "gaussian" or loss_name == "aft":
                mean_loc = out[:, 0]
                log_scale = out[:, 1].clamp(-8.0, 8.0)
                loss = loss_fn((mean_loc, log_scale), yb, censoring=cb, lower_bound=lb, upper_bound=ub)
            else:
                loss = loss_fn(out[:, 0], yb, censoring=cb, lower_bound=lb, upper_bound=ub)
            loss.backward()
            optimizer.step()

def predict_times(model: torch.nn.Module, x: Tensor, loss_name: str) -> Tensor:
    model.eval()
    with torch.no_grad():
        out = model(x)
    if loss_name == "aft":
        return torch.exp(out[:, 0]).clamp(max=1e3)
    return out[:, 0].clamp(max=1e3)

def main() -> None:
    cfg = CensoredComparisonConfig()
    data = simulate_data(cfg)

    methods = [
        ("CensoredGaussianNLL", MLP(cfg.n_features, cfg.hidden, 2), "gaussian"),
        ("CensoredQuantile", MLP(cfg.n_features, cfg.hidden, 1), "quantile"),
        ("AFT", MLP(cfg.n_features, cfg.hidden, 2), "aft"),
    ]

    for name, model, loss_name in methods:
        train_model(model, loss_name, data, cfg)
        pred = predict_times(model, data["x_test"], loss_name)

        mae_true = torch.mean(torch.abs(pred - data["y_true_test"])).item()
        obs_mae = observed_mae(pred, data["y_obs_test"], data["c_test"]).item()
        c_idx = concordance_index(pred, data["y_obs_test"], data["c_test"]).item()

        print(f"Method: {name}")
        print(f"  True MAE: {mae_true:.4f}, Observed MAE: {obs_mae:.4f}, Concordance Index: {c_idx:.4f}")

if __name__ == "__main__":
    main()
```
