# OOD and Selective Prediction Comparison

This guide explains how to use predictive uncertainty to perform selective prediction (abstaining on high-risk inputs) and detect Out-of-Distribution (OOD) covariate shifts.

→ API: [`uncertainty_decomposition`](../api/metrics.md#uncertainty_decomposition), [`mahalanobis_distance`](../api/metrics.md#mahalanobis_distance), [`RiskCoverageCurve`](../api/metrics.md#riskcoveragecurve). Guide: [OOD metrics](../metrics/ood.md).

| # | Reference |
|:-:|:----------|
| 1 | Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). [**Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles**](https://arxiv.org/abs/1612.01474). *NeurIPS*. |
| 2 | Geifman, Y., & El-Yaniv, R. (2017). [**Selective Classification for Deep Neural Networks**](https://arxiv.org/abs/1705.08500). *NeurIPS*. |

---

## Mathematical Formulations

### Ensemble Uncertainty Decomposition

For a heteroscedastic ensemble of $M$ models, each predicting a mean $\mu_m(x)$ and variance $\sigma^2_m(x)$, the total predictive variance is decomposed into epistemic (model disagreement) and aleatoric (data noise) uncertainty:

$$\sigma^2_{\text{total}}(x) = \sigma^2_{\text{epistemic}}(x) + \sigma^2_{\text{aleatoric}}(x)$$

The two components are:

- **Epistemic variance** (model uncertainty):
    $$\sigma^2_{\text{epistemic}}(x) = \frac{1}{M} \sum_{m=1}^M (\mu_m(x) - \bar{\mu}(x))^2 \quad \text{where} \quad \bar{\mu}(x) = \frac{1}{M} \sum_{m=1}^M \mu_m(x)$$
- **Aleatoric variance** (inherent noise):
    $$\sigma^2_{\text{aleatoric}}(x) = \frac{1}{M} \sum_{m=1}^M \sigma^2_m(x)$$

### Selective Prediction (Risk-Coverage)

Selective prediction allows a model to abstain from predicting if its uncertainty score $U(x)$ exceeds a threshold.
1.  **Risk-Coverage Curve (RCC)**: Plots the prediction error (risk) on the subset of samples that are kept, sorted in ascending order of uncertainty, as the coverage (fraction of samples kept) varies from 0 to 1.
2.  **Area Under the Risk-Coverage Curve (AURC)**: Measures uncertainty quality:
    $$\text{AURC} = \int_{0}^1 \text{Risk}(\kappa) d\kappa$$
    where $\kappa$ is the coverage. A smaller AURC indicates that the uncertainty estimator successfully identifies and rejects high-error predictions.

### Out-of-Distribution (OOD) Scoring

OOD metrics identify test points $x_{\text{test}}$ that lie outside the training covariate distribution.
1.  **Mahalanobis Distance**:
    Estimates the distance of features $x$ from the training feature distribution mean $\mu_{\text{train}}$ and covariance $\Sigma_{\text{train}}$:
    $$d_M(x) = \sqrt{(x - \mu_{\text{train}})^\top \Sigma_{\text{train}}^{-1} (x - \mu_{\text{train}})}$$
2.  **Typicality Score**:
    Measures the typicality of target $y$ under the model's predicted distribution parameters.
3.  **Kernel Density Score**:
    Calculates the density of the test point under a kernel density estimate of the training features.

---

## Task-First Context

- **When to Use**: Use selective prediction when your application can defer to a human expert or fallback policy for high-uncertainty samples. Use OOD metrics to trigger safety alerts when the input features shift.
- **Comparison Notes**: Ensembles (e.g. Deep Ensembles) typically provide the most reliable epistemic uncertainty for OOD detection compared to single-model MC dropout proxies.

---

## Code Example

Below is the complete, self-contained code showing how to compare selective prediction and OOD signals across several uncertainty methods.

```python
import copy
from dataclasses import dataclass
import torch
import torch.nn as nn
from torchregress.ensemble import SWAG, BayesianNeuralNetwork
from torchregress.metrics import (
    RejectionPolicy,
    ensemble_mean,
    ensemble_variance_decomposition,
    ood_metrics_report,
    risk_coverage_curve,
)

@dataclass(frozen=True)
class OODConfig:
    seed: int = 123
    n_train: int = 160
    n_cal: int = 48
    n_id_test: int = 80
    n_ood_test: int = 80
    epochs: int = 25
    ensemble_size: int = 3
    mc_samples: int = 16
    lr: float = 0.01
    swag_samples: int = 12
    swag_scale: float = 0.5
    bnn_samples: int = 16
    bnn_beta: float = 0.2
    conformal_alpha: float = 0.1

class PointMLP(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train_point_model(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, epochs: int, lr: float) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()
    model.eval()
    return model

def main() -> None:
    cfg = OODConfig()
    torch.manual_seed(cfg.seed)

    # Generate synthetic train/ID-test/OOD-test splits
    x_train = torch.linspace(-3.0, 3.0, cfg.n_train).unsqueeze(1)
    y_train = torch.sin(x_train) + 0.1 * torch.randn_like(x_train)
    x_test = torch.linspace(-3.5, 3.5, cfg.n_id_test).unsqueeze(1)
    y_test = torch.sin(x_test) + 0.12 * torch.randn_like(x_test)

    # Train a deep ensemble
    ensemble_models = []
    for i in range(cfg.ensemble_size):
        torch.manual_seed(cfg.seed + i)
        model = PointMLP(dropout=0.0)
        train_point_model(model, x_train, y_train, epochs=cfg.epochs, lr=cfg.lr)
        ensemble_models.append(model)

    # Evaluate deep ensemble predictions and uncertainty
    preds_id = torch.stack([m(x_test) for m in ensemble_models])
    mean_id = ensemble_mean(preds_id)
    std_id = preds_id.std(dim=0).clamp(min=1e-8)

    # Compute Selective Prediction (Risk-Coverage AURC)
    rcc = risk_coverage_curve(mean_id, y_test, std_id.view(-1), n_points=25)
    policy = RejectionPolicy(fraction=0.2)
    policy.update(mean_id, y_test, std_id.view(-1))
    policy_res = policy.compute()

    # Compute Mahalanobis Distance for OOD Detection
    train_mean = x_train.mean(dim=0)
    x_centered = x_train - train_mean
    cov = (x_centered.T @ x_centered) / max(1, (x_train.shape[0] - 1))

    ood_report = ood_metrics_report(
        model_output=(mean_id, std_id.pow(2)),
        x_test=x_test,
        x_reference=x_train,
        mean=train_mean,
        cov=cov,
    )

    print("Selective Prediction Evaluation:")
    print(f"  AURC: {rcc['aurc'].item():.6f} (lower is better)")
    print(f"  Risk at 20% rejection: {policy_res['mean_risk'].item():.6f}")
    print(f"  Coverage: {policy_res['coverage'].item():.2f}")
    print("OOD Metrics:")
    print(f"  Mahalanobis Distance (mean): {ood_report['mahalanobis_distance'].mean().item():.4f}")
    print(f"  Typicality Score (mean): {ood_report['typicality_score'].mean().item():.4f}")

if __name__ == "__main__":
    main()
```
