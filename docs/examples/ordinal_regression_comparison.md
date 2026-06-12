# Ordinal Regression Comparison

This guide compares multiple methods for modeling ordered-target data (where targets are discrete classes with a natural sequence or hierarchy).

| # | Reference |
|:-:|:----------|
| 1 | McCullagh, P. (1980). **Regression Models for Ordinal Data**. *Journal of the Royal Statistical Society: Series B*. |
| 2 | Cao, W., Mirjalili, V., & Raschka, S. (2020). **Consistent Rank Logits for Ordinal Regression with Convolutional Neural Networks**. *Pattern Recognition*. |

---

## Mathematical Formulations

In ordinal regression, target labels $y \in \{0, 1, \dots, K-1\}$ carry a natural order. Treating these labels as independent categories (using standard multi-class cross-entropy) discards the ordering information. Conversely, treating them as continuous real numbers (using standard MSE) forces the assumption that distance between adjacent classes is uniform.

### 1. Ordinal Cross Entropy

This baseline optimizes standard multi-class cross-entropy over $K$ independent class probabilities:

$$\mathcal{L}_{\text{CE}} = -\sum_{k=0}^{K-1} \mathbb{I}(y = k) \log p_k$$

It does not penalize a model more for making "farther" ordinal errors (e.g. predicting class $K-1$ when true label is $0$ vs predicting class $1$).

### 2. Cumulative Link Loss

Cumulative link models parameterize the cumulative probability of a sample belonging to a class less than or equal to $k$:

$$P(Y \le k \mid x) = \sigma(\theta_k - \eta(x))$$

where $\theta_0 < \theta_1 < \dots < \theta_{K-2}$ are learned cutpoints (thresholds), $\sigma$ is the sigmoid function, and $\eta(x)$ is the model's predicted scalar score. The individual class probabilities are computed as:

$$P(Y = k \mid x) = P(Y \le k \mid x) - P(Y \le k-1 \mid x)$$

with $P(Y \le -1) = 0$ and $P(Y \le K-1) = 1$. The model is trained by minimizing the negative log-likelihood of the observed classes.

### 3. Consistent Rank Logits (CORAL)

CORAL converts the ordinal classification problem into $K-1$ binary classification tasks. For target $y$, we define binary labels $y_k^* = \mathbb{I}(y > k)$ for $k \in \{0, \dots, K-2\}$. The model predicts $K-1$ logits sharing the same weights but using independent bias terms $b_k$:

$$\hat{p}_k = \sigma(\eta(x) + b_k)$$

The CORAL loss is the sum of binary cross-entropy losses:

$$\mathcal{L}_{\text{CORAL}} = -\sum_{k=0}^{K-2} \left[ y_k^* \log \hat{p}_k + (1 - y_k^*) \log (1 - \hat{p}_k) \right]$$

Sharing the weights guarantees **classifier consistency**: the predicted binary probabilities satisfy $\hat{p}_0 \ge \hat{p}_1 \ge \dots \ge \hat{p}_{K-2}$, preventing contradictory decisions.

---

## Task-First Context

*   **When to Use**: Use ordinal regression when predicting ordered survey answers, rating scales, cancer stages, or age groups.
*   **Comparison Metrics**: Always evaluate models using **Accuracy** (exact match rate), **Ordinal MAE** (average class-index distance), and **Quadratic Weighted Kappa (QWK)** (overall agreement penalizing larger class distance errors quadratically).

---

## Code Example

Below is the complete, self-contained code showing how to compare these ordinal losses on synthetic data.

```python
import argparse
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import CORALLoss, CumulativeLinkLoss, OrdinalCrossEntropyLoss
from torchregress.metrics import (
    mean_absolute_class_error,
    ordinal_accuracy,
    quadratic_weighted_kappa,
)
from torchregress.utils import ordinal_predict

@dataclass(frozen=True)
class OrdinalComparisonConfig:
    seed: int = 260227
    n_train: int = 512
    n_test: int = 256
    n_features: int = 6
    num_classes: int = 5
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

def make_data(cfg: OrdinalComparisonConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(cfg.seed)
    x = torch.randn(cfg.n_train + cfg.n_test, cfg.n_features)

    # Latent linear model with non-linear addition
    w = torch.tensor([0.9, -0.7, 0.4, 0.2, -0.3, 0.1])[: cfg.n_features]
    latent = x @ w + 0.35 * x[:, 0] * x[:, 1] - 0.2 * x[:, 2] ** 2 + 0.4 * torch.randn(x.shape[0])

    # Bin into classes using thresholds
    cutpoints = torch.tensor([-1.0, -0.25, 0.4, 1.1])
    y = torch.bucketize(latent, cutpoints).long()

    split = cfg.n_train
    return x[:split], y[:split], x[split:], y[split:]

def train_model(model: torch.nn.Module, loss_fn: torch.nn.Module, x_tr: Tensor, y_tr: Tensor, cfg: OrdinalComparisonConfig) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=cfg.batch_size, shuffle=True)
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

def evaluate(model: torch.nn.Module, x_te: Tensor, y_te: Tensor, encoding: str) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x_te)
    pred = ordinal_predict(logits, encoding=encoding)
    return {
        "Accuracy": float(ordinal_accuracy(pred, y_te, encoding="labels").item()),
        "OrdinalMAE": float(mean_absolute_class_error(pred, y_te, encoding="labels").item()),
        "QWK": float(quadratic_weighted_kappa(pred, y_te, encoding="labels").item()),
    }

def main() -> None:
    cfg = OrdinalComparisonConfig()
    x_tr, y_tr, x_te, y_te = make_data(cfg)

    # Compare baseline independent logits vs ordinal cumulative link and CORAL
    methods = [
        ("OrdinalCrossEntropy", MLP(cfg.n_features, cfg.hidden, cfg.num_classes), OrdinalCrossEntropyLoss(), "class_logits"),
        ("CumulativeLink", MLP(cfg.n_features, cfg.hidden, cfg.num_classes - 1), CumulativeLinkLoss(), "cumulative_logits"),
        ("CORAL", MLP(cfg.n_features, cfg.hidden, cfg.num_classes - 1), CORALLoss(), "cumulative_logits"),
    ]

    for name, model, loss_fn, encoding in methods:
        train_model(model, loss_fn, x_tr, y_tr, cfg)
        metrics = evaluate(model, x_te, y_te, encoding)
        print(f"Method: {name}")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}, OrdinalMAE: {metrics['OrdinalMAE']:.4f}, QWK: {metrics['QWK']:.4f}")

if __name__ == "__main__":
    main()
```
