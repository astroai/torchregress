# Ordinal Regression Losses

Ordinal regression is for targets with a **natural ordering** where the distances between levels are unknown or non-uniform.  Unlike classification, ordinal losses exploit the fact that **nearby predictions are better than distant ones**.

---

## When to Use Ordinal Regression

!!! example "Common scenarios"

    - **Quality ratings** — 1–5 stars (predicting 4 when truth is 5 is better than 2)
    - **Survey responses** — strongly disagree → strongly agree
    - **Disease severity** — mild / moderate / severe / critical
    - **Binned continuous targets** — age groups, magnitude bins, discretised redshifts

---

## Mathematical Background

### Cumulative Link Model

The cumulative model defines ordered probabilities through **shared thresholds** $\theta_1 < \theta_2 < \cdots < \theta_{K-1}$:

$$P(Y \leq k \mid x) = g\!\bigl(\theta_k - f(x)\bigr), \qquad k = 1, \ldots, K-1$$

where $g$ is a monotone link function (logistic, probit) and $f(x)$ is a **scalar** model output.

The probability of each class is:

$$P(Y = k \mid x) = P(Y \leq k) - P(Y \leq k-1)$$

!!! info "Why shared weights?"
    The model uses a **single scalar** output $f(x)$ plus thresholds — far fewer parameters than $K$ independent logits.  The thresholds enforce the ordering constraint automatically.

### CORAL Objective

**Consistent Rank Logits** (Cao et al., 2020) reformulates ordinal regression as $K-1$ binary tasks with **shared weights** but independent biases:

$$P(Y > k \mid x) = \sigma\!\bigl(f(x) - b_k\bigr)$$

The loss is the sum of binary cross-entropies:

$$\mathcal{L}_{\text{CORAL}} = -\sum_{k=1}^{K-1} \Bigl[\mathbf{1}_{y > k}\,\log\sigma(f - b_k) + \mathbf{1}_{y \leq k}\,\log(1 - \sigma(f - b_k))\Bigr]$$

---

## Available Losses

### OrdinalCrossEntropyLoss

Standard cross-entropy baseline.  Treats each level as an independent class — **does not** exploit ordering.

```python
import torch
from torchregress.losses import OrdinalCrossEntropyLoss

num_classes = 5
logits = torch.randn(32, num_classes)  # K logits
labels = torch.randint(0, num_classes, (32,))

loss_fn = OrdinalCrossEntropyLoss()
loss = loss_fn(logits, labels)
```

!!! tip "When to use"
    As a **baseline** to compare against ordering-aware losses.

### CumulativeLinkLoss

Cumulative-threshold model with $K-1$ logits.

```python
from torchregress.losses import CumulativeLinkLoss

logits = torch.randn(32, num_classes - 1)  # K-1 outputs
loss_fn = CumulativeLinkLoss()
loss = loss_fn(logits, labels)
```

!!! tip "When to use"
    When you want a **principled ordinal model** with the fewest parameters.

### CORALLoss

CORAL-style consistent rank logits.

```python
from torchregress.losses import CORALLoss

logits = torch.randn(32, num_classes - 1)  # K-1 binary logits
loss_fn = CORALLoss()
loss = loss_fn(logits, labels)
```

!!! tip "When to use"
    Best **empirical performance** on many benchmarks (Cao et al., 2020).

---

## Method Comparison

| | OrdinalCE | CumulativeLinkLoss | CORALLoss |
|:-|:---------:|:------------------:|:---------:|
| Exploits ordering | ❌ | ✅ | ✅ |
| Parameters | $K$ logits | $K-1$ logits + thresholds | $K-1$ biases (shared weights) |
| Guarantees monotonicity | ❌ | ✅ (via thresholds) | ✅ (by construction) |
| Typical performance | Baseline | Good | Best |

---

## Complete Example: Product Ratings

```python
import torch
import torch.nn as nn
from torchregress.losses import CORALLoss

K = 5  # 1-5 star ratings

class RatingModel(nn.Module):
    def __init__(self, in_dim, K):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.fc = nn.Linear(32, 1, bias=False)  # shared weights
        self.biases = nn.Parameter(torch.zeros(K - 1))  # K-1 thresholds

    def forward(self, x):
        h = self.fc(self.feature(x))        # scalar per sample
        return h - self.biases.unsqueeze(0)  # [batch, K-1]

model = RatingModel(in_dim=50, K=K)
loss_fn = CORALLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
for x, y in train_loader:
    logits = model(x)       # [batch, K-1]
    loss = loss_fn(logits, y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# Decoding predictions
with torch.no_grad():
    logits = model(x_test)
    probs = torch.sigmoid(logits)               # P(Y > k)
    predicted_class = probs.sum(dim=1).round()   # expected rank
```

---

## Practical Tips

!!! tip "Decoding CORAL predictions"
    Sum the sigmoid-activated outputs: $\hat{y} = \sum_{k=1}^{K-1} \sigma(f(x) - b_k)$.  Round to nearest integer for discrete predictions.

!!! tip "Choosing between losses"
    - **Start with CORALLoss** — best empirical performance
    - Use **CumulativeLinkLoss** for interpretable threshold parameters
    - Use **OrdinalCrossEntropyLoss** only as a baseline

!!! warning "Don't use standard classification metrics"
    Ordinal accuracy ignores ordering.  Use **MAE**, **Quadratic Weighted Kappa**, or **Spearman $\rho$** instead.

---

## Related

- [Ordinal Metrics](../metrics/ordinal.md) — MAE, Quadratic Weighted Kappa, Spearman $\rho$
- [Ordinal Regression Comparison](../examples/ordinal_regression_comparison.md) — side-by-side benchmark
- [Ordinal Regression Comparison (Real Data)](../examples/ordinal_regression_realdata_comparison.md) — real-data benchmark

---

## References

| # | Reference |
|:-:|:----------|
| 1 | P. McCullagh. "Regression Models for Ordinal Data." *JRSS B*, 42(2):109–142, **1980**. |
| 2 | W. Cao, V. Mirjalili, S. Raschka. "Rank Consistent Ordinal Regression for Neural Networks." *Pattern Recognition Letters*, 140:325–331, **2020**. |
| 3 | E. Frank, M. Hall. "A Simple Approach to Ordinal Classification." *ECML*, **2001**. |
