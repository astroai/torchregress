# Super-Level-Set (SLS) Regression

> ← [Transform Losses](transforms.md) | [Loss Functions](index.md) →

SLS regression directly estimates **minimum-volume prediction regions** with conditional coverage guarantees by optimizing a learned level-set boundary under a volume penalty.

!!! tip "API references"
    Training loss: [`SLSLoss`](../api/losses.md). Post-hoc coverage wrapper: [`SLSConformal`](../api/conformal.md).

!!! abstract "Key insight"
    SLS frames conditional coverage as a **volume minimization** problem: find the smallest prediction set $C(x)$ such that $P(Y \in C(x) \mid X = x) \geq \tau$. The boundary of $C(x)$ is parameterized by a Mahalanobis distance in a learned feature space, with a volume-preserving flow ensuring the learned metric is invertible.

---

## When to Use

!!! success "SLS is a good fit when"
    - You need **conditional coverage** guarantees, not just marginal
    - Target distributions are **multivariate** ($d \geq 2$) and possibly multimodal
    - You want **minimum-volume** prediction regions (not just intervals)
    - You're willing to train a specialized architecture end-to-end

!!! warning "SLS is not ideal when"
    - You only need scalar intervals — quantile regression or conformal prediction is simpler
    - Latency is critical — SLS uses a flow-based frontier with multiple transformations
    - You have very high target dimensions ($d > 10$) — the volume proxy and flow transformations scale poorly

---

## Mathematical Background

### The SLS Objective

Given a target coverage level $\tau \in (0, 1)$, SLS learns a **frontier function** $G(y \mid x)$ that defines the prediction region:

$$C(x) = \{ y : G(y \mid x) \leq t(x) \}$$

where $t(x)$ is a learned threshold. The SLS loss simultaneously optimizes the frontier and the threshold to achieve:
- **Coverage**: $P(Y \in C(x) \mid X = x) \approx \tau$
- **Minimal volume**: The region $C(x)$ is as small as possible while maintaining coverage

### Mahalanobis Frontier

The core frontier parameterizes a **Mahalanobis distance** in a warped space:

$$G(y \mid x) = \| L(x) \cdot (\varphi(y; x) - \mu(x)) \|_2^2$$

where:
- $\varphi(y; x)$ is a **volume-preserving flow** that maps $y$ to a warped representation (distinct from the regression function $f(x)$)
- $\mu(x)$ is a learned center
- $L(x)$ is a learned Cholesky factor (full or low-rank) defining the Mahalanobis metric
- $\det L(x)$ contributes a log-volume penalty term

The volume-preserving flow ensures $\det \frac{\partial \varphi}{\partial y} = 1$, so the log-volume penalty depends entirely on $\log|\det L(x)|$ (computed as $\sum_i \log L_{ii}$, i.e. $\mathcal{O}(d)$ once $L$ is in hand).

### Union Frontier (Multimodal)

For $K > 1$ components, `UnionFrontier` models a **union of Mahalanobis regions**:

$$C(x) = \bigcup_{k=1}^K \{ y : G_k(y \mid x) \leq t_k(x) \}$$

with learned mixture weights that assign each point to the most appropriate component via a softmax over the Mahalanobis scores.

---

## Architecture Components

### `MahalanobisFrontier`

Single-component frontier with volume-preserving flow and Mahalanobis metric.

| Mode | Covariance parameterization | Parameter count | Best for |
|:-----|:---------------------------|:---------------|:---------|
| `"full"` | Full Cholesky $L \in \mathbb{R}^{d \times d}$ | $\mathcal{O}(d^2)$ | $d \leq 5$ |
| `"low_rank"` | $D \in \mathbb{R}^d$, $V \in \mathbb{R}^{d \times r}$ | $\mathcal{O}(d \cdot r)$ | $d \geq 5$ |

### `VolumePreservingFlow`

A conditional normalizing flow with **Jacobian determinant = 1** by construction. Uses translation-only coupling layers (no scaling), ensuring the volume in the warped space equals the volume in the original space. The flow is invertible and can be used for sampling.

### `UnionFrontier`

Mixture-of-experts frontier: $K$ independent `MahalanobisFrontier` components whose predictions are combined via a learned softmax over Mahalanobis scores. The temperature parameter $\beta$ is annealed during training to sharpen the assignment.

### `QuantileNetwork`

A small MLP that predicts three learned **score thresholds** $[t_{\text{low}}, t_{\text{mid}}, t_{\text{high}}]$ on the Mahalanobis frontier $G$ representing the coverage window around $\tau$. These are not quantile probability levels — they are thresholds in the score space of $G$. The thresholds are constrained to be ordered via sorting.

---

## Usage

### Basic Setup

```python
import torch
import torch.nn as nn
from torchregress.losses import SLSLoss

# Model backbone: maps input x → context vector
class ContextModel(nn.Module):
    def __init__(self, in_dim, context_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, context_dim),
        )
    def forward(self, x):
        return self.net(x)

model = ContextModel(in_dim=10, context_dim=64)

# SLS loss for 2D targets with a single Mahalanobis component
loss_fn = SLSLoss(
    d=2,
    context_dim=64,
    K=1,                # Single-component frontier
    mode="full",        # Full covariance
    tau=0.9,            # Target coverage
    warmup_steps=500,   # Steps before coverage penalty activates
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `d` | `int` | — | Target dimensionality |
| `context_dim` | `int` | — | Dimension of conditioning vector (model output) |
| `K` | `int` | `1` | Number of mixture components (1 = single, > 1 = union) |
| `mode` | `str` | `"full"` | Covariance mode: `"full"` or `"low_rank"` |
| `rank` | `int` or `None` | `None` | Rank for low-rank mode (default: $\lceil\sqrt{d}\rceil$) |
| `hidden_dim` | `int` | `64` | Hidden dimension for flow and frontier networks |
| `n_transforms` | `int` | `4` | Number of flow transformation layers |
| `tau` | `float` | `0.9` | Target coverage level |
| `warmup_steps` | `int` | `500` | Steps before coverage penalty activates |
| `error_init` | `float` | `0.4` | Initial coverage window half-width |
| `error_min` | `float` | `0.05` | Minimum coverage window half-width |

### Training Loop

```python
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(loss_fn.parameters()),
    lr=1e-3,
)

# Note: y_pred is the context vector from the model
# The loss internally handles the frontier and quantile networks
for epoch in range(200):
    for x, y in train_loader:
        context = model(x)           # [batch, context_dim]
        loss = loss_fn(context, y)   # SLS composite loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Warmup Phase

During the first `warmup_steps`, the **frontier volume penalty** ramps up via a sigmoidal schedule while the **quantile pinball term** is active throughout training. After warmup, the coverage frontier term reaches full strength and the composite loss balances volume vs. coverage.

---

## Inference

### Prediction Regions

```python
with torch.no_grad():
    context = model(x_test)
    # Get Mahalanobis scores for a grid of candidate y values
    G, log_det = loss_fn.frontier(y_candidates, context)
    # Points with G below the quantile threshold are in the prediction set
    quantiles = loss_fn.quantile_net(context)
    threshold = quantiles[..., 1]  # middle quantile
    in_region = G <= threshold
```

### Sampling from the Learned Frontier

The volume-preserving flow is invertible, enabling sampling:

```python
# Sample from the base distribution and invert through the flow
z = torch.randn(1000, d)  # base samples
with torch.no_grad():
    mu, L_params = loss_fn.frontier._get_params(context)
    y_samples = loss_fn.frontier.flow.inverse(z + mu, context)
```

---

## When to Use Multimodal ($K > 1$)

!!! tip "Start with $K = 1$"
    A single Mahalanobis component is sufficient for unimodal or mildly non-Gaussian target distributions. Only increase $K$ when:
    - The target distribution shows clear multimodal structure (validated via KDE or clustering)
    - Single-component prediction regions are excessively large because they must cover gaps between modes
    - Validation coverage or volume metrics improve meaningfully with $K > 1$

!!! warning "Multimodal tradeoffs"
    - Each additional component adds a full `MahalanobisFrontier` (flow + Cholesky + center network)
    - The softmax temperature $\beta$ must be carefully annealed — too fast and components collapse; too slow and the assignment stays diffuse
    - $K > 3$ rarely improves performance and can cause component collapse

---

## Comparison with Other Methods

| Method | Coverage Type | Region Shape | Multivariate | Multimodal | Volume Optimal |
|:-------|:------------|:-------------|:-----------:|:----------:|:-------------:|
| **SLS** | Conditional (learned) | Mahalanobis / union | ✅ | ✅ ($K > 1$) | ✅ |
| **CQR** | Marginal (conformal) | Quantile band | ❌ | ❌ | ❌ |
| **CTI** | Marginal (conformal) | Density level-set | ❌ | ✅ | ✅ (Neyman–Pearson) |
| **MDN** | Distributional | Gaussian mixture | ✅ (diag/full) | ✅ | ❌ |
| **Conformal + MDN** | Marginal (conformal) | MDN level-set | ✅ | ✅ | ✅ |

---

## Limitations

!!! warning "Current constraints"
- **Target dimensionality**: The Mahalanobis frontier with full covariance has $\mathcal{O}(d^2)$ parameters (lower-triangular $L$) and $\mathcal{O}(d^2)$ forward-pass cost; use `mode="low_rank"` for $d > 5$
- **Warmup sensitivity**: The warmup phase length is critical — too short and the frontier never converges; too long and the coverage penalty activates before the frontier is stable. The sigmoidal schedule parameters ($k=0.005$, $t_0=1000$) are function-level defaults and **not exposed as constructor parameters**.
- **Coverage window tuning**: The `error_init` → `error_min` annealing schedule (sigmoidal) may need adjustment for different target distributions
    - **Training stability**: SLS optimizes frontier parameters, quantile thresholds, and (for $K > 1$) mixture weights simultaneously — gradient conflicts can occur
    - **No conformal guarantee**: Unlike CQR or CTI, SLS provides no finite-sample coverage guarantee. The coverage is learned, not calibrated. Validate coverage empirically on held-out data

---

## Recommendations

- **Start with `MahalanobisFrontier`** for unimodal problems. Use `UnionFrontier` only when multimodality is confirmed (e.g., multiple clusters in the latent space).
- **Tune $d$ on validation**: Grid search over $d \in \{2, 4, 8, 16\}$. Monitor frontier volume and coverage on held-out data.
- **Set $\tau$ near the target coverage**: $\tau$ controls the super-level set threshold. A typical starting value is $\tau = 0.9$ for 90% coverage.
- **Combine with conformal calibration**: For guaranteed coverage, wrap SLS predictions with [SLSConformal](../methods/conformal/predictors.md).

## Next steps

- [Normalizing flows](nflows.md) — building-block architecture for SLS frontiers
- [MDN losses](mdn.md) — multimodal alternatives without the volume-optimal guarantee
- [Conformal prediction](../methods/conformal/index.md) — post-hoc coverage guarantees (CQR, CTI, SLSConformal)

---

## References

| # | Reference |
|:-:|:----------|
| 1 | S. Braun, M.I. Jordan, F. Bach. ["Super-Level-Set Regression: Conditional Quantiles via Volume Minimization."](https://arxiv.org/abs/2605.06210) *arXiv:2605.06210*, **2026**. |
