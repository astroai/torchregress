# Representation-Level Imbalanced Regression: PRIME & ConR

> ← [Imbalanced targets](imbalanced.md) | [Rank-N-Contrast](imbalanced.md) → [Balanced MSE](imbalanced.md)

Long-tailed regression is not only a loss-weighting problem. Even with a perfectly reweighted output loss, a neural network's **hidden representation** can collapse in sparse target regions: the few metal-poor stars, young ages, or deep-depth pixels are absorbed into the feature statistics of the well-populated bulk. Output-level methods ([LDS](imbalanced.md), [Balanced MSE](imbalanced.md), density weighting) re-weight *errors*; representation-level methods shape *features*.

`torchregress` already provides two representation-level tools:

* **FDS** — [`FeatureDistributionSmoother`](imbalanced.md): calibrates feature first/second moments across target bins by kernel smoothing. This is a feature-statistics layer, and it is **not** duplicated here.
* **Rank-N-Contrast** — `RankNContrastLoss`: orders features by pairwise label distance.

This page adds two complementary losses, both implemented in `torchregress.losses.prime`:

* **PRIME** (`PrimeLoss`) — learnable **proxies** whose targets tile the label range uniformly; every mini-batch, even one containing only bulk targets, still pulls features toward the under-populated proxies.
* **ConR** (`ConR`) — a **contrastive regularizer** that penalises feature-space collapse: it selects label-dissimilar but prediction-similar pairs as negatives and pushes them apart proportionally to their label distance and inverse label density.

!!! abstract "The idea"
    Add a feature-space term to the task regression loss. PRIME gives every target region a persistent anchor; ConR gives minority anchors a strong push against the samples the model has collapsed onto them.

---

## 1. PRIME: proxy-based representation learning

PRIME (Lim et al., ICML 2025) introduces $C$ learnable proxies $\{(\mathbf z_j^p, y_j^p)\}_{j=1}^{C}$, with feature points $\mathbf z_j^p$ and targets $y_j^p$ distributed uniformly over the training target range $[y_{\min}, y_{\max}]$ (the $(C+1)$-quantiles for scalar targets). The proxy features are optimized jointly with the model, so the proxy set is a persistent, balanced reference frame for the feature space.

For a sample feature $\mathbf z$ with label $y$, PRIME aligns the sample to the proxies with a soft-label cross-entropy:

$$T_j = \frac{e^{-\tau_t\lvert y - y_j^p\rvert}}{\sum_k e^{-\tau_t\lvert y - y_k^p\rvert}}, \qquad
A_j = \frac{e^{-\tau_f\lVert \mathbf z - \mathbf z_j^p\rVert_2}}{\sum_k e^{-\tau_f\lVert \mathbf z - \mathbf z_k^p\rVert_2}},$$

$$\mathcal L_{\text{align}} = -\sum_{j=1}^{C} c_j\,T_j \log A_j,$$

where $c_j$ are optional class-balanced effective-number weights (`class_balanced=True`, paper Eq. 10, $\beta = 0.99$). As $\tau_t \to 0$ this becomes an InfoNCE-style alignment to the single nearest proxy; with a finite $\tau_t$ it is a smooth soft-label version that shares information between neighbouring target values.

To keep the proxies *ordered* and *separated*, the proxy geometry adds a t-SNE-style KL term and a spread term:

$$p_{ij} = \frac{e^{-\tau_t d_t(y_i^p, y_j^p)}}{\sum_{k\ne l} e^{-\tau_t d_t(y_k^p, y_l^p)}}, \qquad
q_{ij} = \frac{e^{-\tau_f d_f(\mathbf z_i^p, \mathbf z_j^p)}}{\sum_{k\ne l} e^{-\tau_f d_f(\mathbf z_k^p, \mathbf z_l^p)}},$$

$$\mathcal L_{\text{proxy}} = \sum_{i\ne j}\left[p_{ij}\log\frac{p_{ij}}{q_{ij}} - \alpha\, d_t(y_i^p,y_j^p)\,\bigl(1-\cos(\mathbf z_i^p,\mathbf z_j^p)\bigr)^2\right].$$

The total regularizer is $\lambda_p\mathcal L_{\text{proxy}} + \lambda_a\,\overline{\mathcal L_{\text{align}}}$, added to the task loss. Because the proxies are parameters, the **whole** target range receives gradient on every step — a mini-batch that happens to contain no tail objects still updates the tail proxies, which is exactly what [Rank-N-Contrast](imbalanced.md) and batch-contrastive methods lack.

```python
import torch
import torch.nn as nn
from torchregress.losses import PrimeLoss

torch.manual_seed(0)
n = 2000
# Long-tailed target: many objects near 0, few near the edges of [-1, 1].
y = torch.cat([0.15 * torch.randn(1600), torch.rand(400) * 2 - 1]).clamp(-1.0, 1.0)
x = torch.stack([y, torch.sin(3 * y)], dim=1) + 0.05 * torch.randn(n, 2)

feature_dim = 32
encoder = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, feature_dim))
head = nn.Linear(feature_dim, 1)
prime = PrimeLoss(
    feature_dim=feature_dim, n_proxies=50, label_range=(-1.0, 1.0), class_balanced=True
)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(head.parameters()) + list(prime.parameters()),
    lr=1e-3,
)
mse = nn.MSELoss()
for _ in range(50):
    z = encoder(x)
    pred = head(z).squeeze(-1)
    loss = mse(pred, y) + 0.5 * prime(z, y)  # both terms, one backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("tail loss:", float((head(encoder(x[y > 0.8])).squeeze(-1) - y[y > 0.8]).abs().mean()))
```

Call `prime.fit(y_train)` instead of `label_range` to estimate the proxy targets and class-balanced weights from the training labels.

!!! tip "Why proxies help the tail"
    With batch-contrastive losses, tail gradient is proportional to the (rare) number of tail samples in the batch. PRIME's proxies are present in every step, so the tail representation direction is always supervised, and the uniform proxy spacing prevents the bulk from occupying the whole feature manifold.

---

## 2. ConR: contrastive regularizer for collapse

ConR (Keramati et al., ICLR 2024) selects pairs using **both** label and prediction similarity. For anchor $i$ with feature $\mathbf z_i$, label $y_i$ and prediction $p_i$:

$$\text{pos}_i = \{j \ne i:\ \lvert y_i - y_j\rvert \le \omega\}, \qquad
\text{neg}_i = \{j \ne i:\ \lvert y_i - y_j\rvert > \omega,\ \lvert p_i - p_j\rvert \le \omega\},$$

$$s_{ij} = \frac{\hat{\mathbf z}_i^\top \hat{\mathbf z}_j}{\tau}, \qquad
S_{ij} = \eta_i\, e\,\lvert y_i-y_j\rvert, \qquad \hat{\mathbf z} = \frac{\mathbf z}{\lVert \mathbf z\rVert},$$

$$\mathcal L_i^{\text{ConR}} = -\frac{1}{\lvert\text{pos}_i\rvert}\sum_{j\in\text{pos}_i}\log\frac{e^{s_{ij}}}{\sum_{k\in\text{pos}_i} e^{s_{ik}} + \sum_{q\in\text{neg}_i} S_{iq}\,e^{s_{iq}}},$$

with $\mathcal L_i=0$ when the anchor has no positive or no negative partner. The intuition: a *negative* pair is a pair with very different labels that the model has nevertheless mapped to the same region of feature space — that is precisely the collapse that hurts minority targets. The pushing weight $S_{iq}$ grows with the label disagreement and with the anchor's inverse-label-density weight $\eta_i$ (`ConR.fit(train_targets)`), so minority anchors push harder.

```python
from torchregress.losses import ConR

# Reuse the long-tailed data and encoder defined above.
conr = ConR(label_threshold=0.2, temperature=0.07, push_scale=0.2)
conr.fit(y)   # inverse-label-density pushing weights from the training targets
for _ in range(50):
    z = encoder(x)
    pred = head(z).squeeze(-1)
    loss = mse(pred, y) + 4.0 * conr(z, y, prediction=pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

!!! warning "ConR needs the predictions"
    Negative pairs are defined by *prediction* similarity, so `forward` requires the model's regression output (`prediction=`). That is also why `ConR` is not registered with the generic loss registry: it cannot be evaluated from features and labels alone.

---

## Comparison: which representation method?

| Method | Mechanism | Persistent tail signal? | Extra parameters | Batch-composition sensitive? |
|:-------|:-----------|:------------------------|:-----------------|:-----------------------------|
| [LDS](imbalanced.md) (`LDSLoss`) | Kernel-smoothed label density reweighting | No (loss-level) | No | No |
| [Balanced MSE (BMC)](balanced_mse_bmc.md) (`BalancedMSEBMC`) | Noise-adaptive batch posterior balancing | Yes (batch-coupled) | No | Yes |
| `RankSimLoss` | Matches label-distance and feature-similarity rankings | Only via batch pairs | No | Yes ($O(B^2)$) |
| [FDS](imbalanced.md) (`FeatureDistributionSmoother`) | Kernels smooth feature mean/variance across bins | Yes (EMA per bin) | No | No |
| `RankNContrastLoss` | Orders features by pairwise label distance | Only via batch pairs | No | Yes ($O(B^2)$, no tail samples → no tail gradient) |
| **PRIME** (`PrimeLoss`) | Uniform learnable proxies + geometry/alignment | Yes (proxies always present) | $C\times D$ proxies | No |
| **ConR** (`ConR`) | Contrastive collapse penalty with label-density pushing | Only via batch pairs | No | Yes ($O(B^2)$) |

!!! tip "Combining them"
    The families are complementary: FDS calibrates feature *statistics*, PRIME shapes the *global* feature geometry, and ConR penalises *local* collapses of the current batch. A practical recipe is `task_loss + FDS layer + λ1 * PRIME + λ2 * ConR`, but tune one term at a time and evaluate tail-slice metrics (few-shot MAE/RMSE, worst-bin error) rather than overall MSE.

---

## Caveats

!!! warning "Assumptions and failure modes"
    * **Scalar labels**: PRIME's proxy targets are uniformly spaced scalar values; multi-dimensional targets need an ordering or clustering design (`label_range` is scalar). ConR accepts multi-dimensional labels via L2 label distance.
    * **Proxy range**: proxies outside the training label range are unconstrained; the paper's uniform design assumes the target interval is meaningful. `PrimeLoss` requires either `fit(train_targets)` or an explicit `label_range`, and will not silently guess.
    * **Prediction required for ConR**: passing only features and labels raises an error; feed the model's regression output.
    * **Batch contrastive cost**: ConR is $O(B^2)$ per step and remains batch-composition sensitive; PRIME's proxy term is $O(BC)$ and unaffected by batch composition.
    * **Tuning**: `label_threshold`/`prediction_threshold` (ConR) and `temperature`/`proxy_temperature` (PRIME) control how many pairs/proxies are active. Thresholds that are too small leave no pairs (zero gradient); too large treats dissimilar data as positives. Check the returned loss is non-zero on a real batch before training.
    * **No free calibration**: representation regularizers improve tail *features*, not the calibration of predictive intervals. Re-run post-hoc calibration ([`VarianceTemperatureScaler`](../methods/calibration.md)) after switching them on.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Yang, Y., Zha, K., Chen, Y.-C., Wang, H., & Katabi, D. (2021). Delving into deep imbalanced regression. *ICML 2021*. [arXiv:2102.09554](https://arxiv.org/abs/2102.09554) |
| 2 | Lim, J., Lee, S., Um, D., Park, S.-U., & Shin, J. (2025). PRIME: Deep imbalanced regression with proxies. *ICML 2025*. [Poster](https://icml.cc/virtual/2025/poster/44388) |
| 3 | Keramati, M., Meng, L., & Evans, R. D. (2024). ConR: Contrastive regularizer for deep imbalanced regression. *ICLR 2024*. [arXiv:2309.06651](https://arxiv.org/abs/2309.06651) |
| 4 | Zha, K., Cao, P., Son, J., Yang, Y., & Katabi, D. (2023). Rank-N-Contrast: Learning continuous representations for regression. *NeurIPS 2023*. |
| 5 | Ren, J., Zhang, M., Yu, C., & Liu, Z. (2022). Balanced MSE for imbalanced visual regression. *CVPR 2022*. [arXiv:2203.16427](https://arxiv.org/abs/2203.16427) |

## See also

* [Imbalanced targets](imbalanced.md) — LDS, FDS, Balanced MSE, Focal-R.
* [Rank-N-Contrast](imbalanced.md) — pairwise label-order regularizer.
* [Post-hoc calibration](../methods/calibration.md) — re-calibrate after changing the objective.
* [Imbalanced / rare-target regression](../guide/method-selection.md) — task-level selection matrix.
