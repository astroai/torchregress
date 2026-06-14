# Multi-Target Regression

Predicting multiple continuous outcomes $\mathbf{y} \in \mathbb{R}^K$ jointly is a common requirement in physical sciences, finance, and engineering. Instead of treating each target dimension independently (which discards correlation structure), **torchregress** provides native support for **joint multi-target regression**, **correlated uncertainty estimation**, and **multivariate calibration**.

---

## 1. Why Joint Multi-Target Regression?

Standard regression models predict each target independently. However, predicting targets jointly offers key advantages:
1. **Target Correlation**: The noise in one target may be strongly correlated with the noise in another (e.g., coordinates in physical space, or different wavelengths in astronomy). Capturing this correlation is essential for valid probabilistic forecasting.
2. **Aleatoric Uncertainty Covariance**: Jointly modeling the covariance matrix $\mathbf{\Sigma}(x)$ allows for estimating directed confidence ellipsoids rather than coordinate-wise boxes.
3. **Optimisation Efficiency**: Sharing feature-extraction backbones across multiple targets is cheaper and less prone to overfitting than training independent networks.

---

## 2. Multi-Target Probabilistic Modeling

**torchregress** provides several ways to model multi-target outcomes, balancing output capacity, shape flexibility, and computational complexity.

### A. Parametric Gaussian Models

For Gaussian noise, we parameterize the predictive distribution as $\mathcal{N}(\boldsymbol{\mu}(x), \mathbf{\Sigma}(x))$.

#### 1. Full Covariance: `MultivariateGaussianLoss`
Models the full $K \times K$ covariance matrix $\mathbf{\Sigma}(x)$ directly.

- **Outputs**: Mean vector $\boldsymbol{\mu} \in \mathbb{R}^K$ and the full covariance matrix $\mathbf{\Sigma} \in \mathbb{R}^{K \times K}$.
- **Likelihood Formulation**:
  $$\mathcal{L}_{\text{MVN}}(\boldsymbol{\mu}, \mathbf{\Sigma}; \mathbf{y}) = \frac{1}{2} \log \lvert\mathbf{\Sigma}\rvert + \frac{1}{2} (\mathbf{y} - \boldsymbol{\mu})^\top \mathbf{\Sigma}^{-1} (\mathbf{y} - \boldsymbol{\mu}) + \frac{K}{2}\log(2\pi)$$
- **Limitations**: Inversion and determinant costs scale as $\mathcal{O}(K^3)$ with target dimension, limiting usability for large $K$.

#### 2. Low-Rank Covariance: `LowRankGaussianLoss`
Parameterizes covariance as a low-rank matrix plus a diagonal correction, $\mathbf{\Sigma} = \mathbf{U}\mathbf{U}^\top + \text{diag}(\mathbf{d})$, where $\mathbf{U} \in \mathbb{R}^{K \times R}$ ($R \ll K$) and $\mathbf{d} \in \mathbb{R}^K_{>0}$.
- **Outputs**: Mean $\boldsymbol{\mu}$, factor matrix $\mathbf{U}$, diagonal variance $\mathbf{d}$.
- **Advantages**: Scales gracefully to high dimensions ($K > 100$) by avoiding explicit $\mathcal{O}(K^3)$ matrix operations via the Woodbury matrix identity.

---

### B. Distributional/Non-Gaussian Models

When targets are multimodal, skewed, or have bounded support, Gaussian assumptions fail.

#### 1. Normalizing Flows: `NormalizingFlowLoss`
Uses invertible neural networks to map simple base distributions to complex target densities.
- **API**: `NormalizingFlowLoss` (which integrates with `zuko`).
- **Use Case**: Highly complex joint target spaces (e.g., banana-shaped or disjoint structures).

#### 2. Mixture Density Networks: `MDNLoss`
A mixture of multivariate Gaussians:
$$p(\mathbf{y} \mid \mathbf{x}) = \sum_{j=1}^J \pi_j(\mathbf{x}) \, \mathcal{N}\bigl(\mathbf{y} \mid \boldsymbol{\mu}_j(\mathbf{x}), \mathbf{\Sigma}_j(\mathbf{x})\bigr)$$
- **Use Case**: Disjoint modes/conditional branching.

---

## 3. Multi-Target Conformal Calibration

To get prediction regions with finite-sample coverage guarantees, we offer conformal wrappers.

### Coordinate-Wise Conformal: `MultiTargetConformal`
Applies conformal calibration to each target dimension independently.
- **Interval**: Produces a hyper-rectangle:
  $$\hat{C}(\mathbf{x}) = [\hat{y}_1 \pm q_1, \; \hat{y}_2 \pm q_2, \; \dots, \; \hat{y}_K \pm q_K]$$

### Joint Conformal Regions: `SLSConformal`
Constructs prediction regions (ellipsoids or arbitrary volumes) that respect target covariance or density level sets.

---

## 4. Multi-Target Ensembling

You can compose multi-target loss heads with ensembles to decompose uncertainty:
- **Aleatoric**: The expected per-member covariance matrix $\mathbb{E}_m [\mathbf{\Sigma}_m]$.
- **Epistemic**: The covariance of the predicted means across ensemble members:
  $$\mathbf{\Sigma}_{\text{epistemic}} = \frac{1}{M} \sum_{m=1}^M (\boldsymbol{\mu}_m - \bar{\boldsymbol{\mu}})(\boldsymbol{\mu}_m - \bar{\boldsymbol{\mu}})^\top$$

---

## 5. Complete Implementation Example

Below is a self-contained example creating a multi-target network, training it with a full covariance loss, and calibrating it conformally.

```python
import torch
import torch.nn as nn
from torchregress.losses import MultivariateGaussianLoss, MultiTargetConformal

# 1. Define a Multi-Target Model with Covariance Outputs
class MultivariateModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(64, out_features)
        # Parameterize lower triangular Cholesky factor to guarantee PSD covariance
        self.cov_head = nn.Linear(64, out_features * out_features)
        self.out_features = out_features

    def forward(self, x):
        features = self.backbone(x)
        mean = self.mean_head(features)

        # Build lower triangular Cholesky matrix L
        cov_raw = self.cov_head(features).view(-1, self.out_features, self.out_features)
        L = torch.tril(cov_raw)
        # Guarantee diagonal elements are strictly positive
        diag_mask = torch.eye(self.out_features, device=x.device).unsqueeze(0)
        L = L * (1 - diag_mask) + torch.diag_embed(torch.exp(torch.diagonal(L, dim1=-2, dim2=-1)))

        # Compute Sigma = L @ L.T
        cov = torch.bmm(L, L.transpose(-2, -1))
        return mean, cov

# 2. Setup training
in_features, out_features = 10, 3
model = MultivariateModel(in_features, out_features)
loss_fn = MultivariateGaussianLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Dummy training step
x, y = torch.randn(32, in_features), torch.randn(32, out_features)
mean, cov = model(x)
loss = loss_fn(mean, y, covariance_matrices=cov)
loss.backward()
optimizer.step()
optimizer.zero_grad()

# 3. Conformal Calibration for Multi-Target
cp = MultiTargetConformal(alpha=0.1)

# Evaluate model on calibration set
with torch.no_grad():
    y_pred_cal, _ = model(x_cal)
# Calibrate each target dimension
cp.calibrate(y_pred_cal, y_cal)

# 4. Inference
with torch.no_grad():
    y_pred_test, _ = model(x_test)
lower, upper = cp.predict_interval(y_pred_test)  # Returns bounding boxes [lower, upper]
```

---

## 6. Comparison & Selection

| Strategy | Multi-Target Dimension | Non-Gaussian? | Complexity |
|:---|:---:|:---:|:---:|
| `MultivariateGaussianLoss` | Small ($K \le 10$) | No | $\mathcal{O}(K^3)$ |
| `LowRankGaussianLoss` | Large ($K > 10$) | No | $\mathcal{O}(K \cdot R)$ |
| `NormalizingFlowLoss` | Small/Medium | Yes | High |
| `MDNLoss` | Small ($K \le 5$) | Yes | High |

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Bishop, C. M. ["Mixture Density Networks."](https://publications.aston.ac.uk/id/eprint/373/) *Technical Report NCRG/94/004*, Aston University, 1994. |
| 2 | Papamakarios et al. ["Normalizing Flows for Probabilistic Modeling."](https://jmlr.org/papers/v22/19-1028.html) *JMLR*, 2021. |
| 3 | Romano et al. ["Conformalized Quantile Regression."](https://arxiv.org/abs/1905.03222) *NeurIPS*, 2019. |
