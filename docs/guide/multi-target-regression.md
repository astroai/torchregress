# Multi-Target Regression

Predicting a vector of continuous outcomes
$\mathbf{y} \in \mathbb{R}^K$ jointly is a standard requirement in
physical sciences, finance, and engineering. This page formalises the
**joint conditional distribution** $p(\mathbf{y} \mid \mathbf{x})$,
explains when the joint model outperforms independent per-target
models, and surveys the head parameterisations available in
torchregress.

The key insight: the **noise in one target is often correlated with
the noise in another** (e.g. coordinates in physical space,
wavelengths in spectroscopy, joint angles in biomechanics). Modelling
the targets independently discards that structure and produces
invalid joint confidence regions.

---

## 1. Why joint multi-target regression?

A point regression model treats each target dimension independently:

$$
\hat{\mathbf{y}}(\mathbf{x}) = \bigl[\hat y_1(\mathbf{x}), \dots, \hat y_K(\mathbf{x})\bigr]
\quad \text{with} \quad
\hat y_k(\mathbf{x}) = f_k(\mathbf{x}; \boldsymbol\theta)
$$

This discards the off-diagonal structure of the noise covariance.
For probabilistic forecasting, this matters in three ways:

1. **Target correlation.** If the conditional noise
   $\boldsymbol\varepsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma})$
   has $\Sigma_{ij} \neq 0$ for $i \neq j$, the joint predictive
   distribution is an *ellipsoid* $\mathcal{N}(\boldsymbol\mu, \mathbf{\Sigma})$,
   not a *box* $\prod_k \mathcal{N}(\mu_k, \sigma_k^2)$. Joint
   confidence regions are tighter than the product of marginal
   intervals.
2. **Aleatoric uncertainty covariance.** Estimating
   $\mathbf{\Sigma}(\mathbf{x})$ as a function of $\mathbf{x}$ gives
   directed confidence ellipsoids that respond to local noise
   structure.
3. **Optimisation efficiency.** Sharing a feature-extraction
   backbone across multiple targets is cheaper and less prone to
   overfitting than training $K$ independent networks.

The downside is **head complexity**: a full covariance head outputs
$K + K^2$ values per example, and a $K^3$ cost appears in the
likelihood evaluation. The next sections survey the parameterisations
that control this cost.

---

## 2. Parametric Gaussian models

For Gaussian noise, the joint conditional is

$$
p(\mathbf{y} \mid \mathbf{x}) = \mathcal{N}\bigl(\boldsymbol\mu(\mathbf{x}),\, \mathbf{\Sigma}(\mathbf{x})\bigr),
$$

with a per-example mean $\boldsymbol\mu(\mathbf{x}) \in \mathbb{R}^K$
and a per-example positive-definite covariance
$\mathbf{\Sigma}(\mathbf{x}) \in \mathbb{S}_+^K$.

### 2.1 Full covariance: `MultivariateGaussianLoss`

The model outputs the mean $\boldsymbol\mu$ and the full covariance
matrix $\mathbf{\Sigma}$. The per-sample negative log-likelihood is

$$
\mathcal{L}_{\text{MVN}}(\boldsymbol\mu, \mathbf{\Sigma}; \mathbf{y}) = \frac{1}{2} \log \lvert \mathbf{\Sigma} \rvert + \frac{1}{2} (\mathbf{y} - \boldsymbol\mu)^\top \mathbf{\Sigma}^{-1} (\mathbf{y} - \boldsymbol\mu) + \frac{K}{2}\log(2\pi).
$$

**Numerical contract:** $\mathbf{\Sigma}$ must be positive definite
for $\log\lvert \mathbf{\Sigma} \rvert$ to be defined and for the
inverse to exist. Parameterise the Cholesky factor $L$ with
strictly-positive diagonal $\text{diag}(L) = \exp(\ell)$ so
$\mathbf{\Sigma} = LL^\top$ is PSD by construction; see
[§2.3](#23-cholesky-parameterisation).

**Complexity:** the determinant and inverse each cost
$\mathcal{O}(K^3)$. Use `MultivariateGaussianLoss` only for
$K \le 10$–$20$.

### 2.2 Low-rank covariance: `LowRankGaussianLoss`

For $K \gg 1$, parameterise
$\mathbf{\Sigma} = \mathbf{U}\mathbf{U}^\top + \text{diag}(\mathbf{d})$
with a low-rank factor $\mathbf{U} \in \mathbb{R}^{K \times R}$
($R \ll K$) and a strictly positive diagonal $\mathbf{d} \in
\mathbb{R}_{>0}^K$.

The inverse is available in closed form via the **Sherman–Morrison–Woodbury
identity**:

$$
(\mathbf{U}\mathbf{U}^\top + \mathbf{D})^{-1} = \mathbf{D}^{-1} - \mathbf{D}^{-1}\mathbf{U}(\mathbf{I}_R + \mathbf{U}^\top \mathbf{D}^{-1} \mathbf{U})^{-1}\mathbf{U}^\top \mathbf{D}^{-1},
$$

and the log-determinant is

$$
\log \lvert \mathbf{U}\mathbf{U}^\top + \mathbf{D} \rvert = \log \lvert \mathbf{I}_R + \mathbf{U}^\top \mathbf{D}^{-1} \mathbf{U} \rvert + \sum_k \log d_k.
$$

Both quantities are $\mathcal{O}(K R^2)$ — linear in $K$ for fixed
$R$. This makes `LowRankGaussianLoss` the right default for
$K \in \[10, 1000\]$.

**Output layout** in torchregress: the head emits
$\text{cat}(\boldsymbol\mu, \text{vec}(\mathbf{U}), \mathbf{d})$, and
`low_rank_output_dim(D, R)` returns the total output dimension.
Use `split_low_rank_gaussian_output(...)` to recover
$\boldsymbol\mu$, $\mathbf{U}$, $\mathbf{d}$ for the loss.

### 2.3 Cholesky parameterisation

The standard way to guarantee $\mathbf{\Sigma} \succ 0$ is the
lower-triangular Cholesky factorisation $\mathbf{\Sigma} = LL^\top$
with $L_{ii} > 0$. Parameterise the diagonal as
$L_{ii} = \exp(\ell_i)$ (unconstrained $\ell_i \in \mathbb{R}$) and
the off-diagonal as raw real values. The head emits $\binom{K+1}{2}$
values per example; the loss reconstructs $L$ and computes
$\log \lvert \mathbf{\Sigma} \rvert = 2 \sum_i \log L_{ii}$.

```python
class FullCovHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, out_features)
        # Lower-triangular entries (including diagonal) before exp
        self.cov_head = nn.Linear(64, out_features * (out_features + 1) // 2)
        self.out_features = out_features

    def forward(self, x):
        h = self.backbone(x)
        mean = self.mean_head(h)
        L_raw = self.cov_head(h).view(-1, self.out_features, self.out_features)
        L = torch.tril(L_raw)
        # Exponentiate the diagonal entries to keep them strictly positive
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        L = L - torch.diag_embed(diag) + torch.diag_embed(torch.exp(diag))
        cov = L @ L.transpose(-2, -1)
        return mean, cov
```

### 2.4 Diagonal covariance: independent heads

When target correlations are small, a diagonal covariance
$\mathbf{\Sigma} = \text{diag}(\boldsymbol\sigma^2)$ is a reasonable
approximation and reduces the head to $2K$ outputs
($[\boldsymbol\mu, \log\boldsymbol\sigma^2]$). This is the
**independent Gaussian** case and is implemented by stacking
$K$ independent [`GaussianNLLLoss`](../api/losses.md#gaussiannllloss) heads or by using a single
$2K$-output head with the diagonal Gaussian NLL.

The diagonal assumption understates joint uncertainty; the resulting
hyper-rectangular confidence regions are conservative but valid.

---

## 3. Non-Gaussian multi-target models

When targets are multimodal, skewed, or have bounded support, Gaussian
assumptions fail. torchregress provides:

### 3.1 Mixture density networks: [`MDNLoss`](../api/losses.md#mdnloss)

A mixture of multivariate Gaussians,

$$
p(\mathbf{y} \mid \mathbf{x}) = \sum_{j=1}^{J} \pi_j(\mathbf{x})\, \mathcal{N}\bigl(\mathbf{y} \mid \boldsymbol\mu_j(\mathbf{x}), \mathbf{\Sigma}_j(\mathbf{x})\bigr),
$$

with mixture weights $\pi_j$ (softmax-normalised), means
$\boldsymbol\mu_j \in \mathbb{R}^K$, and (optionally) per-component
covariances $\mathbf{\Sigma}_j$. Suitable for disjoint modes and
conditional branching.

**Tradeoffs:** the number of components $J$ is a hyperparameter; the
loss is well-defined only for $J \ge 2$. The head emits
$J \cdot (1 + K + K^2)$ values per example (or fewer for diagonal
per-component covariance).

### 3.2 Normalizing flows: `NormalizingFlowLoss`

Invertible neural networks map a simple base distribution to the
target density via a sequence of bijective transforms. The
log-density is

$$
\log p(\mathbf{y} \mid \mathbf{x}) = \log p_z\!\bigl(\mathbf{T}(\mathbf{y}; \mathbf{x})\bigr) + \log \left\lvert \det \frac{\partial \mathbf{T}}{\partial \mathbf{y}} \right\rvert,
$$

where $\mathbf{T}$ is the flow's transformer. Suitable for
*arbitrarily complex* joint densities (banana-shaped, disjoint,
skewed). Requires the optional `[flows]` extra (`pip install
torchregress[flows]`).

**Tradeoffs:** training is the most expensive of the alternatives;
the head is the deepest of all the parameterisations discussed; the
density is exact (up to the flow's expressivity), unlike the mixture
which is constrained by the family.

---

## 4. Multi-target conformal calibration

Conformal calibration is **per-method** and **per-task**. For
multi-target, the choice is between **coordinate-wise** (marginal
calibration per dimension) and **joint** (calibration of the joint
region).

### 4.1 Coordinate-wise: `MultiTargetConformal`

Applies conformal calibration to each target dimension independently.
The result is a hyper-rectangle

$$
\hat{C}(\mathbf{x}) = [\hat y_1 \pm q_1, \hat y_2 \pm q_2, \dots, \hat y_K \pm q_K],
$$

with marginal coverage $P(\hat y_k - q_k \le Y_k \le \hat y_k + q_k) \ge 1 - \alpha_k$
per dimension. The **joint** coverage is *at least*
$1 - \sum_k \alpha_k$ (Bonferroni) but typically much higher.

**Use when:** you need marginal per-dimension intervals and the
correlation structure is not critical to the deployment.

### 4.2 Joint conformal regions: `SLSConformal`

Constructs a joint prediction region (ellipsoid or arbitrary volume)
that respects the target covariance or density level sets. The
result is a region $\hat C(\mathbf{x}) \subset \mathbb{R}^K$ with
joint coverage $P(\mathbf{Y} \in \hat C(\mathbf{X})) \ge 1 - \alpha$.

**Use when:** joint coverage is required (e.g. safety constraints
spanning multiple targets) and the head provides a density or
covariance.

---

## 5. Multi-target ensembling

A multi-target loss head can be composed with an ensemble to obtain
decomposable uncertainty. With $M$ members, each predicting
$\boldsymbol\mu_m(\mathbf{x})$ and $\mathbf{\Sigma}_m(\mathbf{x})$,
the predictive distribution decomposes as

$$
\underbrace{\text{Cov}_{\text{total}}[\mathbf{y} \mid \mathbf{x}]}_{\text{predictive spread}}
= \underbrace{\frac{1}{M} \sum_{m=1}^{M} \mathbf{\Sigma}_m(\mathbf{x})}_{\text{aleatoric (expected per-member noise)}}
+ \underbrace{\frac{1}{M} \sum_{m=1}^{M} (\boldsymbol\mu_m - \bar{\boldsymbol\mu})(\boldsymbol\mu_m - \bar{\boldsymbol\mu})^\top}_{\text{epistemic (covariance of the means)}}
$$

with $\bar{\boldsymbol\mu} = \frac{1}{M} \sum_m \boldsymbol\mu_m$.
The decomposition is a direct application of the Law of Total
Covariance to the per-member predictive distributions.

For **MDN ensembles**, the ensemble disagreement is computed on the
mixture weights and per-component means; the within-mixture spread
contributes to the aleatoric component.

---

## 6. Worked example

The end-to-end example below trains a multi-target network with a
full covariance head and wraps it in a coordinate-wise conformal
calibrator.

```python
import torch
import torch.nn as nn
from torchregress.losses import MultivariateGaussianLoss, MultiTargetConformal

class MultivariateModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.mean_head = nn.Linear(64, out_features)
        self.cov_head = nn.Linear(64, out_features * out_features)
        self.out_features = out_features

    def forward(self, x):
        h = self.backbone(x)
        mean = self.mean_head(h)
        # Cholesky parameterisation: lower triangular L, diagonal exp
        cov_raw = self.cov_head(h).view(-1, self.out_features, self.out_features)
        L = torch.tril(cov_raw)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        L = L - torch.diag_embed(diag) + torch.diag_embed(torch.exp(diag))
        cov = L @ L.transpose(-2, -1)
        return mean, cov

in_features, out_features = 10, 3
model = MultivariateModel(in_features, out_features)
loss_fn = MultivariateGaussianLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
x, y = torch.randn(32, in_features), torch.randn(32, out_features)
mean, cov = model(x)
loss = loss_fn(mean, y, covariance_matrices=cov)
loss.backward()
optimizer.step(); optimizer.zero_grad()

# Conformal calibration
cp = MultiTargetConformal(alpha=0.1)
with torch.no_grad():
    y_pred_cal, _ = model(x_cal)
cp.calibrate(y_pred_cal, y_cal)

# Inference
with torch.no_grad():
    y_pred_test, _ = model(x_test)
lower, upper = cp.predict_interval(y_pred_test)
```

---

## 7. Comparison and selection

| Strategy | Target dim $K$ | Non-Gaussian | Complexity | When to use |
|:---|:---:|:---:|:---:|:---|
| Independent diagonal Gaussian | any | no | $\mathcal{O}(K)$ | Targets roughly independent; cheap baseline |
| `MultivariateGaussianLoss` | $K \le 10$ | no | $\mathcal{O}(K^3)$ | Strong correlations; small $K$ |
| `LowRankGaussianLoss` | $K > 10$ | no | $\mathcal{O}(K R^2)$ | High $K$; rank-$R$ structure |
| `MDNLoss` | $K \le 5$ | yes | $\mathcal{O}(J K^2)$ | Disjoint modes; conditional branching |
| `NormalizingFlowLoss` | small/medium | yes | high | Arbitrarily complex joint densities |

The choice is governed by:

- **$K$:** for $K \le 10$, use `MultivariateGaussianLoss`. For
  $K \in \[10, 1000\]$, use `LowRankGaussianLoss` with $R \approx 5$–$10$.
- **Multimodality:** for disjoint modes, use `MDNLoss`. For arbitrary
  shapes, use `NormalizingFlowLoss`.
- **Joint coverage requirements:** for hyper-rectangular regions,
  use `MultiTargetConformal`. For ellipsoidal or density-level-set
  regions, use `SLSConformal`.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Bishop. ["Mixture Density Networks"](https://publications.aston.ac.uk/id/eprint/373/) *NCRG/94/004*, 1994. |
| 2 | Papamakarios et al. ["Normalizing Flows for Probabilistic Modeling"](https://jmlr.org/papers/v22/19-1028.html) *JMLR*, 2021. |
| 3 | Romano, Patterson & Candès. ["Conformalized Quantile Regression"](https://arxiv.org/abs/1905.03222) *NeurIPS*, 2019. |
| 4 | Woodbury. *Inverting Modified Matrices*. *Statistical Research Group Memo*, 1950. |
| 5 | Lakshminarayanan, Pritzel & Blundell. ["Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"](https://arxiv.org/abs/1612.01474) *NeurIPS*, 2017. |

---

## Next steps

- [Uncertainty Decomposition](uncertainty-decomposition.md) — full taxonomy of aleatoric vs. epistemic contracts.
- [Choosing by Constraint](choosing-by-constraint.md) — latency / coverage / decomposition tradeoffs.
- [Practical Usage](practical-usage.md) — concrete loss recipes.
- [Method Selection Matrix](method-selection.md) — task-first capability matrix.
