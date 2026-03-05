# Uncertain Ground-Truth Losses

These losses handle settings where the ground-truth labels themselves are **noisy, uncertain, or only partially trusted**.  Instead of fitting to potentially corrupted targets, they explicitly model the label noise.

---

## When Does This Arise?

!!! example "Common scenarios"

    - **Crowdsourced labels** — aggregated from multiple noisy annotators
    - **Proxy labels** — e.g., photometric redshift from template fitting (imprecise)
    - **Semi-supervised learning** — mixing labelled and pseudo-labelled data
    - **Weak supervision** — labels from heuristic rules or older, less accurate models
    - **Known measurement uncertainty** — target values come with reported error bars ($\pm\sigma_y$)

---

## Mathematical Background

### Noise-Aware Gaussian NLL

When the target $y$ is observed with **known variance** $\sigma_y^2$:

$$y = y^* + \epsilon_y, \qquad \epsilon_y \sim \mathcal{N}(0,\, \sigma_y^2)$$

The noise-aware NLL simply **adds** the target noise variance to the model's predicted variance:

$$\boxed{\,\mathcal{L} = \frac{1}{2}\log\!\bigl(2\pi(\sigma_{\text{pred}}^2 + \sigma_y^2)\bigr) + \frac{(y - \mu_{\text{pred}})^2}{2(\sigma_{\text{pred}}^2 + \sigma_y^2)}\,}$$

!!! info "Why this works"
    The combined variance $\sigma_{\text{pred}}^2 + \sigma_y^2$ models the **total** observation noise.  This prevents the model from overfitting to measurement noise while still learning the true underlying variance structure.

### Consistency Regularisation

Encourages the student model to agree with a teacher (which may be an EMA or ensemble):

$$\mathcal{L}_{\text{consist}} = \bigl\lVert f_{\text{student}}(x) - \operatorname{sg}\!\bigl[f_{\text{teacher}}(x)\bigr]\bigr\rVert^2$$

where $\operatorname{sg}[\cdot]$ denotes stop-gradient.

### Pseudo-Label Blending

Soft combination of observed and pseudo labels weighted by confidence:

$$\tilde{y} = w \cdot y_{\text{obs}} + (1 - w) \cdot y_{\text{pseudo}}$$

where $w \in [0, 1]$ is the per-sample confidence weight.

---

## Available Losses

### NoisyTargetGaussianNLL

Gaussian NLL that incorporates **known per-sample target uncertainty**.  Ideal when each observation comes with a reported error bar.

```python
import torch
from torchregress.losses import NoisyTargetGaussianNLL

loss_fn = NoisyTargetGaussianNLL()

# Model predicts (mean, log_var)
mean = torch.randn(64, 1)
log_var = torch.randn(64, 1)

# Targets with known uncertainties
y_obs = torch.randn(64, 1)
y_obs_var = torch.rand(64, 1) * 0.5  # known variance per sample

loss = loss_fn((mean, log_var), y_obs, target_variance=y_obs_var)
```

!!! tip "When to use"
    Whenever your labels come with **reported measurement uncertainties** (e.g., photometric redshifts with $\sigma_z$, sensor readings with known noise level).

### ConsistencyRegLoss

Teacher-student consistency regularisation for semi-supervised or self-training settings:

```python
from torchregress.losses import ConsistencyRegLoss

loss_fn = ConsistencyRegLoss()
loss = loss_fn(student_pred, y_obs, teacher_pred=teacher_pred)
```

!!! tip "When to use"
    When you have a **teacher model** (trained on cleaner data, or an EMA of the student) and want to regularise the student to be consistent.

### PseudoLabelNLL

Blends observed labels with pseudo-labels via per-sample confidence weights:

```python
from torchregress.losses import PseudoLabelNLL

loss_fn = PseudoLabelNLL()
loss = loss_fn(
    y_pred, y_obs,
    pseudo_labels=y_pseudo,
    confidence=confidence_weights,  # per-sample blending weight
)
```

!!! tip "When to use"
    Semi-supervised learning where a subset of predictions are confident enough to use as **soft labels**.

---

## Propensity Score Support

All uncertain-GT losses support optional **propensity scores** for correcting **selection bias** in the labelled data.

!!! example "Selection-bias scenario"
    If only bright sources get spectroscopic redshifts, the labelled sample is biased toward bright objects.  Propensity weighting upweights faint sources to correct this.

    ```python
    loss = loss_fn(
        y_pred, y_obs,
        target_variance=y_obs_var,
        propensity_scores=propensity,   # P(labelled | x)
    )
    ```

---

## Complete Example

```python
import torch
import torch.nn as nn
from torchregress.losses import NoisyTargetGaussianNLL

# Simulated photometric redshifts with per-object error bars
torch.manual_seed(42)
n = 500
x_features = torch.randn(n, 10)                  # photometric features
z_true = 0.5 + torch.rand(n, 1)                   # true redshift
z_err = 0.02 + 0.05 * torch.rand(n, 1)            # reported spectro-z error
z_obs = z_true + z_err * torch.randn(n, 1)         # noisy observation

# Model predicts (mean, log_var) — learned aleatoric uncertainty
class PhotoZModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.mean_head = nn.Linear(32, 1)
        self.logvar_head = nn.Linear(32, 1)

    def forward(self, x):
        h = self.net(x)
        return self.mean_head(h), self.logvar_head(h)

model = PhotoZModel()
loss_fn = NoisyTargetGaussianNLL()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(300):
    mean, logvar = model(x_features)
    # target_variance: known per-sample noise from spectroscopy
    loss = loss_fn((mean, logvar), z_obs, target_variance=z_err**2)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

!!! tip "Practical advice"
    - **Don't double-count noise**: if your model already learns aleatoric variance, adding `target_variance` prevents fitting to known measurement noise
    - **Use propensity scores** when the labelled sample is biased (e.g., only bright sources have spectroscopic redshifts)

---

## Related

- [Noisy Labels](noisy_labels.md) — additional robustness approaches
- [Imbalanced Regression](imbalanced.md) — propensity-based reweighting
- [Uncertain-GT + Density Conformal](../examples/uncertain_gt_density_conformal_comparison.md) — end-to-end example
- [Uncertain-GT + Density Conformal (Real Data)](../examples/uncertain_gt_density_conformal_realdata_comparison.md) — real-data benchmark

---

## References

| # | Reference |
|:-:|:----------|
| 1 | S. Laine, T. Aila. "Temporal Ensembling for Semi-Supervised Learning." *ICLR*, **2017**. |
| 2 | D.H. Lee. "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method." *ICML Workshop*, **2013**. |
| 3 | A. Natarajan, I.S. Dhillon, P. Ravikumar, A. Tewari. "Learning with Noisy Labels." *NeurIPS*, **2013**. |
