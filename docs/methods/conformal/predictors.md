# Conformal Predictors

Reference for all point-prediction and quantile-based conformal predictors.
For distributional methods (CDF/density-based), see [Distributional Conformal](distributional.md).

---

## Base Class — `ConformalPredictor`

All predictors inherit from this base class which provides the calibrate → predict workflow and composable features.

```python
class ConformalPredictor:
    def __init__(self, alpha=0.1, normalize_fn=None): ...
    def calibrate(self, y_pred, target, *, mask=None, groups=None, weights=None, x=None): ...
    def predict_interval(self, y_pred, *, groups=None, x=None) -> (lower, upper): ...
```

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `alpha` | `float` | Miscoverage rate. $\alpha = 0.1$ → 90 % coverage target. Must be in $(0, 1)$. |
| `normalize_fn` | `callable` | Optional. $(\hat{y}, x) \mapsto d(x)$ returning a per-sample difficulty scalar.  Nonconformity scores are divided by $d(x)$ for adaptive intervals. |
| `mask` | `Tensor` | Boolean mask indicating valid calibration samples. |
| `groups` | `Tensor` | Integer group labels for Mondrian conditional calibration. |
| `weights` | `Tensor` | Importance weights for covariate-shift-robust quantiles. |

---

## SplitConformal

The simplest and most widely used conformal predictor.

!!! abstract "Summary"
    **Score:**  $\;s_i = \lvert y_i - \hat{y}_i \rvert$
    **Interval:**  $\;\hat{y} \pm \hat{q}$  (constant width unless normalised)
    **Requires:**  Point predictions only

```python
from torchregress.losses import SplitConformal

cp = SplitConformal(alpha=0.1)
cp.calibrate(y_pred_cal, y_cal)
lower, upper = cp.predict_interval(y_pred_test)
```

→ See [SplitConformal API](../../api/losses.md#splitconformal).

!!! tip "When to use"
    Use as a **baseline**.  If residuals are roughly homoscedastic, SplitConformal works well.
    For heteroscedastic data, switch to CQR or add `normalize_fn`.

!!! warning "Constant-width limitation"
    `SplitConformal` produces intervals of **constant width** ($\pm \hat{q}$) for all test points (unless `normalize_fn` is provided). For heteroscedastic data where noise varies strongly with $x$, this is inefficient: intervals are unnecessarily wide in low-noise regions and risk under-coverage in high-noise regions. Use `CQR` or `MonteCarloConformal` for adaptive-width intervals.

---

## CQR

**Conformalized Quantile Regression** — the go-to method for heteroscedastic data.

!!! abstract "Summary"
    **Score:**  $\;s_i = \max\!\bigl(\hat{q}_{\mathrm{lo}}^{(i)} - y_i,\; y_i - \hat{q}_{\mathrm{hi}}^{(i)}\bigr)$
    **Interval:**  $\;[\hat{q}_{\mathrm{lo}} - \hat{q},\; \hat{q}_{\mathrm{hi}} + \hat{q}]$
    **Requires:**  Model that predicts $[\hat{q}_{\mathrm{lo}},\, \hat{q}_{\mathrm{hi}}]$

```python
from torchregress.losses import CQR

cqr = CQR(alpha=0.1, debias=True)

# y_pred_cal: Tensor of shape (n, 2) = [q_lo, q_hi] per sample
cqr.calibrate(y_pred_cal, y_cal)
lower, upper = cqr.predict_interval(y_pred_test)
```

→ See [CQR API](../../api/losses.md#cqr).

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `debias` | `bool` | `False` | Apply finite-sample bias correction (Gibbs et al., 2025). |

!!! tip "When to use"
    Whenever you have a **quantile regression** model (e.g., `MultiQuantileLoss` with quantiles $[\alpha/2, 1-\alpha/2]$).  CQR inherits the model's adaptive width and adds the coverage guarantee.

!!! warning "Base model quality matters"
    CQR adds a conformal correction to a quantile regression model, but it cannot fix a fundamentally broken quantile model. If the base quantile predictions are extremely poor (e.g., the 90% quantile is consistently below the median), the conformal correction $\hat{q}$ grows very large, producing intervals that are technically valid but uselessly wide. Always validate the base quantile model's calibration before applying CQR.

!!! quote "Reference"
    Y. Romano, E. Patterson, E. Candès. "Conformalized Quantile Regression." *NeurIPS*, **2019**.

---

## UACQR

**Uncertainty-aware CQR** — same CQR score and interval construction as [CQR](../../api/conformal.md#cqr), but nonconformity scores are divided by the predicted quantile band width \((\hat{q}_{\mathrm{hi}} - \hat{q}_{\mathrm{lo}})\) (clamped), so calibration respects the model’s own uncertainty scale. Implemented as a thin subclass; training uses the same pinball loss as CQR via [`ConformalLoss(method="uacqr")`](../../api/conformal.md#conformalloss).

```python
from torchregress.losses import UACQR

u = UACQR(alpha=0.1, debias=False, min_width=1e-6, aggregation="mean")
u.calibrate(y_pred_cal, y_cal)  # x optional; width from y_pred only
lower, upper = u.predict_interval(y_pred_test)
```

---

## DensityConformal

Density-adaptive split conformal — widens intervals where the **target distribution is sparse**.

!!! abstract "Summary"
    **Score:**  Density-weighted absolute residuals
    **Interval:**  Wider in low-density target regions
    **Requires:**  Point predictions (density estimated internally)

```python
from torchregress.losses import DensityConformal

dcp = DensityConformal(alpha=0.1)
dcp.calibrate(y_pred_cal, y_cal)
lower, upper = dcp.predict_interval(y_pred_test)
```

!!! tip "When to use"
    When the target distribution has **long tails** or extreme class imbalance (e.g., few samples at high magnitudes, rare ages).

!!! warning "Extrapolation beyond calibration range"
    `DensityConformal` estimates target density from the calibration set $y_{\text{cal}}$. If test targets fall outside the range of calibration targets, the density estimate extrapolates poorly, and the resulting intervals may be unreliable. Ensure the calibration set spans the full range of expected test targets.

---

## PrevalenceAdjustedCP

Group-prevalence-aware conformal prediction.  Adjusts thresholds to account for **varying group sizes** so that rare groups still receive adequate coverage.

!!! abstract "Summary"
    **Score:**  Group-adjusted residuals
    **Requires:**  Group labels for both calibration and test data

```python
from torchregress.losses import PrevalenceAdjustedCP

pacp = PrevalenceAdjustedCP(alpha=0.1)
pacp.calibrate(y_pred_cal, y_cal, groups=group_labels_cal)
lower, upper = pacp.predict_interval(y_pred_test, groups=group_labels_test)
```

!!! tip "When to use"
    When you have **natural grouping** (instrument, source type, demographic) and want coverage guarantees per group, even for rare subpopulations.

---

## MonteCarloConformal

Conformal prediction using **stochastic forward passes** (MC-Dropout, ensemble, variational inference) with uncertainty-normalised scores.

!!! abstract "Summary"
    **Score:**  $\;s_i = \lvert y_i - \hat{y}_i \rvert \,/\, \hat{\sigma}_i$
    **Interval:**  $\;\hat{y} \pm \hat{q}\cdot\hat{\sigma}$
    **Requires:**  Model providing predictive mean + std from multiple forward passes

```python
import torch
from torchregress.losses import MonteCarloConformal

mccp = MonteCarloConformal(alpha=0.1)
# mc_samples: [n_mc, n_cal, ...] stochastic predictive samples
n_mc, n_cal = 20, y_cal.shape[0]
mc_cal = y_pred_cal.unsqueeze(0) + 0.4 * torch.randn(n_mc, n_cal, *y_cal.shape[1:])
mccp.calibrate(mc_cal, y_cal)

mc_test = y_pred_test.unsqueeze(0) + 0.4 * torch.randn(n_mc, *y_pred_test.shape)
lower, upper = mccp.predict_interval(mc_test)
```

!!! tip "When to use"
    When your model already provides **stochastic predictions** (dropout active at test time, ensemble, SWAG samples).  The normalised scores automatically adapt interval width to the predicted uncertainty.

---

## MultiTargetConformal

Per-dimension conformal calibration for **multi-output** regression. Each output dimension receives its own conformal quantile threshold.

!!! abstract "Summary"
    **Score:**  Per-dimension absolute residuals: $s_{i, d} = |y_{i, d} - \hat{y}_{i, d}|$
    **Interval:**  Independent $[\hat{y}_{i, d} - \hat{q}_d,\, \hat{y}_{i, d} + \hat{q}_d]$ per dimension $d \in \{1, \dots, D\}$
    **Requires:**  Vector predictions of shape $(n, D)$

```python
from torchregress.losses import MultiTargetConformal

# For coordinate-wise 90% coverage
mtcp = MultiTargetConformal(alpha=0.1)
mtcp.calibrate(y_pred_cal, y_cal)     # shapes: (n_cal, D)
lower, upper = mtcp.predict_interval(y_pred_test)  # each shape: (n_test, D)
```

!!! warning "Joint Coverage & Bonferroni Correction"
    `MultiTargetConformal` performs **coordinate-wise** calibration. If you calibrate with $\alpha$, the interval for each individual dimension $d$ will cover the true target $y_d$ with probability at least $1 - \alpha$.

    However, the probability that the **entire** target vector is covered simultaneously (joint coverage) is not guaranteed to be $1 - \alpha$. In the worst case, the joint coverage can drop to $1 - D\alpha$.

    To guarantee a joint coverage of at least $1 - \alpha$ across all $D$ dimensions, you should apply the **Bonferroni correction** by initializing the predictor with:
    $$\alpha_{\text{adjusted}} = \frac{\alpha}{D}$$

    ```python
    D = y_cal.shape[-1]
    joint_alpha = 0.1
    # Calibrate each dimension at (1 - 0.1/D) to guarantee 90% joint coverage
    mtcp = MultiTargetConformal(alpha=joint_alpha / D)
    ```

!!! tip "When to use"
    Use for multi-output regression where you want independent coverage guarantees per dimension or need a simple joint bounding box via Bonferroni correction. If target dimensions are highly correlated, Bonferroni-corrected intervals can be overly conservative; in such cases, projecting residuals into a joint space (e.g. Mahalanobis distance) is preferred.

---

## LocalConformal

**Locally Valid and Discriminative Conformal Prediction (LVD)** — constructs prediction intervals that are valid locally in the feature space by using a kernel-weighted conformal quantile.

!!! abstract "Summary"
    **Score:**  $\;s_i = \lvert y_i - \hat{y}_i \rvert$
    **Interval:**  $\;\hat{y} \pm q(x)$ (width varies locally based on kernel similarity)
    **Requires:**  Point predictions and representation/feature space embeddings ($x$)

```python
from torchregress.losses import LocalConformal

# Initialize with alpha = 0.1 and a default Gaussian kernel with bandwidth = 0.5
cp = LocalConformal(alpha=0.1, bandwidth=0.5)

# x_cal: representation/embedding features for the calibration set
cp.calibrate(y_pred_cal, y_cal, x=x_cal)

# Predict intervals using test representations x_test
lower, upper = cp.predict_interval(y_pred_test, x=x_test)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `K_obj` | `Any` | `None` | Optional custom kernel object implementing `K(x1, x2)` and `Ki(xi, Xs)`. |
| `bandwidth` | `float` | `1.0` | Bandwidth $h$ of the default Gaussian kernel: $K(x_1, x_2) = \exp(-\|x_1 - x_2\|^2 / (2 h^2))$. |

!!! tip "When to use"
    Whenever you want **locally valid** coverage guarantees rather than standard marginal ones, meaning the coverage target is met even within localized neighborhoods in feature space. This is highly effective for deep learning models when using representation embeddings (e.g., from the penultimate layer).

!!! quote "Reference"
    Z. Lin, S. Trivedi, J. Sun. "Locally Valid and Discriminative Prediction Intervals for Deep Learning Models." *NeurIPS*, **2021**.

---

## LocalConformalMAD

Difficulty-normalized variant of `LocalConformal` where residuals are scaled by a predicted mean absolute deviation (MAD) or local uncertainty scale, providing discriminative heteroscedastic local coverage.

!!! abstract "Summary"
    **Score:**  $\;s_i = \lvert y_i - \hat{y}_i \rvert \,/\, (\epsilon + \hat{\sigma}_i)$
    **Interval:**  $\;\hat{y} \pm q(x) \cdot (\epsilon + \hat{\sigma})$
    **Requires:**  Point predictions, features ($x$), and uncertainty/MAD estimates ($\hat{\sigma}$)

```python
from torchregress.losses import LocalConformalMAD

cp = LocalConformalMAD(alpha=0.1, bandwidth=0.5, eps=1e-5)

# x_cal: features, mad_cal: predicted residual/MAD scales
cp.calibrate(y_pred_cal, y_cal, x=x_cal, mad=mad_cal)

# Predict intervals using test features and predicted MAD scales
lower, upper = cp.predict_interval(y_pred_test, x=x_test, mad=mad_test)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `eps` | `float` | `1e-5` | Small epsilon clamped for division stability. |

!!! tip "When to use"
    When you have a secondary model predicting local residual/error scales (MAD) and want local coverage that combines both local density weighting and heteroscedastic scaling.

---

## CVPlus & JackknifePlus

**CV+** and **Jackknife+** (leave-one-out variant) conformal predictors are designed specifically for ensemble models (e.g., cross-validation ensembles). Instead of holding out a single calibration dataset, they reuse the out-of-fold/out-of-bag predictions from a $K$-fold training procedure.

!!! abstract "Summary"
    **Score:** $\;s_i = \lvert y_i - \hat{y}_{-f(i)}(x_i) \rvert$ (out-of-fold absolute residual)
    **Interval:** $\;\left[ \text{Quantile}\left(\{\hat{y}_{-f(i)}(x) - s_i\}_{i=1}^n, \alpha\right), \; \text{Quantile}\left(\{\hat{y}_{-f(i)}(x) + s_i\}_{i=1}^n, 1-\alpha\right) \right]$
    **Requires:** Out-of-fold predictions on calibration data, and all member models for test prediction.

```python
from torchregress.losses import CVPlus
import torch

cp = CVPlus(alpha=0.1)

# Predict on calibration set using out-of-fold models
# y_pred_oob: shape [n_cal, output_dim]
# fold_indices: shape [n_cal] indicating which model was held out
cp.calibrate_ensemble(y_pred_oob, y_cal, fold_indices)

# Predict on test set using all K models
# y_pred_members: shape [K, n_test, output_dim]
lower, upper = cp.predict_interval(y_pred_members)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `alpha` | `float` | `0.1` | Miscoverage rate. |

!!! tip "When to use"
    Use when you are training a **$K$-fold ensemble** or a **leave-one-out ensemble**. This allows you to perform conformal prediction without splitting off a separate calibration set, which is highly sample-efficient for small datasets.

!!! quote "Reference"
    | # | Reference |
    |:-:|:----------|
    | 1 | Barber et al. ["Predictive inference with the jackknife+."](https://arxiv.org/abs/1905.02928) *The Annals of Statistics*, 2021. |

---

## EnsembleBatchCP

**Ensemble Batch Conformal Prediction (EnbPI)** (also known as Bootstrap+) uses out-of-bag (OOB) predictions from bootstrap ensembles (e.g., bagging, Random Forests) to obtain conformal intervals around the ensemble mean prediction.

!!! abstract "Summary"
    **Score:** $\;s_i = \lvert y_i - \hat{y}_{\text{OOB}}(x_i) \rvert$ (out-of-bag residual)
    **Interval:** $\;\hat{y}_{\text{mean}} \pm \hat{q}$
    **Requires:** Out-of-bag ensemble predictions for training points, and the ensemble mean prediction for test points.

```python
from torchregress.losses import EnsembleBatchCP

cp = EnsembleBatchCP(alpha=0.1)

# y_pred_oob: out-of-bag ensemble predictions for calibration samples
cp.calibrate(y_pred_oob, y_cal)

# y_pred_mean: ensemble mean prediction for test samples
lower, upper = cp.predict_interval(y_pred_mean)
```

!!! tip "When to use"
    Use when wrapping a **bagged ensemble** or **Random Forest** where out-of-bag predictions are readily available. It avoids training separate cross-validation models and provides homoscedastic conformal intervals.

!!! quote "Reference"
    | # | Reference |
    |:-:|:----------|
    | 1 | Xu & Xie. ["Conformal prediction interval for dynamic time-series."](https://arxiv.org/abs/2010.14144) *ICML*, 2021. |

---

## ConformalLoss (Legacy)

!!! warning "Deprecated in favour of standalone predictors"
    `ConformalLoss` mixes training-loss and calibration logic in one object.
    **Prefer the standalone classes above** for clarity and composability.

```python
from torchregress.losses import ConformalLoss

loss_fn = ConformalLoss(method="split", alpha=0.1)
```

→ For distributional conformal methods, see [Distributional Conformal](distributional.md).
