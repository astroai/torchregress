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

!!! tip "When to use"
    Use as a **baseline**.  If residuals are roughly homoscedastic, SplitConformal works well.  
    For heteroscedastic data, switch to CQR or add `normalize_fn`.

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

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `debias` | `bool` | `False` | Apply finite-sample bias correction (Gibbs et al., 2025). |

!!! tip "When to use"
    Whenever you have a **quantile regression** model (e.g., `MultiQuantileLoss` with quantiles $[\alpha/2, 1-\alpha/2]$).  CQR inherits the model's adaptive width and adds the coverage guarantee.

!!! quote "Reference"
    Y. Romano, E. Patterson, E. Candès. "Conformalized Quantile Regression." *NeurIPS*, **2019**.

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
from torchregress.losses import MonteCarloConformal

mccp = MonteCarloConformal(alpha=0.1)
mccp.calibrate(y_pred_cal, y_cal)
lower, upper = mccp.predict_interval(y_pred_test)
```

!!! tip "When to use"
    When your model already provides **stochastic predictions** (dropout active at test time, ensemble, SWAG samples).  The normalised scores automatically adapt interval width to the predicted uncertainty.

---

## MultiTargetConformal

Per-dimension conformal calibration for **multi-output** regression.  Each output dimension receives its own conformal quantile.

!!! abstract "Summary"
    **Score:**  Per-dimension residuals  
    **Interval:**  Independent $[\mathrm{lo}_d, \mathrm{hi}_d]$ per output dimension  
    **Requires:**  Vector predictions of shape $(n, d)$

```python
from torchregress.losses import MultiTargetConformal

mtcp = MultiTargetConformal(alpha=0.1)
mtcp.calibrate(y_pred_cal, y_cal)     # (n_cal, d)
lower, upper = mtcp.predict_interval(y_pred_test)  # each: (n_test, d)
```

!!! tip "When to use"
    Multi-output regression where you want per-dimension coverage guarantees (e.g., predicting RA + Dec + redshift simultaneously).

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
