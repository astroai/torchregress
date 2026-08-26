# Inference API

Complete reference for `torchregress.inference`. This package implements
**Prediction-Powered Inference (PPI)** — confidence-interval estimators that
combine a small trusted-labeled set with a large model-predicted unlabeled set
to improve statistical efficiency while preserving frequentist coverage.

For background, see [PPI + conformal](../guide/method-selection.md) and the
[inference example](../examples/index.md).

---

## Configuration

| Symbol | Description |
|:-------|:------------|
| `PPIConfig` | Frozen dataclass with `alpha` (target error rate), `method` (always `"bootstrap"`), `n_boot` (default `2000`), `seed`. |

---

## Mean CI (rectified)

| Symbol | Description |
|:-------|:------------|
| `ppi_mean_ci(y_labeled, pred_labeled, pred_unlabeled, *, config=None)` | PPI CI for a population mean. Estimator: `E[Y] ≈ mean(pred_unlabeled) + mean(y_labeled − pred_labeled)`. Returns `{"estimate", "se", "ci_lower", "ci_upper", "alpha", "n_labeled", "n_unlabeled", "bootstrap_samples"}`. |
| `ppi_pp_mean_ci` | `(y_labeled, pred_labeled, pred_unlabeled, *, lambdas=None, cross_fits=0, alpha=0.05)` — PPI++ confidence interval for a population mean: selects the power-tuning parameter λ minimizing first-order PPI variance (optionally cross-fitted). |

**Reference:** Angelopoulos, Bates, Fannjiang, Jordan, Zrnic,
"Prediction-Powered Inference" (Science 2023).

```python
from torchregress.inference import ppi_mean_ci, PPIConfig

cfg = PPIConfig(alpha=0.1, n_boot=2000, seed=42)
res = ppi_mean_ci(y_labeled, pred_labeled, pred_unlabeled, config=cfg)
# res["estimate"], res["ci_lower"], res["ci_upper"]
```

---

## Linearly-calibrated mean CI

| Symbol | Description |
|:-------|:------------|
| `ppi_calibrated_mean_ci` | `(y_labeled, pred_labeled, pred_unlabeled, *, config=None)` — Like `ppi_mean_ci`, but fits an affine map `m⋆(x) = â + b̂ m(x)` by OLS on labeled pairs, then refits `(â, b̂)` on every bootstrap resample. Returns the same dict as `ppi_mean_ci` but with `method="ppi_calibrated_mean_ci"`. |

**Reference:** Chen et al., "Linearly Calibrated Prediction-Powered Inference" (arXiv 2026).

```python
from torchregress.inference import ppi_calibrated_mean_ci
res = ppi_calibrated_mean_ci(y_labeled, pred_labeled, pred_unlabeled,
                              config=PPIConfig(alpha=0.1, n_boot=2000, seed=42))
```

---

## Quantile CI

| Symbol | Description |
|:-------|:------------|
| `ppi_quantile_ci` | `(y_labeled, pred_labeled, pred_unlabeled, *, q=0.5, config=None)` — PPI CI for a target quantile. Estimator: `Q_q(Y) ≈ Q_q(pred_unlabeled) + median(y_labeled − pred_labeled)`. Bootstrap re-estimates the shift and target quantile. |

---

## OLS coefficient CI

| Symbol | Description |
|:-------|:------------|
| `ppi_ols_ci` | `(x_labeled, y_labeled, x_unlabeled, pred_labeled, pred_unlabeled, *, add_intercept=True, config=None)` — PPI CI for linear-regression coefficients. `β̂ = β̂_unlabeled + β̂_(labeled residual)`. Bootstrap refits both regressions. Returns `{"coef", "se", "ci_lower", "ci_upper", "alpha", "n_labeled", "n_unlabeled", "bootstrap_samples"}`. Default `n_boot=1000`. |

---

## Diagnostics

| Symbol | Description |
|:-------|:------------|
| `ppi_diagnostics` | `(y_labeled, pred_labeled, pred_unlabeled)` — Returns `{"n_labeled", "n_unlabeled", "prediction_label_correlation", "residual_rmse_labeled", "residual_mean_labeled", "prediction_mean_shift_unlabeled_vs_labeled", "prediction_range_overlap_ratio"}`. Use to assess PPI validity before trusting a CI. |

---

## Quick example

```python
import torch
from torchregress.inference import ppi_mean_ci, ppi_quantile_ci, ppi_ols_ci, PPIConfig

# Toy data
y_l = torch.randn(40)
p_l = y_l + 0.1 * torch.randn(40)
p_u = torch.randn(2000) + 0.05    # model on unlabeled
x_l = torch.randn(40, 3)
x_u = torch.randn(2000, 3)

# Mean CI
cfg = PPIConfig(alpha=0.1, n_boot=2000, seed=42)
mean_ci = ppi_mean_ci(y_l, p_l, p_u, config=cfg)

# Quantile CI
q_ci = ppi_quantile_ci(y_l, p_l, p_u, q=0.9, config=cfg)

# OLS coefficient CI
ols_ci = ppi_ols_ci(x_l, y_l, x_u, p_l, p_u, add_intercept=True, config=cfg)
# ols_ci["coef"] -> list of length 4 (intercept + 3 coefs)
```

---

## Next steps

- [Uncertainty decomposition](../guide/uncertainty-decomposition.md)
- [PPI examples](../examples/index.md) — coverage and efficiency in practice
- [Method selection guide](../guide/method-selection.md) — when PPI vs other UQ approaches
