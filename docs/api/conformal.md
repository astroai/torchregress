# Conformal API

Conformal prediction (CP) in **torchregress** lives in
[`torchregress.losses.conformal`](../api/losses.md#conformal-prediction). This page
lists every conformal loss, calibrator, and predictor, plus the test-time
adapters in `torchregress.test_time.ot_conformal`. For background and the
unified coverage theorem, see [Conformal prediction](../methods/conformal/index.md).

---

## Method matrix

| Symbol | Strategy | When to use |
|:-------|:---------|:------------|
| `SplitConformal` | Residual-based | Baseline; homoscedastic noise; fast |
| `CQR` | Conformalized Quantile Regression | Heteroscedastic, skewed noise; adaptive width |
| `UACQR` | Width-normalized CQR | Same as CQR; penalized by width |
| `DensityConformal` | Density-weighted residuals | Imbalanced or long-tail data |
| `MonteCarloConformal` | MC-sample normalized | Bayesian models, ensembles |
| `LocalConformal` | Local feature-space coverage | Coverage guaranteed in feature subgroups |
| `LocalConformalMAD` | Local + MAD scaling | Robust to outliers in calibration scores |
| `CTI` | Density level-set | Multimodal, complex distributions |
| `SLSConformal` | Super-level set | Multi-target regression |
| `DistributionalConformal` | Distributional coverage | When full predictive CDF is available |
| `LevelSetConformalPredictor` | Level-set predictor | Construct smallest intervals from CDF |
| `MultiDimensionalConformalLoss` | Multi-dim legacy wrapper | Vector targets |
| `MultiTargetConformal` | Multi-target coverage | Joint coverage on `[B, D]` targets |
| `PrevalenceAdjustedCP` | Group-prevalence-adjusted | Subgroup prior shift |
| `R2CConformal` | R²-style coverage | Coverage that accounts for inherent variance |
| `SLSConformal` | Super-level set conformal | Density-based / SLS regression |
| `ConformalLoss(method=…)` | Unified training+calibration wrapper | One-call training of any method above |
| `conformal_loss(...)` | Functional form of `ConformalLoss` | Inline / functional API |

---

## Calibrators (post-hoc)

| Symbol | Description |
|:-------|:------------|
| `ConformalPredictor` | Train + calibrate wrapper that returns a predictor with `predict_interval(x_test) -> (lower, upper)`. |
| `MultiTargetConformal` | Multi-target version. Returns per-target coverage. |

---

## Test-time shift-aware conformal

In `torchregress.test_time.ot_conformal`:

| Symbol | Description |
|:-------|:------------|
| `OTShiftReweighter` | Estimates sample weights `w(x) = p_target(x) / p_source(x)` via optimal transport. |
| `OptimalTransportCoverageGap` | Computes empirical coverage gap between source and target before/after reweighting. |
| `WeightedSplitConformalAdapter` | Weighted split conformal with per-test-point `w(x)`. Maintains coverage under covariate shift. |
| `weighted_split_classification_predictive_batch` | Classification-style adaptation; useful for ordinal / binned heads. |

**References:** Tibshirani et al., "Conformal Prediction Under Covariate Shift" (NeurIPS 2019).

---

## Unified training+calibration

```python
from torchregress.losses import ConformalLoss, CQR

# CQR as a single training objective
loss_fn = ConformalLoss(method="cqr", target_coverage=0.9, alpha=0.1)
loss = loss_fn(y_pred, y_true)  # training + calibration in one call

# Or use CQR directly as a calibrator
cp = CQR(target_coverage=0.9, alpha=0.1)
cp.calibrate(scores_cal, y_cal)
lower, upper = cp.predict_interval(y_pred_test)
```

## Decision guide

| Situation | Use |
|:----------|:----|
| Plain regression, homoscedastic noise | `SplitConformal` |
| Heteroscedastic, want adaptive width | `CQR` |
| Have predictive CDF | `DistributionalConformal` or `LevelSetConformalPredictor` |
| Sample efficiency / subgroup shift | `DensityConformal` or `LocalConformal` |
| Ensembles / Bayesian MC | `MonteCarloConformal` |
| Distribution shift at test time | `WeightedSplitConformalAdapter` + `OTShiftReweighter` |
| Need multi-target joint coverage | `MultiTargetConformal` |

## Detailed Class References

### ConformalLoss

Loss wrapper that trains a base model and applies conformal calibration on target outputs:

```python
ConformalLoss(base_loss, method="cqr", alpha=0.1)
```

### SplitConformal

Standard split conformal prediction using absolute residuals:

$$
s_i = |y_i - \hat{y}_i|
$$

### CQR

Conformalized Quantile Regression calibration wrapper using quantile interval width:

$$
s_i = \max\left(\hat{q}_{\alpha/2}(x_i) - y_i, y_i - \hat{q}_{1-\alpha/2}(x_i)\right)
$$

### UACQR

Uncertainty-Aware CQR scaling quantile residuals by interval width:

$$
s_i = \frac{\max\left(\hat{q}_{\alpha/2}(x_i) - y_i, y_i - \hat{q}_{1-\alpha/2}(x_i)\right)}{\hat{q}_{1-\alpha/2}(x_i) - \hat{q}_{\alpha/2}(x_i) + \varepsilon}
$$

### DensityConformal

Density-weighted split conformal prediction for long-tailed targets:

$$
s_i = \frac{|y_i - \hat{y}_i|}{\hat{p}(y_i) + \varepsilon}
$$

### MonteCarloConformal

MC-dropout or ensemble normalized split conformal prediction:

$$
s_i = \frac{|y_i - \hat{y}_i|}{\hat{\sigma}_{\text{MC}}(x_i) + \varepsilon}
$$

### LocalConformal

Locally valid split conformal prediction in embedding space:

$$
\hat{C}(x) = [\hat{y} - \hat{q}(x), \hat{y} + \hat{q}(x)]
$$

### LocalConformalMAD

Robust locally valid split conformal prediction utilizing Local Median Absolute Deviation (MAD) scaling.

### CTI

Conformalized Temperature Integration for multimodal distribution levels:

$$
\hat{C}(x) = \{ y : p(y \mid x) \ge \tau_x \}
$$

### SLSConformal

Conformal wrapper for Super-Level-Set ([`SLSLoss`](../api/losses.md#slsloss)) frontiers — calibrates level-set thresholds for multi-target prediction regions.

## Next steps

- [Conformal prediction overview](../methods/conformal/index.md)
- [Uncertainty decomposition](../guide/uncertainty-decomposition.md)
- [Test-time shift](../methods/test-time/ot-shift-conformal.md)
