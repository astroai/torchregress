# Conformal API

Conformal prediction (CP) in **torchregress** lives in
[`torchregress.losses.conformal`](../api/losses.md). This page
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
| `DistributionalConformal` | Distributional coverage | When full predictive CDF is available |
| `LevelSetConformalPredictor` | Level-set predictor | Construct smallest intervals from CDF |
| `MultiDimensionalConformalLoss` | Multi-dim legacy wrapper | Vector targets |
| `MultiTargetConformal` | Multi-target coverage | Joint coverage on `[B, D]` targets |
| `PrevalenceAdjustedCP` | Group-prevalence-adjusted | Subgroup prior shift |
| `R2CConformal` | Regression-as-classification | Multimodal / binned targets |
| `SLSConformal` | Super-level set conformal | Density-based / SLS regression |
| `CVPlus` | Out-of-fold residuals | Cross-validation ensembles |
| `JackknifePlus` | Leave-one-out residuals | Leave-one-out cross-validation ensembles (alias of CVPlus) |
| `EnsembleBatchCP` | Out-of-bag residuals | Bootstrap-based ensemble predictors (EnbPI) |
| `ConformalLoss(method=…)` | Unified CQR/UACQR/split training wrapper | One-call training of supported methods |
| `conformal_loss(...)` | Functional form of `ConformalLoss` | Inline / functional API |

---

## Calibrators (post-hoc)

| Symbol | Description |
|:-------|:------------|
| `ConformalPredictor` | Base post-hoc calibrator: `calibrate(y_pred, target)` then `predict_interval(y_pred)`. |
| `MultiTargetConformal` | Multi-target version. Returns per-target coverage. |

---

## Test-time shift-aware conformal

In `torchregress.test_time.ot_conformal`:

| Symbol | Description |
|:-------|:------------|
| `ScoreCDFReweighter` | Estimates sample weights `w(x) = p_target(x) / p_source(x)` via optimal transport. |
| `OptimalTransportCoverageGap` | Computes empirical coverage gap between source and target before/after reweighting. |
| `WeightedSplitConformalAdapter` | Weighted split conformal with per-test-point `w(x)`. Maintains coverage under covariate shift. |
| `weighted_split_classification_predictive_batch` | Classification-style adaptation; useful for ordinal / binned heads. |

**References:** Tibshirani et al., "Conformal Prediction Under Covariate Shift" (NeurIPS 2019).

---

## Quick example

```python
from torchregress.losses import ConformalLoss, CQR

# CQR as a training wrapper (alpha = miscoverage; 1-alpha = target coverage)
loss_fn = ConformalLoss(method="cqr", alpha=0.1)
loss = loss_fn(y_pred, y_true)

# Or use CQR directly as a post-hoc calibrator
cp = CQR(alpha=0.1)
cp.calibrate(y_pred_cal, y_cal)
lower, upper = cp.predict_interval(y_pred_test)
```

## Next steps

- [Conformal overview](../methods/conformal/index.md) — unified coverage theorem, full method details
- [Conformal predictors](../methods/conformal/predictors.md) — score formulas, parameter tables, best practices
- [Distributional conformal](../methods/conformal/distributional.md) — CDF/density-based methods
- [Test-time shift](../methods/test-time/ot-shift-conformal.md) — shift-aware conformal
