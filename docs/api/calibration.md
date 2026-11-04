# Calibration API

Complete reference for `torchregress.calibration`. This package groups
**post-hoc transforms** (validation-set calibration of means / variances / PIT),
**metrics** (re-exported for convenience under the same umbrella), and
**shift-aware test-time calibrators**.

For background, see [Calibration methods](../methods/calibration.md) and
[Uncertainty decomposition](../guide/uncertainty-decomposition.md).

---

## Multi-source observation maps (`calibration.multisource`)

`bspline_basis_with_derivative` returns basis values, analytic derivatives and extrapolation flags with explicit linear, constant or error boundary behavior. `RelativeSurveySplineHead` imposes a per-target weighted zero-sum on relative source offsets. `CommonScaleSplineHead` represents a separate common correction, whose physical interpretation requires independent anchors. `ConditionBiasHead` is zero at each source's reference condition. The heads return both offsets and Jacobians. These components do not themselves certify calibration or external support; see [Multi-source Gaussian likelihood](../losses/multisource.md).

## Post-hoc transforms (`calibration.posthoc`)

| Symbol | Description |
|:-------|:------------|
| `VarianceTemperatureScaler` | Scalar temperature `T` scaling on Gaussian predictive variance. `fit(pred_mean, pred_var, target, *, max_iter=200, lr=0.05, target_var=None, fit_floor=False, clip=None, temperature_bounds=None)` minimises NLL (closed form `T = mean(r^2 / var)` when no floor or target noise); `transform(pred_var) -> T * pred_var + f`. `T` is unbounded unless `temperature_bounds=(t_min, t_max)` is given (pre-0.3 versions clamped silently to `[0.05, 20]`). Reference: Guo et al. 2017. |
| `IsotonicMeanCalibrator` | Isotonic regression on point predictions, implemented natively via PAVA (no sklearn dependency). Ties are pooled and block end points are kept, so it matches `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`. `fit(pred_mean, target)`, `transform(pred_mean)`. `out_of_bounds="clip"` (default) or any other value for linear extrapolation of the end segments. |
| `PITCalibrator` | Monotonic empirical-CDF mapping for Probability Integral Transform values. `pit_from_gaussian(pred_mean, pred_std, target)` computes PIT; `fit(pit_values)`, `transform(pit_values)` returns remapped values. |
| `VarianceRestoringCalibrator` | Marginal distribution-matching de-shrinkage of point predictions. `method="affine"` (moment matching, two parameters) or `"quantile"` (1-D optimal transport; `out_of_bounds="clip"|"linear"`). `fit(pred_mean, target, *, prior_mean=None)`, `transform(pred_mean)`, `describe(pred_mean, target)` reporting `mse_before/after`, `wasserstein1_before/after`, `var_ratio_before/after`. Matches the population distribution/tails, **not** the conditional mean: can increase MSE for an already calibrated predictor. |

!!! tip "`clip` trade-off"
    On the torchregress-harness `synthetic_outliers` dataset, `clip=5` gave the best CRPS
    but raised test NLL from 1.64 to 2.71: test outliers meet a variance fitted without them.
    Keep the default (no clip) when NLL matters; use `clip=5` when CRPS or robustness to
    outlier-inflated variance matters. See [Calibration methods](../methods/calibration.md).

```python
from torchregress.calibration import (
    VarianceTemperatureScaler, IsotonicMeanCalibrator, PITCalibrator,
)

# Variance temperature
vts = VarianceTemperatureScaler().fit(pred_mean, pred_var, y)
pred_var_cal = vts.transform(pred_var)

# Isotonic mean
iso = IsotonicMeanCalibrator().fit(pred_mean, y)
pred_mean_cal = iso.transform(pred_mean)

# PIT
from torchregress.calibration.posthoc import PITCalibrator
pit = PITCalibrator.pit_from_gaussian(pred_mean, pred_std, y)
cal = PITCalibrator().fit(pit)
pit_cal = cal.transform(pit)
```

---

## Shift-aware test-time calibration (`calibration.shift`)

| Symbol | Description |
|:-------|:------------|
| `RepresentationShiftInflator` | Maps Mahalanobis shift in representation space to a conservative temperature factor. `fit(source_representations)`; `shift_scores(target)`, `temperatures(target)`, `calibrate_probabilities(probs, target)`, `calibrate_std(std, target)`. |
| `BinnedLabelShiftEstimator` | Black-Box Shift Estimation (`bbse`) or EM (Saerens) prior correction. Continuous targets are binned (`adaptive` quantiles or `uniform`). `fit(y_source, pred_source, pred_target)`; `get_bin_weights()`, `sample_weights(y)`. Reference: Lipton et al. 2018. |

```python
from torchregress.calibration import (
    RepresentationShiftInflator, BinnedLabelShiftEstimator,
)

# Representation shift
rsc = RepresentationShiftInflator(base_temperature=1.0, slope=1.0)
rsc.fit(source_features)
T = rsc.temperatures(target_features)
probs_cal = rsc.calibrate_probabilities(probs, target_features)

# Label shift
bls = BinnedLabelShiftEstimator(n_bins=10, method="em")
bls.fit(y_source, pred_source, pred_target)
weights = bls.sample_weights(y_target)  # IPW weights
```

---

## Semi-supervised conformal (`calibration.semicp`)

| Symbol | Description |
|:-------|:------------|
| `SemiConformalCalibrator` | Weighted split-conformal calibrator using unlabeled target weights. `fit(nonconformity_scores_cal, weights_cal=…)`; `compute_thresholds(weights_target, alpha=0.1)` returns per-target thresholds; `calibrate_interval(lower, upper, weights_target, alpha=0.1)` widens the interval by the threshold. |

Reference: Tibshirani et al. 2019; Cauchois, Gupta, Duchi 2020.

```python
from torchregress.calibration import SemiConformalCalibrator

semi = SemiConformalCalibrator()
semi.fit(scores_cal, weights_cal=weights_cal)
thresholds = semi.compute_thresholds(weights_target, alpha=0.1)
lower_cal, upper_cal = semi.calibrate_interval(lower, upper, weights_target, alpha=0.1)
```

---

## Calibration metrics (re-exports)

The following metrics are also re-exported here for convenience — see
[Metrics API](metrics.md#calibration-metrics) for full details.

| Symbol | Origin |
|:-------|:-------|
| `ExpectedCalibrationError`, `MarginalCalibrationError` | `calibration.metrics` (also exported from `torchregress.metrics`) |
| `expected_calibration_error`, `marginal_calibration_error` | functional form |
| `bias`, `calibration_score`, `calibration_metrics_report` | functional helpers |

---

## Next steps

- [Calibration methods](../methods/calibration.md) — full conceptual guide, best practices, limitations
- [Test-time shift (OT conformal)](../methods/test-time/ot-shift-conformal.md)
- [Conformal prediction](../methods/conformal/index.md)
