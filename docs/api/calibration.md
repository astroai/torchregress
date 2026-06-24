# Calibration API

Complete reference for `torchregress.calibration`. This package groups
**post-hoc transforms** (validation-set calibration of means / variances / PIT),
**metrics** (re-exported for convenience under the same umbrella), and
**shift-aware test-time calibrators**.

For background, see [Calibration methods](../methods/calibration.md) and
[Uncertainty decomposition](../guide/uncertainty-decomposition.md).

---

## Post-hoc transforms (`calibration.posthoc`)

| Symbol | Description |
|:-------|:------------|
| `VarianceTemperatureScaler` | Scalar temperature `T` scaling on Gaussian predictive variance. `fit(pred_mean, pred_var, target)` minimises NLL; `transform(pred_var) -> T * pred_var`. Reference: Guo et al. 2017. |
| `IsotonicMeanCalibrator` | Isotonic regression on point predictions, implemented natively via PAVA (no sklearn dependency). `fit(pred_mean, target)`, `transform(pred_mean)`. `out_of_bounds ∈ {"clip"}`. |
| `PITCalibrator` | Monotonic empirical-CDF mapping for Probability Integral Transform values. `pit_from_gaussian(pred_mean, pred_std, target)` computes PIT; `fit(pit_values)`, `transform(pit_values)` returns remapped values. |

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
[Metrics API](metrics.md#calibration-metrics-metricscalibration) for full details.

| Symbol | Origin |
|:-------|:-------|
| `ExpectedCalibrationError`, `MarginalCalibrationError` | `metrics.calibration` |
| `expected_calibration_error`, `marginal_calibration_error` | functional form |
| `bias`, `calibration_score`, `calibration_metrics_report` | functional helpers |

---

## Decision guide

| Situation | Use |
|:----------|:----|
| Gaussian variance miscalibration | `VarianceTemperatureScaler` |
| Point-prediction miscalibration | `IsotonicMeanCalibrator` |
| PIT-based full-CDF calibration | `PITCalibrator` |
| Representation shift at test time | `RepresentationShiftInflator` |
| Label prior shift at test time | `BinnedLabelShiftEstimator` |
| Conformal + covariate shift | `SemiConformalCalibrator` |

## Next steps

- [Calibration methods](../methods/calibration.md)
- [Test-time shift (OT conformal)](../methods/test-time/ot-shift-conformal.md)
- [Conformal prediction](../methods/conformal/index.md)


## Detailed Class References

### VarianceTemperatureScaler

Optimises a scalar temperature $T$ on the validation set to rescale predicted variances:

$$
\sigma^2_{\text{cal}} = T \cdot \sigma^2
$$

by minimising the negative log-likelihood of the validation data.

### IsotonicMeanCalibrator

Calibrates point predictions $\hat{y}$ using a monotonic function fit via pool-adjacent-violators algorithm (PAVA):

$$
\hat{y}_{\text{cal}} = f_{\text{iso}}(\hat{y})
$$

### PITCalibrator

Monotonic empirical-CDF mapping for Probability Integral Transform values:

$$
\text{PIT}_{\text{cal}} = F_{\text{cal}}(\text{PIT})
$$

### BinnedLabelShiftEstimator

Estimates target label prior shift using black-box shift estimation (BBSE) or EM prior correction under marginal label shift.
