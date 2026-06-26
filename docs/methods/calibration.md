# Post-Hoc Calibration

> ← [Methods Overview](index.md) | [Constraints](constraints.md) →

Post-hoc calibration is the process of adjusting a trained model's uncertainty estimates so that **predicted confidence matches observed frequency**. These methods are applied to a held-out **calibration set** and require **no retraining** of the base model.

---

## The Calibration Gap

Modern neural networks are notoriously **overconfident** \[1\]. A model might predict a 95% confidence interval that only covers 70% of the true values. This "calibration gap" can lead to risky decision-making in safety-critical applications.

!!! success "The Goal"

    A model is **perfectly calibrated** if its predicted probability $\tau$ always corresponds to the actual observed frequency:
    $$P\bigl(Y \leq F^{-1}(\tau \mid X)\bigr) = \tau \quad \forall\,\tau \in [0, 1]$$

---

## Why Calibrate?

| Benefit | Description |
|:--------|:------------|
| **Trustworthiness** | Intervals mean what they say (e.g., 90% coverage means exactly 90%). |
| **Reliability** | Consistent performance across different regions of the input space. |
| **Decision Support** | Enables better risk-aware decision making (e.g., in medical or financial AI). |
| **No Retraining** | Calibrate any pre-trained model in seconds. |

---

## Available Calibrators

**torchregress** provides three main calibrators, each targeting a different type of miscalibration.

### 1. Variance Temperature Scaling ([VarianceTemperatureScaler](../api/calibration.md))

Rescales the predicted variance $\sigma^2$ by a single learned "temperature" $T$. This is the most popular method for fixing over/under-confidence while preserving the model's heteroscedasticity.

$$\sigma_{\text{cal}}^2(x) = T \cdot \sigma_{\text{pred}}^2(x)$$

```python
from torchregress.calibration import VarianceTemperatureScaler

scaler = VarianceTemperatureScaler()
scaler.fit(pred_mean_cal, pred_var_cal, y_cal)
calibrated_var = scaler.transform(pred_var_test)
```

!!! warning "Calibration Set Requirements & Risks"
    * **Independent Calibration Set**: The calibration dataset **must** be strictly held out from model training. If the model has seen the calibration data, its predicted variances $\sigma^2_{\text{pred}}$ will be artificially small relative to the residuals, forcing the scaler to converge to an excessively large temperature $T \gg 1$, which will over-inflate (make too wide) prediction intervals at test time.
    * **Covariate Representation**: The calibration set must share the same covariate distribution as the test set. Under covariate shift, a single global temperature $T$ may fail to calibrate variance uniformly across feature space.
    * **Single-T limitation**: `VarianceTemperatureScaler` applies one global scale factor $T$ to all predictions. If the model is overconfident in some input regions and underconfident in others, a single $T$ cannot fix both simultaneously — it will be a compromise. For region-dependent miscalibration, consider `PITCalibrator`.
    * **Small-set risks**: Non-parametric calibrators (`IsotonicMeanCalibrator`, `PITCalibrator`) can **overfit** the calibration set when it is small ($n < 500$), producing jagged mappings that don't generalize. For small calibration sets, prefer `VarianceTemperatureScaler` (1 parameter).


### 2. Isotonic Mean Calibration ([IsotonicMeanCalibrator](../api/calibration.md))

Corrects systematic **bias** in point predictions. If your model consistently over-predicts in some regions and under-predicts in others, isotonic regression learns a monotone mapping to fix it.

```python
from torchregress.calibration import IsotonicMeanCalibrator

cal = IsotonicMeanCalibrator()
cal.fit(mu_cal, y_cal)
calibrated_mu = cal.transform(mu_test)
```

### 3. PIT Calibration ([PITCalibrator](../api/calibration.md))

The most flexible non-parametric method \[2\]. It learns a monotone mapping from **PIT values** (how far predictive ranks deviate from uniform) to better-calibrated ranks. This can fix distributional miscalibration that a single global temperature cannot.

```python
from torchregress.calibration import PITCalibrator

cal = PITCalibrator()
pit_cal = PITCalibrator.pit_from_gaussian(pred_mean_cal, pred_std_cal, y_cal)
cal.fit(pit_cal)

# Remap PIT values on held-out data (e.g., for reliability / coverage diagnostics)
pit_test = PITCalibrator.pit_from_gaussian(pred_mean_test, pred_std_test, y_test)
calibrated_pit = cal.transform(pit_test)
```

---

## Comparison Matrix

| Method | Target | Parameters | API Reference | Best For |
|:-------|:-------|:-----------|:--------------|:---------|
| **Temperature** | Variance | 1 (Scalar) | [VarianceTemperatureScaler](../api/calibration.md) | Heteroscedastic Gaussian models |
| **Isotonic** | Mean Bias | Non-parametric | [IsotonicMeanCalibrator](../api/calibration.md) | Systematic point-prediction errors |
| **PIT** | PIT ranks | Non-parametric | [PITCalibrator](../api/calibration.md) | Any model with CDF / Gaussian predictive std |

---

## Best Practices

!!! tip "Calibration Set Size"

    For temperature scaling, a few hundred samples (e.g., $n=200$) are usually sufficient. For non-parametric methods like PIT or Isotonic, larger sets ($n > 500$) provide smoother and more reliable mappings.

!!! warning "Chaining Calibrators"

    You can chain calibrators (e.g., Mean → Variance → PIT). However, be careful not to **overfit** the calibration set if it is small.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Guo et al. ["On Calibration of Modern Neural Networks."](https://arxiv.org/abs/1706.04599) *ICML*, 2017. |
| 2 | Kuleshov et al. ["Accurate Uncertainties for Deep Learning Using Calibrated Regression."](https://arxiv.org/abs/1807.00263) *ICML*, 2018. |
| 3 | Levi et al. ["Evaluating and Calibrating Uncertainty Prediction in Regression Tasks."](https://arxiv.org/abs/1905.11659) *Sensors*, 2022. |

---

## Limitations

1. **Single global temperature**: `VarianceTemperatureScaler` applies one scale factor $T$ to all predictions. If the model is overconfident in some regions and underconfident in others, a single $T$ compromises. For region-dependent miscalibration, use `PITCalibrator`.
2. **Calibration set independence**: The calibration set must be strictly held out from model training. If the model has seen the calibration data, predicted variances will be artificially small, forcing $T \gg 1$ and over-inflating test-time intervals.
3. **Small calibration sets**: Non-parametric calibrators (`IsotonicMeanCalibrator`, `PITCalibrator`) can overfit when $n < 500$, producing jagged mappings. Prefer `VarianceTemperatureScaler` (1 parameter) for small sets.
4. **Covariate shift**: All calibrators assume the calibration and test sets share the same covariate distribution. Under shift, a fixed mapping learned on the calibration set may not transfer.
5. **Calibration is not coverage**: Post-hoc calibration improves agreement between predicted confidence and observed frequency, but provides no finite-sample coverage guarantee. For coverage guarantees, use [Conformal Prediction](conformal/index.md).

## Recommendations

- **Start with `VarianceTemperatureScaler`**: One parameter, fast to fit, and effective for the most common failure mode (global over/under-confidence).
- **Upgrade to `PITCalibrator`** when calibration errors vary across the input space or when the predictive distribution is non-Gaussian.
- **Use `IsotonicMeanCalibrator`** for systematic point-prediction bias (e.g., consistently over-predicting in low-target regions).
- **Calibration set size**: Target $n \ge 200$ for temperature scaling, $n \ge 500$ for non-parametric methods. See the [calibration comparison example](../examples/constraints_calibration_comparison.md).
- **Chain calibrators carefully**: Mean → Variance → PIT is valid but each step consumes degrees of freedom from the calibration set. Don't chain more than two non-parametric calibrators on small sets.
- **Validate after calibration**: Always re-evaluate calibration metrics (ECE, PIT) on a separate test set after applying calibrators. See [Calibration metrics](../metrics/calibration.md).

## Next Steps
- [Calibration Metrics](../metrics/calibration.md) — [`expected_calibration_error`](../api/metrics.md), [`marginal_calibration_error`](../api/metrics.md)
- [Conformal Prediction](../methods/conformal/index.md) (coverage guarantees vs post-hoc calibration)
- [Calibration comparison example](../examples/constraints_calibration_comparison.md)
- [Calibration API](../api/calibration.md) (calibrator classes)
