# Post-Hoc Calibration

Post-hoc calibration is the process of adjusting a trained model's uncertainty estimates so that **predicted confidence matches observed frequency**. These methods are applied to a held-out **calibration set** and require **no retraining** of the base model.

---

## The Calibration Gap

Modern neural networks are notoriously **overconfident** [1]. A model might predict a 95% confidence interval that only covers 70% of the true values. This "calibration gap" can lead to risky decision-making in safety-critical applications.

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

### 1. Variance Temperature Scaling ([`VarianceTemperatureScaler`](../api/calibration.md#torchregress.calibration.VarianceTemperatureScaler))

Rescales the predicted variance $\sigma^2$ by a single learned "temperature" $T$. This is the most popular method for fixing over/under-confidence while preserving the model's heteroscedasticity.

$$\sigma_{\text{cal}}^2(x) = T \cdot \sigma_{\text{pred}}^2(x)$$

```python
from torchregress.calibration import VarianceTemperatureScaler

scaler = VarianceTemperatureScaler()
scaler.fit(mu_cal, var_cal, y_cal)
calibrated_var = scaler.transform(var_test)
```

!!! warning "Calibration Set Requirements & Risks"
    * **Independent Calibration Set**: The calibration dataset **must** be strictly held out from model training. If the model has seen the calibration data, its predicted variances $\sigma^2_{\text{pred}}$ will be artificially small relative to the residuals, forcing the scaler to converge to an excessively large temperature $T \gg 1$, which will over-inflate (make too wide) prediction intervals at test time.
    * **Covariate Representation**: The calibration set must share the same covariate distribution as the test set. Under covariate shift, a single global temperature $T$ may fail to calibrate variance uniformly across feature space.
    * **Single-T limitation**: `VarianceTemperatureScaler` applies one global scale factor $T$ to all predictions. If the model is overconfident in some input regions and underconfident in others, a single $T$ cannot fix both simultaneously — it will be a compromise. For region-dependent miscalibration, consider `PITCalibrator`.
    * **Small-set risks**: Non-parametric calibrators (`IsotonicMeanCalibrator`, `PITCalibrator`) can **overfit** the calibration set when it is small ($n < 500$), producing jagged mappings that don't generalize. For small calibration sets, prefer `VarianceTemperatureScaler` (1 parameter).


### 2. Isotonic Mean Calibration ([`IsotonicMeanCalibrator`](../api/calibration.md#torchregress.calibration.IsotonicMeanCalibrator))

Corrects systematic **bias** in point predictions. If your model consistently over-predicts in some regions and under-predicts in others, isotonic regression learns a monotone mapping to fix it.

```python
from torchregress.calibration import IsotonicMeanCalibrator

cal = IsotonicMeanCalibrator()
cal.fit(mu_cal, y_cal)
calibrated_mu = cal.transform(mu_test)
```

### 3. PIT Calibration ([`PITCalibrator`](../api/calibration.md#torchregress.calibration.PITCalibrator))

The most powerful method [2]. It calibrates the **entire CDF** by remapping the Probability Integral Transform (PIT) values. This can fix complex distributional miscalibration that temperature scaling misses.

```python
from torchregress.calibration import PITCalibrator

cal = PITCalibrator()
# Fits a mapping from predicted PIT to uniform PIT
cal.fit_from_gaussian(mu_cal, std_cal, y_cal)
calibrated_dist = cal.transform_dist(mu_test, std_test)
```

---

## Comparison Matrix

| Method | Target | Parameters | API Reference | Best For |
|:-------|:-------|:-----------|:--------------|:---------|
| **Temperature** | Variance | 1 (Scalar) | [`VarianceTemperatureScaler`](../api/calibration.md#torchregress.calibration.VarianceTemperatureScaler) | Heteroscedastic Gaussian models |
| **Isotonic** | Mean Bias | Non-parametric | [`IsotonicMeanCalibrator`](../api/calibration.md#torchregress.calibration.IsotonicMeanCalibrator) | Systematic point-prediction errors |
| **PIT** | Full CDF | Non-parametric | [`PITCalibrator`](../api/calibration.md#torchregress.calibration.PITCalibrator) | Any distributional model |

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

## Next Steps
- Learn about [Calibration Metrics](../metrics/calibration.md)
- Explore [Conformal Prediction](../methods/conformal/index.md) (an alternative to post-hoc calibration)
- View the [Calibration Example](../examples/constraints_calibration_comparison.md)
