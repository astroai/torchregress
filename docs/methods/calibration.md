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

**torchregress** provides four main calibrators, each targeting a different type of miscalibration.

### 1. Variance Temperature Scaling ([VarianceTemperatureScaler](../api/calibration.md))

Rescales the predicted variance $\sigma^2$ by a single learned "temperature" $T$. This is the most popular method for fixing over/under-confidence while preserving the model's heteroscedasticity.

$$\sigma_{\text{cal}}^2(x) = T \cdot \sigma_{\text{pred}}^2(x)$$

```python
from torchregress.calibration import VarianceTemperatureScaler

scaler = VarianceTemperatureScaler()
scaler.fit(pred_mean_cal, pred_var_cal, y_cal)
calibrated_var = scaler.transform(pred_var_test)
```

Three optional extensions, all off by default:

* `fit_floor=True` also fits an additive variance floor $f \ge 0$, giving
  $\sigma_{\text{cal}}^2 = T\,\sigma_{\text{pred}}^2 + f$. Use it when the
  error budget has a component the model does not see (systematics, template
  mismatch). A single $T$ then over-covers the noisiest samples and
  under-covers the best-measured ones.
* `target_var=...` gives per-sample label variances (noisy ground truth). The
  fit uses $T\,\sigma_{\text{pred}}^2 + f + \sigma_{\text{target}}^2$, so
  label noise is not charged to the model. `transform` returns the model's
  part only.
* `clip=5.0` ignores calibration samples more than 5 robust standard
  deviations from the median residual, so catastrophic failures do not set
  the scale.

```python
scaler.fit(mu_cal, var_cal, y_cal, target_var=label_var_cal, fit_floor=True, clip=5.0)
```

Without `fit_floor`, `target_var` or a preset `variance_floor`, the NLL
optimum has a closed form and `fit` returns it exactly (no iterations):

$$\hat T = \frac{1}{n}\sum_{i=1}^{n} \frac{(y_i - \mu_i)^2}{\sigma^2_{\text{pred},i}}$$

With a floor or noisy targets, the closed-form value seeds the Adam
optimisation over $\log T$ (and $\log f$).

!!! info "Temperature bounds (changed in 0.3)"
    $T$ is unbounded by default. Earlier versions silently clamped it to
    $[0.05, 20]$, which left strongly over-confident models under-covered (a
    model needing $T \approx 50$ got $T = 20$, so nominal 90% intervals
    covered about 71%). To constrain $T$ deliberately, pass
    `temperature_bounds=(t_min, t_max)`. For example,
    `temperature_bounds=(0.05, 20.0)` reproduces the old behaviour. A fitted
    $T$ far outside $[0.1, 10]$ usually means a leaked or mismatched
    calibration split (see the warning below), not a scale the model needs.

!!! tip "The `clip` trade-off"
    `clip` fits the variance *without* the outliers, so outliers that turn up
    at test time are scored against a narrower predictive distribution. In
    the torchregress-harness study on synthetic data with catastrophic
    outliers, `clip=5` gave the best CRPS but raised the test NLL from 1.64
    to 2.71. Use `clip` when you care about sharp intervals for the bulk
    (CRPS, coverage of good fits). Leave it off when the log-likelihood of
    every test point matters, or model the outliers explicitly (mixture or
    Student-t likelihood).

!!! warning "Calibration Set Requirements & Risks"
    * **Independent Calibration Set**: The calibration dataset **must** be strictly held out from model training. If the model has seen the calibration data, its predicted variances $\sigma^2_{\text{pred}}$ will be artificially small relative to the residuals, forcing the scaler to converge to an excessively large temperature $T \gg 1$, which will over-inflate (make too wide) prediction intervals at test time.
    * **Covariate Representation**: The calibration set must share the same covariate distribution as the test set. Under covariate shift, a single global temperature $T$ may fail to calibrate variance uniformly across feature space.
    * **Single-T limitation**: `VarianceTemperatureScaler` applies one global scale factor $T$ to all predictions. If the model is overconfident in some input regions and underconfident in others, a single $T$ cannot fix both simultaneously — it will be a compromise. For region-dependent miscalibration, consider `PITCalibrator`.
    * **Small-set risks**: Non-parametric calibrators (`IsotonicMeanCalibrator`, `PITCalibrator`) can **overfit** the calibration set when it is small ($n < 500$), producing jagged mappings that don't generalize. For small calibration sets, prefer `VarianceTemperatureScaler` (1 parameter).


### 2. Isotonic Mean Calibration ([IsotonicMeanCalibrator](../api/calibration.md))

Corrects systematic **bias** in point predictions. If your model consistently over-predicts in some regions and under-predicts in others, isotonic regression learns a monotone mapping to fix it.

The fit is the least-squares non-decreasing map

$$\hat g = \arg\min_{g \text{ non-decreasing}} \sum_{i=1}^{n} \big(y_i - g(\mu_i)\big)^2,$$

solved by the Pool Adjacent Violators Algorithm. Targets that share a
prediction are averaged first. Each fitted block is stored by its first and
last prediction, so `transform` is constant on a block and linear between
blocks. The output matches `sklearn.isotonic.IsotonicRegression(out_of_bounds="clip")`
to round-off. Versions before 0.3 interpolated between block centroids and did
not pool ties, so they did not return the isotonic fit on the training points.

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

### 4. Variance-Restoring / Distribution-Matching Calibration ([VarianceRestoringCalibrator](../api/calibration.md))

The first three calibrators adjust **conditional** quantities: the mean, the variance, or the PIT distribution of an individual prediction given its input. None of them restores the **marginal** distribution of the targets. Yet a predictor trained with mean-squared error estimates $\mathbb E[y\mid x]$, and by the law of total variance

$$\operatorname{Var}(y) = \operatorname{Var}\bigl(\mathbb E[y\mid x]\bigr) + \mathbb E\bigl[\operatorname{Var}(y\mid x)\bigr],$$

its predictions are generically **under-dispersed**: the conditional-mean map compresses extreme targets (e.g. $\mathrm{[Fe/H]} < -2.5$) toward the bulk and has a smaller variance than $y$. This calibrator restores the marginal distribution of the point predictions, with two maps:

* **`method="affine"`** — moment matching. With prior mean $\mu_0$ (default $\overline{y}$) and

  $$s = \sqrt{\frac{\operatorname{Var}(y)}{\operatorname{Var}(\hat\mu)}}, \qquad T(z) = \mu_0 + s\,(z - \mu_0).$$

* **`method="quantile"`** — 1-D optimal transport. Sorting the calibration predictions and targets and interpolating between the ordered pairs gives the monotone map that minimises the Wasserstein-2 distance between the empirical marginals, i.e. it reproduces the calibration target marginal exactly (up to interpolation); outside the calibration range it clips (`out_of_bounds="clip"`, default) or extrapolates linearly (`"linear"`).

```python
import torch
from torchregress.calibration import VarianceRestoringCalibrator

# A deliberately shrunk predictor: pred = 0.6 * y + noise.
torch.manual_seed(0)
y_cal = torch.randn(2000)
pred_cal = 0.6 * y_cal + 0.2 * torch.randn(2000)
y_test = torch.randn(2000)
pred_test = 0.6 * y_test + 0.2 * torch.randn(2000)

restorer = VarianceRestoringCalibrator(method="quantile")
restorer.fit(pred_cal, y_cal)
restored = restorer.transform(pred_test)
print(restorer.describe(pred_test, y_test))
# var_ratio_before ~ 0.35, var_ratio_after ~ 1.0,
# mse_after < mse_before (the predictor is attenuated),
# wasserstein1_after < wasserstein1_before
```

!!! warning "Marginal matching is not conditional calibration"
    For a *perfectly calibrated* MSE-optimal predictor ($\mathbb E[y\mid z] = z$), the missing spread is irreducible target noise: stretching restores the marginal variance and tails but **increases** the MSE. For a shrunk/attenuated predictor ($\hat y = \rho y + \text{noise}$ with $\rho < 1$) it can reduce both the Wasserstein-1 distance and the MSE. There is no unconditional improvement guarantee, so always report both the point error and the marginal diagnostics (`describe` returns `mse_before`, `mse_after`, `mse_ratio`, `wasserstein1_before/after`, `var_ratio_before/after`).

    Use this calibrator when the **population distribution** of predictions is the quantity of interest — e.g. population-level metallicity studies, extreme-value recovery, or matching an external reference distribution — and **not** as a substitute for fixing conditional bias with [isotonic calibration](#2-isotonic-mean-calibration-isotonicmeancalibrator) or the mean calibration methods above.

    A per-sample heteroscedastic rescaling $\sigma\sqrt{1 + \sigma^2/\operatorname{Var}(\hat\mu)}$ was considered and intentionally not implemented: it is an ad-hoc rule rather than the solution of a transport or moment-matching objective, and it diverges where the predicted-mean variance is small. Global moment matching and the 1-D OT map are the principled versions of the same intuition.

---

## Comparison Matrix

| Method | Target | Parameters | API Reference | Best For |
|:-------|:-------|:-----------|:--------------|:---------|
| **Temperature** | Variance | 1 (Scalar) | [VarianceTemperatureScaler](../api/calibration.md) | Heteroscedastic Gaussian models |
| **Isotonic** | Mean Bias | Non-parametric | [IsotonicMeanCalibrator](../api/calibration.md) | Systematic point-prediction errors |
| **PIT** | PIT ranks | Non-parametric | [PITCalibrator](../api/calibration.md) | Any model with CDF / Gaussian predictive std |
| **Variance-restoring** | Marginal distribution | 2 (affine) / non-parametric (quantile) | [VarianceRestoringCalibrator](../api/calibration.md) | Population/tail distribution, attenuated predictions |

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
| 4 | Villani, C. *Optimal Transport: Old and New.* Springer, 2009; Peyré & Cuturi, ["Computational Optimal Transport."](https://arxiv.org/abs/1803.00567) *Foundations and Trends in ML*, 2019. |
| 5 | Casella, G., & Berger, R. L. *Statistical Inference* (2nd ed.). Duxbury, 2002. (Law of total variance.) |

---

## Limitations

1. **Single global temperature**: `VarianceTemperatureScaler` applies one scale factor $T$ to all predictions. If the model is overconfident in some regions and underconfident in others, a single $T$ compromises. For region-dependent miscalibration, use `PITCalibrator`.
2. **Calibration set independence**: The calibration set must be strictly held out from model training. If the model has seen the calibration data, predicted variances will be artificially small, forcing $T \gg 1$ and over-inflating test-time intervals.
3. **Small calibration sets**: Non-parametric calibrators (`IsotonicMeanCalibrator`, `PITCalibrator`) can overfit when $n < 500$, producing jagged mappings. Prefer `VarianceTemperatureScaler` (1 parameter) for small sets.
4. **Covariate shift**: All calibrators assume the calibration and test sets share the same covariate distribution. Under shift, a fixed mapping learned on the calibration set may not transfer.
5. **Calibration is not coverage**: Post-hoc calibration improves agreement between predicted confidence and observed frequency, but provides no finite-sample coverage guarantee. For coverage guarantees, use [Conformal Prediction](conformal/index.md).
6. **Marginal calibration is not conditional calibration**: `VarianceRestoringCalibrator` matches the distribution of predictions to the target distribution; it can increase the MSE of an already conditional-mean-calibrated predictor. Use it when the population distribution (especially the tails) is what matters, and check `describe` diagnostics on a separate split.

## Recommendations

- **Start with `VarianceTemperatureScaler`**: One parameter, fast to fit, and effective for the most common failure mode (global over/under-confidence).
- **Upgrade to `PITCalibrator`** when calibration errors vary across the input space or when the predictive distribution is non-Gaussian.
- **Use `IsotonicMeanCalibrator`** for systematic point-prediction bias (e.g., consistently over-predicting in low-target regions).
- **Calibration set size**: Target $n \ge 200$ for temperature scaling, $n \ge 500$ for non-parametric methods. See the [calibration comparison example](../examples/constraints_calibration_comparison.md).
- **Chain calibrators carefully**: Mean → Variance → PIT is valid but each step consumes degrees of freedom from the calibration set. Don't chain more than two non-parametric calibrators on small sets.
- **Use `VarianceRestoringCalibrator` only for marginal/tail restoration**: choose `method="affine"` for a two-parameter moment match, `method="quantile"` for an exact marginal shape (1-D optimal transport). Inspect `describe(...)` before deploying; it reports the point-error cost next to the marginal gain.
- **Validate after calibration**: Always re-evaluate calibration metrics (ECE, PIT) on a separate test set after applying calibrators. See [Calibration metrics](../metrics/calibration.md).

## Next Steps
- [Calibration Metrics](../metrics/calibration.md) — [`expected_calibration_error`](../api/metrics.md), [`marginal_calibration_error`](../api/metrics.md)
- [Conformal Prediction](../methods/conformal/index.md) (coverage guarantees vs post-hoc calibration)
- [Calibration comparison example](../examples/constraints_calibration_comparison.md)
- [Calibration API](../api/calibration.md) (calibrator classes)
