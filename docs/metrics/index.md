# Evaluation Metrics

torchregress provides a comprehensive set of metrics for evaluating regression
models, with special focus on uncertainty quantification. This page is the
**catalogue**: every metric category with links to full guides. For method
selection guidance ("which metric for my task?"), see the [selection
guide](#metric-selection-guide) at the bottom.

## How to read a metric

A regression model does not emit one object, so no single number can score it. The four forecast objects this library evaluates are:

| Forecast | Notation | Scored by |
|:---------|:---------|:----------|
| Point prediction | $\hat{y}(x)$ | [Point metrics](point.md) |
| Prediction interval | $[L(x), U(x)]$ | [Interval metrics](interval.md) |
| Predictive distribution | $F(y \mid x)$ (parametric or sampled) | [Distribution metrics](distribution.md) |
| Ensemble of members | $\{y^{(m)}(x)\}_{m=1}^{M}$ | [Ensemble metrics](ensemble.md) |

The guiding principle is the **proper scoring rule**. A score $S(F, y)$ is *proper* if the true distribution is optimal in expectation,

$$
\mathbb{E}_{Y \sim G}\big[S(G, Y)\big] \;\le\; \mathbb{E}_{Y \sim G}\big[S(F, Y)\big] \quad \text{for every forecast } F,
$$

and *strictly proper* if equality holds only when $F = G$. Squared error is a consistent scoring function for the conditional mean, absolute error for the median, and pinball for a quantile; CRPS and the energy score are proper for the whole distribution. Summaries that do not score the full distribution — a point-only MAE or $R^2$, or coverage without width — are still useful diagnostics, but they can be gamed and should not be the only reported number.

Calibration and sharpness are the two halves of forecast quality: calibration says *the stated probabilities are honest* ($P(Y \le F^{-1}(\tau \mid X)) = \tau$ for all $\tau$), while sharpness says *the forecasts are informative* (narrow intervals, concentrated densities). Proper scores reward both; a calibrated but useless forecast and a sharp but lying forecast both score badly. This is why the library pairs every distributional metric with a calibration diagnostic.

## torchregress vs torchmetrics — why both exist

Torchregress no longer re-exports `MeanSquaredError`, `MeanAbsoluteError`, or
`R2Score` from torchmetrics.  **Import them directly from `torchmetrics`:**

```python
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
```

| torchregress | torchmetrics nearest | Why torchregress exists |
|:-------------|:---------------------|:------------------------|
| `mean_squared_error`, `rmse`, `mean_absolute_error` | `MeanSquaredError`, `MeanAbsoluteError` | **Per-sample** `reduction`, `sample_weight` support, `tail_rmse` / `tail_mae` tail-focus variants |
| `r2_score` | `R2Score` | Functional form (NumPy or tensor input) |
| `huber_loss` | `HuberLoss` | Functional form with `sample_weight` support |

**The remaining metrics — well over 100 exported names — have no torchmetrics equivalent.**  They cover
calibration, CRPS/EnergyScore, prediction intervals, ensemble uncertainties,
OOD detection, censored/ordinal regression, selective prediction, and more.

!!! tip "Rule of thumb"
    - Use **torchmetrics classes** (`torchmetrics.MeanSquaredError`, `torchmetrics.MeanAbsoluteError`, `torchmetrics.R2Score`)
  when you want stateful metric accumulation across epochs.
    - Use **torchregress functional metrics** (`mean_squared_error`, `rmse`, `mean_absolute_error`, `r2_score`)
      when you need per-sample outputs, tail-focus, or `sample_weight` support.
    - Use **torchregress metric classes** ([`ExpectedCalibrationError`](../api/metrics.md), [`MarginalCalibrationError`](../api/metrics.md), …)
      for domain-specific evaluation that torchmetrics does not cover.

---

## Metric Categories

### Point Prediction Metrics

Basic accuracy measures for point predictions — use these as secondary
summaries alongside proper scoring rules.

[Learn more about point metrics →](point.md)

### Ordinal Metrics

Accuracy and agreement for ordered categorical targets.

[Learn more about ordinal metrics →](ordinal.md)

### Censored Metrics

Metrics for censored and interval-censored targets (survival analysis, sensor limits).

[Learn more about censored metrics →](censored.md)

### Distribution Metrics

Proper scoring rules for evaluating full predictive distributions — CRPS, NLL, energy score, PIT.

[Learn more about distribution metrics →](distribution.md)

### Interval Metrics

Coverage and width metrics for prediction intervals — PICP, MPIW, interval score.

[Learn more about interval metrics →](interval.md)

### Calibration Metrics

ECE, MCE, calibration score, and bias — validate that your uncertainty estimates are honest.
For simulation-based-inference posteriors, the same page also covers SBC rank uniformity and
TARP coverage diagnostics (`simulation_based_calibration_ranks`, `sbc_uniformity_pvalue`,
`tarp_coverage`).

[Learn more about calibration metrics →](calibration.md)

### Out-of-Distribution Detection Metrics

Scores for flagging shifted or anomalous inputs — Mahalanobis, typicality, entropy, KDE.

[Learn more about OOD metrics →](ood.md)

### Ensemble Metrics

Member-level statistics and uncertainty decomposition for ensemble models.

[Learn more about ensemble metrics →](ensemble.md)

### Multivariate Metrics

Vector-output RMSE and MAE for multi-target regression, plus task-agnostic correlations.

[Learn more about multivariate metrics →](multivariate.md)

### Decision Metrics

Selective prediction: risk-coverage curves (AURC) and rejection policies.

[Learn more about decision metrics →](decision.md)

### Representation Geometry

Frozen-feature diagnostics — neighbourhood target gap, target ordering, and readout quality — for comparing encoders independently of the head.

[Learn more about representation metrics →](representation.md)

### Photo-z PDF Diagnostics

Global and local photo-$z$ PDF calibration: PIT Cramér–von Mises, local PIT Wasserstein distance, and central PDF width, scaled by $1+z$.

[Learn more about photo-z metrics →](photo_z.md)

## Metric Selection Guide

| If you need to evaluate... | Consider using... |
|---------------------------|-------------------|
| Point prediction accuracy | `rmse`, `mae`, `r2_score` |
| Prediction intervals | `prediction_interval_coverage_probability`, `MeanPredictionIntervalWidth`, `interval_score` |
| Full predictive distributions | `gaussian_nll`, `crps_gaussian`, `energy_score` |
| Model calibration | `expected_calibration_error`, `marginal_calibration_error`, `calibration_score` |
| OOD scoring | `mahalanobis_distance`, `typicality_score`, `entropy_score`, `kernel_density_score` |
| Selective prediction | `RiskCoverageCurve`, `RejectionPolicy`, [`risk_coverage_curve`](../api/metrics.md) |

For detailed guidance on metric selection and interpretation, see the [practical usage guide](../guide/practical-usage.md) and [uncertainty decomposition](../guide/uncertainty-decomposition.md).
