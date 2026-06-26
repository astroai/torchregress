# Evaluation Metrics

torchregress provides a comprehensive set of metrics for evaluating regression
models, with special focus on uncertainty quantification. This page is the
**catalogue**: every metric category with links to full guides. For method
selection guidance ("which metric for my task?"), see the [selection
guide](#metric-selection-guide) at the bottom.

## torchregress vs torchmetrics — why both exist

Torchregress no longer re-exports `MeanSquaredError`, `MeanAbsoluteError`, or
`R2Score` from torchmetrics.  **Import them directly from `torchmetrics`:**

```python
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
```

| torchregress | torchmetrics nearest | Why torchregress exists |
|:-------------|:---------------------|:------------------------|
| `mse`, `rmse`, `mae` | `MeanSquaredError`, `MeanAbsoluteError` | **Per-sample** mode, **mask** support, `tail_rmse` / `tail_mae` tail-focus variants |
| `r2_score` | `R2Score` | Per-sample + mask support |
| `huber_loss` | `HuberLoss` | Functional form with mask support |
| `explained_variance_score` | `ExplainedVariance` | Mask support |
| `mean_absolute_percentage_error` | `MeanAbsolutePercentageError` | Functional form |
| `mean_squared_log_error` | `MeanSquaredLogError` | Functional form |

**The remaining ~50 metrics have no torchmetrics equivalent.**  They cover
calibration, CRPS/EnergyScore, prediction intervals, ensemble uncertainties,
OOD detection, censored/ordinal regression, selective prediction, and more.

!!! tip "Rule of thumb"
    - Use **torchmetrics classes** (`torchmetrics.MeanSquaredError`, `torchmetrics.MeanAbsoluteError`, `torchmetrics.R2Score`)
  when you want stateful metric accumulation across epochs.
    - Use **torchregress functional metrics** (`mse`, `rmse`, `mae`, `r2_score`)
      when you need per-sample outputs, tail-focus, or mask support.
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

[Learn more about calibration metrics →](calibration.md)

### Out-of-Distribution Detection Metrics

Scores for flagging shifted or anomalous inputs — Mahalanobis, typicality, entropy, KDE.

[Learn more about OOD metrics →](ood.md)

### Ensemble Metrics

Member-level statistics and uncertainty decomposition for ensemble models.

[Learn more about ensemble metrics →](ensemble.md)

### Multivariate Metrics

Vector-output RMSE and MAE for multi-target regression.

[Learn more about multivariate metrics →](multivariate.md)

### Decision Metrics

Selective prediction: risk-coverage curves (AURC) and rejection policies.

[Learn more about decision metrics →](decision.md)

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
