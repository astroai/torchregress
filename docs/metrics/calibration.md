# Calibration Metrics

Calibration metrics evaluate whether predicted probabilistic distributions or intervals accurately reflect the true uncertainty in the data.

→ See [Calibration methods](../methods/calibration.md), [Visualization diagnostics](../methods/visualization.md#1-diagnostic-plots-torchregressvizdiagnostic), and the [calibration API](../api/metrics.md#calibration-metrics-metricscalibration).

---

## Expected Calibration Error (ECE)

ECE measures the discrepancy between predicted quantile levels and their empirical observed coverage:

$$
\text{ECE} = \frac{1}{Q} \sum_{q \in \mathcal{Q}} |q - \hat{p}(q)|
$$

where $\mathcal{Q} = \{q_1, \dots, q_Q\}$ is the set of predicted quantile levels (e.g. $[0.1, 0.5, 0.9]$), and $\hat{p}(q)$ is the observed proportion of targets that fall below the predicted $q$-quantile $\hat{y}_{i, q}$:

$$
\hat{p}(q) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i \le \hat{y}_{i, q})
$$

The Maximum Calibration Error (MCE) is the maximum absolute discrepancy across quantiles:

$$
\text{MCE} = \max_{q \in \mathcal{Q}} |q - \hat{p}(q)|
$$

```python
from torchregress.metrics.calibration import expected_calibration_error

# Using a dictionary of quantile predictions
quantiles = {0.1: q10_pred, 0.5: q50_pred, 0.9: q90_pred}

# Calculate basic ECE metrics
ece_metrics = expected_calibration_error(quantiles, y_true)
print(f"Mean Absolute Calibration Error: {ece_metrics['mean_absolute_calibration_error']}")
print(f"Maximum Calibration Error: {ece_metrics['maximum_calibration_error']}")
```
See also: [expected_calibration_error](../api/metrics.md#expected_calibration_error) and the stateful [`ExpectedCalibrationError`](../api/metrics.md#expectedcalibrationerror) metric class.

---

## Marginal Calibration Error (MCE)

Marginal Calibration Error evaluates how well the predictive distribution's marginal ECDF matches the empirical ECDF of true observations.

For $B$ bins spanning the range of true and predicted targets with edges $b_1, \dots, b_{B+1}$, the observed target CDF at bin edge $k$ is:

$$
F_{\text{obs}}(b_{k+1}) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i < b_{k+1})
$$

Given $S$ Monte Carlo predictions per test point $\hat{y}_i^{(s)}$, the mean predicted CDF is:

$$
\bar{F}_{\text{pred}}(b_{k+1}) = \frac{1}{S \cdot N} \sum_{s=1}^S \sum_{i=1}^N \mathbb{I}(\hat{y}_i^{(s)} < b_{k+1})
$$

The Marginal Calibration Error is the average absolute CDF difference over bins:

$$
\text{MCE}_{\text{marginal}} = \frac{1}{B} \sum_{k=1}^B |F_{\text{obs}}(b_{k+1}) - \bar{F}_{\text{pred}}(b_{k+1})|
$$

The Maximum MCE is:

$$
\text{MaxMCE}_{\text{marginal}} = \max_{k} |F_{\text{obs}}(b_{k+1}) - \bar{F}_{\text{pred}}(b_{k+1})|
$$

```python
from torchregress.metrics.calibration import marginal_calibration_error

# y_pred_samples has shape [n_samples, batch_size]
mce_metrics = marginal_calibration_error(y_pred_samples, y_true)
print(f"Marginal Calibration Error: {mce_metrics['marginal_calibration_error']}")
print(f"Maximum MCE: {mce_metrics['maximum_marginal_calibration_error']}")
```
See also: [marginal_calibration_error](../api/metrics.md#marginal_calibration_error) and the stateful [`MarginalCalibrationError`](../api/metrics.md#marginalcalibrationerror) metric class.

---

## Bias

Measures the mean signed prediction error (mean bias):

$$
\text{Bias}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)
$$

```python
from torchregress.metrics.calibration import bias

mean_bias = bias(y_pred, y_true)
```
See also: [bias](../api/metrics.md#bias).

---

## Calibration Score

A convenience calibration score for Gaussian predictions. It builds a fine grid of 19 quantile levels from `pred_mean` and `pred_std` and computes the quantile ECE on that grid:

$$
\hat{y}_{i, q} = \hat{\mu}_i + z_q \cdot \hat{\sigma}_i
$$

where $z_q = \Phi^{-1}(q)$ is the inverse CDF of the standard normal distribution.

```python
from torchregress.metrics.calibration import calibration_score

# Get ECE computed over a fine normal quantile grid
score = calibration_score(y_true, pred_mean, pred_std)
```
See also: [calibration_score](../api/metrics.md#calibration_score).

---

## Comprehensive Reporting

### Calibration Metrics Report

Generate a comprehensive report of calibration metrics.

```python
from torchregress.metrics.calibration import calibration_metrics_report
from torch.distributions import Normal

# Using a PyTorch distribution
dist = Normal(mean_pred, std_pred)
report = calibration_metrics_report(dist, y_true)
```
See also: [calibration_metrics_report](../api/metrics.md#calibration_metrics_report).
