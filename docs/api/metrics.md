# Metrics API

Complete reference for `torchregress.metrics`. Every metric, class, and report function is listed here. For the conceptual guides, see the [Metrics overview](../metrics/index.md) and the per-category pages.

> **Imports.** `MeanSquaredError`, `MeanAbsoluteError`, and `R2Score` are no longer re-exported from torchregress. Import them directly from `torchmetrics` if you want stateful, epoch-accumulating metrics. The torchregress `mse`, `mae`, `r2_score` etc. are **functional**, support **per-sample** returns, and add **mask** support.

---

## Point metrics (`metrics.point`)

| Symbol | Description |
|:-------|:------------|
| [`mse(y_pred, y, mask=…, weights=…)`](#mse) | Functional MSE with per-sample option. |
| [`rmse(y_pred, y, …)`](#rmse) | Functional RMSE. |
| [`mae(y_pred, y, …)`](#mae) | Functional MAE. |
| [`r2_score(y_pred, y, …)`](#r2_score) | Functional R². |
| [`huber_loss(y_pred, y, delta=…)`](#huber_loss) | Functional Huber. |
| `median_absolute_error` | MedAE. |
| `median_absolute_deviation` | MAD. |
| `mean_absolute_percentage_error` | MAPE. |
| `mean_squared_log_error` | MSLE. |
| `normalized_rmse` | RMSE normalised by `mean(|y|)`. |
| `trimmed_mean_squared_error` | Trimmed-MSE. |
| `attenuation_factor` | EIV attenuation-bias diagnostic. |
| `tail_rmse(y_pred, y, q=0.1)` | RMSE on the upper-`q` tail. |
| `tail_mae(y_pred, y, q=0.1)` | MAE on the upper-`q` tail. |
| `regression_metrics_report(y_pred, y, …)` | Aggregate report dict. |

### Function Details

#### `mse`

Functional Mean Squared Error with mask and weight support:

```python
mse(y_pred, target, mask=None, weights=None, reduction="mean")
```

$$
\text{MSE} = \frac{1}{\sum_{i=1}^N w_i m_i} \sum_{i=1}^N w_i m_i (y_i - \hat{y}_i)^2
$$

#### `rmse`

Functional Root Mean Squared Error:

```python
rmse(y_pred, target, mask=None, weights=None, reduction="mean")
```

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

#### `mae`

Functional Mean Absolute Error:

```python
mae(y_pred, target, mask=None, weights=None, reduction="mean")
```

$$
\text{MAE} = \frac{1}{\sum_{i=1}^N w_i m_i} \sum_{i=1}^N w_i m_i |y_i - \hat{y}_i|
$$

#### `r2_score`

Functional R² coefficient of determination:

```python
r2_score(y_pred, target, mask=None, weights=None)
```

$$
R^2 = 1 - \frac{\sum_{i=1}^N w_i m_i (y_i - \hat{y}_i)^2}{\sum_{i=1}^N w_i m_i (y_i - \bar{y}_w)^2}
$$

#### `huber_loss`

Functional Huber loss metric:

```python
huber_loss(y_pred, target, delta=1.0, mask=None, weights=None, reduction="mean")
```

$$
\text{Huber}(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \le \delta \\ \delta |r| - \frac{1}{2} \delta^2 & \text{otherwise} \end{cases}
$$

#### `mean_absolute_percentage_error`

Functional Mean Absolute Percentage Error:

```python
mean_absolute_percentage_error(y_pred, target, mask=None, weights=None, reduction="mean", eps=1e-8)
```

$$
\text{MAPE} = \frac{1}{\sum_i w_i m_i} \sum_i w_i m_i \frac{|y_i - \hat{y}_i|}{\max(|y_i|, \varepsilon)}
$$

#### `mean_squared_log_error`

Functional Mean Squared Log Error:

```python
mean_squared_log_error(y_pred, target, mask=None, weights=None, reduction="mean")
```

$$
\text{MSLE} = \frac{1}{\sum_i w_i m_i} \sum_i w_i m_i \left(\log(\max(y_i, 0) + 1) - \log(\max(\hat{y}_i, 0) + 1)\right)^2
$$

---

## Distributional metrics (`metrics.distribution`)

| Symbol | Description |
|:-------|:------------|
| [`continuous_ranked_probability_score`](#continuous_ranked_probability_score) | CRPS for Gaussian `(μ, σ)`. |
| `crps_from_samples(samples, y)` | Empirical CRPS. |
| [`energy_score(samples, y)`](#energy_score) | Energy score (multivariate). |
| [`gaussian_nll(y_pred, log_var, y)`](#gaussian_nll) | Diagonal Gaussian NLL. |
| [`probability_integral_transform(y_pred, y_pred_std, y)`](#probability_integral_transform) | PIT values. |
| `kolmogorov_smirnov_uniform_statistic(pit)` | KS-uniform statistic on PIT. |

### Function Details

#### `continuous_ranked_probability_score`

CRPS for a Gaussian predictive distribution $\mathcal{N}(\mu, \sigma^2)$:

```python
continuous_ranked_probability_score(mean, std, target, mask=None, reduction="mean")
```

$$
\text{CRPS}(y) = \sigma \left( \frac{y - \mu}{\sigma} \left( 2\Phi\left(\frac{y - \mu}{\sigma}\right) - 1 \right) + 2\phi\left(\frac{y - \mu}{\sigma}\right) - \frac{1}{\sqrt{\pi}} \right)
$$

where $\Phi$ and $\phi$ are the standard normal cumulative and probability density functions.

#### `energy_score`

Multivariate energy score evaluating sample distributions against targets:

```python
energy_score(samples, target, mask=None, reduction="mean")
```

$$
\text{EnergyScore}(Y, y) = \mathbb{E}\left[\|Y - y\|_2\right] - \frac{1}{2} \mathbb{E}\left[\|Y - Y'\|_2\right]
$$

where $Y, Y'$ are independent samples drawn from the predictive model.

#### `gaussian_nll`

Diagonal Gaussian negative log-likelihood:

```python
gaussian_nll(mean, log_var, target, mask=None, weights=None, reduction="mean")
```

$$
\mathcal{L}_i = \frac{1}{2} \log(2\pi \sigma_i^2) + \frac{(y_i - \mu_i)^2}{2\sigma_i^2}
$$

#### `probability_integral_transform`

Gaussian PIT values for calibration diagnostics:

```python
probability_integral_transform(y_pred, y_pred_std, y_true)
```

$$
u_i = \Phi\left(\frac{y_i - \mu_i}{\sigma_i}\right)
$$

where $\Phi$ is the standard normal CDF. Under perfect calibration, $\{u_i\}$ should be uniform on $[0, 1]$.

---

## Interval metrics (`metrics.interval`)

| Symbol | Description |
|:-------|:------------|
| [`interval_score(lower, upper, y, alpha=…)`](#interval_score) | Functional Winkler interval score. |
| [`IntervalScore`](#intervalscore) | Stateful Winkler interval score (torchmetrics-style). |
| [`prediction_interval_coverage_probability(lower, upper, y)`](#prediction_interval_coverage_probability) | Functional PICP; optional MPIW via `return_diagnostics=True`. |
| [`prediction_interval_coverage(...)`](#prediction_interval_coverage) | Alias for `prediction_interval_coverage_probability`. |
| [`PredictionIntervalCoverageProbability`](#predictionintervalcoverageprobability) | Stateful PICP accumulator. |
| [`MeanPredictionIntervalWidth`](#meanpredictionintervalwidth) | Stateful MPIW accumulator. |
| `interval_metrics_report(lower, upper, y, alpha)` | Aggregate report. |

### Function Details

#### `interval_score`

Winkler interval score at significance level $\alpha$:

```python
interval_score(lower, upper, target, alpha=0.1, reduction="mean")
```

$$
S_\alpha(L_i, U_i; y_i) = (U_i - L_i) + \frac{2}{\alpha} (L_i - y_i) \mathbb{I}(y_i < L_i) + \frac{2}{\alpha} (y_i - U_i) \mathbb{I}(y_i > U_i)
$$

#### `prediction_interval_coverage_probability`

Empirical Prediction Interval Coverage Probability (PICP):

```python
prediction_interval_coverage_probability(lower, upper, target, alpha=0.1, return_diagnostics=False)
```

$$
\text{PICP} = \frac{1}{\sum_{i=1}^N m_i} \sum_{i=1}^N m_i \mathbb{I}(L_i \le y_i \le U_i)
$$

MPIW is returned alongside PICP when `return_diagnostics=True`:

$$
\text{MPIW}(L, U) = \frac{1}{\sum_{i=1}^N m_i} \sum_{i=1}^N m_i (U_i - L_i)
$$

#### `IntervalScore`

Stateful Winkler interval score at significance level $\alpha$:

```python
from torchregress.metrics import IntervalScore

metric = IntervalScore(alpha=0.1)
metric.update(lower_bound, upper_bound, y_true)
score = metric.compute()
```

#### `PredictionIntervalCoverageProbability`

Stateful PICP accumulator:

```python
from torchregress.metrics import PredictionIntervalCoverageProbability

picp = PredictionIntervalCoverageProbability()
picp.update(lower_bound, upper_bound, y_true)
coverage = picp.compute()
```

#### `MeanPredictionIntervalWidth`

Stateful MPIW accumulator:

```python
from torchregress.metrics import MeanPredictionIntervalWidth

mpiw = MeanPredictionIntervalWidth()
mpiw.update(lower_bound, upper_bound)
width = mpiw.compute()
```

#### `prediction_interval_coverage`

Compatibility alias for [`prediction_interval_coverage_probability`](#prediction_interval_coverage_probability).

---

## Calibration metrics (`metrics.calibration`)

| Symbol | Description |
|:-------|:------------|
| [`expected_calibration_error(y_pred_quantiles, y)`](#expected_calibration_error) | ECE for quantile predictions. |
| [`ExpectedCalibrationError`](#expectedcalibrationerror) | Stateful torchmetrics-style ECE accumulator. |
| [`marginal_calibration_error(y_pred_samples, y)`](#marginal_calibration_error) | MCE (max-bin error). |
| [`MarginalCalibrationError`](#marginalcalibrationerror) | Stateful marginal calibration accumulator. |
| [`bias(y_pred, y)`](#bias) | Mean signed error. |
| [`calibration_score(y_pred, y_pred_std, y)`](#calibration_score) | Combined calibration quality. |
| `calibration_metrics_report(y_pred, y_pred_std, y)` | Aggregate report. |

### Function Details

#### `expected_calibration_error`

Quantile Expected Calibration Error (MACE):

```python
expected_calibration_error(y_pred_quantiles, target, return_diagnostics=False)
```

$$
\text{MACE} = \frac{1}{Q} \sum_{q \in \mathcal{Q}} |q - \hat{p}(q)|
$$

where $\hat{p}(q) = \frac{1}{N} \sum_i \mathbb{I}(y_i \le \hat{y}_{i, q})$.

#### `marginal_calibration_error`

Marginal Calibration Error for continuous predictions:

```python
marginal_calibration_error(y_pred_samples, target, n_bins=20)
```

$$
\text{MCE}_{\text{marginal}} = \frac{1}{B} \sum_{k=1}^B |F_{\text{obs}}(b_{k+1}) - \bar{F}_{\text{pred}}(b_{k+1})|
$$

#### `ExpectedCalibrationError`

Stateful `torchmetrics.Metric` wrapper around quantile calibration error. Accumulates batches via `update(y_pred_quantiles, y_true)` and returns MACE, RMSCE, and maximum calibration error from `compute()`.

#### `MarginalCalibrationError`

Stateful `torchmetrics.Metric` for marginal calibration error on sample-based predictive distributions. Accumulates via `update(y_pred_samples, y_true)` and returns marginal and maximum MCE from `compute()`.

#### `bias`

Mean prediction bias:

```python
bias(y_pred, target)
```

$$
\text{Bias} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)
$$

#### `calibration_score`

Evaluates calibration score on a Gaussian distribution using 19 quantile levels:

```python
calibration_score(y_true, pred_mean, pred_std)
```

---

## Ensemble metrics (`metrics.ensemble`)

| Symbol | Description |
|:-------|:------------|
| `ensemble_statistics(preds)` | Aggregate mean and variance. |
| [`uncertainty_decomposition(means, variances)`](#uncertainty_decomposition) | Law of Total Variance decomposition. |
| [`gaussian_nll_ensemble(means, variances, y)`](#gaussian_nll_ensemble) | NLL of ensemble predictions. |

### Function Details

#### `ensemble_statistics`

Aggregates ensemble member predictions to mean and sample variance:

```python
ensemble_statistics(predictions, dim=0)
```

$$
\bar{y}_i = \frac{1}{M} \sum_{m=1}^M y_i^{(m)}, \qquad
\text{Var}(y_i) = \frac{1}{M} \sum_{m=1}^M \left(y_i^{(m)} - \bar{y}_i\right)^2
$$

#### `uncertainty_decomposition`

Decomposes ensembled uncertainty:

```python
uncertainty_decomposition(means, variances, dim=0)
```

- Epistemic uncertainty: $\sigma^2_{\text{epistemic}}(x) = \frac{1}{M}\sum (\mu_m(x) - \bar{\mu}(x))^2$
- Aleatoric uncertainty: $\sigma^2_{\text{aleatoric}}(x) = \frac{1}{M}\sum \sigma^2_m(x)$
- Total uncertainty: $\sigma^2_{\text{total}}(x) = \sigma^2_{\text{epistemic}}(x) + \sigma^2_{\text{aleatoric}}(x)$

#### `gaussian_nll_ensemble`

Negative log-likelihood of ensemble predictions:

```python
gaussian_nll_ensemble(means, variances, target)
```

$$
\mathcal{L} = \frac{1}{2} \log(2\pi \sigma^2_{\text{total}}) + \frac{(y - \bar{\mu})^2}{2\sigma^2_{\text{total}}}
$$

---

## OOD metrics (`metrics.ood`)

| Symbol | Description |
|:-------|:------------|
| [`mahalanobis_distance(x, mean, cov)`](#mahalanobis_distance) | Distance measure. |
| [`typicality_score(model_output, x)`](#typicality_score) | Typicality test. |
| `entropy_score(samples)` | Entropy metric. |
| `kernel_density_score(x_test, x_reference)` | KDE score. |

### Function Details

#### `mahalanobis_distance`

Mahalanobis distance to representation centroid:

```python
mahalanobis_distance(x, mean, cov, reduction="none")
```

$$
d_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
$$

#### `typicality_score`

Typicality score under predictive normal distributions:

```python
typicality_score(model_output, target_x)
```

$$
T(X) = \left| -\frac{1}{N} \sum_{i=1}^N \log p(x_i) - H(p) \right|
$$

#### `entropy_score`

Shannon/differential entropy of predictive samples:

```python
entropy_score(samples, n_bins=50)
```

$$
\hat{H}(p) = -\sum_{k=1}^K \hat{p}_k \log \hat{p}_k
$$

where $\hat{p}_k$ is the empirical bin probability from histogram density estimation.

#### `kernel_density_score`

Kernel density estimate relative to a reference training set:

```python
kernel_density_score(x_test, x_reference, bandwidth=0.5)
```

$$
\hat{f}_h(x) = \frac{1}{R} \sum_{k=1}^R K_h(x - r_k), \qquad
K_h(d) = \frac{1}{(2\pi h^2)^{D/2}} \exp\left(-\frac{\|d\|^2}{2h^2}\right)
$$

---

## Additional metrics reference

### crps_gaussian

Closed-form CRPS calculation for a Gaussian predictive distribution:

$$
\text{CRPS}(y) = \sigma \left( \frac{y - \mu}{\sigma} \left( 2\Phi\left(\frac{y - \mu}{\sigma}\right) - 1 \right) + 2\phi\left(\frac{y - \mu}{\sigma}\right) - \frac{1}{\sqrt{\pi}} \right)
$$

### ensemble_statistics

Aggregates predictions to compute ensemble mean and sample variance:

$$
\bar{y}_i = \frac{1}{M} \sum_{m=1}^M y_i^{(m)}
$$

### ensemble_interval_bounds

Computes symmetric Gaussian prediction interval bounds for ensembles.

### ensemble_interval_metrics

Computes empirical coverage and Winkler interval score for ensemble intervals.

### entropy_score

Calculates the Shannon entropy of predicted distribution samples.

### explained_variance_score

Measures the proportion of variance explained by the model:

$$
\text{ExplainedVariance} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}
$$

### interval_metrics_report

Aggregates interval score, PICP, and MPIW statistics across multiple models.

### kernel_density_score

KDE score relative to reference samples:

$$
\hat{f}_h(x) = \frac{1}{R} \sum_{k=1}^R K_h(x - r_k)
$$

### median_absolute_deviation

$$
\text{MAD} = \text{median}(|e_i - \text{median}(e)|)
$$

### median_absolute_error

$$
\text{MedAE} = \text{median}(|y_i - \hat{y}_i|)
$$

### normalized_median_absolute_deviation

MAD normalized by target statistics for relative scale comparison.

### normalized_rmse

$$
\text{NRMSE} = \frac{\text{RMSE}}{\text{scale}}
$$

### ood_metrics_report

Aggregates Mahalanobis, Typicality, Entropy, and KDE scores in a dict.

### outlier_fraction

$$
\text{OutlierFraction} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(|y_i - \hat{y}_i| > \tau \cdot \text{scale})
$$

### regression_metrics_report

Generates a dictionary report of point metrics (MSE, MAE, R2, etc.).

### RejectionPolicy

Implements decision policies to reject/defer low-confidence predictions.

### RiskCoverageCurve

Evaluates risk-coverage curves for selective prediction.

### tail_mae

$$
\text{TailMAE} = \frac{1}{|\mathcal{I}_q|} \sum_{i \in \mathcal{I}_q} |y_i - \hat{y}_i|
$$

### trimmed_mean_squared_error

$$
\text{TrimmedMSE} = \frac{1}{M} \sum_{j=1}^{M} r_{(j)}^2
$$

### calibration_metrics_report

Aggregates calibration statistics (ECE, MACE, MCE) in a dict.

### tail_rmse

$$
\text{TailRMSE} = \sqrt{\frac{1}{|\mathcal{I}_q|} \sum_{i \in \mathcal{I}_q} (y_i - \hat{y}_i)^2}
$$

where $\mathcal{I}_q$ is the set of indices of the samples with targets in the upper $q$-quantile.

## Censored Metrics

### censoring_rate

$$
\text{CensoringRate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(c_i \neq 0)
$$

### observed_mae

$$
\text{ObservedMAE} = \frac{1}{\sum_{i=1}^N \mathbb{I}(c_i = 0)} \sum_{i: c_i = 0} |y_i - \hat{y}_i|
$$

### concordance_index

Harrell-style concordance index for survival and censored regression:

$$
C = \frac{\sum_{i < j} \mathbb{I}(y_i < y_j) \left( \mathbb{I}(\hat{y}_i < \hat{y}_j) + 0.5 \cdot \mathbb{I}(\hat{y}_i = \hat{y}_j) \right)}{\sum_{i < j} \mathbb{I}(y_i < y_j)}
$$

restricted to comparable pairs.

### interval_overlap_rate

Fraction of samples where predicted interval overlaps censored interval bounds.

## Ordinal Metrics

### quadratic_weighted_kappa

Quadratic Weighted Kappa (QWK) for ordinal agreement. Let $O_{j,k}$ be observed counts and $E_{j,k}$ expected counts under independence:

$$
E_{j,k} = \frac{\sum_a O_{j,a} \cdot \sum_b O_{b,k}}{N}, \qquad
W_{j,k} = \frac{(j - k)^2}{(K - 1)^2}
$$

$$
\kappa = 1 - \frac{\sum_{j,k} W_{j,k} O_{j,k}}{\sum_{j,k} W_{j,k} E_{j,k}}
$$

### mean_absolute_class_error

Mean Absolute Class Error for ordinal targets.

### ordinal_accuracy

Classification accuracy on ordinal categories.
