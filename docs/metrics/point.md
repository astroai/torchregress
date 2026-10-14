# Point Prediction Metrics

> ← [Metrics Overview](index.md) | [Interval Metrics](interval.md) →

Point prediction metrics evaluate the accuracy of single-point predictions without considering uncertainty information.

---

## Basic Metrics

### Mean Squared Error (MSE)

The average of squared differences between predictions and targets:

$$
\text{MSE}(y, \hat{y}) = \frac{1}{\sum_{i=1}^N w_i m_i} \sum_{i=1}^N w_i m_i (y_i - \hat{y}_i)^2
$$

where $m_i \in \{0, 1\}$ marks valid entries and $w_i > 0$ are the optional `sample_weight` values (all ones when omitted).

!!! note "No `mask` argument"
    The functional point metrics take `sample_weight`, not `mask` or `weights`. To exclude missing entries, index them out before the call, e.g. `y_pred[mask]`, `y_true[mask]`. `r2_score` accepts neither a mask nor weights.

```python
from torchregress.metrics.point import mean_squared_error

mse = mean_squared_error(y_pred, y_true, sample_weight=weights)
```
See also: [mean_squared_error](../api/metrics.md).

### Mean Absolute Error (MAE)

The average of absolute differences between predictions and targets:

$$
\text{MAE}(y, \hat{y}) = \frac{1}{\sum_{i=1}^N w_i m_i} \sum_{i=1}^N w_i m_i |y_i - \hat{y}_i|
$$

```python
from torchregress.metrics.point import mean_absolute_error

mae = mean_absolute_error(y_pred, y_true, sample_weight=weights)
```
See also: [mean_absolute_error](../api/metrics.md).

### Root Mean Squared Error (RMSE)

The square root of the mean squared error:

$$
\text{RMSE}(y, \hat{y}) = \sqrt{\text{MSE}(y, \hat{y})}
$$

```python
from torchregress.metrics.point import rmse

y_rmse = rmse(y_pred, y_true, sample_weight=weights)
```
See also: [rmse](../api/metrics.md).

### R² (Coefficient of Determination)

Measures the proportion of variance in the target that is predictable from the model:

$$
R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}
$$

where $\bar{y}$ is the target mean. `r2_score` is unweighted and has no mask; it wraps `torchmetrics.R2Score`.

```python
from torchregress.metrics.point import r2_score

r2 = r2_score(y_pred, y_true)
```
See also: [r2_score](../api/metrics.md).

### MAPE, MSLE and Explained Variance

`torchregress.metrics` does not provide functional forms of mean absolute
percentage error, mean squared log error or explained variance. Use the
`torchmetrics` functional equivalents:

```python
from torchmetrics.functional import (
    explained_variance,
    mean_absolute_percentage_error,
    mean_squared_log_error,
)

mape = mean_absolute_percentage_error(y_pred, y_true)
msle = mean_squared_log_error(y_pred, y_true)  # strictly positive values only
ev = explained_variance(y_pred, y_true)
```

---

## Robust Metrics

### Median Absolute Error

Median of absolute differences, robust to outliers:

$$
\text{MedAE}(y, \hat{y}) = \text{median}\left(\{|y_i - \hat{y}_i|\}_{i=1}^{N}\right)
$$

For an even $N$ the median is the mean of the two middle values, as in
`numpy.median` and `sklearn.metrics.median_absolute_error`. Multi-output
targets $[N, D]$ take the median per output and then average the $D$ medians
(`multioutput="uniform_average"`), or return them (`"raw_values"`).

```python
from torchregress.metrics.point import median_absolute_error

median_ae = median_absolute_error(y_pred, y_true)
per_output = median_absolute_error(y_pred, y_true, multioutput="raw_values")
```
See also: [median_absolute_error](../api/metrics.md).

### Huber Loss

Combines quadratic error for small residuals and linear error for large residuals:

$$
\text{Huber}(y, \hat{y}; \delta) = \frac{1}{\sum_{i=1}^N w_i m_i} \sum_{i=1}^N w_i m_i L_\delta(y_i - \hat{y}_i)
$$

$$
L_\delta(r) = \begin{cases} \frac{1}{2} r^2 & \text{if } |r| \le \delta \\ \delta |r| - \frac{1}{2} \delta^2 & \text{otherwise} \end{cases}
$$

```python
from torchregress.metrics.point import huber_loss

# delta controls the transition point from quadratic to linear error
hl = huber_loss(y_pred, y_true, delta=1.0, sample_weight=weights)
```
See also: [huber_loss](../api/metrics.md).

### Trimmed Mean Squared Error

MSE computed after removing the most extreme squared errors from both tails:

$$
\text{TrimmedMSE}(y, \hat{y}; \alpha) = \frac{1}{N - 2k} \sum_{j=k+1}^{N-k} r_{(j)}^2, \qquad k = \lfloor N\alpha \rfloor
$$

where $r_{(j)}^2$ are the sorted squared residuals. Exactly $k$ values are cut
from **each** tail, as in `scipy.stats.trim_mean`. Before 0.3 the upper cut was
$N - \lfloor N(1-\alpha) \rfloor$, which differs from $k$ when $N\alpha$ is not
an integer. The parameter $\alpha$ must satisfy $0 \le \alpha < 0.5$.

```python
from torchregress.metrics.point import trimmed_mean_squared_error

# trim 10% of data from each end (keeps the middle 80%)
tmse = trimmed_mean_squared_error(y_pred, y_true, proportion=0.1)
```
See also: [trimmed_mean_squared_error](../api/metrics.md).

### Median Absolute Deviation (MAD)

Median of absolute deviations from the median error, scaled by a consistency factor:

$$
\text{MAD}(e) = c \cdot \text{median}\left(\{|e_i - \text{median}(e)|\}_{i=1}^{N}\right)
$$

where $e_i = y_i - \hat{y}_i$ and $c = 1.4826$ by default (the Gaussian
consistency factor that makes MAD a consistent estimator of $\sigma$ for
normally distributed residuals). Both medians average the two middle values
for even $N$, so the result equals
`c * scipy.stats.median_abs_deviation(e)`.

```python
from torchregress.metrics.point import median_absolute_deviation

mad = median_absolute_deviation(y_pred, y_true, scale=1.4826)
```
See also: [median_absolute_deviation](../api/metrics.md).

### Normalized RMSE

RMSE normalized by a target-distribution scale parameter:

$$
\text{NRMSE}(y, \hat{y}) = \frac{\text{RMSE}(y, \hat{y})}{\text{scale}}
$$

where $\text{scale}$ can be the standard deviation (`std`), the range (`range`), the mean (`mean`), or the interquartile range (`iqr`) of $y$.

```python
from torchregress.metrics.point import normalized_rmse

nrmse = normalized_rmse(y_pred, y_true, normalization='std')
```
See also: [normalized_rmse](../api/metrics.md).

### Normalized Median Absolute Deviation

The photometric-redshift NMAD, a stateful metric:

$$
\text{NMAD} = 1.4826 \cdot \text{median}\left(|d_i - \text{median}(d)|\right),
\qquad d_i = \hat{y}_i - y_i \;\; \text{or} \;\; d_i = \frac{\hat{y}_i - y_i}{1 + y_i}
$$

The relative form is selected with `normalization="relative"`. Medians average
the two middle values for even $N$.

```python
from torchregress.metrics import NormalizedMedianAbsoluteDeviation

nmad = NormalizedMedianAbsoluteDeviation(normalization="relative")
nmad.update(z_pred, z_true)
print(nmad.compute())
```
See also: [NormalizedMedianAbsoluteDeviation](../api/metrics.md).

---

## Application-Specific Metrics

### Outlier Fraction

Fraction of predictions whose scaled absolute error exceeds a threshold. It is
a stateful metric (`update` / `compute`), not a function:

$$
\text{OutlierFraction}(y, \hat{y}; \tau) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\frac{|y_i - \hat{y}_i|}{1 + y_i} > \tau\right)
$$

With `mode="relative"` (default) the error is scaled by $1 + y_i$, as above.
Any other `mode` scales by the global standard deviation of the targets seen so far.

```python
from torchregress.metrics import OutlierFraction

of = OutlierFraction(threshold=0.15, mode="relative")
of.update(y_pred, y_true)
print(of.compute())
```
See also: [OutlierFraction](../api/metrics.md).

### Tail Metrics

Evaluate point prediction accuracy specifically on extreme target regions:

$$
\text{TailMAE}(y, \hat{y}; q) = \frac{1}{|\mathcal{I}_q|} \sum_{i \in \mathcal{I}_q} |y_i - \hat{y}_i|
$$

$$
\text{TailRMSE}(y, \hat{y}; q) = \sqrt{\frac{1}{|\mathcal{I}_q|} \sum_{i \in \mathcal{I}_q} (y_i - \hat{y}_i)^2}
$$

where $\mathcal{I}_q$ is the set of indices where targets exceed the $q$-quantile (upper tail) or are below the $q$-quantile (lower tail).

```python
from torchregress.metrics import tail_mae, tail_rmse

# Top 10% target values
mae_tail = tail_mae(y_pred, y_true, quantile=0.9, tail="upper")
rmse_tail = tail_rmse(y_pred, y_true, quantile=0.9, tail="upper")
```
See also: [tail_mae](../api/metrics.md) and [tail_rmse](../api/metrics.md).

---

## Limitations

1. **Ignore uncertainty**: Point metrics (MSE, MAE, R²) evaluate only the point prediction. Two models with identical MSE can have vastly different uncertainty quality — one may be well-calibrated, the other overconfident. Always pair point metrics with distributional or interval metrics.
2. **Sensitive to outliers**: MSE and RMSE are dominated by the largest errors. Use MAE, Huber, or median-based metrics for robust evaluation when outliers are present.
3. **R² is not a goodness-of-fit test**: R² measures explained variance but does not validate model assumptions (normality, homoscedasticity, independence). A high R² with systematically miscalibrated uncertainty is a flawed model.
4. **MAPE instability**: MAPE is undefined when $y_i = 0$ and can be dominated by small true values ($y_i \approx 0$ produces enormous percentage errors). Use with caution; prefer MAE or RMSE for general-purpose evaluation.

## Recommendations

- **Always report multiple metrics**: MSE (or RMSE) + MAE gives a balanced view of typical and worst-case performance. Add R² for interpretability.
- **For outlier-heavy data**: Report median absolute error and MAD alongside mean-based metrics.
- **For imbalanced targets**: Use tail metrics (`tail_mae`, `tail_rmse`) to evaluate performance on extreme target regions. See [Imbalanced regression](../losses/imbalanced.md).
- **[regression_metrics_report](../api/metrics.md)** provides a comprehensive dict of all point metrics in one call.

## References

| # | Reference |
|:-:|:----------|
| 1 | Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73–101. |
| 2 | Theil, H. (1958). *Economic Forecasts and Policy*. North-Holland. |
| 3 | Willmott, C. J. & Matsuura, K. (2005). Advantages of the Mean Absolute Error over the Root Mean Square Error. *Atmospheric Research*, 80(1), 79–93. |
| 4 | Gneiting, T. & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *JASA*, 102(477), 359–378. |

## Next steps

- [Interval metrics](interval.md) — evaluate prediction interval coverage and width alongside point accuracy
- [Distributional metrics](distribution.md) — proper scoring rules (CRPS, NLL) for probabilistic forecasts
- [Calibration metrics](calibration.md) — check whether predicted uncertainty matches observed frequency
- [Visualization diagnostics](../methods/visualization.md) — residual plots, Q-Q, and binned-metric diagnostics

---

## Comprehensive Reporting

### Regression Metrics Report

Generate a comprehensive dictionary report of point metrics.

```python
from torchregress.metrics.point import regression_metrics_report

report = regression_metrics_report(y_pred, y_true, sample_weight=weights)
```
See also: [regression_metrics_report](../api/metrics.md).
