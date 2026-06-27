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

where $m_i \in \{0, 1\}$ is a boolean mask ($1$ for valid, $0$ for missing data) and $w_i > 0$ represents sample weights.

```python
from torchregress.metrics.point import mean_squared_error

mse = mean_squared_error(y_pred, y_true, mask=mask, weights=weights)
```
See also: [mean_squared_error](../api/metrics.md).

### Mean Absolute Error (MAE)

The average of absolute differences between predictions and targets:

$$
\text{MAE}(y, \hat{y}) = \frac{1}{\sum_{i=1}^N w_i m_i} \sum_{i=1}^N w_i m_i |y_i - \hat{y}_i|
$$

```python
from torchregress.metrics.point import mean_absolute_error

mae = mean_absolute_error(y_pred, y_true, mask=mask, weights=weights)
```
See also: [mean_absolute_error](../api/metrics.md).

### Root Mean Squared Error (RMSE)

The square root of the mean squared error:

$$
\text{RMSE}(y, \hat{y}) = \sqrt{\text{MSE}(y, \hat{y})}
$$

```python
from torchregress.metrics.point import rmse

y_rmse = rmse(y_pred, y_true, mask=mask, weights=weights)
```
See also: [rmse](../api/metrics.md).

### Mean Absolute Percentage Error (MAPE)

The average percentage difference between predictions and targets:

$$
\text{MAPE}(y, \hat{y}) = \frac{1}{\sum_{i=1}^N w_i m_i} \sum_{i=1}^N w_i m_i \frac{|y_i - \hat{y}_i|}{\max(|y_i|, \varepsilon)}
$$

```python
from torchregress.metrics.point import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_pred, y_true, mask=mask, weights=weights)
```
See also: [mean_absolute_percentage_error](../api/metrics.md).

### R² (Coefficient of Determination)

Measures the proportion of variance in the target that is predictable from the model:

$$
R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^N w_i m_i (y_i - \hat{y}_i)^2}{\sum_{i=1}^N w_i m_i (y_i - \bar{y}_w)^2}
$$

where $\bar{y}_w = \frac{\sum_i w_i m_i y_i}{\sum_i w_i m_i}$ is the weighted target mean.

```python
from torchregress.metrics.point import r2_score

r2 = r2_score(y_pred, y_true, mask=mask, weights=weights)
```
See also: [r2_score](../api/metrics.md).

### Explained Variance Score

Measures the proportion of variance explained by the model:

$$
\text{ExplainedVariance}(y, \hat{y}) = 1 - \frac{\text{Var}_w(y - \hat{y})}{\text{Var}_w(y)}
$$

where $\text{Var}_w(x)$ is the weighted sample variance of $x$.

```python
from torchregress.metrics.point import explained_variance_score

explained_var = explained_variance_score(y_pred, y_true, mask=mask, weights=weights)
```
See also: [explained_variance_score](../api/metrics.md).

### Mean Squared Log Error (MSLE)

Mean squared error after log transformation:

$$
\text{MSLE}(y, \hat{y}) = \frac{1}{\sum_{i=1}^N w_i m_i} \sum_{i=1}^N w_i m_i \left(\log(1 + y_i) - \log(1 + \hat{y}_i)\right)^2
$$

```python
from torchregress.metrics.point import mean_squared_log_error

# Note: Only works with strictly positive values
msle = mean_squared_log_error(y_pred, y_true, mask=mask, weights=weights)
```
See also: [mean_squared_log_error](../api/metrics.md).

---

## Robust Metrics

### Median Absolute Error

Median of absolute differences, robust to outliers:

$$
\text{MedAE}(y, \hat{y}) = \text{median}\left(\{|y_i - \hat{y}_i|\}_{i: m_i=1}\right)
$$

```python
from torchregress.metrics.point import median_absolute_error

median_ae = median_absolute_error(y_pred, y_true, mask=mask)
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
hl = huber_loss(y_pred, y_true, delta=1.0, mask=mask, weights=weights)
```
See also: [huber_loss](../api/metrics.md).

### Trimmed Mean Squared Error

MSE computed after removing the most extreme squared errors from both tails:

$$
\text{TrimmedMSE}(y, \hat{y}; \alpha) = \frac{1}{M} \sum_{j=\lfloor N\alpha \rfloor + 1}^{\lfloor N(1-\alpha) \rfloor} r_{(j)}^2
$$

where $r_{(j)}^2$ are the sorted squared residuals for valid elements, and $M = \lfloor N(1-\alpha) \rfloor - \lfloor N\alpha \rfloor$ for a trimming fraction $\alpha$ applied to **each** tail. The parameter $\alpha$ must satisfy $0 \le \alpha < 0.5$.

```python
from torchregress.metrics.point import trimmed_mean_squared_error

# trim 10% of data from each end (keeps the middle 80%)
tmse = trimmed_mean_squared_error(y_pred, y_true, proportion=0.1, mask=mask)
```
See also: [trimmed_mean_squared_error](../api/metrics.md).

### Median Absolute Deviation (MAD)

Median of absolute deviations from the median error, scaled by a consistency factor:

$$
\text{MAD}(e) = c \cdot \text{median}\left(\{|e_i - \text{median}(e)|\}_{i: m_i=1}\right)
$$

where $e_i = y_i - \hat{y}_i$ and $c = 1.4826$ by default (the Gaussian consistency factor that makes MAD an unbiased estimator of $\sigma$ for normally distributed residuals).

```python
from torchregress.metrics.point import median_absolute_deviation

mad = median_absolute_deviation(y_pred, y_true, scale=1.4826, mask=mask)
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

nrmse = normalized_rmse(y_pred, y_true, normalization='std', mask=mask)
```
See also: [normalized_rmse](../api/metrics.md).

### Normalized Median Absolute Deviation

MAD normalized by target statistics for relative scale comparison:

$$
\text{NMAD}(y, \hat{y}) = \frac{\text{MAD}(y - \hat{y})}{\text{scale}}
$$

```python
from torchregress.metrics.point import normalized_median_absolute_deviation

nmad = normalized_median_absolute_deviation(y_pred, y_true, mask=mask)
```
See also: [normalized_median_absolute_deviation](../api/metrics.md).

---

## Application-Specific Metrics

### Outlier Fraction

Fraction of predictions with absolute error exceeding a threshold:

$$
\text{OutlierFraction}(y, \hat{y}; \tau) = \frac{1}{\sum_{i=1}^N m_i} \sum_{i=1}^N m_i \mathbb{I}\left(|y_i - \hat{y}_i| > \tau \cdot \text{scale}_i\right)
$$

```python
from torchregress.metrics.point import outlier_fraction

# Standard outlier detection (scaled by std of y_true)
of = outlier_fraction(y_pred, y_true, threshold=0.15, mask=mask)
```
See also: [outlier_fraction](../api/metrics.md).

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
mae_tail = tail_mae(y_pred, y_true, quantile=0.9, tail="upper", mask=mask)
rmse_tail = tail_rmse(y_pred, y_true, quantile=0.9, tail="upper", mask=mask)
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

report = regression_metrics_report(y_pred, y_true, mask=mask, weights=weights)
```
See also: [regression_metrics_report](../api/metrics.md).
