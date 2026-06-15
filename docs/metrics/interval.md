# Interval Metrics

Interval metrics evaluate the quality of prediction intervals, focusing on coverage and width properties.

---

## Interval Score (Winkler Score)

The Winkler interval score evaluates prediction intervals by rewarding narrow bands and penalizing intervals when observations fall outside them:

$$
S_\alpha(L_i, U_i; y_i) = (U_i - L_i) + \frac{2}{\alpha} (L_i - y_i) \mathbb{I}(y_i < L_i) + \frac{2}{\alpha} (y_i - U_i) \mathbb{I}(y_i > U_i)
$$

where $[L_i, U_i]$ is the predicted $(1 - \alpha)$ interval (e.g., a $90\%$ interval corresponds to $\alpha = 0.1$) and $y_i$ is the target. A lower score indicates a higher quality interval.

```python
from torchregress.metrics.interval import interval_score

# lower_bound and upper_bound typically represent a 90% prediction interval
score = interval_score(lower_bound, upper_bound, y_true, alpha=0.1)
```
See also: [interval_score](../api/metrics.md#interval_score).

---

## Prediction Interval Coverage Probability (PICP)

PICP measures the proportion of observations that fall within the predicted intervals:

$$
\text{PICP}(L, U; y) = \frac{1}{\sum_{i=1}^N m_i} \sum_{i=1}^N m_i \mathbb{I}(L_i \le y_i \le U_i)
$$

where $m_i \in \{0, 1\}$ is an optional boolean mask when samples are pre-filtered. Ideally, $\text{PICP} \approx 1 - \alpha$.

```python
from torchregress.metrics.interval import prediction_interval_coverage_probability

# Calculate basic PICP
picp = prediction_interval_coverage_probability(lower_bound, upper_bound, y_true)
```
See also: [prediction_interval_coverage_probability](../api/metrics.md#prediction_interval_coverage_probability).

---

## Mean Prediction Interval Width (MPIW)

MPIW measures the average width of prediction intervals:

$$
\text{MPIW}(L, U) = \frac{1}{\sum_{i=1}^N m_i} \sum_{i=1}^N m_i (U_i - L_i)
$$

```python
from torchregress.metrics.interval import prediction_interval_coverage_probability
from torchregress.metrics import MeanPredictionIntervalWidth

# Via PICP diagnostics dict
results = prediction_interval_coverage_probability(
    lower_bound, upper_bound, y_true,
    return_diagnostics=True,
)
mpiw = results["mpiw"]

# Or accumulate MPIW across batches with the stateful metric
mpiw_metric = MeanPredictionIntervalWidth()
mpiw_metric.update(lower_bound, upper_bound)
width = mpiw_metric.compute()
```

See also: [`MeanPredictionIntervalWidth`](../api/metrics.md#meanpredictionintervalwidth) and [prediction_interval_coverage_probability](../api/metrics.md#prediction_interval_coverage_probability) (`return_diagnostics=True`).

---

## Comprehensive Reporting

### Interval Metrics Report

Compare interval quality across multiple models.

```python
from torchregress.metrics.interval import interval_metrics_report

# Create a dictionary of model predictions
predictions = {
    'model1': {'lower': model1_lower, 'upper': model1_upper},
    'model2': {'lower': model2_lower, 'upper': model2_upper}
}

# Generate comprehensive report
report = interval_metrics_report(predictions, y_true, alpha=0.1)
```
See also: [interval_metrics_report](../api/metrics.md#interval_metrics_report).

---

## Advanced Usage

### Asymmetric Interval Evaluation

Evaluate lower vs. upper miss rates to detect interval asymmetry:

$$
\text{MissRate}_{\text{low}} = \frac{\sum_i m_i \mathbb{I}(y_i < L_i)}{\sum_i m_i}
$$

$$
\text{MissRate}_{\text{high}} = \frac{\sum_i m_i \mathbb{I}(y_i > U_i)}{\sum_i m_i}
$$

```python
coverage = prediction_interval_coverage_probability(
    lower_bound, upper_bound, y_true, alpha=0.1, return_diagnostics=True
)
print(f"Lower miss rate: {coverage['miss_rate_low']}")
print(f"Upper miss rate: {coverage['miss_rate_high']}")
```
