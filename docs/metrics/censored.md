# Censored Metrics

> ← [Multivariate Metrics](multivariate.md) | [Ordinal Metrics](ordinal.md) →

Metrics for censored and interval-censored regression outcomes.

---

## Available Metrics

### Censoring Rate

Measures the fraction of samples in the dataset that are censored:

$$
r_{\text{censor}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(c_i \neq 0)
$$

where $c_i \in \{-1, 0, 1\}$ is the censoring indicator ($0$ for fully observed, $1$ for right-censored, $-1$ for left-censored).

### Observed MAE

Computes the Mean Absolute Error only on the subset of data that is fully observed (not censored):

$$
\text{ObservedMAE}(y, \hat{y}; c) = \frac{\sum_{i=1}^N \mathbb{I}(c_i = 0) |y_i - \hat{y}_i|}{\sum_{i=1}^N \mathbb{I}(c_i = 0)}
$$

### Concordance Index

Measures the ordinal rank agreement between predicted values and censored targets (Harrell's C-index).

A pair of samples $(i, j)$ with observed times $t_i, t_j$ is comparable if $t_i < t_j$ and sample $i$ is fully observed ($c_i = 0$). The concordance index is the fraction of comparable pairs where the predicted order matches the observed order, with tied predictions receiving half credit:

$$
C = \frac{\sum_{(i,j) \in \mathcal{P}} \left[\mathbb{I}(\hat{y}_i < \hat{y}_j) + \tfrac{1}{2}\,\mathbb{I}(\hat{y}_i = \hat{y}_j)\right]}{|\mathcal{P}|}
$$

where $|\mathcal{P}|$ is the number of comparable pairs.

### Interval Overlap Rate

Measures the fraction of samples where a predicted interval $[\hat{L}_i, \hat{U}_i]$ overlaps with a censor-interval $[L_i, U_i]$:

$$
\text{OverlapRate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{U}_i \ge L_i \;\text{and}\; \hat{L}_i \le U_i)
$$

---

## Usage

```python
import torch
import torchregress as tr

pred = torch.tensor([0.5, 1.0, 1.5, 2.0])
target = torch.tensor([0.4, 1.2, 1.3, 2.1])
censoring = torch.tensor([0, 1, 0, -1])

c_rate = tr.metrics.censoring_rate(censoring)
mae_obs = tr.metrics.observed_mae(pred, target, censoring)
c_index = tr.metrics.concordance_index(pred, target, censoring)
```

See the [censored metrics API](../api/metrics.md): [`censoring_rate`](../api/metrics.md), [`observed_mae`](../api/metrics.md), [`concordance_index`](../api/metrics.md), and [`interval_overlap_rate`](../api/metrics.md).

---

## Limitations

1. **Censoring Rate is not a quality metric**: A high censoring rate indicates data sparsity but does not directly measure model performance. Always pair with observed MAE and concordance index.
2. **Concordance Index is rank-based only**: C-index measures whether predictions preserve the correct order of survival times, but does not assess absolute calibration. Two models with the same C-index can have very different survival-time predictions.
3. **Observed MAE only evaluates uncensored points**: It does not account for censored targets, which may bias the evaluation if censoring is informative (non-random).
4. **Interval overlap rate assumes bounded intervals**: The metric is most interpretable when targets are interval-censored with known bounds; for right-censored targets, the information is weaker.

## Recommendations

- **Report multiple censored metrics together**: Observed MAE + Concordance Index + Censoring Rate gives a balanced view of accuracy, ranking, and data missingness.
- **Use Kaplan–Meier or AFT-based evaluation** alongside point metrics for survival settings where the full time-to-event distribution matters.
- **Stratify evaluation by censoring type** (right, left, interval) when the dataset contains mixed censoring patterns.

## Next steps

- [Ordinal metrics](ordinal.md) — related evaluation for ordered-class outcomes
- [Point metrics](point.md) — observed MAE pairs naturally with standard point accuracy on uncensored subsets
- [Censored losses](../losses/censored.md) — censored Gaussian NLL, quantile, and AFT losses for training
- [Censored regression example](../examples/censored_regression_comparison.md) — full comparison of censored methods

---

## References

| # | Reference |
|:-:|:----------|
| 1 | F.E. Harrell, R.M. Califf, D.B. Pryor, K.L. Lee, R.A. Rosati. ["Evaluating the Yield of Medical Tests."](https://doi.org/10.1001/jama.1982.03320430047030) *JAMA*, 247(18):2543–2546, **1982**. |
| 2 | J. Tobin. ["Estimation of Relationships for Limited Dependent Variables."](https://www.jstor.org/stable/1907382) *Econometrica*, 26(1):24–36, **1958**. |
