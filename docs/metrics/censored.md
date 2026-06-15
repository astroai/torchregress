# Censored Metrics

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

A pair of samples $(i, j)$ with observed times $t_i, t_j$ is comparable if $t_i < t_j$ and sample $i$ is fully observed ($c_i = 0$). The concordance index is the fraction of comparable pairs where the predicted order matches the observed order:

$$
C = \frac{\sum_{(i,j) \in \mathcal{P}} \mathbb{I}(\hat{y}_i < \hat{y}_j)}{\mathcal{P}}
$$

where $\mathcal{P}$ is the set of all comparable pairs.

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

See the [censored metrics API](../api/metrics.md#censored-metrics): [`censoring_rate`](../api/metrics.md#censoring_rate), [`observed_mae`](../api/metrics.md#observed_mae), and [`concordance_index`](../api/metrics.md#concordance_index).
