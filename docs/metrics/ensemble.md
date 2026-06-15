# Ensemble Metrics

Ensemble metrics evaluate predictive performance and decompose predictive uncertainty across multiple model predictions.

---

## `ensemble_statistics`

Aggregates individual predictions from $M$ ensemble members $\{y^{(1)}, \dots, y^{(M)}\}$ to compute the ensemble mean and sample variance:

$$
\bar{y}_i = \frac{1}{M} \sum_{m=1}^M y_i^{(m)}
$$

$$
\text{Var}(y_i) = \frac{1}{M} \sum_{m=1}^M \left(y_i^{(m)} - \bar{y}_i\right)^2
$$

```python
import torch
from torchregress.metrics.ensemble import ensemble_statistics

predictions = torch.randn(5, 100)  # 5 ensemble members, 100 predictions each
mean, variance = ensemble_statistics(predictions)
```
See also: [ensemble_statistics](../api/metrics.md#ensemble_statistics).

---

## `uncertainty_decomposition`

Decomposes total predictive uncertainty into **epistemic** (model disagreement) and **aleatoric** (data noise) uncertainty using the Law of Total Variance.

For ensemble members predicting means $\mu_m(x)$ and aleatoric variances $\sigma_m^2(x)$, the decomposition is:

- **Ensemble mean**:
  $$\bar{\mu}(x) = \frac{1}{M} \sum_{m=1}^M \mu_m(x)$$
- **Epistemic uncertainty** (variance of predicted means):
  $$\sigma^2_{\text{epistemic}}(x) = \frac{1}{M} \sum_{m=1}^M (\mu_m(x) - \bar{\mu}(x))^2$$
- **Aleatoric uncertainty** (mean of predicted variances):
  $$\sigma^2_{\text{aleatoric}}(x) = \frac{1}{M} \sum_{m=1}^M \sigma_m^2(x)$$
- **Total uncertainty**:
  $$\sigma^2_{\text{total}}(x) = \sigma^2_{\text{epistemic}}(x) + \sigma^2_{\text{aleatoric}}(x)$$

```python
from torchregress.metrics.ensemble import uncertainty_decomposition

# means: [M, N], variances: [M, N]
uncertainty = uncertainty_decomposition(means, variances)
```
See also: [uncertainty_decomposition](../api/metrics.md#uncertainty_decomposition).

---

## `gaussian_nll_ensemble`

Computes the Gaussian negative log-likelihood of the targets under the ensembled predictive distribution:

$$
\mathcal{L}(y_i) = \frac{1}{2} \log(2\pi \sigma^2_{\text{total}, i}) + \frac{(y_i - \bar{\mu}_i)^2}{2\sigma^2_{\text{total}, i}}
$$

where $\bar{\mu}_i$ is the ensemble mean and $\sigma^2_{\text{total}, i}$ is the total uncertainty.

```python
from torchregress.metrics.ensemble import gaussian_nll_ensemble

nll = gaussian_nll_ensemble(means, variances, y_true)
```
See also: [gaussian_nll_ensemble](../api/metrics.md#gaussian_nll_ensemble).

---

## `ensemble_interval_bounds`

Computes symmetric Gaussian prediction intervals at significance level $\alpha$:

$$
L_i = \bar{\mu}_i - z_{1 - \alpha/2} \cdot \sigma_{\text{total}, i}
$$

$$
U_i = \bar{\mu}_i + z_{1 - \alpha/2} \cdot \sigma_{\text{total}, i}
$$

where $z_{p} = \Phi^{-1}(p)$ is the standard normal quantile.

```python
from torchregress.metrics.ensemble import ensemble_interval_bounds

lower, upper = ensemble_interval_bounds(means, variances, alpha=0.1)
```
See also: [ensemble_interval_bounds](../api/metrics.md#ensemble_interval_bounds).

---

## `ensemble_interval_metrics`

Computes the ensembled PICP (empirical coverage) and Winkler interval score for the generated prediction intervals.

```python
from torchregress.metrics.ensemble import ensemble_interval_metrics

metrics = ensemble_interval_metrics(means, variances, y_true, alpha=0.1)
```
See also: [ensemble_interval_metrics](../api/metrics.md#ensemble_interval_metrics).
