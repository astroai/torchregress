# Out-of-Distribution Detection Metrics

Out-of-Distribution (OOD) detection metrics evaluate how well a model can identify inputs or predictions that differ significantly from the training distribution.

→ See [Decision metrics](decision.md) for selective-prediction evaluation and [OOD API](../api/metrics.md).

---

## Mahalanobis Distance

Measures how many standard deviations a feature vector $x$ is from the mean of the training representation distribution:

$$
d_M(x) = \sqrt{(x - \mu)^\top \Sigma^{-1} (x - \mu)}
$$

where $\mu$ is the mean vector of training representations and $\Sigma$ is the training covariance matrix.

```python
from torchregress.metrics.ood import mahalanobis_distance

# x has shape [batch_size, n_features]
# mean has shape [n_features]
# cov has shape [n_features, n_features]
md = mahalanobis_distance(x, mean, cov)
```
See also: [mahalanobis_distance](../api/metrics.md).

---

## Typicality Score

Typicality measures whether a batch of test samples $X = \{x_1, \dots, x_N\}$ exhibits the expected information content under the training distribution model $p(x)$. Let the empirical entropy of the batch be:

$$
H_p(X) = -\frac{1}{N} \sum_{i=1}^N \log p(x_i)
$$

The typicality score is the absolute deviation from the true model entropy $H(p) = \mathbb{E}_{x \sim p}[-\log p(x)]$:

$$
T(X) = \left| H_p(X) - H(p) \right|
$$

A high typicality score indicates that the batch contains samples that are either highly improbable or too structured, signaling a distribution shift.

```python
from torchregress.metrics.ood import typicality_score

# Using a tuple of (mean, variance) representing distribution parameters
ts = typicality_score((mean_pred, var_pred), x_test)
```
See also: [typicality_score](../api/metrics.md).

---

## Entropy Score

Calculates the Shannon/differential entropy of the predictive distribution $p(y \mid x)$ as a measure of **total** predictive uncertainty:

$$
H(p) = -\int p(y \mid x) \log p(y \mid x) \, dy
$$

For empirical samples, this can be estimated using histogram-based density binning:

$$
\hat{H}(p) = -\sum_{k=1}^K \hat{p}_k \log \hat{p}_k
$$

where $\hat{p}_k$ is the fraction of samples falling in bin $k$.

```python
from torchregress.metrics.ood import entropy_score

# samples has shape [n_samples, batch_size, ...]
es = entropy_score(samples)
```
See also: [entropy_score](../api/metrics.md).

---

## Kernel Density Score

Estimates target density using a reference training set $\{r_1, \dots, r_R\}$ under an RBF (Gaussian) kernel:

$$
\hat{f}_h(x) = \frac{1}{R} \sum_{k=1}^R K_h(x - r_k)
$$

$$
K_h(d) = \frac{1}{(2\pi h^2)^{D/2}} \exp\left(-\frac{\|d\|^2}{2h^2}\right)
$$

where $h > 0$ is the bandwidth parameter and $D$ is the dimensionality of feature space.

```python
from torchregress.metrics.ood import kernel_density_score

kds = kernel_density_score(x_test, x_reference, bandwidth=0.5)
```
See also: [kernel_density_score](../api/metrics.md).

---

## Comprehensive Reporting

### OOD Metrics Report

Generate a comprehensive report of OOD detection metrics.

```python
from torchregress.metrics.ood import ood_metrics_report

report = ood_metrics_report(
    model_output=(mean_pred, var_pred),
    x_test=x_test,
    x_reference=x_train,
    mean=train_mean,
    cov=train_cov,
    samples=pred_samples
)
```
See also: [ood_metrics_report](../api/metrics.md).

---

## References

| # | Reference |
|:-:|:----------|
| 1 | K. Lee, K. Lee, H. Lee, J. Shin. ["A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks."](https://arxiv.org/abs/1807.03888) *NeurIPS*, **2018**. |
| 2 | E. Nalisnick, A. Matsukawa, Y.W. Teh, D. Görür, B. Lakshminarayanan. ["Do Deep Generative Models Know What They Don't Know?"](https://arxiv.org/abs/1810.09136) *ICLR*, **2019**. |
| 3 | E. Parzen. ["On Estimation of a Probability Density Function and Mode."](https://doi.org/10.1214/aoms/1177704472) *Ann. Math. Stat.*, 33(3):1065–1076, **1962**. |
