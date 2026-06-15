# Ordinal Metrics

Metrics for ordered-class regression/classification outputs.

---

## Available Metrics

### Ordinal Accuracy

Measures the exact agreement between the predicted class index $\hat{c}_i$ and the true class index $c_i$:

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{c}_i = c_i)
$$

### Mean Absolute Class Error

Computes the Mean Absolute Error in ordinal class space, evaluating the distance between predicted and true rating classes:

$$
\text{MACE} = \frac{1}{N} \sum_{i=1}^N |\hat{c}_i - c_i|
$$

### Quadratic Weighted Kappa (QWK)

Measures agreement between predictions and targets, penalizing larger ordinal mistakes more heavily.

Let $O_{j,k}$ be the observed rating counts (contingency table) where true class is $j$ and predicted class is $k$. Let $E_{j,k}$ be the expected rating counts under independence:

$$
E_{j,k} = \frac{\sum_a O_{j,a} \cdot \sum_b O_{b,k}}{N}
$$

The quadratic weight matrix $W$ is:

$$
W_{j,k} = \frac{(j - k)^2}{(K - 1)^2}
$$

where $K$ is the total number of ordinal classes. The kappa coefficient is:

$$
\kappa = 1 - \frac{\sum_{j,k} W_{j,k} O_{j,k}}{\sum_{j,k} W_{j,k} E_{j,k}}
$$

---

## Usage

```python
import torch
import torchregress as tr

logits = torch.randn(64, 5)
labels = torch.randint(0, 5, (64,))

acc = tr.metrics.ordinal_accuracy(logits, labels, encoding="class_logits")
mae_cls = tr.metrics.mean_absolute_class_error(logits, labels, encoding="class_logits")
qwk = tr.metrics.quadratic_weighted_kappa(logits, labels, encoding="class_logits")
```

See the [quadratic_weighted_kappa API](../api/metrics.md#quadratic_weighted_kappa) for details.
