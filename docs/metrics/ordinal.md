# Ordinal Metrics

Metrics for ordered-class regression/classification outputs.

## Available Metrics

- `ordinal_accuracy`: exact class-index accuracy.
- `mean_absolute_class_error`: MAE in ordinal class space.
- `quadratic_weighted_kappa`: agreement metric that penalizes larger ordinal mistakes more strongly.

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
