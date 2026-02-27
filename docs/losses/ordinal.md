# Ordinal Regression Losses

Ordinal losses are for ordered discrete targets (for example, quality scores 0-4 where distance between classes matters).

## Available Losses

- `OrdinalCrossEntropyLoss`: class-logit cross-entropy baseline for ordinal labels.
- `CumulativeLinkLoss`: cumulative-threshold ordinal loss over `K-1` logits.
- `CORALLoss`: CORAL-style cumulative ordinal objective.

## Usage

```python
import torch
import torchregress as tr

num_classes = 5
logits = torch.randn(32, num_classes - 1)
labels = torch.randint(0, num_classes, (32,))

loss_fn = tr.losses.CumulativeLinkLoss()
loss = loss_fn(logits, labels)
```

## Decoding Predictions

Use `torchregress.utils.ordinal_predict` to decode cumulative logits or class logits to ordinal labels.
