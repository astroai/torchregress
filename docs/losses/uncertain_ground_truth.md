# Uncertain Ground-Truth Losses

These losses target weak/noisy supervision where labels have uncertainty or are partially
trusted.

## APIs

- `NoisyTargetGaussianNLL`
- `ConsistencyRegLoss`
- `PseudoLabelNLL`

## Quick Start

```python
import torch
from torchregress.losses import NoisyTargetGaussianNLL

loss_fn = NoisyTargetGaussianNLL()
loss = loss_fn(y_pred, y_obs, target_variance=y_obs_var)
```

## Notes

- `NoisyTargetGaussianNLL`: adds target-noise variance to predictive variance.
- `ConsistencyRegLoss`: adds teacher-student consistency regularization.
- `PseudoLabelNLL`: blends observed labels with pseudo-labels and confidence weights.
- All losses support optional propensity hooks (`propensity_scores` or
  `propensity_weights`) for selection-bias-aware training.
