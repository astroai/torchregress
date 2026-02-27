# Censored Metrics

Metrics for censored and interval-censored regression outcomes.

## Available Metrics

- `censoring_rate`: fraction of censored samples.
- `observed_mae`: MAE on observed (`censoring == 0`) samples only.
- `concordance_index`: Harrell-style concordance index.
- `interval_overlap_rate`: overlap rate between predicted and censoring intervals.

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
