# Censored Regression Losses

Losses for right/left and interval-censored targets.

Censor encoding convention:

- `0`: observed target
- `1`: right-censored (`y >= target`)
- `-1`: left-censored (`y <= target`)

Interval-censoring is provided via explicit `lower_bound` and `upper_bound` tensors.

## Available Losses

- `CensoredGaussianNLLLoss`
- `CensoredQuantileLoss`
- `AFTLoss` (log-normal accelerated failure time)

## Usage

```python
import torch
import torchregress as tr

mean = torch.randn(64)
log_var = torch.randn(64)
target = torch.rand(64) * 5.0
censoring = torch.randint(low=-1, high=2, size=(64,))

loss_fn = tr.losses.CensoredGaussianNLLLoss()
loss = loss_fn((mean, log_var), target, censoring=censoring)
```
