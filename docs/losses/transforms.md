# Transform Losses

Transform losses apply a target-space mapping before scoring prediction error. They are useful when regression noise is multiplicative, the target is strongly skewed, or variance grows with magnitude.

Available losses:

- `TransformedTargetLoss`
- `LogTransformLoss`
- `BoxCoxTransformLoss`
- `SqrtTransformLoss`
- `YeoJohnsonTransformLoss`

Core utilities:

- `torchregress.utils.LogTransform`
- `torchregress.utils.BoxCoxTransform`
- `torchregress.utils.SqrtTransform`
- `torchregress.utils.YeoJohnsonTransform`
- `torchregress.utils.make_target_transform`

## Objective

For a transform $T(\cdot)$ and a pointwise base loss $\ell(\cdot, \cdot)$:

$$
\mathcal{L}(\hat{y}, y) = \ell(T(\hat{y}), T(y)).
$$

This is a pragmatic choice when the transformed target is closer to homoscedastic and closer to the geometry the downstream error metric actually cares about.

## Which Transform

| Loss | Target support | Good default for |
|:-----|:---------------|:-----------------|
| `LogTransformLoss` | $y \ge 0$ | multiplicative noise, strong right skew, relative-error style objectives |
| `BoxCoxTransformLoss` | $y \ge 0$ | positive targets when log is too aggressive or too weak |
| `SqrtTransformLoss` | $y \ge 0$ | count-like targets or moderate variance growth |
| `YeoJohnsonTransformLoss` | signed | signed skewed targets where Box-Cox/log are invalid |

!!! tip
    Use the simplest transform that matches the target support and residual structure. Start with `LogTransformLoss` for positive multiplicative noise and `YeoJohnsonTransformLoss` for signed skewed targets.

## API

```python
from torchregress.losses import (
    BoxCoxTransformLoss,
    LogTransformLoss,
    SqrtTransformLoss,
    TransformedTargetLoss,
    YeoJohnsonTransformLoss,
)
```

All transform losses follow the standard loss contract:

```python
loss = loss_fn(y_pred, target, mask=mask, weights=weights)
```

All transform loss classes also expose:

```python
restored = loss_fn.inverse(y_transformed)
```

## Examples

### Positive multiplicative-noise target

```python
import torch
from torchregress.losses import LogTransformLoss

y_pred = torch.tensor([[0.8], [1.4], [3.2]])
y_true = torch.tensor([[0.7], [1.7], [2.8]])

loss_fn = LogTransformLoss()
loss = loss_fn(y_pred, y_true)
```

### Signed skewed target

```python
import torch
from torchregress.losses import YeoJohnsonTransformLoss

y_pred = torch.tensor([[-1.0], [0.3], [2.1]])
y_true = torch.tensor([[-0.8], [0.5], [1.7]])

loss_fn = YeoJohnsonTransformLoss(lam=0.5)
loss = loss_fn(y_pred, y_true)
```

### Generic wrapper

```python
from torchregress.losses import TransformedTargetLoss

loss_fn = TransformedTargetLoss("boxcox", lam=0.25, base_loss="huber")
```

## Comparison Example

See [Transformed-Target Regression Comparison](../examples/transformed_target_regression_comparison.md) for a shared-budget benchmark on skewed positive targets with `MSE`, `LogTransformLoss`, `BoxCoxTransformLoss`, and `SqrtTransformLoss`.

## Caveats

!!! warning
    `LogTransformLoss`, `BoxCoxTransformLoss`, and `SqrtTransformLoss` require positive-support predictions and targets. Use a positive output head such as `Softplus` when training directly in the original target space.

!!! info
    Transform losses change the optimization geometry, not the model family. You still need calibration, OOD checks, and hard-problem evaluation just as you would for plain `WeightedMSELoss`.

## References

| # | Reference |
|:-:|:----------|
| 1 | G.E.P. Box, D.R. Cox. "An Analysis of Transformations." *JRSS B*, **1964**. |
| 2 | I.K. Yeo, R.A. Johnson. "A New Family of Power Transformations to Improve Normality or Symmetry." *Biometrika*, **2000**. |
