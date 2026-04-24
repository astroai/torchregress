# Quantile & Expectile Losses

Quantile and expectile regression estimate **different points of the conditional distribution** $P(Y \mid X)$ — quantiles target specific percentiles, expectiles target weighted means.  Both are **distribution-free**: no Gaussian assumption required.

---

## Quantile vs Expectile at a Glance

| | Quantile | Expectile |
|:--|:---------|:----------|
| **Asymmetric loss** | $\lvert r \rvert$ (L1) | $r^2$ (L2) |
| **At $\tau = 0.5$** | Median | Mean |
| **Robustness** | ⭐⭐⭐ (bounded influence) | ⭐⭐ (sensitive to tails) |
| **Efficiency** | Lower (non-differentiable at 0) | Higher (smooth everywhere) |
| **Financial risk** | VaR | Expected Shortfall (ES) |

---

## Mathematical Background

### Quantile loss (check / pinball)

$$\boxed{\;\rho_\tau(r) = r\,(\tau - \mathbb{1}_{r < 0}) = \begin{cases} \tau\, r, & r \geq 0 \\ (\tau - 1)\, r, & r < 0 \end{cases}\;}$$

### Expectile loss (asymmetric least squares)

$$\boxed{\;L_\tau(r) = |\tau - \mathbb{1}_{r < 0}|\; r^2\;}$$

When $\tau = 0.5$ the expectile loss reduces to standard MSE.

---

## Quantile Losses

### QuantileLoss

Single quantile:

```python
from torchregress.losses import QuantileLoss

loss_fn = QuantileLoss(quantile=0.9)   # 90th percentile
loss = loss_fn(y_pred, y_true)
```

### MultiQuantileLoss

Multiple quantiles simultaneously — the model outputs one prediction per quantile:

```python
from torchregress.losses import MultiQuantileLoss

loss_fn = MultiQuantileLoss(
    quantiles=[0.05, 0.5, 0.95],  # 90% prediction interval + median
    joint_prediction=True,         # model outputs [batch, Q, d]
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `quantiles` | `list[float]` | — | Quantile levels (ascending) |
| `joint_prediction` | `bool` | `True` | If `True`, `y_pred` shape is `[batch, Q, d]` |
| `quantile_weights` | `Tensor` or `None` | `None` | Per-quantile importance weights |

=== "Model architecture"

    ```python
    class QuantileModel(nn.Module):
        def __init__(self, in_dim, out_dim, n_quantiles=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128), nn.ReLU(),
                nn.Linear(128, out_dim * n_quantiles),
            )
            self.out_dim = out_dim
            self.n_quantiles = n_quantiles

        def forward(self, x):
            out = self.net(x)                              # [B, Q*d]
            return out.view(-1, self.n_quantiles, self.out_dim)  # [B, Q, d]
    ```

=== "Training"

    ```python
    model = QuantileModel(10, 1, n_quantiles=3)
    loss_fn = MultiQuantileLoss(quantiles=[0.05, 0.5, 0.95])
    optimizer = torch.optim.Adam(model.parameters())

    for x, y in train_loader:
        loss = loss_fn(model(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    ```

=== "Inference"

    ```python
    with torch.no_grad():
        preds = model(x_test)  # [B, 3, 1]
        lower, median, upper = preds[:, 0], preds[:, 1], preds[:, 2]
    ```

### QuantileCrossoverLoss

Adds a **penalty** when quantile curves cross (lower quantile predicts more than higher):

```python
from torchregress.losses import QuantileCrossoverLoss

loss_fn = QuantileCrossoverLoss(
    quantiles=[0.05, 0.5, 0.95],
    crossover_penalty=10.0,  # strength of non-crossing constraint
)
```

!!! tip "When to use"
    Always prefer `QuantileCrossoverLoss` over plain `MultiQuantileLoss` — it prevents unphysical crossing of prediction bands at essentially no extra cost.

---

## Expectile Losses

### ExpectileLoss

Single expectile:

```python
from torchregress.losses import ExpectileLoss

loss_fn = ExpectileLoss(expectile=0.8)   # 80th expectile
loss = loss_fn(y_pred, y_true)
```

### MultiExpectileLoss

Multiple expectiles simultaneously:

```python
from torchregress.losses import MultiExpectileLoss

loss_fn = MultiExpectileLoss(
    expectiles=[0.1, 0.5, 0.9],
    joint_prediction=True,
)
```

### ExpectileCrossoverLoss

Non-crossing constraint for expectiles (analogous to `QuantileCrossoverLoss`):

```python
from torchregress.losses import ExpectileCrossoverLoss

loss_fn = ExpectileCrossoverLoss(
    expectiles=[0.1, 0.5, 0.9],
    crossover_penalty=10.0,
)
```

### AsymmetricLeastSquaresLoss

Alias for `ExpectileLoss` — uses `tau` instead of `expectile` for compatibility:

```python
from torchregress.losses import AsymmetricLeastSquaresLoss

loss_fn = AsymmetricLeastSquaresLoss(tau=0.75)  # same as ExpectileLoss(0.75)
```

---

## Choosing Between Quantile and Expectile

!!! info "Use quantiles when"
    - You need a specific probability statement ("90% of values fall below this")
    - Data has heavy-tailed outliers
    - Computing VaR

!!! info "Use expectiles when"
    - You want smoother, more efficient estimates
    - You need sensitivity to tail *magnitudes* (not just probabilities)
    - Computing Expected Shortfall (ES)

!!! tip "Conformal calibration"
    For **guaranteed coverage** on top of quantile regression, wrap your model with [Conformalized Quantile Regression (CQR)](../methods/conformal/predictors.md).

---

## References

| # | Reference |
|:-:|:----------|
| 1 | R. Koenker, G. Bassett. "Regression Quantiles." *Econometrica*, 46(1):33–50, **1978**. |
| 2 | W.K. Newey, J.L. Powell. "Asymmetric Least Squares Estimation and Testing." *Econometrica*, 55(4):819–847, **1987**. |
| 3 | V. Chernozhukov, I. Fernández-Val, A. Galichon. "Quantile and Probability Curves without Crossing." *Econometrica*, 78(3):1093–1125, **2010**. |
