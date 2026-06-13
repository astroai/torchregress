# Iteratively Reweighted Least Squares (IRLS)

IRLS solves robust regression by iteratively downweighting outliers — at each step, residuals are computed, weights are updated based on a robust criterion, and the model is re-fitted with the new weights.

---

## Mathematical Background

Standard OLS minimises $\sum_i (y_i - f(x_i))^2$, giving equal weight to all residuals.  IRLS generalises this to:

$$\hat\beta^{(t+1)} = \arg\min_\beta \sum_{i=1}^n w_i^{(t)}\!\left(y_i - f_\beta(x_i)\right)^2$$

where the weights $w_i^{(t)}$ depend on the residuals from iteration $t$:

$$w_i^{(t)} = \psi\!\left(\frac{r_i^{(t)}}{\hat\sigma}\right)$$

and $\psi$ is the weight function (Huber, Tukey, etc.), $\hat\sigma$ is a robust scale estimate (MAD).

!!! info "Convergence"
    For convex weight functions (Huber), IRLS converges to the global M-estimate.  For non-convex weight functions (Tukey), convergence depends on initialisation — start with OLS.

---

## Weight Functions

| Function | Formula | Behaviour | Default $c$ |
|:---------|:--------|:----------|:-----------|
| `"huber"` | $w = \min(1,\; \delta/\lvert r\rvert)$ | Downweights, never rejects | $\delta = 1.0$ |
| `"tukey"` | $w = (1 - (r/c)^2)^2$ if $\lvert r\rvert \leq c$; else $0$ | Rejects outliers | $c = 4.685$ |
| `"power"` | $w = 1/(a + \lvert r\rvert^b)$ | Power-law decay | $a=1, b=2$ |
| Custom `Callable` | User-defined | Flexible | — |

---

## API: `iteratively_reweighted_least_squares`

Use this function inside your own training loop to compute robust precision weights from current model predictions:

```python
from torchregress.algorithms.irls import iteratively_reweighted_least_squares

y_pred, loss_history, final_precision = iteratively_reweighted_least_squares(
    model=my_model,
    x=X,
    y_true=y,
    weight_fn="tukey",
    max_iter=20,
    variance_type="robust",
)
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `model` | `nn.Module` | — | PyTorch model (weights fixed during reweighting) |
| `x`, `y_true` | `Tensor` | — | Batch of inputs and targets |
| `weight_fn` | `str` or `Callable` | `"huber"` | Weight function |
| `base_loss` | `str` | `"gaussian"` | `"gaussian"`, `"huber"`, or `"l1"` |
| `max_iter` | `int` | `10` | IRLS iterations |
| `tol` | `float` | `1e-4` | Convergence tolerance |
| `variance_type` | `str` | `"predicted"` | `"predicted"`, `"fixed"`, or `"robust"` |

**Returns:** `(y_pred, loss_history, final_precision)` — use `final_precision` as sample weights in a weighted loss step.

---

## Complete Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchregress.algorithms.irls import iteratively_reweighted_least_squares
from torchregress.losses import WeightedMSELoss

torch.manual_seed(42)
n = 1000
X = torch.randn(n, 5)
y = X @ torch.tensor([1.0, -0.5, 0.2, 0.7, -0.3]) + 0.1 * torch.randn(n)
outliers = torch.randperm(n)[:50]
y[outliers] += 5.0 * torch.randn(50)

model = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 1))
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = WeightedMSELoss()

for epoch in range(5):
    _, _, precision = iteratively_reweighted_least_squares(
        model,
        X,
        y.unsqueeze(1),
        weight_fn="tukey",
        max_iter=15,
        variance_type="robust",
    )
    opt.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y.unsqueeze(1), weights=precision)
    loss.backward()
    opt.step()

clean = ~torch.isin(torch.arange(n), outliers)
pred = model(X).detach()
print(f"IRLS MAE (clean): {(pred[clean] - y[clean].unsqueeze(1)).abs().mean():.4f}")
```

---

## Comparison with Robust Losses

| Approach | Handles Outliers | Retraining | Best For |
|:---------|:---------------:|:----------:|:---------|
| **Robust losses** (Huber, Cauchy, Tukey) | During training | Requires retraining | End-to-end training |
| **IRLS reweighting** | Post-hoc or iterative | Can refine existing model | Fine-tuning, classical statistics |
| **CVaR** | Tail focus | During training | Worst-case performance |

!!! tip "When to use IRLS vs robust losses"
    Use **robust losses** when training from scratch.  Use **IRLS reweighting** when you want to iteratively refine an existing model or when you need the classical M-estimation framework (e.g., for statistical inference with influence functions).

---

## References

| # | Reference |
|:-:|:----------|
| 1 | P.W. Holland, R.E. Welsch. "Robust Regression Using Iteratively Reweighted Least-Squares." *Commun. Stat. Theory Methods*, 6(9):813–827, **1977**. |
| 2 | P.J. Huber. "Robust Regression: Asymptotics, Conjectures, and Monte Carlo." *Ann. Stat.*, 1(5):799–821, **1973**. |
| 3 | A.E. Beaton, J.W. Tukey. "The Fitting of Power Series, Meaning Polynomials, Illustrated on Band-Spectroscopic Data." *Technometrics*, 16(2):147–185, **1974**. |
| 4 | P.J. Huber. *Robust Statistics*. Wiley, **1981**. |
