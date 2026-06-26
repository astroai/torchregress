# Iteratively Reweighted Least Squares (IRLS)

> ← [Methods Overview](../index.md) | [SIMEX](simex.md) →

IRLS computes robust **sample weights** from current residuals; in **torchregress** the base model is typically refit in an outer training loop using those weights (see complete example below).

---

## Mathematical Background

Standard OLS minimises $\sum_i (y_i - f(x_i))^2$, giving equal weight to all residuals.  IRLS generalises this to:

$$\hat\beta^{(t+1)} = \arg\min_\beta \sum_{i=1}^n w_i^{(t)}\!\left(y_i - f_\beta(x_i)\right)^2$$

where the weights $w_i^{(t)}$ depend on the residuals from iteration $t$:

$$w_i^{(t)} = \psi\!\left(\frac{r_i^{(t)}}{\hat\sigma}\right)$$

and $\psi$ is the weight function (Huber, Tukey, etc.), $\hat\sigma$ is a robust scale estimate (MAD).

!!! info "Convergence & Local Minima"
    Convergence behavior depends on the robust loss used:

    - **Convex Losses (Huber)**: The loss function is convex, and IRLS converges to the unique global M-estimate regardless of initialization.
    - **Non-Convex/Redescending Losses (Tukey, Cauchy)**: The loss function is non-convex. IRLS can get trapped in poor local minima or even diverge if initialized poorly. It is critical to **warm-start** the optimization using a convex loss (e.g., standard OLS or Huber) before switching to Tukey/Cauchy.

!!! warning "Vulnerability to Leverage Points"
    M-estimators (including those solved via IRLS) only downweight outliers based on residuals in the **response space** ($Y$). They are highly sensitive to **leverage points** (outliers in the feature space $X$). A single high-leverage outlier can attract the regression line completely, even if its response residual is relatively small. For robust handling of high-leverage points, bounded-influence estimators (e.g., GM-estimators or Schweppe-type weights) are required.

!!! warning "Practical constraints"
    - **Computational cost per iteration**: Each IRLS iteration requires re-evaluating the model on the full batch to compute residuals and weights, then re-fitting. For large models or datasets, the per-iteration cost is comparable to one full training epoch.
    - **MAD scale collapse**: The robust scale estimate (MAD) can become **zero** if many residuals are identical (e.g., saturated predictions). Division by zero in weight computation produces `NaN` losses. Use `variance_type="predicted"` or `"fixed"` to avoid this, or add a `min_scale` floor.
    - **Warm-start requirement**: For redescending losses (Tukey, Cauchy), IRLS must be warm-started from a convex loss (Huber, OLS). Starting directly with Tukey weights from a random model typically diverges.

---

## Weight Functions

| Function | Formula | Behaviour | Default $c$ |
|:---------|:--------|:----------|:-----------|
| `"huber"` | $w = \min(1,\; \delta/\lvert r\rvert)$ | Downweights, never rejects | $\delta = 1.0$ |
| `"tukey"` | $w = (1 - (r/c)^2)^2$ if $\lvert r\rvert \leq c$; else $0$ | Rejects outliers | $c = 4.685$ |
| `"power"` | $w = 1 / (1 + (\lvert r\rvert/a)^b)$ | Power-law decay | $a=1, b=2$ |
| Custom `Callable` | User-defined | Flexible | — |

Here $r$ denotes the **scaled** residual $r / \hat\sigma$ passed to the weight function.

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

**Returns:** `(y_pred, loss_history, final_precision)` — `final_precision` holds robust precision multipliers for `WeightedMSELoss(..., weights=final_precision)`.

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

## Limitations

1. **Vulnerability to leverage points**: IRLS only downweights outliers based on residuals in the response space ($Y$). It is sensitive to high-leverage points (outliers in $X$). A single high-leverage outlier can attract the regression line even if its residual is small.
2. **MAD scale collapse**: The robust scale estimate (MAD) can become zero if many residuals are identical (e.g., saturated predictions), producing NaN losses. Use `variance_type="predicted"` or `"fixed"` to avoid this.
3. **Warm-start required for redescending losses**: Tukey and Cauchy weight functions are non-convex. IRLS must be warm-started from a convex loss (OLS, Huber) or it diverges.
4. **Per-iteration cost**: Each IRLS iteration requires a full forward pass to compute residuals and weights, comparable to one training epoch. For large models, this multiplies training cost.
5. **Not end-to-end differentiable**: IRLS is an outer-loop reweighting procedure, not a differentiable loss. It cannot be used inside standard `loss.backward()` training — it requires explicit weight computation and re-fitting.

## Recommendations

- **Default weight function**: Start with `weight_fn="huber"` — it is convex, stable, and handles moderate outliers. Upgrade to `"tukey"` only with warm-start.
- **Use robust losses for end-to-end training**: For standard neural network training, prefer [Robust losses](../../losses/robust.md) (Huber, Cauchy, Tukey) which are differentiable and work inside `loss.backward()`.
- **Use IRLS for post-hoc refinement**: IRLS is best applied to refine an already-trained model, especially when you need classical M-estimation diagnostics (influence functions, standard errors).
- **Variance type**: Use `variance_type="predicted"` when the model outputs $\sigma^2$; use `"robust"` (MAD-based) for homoscedastic models without variance heads.

## Next steps

- [Robust losses](../../losses/robust.md) — Huber, Tukey, Cauchy trained end-to-end vs IRLS post-hoc reweighting
- [SIMEX](simex.md) — simulation-based bias correction for measurement error, complementary to robust regression
- [EIV losses](../../losses/eiv.md) — functional and structural error-in-variables losses for noisy inputs
- [Comprehensive comparison](../../examples/comprehensive_comparison.py) — benchmark IRLS against end-to-end robust loss training

---

## References

| # | Reference |
|:-:|:----------|
| 1 | P.W. Holland, R.E. Welsch. ["Robust Regression Using Iteratively Reweighted Least-Squares."](https://doi.org/10.1080/03610927708827533) *Commun. Stat. Theory Methods*, 6(9):813–827, **1977**. |
| 2 | P.J. Huber. ["Robust Regression: Asymptotics, Conjectures, and Monte Carlo."](https://doi.org/10.1214/aos/1176342503) *Ann. Stat.*, 1(5):799–821, **1973**. |
| 3 | A.E. Beaton, J.W. Tukey. ["The Fitting of Power Series, Meaning Polynomials, Illustrated on Band-Spectroscopic Data."](https://doi.org/10.1080/00401706.1974.10489171) *Technometrics*, 16(2):147–185, **1974**. |
| 4 | P.J. Huber. *Robust Statistics*. Wiley, **1981**. |
