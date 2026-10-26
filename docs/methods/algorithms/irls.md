# Iteratively Reweighted Least Squares (IRLS)

> ← [Methods Overview](../index.md) | [SIMEX](simex.md) →

`iteratively_reweighted_fit` is a solver-agnostic IRLS controller. The caller supplies residual and weighted-refit callbacks, so the same controller works with analytic WLS, iterative nonlinear solvers, sparse operators, or compiled training steps without assuming a global dense design matrix.

---

## Mathematical Background

Standard OLS minimises $\sum_i (y_i - f(x_i))^2$, giving equal weight to all residuals.  IRLS generalises this to:

$$\hat\beta^{(t+1)} = \arg\min_\beta \sum_{i=1}^n w_i^{(t)}\!\left(y_i - f_\beta(x_i)\right)^2$$

where the weights $w_i^{(t)}$ depend on the residuals from iteration $t$:

$$u_i^{(t)} = \frac{r_i^{(t)}}{\hat\sigma}, \qquad
w_i^{(t)} = \frac{\psi(u_i^{(t)})}{u_i^{(t)}}.$$

After each update, the controller calls `problem.weighted_fit(state, precision)` and then recomputes residuals from the returned state. With base precision $P_i$, the callback receives $w_iP_i$. For a vector residual block with covariance $C_i$, the robust score uses the Mahalanobis magnitude

$$d_i = \sqrt{r_i^\mathsf{T} C_i^{-1} r_i},$$

and the callback receives the block precision $w_i C_i^{-1}$ unless `base_precision` is supplied explicitly.

!!! info "Convergence & Local Minima"
    Convergence behavior depends on the robust loss used:

    - **Convex Losses (Huber)**: The loss function is convex, and IRLS converges to the unique global M-estimate regardless of initialization.
    - **Non-Convex/Redescending Losses (Tukey, Cauchy)**: The loss function is non-convex. IRLS can get trapped in poor local minima or even diverge if initialized poorly. It is critical to **warm-start** the optimization using a convex loss (e.g., standard OLS or Huber) before switching to Tukey/Cauchy.

!!! warning "Vulnerability to Leverage Points"
    M-estimators (including those solved via IRLS) only downweight outliers based on residuals in the **response space** ($Y$). They are highly sensitive to **leverage points** (outliers in the feature space $X$). A single high-leverage outlier can attract the regression line completely, even if its response residual is relatively small. For robust handling of high-leverage points, bounded-influence estimators (e.g., GM-estimators or Schweppe-type weights) are required.

!!! warning "Practical constraints"
    - **Computational cost per iteration**: Each IRLS iteration requires re-evaluating the model on the full batch to compute residuals and weights, then re-fitting. For large models or datasets, the per-iteration cost is comparable to one full training epoch.
    - **MAD scale collapse**: MAD can be zero when many residuals are identical. `IRLSFitConfig.min_scale` provides an explicit numerical floor; covariance-whitened residuals normally use `scale="none"`.
    - **Warm-start requirement**: For redescending losses (Tukey, Cauchy), IRLS must be warm-started from a convex loss (Huber, OLS). Starting directly with Tukey weights from a random model typically diverges.

---

## Weight Functions

| Function | Formula | Behaviour | Default $c$ |
|:---------|:--------|:----------|:-----------|
| `"huber"` | $w = \min(1,\; \delta/\lvert r\rvert)$ | Downweights, never rejects | $\delta = 1.345$ |
| `"tukey"` | $w = (1 - (r/c)^2)^2$ if $\lvert r\rvert \leq c$; else $0$ | Rejects outliers | $c = 4.685$ |
| `"pseudo_huber"` | $w = [1 + (r/\delta)^2]^{-1/2}$ | Smooth Huber approximation | $\delta = 1.345$ |
| `"cauchy"` | $w = [1 + (r/c)^2]^{-1}$ | Smooth redescending influence | $c=2.3849$ |
| `"barron"` | Barron $\psi(r)/r$ | Continuous robust shape family | $\alpha=0$ |
| Custom `Callable` | User-defined | Flexible | — |

Here $r$ denotes the **scaled** residual $r / \hat\sigma$ passed to the weight function.

---

## API: `iteratively_reweighted_fit`

Implement `IRLSProblem` directly or use `IRLSCallbacks`:

```python
from torchregress.algorithms import IRLSCallbacks, IRLSFitConfig, iteratively_reweighted_fit

problem = IRLSCallbacks(
    residuals_fn=lambda beta: y - X @ beta,
    weighted_fit_fn=lambda beta, precision: torch.linalg.lstsq(
        X * precision.sqrt()[:, None], y * precision.sqrt()
    ).solution,
)

result = iteratively_reweighted_fit(problem, initial_beta, IRLSFitConfig(weight="huber"))
beta = result.state
```

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `problem` | `IRLSProblem` or `IRLSCallbacks` | — | Residual and weighted-refit callbacks; optional covariance, whitening and state-step callbacks |
| `initial_state` | caller-defined | — | Initial model/parameter state |
| `max_iter` | `int` | `50` | Maximum updates when no continuation schedule is supplied |
| `weight` | `str` or `Callable` | `"huber"` | Huber, Pseudo-Huber, Tukey, Cauchy, Barron or custom weights |
| `scale` | `str`, scalar, tensor or callable | `"mad"` | `"mad"`, `"std"`, `"rms"`, `"none"`, fixed or custom scale |
| `base_precision` | `Tensor` or `None` | `None` | Sample, diagonal-output, or block precision multiplied by robust weights |
| `mask` | boolean `Tensor` or `None` | `None` | Sample or element mask; covariance blocks require sample masks |
| `continuation` | tuple of stages | `()` | Ordered convex-to-redescending weight schedule |
| convergence tolerances | `float` or `None` | see API | Relative objective, weight and caller-defined parameter-step criteria |

Kish effective sample size and the weighted mean robust objective are invariant
to a common finite positive multiplier of base precisions. The controller
normalizes only diagnostic sums to avoid overflow or false breakdown; the
callback still receives the original precision units. Explicitly masked
residual entries may be nonfinite and are ignored, including entirely missing
output columns. Nonfinite observed residuals still terminate the fit.

**Returns:** frozen `IRLSFitResult` with the final state, applied robust weights and precision, residuals, scale, robust objective, Kish effective sample size, termination reason and a tuple of frozen per-iteration diagnostics. Termination is one of `"converged"`, `"max_iterations"`, `"empty"`, `"breakdown"`, or `"non_finite"`. A caller-owned rank failure can be reported by raising `IRLSBreakdownError` from `weighted_fit`.

---

## Complete analytic WLS example

```python
import torch
from torchregress.algorithms import IRLSCallbacks, IRLSFitConfig, iteratively_reweighted_fit

torch.manual_seed(42)
n = 1000
X = torch.randn(n, 5)
y = X @ torch.tensor([1.0, -0.5, 0.2, 0.7, -0.3]) + 0.1 * torch.randn(n)
outliers = torch.randperm(n)[:50]
y[outliers] += 5.0 * torch.randn(50)

problem = IRLSCallbacks(
    residuals_fn=lambda beta: y - X @ beta,
    weighted_fit_fn=lambda beta, precision: torch.linalg.lstsq(
        X * precision.sqrt()[:, None], y * precision.sqrt()
    ).solution,
)

result = iteratively_reweighted_fit(
    problem,
    torch.zeros(X.shape[1]),
    IRLSFitConfig(weight="tukey", weight_params={"c": 4.685}),
)

clean = ~torch.isin(torch.arange(n), outliers)
pred = X @ result.state
print(f"IRLS MAE (clean): {(pred[clean] - y[clean]).abs().mean():.4f}")
```

For heteroscedastic vector residuals, provide `covariance_fn=lambda state: covariance` with shape `(samples, target_dim, target_dim)`. For matrix-free or sparse solvers, provide `whiten_fn` and keep any operator representation inside the problem object; the controller never constructs a global design or normal matrix.

## Fixed-prediction compatibility utility

`reweight_fixed_predictions(model, x, y_true, ...)` evaluates a fixed model and returns `(prediction, loss_history, precision)` without refitting. Code that needs genuine IRLS should use `iteratively_reweighted_fit`.

---

## Comparison with Robust Losses

| Approach | Handles Outliers | Retraining | Best For |
|:---------|:---------------:|:----------:|:---------|
| **Robust losses** (Huber, Cauchy, Tukey) | During training | Requires retraining | End-to-end training |
| **Refit-capable IRLS** | Robust outer loop | Yes, through caller callback | Classical, nonlinear, sparse or custom solvers |
| **Fixed-prediction reweighting** | Post-hoc weights | No | Compatibility and diagnostics |
| **CVaR** | Tail focus | During training | Worst-case performance |

!!! tip "When to use IRLS vs robust losses"
    Use **robust losses** when training from scratch.  Use **IRLS reweighting** when you want to iteratively refine an existing model or when you need the classical M-estimation framework (e.g., for statistical inference with influence functions).

---

## Limitations

1. **Vulnerability to leverage points**: IRLS only downweights outliers based on residuals in the response space ($Y$). It is sensitive to high-leverage points (outliers in $X$). A single high-leverage outlier can attract the regression line even if its residual is small.
2. **MAD scale collapse**: The robust scale estimate can become zero if many residuals are identical. The configured `min_scale` prevents division by zero, but callers should diagnose the underlying fit collapse.
3. **Warm-start required for redescending losses**: Tukey and Cauchy weight functions are non-convex. IRLS must be warm-started from a convex loss (OLS, Huber) or it diverges.
4. **Per-iteration cost**: Each IRLS update requires a residual evaluation and caller-owned refit. Large-model callbacks should control batching and stopping internally.
5. **Outer-loop controller**: The controller detaches residual diagnostics and is not an end-to-end differentiable loss. The weighted callback may use autograd internally.

## Recommendations

- **Default weight function**: Start with `weight="huber"` — it is convex, stable, and handles moderate outliers. Upgrade to `"tukey"` only with warm-start.
- **Use robust losses for end-to-end training**: For standard neural network training, prefer [Robust losses](../../losses/robust.md) (Huber, Cauchy, Tukey) which are differentiable and work inside `loss.backward()`.
- **Use continuation for redescending weights**: Start with a Huber stage and switch to Tukey, Cauchy, or Barron through `IRLSContinuationStage`.
- **Use covariance whitening for vector targets**: A full per-sample covariance gives exact block Mahalanobis scores; a custom whitening callback supports structured operators.

## Next steps

- [Robust losses](../../losses/robust.md) — Huber, Tukey, and Cauchy trained end-to-end vs refit-capable IRLS
- [SIMEX](simex.md) — simulation-based bias correction for measurement error, complementary to robust regression
- [EIV losses](../../losses/eiv.md) — functional and structural error-in-variables losses for noisy inputs
- [Comprehensive comparison](https://github.com/astroai/torchregress/blob/main/examples/comprehensive_comparison.py) — benchmark IRLS against end-to-end robust loss training

---

## References

| # | Reference |
|:-:|:----------|
| 1 | P.W. Holland, R.E. Welsch. ["Robust Regression Using Iteratively Reweighted Least-Squares."](https://doi.org/10.1080/03610927708827533) *Commun. Stat. Theory Methods*, 6(9):813–827, **1977**. |
| 2 | P.J. Huber. ["Robust Regression: Asymptotics, Conjectures, and Monte Carlo."](https://doi.org/10.1214/aos/1176342503) *Ann. Stat.*, 1(5):799–821, **1973**. |
| 3 | A.E. Beaton, J.W. Tukey. ["The Fitting of Power Series, Meaning Polynomials, Illustrated on Band-Spectroscopic Data."](https://doi.org/10.1080/00401706.1974.10489171) *Technometrics*, 16(2):147–185, **1974**. |
| 4 | P.J. Huber. *Robust Statistics*. Wiley, **1981**. |
