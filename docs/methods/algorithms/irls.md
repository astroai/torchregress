# Iteratively Reweighted Least Squares (IRLS)

> ← [Methods Overview](../index.md) | [SIMEX](simex.md) →

`iteratively_reweighted_fit` is a solver-agnostic IRLS controller; `batched_iteratively_reweighted_fit` runs many independent small fits at once (see [Batched IRLS](#batched-irls-for-many-independent-problems)). The caller supplies residual and weighted-refit callbacks, so the same controller works with analytic WLS, iterative nonlinear solvers, sparse operators, or compiled training steps without assuming a global dense design matrix.

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
| `"student_t"` | $w = (\nu + 1)/(\nu + r^2)$ | Exact Student-$t$ likelihood weight (not normalised to 1 at $r=0$); `nu=math.inf` gives unit weights | $\nu = 5$ |
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
| `weight` | `str` or `Callable` | `"huber"` | Huber, Pseudo-Huber, Tukey, Cauchy, Barron, Student-t or custom weights |
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

## Batched IRLS for many independent problems

`batched_iteratively_reweighted_fit` fits $B$ independent rows at once — one spectrum, light curve, source, or pixel stack per row — each with its own model $m_b(\theta_b)$, $N$ observations $y_{bi}$ with **known** precisions $\lambda_{bi} = 1/\sigma_{bi}^2$, a mask, and a few parameters. Rows sit on the leading axis; there is no per-row Python loop.

### Objective

With the variance model $1/\lambda_{\mathrm{eff},bi} = 1/\lambda_{bi} + \eta^2 \bar m_{bi}^2$ ($\bar m$ the previous iterate's model, $\eta$ a fractional model error; [`effective_precision`](../../losses/robust.md#student-t-likelihood-with-known-precisions)) and standardized residuals $u_{bi} = (y_{bi} - m_{bi})\sqrt{\lambda_{\mathrm{eff},bi}}$, every unfinished row takes the weighted step

$$
\theta_b^{(t+1)} = \arg\min_{\theta}\; \sum_{i \in \mathcal O_b} w_{bi}^{(t)}\,\bigl(y_{bi} - m_{bi}(\theta)\bigr)^2 + 2\,\Pi_b(\theta),
\qquad
w_{bi}^{(t)} = \lambda_{\mathrm{eff},bi}\,\frac{\psi(u_{bi}^{(t)})}{u_{bi}^{(t)}},
$$

where $\mathcal O_b$ is the set of observed entries with positive precision and $\Pi_b$ an optional penalty (e.g. a Gaussian prior $\tfrac12\sum_m c_m^2/\sigma_m^2$). For the default `weight="student_t"`, $w = \lambda_{\mathrm{eff}}(\nu+1)/(\nu + u^2)$ and the reported objective is the exact negative log-likelihood

$$
\mathcal L_b = \sum_{i \in \mathcal O_b}\left[\log\Gamma\!\left(\tfrac{\nu}{2}\right) - \log\Gamma\!\left(\tfrac{\nu+1}{2}\right) + \tfrac12\log(\nu\pi) - \tfrac12\log\lambda_{\mathrm{eff},bi} + \tfrac{\nu+1}{2}\log\!\left(1 + \frac{u_{bi}^2}{\nu}\right)\right] + \Pi_b(\theta_b),
$$

with `nu=math.inf` giving the Gaussian. At $\eta = 0$ the weighted objective majorises $\mathcal L_b$ (since $\log(1 + z/\nu)$ is concave in $z$), so every accepted step lowers it \[5\]. For $\eta > 0$ the iteration is a fixed point in $\bar m$.

### Convergence certificate

A row converges when the **full** (undamped) step of its weighted solve at the current weights moves every parameter by less than `step_tol` of its standard error,

$$
\max_k \frac{\lvert \theta^{\text{full}}_{bk} - \theta^{(t)}_{bk}\rvert}{\mathrm{se}_{bk}} < \mathtt{step\_tol},
$$

after at least `min_iter` iterations whenever the weights depend on the residuals. With a `damped_solve` (Levenberg–Marquardt \[7, 8\]) a full step is accepted only if it lowers the weighted objective (its value at the start of the iteration comes from the previous weight refresh, not from another model evaluation); otherwise the per-row damping grows ×10 until a damped step does. A damped or rejected step never certifies convergence: a short damped step says nothing about the distance to the fixed point. A row for which no damping helps stops as `STALLED`, unless the solver's `predicted_objective` predicts a non-negative decrease (up to round-off) and shows that the decrease still reachable is below the float resolution of the objective; a model that predicts an increase certifies nothing.

### Ownership

The caller owns the model and the weighted solve, so bounds, priors, Gauss–Newton linearisations and operators stay in caller code. [`solve_normal_equations`](../../api/algorithms.md#irls-robust-regression) covers the common linear case: batched Jacobi-scaled Cholesky with a flagged ridge fallback and standard errors (by default the ridge is taken only where the factorisation fails, so a badly conditioned but identified row is never biased; opt-in `rank_tol > 0` also takes it where the smallest-to-largest eigenvalue ratio of the scaled final system is at most `rank_tol`, e.g. `n·eps` for rows singular to working precision; solution and standard errors on ridged rows are those of the ridged system, i.e. regularised), or, with a `nonnegative` mask, batched Lawson–Hanson NNLS \[6\] with a warm-started free set. With bounds the rank test applies to the final free set only, after the active-set steps (which ridge failed factorisations alone); a flagged row is re-solved as the strictly convex ridged problem, so its solution does not depend on the warm start.

| Callback / field | Signature | Role |
|:-----------------|:----------|:-----|
| `predict` | `(theta, rows) -> model [b, N]` | Model of the given rows |
| `solve` | `(theta, weights, target, rows) -> BatchedIRLSStep` | Full step at fixed weights: `theta`, `standard_errors`, optional `failed`, `predicted_objective` and `ridge_used` |
| `damped_solve` | `(theta, weights, target, rows, damping) -> theta` | Optional damped step for per-row damping $\lambda$ |
| `penalty` | `(theta, rows) -> [b]` | Optional penalty $\Pi_b$ in NLL units |
| `exact_solve` | `bool` | `solve` is the exact minimiser (linear LS, NNLS): one solve converges when the weights are fixed, and is taken without the decrease test |

`rows` is `None` when a callback gets the whole batch in order, otherwise the global indices of the rows it gets: finished rows leave the active set, so a batch costs $\sum_b \text{iterations}_b$ row solves. Every callback of one iteration (`solve`, the weight refresh's `predict` and `penalty`) receives the identical `rows` object, which is reused across iterations until rows finish; every callback of one decrease test or damping attempt (a subset of the rows) also shares one object. A caller can therefore cache its per-row gathers (basis, design, calibration polynomials) in a one-entry cache keyed on `rows is last_rows`, as the example below does. `weights` and `target` are the controller's working buffers, valid during the call only. Each controller iteration has one host synchronisation (with a `damped_solve`, plus one for the decrease test and one per damping attempt), excluding the caller's solve: the non-negative path of `solve_normal_equations` costs about one more per active-set step, at $O(b\,n^3)$ per step on the rows still pending, and reuses each row's last factorisation for the standard errors (the unconstrained path has none on an accelerator; on the CPU, where they are free, both paths read the failure flags to ridge and refactorise only the rows that need it).

`BatchedIRLSFitConfig` sets `weight` (default `"student_t"`), `weight_params` (default `{}`; `nu=5` for `student_t` unless given), `eta`, `max_iter`, `step_tol` (largest full-step change in standard errors, default `1e-3`; unlike the relative-norm `IRLSFitConfig.parameter_tol` it is per parameter), `min_iter` (default 2), `min_observations` (default: parameters + 1) and the damping schedule (`max_damping`, `damping_start`, `damping_min`).

`BatchedIRLSFitResult` holds per-row `theta`, `standard_errors`, `model`, `weights`, `effective_precision`, `objective` (NLL + penalty), `penalty`, `status` (`IRLSRowStatus`: `CONVERGED`, `MAX_ITERATIONS`, `STALLED`, `FAILED`, `EMPTY`; these map onto `IRLSTermination` as converged, max_iterations, —, breakdown/non_finite, empty), `iterations`, `n_observations` and `ridge_used` (any solve of the row was regularised).

!!! warning "Standard errors are conditional, not asymptotic"
    `standard_errors` are those of the final weighted least-squares solve: they set the convergence scale. They are not asymptotic $t_\nu$ errors. The Student-$t$ weight $(\nu+1)/(\nu+u^2)$ is the IRLS/MM weight $\psi(u)/u$, whose mean is 1 under $t_\nu$ noise, so $X^\mathsf{T} W X$ estimates the Gaussian information $X^\mathsf{T}\Lambda_{\mathrm{eff}} X$ rather than the $t_\nu$ Fisher information $\tfrac{\nu+1}{\nu+3} X^\mathsf{T}\Lambda_{\mathrm{eff}} X$: the returned errors are too small by about $\sqrt{(\nu+3)/(\nu+1)}$ (15% at $\nu = 5$) under $t_\nu$ noise and by a few percent under Gaussian noise. For asymptotic errors, build the normal matrix from `result.effective_precision` without the IRLS weights and multiply it by [`student_t_efficiency(nu, noise)`](../../losses/robust.md#student-t-likelihood-with-known-precisions) before inverting, as in the example below, or use a sandwich estimate.

!!! warning "Fail closed"
    Rows with a non-finite target or precision on an observed entry, a non-finite start, too few observations, a failed or non-finite solve, or non-finite weights or objective are `FAILED`; rows with no usable observation are `EMPTY`. Their outputs are `NaN`, never zeros that look like a fit. Masked entries may hold any value, `NaN` included, and model values off the likelihood (masked or zero-precision entries) are not checked, so a template may be undefined there.

### Complete batched Student-t example

```python
import math

import torch
from torchregress.algorithms import (
    BatchedIRLSFitConfig,
    BatchedIRLSProblem,
    BatchedIRLSStep,
    batched_iteratively_reweighted_fit,
    solve_normal_equations,
)
from torchregress.losses import student_t_efficiency

torch.manual_seed(0)
B, N, n = 1000, 512, 4
grid = torch.linspace(-1.0, 1.0, N, dtype=torch.float64)
design = torch.stack([grid**k for k in range(n)], -1).expand(B, N, n).contiguous()
truth = torch.randn(B, n, dtype=torch.float64)
precision = torch.full((B, N), 400.0, dtype=torch.float64)
target = (design @ truth.unsqueeze(-1)).squeeze(-1) + 0.05 * torch.randn(B, N, dtype=torch.float64)
target[:, ::37] += 1.0  # outliers
mask = torch.rand(B, N) > 0.05  # missing pixels


gathered = {"rows": None, "design": design}


def take_design(rows):
    # Callbacks of one iteration (and of one damping attempt) get the same `rows`
    # object, which only changes when rows finish: gather the design once per set.
    if rows is None:
        return design
    if gathered["rows"] is not rows:
        gathered["rows"], gathered["design"] = rows, design[rows]
    return gathered["design"]


def predict(theta, rows):
    return (take_design(rows) @ theta.unsqueeze(-1)).squeeze(-1)


def solve(theta, weights, y, rows):
    x = take_design(rows)
    normal = x.transpose(1, 2) @ (x * weights.unsqueeze(-1))
    rhs = (x.transpose(1, 2) @ (weights * y).unsqueeze(-1)).squeeze(-1)
    out = solve_normal_equations(normal, rhs)
    return BatchedIRLSStep(
        out.solution, out.standard_errors, failed=out.singular, ridge_used=out.ridge_used
    )


problem = BatchedIRLSProblem(predict, solve, exact_solve=True)
result = batched_iteratively_reweighted_fit(
    problem,
    torch.zeros(B, n, dtype=torch.float64),
    target,
    precision,
    mask,
    BatchedIRLSFitConfig(weight_params={"nu": 5.0}),
)
print(result.converged.all().item(), result.iterations.double().mean().item())
# Asymptotic t_5 errors under Gaussian noise: efficiency x information at λ_eff.
lam = result.effective_precision  # zero off the likelihood
info = student_t_efficiency(5.0) * design.transpose(1, 2) @ (design * lam.unsqueeze(-1))
se = torch.linalg.inv(info).diagonal(dim1=-2, dim2=-1).sqrt()
print("max |pull|:", ((result.theta - truth) / se).abs().max().item())
gaussian = batched_iteratively_reweighted_fit(
    problem, torch.zeros(B, n, dtype=torch.float64), target, precision, mask,
    BatchedIRLSFitConfig(weight_params={"nu": math.inf}),
)
print("Gaussian NLL - Student-t NLL:", (gaussian.objective - result.objective).mean().item())
```

For a bilinear or otherwise nonlinear model, make `solve` return one Gauss–Newton step (solving the linearised weighted problem with `solve_normal_equations`, bounds included) together with its `predicted_objective`, and supply `damped_solve` that adds $\lambda\,\mathrm{diag}(N)$ to the same normal matrix. The controller then accepts, damps or stalls each row and certifies convergence only on the full step.

!!! info "Throughput"
    `tools/benchmark_batched_irls.py` fits Student-t linear rows with $N = 4096$ observations and $n = 8$ parameters (float64, CPU, 2 threads, 2% outliers): 260–730 rows/s for $B$ from 100 to 10,000 on a shared 4-CPU host (9 iterations, every row converged), 4–12× a per-row loop over `iteratively_reweighted_fit` in the same run. Absolute rates depend on machine load; the ratio is the stable figure. With `--max-outlier-fraction 0.3` (each row's outlier fraction drawn in $[0, 0.3]$, so rows converge after 8–11 iterations) finished rows leave the solve calls: 10,209 row solves instead of 11 × 1,000 at $B = 1000$. Memory grows linearly in $B$. Besides the caller's target and precision, the controller holds about three $[B, N]$ working buffers of the running rows plus at most two transients, and the result returns three $[B, N]$ outputs after those are released: about $(2 + 5) \cdot N \cdot 8$ bytes per row (≈ 0.23 MB at $N = 4096$). The caller's operator comes on top: a materialised $[B, N, n]$ design adds $n \cdot N \cdot 8$ bytes per row (≈ 0.26 MB at $n = 8$, ≈ 0.5 MB per row in total), plus whatever temporaries its solve builds (the benchmark's chunked normal matrices add a fixed ≈ 0.27 GB).

### When to use which controller

| | `iteratively_reweighted_fit` | `batched_iteratively_reweighted_fit` |
|:--|:--|:--|
| Problems per call | One | Many independent rows |
| Noise scale | Estimated (MAD, std, …) or fixed | Known per-observation precision, optional $\eta^2 m^2$ term |
| Residual structure | Scalar, vector blocks, covariance whitening | Scalar per observation |
| Convergence | Relative objective / weight / parameter change | Full step in standard-error units, per row |
| Continuation schedule | Yes | No (one family per call) |
| Damping / acceptance | Caller-side | Built in (Levenberg–Marquardt hook) |
| Reported objective | Weighted mean robust loss | Exact Student-$t$ / Gaussian NLL per row (+ penalty) |

---

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
| 5 | K.L. Lange, R.J.A. Little, J.M.G. Taylor. ["Robust Statistical Modeling Using the t Distribution."](https://doi.org/10.1080/01621459.1989.10478852) *J. Amer. Statist. Assoc.*, 84(408):881–896, **1989**. |
| 6 | C.L. Lawson, R.J. Hanson. *Solving Least Squares Problems*. Prentice-Hall, **1974** (SIAM reprint, 1995). |
| 7 | K. Levenberg. ["A Method for the Solution of Certain Non-Linear Problems in Least Squares."](https://doi.org/10.1090/qam/10666) *Q. Appl. Math.*, 2(2):164–168, **1944**. |
| 8 | D.W. Marquardt. ["An Algorithm for Least-Squares Estimation of Nonlinear Parameters."](https://doi.org/10.1137/0111030) *J. SIAM*, 11(2):431–441, **1963**. |
