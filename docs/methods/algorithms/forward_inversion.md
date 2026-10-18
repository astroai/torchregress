# Forward Generative Inversion ($y \to x$)

> ← [LatentNN](latentnn.md) | [Regression Calibration](rc.md) → [SIMEX](simex.md)

[`ForwardInversionRegressor`](../../api/algorithms.md) estimates regression in the **forward** direction: a decoder $g_\theta(y) \to x$ is trained to reproduce the observed features from the labels under an explicit measurement-error model, and the labels of a new object are then recovered by solving a weighted least-squares problem at test time. It addresses one specific failure mode of discriminative regression — attenuation when the features are much noisier than the labels — and it has known costs and failure modes of its own.

This page documents the estimator's assumptions, the statistical literature it belongs to, what the unit tests in this repository do and do not establish, and the practical hazards of test-time optimization.

!!! abstract "Estimator in one paragraph"
    Train $g_\theta(y)$ on pairs $(y_i, x_i)$ with a likelihood for the observables; for a new object, solve $\hat{y} = \arg\min_y \lVert \Sigma_x^{-1/2}(x_{\text{obs}} - g_\theta(y)) \rVert_2^2$ and report the local Fisher covariance. The label vector is a latent variable of a forward model, recovered by optimization rather than by a learned inverse map.

---

## Scope and evidence status

!!! info "What the evidence in this repository supports"
    The implementation is tested on **synthetic problems with known generative models**:

    * a linear decoder, where the inversion agrees with the analytic Gaussian MLE to numerical precision;
    * a univariate attenuation example, where a direct regression is biased at the extremes and the generative inversion is not;
    * covariance checks against the analytic Fisher information and against Monte-Carlo draws;
    * a nonlinear (quadratic) decoder, where Gauss–Newton and L-BFGS recover the labels;
    * a robust-likelihood experiment with injected defective pixels (10% of pixels corrupted; Student-t training recovers the true slope 2.00 where a Gaussian likelihood gives 1.13).

    There is **no large-scale survey benchmark** in this repository and no claim is made that forward inversion outperforms discriminative models on real catalogues. Such a claim would require matched-capacity, matched-training-budget comparisons on representative data, including the tail slices that motivate the method.

---

## Why inverse regression can differ from direct regression

Consider the univariate noisy-input problem $w = x + \epsilon$, $\epsilon \sim \mathcal N(0, \sigma_x^2)$, with a linear relation $y = \alpha + \beta x$. Least squares of $y$ on the *observed* $w$ estimates

$$\hat{\beta}_{\text{disc}} = \beta \cdot \frac{\operatorname{Var}(x)}{\operatorname{Var}(x) + \sigma_x^2} = \beta \cdot \frac{1}{1 + \mathrm{SNR}^{-1}},$$

so the fitted slope is diluted by a factor $\lambda < 1$ and predictions of extreme labels are compressed toward the mean. This attenuation persists with training-set size, label precision and label-distribution shape. In the multivariate case it becomes a matrix shrinkage of the regression coefficients.

Fitting the *other* direction — a model of $x$ given $y$, then solving it for $y$ — is the **inverse regression** approach to statistical calibration (Krutchkoff 1967; Osborne 1991). Berkson (1950) pointed out that the two regressions estimate different things and need not agree. The classical inverse estimator is **not unbiased**; its justification is a mean-squared-error argument, and the bias/MSE balance depends on the design and the noise level (Krutchkoff 1967; Osborne 1991; Fuller 1987). The estimator here is the likelihood version of that idea:

* under the assumed observation model and an identifiable, correctly specified decoder, the inversion is the maximum-likelihood estimate of $y$; it is consistent as the information in the observables grows, but it carries the usual finite-sample bias of nonlinear MLEs;
* if the decoder or the noise model is misspecified, that bias propagates into $\hat y$ — unlike the direct regression, which optimizes the prediction objective directly.

So "no attenuation" should be read as *the estimator does not inherit the specific dilution factor of inverse regression*, not as *the estimator is unbiased in general*.

---

## Relation to auto-decoders and energy-based models

Structurally the estimator is not a new architecture:

* **auto-decoder** (Park et al. 2019): a decoder maps a low-dimensional code (the labels) to the high-dimensional observation; the code for a new sample is obtained by optimizing the decoder objective at test time instead of by a separate encoder. Here the codes are *observed* during training (supervised), which makes training easier than unsupervised auto-decoding, but inference is still an optimization problem.
* **energy-based prediction** (LeCun et al. 2006): prediction is $\hat{y} = \arg\min_y E_\theta(x, y)$ with $E_\theta(x, y) = \lVert \Sigma_x^{-1/2}(x - g_\theta(y)) \rVert_2^2$.
* **non-amortised counterpart of an auto-encoder**: an auto-encoder learns an explicit encoder (amortised inference); here the encoder is implicit. If inference cost matters, an encoder can be trained on the training set or on simulated inversions to amortise the fast path, with the decoder optimization used only for refinement. `ForwardInversionRegressor` implements only the decoder + inversion path.

The formulation is also a MAP inverse problem (Tarantola 2005), and closely related to likelihood-based event reconstruction in high-energy physics (matrix element method, Gainer et al. 2013) and to simulation-based inference more broadly (Cranmer et al. 2020).

---

## Training: an explicit observation model

The default objective is the **full heteroscedastic Gaussian likelihood** with known per-observable measurement error $\sigma_{x,i,d}$ and an optional learnable per-observable model-mismatch variance $s_d^2$:

$$\mathcal{L}_{\text{Gauss}}(\theta) = \frac{1}{ND}\sum_{i=1}^{N}\sum_{d=1}^{D}\left[\frac{(x_{i,d} - g_{\theta,d}(y_i))^2}{\sigma_{x,i,d}^2 + s_d^2} + \ln\!\left(\sigma_{x,i,d}^2 + s_d^2\right)\right].$$

The log-determinant is not a constant: it lets the fit learn $s_d$ and down-weight observables that the surrogate cannot reproduce. Two alternatives are available:

| `likelihood=` | Objective | Appropriate when |
|:--|:--|:--|
| `"gaussian"` (default) | Full heteroscedastic Gaussian NLL above | Per-observable errors are known and approximately Gaussian |
| `"student_t"` | $\tfrac12\ln v + \tfrac{\nu+1}{2}\ln\!\left(1 + \frac{r^2}{\nu v}\right)$, $v = \sigma_x^2 + s_d^2$, $\nu$ = `degrees_of_freedom` | Training data contain heavy-tailed outliers (defective pixels, cosmic rays, telluric residuals) that should be down-weighted |
| `"wls"` | Heteroskedastic weighted least squares $\tfrac12 r^2 / (\sigma_x^2 + s_d^2)$, no log-determinant | The noise is known and fixed; requires `learn_noise=False` and known `sigma_x` |

Plain unweighted MSE is not the default; it is obtained only by passing a constant `sigma_x` with `learn_noise=False`, and is the least informative choice when per-pixel errors are known. The full likelihood is what allows different observables to be weighted by their stated reliability.

Because $y$ is the network input, the training label distribution $p_{\text{train}}(y)$ shapes the *design* of the surrogate and, through it, the regions where the decoder is constrained. It does not enter the per-object estimator as a prior factor, unlike a discriminative model whose learned inverse map absorbs the marginal of $y$.

!!! info "Label noise is not modelled"
    The likelihood assumes exact training labels. If labels have non-negligible errors, use the noisy-label losses of [`torchregress.losses`](../../losses/noisy_labels.md) in the discriminative comparison, or extend the forward model hierarchically; the current estimator has no label-noise term.

---

## Inference: weighted least squares

For a new object the estimate is

$$\hat{y} = \arg\min_{y\in\mathbb R^{D_y}} \; \lVert \Sigma_x^{-1/2}\,(x_{\text{obs}} - g_\theta(y)) \rVert_2^2, \qquad \Sigma_x = \operatorname{diag}(\sigma_{x,d}^2 + s_d^2),$$

which is the Gaussian MLE (and the fixed-scale Student-t MLE). The training likelihood changes the fitted decoder and noise floor, not the form of the inversion. Two local optimizers are implemented:

=== "Damped Gauss–Newton (default)"

    With $r(y) = x_{\text{obs}} - g_\theta(y)$ and Jacobian $J_y = \partial g_\theta/\partial y$, each iteration solves the damped normal equations

    $$\left(J_y^\top \Sigma_x^{-1} J_y + \lambda\,\mathbf I\right)\delta = J_y^\top \Sigma_x^{-1} r(y), \qquad y \leftarrow y + \delta,$$

    with a Levenberg–Marquardt-style trust region ($\lambda$ adapted per object) and a backtracking line search. The Jacobian is evaluated with forward-mode autodiff (`torch.func.jacfwd` + `vmap`), costing $O(D_y)$ forward passes — the cheap direction when $D_y \ll D$.

=== "L-BFGS"

    `method="lbfgs"` optimizes the same objective with `torch.optim.LBFGS` (strong-Wolfe line search). It is useful when the decoder is not `vmap`-safe or the problem is strongly non-quadratic; `jacobian_method="loop"` computes Jacobians per sample with forward-mode autodiff.

The local covariance is the inverse Fisher information at the solution,

$$\widehat{\operatorname{Cov}}(\hat y) = \left(J_y^\top \Sigma_x^{-1} J_y\right)^{-1}, \qquad J_y = \nabla_y g_\theta(y)\big|_{\hat y},$$

returned by [`predict_with_covariance`](../../api/algorithms.md). It is a **linearised, local** uncertainty: it propagates the measurement noise under the assumption that the objective is well approximated by a quadratic around the solution. It does not include decoder model error (except through the empirical $s_d^2$), and it can be badly misleading when the problem is weakly identified or strongly nonlinear.

---

## Test-time optimization is nonlinear and often non-convex

This is a central practical caveat: **at inference time the estimator solves a nonlinear least-squares problem, and neither implemented optimizer has a global-convergence guarantee.**

* For a nonlinear decoder, $E(y) = \lVert \Sigma_x^{-1/2}(x_{\text{obs}} - g_\theta(y)) \rVert_2^2$ can be multimodal, can have flat valleys, and can be locally ill-conditioned. Different initializations can converge to different stationary points that fit the same observation comparably well.
* **Gauss–Newton** has local quadratic convergence only near a solution with small residual and a well-conditioned Jacobian; far from it, the undamped step can overshoot or diverge. Levenberg–Marquardt damping and backtracking improve robustness but are heuristics, not guarantees (Bates & Watts 1988; Nocedal & Wright 2006).
* **L-BFGS** is a local descent method: it converges to a stationary point that depends on initialization and line-search behavior, and it can stall on non-smooth or badly scaled objectives. It should be treated as a cross-check, not as a trust-region guarantee.
* **Identifiability** can fail structurally (label directions that the decoder cannot distinguish) or practically (label directions with very little information at the observed noise level). In either case the optimizer may still return a well-converged but wrong estimate; the Fisher condition number and profile likelihood are the appropriate diagnostics (Raue et al. 2009).
* **The reported covariance is local.** Strong nonlinearity, bounded/periodic label directions, or multiple minima make the linearised covariance unreliable. For interval estimates use profile likelihood or parametric bootstrap; for multimodal posteriors use a multi-start or sampling approach.

### Recommended protocol

1. **Initialise sensibly.** The default is the training-label mean; for extreme or structured objects pass `init=` from a physical prior, a nearest-neighbour label, or an amortised encoder prediction. Run several initializations and retain the solution with the lowest weighted SSE.
2. **Inspect the diagnostics.** `predict(..., return_info=True)` reports per-object `final_weighted_sse`, `gradient_norm`, `relative_gradient_norm`, and `fisher_condition` (condition number of $J^\top\Sigma_x^{-1}J$), plus the `max_iter` cap. There are no universal thresholds, but a large `relative_gradient_norm`, a very large `fisher_condition`, or a large final SSE relative to the stated noise indicates a problem with that object.
3. **Cross-check the optimizers.** Run Gauss–Newton and L-BFGS (and two initializations) and compare the weighted SSE. Agreement on the estimate is weak but useful evidence; disagreement is a red flag.
4. **Check the model, not only the optimizer.** Hold out objects with known labels, inject realistic systematics, and compare the recovered bias and scatter against a discriminative baseline. Optimizer convergence cannot repair a wrong decoder.
5. **Use float64 for convergence studies.** In float32, gradient norms and covariance entries can be dominated by round-off long before the tolerance is meaningful; the internal tolerances are heuristics. A `relative_gradient_norm` of order $10^{-4}$–$10^{-3}$ in float32 is often the round-off floor rather than a statement about the optimum; rerun in float64 when stationarity is part of the reported result.
6. **Mask known-bad observables.** Defective pixels are not down-weighted by the Gaussian inversion; pass `mask=` for them. The Student-t likelihood robustifies training only, not the test-time objective.
7. **Report the optimization settings.** For scientific use, record the optimizer, initialization strategy, iteration budget, convergence statistics, masking and noise model, so the estimates are reproducible and auditable.

!!! warning "Do not interpret the point estimate as a posterior mode without checking"
    `predict` returns a local optimizer's output. If the diagnostics indicate non-stationarity, ill-conditioning or optimizer disagreement, the appropriate output is a flag, a wider interval, or a multi-start solution — not a silent point estimate.

---

## What changes relative to direct regression

| Effect | Discriminative $x \to y$ | Forward inversion $y \to x$ |
|:-------|:------------------------|:----------------------------|
| Input-noise attenuation | Dilution factor $\lambda = 1/(1+\mathrm{SNR}^{-1})$ in the linear case | Not present in the estimator's fixed point; finite-sample and misspecification bias remain |
| Training label distribution $p_{\text{train}}(y)$ | Absorbed into the learned inverse map | Shapes the surrogate design; no explicit prior factor at inversion |
| Extrapolation outside the training label range | Predictions collapse toward populated regions | Also unreliable; the decoder is unconstrained where it saw no labels |
| Label measurement error | Inflates residual variance | Not corrected (labels assumed exact) |
| Model misspecification | Optimizes the prediction objective directly | Propagates decoder/noise-model bias into $\hat y$ |
| Identifiability | Learned map can hide unidentifiable directions | Exposed at test time; diagnostics available |
| Cost | One forward pass per object | Surrogate fit + iterative per-object optimization |

---

## When a discriminative model is the better choice

There is no general advantage of the generative direction; the following are reasons to expect a discriminative model to win at scale:

* **Unmodelled structure.** A forward likelihood must assign probability to every observed channel. Unmodelled systematics (bad columns, calibration drifts, missing physics, correlated detector effects) enter the energy unless they are masked or modelled, whereas a discriminative network can learn to down-weight or ignore defective regions. On messy data this robustness can dominate the attenuation benefit.
* **Misspecification.** The generative estimator inherits the decoder's bias; the discriminative estimator at least optimizes the prediction objective. The classical generative-versus-discriminative trade-off (Ng & Jordan 2002) applies: generative models can be statistically efficient when the model is correct, discriminative models are often more robust when it is not.
* **Amortisation and throughput.** A surrogate fit plus iterative Jacobian passes per object is far more expensive than one amortised forward pass. For $10^8$–$10^9$ objects, a discriminative encoder (possibly with robust or EIV-corrected losses, or with a likelihood-based refinement on a subset) is usually the only practical option.
* **Amortising changes the bias.** An explicit encoder trained on the training set is again a discriminative estimator for the fast path, and inherits the same attenuation issue; the likelihood inversion only recovers the dynamic range for the objects on which it is run.

A defensible strategy is to compare both on matched splits and budgets, report tail-slice metrics and recovered dynamic range, and use the generative path where its assumptions hold and its diagnostics are clean — not as a default.

---

## Comparison with other EIV methods

| Method | Direction | Needs noise covariance | Cost | Best when |
|:-------|:----------|:-----------------------|:-----|:----------|
| Plain regression | $x \to y$ | No | 1 forward pass | Noise is negligible |
| [Regression Calibration](rc.md) | $x \to y$ + moment correction | Yes ($\Sigma_x$) | 1 forward pass | Features approximately Gaussian, noise known |
| [SIMEX](simex.md) | $x \to y$, K noise levels | Yes ($\Sigma_x$) | $K$ fits | Extrapolation function trusted, moderate dimensionality |
| [LatentNN](latentnn.md) | $x \to y$ with latent inputs | Yes ($\Sigma_x$) | 1 fit, $N\times D$ latent parameters | Moderate $D$, validation data available |
| [Split / corrected score](../../losses/split_eiv.md) | $x \to y$ with EIV loss | Split: none; corrected score: $\Sigma_x$ | 1–2 forward passes per step | Repeat exposures or known per-pixel errors |
| **Forward inversion** | $y \to x$ | Yes ($\sigma_x$ + optional $s_d$) | Surrogate fit + per-object optimization | Accurate labels, high-dimensional noisy observables, low-dimensional labels, trusted observation model |

---

## Example

```python
import torch
from torchregress.algorithms import ForwardInversionRegressor

torch.manual_seed(0)

# Synthetic problem with a known linear generative model.
n_labels, n_observables, n = 2, 8, 512
y = torch.rand(n, n_labels) * 4 - 2
A = torch.randn(n_observables, n_labels)
X = y @ A.T + 0.05 * torch.randn(n, n_observables)  # known pixel noise = 0.05

reg = ForwardInversionRegressor(
    n_labels=n_labels,
    n_observables=n_observables,
    hidden_dims=(64, 64),
    sigma_x=0.05,
    learn_noise=True,
    likelihood="gaussian",     # full heteroscedastic likelihood (default)
    epochs=150,
    batch_size=128,
    seed=0,
)
reg.fit(X, y)

# Point estimates, local covariance, and inversion diagnostics.
y_hat, cov = reg.predict_with_covariance(X[:8])
y_hat, info = reg.predict(X[:8], return_info=True)
print("predicted labels:", y_hat[:3])
print("marginal std:", torch.diagonal(cov, dim1=1, dim2=2).sqrt()[:3])
print("final weighted SSE:", info["final_weighted_sse"])
print("relative gradient norm:", info["relative_gradient_norm"])
print("Fisher condition:", info["fisher_condition"])
```

A frozen, already-trained surrogate can be inverted without retraining:

```python
from torchregress.algorithms import ForwardInversionRegressor

frozen = ForwardInversionRegressor(
    decoder=reg.decoder_.eval(),
    sigma_x=0.05,
    learn_noise=False,
    epochs=1,
)
frozen.fit(X, y, train_decoder=False)
y_hat = frozen.predict(X, method="lbfgs")
```

For data with defective pixels, train with the robust likelihood and **mask the
defects at inversion time** (reusing `X` and `y` from above):

```python
X_with_defects = X.clone()
X_with_defects[::10] += 20.0   # e.g. cosmic-ray hits / bad columns
defect_mask = torch.ones_like(X_with_defects, dtype=torch.bool)
defect_mask[::10] = False

robust = ForwardInversionRegressor(
    n_labels=n_labels,
    n_observables=n_observables,
    hidden_dims=(64, 64),
    sigma_x=0.05,
    likelihood="student_t",
    degrees_of_freedom=4.0,
    epochs=150,
    seed=0,
)
robust.fit(X_with_defects, y)
y_hat = robust.predict(X_with_defects, mask=defect_mask)
```

---

## Diagnostics

| Diagnostic | Interpretation |
|:-----------|:---------------|
| `loss_history_`, `validation_history_` | Training/validation likelihood per epoch; use early stopping when validation degrades. |
| `noise_variance_` | Learned $s_d^2$ per observable; large values indicate poorly modelled observables or a mis-specified $\sigma_x$. |
| `predict(..., return_info=True)["final_weighted_sse"]` | Inversion objective per object; compare with the expected $\chi^2$-like scale set by the stated noise. |
| `...["relative_gradient_norm"]` | Stationarity of the local solution; large values flag non-convergence. |
| `...["fisher_condition"]` | Local identifiability; large values mean the covariance is unreliable and multiple solutions may exist. |
| `predict_with_covariance` | Local Fisher covariance; do not interpret as a global/posterior uncertainty under strong nonlinearity. |
| Cross-optimizer agreement (GN vs L-BFGS, multiple `init`) | Practical evidence that the solution is not an optimizer artefact. |

---

## Caveats

!!! warning "Assumptions and failure modes"
    * **Identifiability**: the decoder must distinguish different labels; symmetric or degenerate label directions make the inverse problem ill-posed. Check the Fisher condition and, when needed, profile likelihood (Raue et al. 2009).
    * **Model misspecification**: the surrogate's bias propagates into $\hat y$; validate on held-out objects and never extrapolate outside the training label range. Prefer `likelihood="student_t"` when training observables are contaminated.
    * **Unmodelled systematics**: every observable needs a likelihood term and, where relevant, a mask; pixels that are neither modelled nor masked bias the inversion. This is the main reason a discriminative model can be more robust on messy, large data sets.
    * **Non-Gaussian noise**: the Gaussian likelihood assumes additive Gaussian errors; Poisson-dominated counts should be transformed (variance-stabilising) or treated with a count likelihood.
    * **Label measurement error**: not corrected; the likelihood assumes exact design labels.
    * **Test-time optimization**: local, nonlinear and potentially multimodal; see the section above. The reported covariance is local and can understate the true uncertainty.
    * **Cost**: iterative per-object optimization; for very large catalogues an amortised discriminative encoder is much cheaper.
    * **Batch-dependent layers**: dropout and batch-norm batch statistics break the single-object Jacobian; the estimator sets the decoder to `eval()` during inversion.

---

## References

| # | Reference | Used for |
|:-:|:----------|:---------|
| 1 | Krutchkoff, R. G. (1967). Classical and inverse regression methods of calibration. *Technometrics*, 9(3), 425–439. [doi:10.1080/00401706.1967.10490486](https://doi.org/10.1080/00401706.1967.10490486) | Inverse regression estimator and its bias/MSE properties |
| 2 | Berkson, J. (1950). Are there two regressions? *JASA*, 45(250), 164–180. | Direct vs inverse regression are different estimators |
| 3 | Osborne, C. (1991). Statistical calibration: a review. *International Statistical Review*, 59(3), 309–336. | Comparative properties of classical and inverse calibration |
| 4 | Fuller, W. A. (1987). *Measurement Error Models*. Wiley. | Measurement-error theory and corrected estimators |
| 5 | Bates, D. M., & Watts, D. G. (1988). *Nonlinear Regression Analysis and Its Applications*. Wiley. | Gauss–Newton convergence, curvature and parameter-effects nonlinearity |
| 6 | Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer. | Local convergence of Levenberg–Marquardt and L-BFGS; trust regions |
| 7 | Raue, A., Kreutz, C., Maiwald, T., et al. (2009). Structural and practical identifiability analysis … profile likelihood. *Bioinformatics*, 25(15), 1923–1929. [doi:10.1093/bioinformatics/btp358](https://doi.org/10.1093/bioinformatics/btp358) | Practical identifiability and profile-likelihood intervals |
| 8 | Tarantola, A. (2005). *Inverse Problem Theory and Methods for Model Parameter Estimation*. SIAM. [doi:10.1137/1.9780898717921](https://doi.org/10.1137/1.9780898717921) | MAP formulation of inverse problems |
| 9 | LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). A tutorial on energy-based learning. In *Predicting Structured Data*. MIT Press. | Energy-based (argmin) prediction framing |
| 10 | Park, J. J., Florence, P., Straub, J., Newcombe, R., & Lovegrove, S. (2019). DeepSDF: Learning continuous signed distance functions for shape representation. *CVPR 2019*. [arXiv:1901.05103](https://arxiv.org/abs/1901.05103) | Auto-decoder inference (test-time code optimization) |
| 11 | Ng, A. Y., & Jordan, M. I. (2002). On discriminative vs. generative classifiers. *NeurIPS 2001*. | Generative/discriminative trade-off behind the scale caveat |
| 12 | Gainer, J. S., Lykken, J., Matchev, K. T., Mrenna, S., & Park, M. (2013). The matrix element method: past, present, and future. Snowmass white paper. [arXiv:1307.3546](https://arxiv.org/abs/1307.3546) | Per-event likelihood evaluation with a forward model (HEP) |
| 13 | Cranmer, K., Brehmer, J., & Louppe, G. (2020). The frontier of simulation-based inference. *PNAS*, 117(48), 30055–30062. [doi:10.1073/pnas.1912789117](https://doi.org/10.1073/pnas.1912789117) | Forward-model-based inference at scale |

## See also

* [LatentNN](latentnn.md) — joint latent-input EIV training in the discriminative direction.
* [Regression Calibration](rc.md) — moment-based correction of noisy inputs.
* [SIMEX](simex.md) — simulation-extrapolation correction.
* [Split-measurement & corrected-score losses](../../losses/split_eiv.md) — attenuation-free EIV objectives.
* [EIV losses](../../losses/eiv.md) — losses that consume noisy inputs directly.
* [Noisy features / measurement error](../../guide/method-selection.md) — method-selection matrix.
