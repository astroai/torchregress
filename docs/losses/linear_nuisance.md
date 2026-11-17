# Linear-nuisance marginal likelihood

> ← [Structural Similarity (1-D)](structural_similarity.md) | [Gaussian Losses](gaussian.md) →

`marginalize_linear_nuisance` and `MarginalizedNuisanceGaussianNLLLoss` score
predictions of sequences (spectra, light curves, profiles) under a Gaussian
likelihood with **known per-sample noise** while integrating out a smooth
distortion you do not want to be scored on: a continuum or baseline, a
flux-calibration polynomial, a gain and offset, a set of sky or template residual
components. The distortion is linear in unknown coefficients, so the integral is
exact and costs one $m\times m$ Cholesky factorisation per row.

| # | Reference |
|:-:|:----------|
| 1 | Rybicki, G. B. & Press, W. H. (1992). *Interpolation, realization, and reconstruction of noisy, irregularly sampled data*. ApJ 398, 169–176. |
| 2 | Patterson, H. D. & Thompson, R. (1971). *Recovery of inter-block information when block sizes are unequal*. Biometrika 58(3), 545–554. (restricted likelihood) |
| 3 | Harville, D. A. (1974). *Bayesian inference for variance components using only error contrasts*. Biometrika 61(2), 383–385. (REML as the flat-prior marginal) |
| 4 | Woodbury, M. A. (1950). *Inverting modified matrices*. Memorandum Rept. 42, Statistical Research Group, Princeton. |
| 5 | Harville, D. A. (1997). *Matrix Algebra From a Statistician's Perspective*. Springer. (matrix determinant lemma) |

→ API: [`marginalize_linear_nuisance`, `MarginalizedNuisanceGaussianNLLLoss`, `LinearNuisanceMarginal`](../api/losses.md#sequence-shape-losses)

---

## Model and objective

For one row with residual $r = y - \hat y \in \mathbb R^N$, known precisions
$\Lambda = \operatorname{diag}(\lambda)$, a nuisance basis $U\in\mathbb R^{N\times m}$
and coefficients $a$ with prior $a\sim\mathcal N(0, P^{-1})$,

$$
y = \hat y + U a + \varepsilon,\quad \varepsilon\sim\mathcal N(0,\Lambda^{-1})
\quad\Longrightarrow\quad
r \sim \mathcal N\!\left(0,\; \Lambda^{-1} + U P^{-1} U^\top\right).
$$

The Woodbury identity and the matrix determinant lemma reduce its negative
log-likelihood over the $n$ unmasked samples to $m\times m$ algebra:

$$
-\log p(r) = \tfrac12\Big[\,(n - m_0)\log 2\pi - \sum_i\log\lambda_i + \log\lvert A\rvert - \log\lvert P\rvert
+ (r - U\hat a)^\top\Lambda\,(r - U\hat a) + \hat a^\top P\,\hat a\,\Big],
$$

$$
A = P + U^\top\Lambda U,\qquad \hat a = A^{-1}U^\top\Lambda r,\qquad
\operatorname{Cov}(a \mid r) = A^{-1}.
$$

$\hat a$ is the posterior mean (MAP) of the nuisance coefficients and $U\hat a$ the
fitted baseline; $r - U\hat a$ is the conditional residual. The quadratic form is
evaluated in this residual form, which equals $r^\top\Lambda r - \hat a^\top U^\top\Lambda r$
without the cancellation that appears when the nuisance explains most of $r$.

### Priors

| Argument | Meaning |
|:---------|:--------|
| `prior_covariance=S` (`[m, m]` or per row) | $P = S^{-1}$, evaluated in whitened coordinates $a = Lc$, $S = LL^\top$, so $S$ is never inverted |
| `prior_precision=p` (float, `[m]` or per row) | diagonal $P = \operatorname{diag}(p)$, $p_j\ge0$ |
| neither (default) | flat prior in every direction ($p = 0$) |

A **flat direction** ($p_j = 0$) is the improper prior $p(a_j) = 1$; $m_0$ counts
them ($m_0 = 0$ for a proper prior). Its $\log p_j$ is dropped from
$\log\lvert P\rvert$ and it removes one $\log 2\pi$. For an all-flat prior this is
the *restricted* likelihood (REML; Patterson & Thompson 1971, Harville 1974), with
its constants:

$$
-\log p_{\text{flat}}(r) = \lim_{\epsilon\to0}\Big[-\log p_{\epsilon}(r) + \tfrac12 m_0\log\frac{\epsilon}{2\pi}\Big]
= \text{profile NLL} + \tfrac12\log\lvert U^\top\Lambda U\rvert - \tfrac12 m_0\log 2\pi ,
$$

where $p_\epsilon$ uses the proper prior precision $\epsilon$ in the flat directions.
It equals the Gaussian likelihood of the error contrasts $K^\top r$ ($K^\top U = 0$,
$K^\top K = I$) plus $\tfrac12\log\lvert U^\top U\rvert$ (tested).

The profile NLL fixes the nuisance at its weighted least-squares value. The extra
terms do not depend on `y_pred`, so gradients for the model are identical; they
matter only when $\Lambda$ or $U$ are learned. With
a flat prior the NLL is **exactly invariant** to adding any $U a$ to the target.

### Masks and fail-closed conditions

Masked samples drop out exactly (as if absent from the row); `NaN`/`inf` values
there are ignored. Every unmasked sample needs $\lambda_i > 0$ ($\lambda_i = 0$ gives
$+\infty$, as in [`student_t_nll`](robust.md); a negative $\lambda_i$ raises). An
asymmetric or non-positive-definite `prior_covariance` raises. A ValueError is also
raised, rather than a silently regularised answer, when for some row $A$ is not numerically positive
definite: an empty row or a basis column that vanishes on every unmasked sample
under a flat prior, collinear columns, or non-finite precisions/basis on unmasked
samples. The test is a Jacobi-scaled Cholesky whose squared pivots must exceed
$m\,\varepsilon_{\text{mach}}$. An empty row with a proper prior is valid: its NLL is
0 and the loss leaves it out of the mean.

!!! warning "Not a fix for a wrong noise model"
    The marginalisation is exact for the stated Gaussian model. If the noise is
    heavier-tailed, mask or down-weight outliers first (for example with
    [`batched_iteratively_reweighted_fit`](../methods/algorithms/irls.md)); if the
    distortion is not in the span of $U$, it is still scored as signal mismatch.

!!! info "Non-zero prior mean"
    For a prior $a\sim\mathcal N(a_0, P^{-1})$, add $U a_0$ to `y_pred`.

---

## Computation and scaling

All rows are processed together: $U^\top\Lambda U$ and $U^\top\Lambda r$ are batched
matrix products, $A$ is factorised by a batched Cholesky, and a successful call makes
one host synchronisation: all fail-closed checks (precision sign, basis finiteness,
prior validity, posterior positive-definiteness) are combined on the device first, and
only a failing call reads them one by one for its error message. (A masked basis
passed as `[1, N, m]` or as an expanded view adds one synchronisation, to check that
it is finite before taking the shared path.) Nothing of size $N\times N$ is formed.
A shared basis `[N, m]` uses the packed products $U_jU_k$ ($N\,m(m+1)/2$ elements)
instead of a `[B, N, m]` intermediate when $B > (m+1)/2$; a basis passed as
`[1, N, m]` or as an expanded view of one `[N, m]` basis takes the same path (unless
it is masked and non-finite somewhere, which only the per-row path accepts). Cost is
$O(B N m^2 + B m^3)$.

Forward + backward w.r.t. `y_pred`, float64, CPU with 2 threads on a shared 4-CPU
host, flat prior, 10% masked (`tools/benchmark_linear_nuisance.py`; comparator:
`torch.distributions.LowRankMultivariateNormal`, which needs the factor per row and
supports neither masks nor flat priors):

| $B$ | $N$ | $m$ | shared basis, rows/s | per-row basis, rows/s | `LowRankMultivariateNormal`, rows/s |
|--:|--:|--:|--:|--:|--:|
| 256 | 1 000 | 5 | 18 600 | 13 300 | 16 300 |
| 256 | 1 000 | 50 | 5 060 | 1 710 | 930 |
| 256 | 10 000 | 5 | 3 460 | 1 150 | 860 |
| 256 | 10 000 | 50 | 980 | 130 | 170 |
| 1 024 | 10 000 | 50 | 960 | — | — |

---

## When to use which

| Loss | Covariance | Masks | Flat (improper) priors | Returns nuisance fit | Use for |
|:-----|:-----------|:------|:-----------------------|:---------------------|:--------|
| `GaussianNLLLoss` | diagonal | per element | — | — | independent errors |
| `LowRankGaussianLoss` | $WW^\top + \operatorname{diag}(d)$, learned | whole rows | no | no | learned correlated outputs |
| `MultivariateGaussianLoss` | full $D\times D$ | whole rows | no | no | small $D$ |
| [`MarginalizedMultiSourceGaussianNLLLoss`](multisource.md) | $R + H L L^\top H^\top$, dense $R$ `[B, K, K]`, Jacobian $H$ `[B, K, D]` | per sample | no | no | cross-correlated multi-source measurements |
| **`MarginalizedNuisanceGaussianNLLLoss`** | $\Lambda^{-1} + U P^{-1} U^\top$, known noise | per sample, exact | yes (restricted likelihood) | yes (`marginalize_linear_nuisance`) | continuum, baseline and calibration marginalisation |
| [`StructuralSimilarity1dLoss`](structural_similarity.md) | none (not a likelihood) | per sample | — | — | shape similarity, regularisation |

With a proper prior the covariance has the low-rank-plus-diagonal structure of
`LowRankGaussianLoss` (factor $W = UL$, $LL^\top = P^{-1}$, diagonal $1/\lambda$), and
the NLL matches `torch.distributions.LowRankMultivariateNormal` and the dense
`MultivariateNormal` to float64 round-off (tested). This loss is separate because its inputs are a known noise
model and a nuisance basis rather than learned covariance factors, and because it
needs per-sample masks, flat priors, the nuisance posterior and a shared basis that
is not copied per row. `LowRankGaussianLoss` also clamps and jitters its diagonal,
which would bias an exact likelihood with known errors.

`MarginalizedMultiSourceGaussianNLLLoss` computes the same marginal NLL when its
measurement covariance is diagonal and known (`mapped_mean=y_pred`, `jacobian=U`,
`measurement_covariance=diag(1/precision)`, `scale_tril=chol(S)`, zero latent mean;
tested to float64 round-off). It is built for dense cross-correlated measurement
covariances, so it materialises a `[B, N, N]` covariance and a per-row Jacobian
(about 819 GB in float64 at $B = 1024$, $N = 10\,000$), and it has no flat priors and
no nuisance posterior. This loss keeps the noise diagonal, never copies a shared
basis per row, and costs $O(BNm^2)$.

---

## Example

Fit the amplitude of a line on an unknown linear continuum. The model predicts only
the line; the continuum is marginalised with a flat prior, and the fitted baseline
comes back from `marginalize_linear_nuisance`.

```python
import torch
from torchregress.losses import MarginalizedNuisanceGaussianNLLLoss, marginalize_linear_nuisance

torch.manual_seed(0)
dtype = torch.float64
wave = torch.linspace(-1.0, 1.0, 400, dtype=dtype)
line = torch.exp(-0.5 * (wave / 0.05) ** 2)
basis = torch.stack([torch.ones_like(wave), wave, wave**2], dim=-1)  # quadratic continuum

rows = 16
true_amp = torch.rand(rows, 1, dtype=dtype) + 0.5
continuum = 3.0 + torch.randn(rows, 3, dtype=dtype) @ basis.T
precision = torch.full((rows, 400), 2500.0, dtype=dtype)               # sigma = 0.02
target = true_amp * line + continuum + torch.randn(rows, 400, dtype=dtype) * precision.rsqrt()
mask = torch.ones(rows, 400, dtype=torch.bool)
mask[:, 50:80] = False                                                 # bad pixels
target[~mask] = float("nan")

amp = torch.ones(rows, 1, dtype=dtype, requires_grad=True)
loss_fn = MarginalizedNuisanceGaussianNLLLoss()                         # flat prior on the continuum
opt = torch.optim.LBFGS([amp], max_iter=100)

def closure():
    opt.zero_grad()
    loss = loss_fn(amp * line, target, precision, basis, mask=mask)
    loss.backward()
    return loss

opt.step(closure)
assert torch.allclose(amp, true_amp, atol=0.02)

fit = marginalize_linear_nuisance(amp.detach() * line, target, precision, basis, mask=mask)
baseline = fit.nuisance                     # fitted continuum at every pixel, gap included
assert (baseline - continuum).abs().max() < 0.02
sigma_coef = fit.coefficient_covariance.diagonal(dim1=-2, dim2=-1).sqrt()
assert sigma_coef.shape == (rows, 3)
```

→ See also [Gaussian losses](gaussian.md), the [Student-$t$ likelihood with known
precisions](robust.md), [batched IRLS](../methods/algorithms/irls.md) for robust
per-row fits, and [1-D structural similarity](structural_similarity.md).
