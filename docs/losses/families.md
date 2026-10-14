# Flexible-Shape Distributional Families

> ← [Losses Catalogue](index.md) | [API Reference](../api/losses.md#distribution-families-lossesfamilies) →

Parametric NLL losses for **non-Gaussian** conditional distributions — skew, heavy-tailed, bounded, and extreme-value families. Each loss consumes the raw output matrix of a flexible-shape head (`y_pred[..., k]`) and routes the elementwise NLL through the unified `BaseLoss._reduce` pipeline (mask, sample weights, reduction).

All families share the same contract:

```python
loss_fn(y_pred, target, mask=None, weights=None) -> Tensor
```

where `y_pred` columns map to distribution parameters and positivity-constrained parameters are handled internally via `softplus(raw) + eps`.

---

## Catalogue

| Loss | Distribution | Parameters (`y_pred` columns) | Use Case | API |
|:-----|:-------------|:-------------------------------|:---------|:----|
| `SkewNormalNLLLoss` | Skew-normal (Azzalini 1985) | `xi, omega_raw, alpha` → `omega = softplus(omega_raw)+eps` | Asymmetric, unimodal, near-Gaussian skew | [`SkewNormalNLLLoss`](../api/losses.md) |
| `SkewTLoss` | Skew-t (Azzalini & Capitanio 2003) | `xi, omega_raw, alpha, nu_raw` → `omega, nu > 0` | Heavy-tailed + skew | [`SkewTLoss`](../api/losses.md) |
| `BetaRegressionNLLLoss` | Beta (Ferrari & Cribari-Neto 2004) | `logit(mu), phi_raw` → `mu=sigmoid(logit), phi=softplus(phi_raw)+eps` | Targets in (0, 1) (rates, proportions) | [`BetaRegressionNLLLoss`](../api/losses.md) |
| `JohnsonSUNLLLoss` | Johnson SU | `xi, lambda_raw, gamma, delta_raw` → `lambda, delta > 0` | Flexible skew + kurtosis via `(gamma, delta)` | [`JohnsonSUNLLLoss`](../api/losses.md) |
| `SinhArcsinhNLLLoss` | Sinh-arcsinh (Jones & Pewsey 2009) | `mu, sigma_raw, epsilon, delta_raw` → `sigma, delta > 0` | Tunable skew (`epsilon`) + tail (`delta`); Gaussian at `epsilon=0, delta=1` | [`SinhArcsinhNLLLoss`](../api/losses.md) |
| `GEVNLLLoss` | GEV (Coles 2001) | `mu, sigma_raw, xi` → `sigma>0`, `xi` free | Block maxima / extremes; analytic Gumbel limit at `xi→0` | [`GEVNLLLoss`](../api/losses.md) |
| `AsymmetricLaplaceNLLLoss` | Asymmetric Laplace | `mu, sigma_raw, kappa_raw` → `sigma, kappa > 0` | Quantile-like asymmetry; pinball correspondence `tau=1/(1+kappa²)` | [`AsymmetricLaplaceNLLLoss`](../api/losses.md) |
| `SQRLoss` | Sorted quantiles (SQR, Duan et al. 2020) | `L` quantile levels (sorted via `cummax`) | Distribution-free quantile regression | [`SQRLoss`](../api/losses.md) |

Functional forms are also exported (`skew_normal_nll`, `skew_t_nll`, `beta_regression_nll`, `johnson_su_nll`, `sinh_arcsinh_nll`, `gev_nll`, `asymmetric_laplace_nll`, `sqr_loss`).

Each minimises the elementwise negative log-likelihood of its density family:

$$\mathcal{L}(\theta_1, \dots, \theta_K) = -\log f(y;\, \theta_1, \dots, \theta_K), \qquad \theta_k = \text{softplus}(\text{raw}_k) + \varepsilon$$

for positivity-constrained parameters, with $\theta_k = \text{raw}_k$ for unconstrained ones ($\xi$, $\alpha$, $\gamma$, $\epsilon$, $\mu$). The elementwise NLLs are:

- **Skew-normal** (Azzalini 1985), $z = (y-\xi)/\omega$:

$$f(y) = \frac{2}{\omega}\,\varphi(z)\,\Phi(\alpha z)$$

- **Skew-t** (Azzalini & Capitanio 2003), degrees of freedom $\nu > 0$:

$$f(y) = \frac{2}{\omega}\, t_\nu(z)\; T_{\nu+1}\!\Big(\alpha z \sqrt{\tfrac{\nu+1}{\nu+z^2}}\Big)$$

- **Beta regression** (Ferrari & Cribari-Neto 2004), for $y \in (0,1)$ with mean $\mu = \sigma(\text{logit})$ and precision $\phi > 0$:

$$f(y) = \frac{y^{\alpha-1}(1-y)^{\beta-1}}{B(\alpha, \beta)}, \qquad \alpha = \mu\phi,\;\; \beta = (1-\mu)\phi$$

- **Johnson SU**, with $u = (y - \xi)/\lambda$ and the standardized variate $Z = \gamma + \delta\,\operatorname{asinh}(u)$:

$$f(y) = \frac{\delta}{\lambda\sqrt{2\pi}}\;\frac{1}{\sqrt{1+u^2}}\;\exp\!\Big(-\tfrac{1}{2}\big(\gamma + \delta\,\operatorname{asinh}(u)\big)^2\Big)$$

- **Sinh-arcsinh** (Jones & Pewsey 2009), with $z = (y-\mu)/\sigma$ and $s = \sinh\big((\operatorname{asinh}(z) - \epsilon)/\delta\big)$ — the inverse of the transformation $y = \mu + \sigma\sinh(\delta\,\operatorname{asinh}(Z)+\epsilon)$ for $Z \sim \mathcal{N}(0,1)$:

$$\mathcal{L} = \tfrac{1}{2}\log 2\pi + \log(\sigma\delta) + \tfrac{1}{2}\log(1+z^2) - \tfrac{1}{2}\log(1+s^2) + \tfrac{s^2}{2}$$

At $\epsilon=0,\ \delta=1$ this reduces **exactly** to Gaussian NLL.

- **GEV** (Coles 2001 parameterization), $z = (y-\mu)/\sigma$; observations outside the support $1 + \xi z > 0$ receive infinite NLL:

$$\mathcal{L} = \log\sigma + \frac{1+\xi}{\xi}\log(1+\xi z) + \big(1+\xi z\big)^{-1/\xi} \quad \xrightarrow{\ \xi \to 0\ } \quad \mathcal{L}_{\text{Gumbel}} = \log\sigma + z + e^{-z}$$

The analytic Gumbel limit is used for $|\xi|$ below the implementation's numerical threshold, avoiding the $0/0$ limit form.

- **Asymmetric Laplace**, $u = y-\mu$:

$$f(y) = \frac{\sqrt{2}\,\kappa}{\sigma(1+\kappa^2)}\,\exp\!\Big(-\frac{\sqrt{2}}{\sigma\kappa}\max(u,0) - \frac{\sqrt{2}\,\kappa}{\sigma}\max(-u,0)\Big)$$

The NLL reduces to its normalisation constant plus $\dfrac{\sqrt{2}(1+\kappa^2)}{\sigma\kappa}\,\rho_\tau(y-\mu)$ where $\rho_\tau$ is the pinball (check) loss at the level $\tau = 1/(1+\kappa^2)$ — the exact $\kappa$–$\tau$ correspondence, matching the [QuantileLoss](quantile_expectile.md) family.

---

## Positivity and `unconstrained_inputs`

Several parameters must stay strictly positive (`omega, phi, lambda, delta, sigma, kappa, nu`). By default each family assumes **unconstrained** network outputs and maps them with:

```
positive = softplus(raw) + eps   # eps defaults to 1e-6
```

Set `unconstrained_inputs=False` when your head **already** enforces positivity (e.g. a `softplus` or `exp` activation). In that mode the loss avoids `softplus(softplus(x))` by applying the inverse first:

```python
# raw head — default
loss_fn = BetaRegressionNLLLoss(unconstrained_inputs=True)  # expects raw phi
loss = loss_fn(torch.randn(8, 2), targets_in_01)

# head already outputs positive phi/sigma/etc.
loss_fn = BetaRegressionNLLLoss(unconstrained_inputs=False)  # expects phi>0
loss = loss_fn(torch.cat([logits, phi_positive], dim=-1), targets_in_01)
```

Internally (`unconstrained_inputs=False`):

```
raw = softplus_inverse((positive - eps).clamp(min=1e-6)) = s + log(-expm1(-s))
positive_recovered = softplus(raw) + eps  # == positive
```

This keeps the elementwise NLL numerically identical between modes while letting you compose heads that already satisfy constraints. All seven NLL families (`SkewNormalNLLLoss`, `SkewTLoss`, `BetaRegressionNLLLoss`, `JohnsonSUNLLLoss`, `SinhArcsinhNLLLoss`, `GEVNLLLoss`, `AsymmetricLaplaceNLLLoss`) expose `unconstrained_inputs` with the same semantics; `SQRLoss` has no positivity constraint and does not use the flag.

!!! tip "Which mode to use?"
    - Use `unconstrained_inputs=True` (default) when the final linear layer is unconstrained — the loss handles positivity. This is the simplest and most common setup.
    - Use `unconstrained_inputs=False` when you share a positivity-enforcing head across multiple losses or when an upstream wrapper already applies `softplus`/`exp`.

---

## Usage

### Skew-normal / Skew-t

```python
from torchregress.losses import SkewNormalNLLLoss, SkewTLoss

# Skew-normal: [xi, omega_raw, alpha]
sn = SkewNormalNLLLoss(unconstrained_inputs=True)
loss = sn(y_pred_sn, y)  # y_pred_sn: [batch, 3]

# Skew-t: [xi, omega_raw, alpha, nu_raw]
st = SkewTLoss(unconstrained_inputs=True)
loss = st(y_pred_st, y)  # y_pred_st: [batch, 4]
```

The skew-normal reduces exactly to the Gaussian NLL at `alpha=0`; the skew-t reduces to Student-t at `alpha=0`.

### Beta regression

```python
from torchregress.losses import BetaRegressionNLLLoss

loss_fn = BetaRegressionNLLLoss(unconstrained_inputs=True)
# y in (0, 1); y_pred: [logit(mu), phi_raw]
loss = loss_fn(y_pred, y)
```

Targets must lie strictly in (0, 1); `0`/`1` raise `ValueError`.

### Johnson SU / Sinh-arcsinh

```python
from torchregress.losses import JohnsonSUNLLLoss, SinhArcsinhNLLLoss

jsu = JohnsonSUNLLLoss(unconstrained_inputs=True)   # [xi, lam_raw, gamma, delta_raw]
sas = SinhArcsinhNLLLoss(unconstrained_inputs=True) # [mu, sigma_raw, eps, delta_raw]
```

Both families are **Gaussian at the identity** (`gamma=0, delta` scaled, or `epsilon=0, delta=1`) and support flexible skew/kurtosis via their shape parameters.

### GEV (extreme values)

```python
from torchregress.losses import GEVNLLLoss

gev = GEVNLLLoss(unconstrained_inputs=True)  # [mu, sigma_raw, xi]
loss = gev(y_pred, y)
```

For `|xi| < 1e-6` the analytic Gumbel limit is used; observations outside `1 + xi*z > 0` receive infinite NLL.

### Asymmetric Laplace / pinball correspondence

```python
from torchregress.losses import AsymmetricLaplaceNLLLoss

ald = AsymmetricLaplaceNLLLoss(unconstrained_inputs=True)  # [mu, sigma_raw, kappa_raw]
# kappa=1 is symmetric Laplace; tau = 1/(1+kappa^2) gives the equivalent quantile
```

### SQR (sorted quantile regression)

```python
from torchregress.losses import SQRLoss

sqr = SQRLoss(n_levels=32)
# y_pred: [batch, L] quantile levels; sorted internally via cummax
loss = sqr(y_pred, y)
```

---

## Recommendations

1. **Start with the identity**: verify that `SkewNormal(alpha=0)` matches Gaussian NLL and `SinhArcsinh(epsilon=0, delta=1)` matches Gaussian NLL on your data pipeline before adding skew/tail learning.
2. **Bounded targets → Beta**: if `y ∈ (0, 1)`, prefer `BetaRegressionNLLLoss` over transforming `y` to `ℝ`; it preserves the (0, 1) support and calibrates near the boundaries.
3. **Extremes → GEV**: use only for block-maxima or threshold-exceedance targets; validate support (`1 + xi*z > 0`) and monitor for infinite NLLs that indicate out-of-support predictions.
4. **Check positivity wiring**: if you set `unconstrained_inputs=False`, ensure every positivity-constrained column is already `> eps`. Passing raw logits with `False` double-constrains and biases gradients.

---

## References

| # | Reference |
|:-:|:----------|
| 1 | Azzalini, A. "A class of distributions which includes the normal ones." *Scand. J. Statist.* 12, 171–178 (1985). |
| 2 | Azzalini, A. & Capitanio, A. "Distributions generated by perturbation of symmetry with emphasis on a multivariate skew t-distribution." *JRSS-B* 65, 367–389 (2003). |
| 3 | Ferrari, S. & Cribari-Neto, F. "Beta regression for modelling rates and proportions." *J. Appl. Stat.* 31, 799–815 (2004). |
| 4 | Johnson, N. L., Kotz, S. & Balakrishnan, N. *Continuous Univariate Distributions Vol. 1* (1994) — Johnson SU. |
| 5 | Jones, M. C. & Pewsey, A. "Sinh-arcsinh distributions." *Biometrika* 96, 761–780 (2009). |
| 6 | Coles, S. *An Introduction to Statistical Modeling of Extreme Values.* Springer (2001). |
| 7 | Koenker, R. & Machado, J. A. F. "Goodness of fit and related inference processes for quantile regression." *JASA* 94 (1999). |
| 8 | Duan et al. "Adaptive Distributional Regression by Minimizing Sqr Loss." (2020). |
