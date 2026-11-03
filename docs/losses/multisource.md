# Multiple noisy sources for one shared Gaussian prediction

`MarginalizedMultiSourceGaussianNLLLoss` scores all available measurements of an object jointly. It accepts a full measurement covariance, including cross-source correlations. A mask removes unobserved dimensions before factorization; missing values never become zero-noise observations.

Given the predicted latent $\theta\sim\mathcal N(\mu,LL^\top)$, linearize each observation map at $\mu$ to obtain $h(\mu)$ and its stacked Jacobian $H$. The resulting distribution is

$$y\mid x\;\dot\sim\;\mathcal N\!\left(h(\mu),R+HLL^\top H^\top\right).$$

The result is **exact for this linearized Gaussian model**, not a claim of exact nonlinear integration. The normalized loss includes $K\log(2\pi)$ and the determinant term. It uses Cholesky solves; the information-form option avoids subtracting two large quadratic terms.

```python
from torchregress.losses import MarginalizedMultiSourceGaussianNLLLoss, PackedObservations

# mean [B,D], scale_tril [B,D,D], value/mapped_mean [B,K],
# jacobian [B,K,D], covariance [B,K,K], observed [B,K] bool
obs = PackedObservations(value, mapped_mean, jacobian, covariance, observed)
loss = MarginalizedMultiSourceGaussianNLLLoss(method="auto")(mean, scale_tril, obs)
```

The caller provides $h(\mu)$ and $H$ without detaching them, so gradients can reach both the predictor and observation-map parameters. The default mean reduction excludes stars with no observed dimensions. The covariance must be positive definite on the observed subspace. Optional, explicitly bounded jitter is reported in `last_diagnostics`; no unknown error is implicitly converted into a known zero.

## Differentiable observation-map components

`torchregress.calibration.bspline_basis_with_derivative` returns a clamped B-spline basis, its derivative, and flags for numerical extrapolation. Its boundary policy explicitly selects linear tails, constant boundary values, or an error. The spline heads default to constant boundary values; this is a numerical convention, not evidence of physical calibration outside support. These flags indicate the spline domain only, not verified population or physical support.

`RelativeSurveySplineHead` imposes a fixed-weight, per-target zero-sum on the relative survey offsets and their Jacobians. A zero weight identifies an unavailable source/target pair. `CommonScaleSplineHead` supplies a separate common correction and must be fitted only when appropriate independent anchors identify it. `ConditionBiasHead` supplies quality-conditioned corrections that are exactly zero at each source's own reference condition. Those conditional corrections are **not** generally zero-sum between sources.

These tools do not establish an absolute physical scale or resolve correlated/repeated data automatically. The caller must define comparable target quantities and units, record release/pipeline provenance, avoid double-counted labels and anchors, validate extrapolation, and propagate calibration-function uncertainty separately.

## Scalar nonlinear forward observations

`GaussianLatentForwardObservationNLLLoss` evaluates a Gaussian measurement likelihood after integrating a scalar Gaussian latent through a caller-supplied differentiable forward map $f$:

$$-\log p(y)=-\log\int \mathcal N(z;\mu,s^2)\,\mathcal N\!\left(y;f(z),\sigma^2\right)dz .$$

With $z=\mu+s\,t$, the integral is evaluated by **adaptive Gauss–Hermite quadrature**. The $Q$ Hermite nodes $x_q$ and weights $w_q$ are placed on a Gaussian proposal $\mathcal N(c,\tau^2)$ in $t$:

$$p(y)\approx\sqrt2\,\tau\sum_{q=1}^{Q} w_q\,e^{x_q^2}\,\varphi(t_q)\,\mathcal N\!\left(y;f(\mu+s\,t_q),\sigma^2\right),\qquad t_q=c+\sqrt2\,\tau\,x_q .$$

The centre $c$ and scale $\tau\le1$ track the mode and curvature of the integrand (prior times likelihood), not the prior. They come from a fixed number of derivative-free passes that evaluate the map on $[N,Q]$ node grids without autograd. Each pass brackets the peak between the best node's neighbours and fits a parabola through them. The rule is therefore exact for linear maps at any order $Q\ge3$. It stays accurate when the observation noise is much narrower than the latent spread pushed through the map, which is where the prior-centred rule ($c=0,\tau=1$, used before 0.3.0) fails. $c$ and $\tau$ act as a stop-gradient change of variables: the integral does not depend on them, so gradients with respect to $\mu$, $s$, $y$, $\sigma$ and the map's parameters are the integral's gradients, up to quadrature error. Missing observations are masked before the map, and negative observations are valid. Multiple dependent observations of the same latent must be integrated jointly, rather than multiplied as separate marginal likelihoods.

| 200 synthetic fluxes $y=e^z+\varepsilon$ (SNR 7–50, $s\in[0.1,0.8]$, float64); max abs. error vs. `scipy.integrate.quad`, nats | $Q=8$ | $Q=16$ | $Q=32$ | $Q=64$ |
|:--|--:|--:|--:|--:|
| Adaptive (current) | $2\times10^{-3}$ | $6\times10^{-5}$ | $2\times10^{-7}$ | $10^{-10}$ |
| Prior-centred (before 0.3.0) | $10^{3}$ | $7\times10^{2}$ | $3\times10^{2}$ | $2\times10^{2}$ |

!!! warning "When to check convergence"
    Convergence slows when the integrand is far from Gaussian around its peak. This happens when the map's derivative vanishes at the peak (for example $z^3$ near $0$), or when the likelihood keeps a plateau across the prior (an observation within a few $\sigma$ of an asymptote of the map). Non-monotone maps for which $f(z)=y$ has several solutions give multimodal integrands that a single proposal cannot cover. Compare two quadrature orders before relying on results. With one or two nodes the prior-centred rule is used.

!!! info "Map contract"
    The map receives and returns `[N, Q]` tensors and must act elementwise. It is called once per adaptive pass under `torch.no_grad()` and once more for the differentiable result, so it must be deterministic and free of side effects (no dropout, no batch-norm statistics updates).

```python
import torch
from torchregress.losses import GaussianLatentForwardObservationNLLLoss

mu = torch.tensor([0.1, 1.2], requires_grad=True)       # latent log-flux mean
s = torch.tensor([0.55, 0.30], requires_grad=True)       # latent scatter
flux = torch.tensor([1.30, 3.40])                         # observed flux
flux_err = torch.tensor([0.05, 0.08])                     # 1-sigma measurement error

loss_fn = GaussianLatentForwardObservationNLLLoss(quadrature_points=32)
loss = loss_fn(mu, s, flux, flux_err, torch.exp)
loss.backward()
# Convergence check: a higher order should agree to well below the noise you care about.
assert abs(loss.item() - GaussianLatentForwardObservationNLLLoss(64)(mu, s, flux, flux_err, torch.exp).item()) < 1e-5
```

`RobustMultiSourceGaussianNLLLoss` uses a normalized inlier/outlier mixture per measurement block while integrating over the same shared latent Gaussian. It preserves covariance within a block and requires explicit block IDs; it rejects cross-block measurement covariance, because the assumed independent block inflation would be ambiguous. Assignment enumeration grows exponentially and is bounded by configuration. The outlier model and its probabilities must be justified for the data rather than treated as proof that labels are correct.
