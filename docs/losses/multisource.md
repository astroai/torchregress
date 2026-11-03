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

### Batched evaluation

All objects in a batch are factorized together (batched Cholesky and triangular solves on the inputs' device and dtype). Each unobserved slot $j$ becomes a decoupled coordinate with unit variance, zero residual, and a zero Jacobian row:

$$\tilde R_{jj}=1,\qquad \tilde R_{ij}=\tilde R_{ji}=0,\qquad \tilde r_j=0,\qquad \tilde H_{j\cdot}=0 .$$

The padded covariance is block diagonal after a permutation, so the padded Cholesky factor restricted to the observed rows is the factor of the observed submatrix. A padded slot adds $\log 1=0$ to the log-determinant and $0$ to the quadratic form. The $\log(2\pi)$ constant counts observed slots only. Values and gradients match the earlier object-by-object evaluation to floating-point rounding.

!!! info "Validation and diagnostics"
    Inputs are validated once per batch, with one host synchronization. If several objects are invalid in different ways, the error reported is the first failing check in this order: non-finite latent parameters, latent scale, non-finite observed values, asymmetric covariance, then not positive definite. `last_diagnostics` is a read-only mapping. It is copied to Python lists only when you read it. Adaptive jitter (`max_jitter > jitter`) adds one synchronization for each jitter level that is actually tried.

## Differentiable observation-map components

`torchregress.calibration.bspline_basis_with_derivative` returns a clamped B-spline basis, its derivative, and flags for numerical extrapolation. Its boundary policy explicitly selects linear tails, constant boundary values, or an error. The spline heads default to constant boundary values; this is a numerical convention, not evidence of physical calibration outside support. These flags indicate the spline domain only, not verified population or physical support.

`RelativeSurveySplineHead` imposes a fixed-weight, per-target zero-sum on the relative survey offsets and their Jacobians. A zero weight identifies an unavailable source/target pair. `CommonScaleSplineHead` supplies a separate common correction and must be fitted only when appropriate independent anchors identify it. `ConditionBiasHead` supplies quality-conditioned corrections that are exactly zero at each source's own reference condition. Those conditional corrections are **not** generally zero-sum between sources.

These tools do not establish an absolute physical scale or resolve correlated/repeated data automatically. The caller must define comparable target quantities and units, record release/pipeline provenance, avoid double-counted labels and anchors, validate extrapolation, and propagate calibration-function uncertainty separately.

## Scalar nonlinear forward observations

`GaussianLatentForwardObservationNLLLoss` evaluates a Gaussian measurement likelihood after integrating a scalar Gaussian latent through a caller-supplied differentiable forward map. It uses normalized Gauss–Hermite weights and finite-order quadrature. Missing observations are masked before the map, and negative observations are valid. Compare quadrature orders for broad latent distributions or narrow observation noise; multiple dependent observations of the same latent must be integrated jointly rather than multiplied as separate marginal likelihoods.

`RobustMultiSourceGaussianNLLLoss` uses a normalized inlier/outlier mixture per measurement block while integrating over the same shared latent Gaussian. It preserves covariance within a block and requires explicit block IDs; it rejects cross-block measurement covariance, because the assumed independent block inflation would be ambiguous. Assignment enumeration grows exponentially and is bounded by configuration. Each assignment is scored for the whole batch at once, and an object skips assignments of blocks it does not have. The mixture is a batched `logsumexp` over assignments. The outlier model and its probabilities must be justified for the data rather than treated as proof that labels are correct.
