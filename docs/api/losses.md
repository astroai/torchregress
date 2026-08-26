# Losses API

Complete reference for the `torchregress.losses` module. Every exported class, factory, and helper is listed here.

→ **Guides & decision aids:** [Loss Functions](../losses/index.md) · [Method Selection](../guide/method-selection.md) · [Choosing by Constraint](../guide/choosing-by-constraint.md)

---

## Module overview

All losses share a unified contract:

```python
forward(y_pred, target, mask=None, weights=None) -> Tensor | dict
```

- **`mask`**: boolean tensor marking valid (True) vs. missing (False) targets.
- **`weights`**: per-sample weight tensor broadcastable to `target`.
- **Reduction**: every loss supports `mean` / `sum` / `none` via the `reduction` constructor argument.

---

## Base classes

→ **Guide:** [Base Loss Classes](../losses/base.md)

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `BaseLoss` | `BaseLoss(reduction="mean")` | Root class: reduction, mask/weight support |
| `RegressionLoss` | `RegressionLoss(reduction="mean")` | Point-prediction losses (MSE, Huber, …) |
| `DistributionLoss` | `DistributionLoss(reduction="mean")` | Distributional losses (Gaussian NLL, MDN, …) |
| `WeightedLossWrapper` | `WeightedLossWrapper(loss_module, reduction="mean")` | Wraps any `nn.Module` with mask + weight support |
| `WeightedMSELoss` | `WeightedMSELoss(reduction="mean")` | `nn.MSELoss` + mask + weights |
| `WeightedL1Loss` | `WeightedL1Loss(reduction="mean")` | `nn.L1Loss` + mask + weights |
| `WeightedHuberLoss` | `WeightedHuberLoss(delta=1.0, reduction="mean")` | `nn.HuberLoss` + mask + weights |
| `WeightedCrossEntropyLoss` | `WeightedCrossEntropyLoss(...)` | `nn.CrossEntropyLoss` + mask + weights |
| `WeightedNLLLoss` | `WeightedNLLLoss(...)` | `nn.NLLLoss` + mask + weights |

$$\\mathcal{L} = \\frac{1}{\\sum w_i m_i} \\sum w_i m_i \\, \\ell(y_{\\text{pred}, i}, y_i)$$

---

## Gaussian & heteroscedastic

→ **Guides:** [Gaussian losses](../losses/gaussian.md) · [Faithful Gaussian](../losses/faithful_gaussian.md) · [Beta-NLL](../losses/beta_nll.md) · [Wasserstein bound](../losses/gaussian_wasserstein.md)

| Symbol | Signature | Output |
|:-------|:----------|:-------|
| `GaussianNLLLoss` | `GaussianNLLLoss(eps=1e-6, reduction="mean")` | `(mean, log_var)` |
| `FaithfulGaussianLoss` | `FaithfulGaussianLoss(mean_weight=1.0, variance_weight=1.0)` | `(mean, log_var)` |
| `GaussianCRPSLoss` | `GaussianCRPSLoss(eps=1e-6, reduction="mean")` | `(mean, log_var)` |
| `BetaNLLLoss` | `BetaNLLLoss(beta=0.5, eps=1e-6, reduction="mean")` | `(mean, log_var)` |
| `beta_nll_loss` | `beta_nll_loss(y_pred, y_true, ...)` | Functional form of `BetaNLLLoss` |
| `GaussianWassersteinBoundLoss` | `GaussianWassersteinBoundLoss(covariance_parameterization="diagonal")` | mean + cov params |
| `gaussian_wasserstein_bound_loss` | `gaussian_wasserstein_bound_loss(y_pred, y_target, ...)` | Functional form |
| `symmetric_spd_matrix_sqrt` | `symmetric_spd_matrix_sqrt(M)` | Matrix square root for Wasserstein bound |
| `MultivariateGaussianLoss` | `MultivariateGaussianLoss(...)` | mean + full Σ |
| `LowRankGaussianLoss` | `LowRankGaussianLoss(cov_rank, ...)` | mean + W·Wᵀ + D |
| `create_gaussian_nll` | `create_gaussian_nll(covariance_type="diagonal")` | Factory |
| `low_rank_output_dim` | `low_rank_output_dim(n_features, rank) → int` | → See [Utilities API](utils.md) |
| `split_low_rank_gaussian_output` | `split_low_rank_gaussian_output(out, cov_rank, target_dim)` | → See [Utilities API](utils.md) |

Core formulas:

$$\\mathcal{L}_{\\text{NLL}} = \\frac{1}{2} \\log(2\\pi \\sigma^2) + \\frac{(y - \\mu)^2}{2\\sigma^2}$$

$$\\mathcal{L}_{\\text{CRPS}} = \\sigma \\bigl[ z(2\\Phi(z)-1) + 2\\phi(z) - \\tfrac{1}{\\sqrt{\\pi}} \\bigr], \\quad z = \\tfrac{y-\\mu}{\\sigma}$$

$$\\mathcal{L}_{\\beta\\text{-NLL}} = (\\sigma^2 + \\varepsilon)^{-\\beta} \\cdot \\mathcal{L}_{\\text{NLL}}$$

---

## Robust losses

→ **Guide:** [Robust losses](../losses/robust.md)

| Symbol | Signature | Influence |
|:-------|:----------|:----------|
| `PseudoHuberLoss` | `PseudoHuberLoss(delta=1.0, reduction="mean")` | C² smooth, linear tail |
| `LogCoshLoss` | `LogCoshLoss(reduction="mean")` | tanh(r), linear tail |
| `CharbonnierLoss` | `CharbonnierLoss(eps=1e-3, reduction="mean")` | r / √(r² + ε²) |
| `CauchyLoss` | `CauchyLoss(c=1.0, reduction="mean")` | Logarithmic tail |
| `TukeyBiweightLoss` | `TukeyBiweightLoss(c=4.685, reduction="mean")` | Redescending (zero for |r| > c) |
| `AdaptiveRobustLoss` | `AdaptiveRobustLoss(...)` | Trainable α + scale |
| `BarronLoss` | `BarronLoss(alpha=1.0, c=1.0, ...)` | Continuous L1 ↔ L2 family |
| `CVaRLoss` | `CVaRLoss(alpha=0.1, ...)` | Tail α-fraction |
| `huber_elementwise` | `huber_elementwise(residuals, delta)` | Elementwise Huber; quadratic/linear crossover at `delta` |
| `log_cosh` | `log_cosh(u)` | Numerically stable `log(cosh(u))` |
| `tukey_biweight` | `tukey_biweight(residuals, c)` | Tukey biweight; constant saturation beyond `c` |

$$\\mathcal{L}_{\\text{Huber}}(r;\\delta) = \\begin{cases} \\frac{1}{2}r^2 & |r| \\le \\delta \\\\ \\delta|r| - \\frac{1}{2}\\delta^2 & |r| > \\delta \\end{cases}$$

---

## Quantile, expectile, distributional

→ **Guides:** [Quantile & expectile](../losses/quantile_expectile.md) · [MDN](../losses/mdn.md) · [Flows](../losses/nflows.md) · [Evidential](../losses/advanced.md)

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `QuantileLoss` | `QuantileLoss(quantile=0.5, reduction="mean")` | Pinball loss |
| `MultiQuantileLoss` | `MultiQuantileLoss(quantiles=[0.1,0.5,0.9])` | Joint pinball |
| `QuantileCrossoverLoss` | `QuantileCrossoverLoss(quantiles=[...])` | + non-crossing penalty |
| `ExpectileLoss` | `ExpectileLoss(expectile=0.5, reduction="mean")` | Asymmetric L2 |
| `MultiExpectileLoss` | `MultiExpectileLoss(expectiles=[...])` | Joint expectile |
| `ExpectileCrossoverLoss` | `ExpectileCrossoverLoss(expectiles=[...])` | + non-crossing penalty |
| `AsymmetricLeastSquaresLoss` | `AsymmetricLeastSquaresLoss(tau=0.5)` | Alias for ExpectileLoss |
| `QuantileCrossover` | `QuantileCrossover(quantiles=[...])` | Non-crossing penalty helper dataclass |
| `quantile_loss` | `quantile_loss(y_pred, y, tau)` | Functional pinball loss |
| `expectile_loss` | `expectile_loss(y_pred, y, tau)` | Functional expectile loss |
| `MDNLoss` | `MDNLoss(n_components=5, reduction="mean")` | Mixture Density Network NLL |
| `MixtureDensityLoss` | `MixtureDensityLoss(n_components=5, reduction="mean")` | Alias for `MDNLoss` |
| `create_mdn_loss` | `create_mdn_loss(n_components=5, ...)` | Factory for `MixtureDensityLoss` |
| `NormalizingFlowLoss` | `NormalizingFlowLoss(flow=..., reduction="mean")` | Conditional flow NLL (requires `zuko`) |
| `ContrastiveFlowLoss` | `ContrastiveFlowLoss(...)` | Contrastive likelihood-ratio flow |
| `create_flow_model` | `create_flow_model(...)` | Factory: build a flow model |
| `create_flow_loss` | `create_flow_loss(...)` | Factory: build a flow loss |
| `create_contrastive_flow_loss` | `create_contrastive_flow_loss(...)` | Factory: build a contrastive flow loss |
| `EvidentialRegressionLoss` | `EvidentialRegressionLoss(coeff=1e-2, reduction="mean")` | NIG evidential regression |

$$\\mathcal{L}_{\\text{quantile}} = \\max(q(y-\\hat{y}), (q-1)(y-\\hat{y}))$$ $$\\mathcal{L}_{\\text{expectile}} = |e - \\mathbb{I}(y<\\hat{y})| \\cdot (y-\\hat{y})^2$$ $$\\mathcal{L}_{\\text{MDN}} = -\\log \\sum_k \\pi_k \\mathcal{N}(y \\mid \\mu_k, \\sigma_k^2)$$

---

## Distribution families (`losses.families`)

Parametric NLL losses for non-Gaussian target families.

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `SkewNormalNLLLoss` | `SkewNormalNLLLoss(...)` | Skew-normal NLL (Azzalini 1985); reduces exactly to Gaussian NLL at α=0 |
| `skew_normal_nll` | `skew_normal_nll(...)` | Functional skew-normal NLL |
| `SkewTLoss` | `SkewTLoss(...)` | Skew-t NLL (Azzalini & Capitanio 2003), lgamma-normalized |
| `skew_t_nll` | `skew_t_nll(...)` | Functional skew-t NLL |
| `BetaRegressionNLLLoss` | `BetaRegressionNLLLoss(...)` | Beta-regression NLL (Ferrari & Cribari-Neto 2004) for targets in (0, 1) |
| `beta_regression_nll` | `beta_regression_nll(...)` | Functional beta-regression NLL |
| `JohnsonSUNLLLoss` | `JohnsonSUNLLLoss(...)` | Johnson-SU NLL with Shenton-quadrant (γ, δ) shapes |
| `johnson_su_nll` | `johnson_su_nll(...)` | Functional Johnson-SU NLL |
| `SinhArcsinhNLLLoss` | `SinhArcsinhNLLLoss(...)` | Sinh-arcsinh NLL (Jones & Pewsey 2009); Gaussian at ε=0, δ=1 |
| `sinh_arcsinh_nll` | `sinh_arcsinh_nll(...)` | Functional sinh-arcsinh NLL |
| `GEVNLLLoss` | `GEVNLLLoss(...)` | GEV NLL with an analytic Gumbel limit as ξ → 0 |
| `gev_nll` | `gev_nll(...)` | Functional GEV NLL |
| `AsymmetricLaplaceNLLLoss` | `AsymmetricLaplaceNLLLoss(...)` | Asymmetric-Laplace NLL; reduces to the scaled pinball NLL at a common tail scale |
| `asymmetric_laplace_nll` | `asymmetric_laplace_nll(...)` | Functional asymmetric-Laplace NLL |
| `SQRLoss` | `SQRLoss(...)` | Distributional quantile regression via sorted levels + mean pinball |
| `sqr_loss` | `sqr_loss(...)` | Functional SQR loss over `n_levels` sorted quantile levels |

---

## Special target types

→ **Guides:** [Censored](../losses/censored.md) · [Ordinal](../losses/ordinal.md) · [Poisson & Tweedie](../losses/poisson_tweedie.md) · [Poisson-Gaussian](../losses/poisson_gaussian.md)

### Ordinal

| Symbol | Signature | Output |
|:-------|:----------|:-------|
| `CumulativeLinkLoss` | `CumulativeLinkLoss(n_classes, ...)` | `K-1` logits |
| `CORALLoss` | `CORALLoss(n_classes, ...)` | `K-1` binary logits |
| `OrdinalCrossEntropyLoss` | `OrdinalCrossEntropyLoss(...)` | `K` logits |

### Censored / survival

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `CensoredGaussianNLLLoss` | `CensoredGaussianNLLLoss(reduction="mean")` | Gaussian + Φ censoring |
| `CensoredQuantileLoss` | `CensoredQuantileLoss(...)` | Quantile + censoring |
| `AFTLoss` | `AFTLoss(...)` | Accelerated Failure Time |

$$\\mathcal{L}_{\\text{censored}} = \\begin{cases} \\tfrac{1}{2}\\log(2\\pi\\sigma^2) + \\tfrac{(y-\\mu)^2}{2\\sigma^2} & c=0 \\\\ -\\log\\Phi(\\tfrac{y-\\mu}{\\sigma}) & c=-1 \\\\ -\\log(1-\\Phi(\\tfrac{y-\\mu}{\\sigma})) & c=1 \\end{cases}$$

### Count data

| Symbol | Signature | Variance |
|:-------|:----------|:---------|
| `PoissonDevianceLoss` | `PoissonDevianceLoss(log_input=True, ...)` | Var = μ |
| `PoissonLikelihoodRatioLoss` | `PoissonLikelihoodRatioLoss(log_input=True, ...)` | Var = μ (binned) |
| `NegativeBinomialNLLLoss` | `NegativeBinomialNLLLoss(learn_theta=False, ...)` | Var = μ + μ²/r |
| `ZeroInflatedPoissonNLLLoss` | `ZeroInflatedPoissonNLLLoss(log_input=True, ...)` | ZIP |
| `TweedieLoss` | `TweedieLoss(p=1.5, link="log")` | Var = φ μ^p |
| `GammaLoss` | `GammaLoss(link="log")` | Var = φ μ² |
| `InverseGaussianLoss` | `InverseGaussianLoss(link="log")` | Var = φ μ³ |
| `CompoundPoissonLoss` | `CompoundPoissonLoss(p=1.5, link="log")` | 1 < p < 2 |
| `tweedie_loss` | `tweedie_loss(y_pred, y, p=1.5, ...)` | Functional Tweedie loss |
| `PoissonGaussianMixtureLoss` | `PoissonGaussianMixtureLoss(log_input=True, ...)` | Poisson + Gaussian readout |
| `poisson_gaussian_mixture_loss` | `poisson_gaussian_mixture_loss(y_pred, y, ...)` | Functional form |
| `EnhancedPoissonGaussianMixtureLoss` | `EnhancedPoissonGaussianMixtureLoss(...)` | Gain/offset/learnable noise |
| `enhanced_poisson_gaussian_loss` | `enhanced_poisson_gaussian_loss(y_pred, y, ...)` | Functional form |
| `PoissonGaussianLikelihoodRatioLoss` | `PoissonGaussianLikelihoodRatioLoss(...)` | Likelihood-ratio variant |
| `poisson_gaussian_likelihood_ratio_loss` | `poisson_gaussian_likelihood_ratio_loss(...)` | Functional form |

---

## Data quality issues

→ **Guides:** [Imbalanced](../losses/imbalanced.md) · [Uncertain GT](../losses/uncertain_ground_truth.md) · [Noisy labels](../losses/noisy_labels.md)

| Symbol | Signature | Strategy |
|:-------|:----------|:---------|
| `BalancedMSELoss` | `BalancedMSELoss(bin_edges, ...)` | Inverse bin-frequency weights |
| `BinReweightedMSELoss` | `BinReweightedMSELoss(num_bins, *, noise_sigma=1.0, binning="equal", ...)` | Binned inverse-frequency weighted MSE |
| `DensityWeightedLoss` | `DensityWeightedLoss(...)` | Inverse target-density weights |
| `FocalRLoss` | `FocalRLoss(gamma=2.0, ...)` | Focus on hard examples |
| `LDSLoss` | `LDSLoss(...)` | Label Distribution Smoothing |
| `FeatureDistributionSmoother` | `FeatureDistributionSmoother(...)` | Feature Distribution Smoothing |
| `PropensityWeightedLoss` | `PropensityWeightedLoss(...)` | Inverse propensity weights |
| `NoisyTargetGaussianNLL` | `NoisyTargetGaussianNLL(reduction="mean")` | Adds known target-noise σ²_y |
| `PseudoLabelNLL` | `PseudoLabelNLL(pseudo_weight=0.8, ...)` | Pseudo-label + confidence NLL |
| `ConsistencyRegLoss` | `ConsistencyRegLoss(reduction="mean")` | Student–teacher MSE |
| `PseudoLabelConsistencyLoss` | `PseudoLabelConsistencyLoss(confidence_threshold=0.9, ...)` | Pseudo-label + teacher consistency |

---

## Error-in-variables (EIV)

→ **Guides:** [EIV losses](../losses/eiv.md) · **Algorithms:** [SIMEX](../methods/algorithms/simex.md) · [RC](../methods/algorithms/rc.md) · [LatentNN](../methods/algorithms/latentnn.md)

Call pattern: construct with `model=...`, then `loss(x_obs, y_obs, mask=...)`.

| Symbol | Constructor |
|:-------|:------------|
| `StructuralEIVLoss` | `StructuralEIVLoss(model=..., sigma_x=0.5, sigma_y=0.3)` |
| `FunctionalEIVLoss` | `FunctionalEIVLoss(model=..., sigma_x=0.5, sigma_y=0.3)` |
| `OrthogonalDistanceRegressionLoss` | `OrthogonalDistanceRegressionLoss(model=...)` |
| `EnsembleEIVLoss` | `EnsembleEIVLoss(model=...)` |
| `InputNoiseAugmentationLoss` | `InputNoiseAugmentationLoss(...)` |
| `InputNoiseMDNLoss` | `InputNoiseMDNLoss(...)` |
| `InputNoiseBinnedPDFLoss` | `InputNoiseBinnedPDFLoss(...)` |
| `LatentMarginalizationLoss` | `LatentMarginalizationLoss(...)` |
| `NoisyInputPredictor` | `NoisyInputPredictor(...)` |
| `ExplicitEIVAdapter` | `ExplicitEIVAdapter(...)` |
| `create_eiv_loss` | `create_eiv_loss(method="structural", ...)` |

---

## Conformal prediction

→ **Guides:** [Conformal overview](../methods/conformal/index.md) · [Predictors](../methods/conformal/predictors.md) · [Distributional](../methods/conformal/distributional.md) · **API:** [Conformal API](conformal.md)

| Symbol | Signature | Strategy |
|:-------|:----------|:---------|
| `ConformalLoss` | `ConformalLoss(method="cqr", alpha=0.1)` | Training + calibration wrapper |
| `conformal_loss` | `conformal_loss(method="cqr", alpha=0.1)` | Functional form of `ConformalLoss` |
| `ConformalPredictor` | `ConformalPredictor(...)` | Base post-hoc calibrator |
| `MultiDimensionalConformalLoss` | `MultiDimensionalConformalLoss(...)` | Legacy multi-dim wrapper |
| `SplitConformal` | `SplitConformal(alpha=0.1)` | Residual-based |
| `CQR` | `CQR(alpha=0.1, debias=False)` | Conformalized Quantile Regression |
| `UACQR` | `UACQR(alpha=0.1, ...)` | Width-normalized CQR |
| `DensityConformal` | `DensityConformal(alpha=0.1)` | Density-weighted residuals |
| `MonteCarloConformal` | `MonteCarloConformal(alpha=0.1)` | MC-sample normalized |
| `LocalConformal` | `LocalConformal(alpha=0.1, bandwidth=1.0)` | Local feature-space coverage |
| `LocalConformalMAD` | `LocalConformalMAD(alpha=0.1, ...)` | Local + MAD scaling |
| `CTI` | `CTI(alpha=0.1, grid_size=500)` | Density thresholded intervals |
| `DistributionalConformal` | `DistributionalConformal(alpha=0.1)` | PIT-based CDF calibration |
| `LevelSetConformalPredictor` | `LevelSetConformalPredictor(alpha=0.1)` | Smallest density sets |
| `MultiTargetConformal` | `MultiTargetConformal(alpha=0.1)` | Multi-target joint coverage |
| `PrevalenceAdjustedCP` | `PrevalenceAdjustedCP(alpha=0.1)` | Group-prevalence-adjusted |
| `R2CConformal` | `R2CConformal(alpha=0.1, bin_edges=...)` | Regression-as-classification CP |
| `SLSConformal` | `SLSConformal(alpha=0.1, ...)` | Super-level set conformal |
| `CVPlus` / `JackknifePlus` | `CVPlus(alpha=0.1)` | Cross-validation ensemble CP |
| `EnsembleBatchCP` | `EnsembleBatchCP(alpha=0.1)` | Bootstrap/OOB ensemble CP |

Conformal guarantee: $$P(Y_{n+1} \\in \\hat{C}(X_{n+1})) \\geq 1 - \\alpha$$

---

## Transforms & SLS

→ **Guides:** [Transforms](../losses/transforms.md) · [SLS](../losses/sls.md)

| Symbol | Signature | Description |
|:-------|:----------|:------------|
| `LogTransformLoss` | `LogTransformLoss(reduction="mean")` | log(ŷ) → base loss |
| `BoxCoxTransformLoss` | `BoxCoxTransformLoss(lam=0.0, ...)` | Box–Cox → base loss |
| `SqrtTransformLoss` | `SqrtTransformLoss(reduction="mean")` | √ŷ → base loss |
| `YeoJohnsonTransformLoss` | `YeoJohnsonTransformLoss(lam=1.0, ...)` | Yeo–Johnson → base loss |
| `TransformedTargetLoss` | `TransformedTargetLoss(transform="log", ...)` | Generic wrapper |
| `SLSLoss` | `SLSLoss(d=2, context_dim=64, K=1, tau=0.9)` | Super-Level-Set regression |
| `VolumePreservingFlow` | — | Flow with det(J)=1 |
| `MahalanobisFrontier` | — | Mahalanobis-based SLS frontier |
| `UnionFrontier` | — | Multimodal SLS frontier |

---

## Registry

| Symbol | Signature |
|:-------|:----------|
| `get_regression_loss` | `get_regression_loss(name, **kwargs)` |
| `list_regression_losses` | `list_regression_losses()` |
| `create_loss_from_config` | `create_loss_from_config(config_dict)` |
