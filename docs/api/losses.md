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
| `GaussianWassersteinBoundLoss` | `GaussianWassersteinBoundLoss(covariance_parameterization="diagonal")` | mean + cov params |
| `MultivariateGaussianLoss` | `MultivariateGaussianLoss(...)` | mean + full Σ |
| `LowRankGaussianLoss` | `LowRankGaussianLoss(cov_rank, ...)` | mean + W·Wᵀ + D |
| `create_gaussian_nll` | `create_gaussian_nll(covariance_type="diagonal")` | Factory |

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
| `MDNLoss` | `MDNLoss(n_components=5, reduction="mean")` | Mixture Density Network NLL |
| `NormalizingFlowLoss` | `NormalizingFlowLoss(flow=..., reduction="mean")` | Conditional flow NLL (requires `zuko`) |
| `ContrastiveFlowLoss` | `ContrastiveFlowLoss(...)` | Contrastive likelihood-ratio flow |
| `EvidentialRegressionLoss` | `EvidentialRegressionLoss(coeff=1e-2, reduction="mean")` | NIG evidential regression |

$$\\mathcal{L}_{\\text{quantile}} = \\max(q(y-\\hat{y}), (q-1)(y-\\hat{y}))$$ $$\\mathcal{L}_{\\text{expectile}} = |e - \\mathbb{I}(y<\\hat{y})| \\cdot (y-\\hat{y})^2$$ $$\\mathcal{L}_{\\text{MDN}} = -\\log \\sum_k \\pi_k \\mathcal{N}(y \\mid \\mu_k, \\sigma_k^2)$$

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
| `PoissonGaussianMixtureLoss` | `PoissonGaussianMixtureLoss(log_input=True, ...)` | Poisson + Gaussian readout |
| `EnhancedPoissonGaussianMixtureLoss` | `EnhancedPoissonGaussianMixtureLoss(...)` | Gain/offset/learnable noise |

---

## Data quality issues

→ **Guides:** [Imbalanced](../losses/imbalanced.md) · [Uncertain GT](../losses/uncertain_ground_truth.md) · [Noisy labels](../losses/noisy_labels.md)

| Symbol | Signature | Strategy |
|:-------|:----------|:---------|
| `BalancedMSELoss` | `BalancedMSELoss(bin_edges, ...)` | Inverse bin-frequency weights |
| `BMCLoss` | `BMCLoss(bin_edges, noise_sigma=0.0, ...)` | Balanced MSE + Gaussian smoothing |
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
| `InputNoiseMarginalizationLoss` | `InputNoiseMarginalizationLoss(...)` |
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
