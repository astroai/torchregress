# Loss-library test coverage map

This doc cross-references every public symbol exported from
`torchregress.losses` with the dedicated test class that locks its behavior
and the **discriminator invariant** the test guards.

When you change a loss internals formula, factory, or protocol, the table below
tells you which `tests/losses/test_*.py` file (and which `Test*` class inside
it) is the right entry point. The invariants listed are the *mathematical*
or *structural* promises the test enforces — failing any of them is the
canonical way a contract break shows up in CI.

## Conventions

- **DEDICATED** — the symbol has a `Test<S>` class (or top-level `test_*`
  function) that locks in its contract.
- **DIRECT-LIGHT** — covered indirectly via the per-symbol regression suite
  (`test_loss_fixes.py`, `test_sls_internals.py`, `test_eiv_internals.py`,
  `test_functional_wrappers.py`, `test_indirect_utilities.py`), each one
  introduced to fill gaps in the per-class suites.
- **INDIRECT** — exercised only by other tests' incidental usage. Marked
  so refactors catch them on the next review pass.

## Master matrix

> **Note:** `Test*` class names are best-effort.  If a refactor renames or
> splits a class, run `grep '^class Test' tests/losses/<file>.py` for the
> current names rather than trusting this table verbatim.  The test-file
> column is more stable -- use that as the canonical entry point.

| Symbol | Module | Test file | Test class / function | Discriminator invariant |
|---|---|---|---|---|
| `BaseLoss` | `base.py` | `tests/losses/test_base.py` | `TestBaseLoss` | `_reduce` honours `mean`/`sum`/`none`/`min`/`max`; `_reduce_with_mask` filters & re-normalises only the masked entries |
| `RegressionLoss` | `base.py` | `tests/losses/test_base.py` | `TestRegressionLoss` | `forward` raises `NotImplementedError` until subclassed; reduction flows through to subclasses |
| `DistributionLoss` | `base.py` | `tests/losses/test_base.py` | `TestDistributionLoss` | Flags `_extract_distribution_parameters` and `_calculate_nll` as abstract; concrete subclass returns scalar |
| `WeightedLossWrapper` | `base.py` | `tests/losses/test_base.py` | `TestWeightedLossWrapper` + `tests/losses/test_loss_fixes.py::TestWeightedLossWrapperPreservesReduction` | Honours the wrapped loss' pre-set reduction (`sum`/`none`); auto-mean fallback only when no prior reduction |
| `WeightedMSELoss` | `base.py` | `tests/losses/test_base.py` | `TestWeightedLossWrapper` | MSE wrapping honours mask & weights |
| `WeightedL1Loss` | `base.py` | `tests/losses/test_base.py` | `TestWeightedLossWrapper` | L1 wrapping honours reduction inheritance |
| `WeightedHuberLoss` | `base.py` | `tests/losses/test_base.py::TestWeightedClassificationLosses` (slip) + `test_functional_wrappers.py` | `TestWeightedLossWrapper` | Huber's `delta` parameter passes through unchanged |
| `WeightedCrossEntropyLoss` | `base.py` | `tests/losses/test_base.py` | `TestWeightedClassificationLosses` | Per-sample weights propagate to native `CrossEntropyLoss` |
| `WeightedNLLLoss` | `base.py` | `tests/losses/test_base.py` | `TestWeightedClassificationLosses` | Mask-filtered per-sample NLL matches native `NLLLoss` mean |
| `BalancedMSELoss` | `balanced_mse.py` | `tests/losses/test_balanced_mse.py` | (per-class suite) | Effective MSE under target-density weights |
| `BMCLoss` | `balanced_mse.py` | `tests/losses/test_balanced_mse.py` | (per-class suite) | Bayesian-MSE-style marginal under label-shift |
| `BetaNLLLoss` | `beta_nll.py` | `tests/losses/test_beta_nll.py` + `tests/losses/test_functional_wrappers.py` | `TestBetaNLLLossWrapper` | Detached `var⁻ᵝ` rescaling; reduces to vanilla Gaussian NLL at `β=0` |
| `beta_nll_loss` | `beta_nll.py` | `tests/losses/test_functional_wrappers.py` | `TestBetaNLLLossWrapper::test_parity_with_class` | Functional wrapper parity with `BetaNLLLoss` on tuple & concat inputs |
| `CensoredGaussianNLLLoss` | `censored.py` | `tests/losses/test_censored.py` | (per-class) | Censoring mask propagates to NLL |
| `CensoredQuantileLoss` | `censored.py` | `tests/losses/test_censored.py` | (per-class) | Pinball loss with left-/right-censoring adjustments |
| `AFTLoss` | `censored.py` | `tests/losses/test_censored.py` | (per-class suite) | Accelerated-failure-time log-location residual |
| `CQR` | `conformal.py` | `tests/losses/test_conformal.py` | `TestCQR` + `TestConformalLossCQRDebias` | Debias widens via `α·n/(n+1)`; alpha restoration after calibrate; calibrated interval is `[q_lo - q_hat, q_hi + q_hat]` |
| `CTI` | `conformal.py` | `tests/losses/test_conformal.py` | `TestCTI` | Smallest density-level-set interval via grid search |
| `SLSConformal` | `conformal.py` | `tests/losses/test_sls_conformal.py` | (per-class) | Score = `G(X,Y) / q_τ(X)`; interval from frontier level set |
| `ConformalLoss` | `conformal.py` | `tests/losses/test_conformal.py` | `TestSplitConformal`, `TestCQR`, `TestConformalLossCQRDebias`, etc. | Method-routes to underlying predictor; pinball training loss for cqr/uacqr |
| `conformal_loss` | `conformal.py` | `tests/losses/test_functional_wrappers.py` | `TestConformalLossWrapper` | Functional wrapper parity (split + cqr) |
| `ConformalPredictor` | `conformal.py` | `tests/losses/test_conformal.py` | (base for `Test*Conformal`) | Virtual `_compute_scores`/`_build_intervals`; Mondrian + normalised + weighted composable features |
| `LevelSetConformalPredictor` | `conformal.py` | `tests/losses/test_conformal.py` | (via `TestCTI`, `TestDistributionalConformal`, `TestSLSConformal`) | Shared `_grid_search_level_set` utility |
| `LocalConformal` | `conformal.py` | `tests/losses/test_conformal.py` | `TestLocalConformal` + `tests/losses/test_loss_fixes.py::TestLocalConformalSentinelFallback` | Sentinel `True` column guards against underflow when cum-weights miss `1-α` |
| `LocalConformalMAD` | `conformal.py` | `tests/losses/test_conformal.py` | `TestLocalConformalMAD` + `tests/losses/test_loss_fixes.py::TestLocalConformalSentinelFallback` | Same sentinel behaviour under MAD scaling |
| `SplitConformal` | `conformal.py` | `tests/losses/test_conformal.py` | `TestSplitConformal` | \|residual\| score; symmetric `ŷ ± q_hat` interval |
| `UACQR` | `conformal.py` | `tests/losses/test_conformal.py` | (per-class) | CQR with width-normalised scores |
| `DensityConformal` | `conformal.py` | `tests/losses/test_conformal.py` | `TestDensityConformal` | KDE-adaptive score normalisation |
| `DistributionalConformal` | `conformal.py` | `tests/losses/test_conformal.py` | `TestDistributionalConformal` | PIT-based score; ICDF interval construction |
| `MonteCarloConformal` | `conformal.py` | `tests/losses/test_conformal.py` | `TestMonteCarloConformal` | MC sample mean/median + std normalisation |
| `MultiDimensionalConformalLoss` | `conformal.py` | `tests/losses/test_conformal.py` | (per-class) | Per-dimension quantile thresholds |
| `MultiTargetConformal` | `conformal.py` | `tests/losses/test_conformal.py` | (per-class) | Per-dimension calibration independence |
| `PrevalenceAdjustedCP` | `conformal.py` | `tests/losses/test_conformal.py` | `TestPrevalenceAdjustedCP` | Group-prevalence scales miscoverage rate |
| `R2CConformal` | `conformal.py` | `tests/losses/test_conformal.py` | `TestR2CConformal` | APS-style sorted-bin inclusion |
| `BaseEIVLoss` | `eiv.py` | `tests/losses/test_eiv_internals.py` | `TestBaseEIVLossInternals` | `_prepare_covariance_from_sigma` interpretation of scalar / 1d / 2d / 3d sigma; `explicit()` factory returns adapter |
| `EnsembleEIVLoss` | `eiv.py` | `tests/losses/test_eiv.py` | `TestEIVLoss::test_ensemble_eiv_loss` | Gaussian-perturbation ensemble averaging |
| `ExplicitEIVAdapter` | `eiv.py` | `tests/losses/test_eiv_internals.py` | `TestExplicitEIVAdapter` | Adapter forwards `sigma_x`/`sigma_y` override at call-site even if not set in constructor |
| `FunctionalEIVLoss` | `eiv.py` | `tests/losses/test_eiv.py` | `TestEIVLoss::test_functional_eiv_loss` + `TestEIVLossNumericalStability` | Jacobian-variance + NLL; analytical/mc/hybrid branches finite & monotonic in `n_samples` |
| `InputNoiseBinnedPDFLoss` | `eiv.py` | (specialized class — INDIRECT, exercised via `test_eiv.py`) | — | Binned-PDF marginalisation with Ordinal base |
| `InputNoiseMarginalizationLoss` | `eiv.py` | `tests/losses/test_eiv.py` | `TestEIVLoss::test_input_noise_marginalization_loss` + `test_input_noise_predictive_average_*` | `sample_predictions` shape; antithetic sampling; per-sample weights pass through |
| `InputNoiseMDNLoss` | `eiv.py` | (per-class — INDIRECT via `test_mdn.py`) | — | MDN base loss + marginalisation |
| `NoisyInputPredictor` | `eiv.py` | `tests/losses/test_eiv_internals.py` | `TestNoisyInputPredictor` | `forward(x)` returns mean over MC perturbations; antithetic yields distinct rows; non-tensor model is rejected |
| `OrthogonalDistanceRegressionLoss` | `eiv.py` | `tests/losses/test_eiv.py` | `TestEIVLoss::test_odr_loss` + `TestEIVLossNumericalStability::test_odr_gradient_flow` | Latent-x optimisation step (approx) recovers Mahalanobis sum |
| `StructuralEIVLoss` | `eiv.py` | `tests/losses/test_eiv.py` | `TestEIVLoss::test_structural_eiv_loss` | Cross-covariance `σ_xy` propagates through NLL |
| `create_eiv_loss` | `eiv.py` | (INDIRECT — used inside `examples/`) | — | Routes on `loss_type` keyword |
| `EvidentialRegressionLoss` | `evidential.py` | `tests/losses/test_evidential.py` | `TestEvidentialRegressionLoss` | NIG posterior + KL regularisation |
| `ExpectileLoss` | `expectile.py` | `tests/losses/test_expectile.py` + `tests/losses/test_functional_wrappers.py` | `TestExpectileLoss` + `TestExpectileLossWrapper` | `τ=0.5` collapses to MSE; asymmetric weights `τ`/`1-τ` |
| `expectile_loss` | `expectile.py` | `tests/losses/test_functional_wrappers.py` | `TestExpectileLossWrapper` | Parity with `ExpectileLoss` class |
| `MultiExpectileLoss` | `expectile.py` | `tests/losses/test_non_gaussian_consistency.py` | `TestExpectileContract` | Average across expectile levels per sample |
| `AsymmetricLeastSquaresLoss` | `expectile.py` | `tests/losses/test_expectile.py` | (per-class) | Subclass of `ExpectileLoss` accepts `tau=` alias |
| `ExpectileCrossover` | `expectile.py` | `tests/losses/test_indirect_utilities.py` | `TestCrossoverAliases::test_expectile_crossover_is_class_alias` | `ExpectileCrossover IS ExpectileCrossoverLoss` |
| `ExpectileCrossoverLoss` | `expectile.py` | `tests/losses/test_non_gaussian_consistency.py` | `TestExpectileContract` | Crossover penalty on pairs of predictands |
| `FaithfulGaussianLoss` | `faithful_gaussian.py` | `tests/losses/test_faithful_gaussian.py` | (per-class) | Mean / variance decoupled layers |
| `GaussianCRPSLoss` | `gaussian.py` | `tests/losses/test_gaussian.py` | `TestGaussianLosses::test_gaussian_crps_loss_matches_metric` | Analytic CRPS matches shared `crps_gaussian` metric |
| `GaussianNLLLoss` | `gaussian.py` | `tests/losses/test_gaussian.py` | `TestGaussianLosses::test_gaussian_nll_loss` + `test_gaussian_consistency.py` | Accepts tuple / concat / mean-only formats; reduction modes |
| `LowRankGaussianLoss` | `gaussian.py` | `tests/losses/test_gaussian.py` | `TestGaussianLosses::test_low_rank_gaussian_loss` | `cov_factor @ cov_factor.T + diag(cov_diag)` NLL |
| `MultivariateGaussianLoss` | `gaussian.py` | `tests/losses/test_gaussian.py` + consistency suite | `TestGaussianLosses::test_multivariate_gaussian_loss` | Cholesky NLL with jitter; respects shape contracts |
| `create_gaussian_nll` | `gaussian.py` | `tests/losses/test_indirect_utilities.py` | `TestCreateGaussianNll` | Factory routes by `covariance_type`; `full` ≈ `multivariate` route identity; `use_mse_for_unit_variance` → `WeightedMSELoss` |
| `GaussianWassersteinBoundLoss` | `gaussian_wasserstein.py` | `tests/losses/test_gaussian_wasserstein.py` + consistency hybrid tests | (per-class) | \|μ-μ̂\|² + \|Σ½-Σ̂½\|_F² in 4 covariance parameterisations |
| `gaussian_wasserstein_bound_loss` | `gaussian_wasserstein.py` | `tests/losses/test_functional_wrappers.py` | `TestGaussianWassersteinBoundWrapper` | Functional parity across diag / covariance / sqrt / cholesky modes |
| `symmetric_spd_matrix_sqrt` | `gaussian_wasserstein.py` | `tests/losses/test_indirect_utilities.py` | `TestSymmetricSpsMatrixSqrt` | `eigh`-based sqrt: symmetric, `S·S = Σ`, `eps` floors singletons |
| `DensityWeightedLoss` | `imbalanced.py` | `tests/losses/test_imbalanced.py` | (per-class) | Target-density reweighting |
| `FocalRLoss` | `imbalanced.py` | `tests/losses/test_imbalanced.py` | (per-class) | Focal-style per-sample modulation |
| `LDSLoss` | `imbalanced.py` | `tests/losses/test_imbalanced.py` | (per-class) | Label-distribution-shift reweighting |
| `PropensityWeightedLoss` | `imbalanced.py` | `tests/losses/test_imbalanced.py` | (per-class) | Inverse-propensity weighting |
| `get_regression_loss` | `loss_registry.py` | `tests/losses/test_loss_registry.py` | (per-class) | Lookup by registry key |
| `list_regression_losses` | `loss_registry.py` | `tests/losses/test_loss_registry.py` | (per-class) | Sorted list of registered names |
| `create_loss_from_config` | `loss_registry.py` | `tests/losses/test_loss_registry.py` | (per-class) | YAML-style config construction with aliases |
| `MixtureDensityLoss` (re-exported as `MDNLoss`) | `mdn.py` | `tests/losses/test_mdn.py` | (per-class) | Full-cov Cholesky backward finite & differentiable |
| `create_mdn_loss` | `mdn.py` | `tests/losses/test_indirect_utilities.py` | `TestCreateMdnLoss` | Factory routes kwargs to `MixtureDensityLoss`; rejects bad `covariance_type`; end-to-end forward + backward smoke |
| `SLSLoss` | `sls.py` | `tests/losses/test_sls.py` + consistency suite | (top-level `test_sls_loss_warmup_and_forward`, etc.) | Step-counter monotonic advance; warmup window; K>1 unfreeze gating |
| `VolumePreservingFlow` | `sls.py` | `tests/losses/test_sls.py` (round-trip) + `tests/losses/test_sls_internals.py` | `TestVolumePreservingFlow` | Invertibility: `flow.inverse(flow(y)) == y`; alternating half-mask per layer |
| `MahalanobisFrontier` | `sls.py` | `tests/losses/test_sls_internals.py` | `TestMahalanobisFrontierFull`, `TestMahalanobisFrontierLowRank` | Full-mode `num_L_params = d(d+1)/2`; low-rank default `rank = ⌈√d⌉`; `G ≥ 0` invariant |
| `UnionFrontier` | `sls.py` | `tests/losses/test_sls_internals.py` | `TestUnionFrontier` | K-component `MahalanobisFrontier` list; freeze/unfreeze gating; mixture weights sum to 1 |
| `NormalizingFlowLoss` | `nflows.py` | `tests/losses/test_nflows.py` | (per-class) | zuko-backed wrapper; loss wiring |
| `ContrastiveFlowLoss` | `nflows.py` | `tests/losses/test_contrastive_nflows.py` | (per-class) | Contrastive term |
| `create_flow_model` | `nflows.py` | (INDIRECT inside `test_nflows.py`) | — | Build flow from config |
| `create_flow_loss` | `nflows.py` | (INDIRECT inside `test_nflows.py`) | — | Build flow loss from config |
| `create_contrastive_flow_loss` | `nflows.py` | (INDIRECT — special optional) | — | Build contrastive flow loss |
| `OrdinalCrossEntropyLoss` | `ordinal.py` | `tests/losses/test_ordinal.py` | (per-class) | Label-smoothed ordinal CE |
| `CumulativeLinkLoss` | `ordinal.py` | `tests/losses/test_ordinal.py` | (per-class) | Cumulative-link logistic loss |
| `CORALLoss` | `ordinal.py` | `tests/losses/test_ordinal.py` | (per-class) | CORAL rank-consistent ordinal loss |
| `NegativeBinomialNLLLoss` | `poisson.py` | `tests/losses/test_poisson.py` | (per-class) | NB NLL with dispersion parameter |
| `PoissonDevianceLoss` | `poisson.py` | `tests/losses/test_poisson.py` | (per-class) | Poisson deviance residual |
| `PoissonLikelihoodRatioLoss` | `poisson.py` | `tests/losses/test_poisson.py` | (per-class) | LR loss vs. Poisson mean |
| `ZeroInflatedPoissonNLLLoss` | `poisson.py` | `tests/losses/test_poisson.py` | (per-class) | ZIP NLL with mixing parameter |
| `PoissonGaussianMixtureLoss` | `poisson_gaussian.py` | `tests/losses/test_poisson_gaussian.py` | (per-class) + consistency | Poisson+Gaussian NLL mixture |
| `poisson_gaussian_mixture_loss` | `poisson_gaussian.py` | `tests/losses/test_functional_wrappers.py` | `TestPoissonGaussianFactoryWrappers::test_mixture_factory_returns_instance` | Config ↔ kwargs plumbing; class identity |
| `EnhancedPoissonGaussianMixtureLoss` | `poisson_gaussian.py` | `tests/losses/test_poisson_gaussian.py` | (per-class) + consistency | Gain-offset mixture w/ learnable params |
| `enhanced_poisson_gaussian_loss` | `poisson_gaussian.py` | `tests/losses/test_functional_wrappers.py` | `TestPoissonGaussianFactoryWrappers::test_enhanced_factory_returns_instance` | Config plumbing |
| `PoissonGaussianLikelihoodRatioLoss` | `poisson_gaussian.py` | `tests/losses/test_poisson_gaussian.py` + consistency | (per-class) | LR mixture NLL |
| `poisson_gaussian_likelihood_ratio_loss` | `poisson_gaussian.py` | `tests/losses/test_functional_wrappers.py` | `TestPoissonGaussianFactoryWrappers::test_lr_factory_returns_instance` | Config plumbing + default `log_input=False` |
| `MultiQuantileLoss` | `quantile.py` | `tests/losses/test_quantile.py` + `tests/losses/test_non_gaussian_consistency.py` | `TestMultiQuantileLoss` | Average across multiple quantiles per sample |
| `QuantileCrossover` | `quantile.py` | `tests/losses/test_indirect_utilities.py` | `TestCrossoverAliases` | `QuantileCrossover IS QuantileCrossoverLoss`; aliases accept strictly non-default kwargs that reach live attributes |
| `QuantileCrossoverLoss` | `quantile.py` | `tests/losses/test_quantile.py` + consistency | `TestQuantileLoss::test_quantile_crossover_constraint` | Crossover penalty `Σᵢ max(fᵢ - fᵢ₊₁, 0)` |
| `QuantileLoss` | `quantile.py` | `tests/losses/test_quantile.py` + `tests/losses/test_functional_wrappers.py` | `TestQuantileLoss`, `TestQuantileLossWrapper` | Asymmetric pinball; `q=0.5` recovers MAE |
| `quantile_loss` | `quantile.py` | `tests/losses/test_functional_wrappers.py` | `TestQuantileLossWrapper` | Parity with `QuantileLoss` class |
| `PseudoHuberLoss` | `robust.py` | `tests/losses/test_robust.py` + consistency | (per-class) | Smooth L1 alternative |
| `LogCoshLoss` | `robust.py` | `tests/losses/test_robust.py` + consistency | (per-class) | Log-cosh residual |
| `CharbonnierLoss` | `robust.py` | `tests/losses/test_robust.py` + consistency | (per-class) | `√(r²+ε²)` stem |
| `TukeyBiweightLoss` | `robust.py` | `tests/losses/test_robust.py` + consistency | (per-class) | Biweight ψ-function |
| `CauchyLoss` | `robust.py` | `tests/losses/test_robust.py` + consistency | (per-class) | log(1+(r/c)²) |
| `BarronLoss` | `robust.py` | `tests/losses/test_robust.py` + consistency | (per-class) | Continuously-interpolated L2/L1 shape via α |
| `AdaptiveRobustLoss` | `robust.py` | `tests/losses/test_robust.py` + consistency | (per-class) | Learnable α/β/scale |
| `CVaRLoss` | `robust.py` | `tests/losses/test_cvar.py` | (per-class) | Tail-risk expectation above `α` |
| `BoxCoxTransformLoss` | `transforms.py` | `tests/losses/test_transforms.py` | (per-class) | Box-Cox transform + base loss |
| `LogTransformLoss` | `transforms.py` | `tests/losses/test_transforms.py` | (per-class) | log + base loss |
| `SqrtTransformLoss` | `transforms.py` | `tests/losses/test_transforms.py` | (per-class) | √ + base loss |
| `TransformedTargetLoss` | `transforms.py` | `tests/losses/test_transforms.py` | (per-class) | Generic transform wrapper |
| `YeoJohnsonTransformLoss` | `transforms.py` | `tests/losses/test_transforms.py` | (per-class) | Yeo-Johnson (handles non-positive targets) |
| `CompoundPoissonLoss` | `tweedie.py` | `tests/losses/test_tweedie.py` | (per-class) | Compound Poisson-Gamma (1 < p < 2) |
| `GammaLoss` | `tweedie.py` | `tests/losses/test_tweedie.py` | (per-class) | Gamma loss (p=2) |
| `InverseGaussianLoss` | `tweedie.py` | `tests/losses/test_tweedie.py` | (per-class) | IG loss (p=3) |
| `TweedieLoss` | `tweedie.py` | `tests/losses/test_tweedie.py` + `tests/losses/test_functional_wrappers.py` | `TestTweedieLoss`, `TestTweedieLossWrapper` | Routing by `p`: 0→normal, 1→Poisson, 2→gamma, 3→IG, 1<p<2→compound |
| `tweedie_loss` | `tweedie.py` | `tests/losses/test_functional_wrappers.py` | `TestTweedieLossWrapper` | Wrapper parity |
| `ConsistencyRegLoss` | `uncertain_gt.py` | `tests/losses/test_uncertain_gt.py` | (per-class) | EMA-style consistency penalty |
| `NoisyTargetGaussianNLL` | `uncertain_gt.py` | `tests/losses/test_uncertain_gt.py` | (per-class) | Label-noise-aware Gaussian NLL |
| `PseudoLabelConsistencyLoss` | `uncertain_gt.py` | `tests/losses/test_uncertain_gt.py` | (per-class) | Pseudo-label agreement penalty |
| `PseudoLabelNLL` | `uncertain_gt.py` | `tests/losses/test_uncertain_gt.py` | (per-class) | NLL under pseudo-label distribution |
| `low_rank_output_dim` | `utils/gaussian_output.py` | `tests/losses/test_indirect_utilities.py` | `TestLowRankOutputDim` | `2F + F·R` formula; monotonicity; `ValueError` on `n_features<=0` or `rank<=0` |
| `split_low_rank_gaussian_output` | `utils/gaussian_output.py` | `tests/losses/test_indirect_utilities.py` | `TestSplitLowRankGaussianOutput` | Round-trip `[mean \| factor \| diag]`; unbatched shape; `ValueError` on wrong out_dim |

### Internal helpers in `utils/gaussian_output.py`

The three helpers below are not in `__all__` — they remain internal contract
helpers *outside* the public `losses` API. They are exercised indirectly by
their `BetaNLLLoss`/`GaussianNLLLoss`/`FaithfulGaussianLoss` callers but have
no dedicated test class. Treat any change as a cross-module refactor that
should land alongside a direct test in
`tests/losses/test_indirect_utilities.py`.

- `parse_heteroscedastic_output` — tuple/dict/concat-2D routing
- `split_mean_log_variance` — tuple/dict/concat routing; `ValueError` on odd dims
- `variance_from_logvar` — clamp + exp signing

## Per-class test surface (consistency / cross-family)

| File | Purpose |
|---|---|
| `test_gaussian_consistency.py` | Cross-family Gaussian invariants (input formats, mask/weights contracts, gradient contracts, CRPS stability, Wasserstein relationships). Locks every variant of the Gaussian loss family to the same input contract. |
| `test_non_gaussian_consistency.py` | Cross-family contracts for Poisson-Gaussian, censored, Tweedie, quantile, ordinal, conformal, EIV, expectile, evidential, robust, balanced-MSE families. |
| `test_loss_fixes.py` | Regression tests for the **5 documented loss-bugs** — `SLSLoss.step_counter`, `WeightedLossWrapper.reduction`, `MixtureDensityLoss.full-cov Cholesky backward`, `LocalConformal.predict_interval sentinel`, `CQR.debias factor`. Use this file first when investigating these invariants. |
| `test_sls_internals.py` | SLS internal modules — `VolumePreservingFlow` / `MahalanobisFrontier` (full + low-rank) / `UnionFrontier` / `QuantileNetwork`. |
| `test_eiv_internals.py` | EIV internal modules — `ExplicitEIVAdapter` / `NoisyInputPredictor` / `BaseEIVLoss._prepare_covariance_from_sigma`. |
| `test_functional_wrappers.py` | The 9 functional wrappers (`quantile_loss`, `expectile_loss`, `beta_nll_loss`, `tweedie_loss`, `conformal_loss`, `gaussian_wasserstein_bound_loss`, 3× Poisson-Gaussian factories). |
| `test_indirect_utilities.py` | Public indirect utilities — `low_rank_output_dim`, `split_low_rank_gaussian_output`, `symmetric_spd_matrix_sqrt`, `create_gaussian_nll`, `create_mdn_loss`, plus the `QuantileCrossover`/`ExpectileCrossover` aliases. |
| `test_loss_registry.py` | `get_regression_loss` / `list_regression_losses` / `create_loss_from_config`. |

## Discriminator invariants: where to look when refactoring

| Invariant category | Canonical test entrance | Module-level fallback |
|---|---|---|
| **NLL formula and reduction** | `test_gaussian.py::TestGaussianLosses` + `test_gaussian_consistency.py::TestMaskContract` | `test_beta_nll.py`, `test_poisson_gaussian.py` |
| **Pinball / quantile formulation** | `test_quantile.py::TestQuantileLoss` + `tests/losses/test_functional_wrappers.py::TestQuantileLossWrapper` | `test_non_gaussian_consistency.py` |
| **Conformal debias direction & coverage** | `test_loss_fixes.py::TestCQRDebiasDocCodeAlignment` + `test_conformal.py::TestConformalLossCQRDebias` | `test_conformal.py` |
| **Determinism across dataloader calls (sentinel underflow)** | `test_loss_fixes.py::TestLocalConformalSentinelFallback` | `test_conformal.py::TestLocalConformal` |
| **Volume-preserving flow invertibility** | `test_sls.py::test_volume_preserving_*` + `test_sls_internals.py::TestVolumePreservingFlow` | `test_sls_internals.py` |
| **Mahalanobis frontier shape & rank default** | `test_sls_internals.py::TestMahalanobisFrontier{LowRank,Full}` | `test_sls.py::test_mahalanobis_frontier_full_and_low_rank` |
| **K>1 union mixture weights** | `test_sls_internals.py::TestUnionFrontier::test_unfrozen_mixture_weights_sum_to_one_per_sample` | `test_sls.py::test_union_frontier` |
| **MDN Cholesky backward (no leaf-tensor in-place write)** | `test_loss_fixes.py::TestMixtureDensityLossFullCovBackward` | `test_mdn.py` |
| **Wrapper preserves caller-set reduction** | `test_loss_fixes.py::TestWeightedLossWrapperPreservesReduction` | `test_base.py::TestWeightedLossWrapper` |
| **SLS curriculum monotonic advance** | `test_loss_fixes.py::TestSLSLossStepCounterAdvances` | `test_sls.py::test_sls_loss_warmup_and_forward` |
| **Functional-wrapper parity (loss vs class)** | All `Test*Wrapper` classes in `test_functional_wrappers.py` + `test_indirect_utilities.py` | per-class suite |
| **Gaussian Wasserstein bound 4-mode parity** | `test_functional_wrappers.py::TestGaussianWassersteinBoundWrapper::test_*_mode` | `test_gaussian_wasserstein.py` |

## INDIRECT-only symbols (refactor risk)

The following public symbols have **no dedicated test class**:

- `create_eiv_loss` (used in examples)
- `create_flow_model`, `create_flow_loss`, `create_contrastive_flow_loss` (zuko-dependent, optional)

Any **non-trivial** change to these symbols should land alongside a small
direct test in `tests/losses/test_indirect_utilities.py` (or *specific*
file) so their behaviour doesn't drift silently.

## Per-class test surfaces

For matrix rows citing `(per-class suite)`, the dedicated `Test*` classes live
in the file under **Test file**. Use `grep '^class Test' <test_file>` to list
the exact class names — the names change frequently and the matrix does not
attempt to enumerate them. Cross-consistency coverage is documented per file
in `test_gaussian_consistency.py` and `test_non_gaussian_consistency.py`.

## Known silent regressions

The following invariants are documented. Pre-fix code raises the listed
error under older PyTorch but is silently absorbed under modern PyTorch's
version-counter autograd; the listed regression test now discriminates
both states at runtime, so the original caveat no longer applies:

- **`MixtureDensityLoss` H2 in-place scatter.** Post-fix code composes the
  full-cov Cholesky factor functionally via ``torch.where(diag_mask,
  L_diag, L_offdiag)``, yielding a brand-new ``L_matrices`` tensor with
  ``_version == 0`` at forward-return time.  Pre-fix code derived
  ``L_matrices`` via two in-place ``__setitem__`` writes onto
  ``L_offdiag``, bumping its version to ``>= 2`` by the time it was
  returned.The discriminator
``tests/losses/test_loss_fixes.py::TestMixtureDensityLossFullCovBackward::test_full_covariance_L_version_zero``
  wraps ``_extract_distribution_parameters`` to capture
  ``L_matrices._version`` immediately after forward and asserts it
  equals 0, surviving PyTorch's silent absorption of the in-place
  pattern.

## Coverage invariants: what test authors MUST pin

Future test authors must follow this discipline when introducing any
ad-hoc reference to a covariance, sigma, stddev, or Cholesky factor in a
test fixture. **Ad-hoc ``torch.eye(...)`` / ``torch.diag(...)`` /
``torch.diag_embed(...)`` literals MUST pin ``device`` and ``dtype``** so
the fixture does not silently rely on the loss module handling
dtype/device of input fixtures internally. Two accepted forms depending
on whether the literal is on the *result* side or the *input* side of
the comparison:

- **Result comparison** (the literal is the reference for a tensor
  returned by a loss module). Pin the literal to the result's metadata:
  ```python
  ref = torch.eye(n_features, device=cov.device, dtype=cov.dtype) * 0.25
  self.assertTrue(torch.allclose(cov, ref))
  ```
- **Input builder** (the literal is part of a fixture fed *into* a loss
  module — e.g. ``A @ A.T + torch.eye(dim) * jitter``). Pin to the
  co-built anchor tensor that's already on the right device/dtype:
  ```python
  A = torch.randn(dim, dim)
  base = A @ A.T + torch.eye(dim, device=A.device, dtype=A.dtype) * 1e-3
  ```

The legacy ``.to(cov.device, dtype=cov.dtype)`` cast on the literal is
accepted as a fallback when the ``torch.eye(...)`` / ``torch.diag(...)``
form would change the literal's mathematical meaning (e.g. when the
literal is a hand-built third-party reference like ``torch.tensor([...])``
the test materials outside an obvious anchor).

### Why this discipline matters

``torch.allclose`` and ``torch.testing.assert_close`` are
dtype/device-tied at the framework level. ``torch.eye(3)`` silently
defaults to float32/CPU; if a future loss module grows a
device-dispatching code path (e.g. CUDA NLL, autograd-double NLL, the
``to("cuda")`` migration in `test_gaussian.py`'s ``setUp``) the test
would silently pass under old builds while passing against mismatched
metadata under new builds — a covered-by-follow-up symptom of implicit
fixture metadata rather than a single-step failure flagged by CI.

Concretely: under a dtype/device mismatch, ``torch.allclose`` raises
a ``RuntimeError`` rather than returning the comparison result. CI fails
with an unrelated-looking error message (e.g. ``Found dtype Float but
expected Double``), masking the real loss-graph regression behind a
metadata mismatch — the developer fixes the missing dtype pin rather
than the actual loss bug the test was guarding.

### Canonical examples for reference while authoring new tests

- ``tests/losses/test_eiv_internals.py::TestBaseEIVLossInternals::test_prepare_covariance_*`` — the canonical *result-comparison* form.  Pin the reference to ``cov`` after ``_prepare_covariance_from_sigma``: ``torch.eye(n_features, device=cov.device, dtype=cov.dtype) * 0.25``.
- ``tests/losses/test_functional_wrappers.py::TestGaussianWassersteinBoundWrapper.setUp`` — the canonical *input-builder* form.  Pin the SPD builder's eye to the co-built ``A``: ``A @ A.T + torch.eye(self.d, device=A.device, dtype=A.dtype)``.
- All ``_make_spd_cov``-style helpers in ``tests/losses/test_gaussian_consistency.py`` and the ``_psd_matrix`` helper in ``tests/losses/test_indirect_utilities.py`` reproduce the *input-builder* form on the helper's anchor ``A``.
- ``tests/losses/test_gaussian.py::TestGaussianLosses.setUp`` — adds a ``dtype=`` pin on top of an existing ``device=`` pin for the SPD jitter term.

### Cross-cutting rule for code review

When reviewing or extending any test in ``tests/losses/``, ``tests/``,
``notebooks/``, or ``examples/``, re-grep ``torch.eye`` and
``torch.diag`` / ``torch.diag_embed``. Any unpinned site that
- constructs an input fixture (an ``A @ A.T + eye(dim) * jitter``
  builder, an eye-in-a-list literal passed as a sigma parameter,
  etc.), or
- compares against a result tensor in ``torch.allclose`` / ``torch.testing.assert_close``,

is a **coverage-invariance violation** under this rule. Fix before
merging, exactly per the patterns above.

## TL;DR for code review

When reviewing a loss change, look for:

1. The matrix row for the changed symbol — confirm the test class actually
   enforces the changed invariant (not just instantiates the loss).
2. If you can't find a row, add a dedicated test before merging; the
   `INDIRECT`-only section flags where the gap lives.
3. Re-running `torchregress/scripts/revert_verify.py` after a change gives
   empirical evidence the test discriminates the bug class the change
   risks.
