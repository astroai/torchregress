# Generated Method Catalog Snapshot

_Generated provenance_: `tools/render_method_catalog.py:render_markdown`
_Source artifacts_: `reports/method_catalog_latest.json`
_Generated date_: `2026-04-16`

It is a code-backed snapshot used to reduce docs drift in the task-first matrix.

## Summary

- Total methods: `125`
- Peer methods present: `SWAG`, `BayesianNeuralNetwork`, `MDNLoss`

### By Family

| Family | Count |
|---|---:|
| `bnn` | 7 |
| `calibration_transform` | 5 |
| `causal` | 2 |
| `censored` | 3 |
| `conformal` | 13 |
| `constraints` | 2 |
| `eiv` | 15 |
| `ensemble` | 7 |
| `evidential` | 1 |
| `expectile` | 3 |
| `flow` | 3 |
| `imbalanced_loss` | 7 |
| `inference` | 1 |
| `mc_dropout` | 1 |
| `mdn` | 1 |
| `multivariate` | 1 |
| `ordinal` | 3 |
| `point_loss` | 1 |
| `poisson` | 4 |
| `poisson_gaussian` | 3 |
| `probabilistic_loss` | 8 |
| `quantile` | 2 |
| `robust_loss` | 9 |
| `semi_supervised` | 1 |
| `swag` | 2 |
| `target_transform` | 5 |
| `test_time` | 6 |
| `tictac` | 1 |
| `tweedie` | 4 |
| `uncertain_gt` | 4 |

### By Maturity

| Maturity | Count |
|---|---:|
| `Core` | 7 |
| `Strong` | 20 |
| `Available` | 97 |
| `Deprecated` | 1 |

## Method Rows

Legend: `*` suffix means partial support. This snapshot is for discovery and comparison, not absolute ranking.

| Name | Family | Maturity | Task Tags | Core Capability Signals | Public Path |
|---|---|---|---|---|---|
| `BayesianNeuralNetwork` | `bnn` | `Available` | `epistemic_uq`, `ood` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.BayesianNeuralNetwork` |
| `HeteroscedasticBNN` | `bnn` | `Available` | `uq_decomposition`, `ood`, `calibration` | `epistemic`, `aleatoric`, `decomposition`, `calibration*`, `ood*` | `torchregress.ensemble.HeteroscedasticBNN` |
| `HeteroscedasticLaplaceRegressor` | `bnn` | `Available` | `laplace_approximation`, `epistemic_uq`, `aleatoric_uq`, `last_layer` | `epistemic`, `aleatoric`, `decomposition`, `calibration*`, `ood*` | `torchregress.algorithms.HeteroscedasticLaplaceRegressor` |
| `IVON` | `bnn` | `Available` | `variational_inference`, `optimizer`, `natural_gradient`, `bayesian_learning_rule` | `epistemic`, `calibration*`, `ood*` | `torchregress.algorithms.IVON` |
| `NaturalHeteroscedasticHead` | `bnn` | `Available` | `natural_parameterization`, `optimization_stability` | `aleatoric`, `calibration*`, `ood*` | `torchregress.algorithms.NaturalHeteroscedasticHead` |
| `NaturalReparamHead` | `bnn` | `Available` | `natural_parameterization`, `optimization_stability` | `aleatoric`, `calibration*`, `ood*` | `torchregress.algorithms.NaturalReparamHead` |
| `VIDSRegressor` | `bnn` | `Available` | `distribution_shift`, `variational_inference`, `adaptive_prior` | `epistemic`, `aleatoric`, `calibration*`, `ood*` | `torchregress.algorithms.VIDSRegressor` |
| `BinnedLabelShiftEstimator` | `calibration_transform` | `Strong` | `calibration`, `shift_adaptation`, `label_shift` | `calibration`, `ood*` | `torchregress.calibration.BinnedLabelShiftEstimator` |
| `IsotonicMeanCalibrator` | `calibration_transform` | `Available` | `calibration`, `posthoc_calibration` | `calibration`, `ood*` | `torchregress.calibration.IsotonicMeanCalibrator` |
| `PITCalibrator` | `calibration_transform` | `Available` | `calibration`, `posthoc_calibration`, `distribution_calibration` | `calibration`, `ood*` | `torchregress.calibration.PITCalibrator` |
| `SemiConformalCalibrator` | `calibration_transform` | `Strong` | `calibration`, `conformal`, `semi_supervised`, `shift_adaptation` | `calibration`, `ood*` | `torchregress.calibration.SemiConformalCalibrator` |
| `VarianceTemperatureScaler` | `calibration_transform` | `Available` | `calibration`, `posthoc_calibration` | `calibration`, `ood*` | `torchregress.calibration.VarianceTemperatureScaler` |
| `dr_ate` | `causal` | `Available` | `causal_inference`, `ate`, `selection_bias` | `calibration*`, `ood*` | `torchregress.causal.dr_ate` |
| `dr_cate` | `causal` | `Available` | `causal_inference`, `cate`, `selection_bias` | `calibration*`, `ood*` | `torchregress.causal.dr_cate` |
| `AFTLoss` | `censored` | `Available` | `censored`, `interval_censored`, `survival_regression` | `calibration*`, `ood*` | `torchregress.losses.AFTLoss` |
| `CensoredGaussianNLLLoss` | `censored` | `Available` | `censored`, `interval_censored`, `survival_regression` | `calibration*`, `ood*` | `torchregress.losses.CensoredGaussianNLLLoss` |
| `CensoredQuantileLoss` | `censored` | `Available` | `censored`, `interval_censored`, `non_gaussian` | `calibration*`, `ood*` | `torchregress.losses.CensoredQuantileLoss` |
| `CQR` | `conformal` | `Strong` | `coverage_guarantees`, `calibration`, `heteroscedastic_noise` | `calibration`, `ood*` | `torchregress.losses.CQR` |
| `CTI` | `conformal` | `Available` | `coverage_guarantees`, `calibration`, `distributional` | `calibration`, `ood*` | `torchregress.losses.CTI` |
| `CVPlus` | `conformal` | `Available` | `coverage_guarantees`, `calibration`, `ensemble` | `calibration`, `ood*` | `torchregress.losses.CVPlus` |
| `ConformalLoss` | `conformal` | `Core` | `coverage_guarantees`, `calibration` | `calibration`, `ood*` | `torchregress.losses.ConformalLoss` |
| `DensityConformal` | `conformal` | `Available` | `coverage_guarantees`, `density_conformal`, `imbalance` | `calibration`, `ood*` | `torchregress.losses.DensityConformal` |
| `EnsembleBatchCP` | `conformal` | `Available` | `coverage_guarantees`, `calibration`, `ensemble` | `calibration`, `ood*` | `torchregress.losses.EnsembleBatchCP` |
| `JackknifePlus` | `conformal` | `Available` | `coverage_guarantees`, `calibration`, `ensemble` | `calibration`, `ood*` | `torchregress.losses.JackknifePlus` |
| `MonteCarloConformal` | `conformal` | `Available` | `coverage_guarantees`, `density_conformal`, `epistemic_uq` | `epistemic*`, `calibration`, `ood*` | `torchregress.losses.MonteCarloConformal` |
| `MultiDimensionalConformalLoss` | `conformal` | `Available` | `coverage_guarantees`, `calibration`, `multi_target` | `calibration`, `ood*` | `torchregress.losses.MultiDimensionalConformalLoss` |
| `PrevalenceAdjustedCP` | `conformal` | `Available` | `coverage_guarantees`, `density_conformal`, `imbalance` | `calibration`, `ood*` | `torchregress.losses.PrevalenceAdjustedCP` |
| `SLSConformal` | `conformal` | `Available` | `coverage_guarantees`, `multimodal`, `calibration` | `calibration`, `ood*` | `torchregress.losses.SLSConformal` |
| `SplitConformal` | `conformal` | `Core` | `coverage_guarantees`, `calibration`, `baseline` | `calibration`, `ood*` | `torchregress.losses.SplitConformal` |
| `UACQR` | `conformal` | `Available` | `coverage_guarantees`, `calibration`, `heteroscedastic_noise` | `calibration`, `ood*` | `torchregress.losses.UACQR` |
| `BoundedHead` | `constraints` | `Available` | `constraints`, `bounded_outputs` | `calibration*`, `ood*` | `torchregress.constraints.BoundedHead` |
| `NonCrossingSort` | `constraints` | `Available` | `constraints`, `ordered_outputs` | `calibration*`, `ood*` | `torchregress.constraints.NonCrossingSort` |
| `ErrorAwareFeatureEncoder` | `eiv` | `Available` | `noisy_features`, `measurement_error`, `feature_encoding` | `calibration*`, `ood*` | `torchregress.algorithms.ErrorAwareFeatureEncoder` |
| `FunctionalEIVLoss` | `eiv` | `Available` | `noisy_features`, `measurement_error` | `calibration*`, `ood*` | `torchregress.losses.FunctionalEIVLoss` |
| `InputNoiseAugmentationLoss` | `eiv` | `Strong` | `noisy_features`, `measurement_error`, `augmentation` | `calibration*`, `ood*` | `torchregress.losses.InputNoiseAugmentationLoss` |
| `InputNoiseBinnedPDFLoss` | `eiv` | `Available` | `noisy_features`, `measurement_error`, `multimodal` | `calibration*`, `ood*` | `torchregress.losses.InputNoiseBinnedPDFLoss` |
| `InputNoiseMDNLoss` | `eiv` | `Available` | `noisy_features`, `measurement_error`, `multimodal` | `calibration*`, `ood*` | `torchregress.losses.InputNoiseMDNLoss` |
| `InputNoiseMarginalizationLoss` | `eiv` | `Deprecated` | `noisy_features`, `measurement_error`, `marginalization` | `calibration*`, `ood*` | `torchregress.losses.InputNoiseMarginalizationLoss` |
| `LatentMarginalizationLoss` | `eiv` | `Strong` | `noisy_features`, `measurement_error`, `marginalization` | `calibration*`, `ood*` | `torchregress.losses.LatentMarginalizationLoss` |
| `LatentNN` | `eiv` | `Available` | `noisy_features`, `measurement_error`, `latent_input` | `calibration*`, `ood*` | `torchregress.algorithms.LatentNN` |
| `NoiseAwareRegressor` | `eiv` | `Available` | `noisy_features`, `measurement_error`, `feature_encoding` | `calibration*`, `ood*` | `torchregress.algorithms.NoiseAwareRegressor` |
| `NoisyInputPredictor` | `eiv` | `Strong` | `noisy_features`, `measurement_error`, `inference` | `calibration*`, `ood*` | `torchregress.losses.NoisyInputPredictor` |
| `OrthogonalDistanceRegressionLoss` | `eiv` | `Available` | `noisy_features`, `measurement_error` | `calibration*`, `ood*` | `torchregress.losses.OrthogonalDistanceRegressionLoss` |
| `PredictionSIMEX` | `eiv` | `Strong` | `noisy_features`, `measurement_error`, `simex`, `inference` | `calibration*`, `ood*` | `torchregress.algorithms.PredictionSIMEX` |
| `RegressionCalibration` | `eiv` | `Strong` | `noisy_features`, `measurement_error`, `calibration` | `calibration`, `ood*` | `torchregress.algorithms.RegressionCalibration` |
| `SIMEX` | `eiv` | `Strong` | `noisy_features`, `measurement_error`, `simex` | `calibration*`, `ood*` | `torchregress.algorithms.SIMEX` |
| `StructuralEIVLoss` | `eiv` | `Available` | `noisy_features`, `measurement_error` | `calibration*`, `ood*` | `torchregress.losses.StructuralEIVLoss` |
| `BatchEnsembleRegressor` | `ensemble` | `Available` | `uq_decomposition`, `epistemic_uq`, `aleatoric_uq`, `ood`, `low_compute` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.BatchEnsembleRegressor` |
| `BinnedPDFEnsembleModel` | `ensemble` | `Available` | `multimodal_targets`, `non_gaussian`, `calibration` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.BinnedPDFEnsembleModel` |
| `CumulativeLinkEnsembleModel` | `ensemble` | `Available` | `ordinal`, `non_gaussian`, `calibration` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.CumulativeLinkEnsembleModel` |
| `DeepEnsemble` | `ensemble` | `Core` | `epistemic_uq`, `ood`, `selective_prediction` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood` | `torchregress.ensemble.DeepEnsemble` |
| `HeteroscedasticBatchEnsembleModel` | `ensemble` | `Strong` | `uq_decomposition`, `epistemic_uq`, `aleatoric_uq`, `ood`, `low_compute` | `epistemic`, `aleatoric`, `decomposition`, `calibration*`, `ood*` | `torchregress.ensemble.HeteroscedasticBatchEnsembleModel` |
| `HeteroscedasticEnsembleModel` | `ensemble` | `Strong` | `uq_decomposition`, `ood`, `calibration` | `epistemic`, `aleatoric`, `decomposition`, `calibration*`, `ood` | `torchregress.ensemble.HeteroscedasticEnsembleModel` |
| `MDNEnsembleModel` | `ensemble` | `Available` | `multimodal_targets`, `non_gaussian`, `calibration`, `uq_decomposition` | `epistemic`, `aleatoric`, `decomposition`, `calibration*`, `ood*` | `torchregress.ensemble.MDNEnsembleModel` |
| `EvidentialRegressionLoss` | `evidential` | `Available` | `uq_decomposition`, `single_pass`, `evidential` | `epistemic`, `aleatoric`, `decomposition`, `calibration*`, `ood*` | `torchregress.losses.EvidentialRegressionLoss` |
| `AsymmetricLeastSquaresLoss` | `expectile` | `Available` | `prediction_intervals`, `non_gaussian`, `expectile` | `calibration*`, `ood*` | `torchregress.losses.AsymmetricLeastSquaresLoss` |
| `ExpectileLoss` | `expectile` | `Available` | `prediction_intervals`, `non_gaussian`, `expectile` | `calibration*`, `ood*` | `torchregress.losses.ExpectileLoss` |
| `MultiExpectileLoss` | `expectile` | `Available` | `prediction_intervals`, `non_gaussian`, `multi_expectile` | `calibration*`, `ood*` | `torchregress.losses.MultiExpectileLoss` |
| `ContrastiveFlowLoss` | `flow` | `Available` | `parameter_estimation`, `domain_shift`, `simulation_based_inference` | `aleatoric`, `decomposition*`, `calibration*`, `ood*` | `torchregress.losses.ContrastiveFlowLoss` |
| `NormalizingFlowLoss` | `flow` | `Available` | `multimodal`, `non_gaussian`, `multi_target` | `aleatoric`, `decomposition*`, `calibration*`, `ood*` | `torchregress.losses.NormalizingFlowLoss` |
| `SLSLoss` | `flow` | `Available` | `multimodal`, `non_gaussian`, `prediction_intervals` | `calibration*`, `ood*` | `torchregress.losses.SLSLoss` |
| `BMCLoss` | `imbalanced_loss` | `Available` | `imbalance`, `rare_targets`, `long_tail` | `calibration*`, `ood*` | `torchregress.losses.BMCLoss` |
| `BalancedMSELoss` | `imbalanced_loss` | `Available` | `imbalance`, `rare_targets`, `long_tail` | `calibration*`, `ood*` | `torchregress.losses.BalancedMSELoss` |
| `DensityWeightedLoss` | `imbalanced_loss` | `Available` | `imbalance`, `rare_targets` | `calibration*`, `ood*` | `torchregress.losses.DensityWeightedLoss` |
| `FeatureDistributionSmoother` | `imbalanced_loss` | `Available` | `imbalance`, `rare_targets` | `calibration*`, `ood*` | `torchregress.losses.FeatureDistributionSmoother` |
| `FocalRLoss` | `imbalanced_loss` | `Available` | `imbalance`, `rare_targets`, `focal` | `calibration*`, `ood*` | `torchregress.losses.FocalRLoss` |
| `LDSLoss` | `imbalanced_loss` | `Available` | `imbalance`, `rare_targets` | `calibration*`, `ood*` | `torchregress.losses.LDSLoss` |
| `PropensityWeightedLoss` | `imbalanced_loss` | `Available` | `imbalance`, `selection_bias`, `missing_labels` | `calibration*`, `ood*` | `torchregress.losses.PropensityWeightedLoss` |
| `PredictionPoweredInference` | `inference` | `Available` | `inference`, `limited_labels`, `population_estimation` | `calibration*`, `ood*` | `torchregress.inference.ppi_mean_ci` |
| `MCDropoutWrapper` | `mc_dropout` | `Strong` | `epistemic_uq`, `low_compute` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.MCDropoutWrapper` |
| `MDNLoss` | `mdn` | `Available` | `multimodal`, `non_gaussian`, `multi_target` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.MDNLoss` |
| `TaskAgnosticCorrelations` | `multivariate` | `Available` | `covariance_evaluation`, `conditioning` | `calibration*`, `ood*` | `torchregress.metrics.TaskAgnosticCorrelations` |
| `CORALLoss` | `ordinal` | `Available` | `ordinal`, `ordered_targets`, `calibration` | `calibration*`, `ood*` | `torchregress.losses.CORALLoss` |
| `CumulativeLinkLoss` | `ordinal` | `Available` | `ordinal`, `ordered_targets`, `calibration` | `calibration*`, `ood*` | `torchregress.losses.CumulativeLinkLoss` |
| `OrdinalCrossEntropyLoss` | `ordinal` | `Available` | `ordinal`, `ordered_targets` | `calibration*`, `ood*` | `torchregress.losses.OrdinalCrossEntropyLoss` |
| `WeightedMSELoss` | `point_loss` | `Core` | `baseline` | `calibration*`, `ood*` | `torchregress.losses.WeightedMSELoss` |
| `NegativeBinomialNLLLoss` | `poisson` | `Available` | `count_data`, `non_gaussian`, `overdispersion` | `calibration*`, `ood*` | `torchregress.losses.NegativeBinomialNLLLoss` |
| `PoissonDevianceLoss` | `poisson` | `Available` | `count_data`, `non_gaussian`, `poisson` | `calibration*`, `ood*` | `torchregress.losses.PoissonDevianceLoss` |
| `PoissonLikelihoodRatioLoss` | `poisson` | `Available` | `count_data`, `non_gaussian`, `poisson`, `ablation` | `calibration*`, `ood*` | `torchregress.losses.PoissonLikelihoodRatioLoss` |
| `ZeroInflatedPoissonNLLLoss` | `poisson` | `Available` | `count_data`, `non_gaussian`, `zero_inflated` | `calibration*`, `ood*` | `torchregress.losses.ZeroInflatedPoissonNLLLoss` |
| `EnhancedPoissonGaussianMixtureLoss` | `poisson_gaussian` | `Available` | `count_data`, `non_gaussian`, `mixture`, `noisy_counts` | `calibration*`, `ood*` | `torchregress.losses.EnhancedPoissonGaussianMixtureLoss` |
| `PoissonGaussianLikelihoodRatioLoss` | `poisson_gaussian` | `Available` | `count_data`, `non_gaussian`, `mixture`, `ablation` | `calibration*`, `ood*` | `torchregress.losses.PoissonGaussianLikelihoodRatioLoss` |
| `PoissonGaussianMixtureLoss` | `poisson_gaussian` | `Available` | `count_data`, `non_gaussian`, `mixture`, `noisy_counts` | `calibration*`, `ood*` | `torchregress.losses.PoissonGaussianMixtureLoss` |
| `BetaNLLLoss` | `probabilistic_loss` | `Available` | `heteroscedastic_noise`, `aleatoric_uq`, `optimization_stability` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.BetaNLLLoss` |
| `FaithfulGaussianLoss` | `probabilistic_loss` | `Available` | `heteroscedastic_noise`, `aleatoric_uq`, `mean_variance_decoupling` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.FaithfulGaussianLoss` |
| `GaussianCRPSLoss` | `probabilistic_loss` | `Strong` | `heteroscedastic_noise`, `aleatoric_uq`, `calibration` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.GaussianCRPSLoss` |
| `GaussianNLLLoss` | `probabilistic_loss` | `Core` | `heteroscedastic_noise`, `aleatoric_uq` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.GaussianNLLLoss` |
| `GaussianWassersteinBoundLoss` | `probabilistic_loss` | `Available` | `heteroscedastic_noise`, `multivariate_targets`, `mean_covariance_supervision` | `aleatoric*`, `calibration*`, `ood*` | `torchregress.losses.GaussianWassersteinBoundLoss` |
| `LowRankGaussianLoss` | `probabilistic_loss` | `Strong` | `multi_target`, `scalability` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.LowRankGaussianLoss` |
| `MultivariateGaussianLoss` | `probabilistic_loss` | `Strong` | `multi_target`, `correlated_targets` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.MultivariateGaussianLoss` |
| `NeighborhoodCovariancePseudoLabeler` | `probabilistic_loss` | `Available` | `pseudo_labels`, `heteroscedastic_noise`, `self_supervision` | `aleatoric*`, `calibration*`, `ood*` | `torchregress.algorithms.NeighborhoodCovariancePseudoLabeler` |
| `MultiQuantileLoss` | `quantile` | `Strong` | `prediction_intervals`, `non_gaussian`, `multi_quantile` | `calibration`, `ood*` | `torchregress.losses.MultiQuantileLoss` |
| `QuantileLoss` | `quantile` | `Core` | `prediction_intervals`, `non_gaussian` | `calibration`, `ood*` | `torchregress.losses.QuantileLoss` |
| `AdaptiveRobustLoss` | `robust_loss` | `Available` | `outliers`, `adaptive_shape`, `learnable` | `calibration*`, `ood*` | `torchregress.losses.AdaptiveRobustLoss` |
| `BarronLoss` | `robust_loss` | `Available` | `outliers`, `adaptive_shape` | `calibration*`, `ood*` | `torchregress.losses.BarronLoss` |
| `CVaRLoss` | `robust_loss` | `Available` | `outliers`, `tail_focused`, `worst_case`, `noisy_labels` | `calibration*`, `ood*` | `torchregress.losses.CVaRLoss` |
| `CauchyLoss` | `robust_loss` | `Available` | `outliers`, `redescending`, `noisy_labels` | `calibration*`, `ood*` | `torchregress.losses.CauchyLoss` |
| `CharbonnierLoss` | `robust_loss` | `Strong` | `outliers`, `noisy_labels`, `smooth_approx` | `calibration*`, `ood*` | `torchregress.losses.CharbonnierLoss` |
| `LogCoshLoss` | `robust_loss` | `Strong` | `outliers`, `noisy_labels`, `smooth_approx` | `calibration*`, `ood*` | `torchregress.losses.LogCoshLoss` |
| `PseudoHuberLoss` | `robust_loss` | `Strong` | `outliers`, `noisy_labels`, `smooth_approx` | `calibration*`, `ood*` | `torchregress.losses.PseudoHuberLoss` |
| `TukeyBiweightLoss` | `robust_loss` | `Available` | `outliers`, `redescending`, `noisy_labels` | `calibration*`, `ood*` | `torchregress.losses.TukeyBiweightLoss` |
| `WeightedHuberLoss` | `robust_loss` | `Core` | `outliers`, `noisy_labels` | `calibration*`, `ood*` | `torchregress.losses.WeightedHuberLoss` |
| `TeacherStudentTrainer` | `semi_supervised` | `Strong` | `semi_supervised`, `weak_supervision`, `pseudo_labeling` | `calibration*`, `ood*` | `torchregress.semi_supervised.TeacherStudentTrainer` |
| `MultiSWAG` | `swag` | `Available` | `epistemic_uq`, `ood` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.MultiSWAG` |
| `SWAG` | `swag` | `Available` | `epistemic_uq`, `ood` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.SWAG` |
| `BoxCoxTransformLoss` | `target_transform` | `Available` | `target_transform`, `skewed_targets` | `calibration*`, `ood*` | `torchregress.losses.BoxCoxTransformLoss` |
| `LogTransformLoss` | `target_transform` | `Available` | `target_transform`, `skewed_targets` | `calibration*`, `ood*` | `torchregress.losses.LogTransformLoss` |
| `SqrtTransformLoss` | `target_transform` | `Available` | `target_transform`, `skewed_targets` | `calibration*`, `ood*` | `torchregress.losses.SqrtTransformLoss` |
| `TransformedTargetLoss` | `target_transform` | `Available` | `target_transform`, `skewed_targets` | `calibration*`, `ood*` | `torchregress.losses.TransformedTargetLoss` |
| `YeoJohnsonTransformLoss` | `target_transform` | `Available` | `target_transform`, `skewed_targets` | `calibration*`, `ood*` | `torchregress.losses.YeoJohnsonTransformLoss` |
| `BayesianLinearHead` | `test_time` | `Available` | `low_shot`, `test_time_adaptation`, `last_layer`, `linear_head` | `epistemic*`, `aleatoric*`, `decomposition*`, `calibration*` | `torchregress.test_time.BayesianLinearHead` |
| `OTShiftReweighter` | `test_time` | `Available` | `distribution_shift`, `conformal`, `calibration` | `calibration*`, `ood*` | `torchregress.test_time.OTShiftReweighter` |
| `OptimalTransportCoverageGap` | `test_time` | `Available` | `distribution_shift`, `conformal`, `calibration` | `calibration*`, `ood*` | `torchregress.test_time.OptimalTransportCoverageGap` |
| `RecursiveBayesianHead` | `test_time` | `Available` | `low_shot`, `test_time_adaptation`, `last_layer`, `streaming` | `epistemic*`, `aleatoric*`, `decomposition*`, `calibration*` | `torchregress.test_time.RecursiveBayesianHead` |
| `WeightedSplitConformalAdapter` | `test_time` | `Available` | `distribution_shift`, `conformal`, `calibration` | `calibration*`, `ood*` | `torchregress.test_time.WeightedSplitConformalAdapter` |
| `weighted_split_classification_predictive_batch` | `test_time` | `Available` | `distribution_shift`, `conformal`, `calibration` | `calibration*`, `ood*` | `torchregress.test_time.weighted_split_classification_predictive_batch` |
| `TaylorInducedCovarianceHead` | `tictac` | `Available` | `heteroscedastic_noise`, `multivariate_targets`, `gradient_curvature_parameterization` | `aleatoric`, `calibration*`, `ood*` | `torchregress.algorithms.TaylorInducedCovarianceHead` |
| `CompoundPoissonLoss` | `tweedie` | `Available` | `tweedie`, `non_gaussian`, `zero_inflated`, `positive_targets` | `calibration*`, `ood*` | `torchregress.losses.CompoundPoissonLoss` |
| `GammaLoss` | `tweedie` | `Available` | `tweedie`, `non_gaussian`, `positive_targets`, `right_skewed` | `calibration*`, `ood*` | `torchregress.losses.GammaLoss` |
| `InverseGaussianLoss` | `tweedie` | `Available` | `tweedie`, `non_gaussian`, `positive_targets` | `calibration*`, `ood*` | `torchregress.losses.InverseGaussianLoss` |
| `TweedieLoss` | `tweedie` | `Available` | `tweedie`, `non_gaussian`, `positive_targets`, `compound_poisson` | `calibration*`, `ood*` | `torchregress.losses.TweedieLoss` |
| `ConsistencyRegLoss` | `uncertain_gt` | `Available` | `uncertain_ground_truth`, `weak_supervision`, `noisy_labels` | `calibration*`, `ood*` | `torchregress.losses.ConsistencyRegLoss` |
| `NoisyTargetGaussianNLL` | `uncertain_gt` | `Available` | `uncertain_ground_truth`, `noisy_labels`, `weak_supervision` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.NoisyTargetGaussianNLL` |
| `PseudoLabelConsistencyLoss` | `uncertain_gt` | `Available` | `uncertain_ground_truth`, `weak_supervision`, `semi_supervised` | `calibration*`, `ood*` | `torchregress.losses.PseudoLabelConsistencyLoss` |
| `PseudoLabelNLL` | `uncertain_gt` | `Available` | `uncertain_ground_truth`, `weak_supervision`, `noisy_labels` | `calibration*`, `ood*` | `torchregress.losses.PseudoLabelNLL` |

## Capability-Focused Shortlists

### Multimodal (`multimodal=yes`)

`SLSConformal`, `InputNoiseBinnedPDFLoss`, `InputNoiseMDNLoss`, `ContrastiveFlowLoss`, `NormalizingFlowLoss`, `SLSLoss`, `MDNLoss`, `EnhancedPoissonGaussianMixtureLoss`, `PoissonGaussianMixtureLoss`

### Noisy Features / EIV (`noisy_features_eiv=yes`)

`ErrorAwareFeatureEncoder`, `FunctionalEIVLoss`, `InputNoiseAugmentationLoss`, `InputNoiseBinnedPDFLoss`, `InputNoiseMDNLoss`, `InputNoiseMarginalizationLoss`, `LatentMarginalizationLoss`, `LatentNN`, `NoiseAwareRegressor`, `NoisyInputPredictor`, `OrthogonalDistanceRegressionLoss`, `PredictionSIMEX`, `RegressionCalibration`, `SIMEX`, `StructuralEIVLoss`

### Decomposition (`decomposition=yes`)

`HeteroscedasticBNN`, `HeteroscedasticLaplaceRegressor`, `HeteroscedasticBatchEnsembleModel`, `HeteroscedasticEnsembleModel`, `MDNEnsembleModel`, `EvidentialRegressionLoss`
