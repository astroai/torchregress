# Generated Method Catalog Snapshot

_Generated provenance_: `tools/render_method_catalog.py:render_markdown`
_Source artifacts_: `reports/method_catalog_latest.json`
_Generated date_: `2026-04-16`

It is a code-backed snapshot used to reduce docs drift in the task-first matrix.

## Summary

- Total methods: `70`
- Peer methods present: `SWAG`, `BayesianNeuralNetwork`, `MDNLoss`

### By Family

| Family | Count |
|---|---:|
| `bnn` | 2 |
| `calibration_transform` | 3 |
| `causal` | 2 |
| `censored` | 3 |
| `conformal` | 5 |
| `constraints` | 2 |
| `eiv` | 7 |
| `ensemble` | 7 |
| `flow` | 2 |
| `imbalanced_loss` | 5 |
| `inference` | 1 |
| `mc_dropout` | 1 |
| `mdn` | 1 |
| `ordinal` | 3 |
| `point_loss` | 1 |
| `probabilistic_loss` | 7 |
| `quantile` | 1 |
| `robust_loss` | 1 |
| `swag` | 2 |
| `target_transform` | 4 |
| `test_time` | 6 |
| `uncertain_gt` | 4 |

### By Maturity

| Maturity | Count |
|---|---:|
| `Core` | 6 |
| `Strong` | 7 |
| `Available` | 57 |

## Method Rows

Legend: `*` suffix means partial support. This snapshot is for discovery and comparison, not absolute ranking.

| Name | Family | Maturity | Task Tags | Core Capability Signals | Public Path |
|---|---|---|---|---|---|
| `BayesianNeuralNetwork` | `bnn` | `Available` | `epistemic_uq`, `ood` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.BayesianNeuralNetwork` |
| `HeteroscedasticBNN` | `bnn` | `Available` | `uq_decomposition`, `ood`, `calibration` | `epistemic`, `aleatoric`, `decomposition`, `calibration*`, `ood*` | `torchregress.ensemble.HeteroscedasticBNN` |
| `IsotonicMeanCalibrator` | `calibration_transform` | `Available` | `calibration`, `posthoc_calibration` | `calibration`, `ood*` | `torchregress.calibration.IsotonicMeanCalibrator` |
| `PITCalibrator` | `calibration_transform` | `Available` | `calibration`, `posthoc_calibration`, `distribution_calibration` | `calibration`, `ood*` | `torchregress.calibration.PITCalibrator` |
| `VarianceTemperatureScaler` | `calibration_transform` | `Available` | `calibration`, `posthoc_calibration` | `calibration`, `ood*` | `torchregress.calibration.VarianceTemperatureScaler` |
| `dr_ate` | `causal` | `Available` | `causal_inference`, `ate`, `selection_bias` | `calibration*`, `ood*` | `torchregress.causal.dr_ate` |
| `dr_cate` | `causal` | `Available` | `causal_inference`, `cate`, `selection_bias` | `calibration*`, `ood*` | `torchregress.causal.dr_cate` |
| `AFTLoss` | `censored` | `Available` | `censored`, `interval_censored`, `survival_regression` | `calibration*`, `ood*` | `torchregress.losses.AFTLoss` |
| `CensoredGaussianNLLLoss` | `censored` | `Available` | `censored`, `interval_censored`, `survival_regression` | `calibration*`, `ood*` | `torchregress.losses.CensoredGaussianNLLLoss` |
| `CensoredQuantileLoss` | `censored` | `Available` | `censored`, `interval_censored`, `non_gaussian` | `calibration*`, `ood*` | `torchregress.losses.CensoredQuantileLoss` |
| `ConformalLoss` | `conformal` | `Core` | `coverage_guarantees`, `calibration` | `calibration`, `ood*` | `torchregress.losses.ConformalLoss` |
| `DensityConformal` | `conformal` | `Available` | `coverage_guarantees`, `density_conformal`, `imbalance` | `calibration`, `ood*` | `torchregress.losses.DensityConformal` |
| `MonteCarloConformal` | `conformal` | `Available` | `coverage_guarantees`, `density_conformal`, `epistemic_uq` | `epistemic*`, `calibration`, `ood*` | `torchregress.losses.MonteCarloConformal` |
| `PrevalenceAdjustedCP` | `conformal` | `Available` | `coverage_guarantees`, `density_conformal`, `imbalance` | `calibration`, `ood*` | `torchregress.losses.PrevalenceAdjustedCP` |
| `UACQR` | `conformal` | `Available` | `coverage_guarantees`, `calibration`, `heteroscedastic_noise` | `calibration`, `ood*` | `torchregress.losses.UACQR` |
| `BoundedHead` | `constraints` | `Available` | `constraints`, `bounded_outputs` | `calibration*`, `ood*` | `torchregress.constraints.BoundedHead` |
| `NonCrossingSort` | `constraints` | `Available` | `constraints`, `ordered_outputs` | `calibration*`, `ood*` | `torchregress.constraints.NonCrossingSort` |
| `FunctionalEIVLoss` | `eiv` | `Available` | `noisy_features`, `measurement_error` | `calibration*`, `ood*` | `torchregress.losses.FunctionalEIVLoss` |
| `InputNoiseBinnedPDFLoss` | `eiv` | `Available` | `noisy_features`, `measurement_error`, `multimodal` | `calibration*`, `ood*` | `torchregress.losses.InputNoiseBinnedPDFLoss` |
| `InputNoiseMDNLoss` | `eiv` | `Available` | `noisy_features`, `measurement_error`, `multimodal` | `calibration*`, `ood*` | `torchregress.losses.InputNoiseMDNLoss` |
| `InputNoiseMarginalizationLoss` | `eiv` | `Strong` | `noisy_features`, `measurement_error`, `marginalization` | `calibration*`, `ood*` | `torchregress.losses.InputNoiseMarginalizationLoss` |
| `NoisyInputPredictor` | `eiv` | `Strong` | `noisy_features`, `measurement_error`, `inference` | `calibration*`, `ood*` | `torchregress.losses.NoisyInputPredictor` |
| `OrthogonalDistanceRegressionLoss` | `eiv` | `Available` | `noisy_features`, `measurement_error` | `calibration*`, `ood*` | `torchregress.losses.OrthogonalDistanceRegressionLoss` |
| `StructuralEIVLoss` | `eiv` | `Available` | `noisy_features`, `measurement_error` | `calibration*`, `ood*` | `torchregress.losses.StructuralEIVLoss` |
| `BinnedPDFEnsembleModel` | `ensemble` | `Available` | `multimodal_targets`, `non_gaussian`, `calibration` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.BinnedPDFEnsembleModel` |
| `CumulativeLinkEnsembleModel` | `ensemble` | `Available` | `ordinal`, `non_gaussian`, `calibration` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.CumulativeLinkEnsembleModel` |
| `DeepEnsemble` | `ensemble` | `Core` | `epistemic_uq`, `ood`, `selective_prediction` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood` | `torchregress.ensemble.DeepEnsemble` |
| `HeteroscedasticBatchEnsembleModel` | `ensemble` | `Strong` | `uq_decomposition`, `epistemic_uq`, `aleatoric_uq`, `ood`, `low_compute` | `epistemic`, `aleatoric`, `decomposition`, `calibration*`, `ood*` | `torchregress.ensemble.HeteroscedasticBatchEnsembleModel` |
| `HeteroscedasticEnsembleModel` | `ensemble` | `Strong` | `uq_decomposition`, `ood`, `calibration` | `epistemic`, `aleatoric`, `decomposition`, `calibration*`, `ood` | `torchregress.ensemble.HeteroscedasticEnsembleModel` |
| `MDNEnsembleModel` | `ensemble` | `Available` | `multimodal_targets`, `non_gaussian`, `calibration`, `uq_decomposition` | `epistemic`, `aleatoric`, `decomposition`, `calibration*`, `ood*` | `torchregress.ensemble.MDNEnsembleModel` |
| `PackedEnsembleRegressor` | `ensemble` | `Available` | `uq_decomposition`, `epistemic_uq`, `aleatoric_uq`, `ood`, `low_compute` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.PackedEnsembleRegressor` |
| `ContrastiveFlowLoss` | `flow` | `Available` | `parameter_estimation`, `domain_shift`, `simulation_based_inference` | `aleatoric`, `decomposition*`, `calibration*`, `ood*` | `torchregress.losses.ContrastiveFlowLoss` |
| `NormalizingFlowLoss` | `flow` | `Available` | `multimodal`, `non_gaussian`, `multi_target` | `aleatoric`, `decomposition*`, `calibration*`, `ood*` | `torchregress.losses.NormalizingFlowLoss` |
| `BMCLoss` | `imbalanced_loss` | `Available` | `imbalance`, `rare_targets`, `long_tail` | `calibration*`, `ood*` | `torchregress.losses.BMCLoss` |
| `BalancedMSELoss` | `imbalanced_loss` | `Available` | `imbalance`, `rare_targets`, `long_tail` | `calibration*`, `ood*` | `torchregress.losses.BalancedMSELoss` |
| `DensityWeightedLoss` | `imbalanced_loss` | `Available` | `imbalance`, `rare_targets` | `calibration*`, `ood*` | `torchregress.losses.DensityWeightedLoss` |
| `LDSLoss` | `imbalanced_loss` | `Available` | `imbalance`, `rare_targets` | `calibration*`, `ood*` | `torchregress.losses.LDSLoss` |
| `PropensityWeightedLoss` | `imbalanced_loss` | `Available` | `imbalance`, `selection_bias`, `missing_labels` | `calibration*`, `ood*` | `torchregress.losses.PropensityWeightedLoss` |
| `PredictionPoweredInference` | `inference` | `Available` | `inference`, `limited_labels`, `population_estimation` | `calibration*`, `ood*` | `torchregress.inference.ppi_mean_ci` |
| `MCDropoutWrapper` | `mc_dropout` | `Strong` | `epistemic_uq`, `low_compute` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.MCDropoutWrapper` |
| `MDNLoss` | `mdn` | `Available` | `multimodal`, `non_gaussian`, `multi_target` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.MDNLoss` |
| `CORALLoss` | `ordinal` | `Available` | `ordinal`, `ordered_targets`, `calibration` | `calibration*`, `ood*` | `torchregress.losses.CORALLoss` |
| `CumulativeLinkLoss` | `ordinal` | `Available` | `ordinal`, `ordered_targets`, `calibration` | `calibration*`, `ood*` | `torchregress.losses.CumulativeLinkLoss` |
| `OrdinalCrossEntropyLoss` | `ordinal` | `Available` | `ordinal`, `ordered_targets` | `calibration*`, `ood*` | `torchregress.losses.OrdinalCrossEntropyLoss` |
| `WeightedMSELoss` | `point_loss` | `Core` | `baseline` | `calibration*`, `ood*` | `torchregress.losses.WeightedMSELoss` |
| `BetaNLLLoss` | `probabilistic_loss` | `Available` | `heteroscedastic_noise`, `aleatoric_uq`, `optimization_stability` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.BetaNLLLoss` |
| `FaithfulGaussianLoss` | `probabilistic_loss` | `Available` | `heteroscedastic_noise`, `aleatoric_uq`, `mean_variance_decoupling` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.FaithfulGaussianLoss` |
| `GaussianNLLLoss` | `probabilistic_loss` | `Core` | `heteroscedastic_noise`, `aleatoric_uq` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.GaussianNLLLoss` |
| `GaussianWassersteinBoundLoss` | `probabilistic_loss` | `Available` | `heteroscedastic_noise`, `multivariate_targets`, `mean_covariance_supervision` | `aleatoric*`, `calibration*`, `ood*` | `torchregress.losses.GaussianWassersteinBoundLoss` |
| `LowRankGaussianLoss` | `probabilistic_loss` | `Strong` | `multi_target`, `scalability` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.LowRankGaussianLoss` |
| `MultivariateGaussianLoss` | `probabilistic_loss` | `Strong` | `multi_target`, `correlated_targets` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.MultivariateGaussianLoss` |
| `NeighborhoodCovariancePseudoLabeler` | `probabilistic_loss` | `Available` | `pseudo_labels`, `heteroscedastic_noise`, `self_supervision` | `aleatoric*`, `calibration*`, `ood*` | `torchregress.algorithms.NeighborhoodCovariancePseudoLabeler` |
| `QuantileLoss` | `quantile` | `Core` | `prediction_intervals`, `non_gaussian` | `calibration`, `ood*` | `torchregress.losses.QuantileLoss` |
| `HuberLoss` | `robust_loss` | `Core` | `outliers`, `noisy_labels` | `calibration*`, `ood*` | `torchregress.losses.HuberLoss` |
| `MultiSWAG` | `swag` | `Available` | `epistemic_uq`, `ood` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.MultiSWAG` |
| `SWAG` | `swag` | `Available` | `epistemic_uq`, `ood` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.SWAG` |
| `BoxCoxTransformLoss` | `target_transform` | `Available` | `target_transform`, `skewed_targets` | `calibration*`, `ood*` | `torchregress.losses.BoxCoxTransformLoss` |
| `LogTransformLoss` | `target_transform` | `Available` | `target_transform`, `skewed_targets` | `calibration*`, `ood*` | `torchregress.losses.LogTransformLoss` |
| `SqrtTransformLoss` | `target_transform` | `Available` | `target_transform`, `skewed_targets` | `calibration*`, `ood*` | `torchregress.losses.SqrtTransformLoss` |
| `YeoJohnsonTransformLoss` | `target_transform` | `Available` | `target_transform`, `skewed_targets` | `calibration*`, `ood*` | `torchregress.losses.YeoJohnsonTransformLoss` |
| `BayesianLinearHead` | `test_time` | `Available` | `low_shot`, `test_time_adaptation`, `last_layer`, `linear_head` | `epistemic*`, `aleatoric*`, `decomposition*`, `calibration*` | `torchregress.test_time.BayesianLinearHead` |
| `OTShiftReweighter` | `test_time` | `Available` | `distribution_shift`, `conformal`, `calibration` | `calibration*`, `ood*` | `torchregress.test_time.OTShiftReweighter` |
| `OptimalTransportCoverageGap` | `test_time` | `Available` | `distribution_shift`, `conformal`, `calibration` | `calibration*`, `ood*` | `torchregress.test_time.OptimalTransportCoverageGap` |
| `RecursiveBayesianHead` | `test_time` | `Available` | `low_shot`, `test_time_adaptation`, `last_layer`, `streaming` | `epistemic*`, `aleatoric*`, `decomposition*`, `calibration*` | `torchregress.test_time.RecursiveBayesianHead` |
| `WeightedSplitConformalAdapter` | `test_time` | `Available` | `distribution_shift`, `conformal`, `calibration` | `calibration*`, `ood*` | `torchregress.test_time.WeightedSplitConformalAdapter` |
| `weighted_split_classification_predictive_batch` | `test_time` | `Available` | `distribution_shift`, `conformal`, `calibration` | `calibration*`, `ood*` | `torchregress.test_time.weighted_split_classification_predictive_batch` |
| `ConsistencyRegLoss` | `uncertain_gt` | `Available` | `uncertain_ground_truth`, `weak_supervision`, `noisy_labels` | `calibration*`, `ood*` | `torchregress.losses.ConsistencyRegLoss` |
| `NoisyTargetGaussianNLL` | `uncertain_gt` | `Available` | `uncertain_ground_truth`, `noisy_labels`, `weak_supervision` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.NoisyTargetGaussianNLL` |
| `PseudoLabelConsistencyLoss` | `uncertain_gt` | `Available` | `uncertain_ground_truth`, `weak_supervision`, `semi_supervised` | `calibration*`, `ood*` | `torchregress.losses.PseudoLabelConsistencyLoss` |
| `PseudoLabelNLL` | `uncertain_gt` | `Available` | `uncertain_ground_truth`, `weak_supervision`, `noisy_labels` | `calibration*`, `ood*` | `torchregress.losses.PseudoLabelNLL` |

## Capability-Focused Shortlists

### Multimodal (`multimodal=yes`)

`InputNoiseBinnedPDFLoss`, `InputNoiseMDNLoss`, `ContrastiveFlowLoss`, `NormalizingFlowLoss`, `MDNLoss`

### Noisy Features / EIV (`noisy_features_eiv=yes`)

`FunctionalEIVLoss`, `InputNoiseBinnedPDFLoss`, `InputNoiseMDNLoss`, `InputNoiseMarginalizationLoss`, `NoisyInputPredictor`, `OrthogonalDistanceRegressionLoss`, `StructuralEIVLoss`

### Decomposition (`decomposition=yes`)

`HeteroscedasticBNN`, `HeteroscedasticBatchEnsembleModel`, `HeteroscedasticEnsembleModel`, `MDNEnsembleModel`
