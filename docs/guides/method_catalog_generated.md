# Generated Method Catalog Snapshot

_Generated provenance_: `tools/render_method_catalog.py:render_markdown`
_Source artifacts_: `reports/method_catalog_latest.json`
_Generated date_: `2026-02-26`

It is a code-backed snapshot used to reduce docs drift in the task-first matrix.

## Summary

- Total methods: `22`
- Peer methods present: `SWAG`, `BayesianNeuralNetwork`, `MDNLoss`

### By Family

| Family | Count |
|---|---:|
| `bnn` | 2 |
| `conformal` | 1 |
| `eiv` | 3 |
| `ensemble` | 2 |
| `flow` | 1 |
| `imbalanced_loss` | 2 |
| `inference` | 1 |
| `mc_dropout` | 1 |
| `mdn` | 1 |
| `point_loss` | 1 |
| `probabilistic_loss` | 3 |
| `quantile` | 1 |
| `robust_loss` | 1 |
| `swag` | 2 |

### By Maturity

| Maturity | Count |
|---|---:|
| `Core` | 6 |
| `Strong` | 5 |
| `Available` | 11 |

## Method Rows

Legend: `*` suffix means partial support. This snapshot is for discovery and comparison, not absolute ranking.

| Name | Family | Maturity | Task Tags | Core Capability Signals | Public Path |
|---|---|---|---|---|---|
| `BayesianNeuralNetwork` | `bnn` | `Available` | `epistemic_uq`, `ood` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.BayesianNeuralNetwork` |
| `HeteroscedasticBNN` | `bnn` | `Available` | `uq_decomposition`, `ood`, `calibration` | `epistemic`, `aleatoric`, `decomposition`, `calibration*`, `ood*` | `torchregress.ensemble.HeteroscedasticBNN` |
| `ConformalLoss` | `conformal` | `Core` | `coverage_guarantees`, `calibration` | `calibration`, `ood*` | `torchregress.losses.ConformalLoss` |
| `FunctionalEIVLoss` | `eiv` | `Available` | `noisy_features`, `measurement_error` | `calibration*`, `ood*` | `torchregress.losses.FunctionalEIVLoss` |
| `OrthogonalDistanceRegressionLoss` | `eiv` | `Available` | `noisy_features`, `measurement_error` | `calibration*`, `ood*` | `torchregress.losses.OrthogonalDistanceRegressionLoss` |
| `StructuralEIVLoss` | `eiv` | `Available` | `noisy_features`, `measurement_error` | `calibration*`, `ood*` | `torchregress.losses.StructuralEIVLoss` |
| `DeepEnsemble` | `ensemble` | `Core` | `epistemic_uq`, `ood`, `selective_prediction` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood` | `torchregress.ensemble.DeepEnsemble` |
| `HeteroscedasticEnsembleModel` | `ensemble` | `Strong` | `uq_decomposition`, `ood`, `calibration` | `epistemic`, `aleatoric`, `decomposition`, `calibration*`, `ood` | `torchregress.ensemble.HeteroscedasticEnsembleModel` |
| `NormalizingFlowLoss` | `flow` | `Available` | `multimodal`, `non_gaussian`, `multi_target` | `aleatoric`, `decomposition*`, `calibration*`, `ood*` | `torchregress.losses.NormalizingFlowLoss` |
| `DensityWeightedLoss` | `imbalanced_loss` | `Strong` | `imbalance`, `rare_targets` | `calibration*`, `ood*` | `torchregress.losses.DensityWeightedLoss` |
| `LDSLoss` | `imbalanced_loss` | `Available` | `imbalance`, `rare_targets` | `calibration*`, `ood*` | `torchregress.losses.LDSLoss` |
| `PredictionPoweredInference` | `inference` | `Available` | `inference`, `limited_labels`, `population_estimation` | `calibration*`, `ood*` | `torchregress.inference.ppi_mean_ci` |
| `MCDropoutWrapper` | `mc_dropout` | `Strong` | `epistemic_uq`, `low_compute` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.MCDropoutWrapper` |
| `MDNLoss` | `mdn` | `Available` | `multimodal`, `non_gaussian`, `multi_target` | `aleatoric`, `decomposition`, `calibration*`, `ood*` | `torchregress.losses.MDNLoss` |
| `WeightedMSELoss` | `point_loss` | `Core` | `baseline` | `calibration*`, `ood*` | `torchregress.losses.WeightedMSELoss` |
| `GaussianNLLLoss` | `probabilistic_loss` | `Core` | `heteroscedastic_noise`, `aleatoric_uq` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.GaussianNLLLoss` |
| `LowRankGaussianLoss` | `probabilistic_loss` | `Strong` | `multi_target`, `scalability` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.LowRankGaussianLoss` |
| `MultivariateGaussianLoss` | `probabilistic_loss` | `Strong` | `multi_target`, `correlated_targets` | `aleatoric`, `calibration*`, `ood*` | `torchregress.losses.MultivariateGaussianLoss` |
| `QuantileLoss` | `quantile` | `Core` | `prediction_intervals`, `non_gaussian` | `calibration`, `ood*` | `torchregress.losses.QuantileLoss` |
| `HuberLoss` | `robust_loss` | `Core` | `outliers`, `noisy_labels` | `calibration*`, `ood*` | `torchregress.losses.HuberLoss` |
| `MultiSWAG` | `swag` | `Available` | `epistemic_uq`, `ood` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.MultiSWAG` |
| `SWAG` | `swag` | `Available` | `epistemic_uq`, `ood` | `epistemic`, `aleatoric*`, `decomposition*`, `calibration*`, `ood*` | `torchregress.ensemble.SWAG` |

## Capability-Focused Shortlists

### Multimodal (`multimodal=yes`)

`NormalizingFlowLoss`, `MDNLoss`

### Noisy Features / EIV (`noisy_features_eiv=yes`)

`FunctionalEIVLoss`, `OrthogonalDistanceRegressionLoss`, `StructuralEIVLoss`

### Decomposition (`decomposition=yes`)

`HeteroscedasticBNN`, `HeteroscedasticEnsembleModel`, `MDNLoss`

