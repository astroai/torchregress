"""Export and signature contract tests for non-loss public modules."""

from __future__ import annotations

import importlib
import inspect
import re

import torchregress as tr

EXPECTED_EXPORTS = {
    "torchregress": [
        "losses",
        "metrics",
        "algorithms",
        "ensemble",
        "semi_supervised",
        "test_time",
        "method_catalog",
        "inference",
        "constraints",
        "calibration",
        "causal",
        "prediction",
        "viz",
        "utils",
        "BaseLoss",
        "RegressionLoss",
        "DistributionLoss",
        "iteratively_reweighted_least_squares",
        "__version__",
    ],
    "metrics": [
        "ExpectedCalibrationError",
        "MarginalCalibrationError",
        "bias",
        "expected_calibration_error",
        "marginal_calibration_error",
        "calibration_score",
        "calibration_metrics_report",
        "censoring_rate",
        "observed_mae",
        "concordance_index",
        "interval_overlap_rate",
        "ContinuousRankedProbabilityScore",
        "EnergyScore",
        "conditional_density_estimation_loss",
        "continuous_ranked_probability_score",
        "crps_from_samples",
        "energy_score",
        "gaussian_nll",
        "crps_gaussian",
        "highest_posterior_density_coverage",
        "highest_posterior_density_level",
        "kolmogorov_smirnov_uniform_statistic",
        "probability_integral_transform",
        "distribution_metrics_report",
        "RiskCoverageCurve",
        "RejectionPolicy",
        "risk_coverage_curve",
        "GaussianNLLEnsemble",
        "EnsembleIntervalMetrics",
        "gaussian_nll_ensemble",
        "ensemble_interval_bounds",
        "ensemble_interval_metrics",
        "ensemble_mean",
        "ensemble_std",
        "ensemble_statistics",
        "ensemble_variance_decomposition",
        "uncertainty_decomposition",
        "IntervalScore",
        "MeanPredictionIntervalWidth",
        "PredictionIntervalCoverageProbability",
        "interval_score",
        "prediction_interval_coverage",
        "prediction_interval_coverage_probability",
        "interval_metrics_report",
        "MultivariateMAE",
        "MultivariateRMSE",
        "EntropyScore",
        "KernelDensityScore",
        "MahalanobisDistance",
        "TypicalityScore",
        "entropy_score",
        "kernel_density_score",
        "mahalanobis_distance",
        "typicality_score",
        "ood_metrics_report",
        "ordinal_accuracy",
        "mean_absolute_class_error",
        "quadratic_weighted_kappa",
        "HuberMetric",
        "MedianAbsoluteError",
        "MedianAbsoluteDeviation",
        "NormalizedMedianAbsoluteDeviation",
        "NormalizedRMSE",
        "OutlierFraction",
        "TrimmedMeanSquaredError",
        "attenuation_factor",
        "mse",
        "mean_squared_error",
        "mae",
        "mean_absolute_error",
        "median_absolute_error",
        "huber_loss",
        "rmse",
        "r2_score",
        "trimmed_mean_squared_error",
        "median_absolute_deviation",
        "normalized_rmse",
        "tail_mae",
        "tail_rmse",
        "regression_metrics_report",
        "noisy_target_gaussian_nll",
        "consistency_error",
        "pseudo_label_acceptance_rate",
        "uncertain_gt_metrics_report",
        "MeanSquaredError",
        "MeanAbsoluteError",
        "R2Score",
    ],
    "ensemble": [
        "BaseEnsembleModel",
        "BatchEnsembleLinear",
        "HeteroscedasticEnsembleModel",
        "DeepEnsemble",
        "BatchEnsembleMLPBackbone",
        "BinnedPDFEnsembleModel",
        "CumulativeLinkEnsembleModel",
        "MDNEnsembleModel",
        "RandomPartitionEnsembleModel",
        "HeteroscedasticBatchEnsembleModel",
        "BayesianModelAveraging",
        "StackingEnsemble",
        "DynamicEnsembleWeighting",
        "SWAG",
        "MultiSWAG",
        "parse_heteroscedastic_output",
        "MCDropoutWrapper",
        "MCDropoutModel",
        "enable_dropout",
        "VariationalLinear",
        "BayesianNeuralNetwork",
        "HeteroscedasticBNN",
    ],
    "algorithms": [
        "iteratively_reweighted_least_squares",
        "IRLS",
        "RegressionCalibration",
        "SIMEX",
        "LatentNN",
        "ErrorAwareFeatureEncoder",
        "NoiseAwareRegressor",
    ],
    "causal": [
        "dr_ate",
        "dr_cate",
        "dr_policy_value",
        "causal_overlap_report",
    ],
    "inference": [
        "ppi_mean_ci",
        "ppi_quantile_ci",
        "ppi_ols_ci",
        "ppi_diagnostics",
    ],
    "constraints": [
        "NonNegativeHead",
        "BoundedHead",
        "SimplexHead",
        "NonCrossingSort",
        "SpectralNormWrapper",
    ],
    "calibration": [
        "VarianceTemperatureScaler",
        "IsotonicMeanCalibrator",
        "PITCalibrator",
    ],
    "test_time": [
        "AdaptationBatch",
        "FeatureStatNormalizer",
        "LabelShiftEstimate",
        "ParameterEMA",
        "PosteriorLabelShiftAdapter",
        "RepresentationShiftCalibrator",
        "SignificantSubspaceAligner",
        "ShiftFactoredPredictiveTransport",
        "ShiftFactoredTransportConfig",
        "ShiftFactoredTransportState",
        "SubspaceAlignmentState",
        "SupportsAdaptationParameters",
        "SupportsPredictiveBatch",
        "SupportsRepresentation",
        "apply_label_shift_correction",
        "confidence_scores",
        "correct_gaussian_predictions_for_label_shift",
        "entropy_scores",
        "estimate_target_prior_em",
        "flatten_adaptation_parameters",
        "gaussian_bin_edges_from_targets",
        "gaussian_bin_probabilities",
        "gaussian_moments_from_binned_probabilities",
        "local_consistency_weights",
        "pseudo_label_targets",
        "select_high_confidence",
    ],
    "prediction": [
        "PredictiveBatch",
        "bars_to_density_grid",
        "quantiles_to_density_grid",
        "samples_to_density_grid",
    ],
    "semi_supervised": [
        "SAGERegAgreement",
        "SAGERegLoss",
        "SAGERegOutput",
        "SelfAgreementTrainer",
        "build_consensus_predictive_batch",
        "disagreement_to_weight",
        "distributional_pseudo_loss",
        "predictive_agreement_score",
    ],
}


EXPECTED_SIGNATURES = {
    "algorithms.iteratively_reweighted_least_squares": (
        "(model: torch.nn.modules.module.Module, x: torch.Tensor, y_true: torch.Tensor, "
        "initial_precision: torch.Tensor | None = None, "
        "covariance_matrices: torch.Tensor | None = None, "
        "mask: torch.Tensor | None = None, base_loss: str = 'gaussian', max_iter: int = 10, "
        "tol: float = 0.0001, delta: float = 1.0, weight_fn: Union[str, Callable] = 'huber', "
        "weight_params: Optional[Dict[str, Any]] = None, variance_type: str = 'predicted', "
        "epsilon: float = 1.1920928955078125e-07, return_all_predictions: bool = False, "
        "batch_size: int = 1024) -> Union[Tuple[torch.Tensor, List[float], torch.Tensor], "
        "Tuple[torch.Tensor, List[float], torch.Tensor, List[torch.Tensor]]]"
    ),
    "algorithms.RegressionCalibration.fit": (
        "(self, X_observed: torch.Tensor) -> 'RegressionCalibration'"
    ),
    "algorithms.RegressionCalibration.transform": (
        "(self, X_observed: torch.Tensor) -> torch.Tensor"
    ),
    "algorithms.SIMEX.fit": "(self, X_train: torch.Tensor, y_train: torch.Tensor) -> 'SIMEX'",
    "algorithms.SIMEX.predict": "(self, X: torch.Tensor) -> torch.Tensor",
    "algorithms.LatentNN.fit": (
        "(self, X_observed: 'torch.Tensor', y_observed: 'torch.Tensor', *, "
        "X_val: 'torch.Tensor | None' = None, y_val: 'torch.Tensor | None' = None) -> \"'LatentNN'\""
    ),
    "algorithms.LatentNN.predict": "(self, X: 'torch.Tensor') -> 'torch.Tensor'",
    "algorithms.ErrorAwareFeatureEncoder.forward": (
        "(self, x: 'torch.Tensor', sigma_x: 'float | torch.Tensor') -> 'torch.Tensor'"
    ),
    "algorithms.NoiseAwareRegressor.forward": (
        "(self, x: 'torch.Tensor', sigma_x: 'float | torch.Tensor') -> 'torch.Tensor'"
    ),
    "metrics.mse": (
        "(y_pred: torch.Tensor | numpy.ndarray, y_true: torch.Tensor | numpy.ndarray, "
        "sample_weight: torch.Tensor | numpy.ndarray | None = None, "
        "reduction: str = 'mean', "
        "as_numpy: bool = False) -> torch.Tensor | float | numpy.ndarray"
    ),
    "metrics.rmse": (
        "(y_pred: torch.Tensor | numpy.ndarray, y_true: torch.Tensor | numpy.ndarray, "
        "sample_weight: torch.Tensor | numpy.ndarray | None = None, "
        "reduction: str = 'mean', "
        "as_numpy: bool = False) -> torch.Tensor | float | numpy.ndarray"
    ),
    "metrics.expected_calibration_error": (
        "(y_pred_quantiles: Dict[float, torch.Tensor | numpy.ndarray], "
        "y_true: torch.Tensor | numpy.ndarray, return_diagnostics: bool = False, "
        "as_numpy: bool = False) -> Dict[str, torch.Tensor | float | numpy.ndarray]"
    ),
    "metrics.calibration_score": (
        "(y_true: torch.Tensor | numpy.ndarray, "
        "pred_mean: torch.Tensor | numpy.ndarray, "
        "pred_std: torch.Tensor | numpy.ndarray, n_levels: int = 19, as_numpy: bool = False) "
        "-> Dict[str, torch.Tensor | float | numpy.ndarray]"
    ),
    "metrics.prediction_interval_coverage_probability": (
        "(lower_bound: torch.Tensor | numpy.ndarray, "
        "upper_bound: torch.Tensor | numpy.ndarray, "
        "y_true: torch.Tensor | numpy.ndarray, alpha: float = 0.1, "
        "return_diagnostics: bool = False) "
        "-> torch.Tensor | float | Dict[str, torch.Tensor]"
    ),
    "metrics.ensemble_variance_decomposition": (
        "(means: torch.Tensor | numpy.ndarray, "
        "variances: torch.Tensor | numpy.ndarray, "
        "dim: int = 0) -> Tuple[torch.Tensor, torch.Tensor]"
    ),
    "metrics.mahalanobis_distance": (
        "(x: torch.Tensor | numpy.ndarray, mean: torch.Tensor | numpy.ndarray, "
        "cov: torch.Tensor | numpy.ndarray, reduction: str = 'none') -> torch.Tensor"
    ),
    "ensemble.BaseEnsembleModel.predict": "(self, x: torch.Tensor) -> Dict[str, torch.Tensor]",
    "ensemble.HeteroscedasticEnsembleModel.predict": (
        "(self, x: torch.Tensor) -> Dict[str, torch.Tensor]"
    ),
    "ensemble.HeteroscedasticBatchEnsembleModel.predict": (
        "(self, x: torch.Tensor) -> Dict[str, torch.Tensor]"
    ),
    "ensemble.MCDropoutWrapper.predict_with_uncertainty": (
        "(self, x: torch.Tensor, n_samples: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]"
    ),
    "ensemble.SWAG.sample": "(self, scale: float = 1.0, diag_noise: bool = True) -> None",
    "ensemble.MultiSWAG.predict_with_uncertainty": (
        "(self, x: torch.Tensor, n_samples: int = 30, scale: float = 1.0) -> "
        "tuple[torch.Tensor, torch.Tensor, torch.Tensor]"
    ),
}


EXPECTED_PARAM_ORDERS = {
    "inference.ppi_mean_ci": [
        "y_labeled",
        "pred_labeled",
        "pred_unlabeled",
        "alpha",
        "method",
        "n_boot",
        "seed",
    ],
    "inference.ppi_quantile_ci": [
        "y_labeled",
        "pred_labeled",
        "pred_unlabeled",
        "q",
        "alpha",
        "method",
        "n_boot",
        "seed",
    ],
    "inference.ppi_ols_ci": [
        "x_labeled",
        "y_labeled",
        "x_unlabeled",
        "pred_labeled",
        "pred_unlabeled",
        "alpha",
        "add_intercept",
        "n_boot",
        "seed",
    ],
    "inference.ppi_diagnostics": ["y_labeled", "pred_labeled", "pred_unlabeled"],
    "causal.dr_ate": [
        "x",
        "t",
        "y",
        "outcome_model",
        "propensity_model",
        "folds",
        "alpha",
        "seed",
        "trim_threshold",
        "eps",
    ],
    "causal.dr_cate": [
        "x",
        "t",
        "y",
        "cate_model",
        "outcome_model",
        "propensity_model",
        "folds",
        "alpha",
        "seed",
        "trim_threshold",
        "eps",
    ],
    "calibration.VarianceTemperatureScaler.fit": [
        "self",
        "pred_mean",
        "pred_var",
        "target",
        "max_iter",
        "lr",
    ],
    "calibration.VarianceTemperatureScaler.transform": ["self", "pred_var"],
    "calibration.IsotonicMeanCalibrator.fit": ["self", "pred_mean", "target"],
    "calibration.IsotonicMeanCalibrator.transform": ["self", "pred_mean"],
    "calibration.PITCalibrator.fit": ["self", "pit_values"],
    "calibration.PITCalibrator.transform": ["self", "pit_values"],
    "calibration.PITCalibrator.pit_from_gaussian": ["pred_mean", "pred_std", "target"],
    "constraints.NonNegativeHead.forward": ["self", "x", "args", "kwargs"],
    "constraints.BoundedHead.forward": ["self", "x", "args", "kwargs"],
    "constraints.SimplexHead.forward": ["self", "x", "args", "kwargs"],
    "constraints.NonCrossingSort.forward": ["self", "x"],
    "constraints.SpectralNormWrapper.forward": ["self", "args", "kwargs"],
}


def _resolve(name: str):
    obj = tr
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        elif ch == delimiter and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return [p for p in parts if p]


def _find_matching_bracket(text: str, open_idx: int) -> int:
    depth = 0
    for idx in range(open_idx, len(text)):
        if text[idx] == "[":
            depth += 1
        elif text[idx] == "]":
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError(f"Unmatched bracket in signature text: {text!r}")


def _normalize_union_optional(text: str) -> str:
    i = 0
    out: list[str] = []
    while i < len(text):
        if text.startswith("Union[", i):
            open_idx = i + len("Union")
            close_idx = _find_matching_bracket(text, open_idx)
            inner = text[open_idx + 1 : close_idx]
            parts = _split_top_level(inner)
            normalized_parts = [_normalize_union_optional(part) for part in parts]
            out.append(" | ".join(normalized_parts))
            i = close_idx + 1
            continue
        if text.startswith("Optional[", i):
            open_idx = i + len("Optional")
            close_idx = _find_matching_bracket(text, open_idx)
            inner = text[open_idx + 1 : close_idx]
            out.append(f"{_normalize_union_optional(inner)} | None")
            i = close_idx + 1
            continue
        out.append(text[i])
        i += 1

    normalized = "".join(out).replace("NoneType", "None")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s*\|\s*", " | ", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    return normalized


def test_public_exports_snapshot_non_losses() -> None:
    module_map = {
        "torchregress": tr,
        "metrics": tr.metrics,
        "ensemble": tr.ensemble,
        "semi_supervised": tr.semi_supervised,
        "algorithms": tr.algorithms,
        "test_time": tr.test_time,
        "prediction": tr.prediction,
        "causal": tr.causal,
        "inference": tr.inference,
        "constraints": tr.constraints,
        "calibration": tr.calibration,
    }
    for module_name, expected in EXPECTED_EXPORTS.items():
        actual = list(module_map[module_name].__all__)
        assert actual == expected, module_name
        for symbol in actual:
            assert hasattr(module_map[module_name], symbol), f"{module_name}.{symbol}"


def test_signature_snapshots_non_losses() -> None:
    for path, expected in EXPECTED_SIGNATURES.items():
        actual = str(inspect.signature(_resolve(path)))
        assert _normalize_union_optional(actual) == _normalize_union_optional(
            expected
        ), f"{path}\nEXPECTED: {expected}\nACTUAL:   {actual}"


def test_parameter_order_contracts_for_new_helper_apis() -> None:
    for path, expected_order in EXPECTED_PARAM_ORDERS.items():
        actual_order = list(inspect.signature(_resolve(path)).parameters.keys())
        assert (
            actual_order == expected_order
        ), f"{path}\nEXPECTED PARAMS: {expected_order}\nACTUAL PARAMS:   {actual_order}"


def test_top_level_submodules_are_lazy_loaded() -> None:
    module = importlib.import_module("torchregress")
    module.__dict__.pop("viz", None)
    module.__dict__.pop("metrics", None)
    module = importlib.reload(module)
    assert "viz" not in module.__dict__
    assert "metrics" not in module.__dict__
    _ = module.viz
    _ = module.metrics
    assert "viz" in module.__dict__
    assert "metrics" in module.__dict__
