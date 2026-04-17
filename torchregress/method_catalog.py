"""
Task-first method capability catalog for discovery and docs integration.

This is an evidence-neutral metadata layer: it helps users shortlist methods by
capability and family without forcing a Bayesian-first or non-Bayesian-first API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional

CapabilityValue = Literal["yes", "partial", "no"]


@dataclass(frozen=True)
class MethodMetadata:
    name: str
    family: str
    public_path: str
    task_tags: tuple[str, ...]
    maturity: str
    multimodal: CapabilityValue = "no"
    multi_target: CapabilityValue = "yes"
    non_gaussian: CapabilityValue = "partial"
    epistemic: CapabilityValue = "no"
    aleatoric: CapabilityValue = "no"
    decomposition: CapabilityValue = "no"
    calibration: CapabilityValue = "partial"
    ood_support: CapabilityValue = "partial"
    imbalance: CapabilityValue = "no"
    noisy_features_eiv: CapabilityValue = "no"
    notes: str = ""


@dataclass(frozen=True)
class TaskRecommendation:
    task: str
    recommended_start: str
    strong_alternatives: tuple[str, ...]
    notes: str


@dataclass(frozen=True)
class DecisionWorkflowStep:
    order: int
    question: str
    primary_recommendation: str
    alternatives: tuple[str, ...] = ()
    caveat: str = ""


@dataclass(frozen=True)
class ComparativeEvidenceRow:
    task: str
    examples: tuple[str, ...]
    comparison_grade: str
    fairness_controls: tuple[str, ...]
    metrics_coverage: tuple[str, ...]
    peer_methods_visible: tuple[str, ...]
    gaps: str
    notes: str = ""


_METHODS: tuple[MethodMetadata, ...] = (
    MethodMetadata(
        name="WeightedMSELoss",
        family="point_loss",
        public_path="torchregress.losses.WeightedMSELoss",
        task_tags=("baseline",),
        maturity="Core",
        non_gaussian="no",
        calibration="partial",
        notes=(
            "Strong clean-data baseline; pair with calibration/OOD metrics for deployment checks."
        ),
    ),
    MethodMetadata(
        name="HuberLoss",
        family="robust_loss",
        public_path="torchregress.losses.HuberLoss",
        task_tags=("outliers", "noisy_labels"),
        maturity="Core",
        notes="Robust default for moderate outliers before heavier methods.",
    ),
    MethodMetadata(
        name="DensityWeightedLoss",
        family="imbalanced_loss",
        public_path="torchregress.losses.DensityWeightedLoss",
        task_tags=("imbalance", "rare_targets"),
        maturity="Available",
        imbalance="yes",
        calibration="partial",
        notes=(
            "Research/Advanced method. Validate calibration after reweighting; "
            "photo-z benchmarks currently show this is highly data- and "
            "protocol-sensitive. Not recommended as a default."
        ),
    ),
    MethodMetadata(
        name="PropensityWeightedLoss",
        family="imbalanced_loss",
        public_path="torchregress.losses.PropensityWeightedLoss",
        task_tags=("imbalance", "selection_bias", "missing_labels"),
        maturity="Available",
        imbalance="yes",
        calibration="partial",
        notes="IPW-style loss for covariate-dependent label observation bias.",
    ),
    MethodMetadata(
        name="LDSLoss",
        family="imbalanced_loss",
        public_path="torchregress.losses.LDSLoss",
        task_tags=("imbalance", "rare_targets"),
        maturity="Available",
        imbalance="yes",
        calibration="partial",
        notes=(
            "Research/Advanced method. Aggressive reweighting can distort "
            "calibration; check intervals post-hoc."
        ),
    ),
    MethodMetadata(
        name="GaussianNLLLoss",
        family="probabilistic_loss",
        public_path="torchregress.losses.GaussianNLLLoss",
        task_tags=("heteroscedastic_noise", "aleatoric_uq"),
        maturity="Core",
        aleatoric="yes",
        calibration="partial",
    ),
    MethodMetadata(
        name="MultivariateGaussianLoss",
        family="probabilistic_loss",
        public_path="torchregress.losses.MultivariateGaussianLoss",
        task_tags=("multi_target", "correlated_targets"),
        maturity="Strong",
        multi_target="yes",
        aleatoric="yes",
        calibration="partial",
    ),
    MethodMetadata(
        name="LowRankGaussianLoss",
        family="probabilistic_loss",
        public_path="torchregress.losses.LowRankGaussianLoss",
        task_tags=("multi_target", "scalability"),
        maturity="Strong",
        multi_target="yes",
        aleatoric="yes",
        calibration="partial",
    ),
    MethodMetadata(
        name="QuantileLoss",
        family="quantile",
        public_path="torchregress.losses.QuantileLoss",
        task_tags=("prediction_intervals", "non_gaussian"),
        maturity="Core",
        non_gaussian="yes",
        calibration="yes",
    ),
    MethodMetadata(
        name="ConformalLoss",
        family="conformal",
        public_path="torchregress.losses.ConformalLoss",
        task_tags=("coverage_guarantees", "calibration"),
        maturity="Core",
        non_gaussian="yes",
        calibration="yes",
        decomposition="no",
        notes="Coverage guarantees, not epistemic/aleatoric decomposition.",
    ),
    MethodMetadata(
        name="UACQR",
        family="conformal",
        public_path="torchregress.losses.UACQR",
        task_tags=("coverage_guarantees", "calibration", "heteroscedastic_noise"),
        maturity="Available",
        non_gaussian="yes",
        calibration="yes",
        notes="CQR with nonconformity scores divided by predicted quantile band width.",
    ),
    MethodMetadata(
        name="DensityConformal",
        family="conformal",
        public_path="torchregress.losses.DensityConformal",
        task_tags=("coverage_guarantees", "density_conformal", "imbalance"),
        maturity="Available",
        non_gaussian="yes",
        calibration="yes",
        imbalance="yes",
        notes="Density-adaptive conformal intervals for long-tail target regions.",
    ),
    MethodMetadata(
        name="PrevalenceAdjustedCP",
        family="conformal",
        public_path="torchregress.losses.PrevalenceAdjustedCP",
        task_tags=("coverage_guarantees", "density_conformal", "imbalance"),
        maturity="Available",
        non_gaussian="yes",
        calibration="yes",
        imbalance="yes",
        notes="Group-prevalence adjusted conformal intervals for rare-target coverage.",
    ),
    MethodMetadata(
        name="MonteCarloConformal",
        family="conformal",
        public_path="torchregress.losses.MonteCarloConformal",
        task_tags=("coverage_guarantees", "density_conformal", "epistemic_uq"),
        maturity="Available",
        non_gaussian="yes",
        calibration="yes",
        epistemic="partial",
        notes="Conformal on MC predictive samples with uncertainty-normalized residual scores.",
    ),
    MethodMetadata(
        name="NoisyTargetGaussianNLL",
        family="uncertain_gt",
        public_path="torchregress.losses.NoisyTargetGaussianNLL",
        task_tags=("uncertain_ground_truth", "noisy_labels", "weak_supervision"),
        maturity="Available",
        non_gaussian="partial",
        aleatoric="yes",
        calibration="partial",
        notes=(
            "Adds target-noise variance to predictive variance before NLL scoring; "
            "strongest when probabilistic label uncertainty is genuinely available."
        ),
    ),
    MethodMetadata(
        name="ConsistencyRegLoss",
        family="uncertain_gt",
        public_path="torchregress.losses.ConsistencyRegLoss",
        task_tags=("uncertain_ground_truth", "weak_supervision", "noisy_labels"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes="Teacher/student consistency regularization with regression base loss.",
    ),
    MethodMetadata(
        name="PseudoLabelNLL",
        family="uncertain_gt",
        public_path="torchregress.losses.PseudoLabelNLL",
        task_tags=("uncertain_ground_truth", "weak_supervision", "noisy_labels"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes="Blends observed labels and pseudo-labels with confidence weighting.",
    ),
    MethodMetadata(
        name="PseudoLabelConsistencyLoss",
        family="uncertain_gt",
        public_path="torchregress.losses.PseudoLabelConsistencyLoss",
        task_tags=("uncertain_ground_truth", "weak_supervision", "semi_supervised"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes="Composite point-regression loss for pseudo-label and teacher-consistency training.",
    ),
    MethodMetadata(
        name="LogTransformLoss",
        family="target_transform",
        public_path="torchregress.losses.LogTransformLoss",
        task_tags=("target_transform", "skewed_targets"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes="Positive-support log-space loss for multiplicative noise and strong right skew.",
    ),
    MethodMetadata(
        name="BoxCoxTransformLoss",
        family="target_transform",
        public_path="torchregress.losses.BoxCoxTransformLoss",
        task_tags=("target_transform", "skewed_targets"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes="Tunable positive-support power transform for skewed targets.",
    ),
    MethodMetadata(
        name="SqrtTransformLoss",
        family="target_transform",
        public_path="torchregress.losses.SqrtTransformLoss",
        task_tags=("target_transform", "skewed_targets"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes="Square-root-space loss for count-like or moderate variance-growth targets.",
    ),
    MethodMetadata(
        name="YeoJohnsonTransformLoss",
        family="target_transform",
        public_path="torchregress.losses.YeoJohnsonTransformLoss",
        task_tags=("target_transform", "skewed_targets"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes="Signed-target power transform when Box-Cox/log are invalid.",
    ),
    MethodMetadata(
        name="PredictionPoweredInference",
        family="inference",
        public_path="torchregress.inference.ppi_mean_ci",
        task_tags=("inference", "limited_labels", "population_estimation"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes=(
            "Frequentist population/parameter inference layer that combines "
            "small labeled sets with larger prediction-only sets."
        ),
    ),
    MethodMetadata(
        name="dr_ate",
        family="causal",
        public_path="torchregress.causal.dr_ate",
        task_tags=("causal_inference", "ate", "selection_bias"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes="Cross-fitted doubly-robust ATE with overlap diagnostics and robust CI.",
    ),
    MethodMetadata(
        name="dr_cate",
        family="causal",
        public_path="torchregress.causal.dr_cate",
        task_tags=("causal_inference", "cate", "selection_bias"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes="Cross-fitted DR pseudo-outcome regression for heterogeneous treatment effects.",
    ),
    MethodMetadata(
        name="OrdinalCrossEntropyLoss",
        family="ordinal",
        public_path="torchregress.losses.OrdinalCrossEntropyLoss",
        task_tags=("ordinal", "ordered_targets"),
        maturity="Available",
        non_gaussian="yes",
        calibration="partial",
        notes=(
            "Class-logit baseline for ordered targets; also accepts soft bin targets "
            "for ambiguous labels and soft pseudo labels."
        ),
    ),
    MethodMetadata(
        name="CumulativeLinkLoss",
        family="ordinal",
        public_path="torchregress.losses.CumulativeLinkLoss",
        task_tags=("ordinal", "ordered_targets", "calibration"),
        maturity="Available",
        non_gaussian="yes",
        calibration="partial",
        notes=(
            "Cumulative-threshold objective with K-1 logits for ordinal labels or "
            "soft ordered-bin PMFs."
        ),
    ),
    MethodMetadata(
        name="CORALLoss",
        family="ordinal",
        public_path="torchregress.losses.CORALLoss",
        task_tags=("ordinal", "ordered_targets", "calibration"),
        maturity="Available",
        non_gaussian="yes",
        calibration="partial",
        notes="CORAL-style cumulative ordinal objective.",
    ),
    MethodMetadata(
        name="CensoredGaussianNLLLoss",
        family="censored",
        public_path="torchregress.losses.CensoredGaussianNLLLoss",
        task_tags=("censored", "interval_censored", "survival_regression"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes="Gaussian censored likelihood for left/right/interval-censored targets.",
    ),
    MethodMetadata(
        name="CensoredQuantileLoss",
        family="censored",
        public_path="torchregress.losses.CensoredQuantileLoss",
        task_tags=("censored", "interval_censored", "non_gaussian"),
        maturity="Available",
        non_gaussian="yes",
        calibration="partial",
        notes="Quantile-style objective for censored and bounded target intervals.",
    ),
    MethodMetadata(
        name="AFTLoss",
        family="censored",
        public_path="torchregress.losses.AFTLoss",
        task_tags=("censored", "interval_censored", "survival_regression"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes="Log-normal accelerated failure-time objective with censoring support.",
    ),
    MethodMetadata(
        name="BoundedHead",
        family="constraints",
        public_path="torchregress.constraints.BoundedHead",
        task_tags=("constraints", "bounded_outputs"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes="Enforces bounded output support via sigmoid scaling.",
    ),
    MethodMetadata(
        name="NonCrossingSort",
        family="constraints",
        public_path="torchregress.constraints.NonCrossingSort",
        task_tags=("constraints", "ordered_outputs"),
        maturity="Available",
        non_gaussian="partial",
        calibration="partial",
        notes="Post-head sorting transform to remove crossing violations.",
    ),
    MethodMetadata(
        name="VarianceTemperatureScaler",
        family="calibration_transform",
        public_path="torchregress.calibration.VarianceTemperatureScaler",
        task_tags=("calibration", "posthoc_calibration"),
        maturity="Available",
        calibration="yes",
        notes="Scalar variance temperature fitted by NLL minimization.",
    ),
    MethodMetadata(
        name="IsotonicMeanCalibrator",
        family="calibration_transform",
        public_path="torchregress.calibration.IsotonicMeanCalibrator",
        task_tags=("calibration", "posthoc_calibration"),
        maturity="Available",
        calibration="yes",
        notes="Monotonic regression map for mean prediction calibration.",
    ),
    MethodMetadata(
        name="PITCalibrator",
        family="calibration_transform",
        public_path="torchregress.calibration.PITCalibrator",
        task_tags=("calibration", "posthoc_calibration", "distribution_calibration"),
        maturity="Available",
        calibration="yes",
        notes="Empirical CDF remapping for PIT distribution calibration.",
    ),
    MethodMetadata(
        name="MDNLoss",
        family="mdn",
        public_path="torchregress.losses.MDNLoss",
        task_tags=("multimodal", "non_gaussian", "multi_target"),
        maturity="Available",
        multimodal="yes",
        multi_target="yes",
        non_gaussian="yes",
        aleatoric="yes",
        decomposition="yes",
    ),
    MethodMetadata(
        name="NormalizingFlowLoss",
        family="flow",
        public_path="torchregress.losses.NormalizingFlowLoss",
        task_tags=("multimodal", "non_gaussian", "multi_target"),
        maturity="Available",
        multimodal="yes",
        multi_target="yes",
        non_gaussian="yes",
        aleatoric="yes",
        decomposition="partial",
        notes="Optional dependency (`zuko`); often compared after MDN baselines.",
    ),
    MethodMetadata(
        name="ContrastiveFlowLoss",
        family="flow",
        public_path="torchregress.losses.ContrastiveFlowLoss",
        task_tags=("parameter_estimation", "domain_shift", "simulation_based_inference"),
        maturity="Available",
        multimodal="yes",
        multi_target="yes",
        non_gaussian="yes",
        aleatoric="yes",
        decomposition="partial",
        calibration="partial",
        notes=(
            "Likelihood-ratio style flow objective for parameter ranking against alternate "
            "hypotheses; strongest when simulation under alternate settings is available."
        ),
    ),
    MethodMetadata(
        name="FunctionalEIVLoss",
        family="eiv",
        public_path="torchregress.losses.FunctionalEIVLoss",
        task_tags=("noisy_features", "measurement_error"),
        maturity="Available",
        noisy_features_eiv="yes",
        calibration="partial",
        notes=(
            "Research/Advanced method. Powerful but ergonomically awkward; explicit "
            "adapters or simpler MC input-noise baselines can be easier to debug in "
            "practice."
        ),
    ),
    MethodMetadata(
        name="InputNoiseMarginalizationLoss",
        family="eiv",
        public_path="torchregress.losses.InputNoiseMarginalizationLoss",
        task_tags=("noisy_features", "measurement_error", "marginalization"),
        maturity="Strong",
        noisy_features_eiv="yes",
        calibration="partial",
        notes=(
            "Recommended default interface for Noisy-Input models. Integrates over "
            "expected input errors using Monte Carlo sampling instead of explicit "
            "Jacobian structures."
        ),
    ),
    MethodMetadata(
        name="NoisyInputPredictor",
        family="eiv",
        public_path="torchregress.losses.NoisyInputPredictor",
        task_tags=("noisy_features", "measurement_error", "inference"),
        maturity="Strong",
        noisy_features_eiv="yes",
        notes="High-level wrapper for performing inference on noisy inputs via marginalization.",
    ),
    MethodMetadata(
        name="InputNoiseMDNLoss",
        family="eiv",
        public_path="torchregress.losses.InputNoiseMDNLoss",
        task_tags=("noisy_features", "measurement_error", "multimodal"),
        maturity="Available",
        multimodal="yes",
        noisy_features_eiv="yes",
        notes="Marginalization for MDN heads; robust to input noise in multimodal scenarios.",
    ),
    MethodMetadata(
        name="InputNoiseBinnedPDFLoss",
        family="eiv",
        public_path="torchregress.losses.InputNoiseBinnedPDFLoss",
        task_tags=("noisy_features", "measurement_error", "multimodal"),
        maturity="Available",
        multimodal="yes",
        noisy_features_eiv="yes",
        notes="Marginalization for binned-PDF/ordinal heads.",
    ),
    MethodMetadata(
        name="StructuralEIVLoss",
        family="eiv",
        public_path="torchregress.losses.StructuralEIVLoss",
        task_tags=("noisy_features", "measurement_error"),
        maturity="Available",
        noisy_features_eiv="yes",
        calibration="partial",
    ),
    MethodMetadata(
        name="OrthogonalDistanceRegressionLoss",
        family="eiv",
        public_path="torchregress.losses.OrthogonalDistanceRegressionLoss",
        task_tags=("noisy_features", "measurement_error"),
        maturity="Available",
        noisy_features_eiv="yes",
        calibration="partial",
    ),
    MethodMetadata(
        name="DeepEnsemble",
        family="ensemble",
        public_path="torchregress.ensemble.DeepEnsemble",
        task_tags=("epistemic_uq", "ood", "selective_prediction"),
        maturity="Core",
        epistemic="yes",
        ood_support="yes",
        aleatoric="partial",
        decomposition="partial",
    ),
    MethodMetadata(
        name="HeteroscedasticEnsembleModel",
        family="ensemble",
        public_path="torchregress.ensemble.HeteroscedasticEnsembleModel",
        task_tags=("uq_decomposition", "ood", "calibration"),
        maturity="Strong",
        epistemic="yes",
        aleatoric="yes",
        decomposition="yes",
        ood_support="yes",
        calibration="partial",
    ),
    MethodMetadata(
        name="HeteroscedasticBatchEnsembleModel",
        family="ensemble",
        public_path="torchregress.ensemble.HeteroscedasticBatchEnsembleModel",
        task_tags=("uq_decomposition", "epistemic_uq", "aleatoric_uq", "ood", "low_compute"),
        maturity="Strong",
        epistemic="yes",
        aleatoric="yes",
        decomposition="yes",
        ood_support="partial",
        calibration="partial",
        notes="Shared-weight ensemble variant for lower-latency uncertainty decomposition.",
    ),
    MethodMetadata(
        name="PackedEnsembleRegressor",
        family="ensemble",
        public_path="torchregress.ensemble.PackedEnsembleRegressor",
        task_tags=("uq_decomposition", "epistemic_uq", "aleatoric_uq", "ood", "low_compute"),
        maturity="Available",
        epistemic="yes",
        aleatoric="partial",
        decomposition="partial",
        ood_support="partial",
        calibration="partial",
        notes=(
            "Facade over batch-ensemble heads with optional alpha scaling and "
            "PackedEnsembleOutput for mean/std_epistemic-style access."
        ),
    ),
    MethodMetadata(
        name="BinnedPDFEnsembleModel",
        family="ensemble",
        public_path="torchregress.ensemble.BinnedPDFEnsembleModel",
        task_tags=("multimodal_targets", "non_gaussian", "calibration"),
        maturity="Available",
        epistemic="yes",
        aleatoric="partial",
        decomposition="partial",
        non_gaussian="yes",
        calibration="partial",
        notes=(
            "Averages bin-wise probabilities in predictive-distribution space; "
            "strongest when discrete PDFs are the desired output surface."
        ),
    ),
    MethodMetadata(
        name="CumulativeLinkEnsembleModel",
        family="ensemble",
        public_path="torchregress.ensemble.CumulativeLinkEnsembleModel",
        task_tags=("ordinal", "non_gaussian", "calibration"),
        maturity="Available",
        epistemic="yes",
        aleatoric="partial",
        decomposition="partial",
        non_gaussian="yes",
        calibration="partial",
        notes=(
            "Averages ordinal CDF / PMF predictions across members rather than "
            "averaging thresholds."
        ),
    ),
    MethodMetadata(
        name="MDNEnsembleModel",
        family="ensemble",
        public_path="torchregress.ensemble.MDNEnsembleModel",
        task_tags=("multimodal_targets", "non_gaussian", "calibration", "uq_decomposition"),
        maturity="Available",
        epistemic="yes",
        aleatoric="yes",
        decomposition="yes",
        non_gaussian="yes",
        calibration="partial",
        notes=(
            "Uses a mixture-of-mixtures aggregation to avoid MDN component "
            "label-switching across members."
        ),
    ),
    MethodMetadata(
        name="MCDropoutWrapper",
        family="mc_dropout",
        public_path="torchregress.ensemble.MCDropoutWrapper",
        task_tags=("epistemic_uq", "low_compute"),
        maturity="Strong",
        epistemic="yes",
        aleatoric="partial",
        decomposition="partial",
        ood_support="partial",
    ),
    MethodMetadata(
        name="SWAG",
        family="swag",
        public_path="torchregress.ensemble.SWAG",
        task_tags=("epistemic_uq", "ood"),
        maturity="Available",
        epistemic="yes",
        aleatoric="partial",
        decomposition="partial",
        ood_support="partial",
    ),
    MethodMetadata(
        name="MultiSWAG",
        family="swag",
        public_path="torchregress.ensemble.MultiSWAG",
        task_tags=("epistemic_uq", "ood"),
        maturity="Available",
        epistemic="yes",
        aleatoric="partial",
        decomposition="partial",
        ood_support="partial",
    ),
    MethodMetadata(
        name="BayesianNeuralNetwork",
        family="bnn",
        public_path="torchregress.ensemble.BayesianNeuralNetwork",
        task_tags=("epistemic_uq", "ood"),
        maturity="Available",
        epistemic="yes",
        aleatoric="partial",
        decomposition="partial",
        ood_support="partial",
    ),
    MethodMetadata(
        name="HeteroscedasticBNN",
        family="bnn",
        public_path="torchregress.ensemble.HeteroscedasticBNN",
        task_tags=("uq_decomposition", "ood", "calibration"),
        maturity="Available",
        epistemic="yes",
        aleatoric="yes",
        decomposition="yes",
        ood_support="partial",
        calibration="partial",
    ),
    MethodMetadata(
        name="BayesianLinearHead",
        family="test_time",
        public_path="torchregress.test_time.BayesianLinearHead",
        task_tags=("low_shot", "test_time_adaptation", "last_layer", "linear_head"),
        maturity="Available",
        epistemic="partial",
        aleatoric="partial",
        decomposition="partial",
        calibration="partial",
        ood_support="no",
        notes=(
            "Exact conjugate Gaussian linear regression on fixed features; pairs with frozen "
            "representations. See examples/benchmarks/bayesian_linear_head_lowshot_adaptation.py "
            "and bayesian_linear_head_online_drift.py for synthetic adaptation baselines."
        ),
    ),
    MethodMetadata(
        name="RecursiveBayesianHead",
        family="test_time",
        public_path="torchregress.test_time.RecursiveBayesianHead",
        task_tags=("low_shot", "test_time_adaptation", "last_layer", "streaming"),
        maturity="Available",
        epistemic="partial",
        aleatoric="partial",
        decomposition="partial",
        calibration="partial",
        ood_support="no",
        notes=(
            "Streaming conjugate updates with optional precision forgetting; not a substitute "
            "for ensembles/SWAG/BNN when nonlinear epistemic UQ is required."
        ),
    ),
    MethodMetadata(
        name="BetaNLLLoss",
        family="probabilistic_loss",
        public_path="torchregress.losses.BetaNLLLoss",
        task_tags=("heteroscedastic_noise", "aleatoric_uq", "optimization_stability"),
        maturity="Available",
        aleatoric="yes",
        calibration="partial",
        notes="Variance-detached weighting on Gaussian NLL for heteroscedastic heads.",
    ),
    MethodMetadata(
        name="FaithfulGaussianLoss",
        family="probabilistic_loss",
        public_path="torchregress.losses.FaithfulGaussianLoss",
        task_tags=("heteroscedastic_noise", "aleatoric_uq", "mean_variance_decoupling"),
        maturity="Available",
        aleatoric="yes",
        calibration="partial",
        notes="MSE on mean plus NLL on variance with detached mean in the residual.",
    ),
    MethodMetadata(
        name="BalancedMSELoss",
        family="imbalanced_loss",
        public_path="torchregress.losses.BalancedMSELoss",
        task_tags=("imbalance", "rare_targets", "long_tail"),
        maturity="Available",
        imbalance="yes",
        calibration="partial",
        notes="Inverse bin-frequency weighted MSE; call fit(train_targets) on fixed bin edges.",
    ),
    MethodMetadata(
        name="BMCLoss",
        family="imbalanced_loss",
        public_path="torchregress.losses.BMCLoss",
        task_tags=("imbalance", "rare_targets", "long_tail"),
        maturity="Available",
        imbalance="yes",
        calibration="partial",
        notes=(
            "Inverse bin counts with Laplace-style noise_sigma smoothing; "
            "optional quantile bins."
        ),
    ),
    MethodMetadata(
        name="GaussianWassersteinBoundLoss",
        family="probabilistic_loss",
        public_path="torchregress.losses.GaussianWassersteinBoundLoss",
        task_tags=("heteroscedastic_noise", "multivariate_targets", "mean_covariance_supervision"),
        maturity="Available",
        aleatoric="partial",
        calibration="partial",
        notes=(
            "Frobenius surrogate on matrix roots for mean+covariance supervision; pair with "
            "pseudo-labels or oracle covariance targets."
        ),
    ),
    MethodMetadata(
        name="NeighborhoodCovariancePseudoLabeler",
        family="probabilistic_loss",
        public_path="torchregress.algorithms.NeighborhoodCovariancePseudoLabeler",
        task_tags=("pseudo_labels", "heteroscedastic_noise", "self_supervision"),
        maturity="Available",
        aleatoric="partial",
        calibration="partial",
        notes=(
            "Heuristic local target-covariance estimates from input-space neighbours; "
            "experimental. See examples/wasserstein_bound_hybrid_pretrain_demo.py."
        ),
    ),
    MethodMetadata(
        name="OptimalTransportCoverageGap",
        family="test_time",
        public_path="torchregress.test_time.OptimalTransportCoverageGap",
        task_tags=("distribution_shift", "conformal", "calibration"),
        maturity="Available",
        calibration="partial",
        ood_support="partial",
        notes="Score ECDF gap diagnostics between calibration and target pools.",
    ),
    MethodMetadata(
        name="OTShiftReweighter",
        family="test_time",
        public_path="torchregress.test_time.OTShiftReweighter",
        task_tags=("distribution_shift", "conformal", "calibration"),
        maturity="Available",
        calibration="partial",
        ood_support="partial",
        notes="Simplex weights on calibration scores via CDF matching (OT-inspired surrogate).",
    ),
    MethodMetadata(
        name="WeightedSplitConformalAdapter",
        family="test_time",
        public_path="torchregress.test_time.WeightedSplitConformalAdapter",
        task_tags=("distribution_shift", "conformal", "calibration"),
        maturity="Available",
        calibration="partial",
        notes="Weighted split conformal threshold for classification-style nonconformity scores.",
    ),
    MethodMetadata(
        name="weighted_split_classification_predictive_batch",
        family="test_time",
        public_path="torchregress.test_time.weighted_split_classification_predictive_batch",
        task_tags=("distribution_shift", "conformal", "calibration"),
        maturity="Available",
        calibration="partial",
        notes="Builds PredictiveBatch from a calibrated WeightedSplitConformalAdapter.",
    ),
)


_TASK_RECOMMENDATIONS: tuple[TaskRecommendation, ...] = (
    TaskRecommendation(
        task="Clean regression baseline",
        recommended_start="WeightedMSELoss",
        strong_alternatives=("HuberLoss",),
        notes="Start simple; add UQ only if needed.",
    ),
    TaskRecommendation(
        task="Outliers / robust regression",
        recommended_start="HuberLoss",
        strong_alternatives=("CauchyLoss", "TukeyBiweightLoss", "CharbonnierLoss"),
        notes="Huber is the best default tradeoff.",
    ),
    TaskRecommendation(
        task="Heteroscedastic noise (aleatoric UQ)",
        recommended_start="GaussianCRPSLoss",
        strong_alternatives=("GaussianNLLLoss", "HeteroscedasticEnsembleModel", "MDNLoss"),
        notes=(
            "Photo-z benchmarks favor CRPS-trained Gaussian heads as the safest "
            "calibrated Gaussian baseline."
        ),
    ),
    TaskRecommendation(
        task="Epistemic uncertainty",
        recommended_start="DeepEnsemble",
        strong_alternatives=(
            "HeteroscedasticBatchEnsembleModel",
            "BinnedPDFEnsembleModel",
            "MDNEnsembleModel",
            "SWAG",
            "BayesianNeuralNetwork",
            "MCDropoutWrapper",
        ),
        notes="Deep ensembles are easiest operationally.",
    ),
    TaskRecommendation(
        task="Low-shot / streaming linear head on fixed features",
        recommended_start="BayesianLinearHead",
        strong_alternatives=(
            "RecursiveBayesianHead",
            "WeightedMSELoss (ridge MAP, matched L2)",
        ),
        notes=(
            "Conjugate exact BLR for last-layer adaptation; synthetic RMSE/NLL and drift sweeps "
            "live under examples/benchmarks/. Prefer ensembles/SWAG/BNN when epistemic UQ must "
            "track representation-level ambiguity."
        ),
    ),
    TaskRecommendation(
        task="Epistemic + aleatoric decomposition",
        recommended_start="HeteroscedasticEnsembleModel",
        strong_alternatives=("HeteroscedasticBNN", "MDNLoss", "NormalizingFlowLoss"),
        notes="Requires variance/distribution modeling.",
    ),
    TaskRecommendation(
        task="Multimodal targets",
        recommended_start="MDNLoss",
        strong_alternatives=("MDNEnsembleModel", "BinnedPDFEnsembleModel", "NormalizingFlowLoss"),
        notes=(
            "MDN is usually easier to debug first; ensembles of MDN or ordered-bin "
            "heads are the next move when mode averaging matters."
        ),
    ),
    TaskRecommendation(
        task="Non-Gaussian / skewed tails",
        recommended_start="QuantileLoss / ExpectileLoss / TweedieLoss",
        strong_alternatives=("MDNLoss", "NormalizingFlowLoss"),
        notes="Choose by target support and evaluation metric.",
    ),
    TaskRecommendation(
        task="Multi-target correlated outputs",
        recommended_start="MultivariateGaussianLoss / LowRankGaussianLoss",
        strong_alternatives=("MDNLoss", "NormalizingFlowLoss"),
        notes="Prefer low-rank/full covariance when Gaussian is enough.",
    ),
    TaskRecommendation(
        task="Noisy features / measurement error",
        recommended_start=(
            "InputNoiseMarginalizationLoss + GaussianCRPSLoss / MDNLoss / BinnedPDF"
        ),
        strong_alternatives=(
            "FunctionalEIVLoss",
            "StructuralEIVLoss",
            "OrthogonalDistanceRegressionLoss",
        ),
        notes=(
            "Start with explicit input-noise marginalization and test-time "
            "predictive averaging, then escalate to Jacobian-based EIV losses "
            "only if they clearly help."
        ),
    ),
    TaskRecommendation(
        task="Noisy labels / label corruption",
        recommended_start="HuberLoss",
        strong_alternatives=("DeepEnsemble", "ConformalLoss"),
        notes="Prefer robust baselines before heavier methods.",
    ),
    TaskRecommendation(
        task="Imbalanced / rare-target regression",
        recommended_start="GaussianCRPSLoss / QuantileLoss + tail-slice evaluation",
        strong_alternatives=("DensityConformal",),
        notes=(
            "Photo-z benchmarks do not justify density weighting as default. "
            "Advanced research methods (DensityWeightedLoss, LDSLoss) should "
            "only be tried if coverage/calibration allow for tail gains."
        ),
    ),
    TaskRecommendation(
        task="Selection bias / covariate-dependent missing labels",
        recommended_start="PropensityWeightedLoss",
        strong_alternatives=("DensityWeightedLoss",),
        notes="Estimate p(observed|x) and apply IPW to reduce selection bias.",
    ),
    TaskRecommendation(
        task="Output constraints / monotonicity",
        recommended_start="BoundedHead / NonNegativeHead / NonCrossingSort",
        strong_alternatives=("SimplexHead", "SpectralNormWrapper"),
        notes="Apply structural constraints in the head before post-hoc calibration.",
    ),
    TaskRecommendation(
        task="Post-hoc calibration transforms",
        recommended_start="VarianceTemperatureScaler",
        strong_alternatives=("IsotonicMeanCalibrator", "PITCalibrator"),
        notes="Fit transforms on a held-out calibration split.",
    ),
    TaskRecommendation(
        task="Calibrated intervals with coverage guarantees",
        recommended_start="ConformalLoss on top of a strong probabilistic backbone",
        strong_alternatives=("QuantileLoss", "MonteCarloConformal", "DensityConformal"),
        notes=(
            "Conformal gives coverage, not density estimation; keep CRPS/NLL "
            "reporting from the underlying predictive model."
        ),
    ),
    TaskRecommendation(
        task="Density-aware conformal under long-tail targets",
        recommended_start="DensityConformal",
        strong_alternatives=("PrevalenceAdjustedCP", "MonteCarloConformal"),
        notes=(
            "Prefer density/prevalence variants when tail-region coverage is a "
            "key objective; point-accuracy gains are not guaranteed."
        ),
    ),
    TaskRecommendation(
        task="Uncertain ground-truth / weak labels",
        recommended_start="NoisyTargetGaussianNLL",
        strong_alternatives=(
            "OrdinalCrossEntropyLoss",
            "PseudoLabelConsistencyLoss",
            "PseudoLabelNLL",
        ),
        notes=(
            "Use NoisyTargetGaussianNLL for Gaussian label uncertainty and soft-bin / "
            "PMF supervision when target PDFs or intervals are the natural label form."
        ),
    ),
    TaskRecommendation(
        task="Semi-supervised regression",
        recommended_start="PseudoLabelConsistencyLoss",
        strong_alternatives=("PseudoLabelNLL", "NoisyTargetGaussianNLL"),
        notes=("Use confidence-gated pseudo labels and keep a clean held-out evaluation split."),
    ),
    TaskRecommendation(
        task="Target transforms for skewed / multiplicative-noise regression",
        recommended_start="LogTransformLoss",
        strong_alternatives=(
            "BoxCoxTransformLoss",
            "SqrtTransformLoss",
            "YeoJohnsonTransformLoss",
        ),
        notes=("Match transform support to target support before tuning model complexity."),
    ),
    TaskRecommendation(
        task="Causal inference regression (ATE/CATE)",
        recommended_start="dr_ate / dr_cate",
        strong_alternatives=("PredictionPoweredInference",),
        notes=("Use cross-fitting and overlap diagnostics before interpreting treatment effects."),
    ),
    TaskRecommendation(
        task="Population inference with few labels",
        recommended_start="PredictionPoweredInference",
        strong_alternatives=("ConformalLoss", "QuantileLoss"),
        notes=("Use PPI for means/quantiles/regression coefficients with limited labels."),
    ),
    TaskRecommendation(
        task="Ordinal / ordered targets",
        recommended_start="CumulativeLinkLoss",
        strong_alternatives=("CORALLoss", "OrdinalCrossEntropyLoss"),
        notes="Prefer cumulative objectives when rank-distance errors matter.",
    ),
    TaskRecommendation(
        task="Censored / interval-censored regression",
        recommended_start="CensoredGaussianNLLLoss",
        strong_alternatives=("AFTLoss", "CensoredQuantileLoss"),
        notes="Use censoring code 0/1/-1 and explicit interval bounds when available.",
    ),
    TaskRecommendation(
        task="OOD scoring / selective prediction",
        recommended_start="DeepEnsemble + OOD metrics",
        strong_alternatives=(
            "HeteroscedasticBatchEnsembleModel + OOD metrics",
            "SWAG + OOD metrics",
            "BayesianNeuralNetwork + OOD metrics",
        ),
        notes="Use multiple signals; no single OOD score is sufficient.",
    ),
)


_DECISION_WORKFLOW: tuple[DecisionWorkflowStep, ...] = (
    DecisionWorkflowStep(
        order=1,
        question="Need coverage guarantees?",
        primary_recommendation="ConformalLoss (split / CQR / ACI patterns)",
        caveat="Coverage guarantees are not uncertainty decomposition.",
    ),
    DecisionWorkflowStep(
        order=2,
        question="Need epistemic + aleatoric decomposition?",
        primary_recommendation="HeteroscedasticEnsembleModel",
        alternatives=("HeteroscedasticBNN", "MDNLoss", "NormalizingFlowLoss (+ ensemble)"),
        caveat="Requires variance/distribution modeling and stronger compute budget.",
    ),
    DecisionWorkflowStep(
        order=3,
        question="Need multimodal outputs?",
        primary_recommendation="MDNLoss",
        alternatives=("NormalizingFlowLoss",),
        caveat="Move to flows when MDN component count/training stability is the bottleneck.",
    ),
    DecisionWorkflowStep(
        order=4,
        question="Have noisy features / measurement error?",
        primary_recommendation=(
            "InputNoiseMarginalizationLoss + GaussianCRPSLoss / MDNLoss / BinnedPDF"
        ),
        alternatives=("FunctionalEIVLoss / StructuralEIVLoss / OrthogonalDistanceRegressionLoss",),
        caveat=(
            "Use the simpler explicit input-noise path first, including test-time "
            "predictive averaging; Jacobian-style EIV losses are more fragile and "
            "need careful benchmarking."
        ),
    ),
    DecisionWorkflowStep(
        order=5,
        question="Have imbalanced tails / rare targets?",
        primary_recommendation="GaussianCRPSLoss / QuantileLoss + tail-slice evaluation",
        alternatives=("DensityConformal", "DensityWeightedLoss", "LDSLoss"),
        caveat=(
            "Density-aware weighting is not yet a universally strong default on photo-z benchmarks."
        ),
    ),
    DecisionWorkflowStep(
        order=6,
        question="Need OOD scoring / selective prediction under a latency budget?",
        primary_recommendation="DeepEnsemble + OOD metrics",
        alternatives=(
            "HeteroscedasticBatchEnsembleModel + OOD metrics",
            "SWAG + OOD metrics",
            "BayesianNeuralNetwork + OOD metrics",
            "MCDropoutWrapper",
        ),
        caveat="Use multiple signals and benchmark runtime against deployment latency targets.",
    ),
    DecisionWorkflowStep(
        order=7,
        question="Are labels uncertain or weak (noisy targets, pseudo-labels, partial trust)?",
        primary_recommendation="NoisyTargetGaussianNLL",
        alternatives=("OrdinalCrossEntropyLoss", "PseudoLabelNLL", "ConsistencyRegLoss"),
        caveat=(
            "Retain held-out clean-label evaluation where available to avoid self-confirming loops."
        ),
    ),
    DecisionWorkflowStep(
        order=8,
        question="Need treatment-effect estimation under confounding (ATE/CATE)?",
        primary_recommendation="dr_ate / dr_cate",
        alternatives=("PredictionPoweredInference",),
        caveat=(
            "Check overlap/ESS diagnostics and avoid causal claims under severe "
            "positivity violations."
        ),
    ),
)


_COMPARATIVE_EVIDENCE_ROWS: tuple[ComparativeEvidenceRow, ...] = (
    ComparativeEvidenceRow(
        task="Robust regression / outliers",
        examples=(
            "examples/comprehensive_comparison.py",
            "examples/comprehensive_loss_comparison.py",
            "examples/photoz_benchmark_comparison.py",
        ),
        comparison_grade="Decision-grade",
        fairness_controls=(
            "fixed seeds",
            "shared scenarios / domain split",
            "runtime summaries",
        ),
        metrics_coverage=("MSE", "MAE", "R2", "NMAD", "catastrophic outlier rate", "runtime"),
        peer_methods_visible=("HuberLoss", "CauchyLoss", "WeightedMSELoss", "WeightedHuberLoss"),
        gaps="Only one domain benchmark (photo-z) so far; needs broader domain coverage.",
        notes=(
            "Photo-z benchmark adds SDSS-style domain metrics (NMAD, catastrophic outlier rate, "
            "high-z MAE) and robust/probabilistic/EIV comparisons under shared budgets; "
            "RAIL baseline merge support is available via tools/photoz_rail_compare.py."
        ),
    ),
    ComparativeEvidenceRow(
        task="Imbalanced / rare-target regression",
        examples=(
            "examples/imbalanced_regression.py",
            "examples/propensity_tail_regression_comparison.py",
        ),
        comparison_grade="Strong",
        fairness_controls=("shared split", "summary tables"),
        metrics_coverage=(
            "tail vs dense metrics",
            "tail MAE/RMSE",
            "native interval coverage/width",
            "runtime",
        ),
        peer_methods_visible=(
            "DensityWeightedLoss",
            "PropensityWeightedLoss",
            "LDSLoss",
            "GaussianNLLLoss",
            "MultiQuantileLoss",
        ),
        gaps="Needs additional real-data long-tail benchmarks beyond synthetic selection proxies.",
    ),
    ComparativeEvidenceRow(
        task="Selection bias / long-tail with missing labels",
        examples=("examples/propensity_tail_regression_comparison.py",),
        comparison_grade="Strong",
        fairness_controls=("fixed seed", "shared selection process", "matched model capacity"),
        metrics_coverage=(
            "MAE",
            "tail MAE/RMSE",
            "native interval coverage/width",
            "observed-rate diagnostics",
            "runtime",
        ),
        peer_methods_visible=(
            "PropensityWeightedLoss",
            "DensityWeightedLoss",
            "WeightedMSELoss",
            "GaussianNLLLoss",
            "MultiQuantileLoss",
        ),
        gaps="Needs real-data selection-bias benchmarks beyond synthetic generation.",
    ),
    ComparativeEvidenceRow(
        task="Output constraints + post-hoc calibration transforms",
        examples=("examples/constraints_calibration_comparison.py",),
        comparison_grade="Strong",
        fairness_controls=("fixed seed", "shared synthetic split", "shared calibrator fit budget"),
        metrics_coverage=("MAE", "NLL", "PIT chi-square", "crossing/bound violations", "runtime"),
        peer_methods_visible=(
            "BoundedHead",
            "NonCrossingSort",
            "VarianceTemperatureScaler",
            "IsotonicMeanCalibrator",
            "PITCalibrator",
        ),
        gaps="Needs additional domain benchmarks beyond synthetic stress tests.",
    ),
    ComparativeEvidenceRow(
        task="Target transforms for skewed regression",
        examples=("examples/transformed_target_regression_comparison.py",),
        comparison_grade="Strong",
        fairness_controls=("fixed seed", "shared architecture", "shared optimizer/epoch budget"),
        metrics_coverage=("MSE", "MAE", "R2", "MAPE", "upper-tail MAE", "runtime"),
        peer_methods_visible=(
            "WeightedMSELoss",
            "LogTransformLoss",
            "BoxCoxTransformLoss",
            "SqrtTransformLoss",
        ),
        gaps=(
            "Needs real-data positive-target benchmarks beyond synthetic "
            "multiplicative-noise tasks."
        ),
        notes="Focused evidence for target-transform choice under skewed positive targets.",
    ),
    ComparativeEvidenceRow(
        task="Semi-supervised regression / limited labels",
        examples=("examples/semi_supervised_regression_comparison.py",),
        comparison_grade="Strong",
        fairness_controls=(
            "fixed seed",
            "shared architecture",
            "shared labeled/unlabeled split",
            "shared teacher bootstrap",
        ),
        metrics_coverage=("MSE", "MAE", "R2", "pseudo-label acceptance", "runtime"),
        peer_methods_visible=(
            "PseudoLabelConsistencyLoss",
            "PseudoLabelNLL",
            "WeightedMSELoss",
        ),
        gaps=(
            "Current evidence is one real-data proxy benchmark; add domain-native "
            "SSL regression tracks."
        ),
        notes=(
            "Real-data proxy benchmark uses Diabetes with masked labels and "
            "confidence-gated pseudo labels."
        ),
    ),
    ComparativeEvidenceRow(
        task="Uncertain ground-truth + density-aware conformal",
        examples=(
            "examples/semi_supervised_regression_comparison.py",
            "examples/uncertain_gt_density_conformal_comparison.py",
            "examples/uncertain_gt_density_conformal_realdata_comparison.py",
        ),
        comparison_grade="Strong",
        fairness_controls=(
            "fixed seed",
            "shared synthetic/real-data split",
            "shared calibration budget",
        ),
        metrics_coverage=(
            "coverage",
            "interval width",
            "noisy-target NLL",
            "consistency loss",
            "pseudo-label NLL",
            "runtime",
        ),
        peer_methods_visible=(
            "SplitConformal",
            "DensityConformal",
            "PrevalenceAdjustedCP",
            "MonteCarloConformal",
            "NoisyTargetGaussianNLL",
            "PseudoLabelNLL",
            "ConsistencyRegLoss",
            "PseudoLabelConsistencyLoss",
        ),
        gaps=(
            "Includes one real-data proxy benchmark; needs domain-native uncertain-label datasets "
            "for stronger external validity."
        ),
        notes=(
            "Evidence now includes synthetic conformal tracks plus a real-data "
            "masked-label SSL proxy benchmark and real-data (Diabetes) "
            "uncertain-label evaluation."
        ),
    ),
    ComparativeEvidenceRow(
        task="Causal inference regression (DR ATE/CATE)",
        examples=(
            "examples/causal_dr_uplift_comparison.py",
            "examples/causal_dr_realdata_comparison.py",
        ),
        comparison_grade="Strong",
        fairness_controls=(
            "fixed seed",
            "shared synthetic/real-covariate scenarios",
            "matched nuisance models",
        ),
        metrics_coverage=(
            "ATE absolute error",
            "CI contains true ATE",
            "CI width",
            "overlap rate",
            "effective sample size",
            "runtime",
        ),
        peer_methods_visible=("dr_ate", "dr_cate", "naive difference-in-means"),
        gaps=(
            "Includes real-covariate proxy benchmarks; needs external treatment-effect datasets "
            "for stronger external validity."
        ),
        notes=(
            "Evidence includes synthetic uplift/astronomy-style scenarios plus real-covariate "
            "Diabetes proxy scenarios, all with cross-fitted DR estimators and overlap diagnostics."
        ),
    ),
    ComparativeEvidenceRow(
        task="Calibrated intervals / coverage",
        examples=(
            "examples/evaluate_conformal_methods.py",
            "examples/ood_selective_prediction_comparison.py",
            "examples/ood_selective_prediction_realdata_comparison.py",
            "examples/photoz_benchmark_comparison.py",
            "examples/photoz_nnc_crps_rail_comparison.py",
        ),
        comparison_grade="Strong",
        fairness_controls=(
            "shared evaluation budget",
            "coverage/width summaries",
            "shared train budget",
        ),
        metrics_coverage=("coverage", "interval width", "runtime", "domain error metrics"),
        peer_methods_visible=(
            "ConformalLoss",
            "QuantileLoss",
            "GaussianNLLLoss",
            "DeepEnsemble",
            "SWAG",
            "BayesianNeuralNetwork",
        ),
        gaps=(
            "Coverage evidence now spans ensemble/SWAG/BNN base models; still needs "
            "multi-domain real-data calibration benchmarks under stronger shift."
        ),
        notes=(
            "Photo-z benchmark adds domain-realistic coverage/width evaluation for Gaussian and "
            "quantile intervals alongside photo-z metrics; OOD/selective comparisons now include "
            "split-conformal interval diagnostics across DeepEnsemble/MCDropout/SWAG/BNN. "
            "Ordered-bin NNC-CRPS-style comparisons are available in "
            "examples/photoz_nnc_crps_rail_comparison.py."
        ),
    ),
    ComparativeEvidenceRow(
        task="Population/parameter inference (few labels)",
        examples=("examples/ppi_photoz_inference_comparison.py",),
        comparison_grade="Strong",
        fairness_controls=("fixed seed", "shared labeled/unlabeled split", "runtime summaries"),
        metrics_coverage=("estimate bias", "CI width", "CI coverage", "runtime"),
        peer_methods_visible=("PredictionPoweredInference", "labeled-only baseline"),
        gaps="Needs more than one real-data benchmark for generalization claims.",
        notes=(
            "Prediction-powered inference example demonstrates mean/quantile/OLS coefficient "
            "intervals with diagnostics under small-label settings."
        ),
    ),
    ComparativeEvidenceRow(
        task="Ordinal regression / ordered targets",
        examples=(
            "examples/ordinal_regression_comparison.py",
            "examples/ordinal_regression_realdata_comparison.py",
            "examples/ordinal_uncertain_ground_truth_comparison.py",
        ),
        comparison_grade="Strong",
        fairness_controls=(
            "fixed seed",
            "shared synthetic/real-data split",
            "matched model capacity",
        ),
        metrics_coverage=("accuracy", "ordinal class MAE", "QWK", "true-class NLL", "runtime"),
        peer_methods_visible=("OrdinalCrossEntropyLoss", "CumulativeLinkLoss", "CORALLoss"),
        gaps=(
            "Includes one quantile-binned real-data benchmark; needs domain-native ordinal-label "
            "datasets for stronger external validity."
        ),
        notes=(
            "Comparison examples evaluate hard-label, soft-plausibility, and cumulative "
            "objectives under shared training budgets and ordinal-aware metrics."
        ),
    ),
    ComparativeEvidenceRow(
        task="Censored / interval-censored regression",
        examples=(
            "examples/censored_regression_comparison.py",
            "examples/censored_regression_realdata_comparison.py",
        ),
        comparison_grade="Strong",
        fairness_controls=(
            "fixed seed",
            "shared synthetic/real-data censoring split",
            "matched model capacity",
        ),
        metrics_coverage=("true-target MAE", "observed MAE", "concordance index", "runtime"),
        peer_methods_visible=("CensoredGaussianNLLLoss", "CensoredQuantileLoss", "AFTLoss"),
        gaps=(
            "Includes one real-data synthetic-censoring benchmark; needs naturally censored "
            "datasets for stronger external validity."
        ),
        notes=(
            "Examples include right/left censoring plus explicit interval-censored samples "
            "under shared training budgets on synthetic and real-data tracks."
        ),
    ),
    ComparativeEvidenceRow(
        task="OOD robustness / selective prediction",
        examples=(
            "examples/ood_selective_prediction_comparison.py",
            "examples/ood_selective_prediction_realdata_comparison.py",
            "examples/comprehensive_comparison.py",
        ),
        comparison_grade="Decision-grade",
        fairness_controls=(
            "fixed seeds",
            "shared synthetic ID/OOD splits",
            "shared train budget",
        ),
        metrics_coverage=(
            "ID/OOD MSE",
            "AURC",
            "rejection policy",
            "OOD uncertainty gap",
            "runtime",
        ),
        peer_methods_visible=(
            "DeepEnsemble",
            "HeteroscedasticEnsembleModel",
            "MCDropoutWrapper",
            "SWAG",
            "BayesianNeuralNetwork",
        ),
        gaps=(
            "Needs multiple real-data OOD/selective benchmarks (beyond one covariate-shift proxy) "
            "for stronger external validity and regression tracking."
        ),
        notes=(
            "Dedicated OOD/selective examples now include synthetic and real-data (Diabetes "
            "covariate-shift proxy) comparisons with SWAG and BNN under shared budgets; "
            "committed smoke/audit/full JSON summary artifacts are available."
        ),
    ),
    ComparativeEvidenceRow(
        task="Multimodal / multi-target non-Gaussian",
        examples=(
            "examples/multimodal_method_comparison.py",
            "examples/multimodal_method_realdata_comparison.py",
            "examples/normalizing_flows_multitarget.py",
        ),
        comparison_grade="Strong",
        fairness_controls=(
            "fixed seeds",
            "shared synthetic / real-covariate multimodal splits",
            "shared train budget",
        ),
        metrics_coverage=(
            "point metrics",
            "NLL",
            "energy score",
            "marginal calibration error",
            "runtime",
        ),
        peer_methods_visible=("GaussianNLLLoss", "MDNLoss", "NormalizingFlowLoss"),
        gaps=(
            "Needs domain-specific real-data multimodal benchmark(s) (beyond synthetic multimodal "
            "targets on real covariates) and optional-dependency CI coverage for zuko flow runs."
        ),
        notes=(
            "Dedicated comparisons now include synthetic and real-covariate tracks; flow path is "
            "optional and may be skipped in minimal environments without zuko."
        ),
    ),
    ComparativeEvidenceRow(
        task="Noisy features / EIV",
        examples=(
            "examples/eiv_method_comparison.py",
            "examples/eiv_method_realdata_comparison.py",
            "examples/photoz_benchmark_comparison.py",
        ),
        comparison_grade="Strong",
        fairness_controls=(
            "fixed seeds",
            "shared synthetic/real-data measurement-error splits",
            "shared train budget",
        ),
        metrics_coverage=("clean test MSE", "observed test MSE", "stress MSE", "runtime"),
        peer_methods_visible=(
            "FunctionalEIVLoss",
            "StructuralEIVLoss",
            "OrthogonalDistanceRegressionLoss",
            "EnsembleEIVLoss",
        ),
        gaps=(
            "Needs additional larger-scale/nonlinear real-data benchmarks (beyond Diabetes and "
            "one photo-z domain benchmark) for stronger "
            "external validity."
        ),
        notes=(
            "EIV comparisons now include synthetic and real-data (Diabetes with synthetic "
            "measurement-error injection) tracks, plus a photo-z benchmark with SDSS-style "
            "feature errors."
        ),
    ),
    ComparativeEvidenceRow(
        task="Noisy labels / corruption",
        examples=(
            "examples/noisy_label_comparison.py",
            "examples/noisy_label_realdata_comparison.py",
            "examples/comprehensive_loss_comparison.py",
        ),
        comparison_grade="Strong",
        fairness_controls=(
            "fixed seeds",
            "shared corrupted train/cal/test splits",
            "shared train budget",
        ),
        metrics_coverage=(
            "clean/observed point metrics",
            "split-conformal coverage/width",
            "interval score",
            "runtime",
        ),
        peer_methods_visible=(
            "WeightedHuberLoss",
            "CauchyLoss",
            "TukeyBiweightLoss",
            "GaussianNLLLoss",
            "MultiQuantileLoss",
        ),
        gaps=(
            "Needs comparisons against explicit noisy-label algorithms "
            "(co-teaching / sample-weight meta-learning) if/when implemented, plus more than "
            "one real dataset for stronger external validity."
        ),
        notes=(
            "Dedicated runnable comparisons now include synthetic and real-data (Diabetes with "
            "synthetic corruption) calibration-aware evaluations via shared split-conformal "
            "metrics."
        ),
    ),
    ComparativeEvidenceRow(
        task="Low-shot linear adaptation on fixed features (last layer)",
        examples=(
            "examples/benchmarks/bayesian_linear_head_lowshot_adaptation.py",
            "examples/benchmarks/bayesian_linear_head_online_drift.py",
        ),
        comparison_grade="Strong",
        fairness_controls=(
            "fixed seeds",
            "matched ridge L2 vs conjugate prior in low-shot script",
            "tabular stdout summaries (RMSE / NLL / streaming RMSE)",
        ),
        metrics_coverage=("RMSE", "Gaussian NLL", "streaming RMSE under forgetting"),
        peer_methods_visible=(
            "BayesianLinearHead",
            "RecursiveBayesianHead",
            "WeightedMSELoss",
        ),
        gaps=(
            "Synthetic Gaussian linear benchmarks only; needs real frozen-backbone protocols "
            "before deployment-grade ranking against ensembles or SWAG."
        ),
        notes=(
            "CPU-only benchmarks: low-shot path aligns BLR posterior mean with ridge MAP under "
            "matched L2 and contrasts predictive NLL with an oracle homoscedastic ridge baseline; "
            "drift path evaluates recursive partial_fit versus a phase-only batch oracle."
        ),
    ),
)


def list_methods(
    *,
    family: Optional[str] = None,
    task_tag: Optional[str] = None,
    capability_filters: Optional[Dict[str, CapabilityValue]] = None,
    maturity: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """
    List cataloged methods with optional filtering.

    Filters are exact matches on capability values (`yes`/`partial`/`no`).
    """
    maturity_set = set(maturity) if maturity is not None else None
    rows: list[MethodMetadata] = list(_METHODS)

    if family is not None:
        rows = [m for m in rows if m.family == family]
    if task_tag is not None:
        rows = [m for m in rows if task_tag in m.task_tags]
    if maturity_set is not None:
        rows = [m for m in rows if m.maturity in maturity_set]
    if capability_filters:
        filtered: list[MethodMetadata] = []
        for m in rows:
            ok = True
            for key, expected in capability_filters.items():
                if not hasattr(m, key) or getattr(m, key) != expected:
                    ok = False
                    break
            if ok:
                filtered.append(m)
        rows = filtered

    return [asdict(m) for m in rows]


def get_method_metadata(name: str) -> Dict[str, Any]:
    """Return metadata for a single method by display name."""
    for method in _METHODS:
        if method.name == name:
            return asdict(method)
    raise KeyError(f"Unknown method '{name}'.")


def list_method_names() -> List[str]:
    """Convenience list of cataloged method names."""
    return [m.name for m in _METHODS]


def list_task_recommendations() -> List[Dict[str, Any]]:
    """Return task-first starting-point recommendations for docs and onboarding."""
    return [asdict(row) for row in _TASK_RECOMMENDATIONS]


def list_decision_workflow_steps() -> List[Dict[str, Any]]:
    """Return fast-path decision workflow steps for onboarding docs."""
    return [asdict(row) for row in _DECISION_WORKFLOW]


def list_comparative_evidence_rows() -> List[Dict[str, Any]]:
    """Return comparative example evidence coverage rows used for docs/audit reporting."""
    return [asdict(row) for row in _COMPARATIVE_EVIDENCE_ROWS]
