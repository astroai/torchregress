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
        maturity="Strong",
        imbalance="yes",
        calibration="partial",
        notes="Validate calibration after reweighting.",
    ),
    MethodMetadata(
        name="LDSLoss",
        family="imbalanced_loss",
        public_path="torchregress.losses.LDSLoss",
        task_tags=("imbalance", "rare_targets"),
        maturity="Available",
        imbalance="yes",
        calibration="partial",
        notes="Aggressive reweighting can distort calibration; check intervals post-hoc.",
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
        name="FunctionalEIVLoss",
        family="eiv",
        public_path="torchregress.losses.FunctionalEIVLoss",
        task_tags=("noisy_features", "measurement_error"),
        maturity="Available",
        noisy_features_eiv="yes",
        calibration="partial",
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
        recommended_start="GaussianNLLLoss",
        strong_alternatives=("HeteroscedasticEnsembleModel", "MDNLoss", "NormalizingFlowLoss"),
        notes="Single-model heteroscedastic is the cheapest upgrade.",
    ),
    TaskRecommendation(
        task="Epistemic uncertainty",
        recommended_start="DeepEnsemble",
        strong_alternatives=("SWAG", "BayesianNeuralNetwork", "MCDropoutWrapper"),
        notes="Deep ensembles are easiest operationally.",
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
        strong_alternatives=("NormalizingFlowLoss",),
        notes="MDN is usually easier to debug first.",
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
            "FunctionalEIVLoss / StructuralEIVLoss / OrthogonalDistanceRegressionLoss"
        ),
        strong_alternatives=("EnsembleEIVLoss",),
        notes="EIV losses change the call pattern (input noise is modeled).",
    ),
    TaskRecommendation(
        task="Noisy labels / label corruption",
        recommended_start="HuberLoss",
        strong_alternatives=("DeepEnsemble", "ConformalLoss"),
        notes="Prefer robust baselines before heavier methods.",
    ),
    TaskRecommendation(
        task="Imbalanced / rare-target regression",
        recommended_start="DensityWeightedLoss",
        strong_alternatives=("LDSLoss",),
        notes="Check calibration after aggressive reweighting.",
    ),
    TaskRecommendation(
        task="Calibrated intervals with coverage guarantees",
        recommended_start="ConformalLoss",
        strong_alternatives=("QuantileLoss",),
        notes="Conformal gives coverage, not UQ decomposition.",
    ),
    TaskRecommendation(
        task="Population inference with few labels",
        recommended_start="PredictionPoweredInference",
        strong_alternatives=("ConformalLoss", "QuantileLoss"),
        notes="Use PPI for means/quantiles/regression coefficients with limited labels.",
    ),
    TaskRecommendation(
        task="OOD scoring / selective prediction",
        recommended_start="DeepEnsemble + OOD metrics",
        strong_alternatives=("SWAG + OOD metrics", "BayesianNeuralNetwork + OOD metrics"),
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
            "FunctionalEIVLoss / StructuralEIVLoss / OrthogonalDistanceRegressionLoss"
        ),
        alternatives=("EnsembleEIVLoss",),
        caveat="EIV losses use a different call pattern (observed x passed as loss input).",
    ),
    DecisionWorkflowStep(
        order=5,
        question="Have imbalanced tails / rare targets?",
        primary_recommendation="DensityWeightedLoss",
        alternatives=("LDSLoss",),
        caveat="Validate calibration after aggressive reweighting.",
    ),
    DecisionWorkflowStep(
        order=6,
        question="Need OOD scoring / selective prediction under a latency budget?",
        primary_recommendation="DeepEnsemble + OOD metrics",
        alternatives=(
            "SWAG + OOD metrics",
            "BayesianNeuralNetwork + OOD metrics",
            "MCDropoutWrapper",
        ),
        caveat="Use multiple signals and benchmark runtime against deployment latency targets.",
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
        examples=("examples/imbalanced_regression.py",),
        comparison_grade="Strong",
        fairness_controls=("shared split", "summary tables"),
        metrics_coverage=("tail vs dense metrics", "calibration", "runtime"),
        peer_methods_visible=("DensityWeightedLoss", "LDSLoss"),
        gaps="Needs more model-family comparisons beyond reweighting losses.",
    ),
    ComparativeEvidenceRow(
        task="Calibrated intervals / coverage",
        examples=(
            "examples/evaluate_conformal_methods.py",
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
        peer_methods_visible=("ConformalLoss", "QuantileLoss", "GaussianNLLLoss"),
        gaps=(
            "Broader base-model diversity (especially ensembles/BNN/SWAG + conformal wrappers) "
            "needed for stronger generalization claims."
        ),
        notes=(
            "Photo-z benchmark adds domain-realistic coverage/width evaluation for Gaussian and "
            "quantile intervals alongside photo-z metrics; conformal method comparisons remain "
            "the primary coverage-guarantee benchmark. Ordered-bin NNC-CRPS-style comparisons "
            "are available in examples/photoz_nnc_crps_rail_comparison.py."
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
