import pytest

import torchregress as tr
from torchregress import method_catalog


def test_method_catalog_includes_peer_uq_methods_without_experimental_default_label() -> None:
    names = set(method_catalog.list_method_names())
    assert {"SWAG", "BayesianNeuralNetwork", "MDNLoss"} <= names
    assert {"BayesianLinearHead", "RecursiveBayesianHead"} <= names

    for name in ("SWAG", "BayesianNeuralNetwork", "MDNLoss"):
        meta = method_catalog.get_method_metadata(name)
        assert meta["maturity"] != "Experimental"
        assert meta["family"] in {"swag", "bnn", "mdn"}

    for name in ("BayesianLinearHead", "RecursiveBayesianHead"):
        meta = method_catalog.get_method_metadata(name)
        assert meta["maturity"] == "Available"
        assert meta["family"] == "test_time"


def test_method_catalog_filtering_by_capability_and_task_tag() -> None:
    multimodal = method_catalog.list_methods(capability_filters={"multimodal": "yes"})
    multimodal_names = {row["name"] for row in multimodal}
    assert {"MDNLoss", "NormalizingFlowLoss", "ContrastiveFlowLoss"} <= multimodal_names

    eiv = method_catalog.list_methods(task_tag="noisy_features")
    eiv_names = {row["name"] for row in eiv}
    assert "FunctionalEIVLoss" in eiv_names
    assert "OrthogonalDistanceRegressionLoss" in eiv_names

    decomposition = method_catalog.list_methods(capability_filters={"decomposition": "yes"})
    decomp_names = {row["name"] for row in decomposition}
    assert {"HeteroscedasticEnsembleModel", "HeteroscedasticBNN", "MDNLoss"} <= decomp_names

    inference = method_catalog.list_methods(task_tag="inference")
    inference_names = {row["name"] for row in inference}
    assert "PredictionPoweredInference" in inference_names

    param_est = method_catalog.list_methods(task_tag="parameter_estimation")
    param_est_names = {row["name"] for row in param_est}
    assert "ContrastiveFlowLoss" in param_est_names

    ordinal = method_catalog.list_methods(task_tag="ordinal")
    ordinal_names = {row["name"] for row in ordinal}
    assert {"OrdinalCrossEntropyLoss", "CumulativeLinkLoss", "CORALLoss"} <= ordinal_names

    censored = method_catalog.list_methods(task_tag="censored")
    censored_names = {row["name"] for row in censored}
    assert {"CensoredGaussianNLLLoss", "CensoredQuantileLoss", "AFTLoss"} <= censored_names

    selection_bias = method_catalog.list_methods(task_tag="selection_bias")
    selection_names = {row["name"] for row in selection_bias}
    assert "PropensityWeightedLoss" in selection_names

    posthoc = method_catalog.list_methods(task_tag="posthoc_calibration")
    posthoc_names = {row["name"] for row in posthoc}
    assert {
        "VarianceTemperatureScaler",
        "IsotonicMeanCalibrator",
        "PITCalibrator",
    } <= posthoc_names

    uncertain_gt = method_catalog.list_methods(task_tag="uncertain_ground_truth")
    uncertain_gt_names = {row["name"] for row in uncertain_gt}
    assert {
        "NoisyTargetGaussianNLL",
        "PseudoLabelNLL",
        "ConsistencyRegLoss",
        "PseudoLabelConsistencyLoss",
    } <= uncertain_gt_names

    target_transforms = method_catalog.list_methods(task_tag="target_transform")
    transform_names = {row["name"] for row in target_transforms}
    assert {
        "LogTransformLoss",
        "BoxCoxTransformLoss",
        "SqrtTransformLoss",
        "YeoJohnsonTransformLoss",
    } <= transform_names

    density_cp = method_catalog.list_methods(task_tag="density_conformal")
    density_cp_names = {row["name"] for row in density_cp}
    assert {"DensityConformal", "PrevalenceAdjustedCP", "MonteCarloConformal"} <= density_cp_names

    causal = method_catalog.list_methods(task_tag="causal_inference")
    causal_names = {row["name"] for row in causal}
    assert {"dr_ate", "dr_cate"} <= causal_names

    low_compute = method_catalog.list_methods(task_tag="low_compute")
    low_compute_names = {row["name"] for row in low_compute}
    assert {"HeteroscedasticBatchEnsembleModel", "MCDropoutWrapper"} <= low_compute_names

    low_shot = method_catalog.list_methods(task_tag="low_shot")
    low_shot_names = {row["name"] for row in low_shot}
    assert {"BayesianLinearHead", "RecursiveBayesianHead"} <= low_shot_names


def test_method_catalog_is_exposed_via_top_level_module_namespace() -> None:
    assert hasattr(tr, "method_catalog")
    assert tr.method_catalog.get_method_metadata("ConformalLoss")["family"] == "conformal"


def test_method_catalog_unknown_method_raises_key_error() -> None:
    with pytest.raises(KeyError):
        method_catalog.get_method_metadata("DoesNotExist")


def test_task_recommendations_include_hard_problem_rows_and_peer_methods() -> None:
    rows = method_catalog.list_task_recommendations()
    tasks = {row["task"] for row in rows}
    assert {
        "Imbalanced / rare-target regression",
        "Calibrated intervals with coverage guarantees",
        "Population inference with few labels",
        "Ordinal / ordered targets",
        "Censored / interval-censored regression",
        "Selection bias / covariate-dependent missing labels",
        "Output constraints / monotonicity",
        "Post-hoc calibration transforms",
        "Density-aware conformal under long-tail targets",
        "Uncertain ground-truth / weak labels",
        "Semi-supervised regression",
        "Target transforms for skewed / multiplicative-noise regression",
        "Causal inference regression (ATE/CATE)",
        "OOD scoring / selective prediction",
        "Noisy features / measurement error",
        "Multimodal targets",
        "Low-shot / streaming linear head on fixed features",
    } <= tasks

    ood_row = next(row for row in rows if row["task"] == "OOD scoring / selective prediction")
    assert ood_row["recommended_start"] == "DeepEnsemble + OOD metrics"
    assert any("HeteroscedasticBatchEnsembleModel" in alt for alt in ood_row["strong_alternatives"])
    assert any("SWAG" in alt for alt in ood_row["strong_alternatives"])
    assert any("BayesianNeuralNetwork" in alt for alt in ood_row["strong_alternatives"])


def test_decision_workflow_and_comparative_evidence_metadata_cover_hard_tasks() -> None:
    workflow = method_catalog.list_decision_workflow_steps()
    assert any("coverage guarantees" in row["question"].lower() for row in workflow)
    assert any("OOD" in row["question"] for row in workflow)
    assert any("SWAG" in alt for row in workflow for alt in row["alternatives"])

    evidence = method_catalog.list_comparative_evidence_rows()
    tasks = {row["task"] for row in evidence}
    assert {
        "Imbalanced / rare-target regression",
        "Calibrated intervals / coverage",
        "Ordinal regression / ordered targets",
        "Censored / interval-censored regression",
        "Selection bias / long-tail with missing labels",
        "Output constraints + post-hoc calibration transforms",
        "Target transforms for skewed regression",
        "Semi-supervised regression / limited labels",
        "Uncertain ground-truth + density-aware conformal",
        "Causal inference regression (DR ATE/CATE)",
        "OOD robustness / selective prediction",
        "Noisy features / EIV",
        "Multimodal / multi-target non-Gaussian",
        "Low-shot linear adaptation on fixed features (last layer)",
    } <= tasks
    grades = {row["comparison_grade"] for row in evidence}
    assert {"Decision-grade", "Strong"} <= grades
    assert "Missing" not in grades
    assert "Emerging" not in grades
    assert "Demo-only" not in grades
