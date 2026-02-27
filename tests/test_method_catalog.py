import pytest

import torchregress as tr
from torchregress import method_catalog


def test_method_catalog_includes_peer_uq_methods_without_experimental_default_label() -> None:
    names = set(method_catalog.list_method_names())
    assert {"SWAG", "BayesianNeuralNetwork", "MDNLoss"} <= names

    for name in ("SWAG", "BayesianNeuralNetwork", "MDNLoss"):
        meta = method_catalog.get_method_metadata(name)
        assert meta["maturity"] != "Experimental"
        assert meta["family"] in {"swag", "bnn", "mdn"}


def test_method_catalog_filtering_by_capability_and_task_tag() -> None:
    multimodal = method_catalog.list_methods(capability_filters={"multimodal": "yes"})
    multimodal_names = {row["name"] for row in multimodal}
    assert {"MDNLoss", "NormalizingFlowLoss"} <= multimodal_names

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

    ordinal = method_catalog.list_methods(task_tag="ordinal")
    ordinal_names = {row["name"] for row in ordinal}
    assert {"OrdinalCrossEntropyLoss", "CumulativeLinkLoss", "CORALLoss"} <= ordinal_names

    censored = method_catalog.list_methods(task_tag="censored")
    censored_names = {row["name"] for row in censored}
    assert {"CensoredGaussianNLLLoss", "CensoredQuantileLoss", "AFTLoss"} <= censored_names


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
        "OOD scoring / selective prediction",
        "Noisy features / measurement error",
        "Multimodal targets",
    } <= tasks

    ood_row = next(row for row in rows if row["task"] == "OOD scoring / selective prediction")
    assert ood_row["recommended_start"] == "DeepEnsemble + OOD metrics"
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
        "OOD robustness / selective prediction",
        "Noisy features / EIV",
        "Multimodal / multi-target non-Gaussian",
    } <= tasks
    grades = {row["comparison_grade"] for row in evidence}
    assert {"Decision-grade", "Strong"} <= grades
    assert "Missing" not in grades
    assert "Emerging" not in grades
    assert "Demo-only" not in grades
