from __future__ import annotations

import json
from pathlib import Path

from tools import render_example_summaries


def test_render_example_summaries_smoke_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["ood_selective_prediction_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert payload["task"] == "OOD robustness / selective prediction"
    methods = {row["Method"] for row in payload["rows"]}
    assert {"SWAG", "BayesianNeuralNetwork"} <= methods


def test_render_example_summaries_ood_realdata_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["ood_selective_prediction_realdata_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "real-data" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"SWAG", "BayesianNeuralNetwork"} <= methods


def test_render_example_summaries_eiv_realdata_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["eiv_method_realdata_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "real-data" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"Baseline MSE", "FunctionalEIV (analytic)", "ODR"} <= methods


def test_render_example_summaries_multimodal_realdata_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["multimodal_method_realdata_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "real-data" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"GaussianNLL", "MDN", "NormalizingFlow"} <= methods


def test_render_example_summaries_contrastive_flow_synth_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["contrastive_flow_parameter_estimation_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "parameter estimation" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"GaussianSummary", "NormalizingFlow", "ContrastiveFlow"} <= methods


def test_render_example_summaries_realdata_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["noisy_label_realdata_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "real-data" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"MSE", "Huber", "GaussianNLL", "Quantile90"} <= methods


def test_render_example_summaries_ordinal_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["ordinal_regression_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "ordinal" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"OrdinalCrossEntropy", "CumulativeLink", "CORAL"} <= methods


def test_render_example_summaries_ordinal_realdata_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["ordinal_regression_realdata_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "real-data" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"OrdinalCrossEntropy", "CumulativeLink", "CORAL"} <= methods


def test_render_example_summaries_ordinal_ugt_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["ordinal_uncertain_ground_truth_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "ordinal" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {
        "HardOrdinalCE",
        "SoftOrdinalCE",
        "SoftOrdinalCE+Pseudo",
        "SoftCumulativeLink",
    } <= methods


def test_render_example_summaries_censored_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["censored_regression_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "censored" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"CensoredGaussianNLL", "CensoredQuantile", "AFT"} <= methods


def test_render_example_summaries_censored_realdata_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["censored_regression_realdata_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "real-data" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"CensoredGaussianNLL", "CensoredQuantile", "AFT"} <= methods


def test_render_example_summaries_propensity_tail_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["propensity_tail_regression_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "selection bias" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"MSE", "DensityWeighted", "PropensityWeighted", "GaussianNLL", "Quantile90"} <= methods


def test_render_example_summaries_constraints_calibration_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["constraints_calibration_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "calibration transforms" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"Raw", "Calibrated+Constrained"} <= methods


def test_render_example_summaries_transformed_target_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["transformed_target_regression_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "target transforms" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"MSE", "LogTransform", "BoxCox(0.25)", "SqrtTransform"} <= methods


def test_render_example_summaries_semi_supervised_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["semi_supervised_regression_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "semi-supervised" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"SupervisedMSE", "PseudoLabelConsistency", "PseudoLabelNLL"} <= methods


def test_render_example_summaries_uncertain_gt_density_conformal_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["uncertain_gt_density_conformal_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "uncertain ground-truth" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {
        "SplitConformal",
        "DensityConformal",
        "PrevalenceAdjustedCP",
        "MonteCarloConformal",
    } <= methods


def test_render_example_summaries_uncertain_gt_density_conformal_realdata_subset(
    tmp_path: Path,
) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["uncertain_gt_density_conformal_realdata_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "real-data" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {
        "SplitConformal",
        "DensityConformal",
        "PrevalenceAdjustedCP",
        "MonteCarloConformal",
    } <= methods


def test_render_example_summaries_causal_dr_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["causal_dr_uplift_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "causal" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"Uplift-NaiveDiff", "Uplift-DRATE", "AstronomyBias-DRATE"} <= methods


def test_render_example_summaries_causal_dr_realdata_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["causal_dr_realdata_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "real covariates" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {
        "DiabetesProxy-NaiveDiff",
        "DiabetesProxy-DRATE",
        "DiabetesSelectionBias-DRATE",
    } <= methods
