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


def test_render_example_summaries_photoz_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["photoz_benchmark_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "photo-z" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {"MSE", "Huber", "GaussianNLL", "Quantile90", "FunctionalEIV"} <= methods


def test_render_example_summaries_photoz_nnc_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["photoz_nnc_crps_rail_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "ordered-bin" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {
        "BinnedCE",
        "BinnedCE+TempScaling",
        "OrderedBinCRPS",
        "OrderedBinCRPS+TempScaling",
        "GaussianNLL",
        "MultiQuantileLoss",
    } <= methods


def test_render_example_summaries_ppi_photoz_subset(tmp_path: Path) -> None:
    paths = render_example_summaries.render_all(
        profile="smoke",
        output_dir=tmp_path,
        examples=["ppi_photoz_inference_comparison"],
    )
    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert "inference" in payload["task"].lower()
    methods = {row["Method"] for row in payload["rows"]}
    assert {
        "LabeledOnlyMeanCI",
        "PPIMeanCI",
        "LabeledOnlyQuantileCI",
        "PPIQuantileCI",
    } <= methods


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
    assert {"MSE", "DensityWeighted", "PropensityWeighted"} <= methods


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


def test_render_photoz_rail_merge_helper(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    tr_summary = tmp_path / "tr_summary.json"
    rail_summary = tmp_path / "rail_summary.json"
    output = tmp_path / "merged.json"

    manifest.write_text(
        json.dumps(
            {
                "artifact": "rail_photoz_manifest",
                "dataset_id": "dset",
                "split_id": "split",
                "core_baselines": ["flexzboost", "pzflow", "delight", "bpz"],
                "optional_baselines": ["lephare"],
            }
        ),
        encoding="utf-8",
    )
    tr_summary.write_text(
        json.dumps(
            {
                "artifact": "comparison_example_summary",
                "version": 1,
                "rows": [{"Method": "BinnedCE", "NMAD": 0.05}],
            }
        ),
        encoding="utf-8",
    )
    rail_summary.write_text(
        json.dumps(
            {
                "artifact": "rail_photoz_summary",
                "dataset_id": "dset",
                "split_id": "split",
                "rows": [
                    {"Method": "flexzboost", "NMAD": 0.05},
                    {"Method": "pzflow", "NMAD": 0.06},
                    {"Method": "delight", "NMAD": 0.07},
                    {"Method": "bpz", "NMAD": 0.08},
                ],
            }
        ),
        encoding="utf-8",
    )

    merged = render_example_summaries.render_photoz_rail_merge(
        manifest_path=manifest,
        torchregress_summary_path=tr_summary,
        rail_input_paths=[rail_summary],
        output_path=output,
        paper_parity=True,
    )
    assert merged == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    methods = {row["Method"] for row in payload["rows"]}
    assert {"BinnedCE", "flexzboost", "pzflow", "delight", "bpz"} <= methods
