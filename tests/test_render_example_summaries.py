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
