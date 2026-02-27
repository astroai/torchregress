from __future__ import annotations

import json
from pathlib import Path

from tools import compare_example_summary_profiles


def _write_payload(
    path: Path,
    *,
    task: str,
    methods: list[dict[str, object]],
    cfg: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact": "comparison_example_summary",
                "version": 1,
                "example": "examples/dummy.py",
                "task": task,
                "config": cfg,
                "rows": methods,
                "notes": [],
            }
        ),
        encoding="utf-8",
    )


def test_compare_profiles_reports_method_and_budget_consistency(tmp_path: Path) -> None:
    for (
        example_name,
        spec,
    ) in compare_example_summary_profiles.render_example_summaries.EXAMPLE_SPECS.items():
        stem = spec["filename"]
        _write_payload(
            tmp_path / f"{stem}_audit.json",
            task=example_name,
            cfg={"epochs": 10, "n_train": 100},
            methods=[
                {"Method": "A", "MSE": 1.0, "train_s": 1.0},
                {"Method": "B", "MSE": 2.0, "train_s": 2.0},
            ],
        )
        _write_payload(
            tmp_path / f"{stem}_full.json",
            task=example_name,
            cfg={"epochs": 20, "n_train": 200},
            methods=[
                {"Method": "A", "MSE": 0.8, "train_s": 2.0, "eval_s": 0.2},
                {"Method": "B", "MSE": 1.5, "train_s": 3.0, "eval_s": 0.3},
            ],
        )

    report = compare_example_summary_profiles.compare_profiles(
        base_dir=tmp_path,
        source_profile="audit",
        target_profile="full",
    )
    assert report["artifact"] == "example_summary_profile_comparison"
    assert report["ok"] is True
    for row in report["rows"]:
        assert row["method_set_equal"] is True
        assert row["numeric_key_coverage_ok"] is True
        assert row["source_semantic_ok"] is True
        assert row["target_semantic_ok"] is True
        assert row["directionality_ok"] is True
        assert row["runtime_scaling_ok"] is True
        assert "epochs" in row["monotone_budget_keys"]


def test_compare_profiles_detects_semantic_domain_issue(tmp_path: Path) -> None:
    for (
        example_name,
        spec,
    ) in compare_example_summary_profiles.render_example_summaries.EXAMPLE_SPECS.items():
        stem = spec["filename"]
        _write_payload(
            tmp_path / f"{stem}_audit.json",
            task=example_name,
            cfg={"epochs": 10, "n_train": 100},
            methods=[{"Method": "A", "ConformalCov90": 0.8, "train_s": 1.0}],
        )
        _write_payload(
            tmp_path / f"{stem}_full.json",
            task=example_name,
            cfg={"epochs": 20, "n_train": 200},
            methods=[{"Method": "A", "ConformalCov90": 1.2, "train_s": 2.0}],
        )

    report = compare_example_summary_profiles.compare_profiles(
        base_dir=tmp_path,
        source_profile="audit",
        target_profile="full",
    )
    assert report["ok"] is False
    assert any(row["target_semantic_ok"] is False for row in report["rows"])


def test_compare_profiles_detects_catastrophic_directionality_regression(tmp_path: Path) -> None:
    for (
        example_name,
        spec,
    ) in compare_example_summary_profiles.render_example_summaries.EXAMPLE_SPECS.items():
        stem = spec["filename"]
        _write_payload(
            tmp_path / f"{stem}_audit.json",
            task=example_name,
            cfg={"epochs": 10, "n_train": 100},
            methods=[{"Method": "A", "MSE": 1.0, "ConformalCov90": 0.85, "train_s": 1.0}],
        )
        _write_payload(
            tmp_path / f"{stem}_full.json",
            task=example_name,
            cfg={"epochs": 20, "n_train": 200},
            methods=[{"Method": "A", "MSE": 20.0, "ConformalCov90": 0.84, "train_s": 2.0}],
        )

    report = compare_example_summary_profiles.compare_profiles(
        base_dir=tmp_path,
        source_profile="audit",
        target_profile="full",
    )
    assert report["ok"] is False
    assert any(row["directionality_ok"] is False for row in report["rows"])


def test_compare_profiles_detects_ood_uncertainty_gap_mismatch(tmp_path: Path) -> None:
    for (
        example_name,
        spec,
    ) in compare_example_summary_profiles.render_example_summaries.EXAMPLE_SPECS.items():
        stem = spec["filename"]
        task = "OOD robustness / selective prediction" if "ood" in stem else example_name
        _write_payload(
            tmp_path / f"{stem}_audit.json",
            task=task,
            cfg={"epochs": 10, "n_train": 100},
            methods=[
                {"Method": "A", "MSE_ID": 1.0, "MSE_OOD": 1.4, "ood_unc_gap": 0.1, "train_s": 1.0}
            ],
        )
        _write_payload(
            tmp_path / f"{stem}_full.json",
            task=task,
            cfg={"epochs": 20, "n_train": 200},
            methods=[
                {
                    "Method": "A",
                    "MSE_ID": 1.0,
                    "MSE_OOD": 3.0,
                    "ood_unc_gap": -0.3,
                    "train_s": 2.0,
                }
            ],
        )

    report = compare_example_summary_profiles.compare_profiles(
        base_dir=tmp_path,
        source_profile="audit",
        target_profile="full",
    )
    assert report["ok"] is False
    assert any((not row["directionality_ok"]) and "ood" in row["example"] for row in report["rows"])


def test_compare_profiles_detects_multimodal_nll_mce_blowup(tmp_path: Path) -> None:
    for (
        example_name,
        spec,
    ) in compare_example_summary_profiles.render_example_summaries.EXAMPLE_SPECS.items():
        stem = spec["filename"]
        task = "Multimodal / multi-target non-Gaussian" if "multimodal" in stem else example_name
        _write_payload(
            tmp_path / f"{stem}_audit.json",
            task=task,
            cfg={"epochs": 10, "n_train": 100},
            methods=[{"Method": "A", "NLL": 1.2, "MCE": 0.1, "train_s": 1.0}],
        )
        _write_payload(
            tmp_path / f"{stem}_full.json",
            task=task,
            cfg={"epochs": 20, "n_train": 200},
            methods=[{"Method": "A", "NLL": 25.0, "MCE": 1.5, "train_s": 2.0}],
        )

    report = compare_example_summary_profiles.compare_profiles(
        base_dir=tmp_path,
        source_profile="audit",
        target_profile="full",
    )
    assert report["ok"] is False
    assert any(
        (not row["directionality_ok"]) and "multimodal" in row["example"] for row in report["rows"]
    )


def test_compare_profiles_detects_noisy_label_coverage_collapse_mismatch(tmp_path: Path) -> None:
    for (
        example_name,
        spec,
    ) in compare_example_summary_profiles.render_example_summaries.EXAMPLE_SPECS.items():
        stem = spec["filename"]
        task = "Noisy labels / corruption" if "noisy_label" in stem else example_name
        _write_payload(
            tmp_path / f"{stem}_audit.json",
            task=task,
            cfg={"epochs": 10, "n_train": 100},
            methods=[
                {
                    "Method": "A",
                    "CleanMSE": 0.4,
                    "ObsMSE": 0.7,
                    "ConformalCov90": 0.88,
                    "NativeCov90": 0.83,
                    "ConformalWidth90": 2.2,
                    "NativeWidth90": 2.0,
                    "train_s": 1.0,
                }
            ],
        )
        _write_payload(
            tmp_path / f"{stem}_full.json",
            task=task,
            cfg={"epochs": 20, "n_train": 200},
            methods=[
                {
                    "Method": "A",
                    "CleanMSE": 0.5,
                    "ObsMSE": 1.5,
                    "ConformalCov90": 0.25,
                    "NativeCov90": 0.3,
                    "ConformalWidth90": 2.1,
                    "NativeWidth90": 1.95,
                    "train_s": 2.0,
                }
            ],
        )

    report = compare_example_summary_profiles.compare_profiles(
        base_dir=tmp_path,
        source_profile="audit",
        target_profile="full",
    )
    assert report["ok"] is False
    assert any(
        (not row["directionality_ok"]) and "noisy_label" in row["example"] for row in report["rows"]
    )


def test_compare_profiles_detects_photoz_quality_blowup(tmp_path: Path) -> None:
    for (
        example_name,
        spec,
    ) in compare_example_summary_profiles.render_example_summaries.EXAMPLE_SPECS.items():
        stem = spec["filename"]
        task = (
            "Photometric redshift benchmark (photo-z, SDSS-style)"
            if "photoz_benchmark" in stem
            else example_name
        )
        _write_payload(
            tmp_path / f"{stem}_audit.json",
            task=task,
            cfg={"epochs": 10, "n_train": 100},
            methods=[
                {
                    "Method": "A",
                    "NMAD": 0.05,
                    "CatastrophicRate": 0.10,
                    "HighZ_MAE": 0.2,
                    "train_s": 1.0,
                }
            ],
        )
        _write_payload(
            tmp_path / f"{stem}_full.json",
            task=task,
            cfg={"epochs": 20, "n_train": 200},
            methods=[
                {
                    "Method": "A",
                    "NMAD": 0.25,
                    "CatastrophicRate": 0.70,
                    "HighZ_MAE": 0.8,
                    "train_s": 2.0,
                }
            ],
        )

    report = compare_example_summary_profiles.compare_profiles(
        base_dir=tmp_path,
        source_profile="audit",
        target_profile="full",
    )
    assert report["ok"] is False
    assert any(
        (not row["directionality_ok"]) and "photoz_benchmark_comparison" in row["example"]
        for row in report["rows"]
    )


def test_compare_profiles_detects_photoz_pdf_quality_blowup(tmp_path: Path) -> None:
    for (
        example_name,
        spec,
    ) in compare_example_summary_profiles.render_example_summaries.EXAMPLE_SPECS.items():
        stem = spec["filename"]
        task = (
            "Photometric redshift benchmark (photo-z, ordered-bin NNC-CRPS style)"
            if "photoz_nnc_crps_rail" in stem
            else example_name
        )
        if "photoz_nnc_crps_rail" in stem:
            audit_row = {
                "Method": "A",
                "CRPS": 0.05,
                "PDF_NLL": 0.2,
                "NMAD": 0.04,
                "CatastrophicRate": 0.12,
                "HighZ_MAE": 0.2,
                "train_s": 1.0,
            }
            full_row = {
                "Method": "A",
                "CRPS": 0.4,
                "PDF_NLL": 1.2,
                "NMAD": 0.05,
                "CatastrophicRate": 0.14,
                "HighZ_MAE": 0.22,
                "train_s": 2.0,
            }
        else:
            audit_row = {"Method": "A", "MSE": 1.0, "train_s": 1.0}
            full_row = {"Method": "A", "MSE": 0.9, "train_s": 2.0}
        _write_payload(
            tmp_path / f"{stem}_audit.json",
            task=task,
            cfg={"epochs": 10, "n_train": 100},
            methods=[audit_row],
        )
        _write_payload(
            tmp_path / f"{stem}_full.json",
            task=task,
            cfg={"epochs": 20, "n_train": 200},
            methods=[full_row],
        )

    report = compare_example_summary_profiles.compare_profiles(
        base_dir=tmp_path,
        source_profile="audit",
        target_profile="full",
    )
    assert report["ok"] is False
    assert any(
        (not row["directionality_ok"]) and "photoz_nnc_crps_rail_comparison" in row["example"]
        for row in report["rows"]
    )


def test_compare_profiles_detects_ordinal_quality_blowup(tmp_path: Path) -> None:
    for (
        example_name,
        spec,
    ) in compare_example_summary_profiles.render_example_summaries.EXAMPLE_SPECS.items():
        stem = spec["filename"]
        task = "Ordinal regression / ordered targets" if "ordinal" in stem else example_name
        if "ordinal" in stem:
            audit_row = {
                "Method": "A",
                "Accuracy": 0.85,
                "OrdinalMAE": 0.2,
                "QWK": 0.8,
                "train_s": 1.0,
            }
            full_row = {
                "Method": "A",
                "Accuracy": 0.2,
                "OrdinalMAE": 2.2,
                "QWK": -0.1,
                "train_s": 2.0,
            }
        else:
            audit_row = {"Method": "A", "MSE": 1.0, "train_s": 1.0}
            full_row = {"Method": "A", "MSE": 0.9, "train_s": 2.0}
        _write_payload(
            tmp_path / f"{stem}_audit.json",
            task=task,
            cfg={"epochs": 10, "n_train": 100},
            methods=[audit_row],
        )
        _write_payload(
            tmp_path / f"{stem}_full.json",
            task=task,
            cfg={"epochs": 20, "n_train": 200},
            methods=[full_row],
        )

    report = compare_example_summary_profiles.compare_profiles(
        base_dir=tmp_path,
        source_profile="audit",
        target_profile="full",
    )
    assert report["ok"] is False
    assert any(
        (not row["directionality_ok"]) and "ordinal_regression_comparison" in row["example"]
        for row in report["rows"]
    )
