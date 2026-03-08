from __future__ import annotations

import json
from pathlib import Path

from tools import photoz_rail_pipeline


def test_photoz_rail_pipeline_end_to_end_without_manual_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import photoz_rail_assets

    def mock_download(url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if "train.parquet" in url:
            target.write_bytes(b"train")
        elif "test.parquet" in url:
            target.write_bytes(b"test")
        elif "cal.parquet" in url or "calibration.parquet" in url:
            target.write_bytes(b"cal")
        elif ".json" in url:
            method = url.split("/")[-1].replace(".json", "")
            target.write_text(
                json.dumps(
                    {
                        "artifact": "rail_photoz_summary",
                        "dataset_id": "dset",
                        "split_id": "split",
                        "rows": [{"Method": method, "NMAD": 0.05}],
                    }
                ),
                encoding="utf-8",
            )
        else:
            target.write_bytes(b"dummy")

    monkeypatch.setattr(photoz_rail_assets, "_download_to_path", mock_download)

    dataset_src = tmp_path / "dataset_src"
    baseline_src = tmp_path / "baseline_src"
    output_dir = tmp_path / "reports/example_summaries"
    dataset_src.mkdir(parents=True)
    baseline_src.mkdir(parents=True)

    train_src = dataset_src / "train.parquet"
    test_src = dataset_src / "test.parquet"
    cal_src = dataset_src / "cal.parquet"

    dataset_id = "dset"
    split_id = "split"
    method_files = {
        "flexzboost": baseline_src / "flexzboost.json",
        "pzflow": baseline_src / "pzflow.json",
        "delight": baseline_src / "delight.json",
        "bpz": baseline_src / "bpz.json",
    }

    tr_summary = tmp_path / "precomputed_tr_summary.json"
    tr_summary.write_text(
        json.dumps(
            {
                "artifact": "comparison_example_summary",
                "version": 1,
                "rows": [{"Method": "BinnedCE", "NMAD": 0.04}],
            }
        ),
        encoding="utf-8",
    )

    materialized_train = tmp_path / "data/rail/datasets/train.parquet"
    materialized_test = tmp_path / "data/rail/datasets/test.parquet"
    materialized_cal = tmp_path / "data/rail/datasets/calibration.parquet"
    manifest = {
        "artifact": "rail_photoz_manifest",
        "version": 1,
        "dataset_id": dataset_id,
        "split_id": split_id,
        "core_baselines": ["flexzboost", "pzflow", "delight", "bpz"],
        "optional_baselines": ["lephare"],
        "dataset_files": [
            {
                "key": "train_catalog_sha256",
                "path": str(materialized_train),
                "url": f"http://localhost/{train_src.name}",
                "required": True,
            },
            {
                "key": "test_catalog_sha256",
                "path": str(materialized_test),
                "url": f"http://localhost/{test_src.name}",
                "required": True,
            },
            {
                "key": "calibration_catalog_sha256",
                "path": str(materialized_cal),
                "url": f"http://localhost/{cal_src.name}",
                "required": True,
            },
        ],
        "baseline_payloads": [
            {
                "method": method,
                "path": str(tmp_path / f"data/rail/baselines/{method}.json"),
                "url": f"http://localhost/{path.name}",
                "required": True,
            }
            for method, path in method_files.items()
        ],
        "checksum_policy": {
            "train_catalog_sha256": "",
            "test_catalog_sha256": "",
            "calibration_catalog_sha256": "",
        },
    }
    manifest_path = tmp_path / "rail_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = photoz_rail_pipeline.run_pipeline(
        manifest_path=manifest_path,
        output_dir=output_dir,
        profile="full",
        torchregress_summary_path=tr_summary,
        merged_output_path=None,
        render_torchregress_summary=False,
        allow_download=True,
        overwrite_downloads=False,
        strict_checksums=False,
        paper_parity=True,
        write_manifest_path=manifest_path,
        materialization_report_path=tmp_path / "materialization.json",
    )

    assert report["artifact"] == "photoz_rail_pipeline_report"
    merge_analysis = report.get("merge_analysis")
    assert isinstance(merge_analysis, dict)
    assert merge_analysis["n_rail_rows"] == 4
    assert merge_analysis["n_torchregress_rows"] == 1
    merged_path = Path(report["merged_output_path"])
    assert merged_path.exists()
    payload = json.loads(merged_path.read_text(encoding="utf-8"))
    methods = {row["Method"] for row in payload["rows"]}
    assert {"BinnedCE", "flexzboost", "pzflow", "delight", "bpz"} <= methods
    assert Path(report["manifest_path"]).exists()
    assert Path(tmp_path / "materialization.json").exists()


def test_photoz_rail_pipeline_supports_override_only_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import photoz_rail_assets

    def mock_download(url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if "train.parquet" in url:
            target.write_bytes(b"train")
        elif "test.parquet" in url:
            target.write_bytes(b"test")
        elif "cal.parquet" in url or "calibration.parquet" in url:
            target.write_bytes(b"cal")
        elif ".json" in url:
            method = url.split("/")[-1].replace(".json", "")
            target.write_text(
                json.dumps(
                    {
                        "artifact": "rail_photoz_summary",
                        "dataset_id": "dset",
                        "split_id": "split",
                        "rows": [{"Method": method, "NMAD": 0.05}],
                    }
                ),
                encoding="utf-8",
            )
        else:
            target.write_bytes(b"dummy")

    monkeypatch.setattr(photoz_rail_assets, "_download_to_path", mock_download)

    dataset_src = tmp_path / "dataset_src"
    baseline_src = tmp_path / "baseline_src"
    dataset_src.mkdir(parents=True)
    baseline_src.mkdir(parents=True)

    train_src = dataset_src / "train.parquet"
    test_src = dataset_src / "test.parquet"
    cal_src = dataset_src / "calibration.parquet"

    dataset_id = "dset"
    split_id = "split"

    tr_summary = tmp_path / "tr_summary.json"
    tr_summary.write_text(
        json.dumps(
            {
                "artifact": "comparison_example_summary",
                "version": 1,
                "rows": [{"Method": "BinnedCE", "NMAD": 0.04}],
            }
        ),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "legacy_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact": "rail_photoz_manifest",
                "version": 1,
                "dataset_id": dataset_id,
                "split_id": split_id,
                "core_baselines": ["flexzboost", "pzflow", "delight", "bpz"],
                "optional_baselines": ["lephare"],
                "checksum_policy": {},
            }
        ),
        encoding="utf-8",
    )

    report = photoz_rail_pipeline.run_pipeline(
        manifest_path=manifest_path,
        output_dir=tmp_path / "reports/example_summaries",
        profile="full",
        torchregress_summary_path=tr_summary,
        render_torchregress_summary=False,
        allow_download=True,
        dataset_urls={
            "train_catalog_sha256": f"http://localhost/{train_src.name}",
            "test_catalog_sha256": f"http://localhost/{test_src.name}",
            "calibration_catalog_sha256": f"http://localhost/{cal_src.name}",
        },
        baseline_urls={
            method: f"http://localhost/{method}.json"
            for method in ("flexzboost", "pzflow", "delight", "bpz")
        },
        strict_checksums=False,
        paper_parity=True,
    )

    merged = Path(report["merged_output_path"])
    assert merged.exists()
    merge_analysis = report.get("merge_analysis")
    assert isinstance(merge_analysis, dict)
    assert merge_analysis["n_rail_rows"] == 4
    payload = json.loads(merged.read_text(encoding="utf-8"))
    methods = {row["Method"] for row in payload["rows"]}
    assert {"BinnedCE", "flexzboost", "pzflow", "delight", "bpz"} <= methods


def test_photoz_rail_pipeline_can_bootstrap_missing_manifest_with_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import photoz_rail_assets

    def mock_download(url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if "train.parquet" in url:
            target.write_bytes(b"train")
        elif "test.parquet" in url:
            target.write_bytes(b"test")
        elif "cal.parquet" in url or "calibration.parquet" in url:
            target.write_bytes(b"cal")
        elif ".json" in url:
            method = url.split("/")[-1].replace(".json", "")
            target.write_text(
                json.dumps(
                    {
                        "artifact": "rail_photoz_summary",
                        "dataset_id": "nnc_crps_photoz_paper_reference_dataset",
                        "split_id": "nnc_crps_photoz_paper_reference_split_v1",
                        "rows": [{"Method": method, "NMAD": 0.05}],
                    }
                ),
                encoding="utf-8",
            )
        else:
            target.write_bytes(b"dummy")

    monkeypatch.setattr(photoz_rail_assets, "_download_to_path", mock_download)

    dataset_src = tmp_path / "dataset_src"
    baseline_src = tmp_path / "baseline_src"
    dataset_src.mkdir(parents=True)
    baseline_src.mkdir(parents=True)

    train_src = dataset_src / "train.parquet"
    test_src = dataset_src / "test.parquet"
    cal_src = dataset_src / "calibration.parquet"

    tr_summary = tmp_path / "tr_summary.json"
    tr_summary.write_text(
        json.dumps(
            {
                "artifact": "comparison_example_summary",
                "version": 1,
                "rows": [{"Method": "BinnedCE", "NMAD": 0.04}],
            }
        ),
        encoding="utf-8",
    )

    missing_manifest = tmp_path / "manifest_missing.json"
    report = photoz_rail_pipeline.run_pipeline(
        manifest_path=missing_manifest,
        output_dir=tmp_path / "reports/example_summaries",
        profile="full",
        torchregress_summary_path=tr_summary,
        render_torchregress_summary=False,
        allow_download=True,
        dataset_urls={
            "train_catalog_sha256": f"http://localhost/{train_src.name}",
            "test_catalog_sha256": f"http://localhost/{test_src.name}",
            "calibration_catalog_sha256": f"http://localhost/{cal_src.name}",
        },
        baseline_urls={
            method: f"http://localhost/{method}.json"
            for method in ("flexzboost", "pzflow", "delight", "bpz")
        },
        strict_checksums=False,
        paper_parity=True,
        preset="nnc_crps",
    )

    assert missing_manifest.exists()
    manifest_payload = json.loads(missing_manifest.read_text(encoding="utf-8"))
    assert manifest_payload["dataset_id"] == "nnc_crps_photoz_paper_reference_dataset"
    assert manifest_payload["split_id"] == "nnc_crps_photoz_paper_reference_split_v1"
    merged = Path(report["merged_output_path"])
    assert merged.exists()
    merge_analysis = report.get("merge_analysis")
    assert isinstance(merge_analysis, dict)
    assert merge_analysis["n_rail_rows"] == 4
