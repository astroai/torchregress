from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import photoz_rail_assets


def test_materialize_manifest_assets_downloads_and_updates_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mock_download(url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if "train" in url:
            target.write_bytes(b"train-bytes")
        elif "test" in url:
            target.write_bytes(b"test-bytes")
        elif "cal" in url:
            target.write_bytes(b"cal-bytes")
        elif "json" in url:
            method = url.split("/")[-1].replace("src_", "").replace(".json", "")
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

    monkeypatch.setattr(photoz_rail_assets, "_download_to_path", mock_download)

    src_train = tmp_path / "src_train.parquet"
    src_test = tmp_path / "src_test.parquet"
    src_cal = tmp_path / "src_cal.parquet"
    src_flex = tmp_path / "src_flex.json"
    src_pzflow = tmp_path / "src_pzflow.json"
    src_delight = tmp_path / "src_delight.json"
    src_bpz = tmp_path / "src_bpz.json"

    src_train.write_bytes(b"train-bytes")
    src_test.write_bytes(b"test-bytes")
    src_cal.write_bytes(b"cal-bytes")
    for src, method in (
        (src_flex, "flexzboost"),
        (src_pzflow, "pzflow"),
        (src_delight, "delight"),
        (src_bpz, "bpz"),
    ):
        src.write_text(
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

    manifest: dict[str, object] = {
        "artifact": "rail_photoz_manifest",
        "version": 1,
        "dataset_id": "dset",
        "split_id": "split",
        "core_baselines": ["flexzboost", "pzflow", "delight", "bpz"],
        "optional_baselines": ["lephare"],
        "dataset_files": [
            {
                "key": "train_catalog_sha256",
                "path": "data/rail/datasets/train.parquet",
                "url": f"http://localhost/{src_train.name}",
                "required": True,
            },
            {
                "key": "test_catalog_sha256",
                "path": "data/rail/datasets/test.parquet",
                "url": f"http://localhost/{src_test.name}",
                "required": True,
            },
            {
                "key": "calibration_catalog_sha256",
                "path": "data/rail/datasets/calibration.parquet",
                "url": f"http://localhost/{src_cal.name}",
                "required": True,
            },
        ],
        "baseline_payloads": [
            {
                "method": "flexzboost",
                "path": "data/rail/baselines/flexzboost.json",
                "url": f"http://localhost/{src_flex.name}",
                "required": True,
            },
            {
                "method": "pzflow",
                "path": "data/rail/baselines/pzflow.json",
                "url": f"http://localhost/{src_pzflow.name}",
                "required": True,
            },
            {
                "method": "delight",
                "path": "data/rail/baselines/delight.json",
                "url": f"http://localhost/{src_delight.name}",
                "required": True,
            },
            {
                "method": "bpz",
                "path": "data/rail/baselines/bpz.json",
                "url": f"http://localhost/{src_bpz.name}",
                "required": True,
            },
            {
                "method": "lephare",
                "path": "data/rail/baselines/lephare.json",
                "url": "",
                "required": False,
            },
        ],
        "checksum_policy": {
            "train_catalog_sha256": "",
            "test_catalog_sha256": "",
            "calibration_catalog_sha256": "",
        },
    }

    report = photoz_rail_assets.materialize_manifest_assets(
        manifest,
        repo_root=tmp_path,
        allow_download=True,
        overwrite=False,
        strict_checksums=False,
        fail_on_missing=True,
    )
    assert report["artifact"] == "photoz_rail_materialization_report"
    assert not report["missing_required"]
    checksum_policy = manifest["checksum_policy"]
    assert isinstance(checksum_policy, dict)
    assert checksum_policy["train_catalog_sha256"]
    assert checksum_policy["test_catalog_sha256"]
    assert checksum_policy["calibration_catalog_sha256"]
    assert (tmp_path / "data/rail/datasets/train.parquet").exists()
    assert (tmp_path / "data/rail/baselines/flexzboost.json").exists()

    baseline_paths = photoz_rail_assets.collect_baseline_input_paths(
        manifest,
        repo_root=tmp_path,
        require_required=True,
    )
    names = {path.name for path in baseline_paths}
    assert {"flexzboost.json", "pzflow.json", "delight.json", "bpz.json"} <= names


def test_materialize_manifest_assets_strict_checksum_mismatch_raises(tmp_path: Path) -> None:
    target = tmp_path / "data/rail/datasets/train.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"current-data")
    manifest = {
        "artifact": "rail_photoz_manifest",
        "version": 1,
        "dataset_files": [
            {
                "key": "train_catalog_sha256",
                "path": str(target),
                "url": "",
                "required": True,
            }
        ],
        "baseline_payloads": [],
        "checksum_policy": {"train_catalog_sha256": "deadbeef"},
    }
    with pytest.raises(ValueError, match="Checksum mismatch"):
        photoz_rail_assets.materialize_manifest_assets(
            manifest,
            repo_root=tmp_path,
            allow_download=False,
            overwrite=False,
            strict_checksums=True,
            fail_on_missing=True,
        )


def test_manifest_defaults_are_inferred_for_legacy_schema() -> None:
    manifest: dict[str, object] = {
        "artifact": "rail_photoz_manifest",
        "version": 1,
        "core_baselines": ["flexzboost", "pzflow", "delight", "bpz"],
        "optional_baselines": ["lephare"],
        "checksum_policy": {},
    }
    photoz_rail_assets.ensure_manifest_defaults(manifest)

    dataset_files = manifest["dataset_files"]
    baseline_payloads = manifest["baseline_payloads"]
    assert isinstance(dataset_files, list) and len(dataset_files) == 3
    assert isinstance(baseline_payloads, list) and len(baseline_payloads) == 5
    required_methods = {
        entry["method"]
        for entry in baseline_payloads
        if isinstance(entry, dict) and entry.get("required") is True
    }
    assert {"flexzboost", "pzflow", "delight", "bpz"} <= required_methods


def test_apply_manifest_overrides_updates_urls_and_paths() -> None:
    manifest: dict[str, object] = {
        "artifact": "rail_photoz_manifest",
        "version": 1,
        "core_baselines": ["flexzboost", "pzflow", "delight", "bpz"],
        "optional_baselines": ["lephare"],
        "checksum_policy": {},
    }
    photoz_rail_assets.apply_manifest_overrides(
        manifest,
        dataset_urls={"train_catalog_sha256": "https://example.test/train.parquet"},
        baseline_urls={"flexzboost": "https://example.test/flexzboost.json"},
        dataset_paths={"train_catalog_sha256": "/tmp/train_override.parquet"},
        baseline_paths={"flexzboost": "/tmp/flexzboost_override.json"},
    )
    dataset_entries = manifest["dataset_files"]
    baseline_entries = manifest["baseline_payloads"]
    assert isinstance(dataset_entries, list)
    assert isinstance(baseline_entries, list)
    train_entry = next(
        entry
        for entry in dataset_entries
        if isinstance(entry, dict) and entry.get("key") == "train_catalog_sha256"
    )
    flex_entry = next(
        entry
        for entry in baseline_entries
        if isinstance(entry, dict) and entry.get("method") == "flexzboost"
    )
    assert train_entry["url"] == "https://example.test/train.parquet"
    assert train_entry["path"] == "/tmp/train_override.parquet"
    assert flex_entry["url"] == "https://example.test/flexzboost.json"
    assert flex_entry["path"] == "/tmp/flexzboost_override.json"


def test_load_manifest_bootstraps_from_template(tmp_path: Path) -> None:
    template = tmp_path / "template.json"
    template_payload = {
        "artifact": "rail_photoz_manifest",
        "version": 1,
        "dataset_id": "template_dataset",
        "split_id": "template_split",
    }
    template.write_text(json.dumps(template_payload), encoding="utf-8")

    loaded = photoz_rail_assets.load_manifest(
        tmp_path / "missing_manifest.json",
        preset="rail",
        template_path=template,
    )
    assert loaded["dataset_id"] == "template_dataset"
    assert loaded["split_id"] == "template_split"
