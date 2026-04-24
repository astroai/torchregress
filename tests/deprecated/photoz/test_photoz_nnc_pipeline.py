from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tools import photoz_nnc_pipeline


def test_normalize_nnc_catalog_from_csv(tmp_path: Path) -> None:
    raw = tmp_path / "nnc_raw.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "redshift": [0.1, 0.2, 0.3, 0.4],
            "redshift_err": [0.01, 0.01, 0.02, 0.02],
            "u_mag": [22.1, 22.2, 22.3, 22.4],
            "g_mag": [21.7, 21.8, 21.9, 22.0],
            "r_mag": [21.2, 21.3, 21.4, 21.5],
            "i_mag": [20.8, 20.9, 21.0, 21.1],
            "z_mag": [20.4, 20.5, 20.6, 20.7],
            "u_magerr": [0.03, 0.03, 0.03, 0.03],
            "g_magerr": [0.02, 0.02, 0.02, 0.02],
            "r_magerr": [0.02, 0.02, 0.02, 0.02],
            "i_magerr": [0.02, 0.02, 0.02, 0.02],
            "z_magerr": [0.03, 0.03, 0.03, 0.03],
        }
    ).to_csv(raw, index=False)

    out = tmp_path / "nnc_photoz_real.csv"
    report = photoz_nnc_pipeline.normalize_nnc_catalog(
        raw_catalog_path=raw,
        output_path=out,
    )

    normalized = pd.read_csv(out)
    assert report["rows_normalized"] == 4
    assert "spec_z" in normalized.columns
    assert "u_g" in normalized.columns
    assert "g_r" in normalized.columns
    assert "r_i" in normalized.columns
    assert "i_z" in normalized.columns


def test_photoz_nnc_pipeline_runs_suite(monkeypatch, tmp_path: Path) -> None:
    raw = tmp_path / "nnc_raw.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "z_true": [0.1, 0.2, 0.3, 0.4],
            "u": [22.1, 22.2, 22.3, 22.4],
            "g": [21.7, 21.8, 21.9, 22.0],
            "r": [21.2, 21.3, 21.4, 21.5],
            "i": [20.8, 20.9, 21.0, 21.1],
            "z_mag": [20.4, 20.5, 20.6, 20.7],
            "u_err": [0.03, 0.03, 0.03, 0.03],
            "g_err": [0.02, 0.02, 0.02, 0.02],
            "r_err": [0.02, 0.02, 0.02, 0.02],
            "i_err": [0.02, 0.02, 0.02, 0.02],
            "z_mag_err": [0.03, 0.03, 0.03, 0.03],
        }
    ).to_csv(raw, index=False)

    def _fake_run_suite(**kwargs):
        out_dir = kwargs["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        standard = out_dir / "photoz_benchmark_comparison_smoke.json"
        ordered = out_dir / "photoz_nnc_crps_rail_comparison_smoke.json"
        standard.write_text(json.dumps({"rows": [{"Method": "MSE"}]}), encoding="utf-8")
        ordered.write_text(json.dumps({"rows": [{"Method": "BinnedCE"}]}), encoding="utf-8")
        markdown_path = kwargs["markdown_report_path"]
        assert markdown_path is not None
        markdown_path.write_text("# report\n", encoding="utf-8")
        return {
            "artifact": "photoz_benchmark_suite_report",
            "summary_paths": {
                "photoz_benchmark_comparison": str(standard),
                "photoz_nnc_crps_rail_comparison": str(ordered),
            },
            "markdown_report_path": str(markdown_path),
        }

    monkeypatch.setattr(photoz_nnc_pipeline.photoz_benchmark_suite, "run_suite", _fake_run_suite)

    report = photoz_nnc_pipeline.run_pipeline(
        profile="smoke",
        output_dir=tmp_path / "reports",
        raw_catalog_path=raw,
        normalized_output_path=tmp_path / "nnc_photoz_real.csv",
        download_if_missing=False,
        record_id=0,
        preferred_suffix=".csv",
        suite_report_path=tmp_path / "reports" / "suite.json",
        markdown_report_path=tmp_path / "reports" / "suite.md",
    )

    assert Path(report["suite_report_path"]).exists()
    assert Path(report["suite_markdown_report_path"]).exists()
    normalized = Path(report["normalization_report"]["normalized_output_path"])
    assert normalized.exists()


def test_photoz_nnc_pipeline_rejects_too_small_local_catalog(tmp_path: Path) -> None:
    raw_dir = tmp_path / "catalogs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    small = raw_dir / "tiny.csv"
    pd.DataFrame(
        {
            "id": [1, 2],
            "redshift": [0.1, 0.2],
            "u_mag": [22.1, 22.2],
            "g_mag": [21.7, 21.8],
            "r_mag": [21.2, 21.3],
            "i_mag": [20.8, 20.9],
            "z_mag": [20.4, 20.5],
        }
    ).to_csv(small, index=False)

    old_dir = photoz_nnc_pipeline.DEFAULT_RAW_DIR
    photoz_nnc_pipeline.DEFAULT_RAW_DIR = raw_dir
    try:
        try:
            photoz_nnc_pipeline.run_pipeline(
                profile="smoke",
                output_dir=tmp_path / "reports",
                raw_catalog_path=None,
                normalized_output_path=tmp_path / "nnc_photoz_real.csv",
                download_if_missing=False,
                record_id=0,
                preferred_suffix=".csv",
                suite_report_path=tmp_path / "reports" / "suite.json",
                markdown_report_path=tmp_path / "reports" / "suite.md",
            )
        except FileNotFoundError as exc:
            assert "Need at least 112 rows" in str(exc)
        else:
            raise AssertionError("Expected insufficient-row FileNotFoundError")
    finally:
        photoz_nnc_pipeline.DEFAULT_RAW_DIR = old_dir
