from __future__ import annotations

import json
from pathlib import Path

from tools import photoz_transferz_pipeline


def test_photoz_transferz_pipeline_runs_suite(monkeypatch, tmp_path: Path) -> None:
    normalized_dir = tmp_path / "normalized"
    raw_dir = tmp_path / "raw"
    suite_output_dir = tmp_path / "reports"
    report_path = tmp_path / "transferz_pipeline.json"
    suite_report_path = tmp_path / "transferz_suite.json"
    markdown_report_path = tmp_path / "transferz_suite.md"

    collection_payload = {
        "artifact": "photoz_transferz_collection_report",
        "normalized_paths": {},
    }
    header = "spec_z,spec_z_err,g_r,r_i,i_z,z_y,g_r_err,r_i_err,i_z_err,z_y_err\n"
    for split in ("train", "cal", "test", "conformal"):
        path = normalized_dir / f"transferz_{split}_photoz.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header, encoding="utf-8")
        collection_payload["normalized_paths"][split] = str(path)

    def _fake_collect(**kwargs):
        raw_dir.mkdir(parents=True, exist_ok=True)
        return collection_payload

    def _fake_run_suite(**kwargs):
        summary_path = suite_output_dir / "photoz_benchmark_comparison_smoke.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"artifact": "comparison_example_summary", "rows": [{"Method": "Huber"}]}),
            encoding="utf-8",
        )
        return {
            "artifact": "photoz_benchmark_suite_report",
            "summary_paths": {"photoz_benchmark_comparison": str(summary_path)},
            "recommended_read_order": ["photoz_benchmark_comparison"],
        }

    def _fake_run_conformal_example(**kwargs):
        path = suite_output_dir / "photoz_transferz_conformal_comparison_smoke.json"
        path.write_text(
            json.dumps(
                {"artifact": "comparison_example_summary", "rows": [{"Method": "SplitConformal"}]}
            ),
            encoding="utf-8",
        )
        return path

    monkeypatch.setattr(
        photoz_transferz_pipeline.photoz_collect_real_data,
        "collect_transferz_splits",
        _fake_collect,
    )
    monkeypatch.setattr(
        photoz_transferz_pipeline.photoz_benchmark_suite,
        "run_suite",
        _fake_run_suite,
    )
    monkeypatch.setattr(
        photoz_transferz_pipeline,
        "_run_conformal_example",
        _fake_run_conformal_example,
    )

    report = photoz_transferz_pipeline.run_pipeline(
        profile="smoke",
        raw_output_dir=raw_dir,
        normalized_output_dir=normalized_dir,
        suite_output_dir=suite_output_dir,
        suite_report_path=suite_report_path,
        markdown_report_path=markdown_report_path,
        report_path=report_path,
        download_if_missing=True,
    )

    assert report["artifact"] == "photoz_transferz_pipeline_report"
    assert report["split_policy"]["conformal_reserved"].endswith("transferz_conformal_photoz.csv")
    assert report["conformal_summary_path"].endswith(
        "photoz_transferz_conformal_comparison_smoke.json"
    )
    assert suite_report_path.exists()
    assert report_path.exists()
