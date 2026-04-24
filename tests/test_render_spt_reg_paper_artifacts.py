from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import render_spt_reg_paper_artifacts


def test_render_spt_reg_paper_artifacts_smoke(tmp_path: Path) -> None:
    report_path = tmp_path / "artifact_manifest.json"
    report = render_spt_reg_paper_artifacts.run_render(
        profile="smoke",
        output_dir=tmp_path,
        report_path=report_path,
    )

    assert report["artifact"] == "spt_reg_paper_artifact_manifest"
    assert report["profile"] == "smoke"
    assert "synthetic_local" in str(report.get("year_track_data", ""))
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summaries = payload["summaries"]
    assert isinstance(summaries, dict)
    assert set(summaries) == {"synthetic", "tabular_small", "tabular_large"}
    for key in ("synthetic", "tabular_small", "tabular_large"):
        summary_path = Path(str(summaries[key]))
        assert summary_path.exists()
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary_payload["artifact"] == "comparison_example_summary"
        methods = {row["Method"] for row in summary_payload["rows"]}
        assert "SPTRegGaussian" in methods
        assert "SPTRegBinnedPDF" in methods


def test_render_spt_reg_paper_artifacts_year_external_dataset(tmp_path: Path) -> None:
    """Large-tabular track uses real path when year_dataset_path is passed to run_render."""
    rng = np.random.default_rng(0)
    n_rows = 400
    frame = pd.DataFrame({f"f{i}": rng.normal(size=n_rows).astype("float32") for i in range(8)})
    frame["target"] = rng.normal(size=n_rows).astype("float32")
    year_csv = tmp_path / "external_year.csv"
    frame.to_csv(year_csv, index=False)

    report_path = tmp_path / "manifest_ext.json"
    report = render_spt_reg_paper_artifacts.run_render(
        profile="smoke",
        output_dir=tmp_path,
        report_path=report_path,
        year_dataset_path=year_csv,
    )
    assert "external_year.csv" in str(report["year_track_data"])
    year_summary = json.loads(
        Path(str(report["summaries"]["tabular_large"])).read_text(encoding="utf-8")
    )
    assert "external_year.csv" in year_summary["rows"][0].get("Notes", "")


def test_run_render_rejects_both_year_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at most one"):
        render_spt_reg_paper_artifacts.run_render(
            profile="smoke",
            output_dir=tmp_path,
            report_path=tmp_path / "m.json",
            year_cache_path=tmp_path / "a.csv",
            year_dataset_path=tmp_path / "b.csv",
        )


def test_run_render_rejects_openml_with_year_dataset_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        render_spt_reg_paper_artifacts.run_render(
            profile="smoke",
            output_dir=tmp_path,
            report_path=tmp_path / "m.json",
            year_dataset_path=tmp_path / "y.csv",
            year_openml_data_id=42225,
        )
