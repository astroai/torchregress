from __future__ import annotations

import json
from pathlib import Path

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
    assert report_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summaries = payload["summaries"]
    assert isinstance(summaries, dict)
    assert payload["include_photoz"] is False
    assert set(summaries) == {"synthetic", "tabular_small", "tabular_large"}
    for key in ("synthetic", "tabular_small", "tabular_large"):
        summary_path = Path(str(summaries[key]))
        assert summary_path.exists()
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary_payload["artifact"] == "comparison_example_summary"
        methods = {row["Method"] for row in summary_payload["rows"]}
        assert "SPTRegGaussian" in methods
        assert "SPTRegBinnedPDF" in methods


def test_render_spt_reg_paper_artifacts_can_include_photoz(tmp_path: Path) -> None:
    report_path = tmp_path / "artifact_manifest_photoz.json"
    report = render_spt_reg_paper_artifacts.run_render(
        profile="smoke",
        include_photoz=True,
        output_dir=tmp_path,
        report_path=report_path,
    )

    assert report["include_photoz"] is True
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summaries = payload["summaries"]
    assert isinstance(summaries, dict)
    assert set(summaries) == {"synthetic", "tabular_small", "tabular_large", "photoz"}
    photoz_path = Path(str(summaries["photoz"]))
    assert photoz_path.exists()
    photoz_payload = json.loads(photoz_path.read_text(encoding="utf-8"))
    methods = {row["Method"] for row in photoz_payload["rows"]}
    assert "SPTRegGaussian" in methods
    assert "SPTRegBinnedPDF" in methods
