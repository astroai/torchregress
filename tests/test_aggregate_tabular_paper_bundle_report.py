from __future__ import annotations

import json
from pathlib import Path

from tools.aggregate_tabular_paper_bundle_report import build_report


def test_build_tabular_paper_bundle_report_roundtrip(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "sage/year_direct").mkdir(parents=True)
    (bundle / "sage/multiseed").mkdir(parents=True)
    (bundle / "spt/full").mkdir(parents=True)

    (bundle / "bundle_meta.json").write_text(
        json.dumps({"spt_profile": "smoke"}, indent=2),
        encoding="utf-8",
    )
    (bundle / "sage/year_direct/summary.json").write_text(
        json.dumps(
            {
                "artifact": "comparison_example_summary",
                "version": 1,
                "example": "year",
                "task": "t",
                "config": {},
                "rows": [
                    {
                        "Method": "SAGE-Reg",
                        "UnlabeledFraction": 1.0,
                        "NLL": 1.2,
                        "CRPS": 0.5,
                        "Cov90": 0.9,
                        "CalibMAE": 0.1,
                        "RMSE": 1.0,
                    }
                ],
                "notes": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (bundle / "sage/multiseed/multiseed_summary.json").write_text(
        json.dumps(
            {
                "aggregate_rows": [
                    {
                        "Benchmark": "year",
                        "Seeds": 3,
                        "SAGEMinusSupervisedMean": -0.1,
                        "SAGEMinusSupervisedStd": 0.05,
                        "ConfidenceMinusSupervisedMean": 0.2,
                    }
                ],
                "seed_rows": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (bundle / "spt/full/year_competing_methods_smoke.json").write_text(
        json.dumps(
            {
                "artifact": "comparison_example_summary",
                "rows": [
                    {
                        "Method": "SPTRegGaussian",
                        "NLL": 1.0,
                        "CRPS": 0.4,
                        "Cov90": 0.95,
                        "Width90": 2.0,
                        "MSE": 0.1,
                        "MAE": 0.2,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (bundle / "spt/full/artifact_manifest.json").write_text("{}", encoding="utf-8")

    report = build_report(bundle)
    assert report["artifact"] == "tabular_paper_bundle_report"
    assert report["paths"]["spt_profile"] == "smoke"
    assert report["sage_year_direct"]["by_method"]["SAGE-Reg"][0]["NLL"] == 1.2
    assert report["sage_multiseed"]["aggregate"][0]["Benchmark"] == "year"
    assert "SPTRegGaussian" in report["spt_year_full"]["methods"]
