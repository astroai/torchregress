from __future__ import annotations

import json
from pathlib import Path

from tools import render_photoz_benchmark_report


def test_render_photoz_benchmark_report_writes_markdown(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports/example_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    standard = output_dir / "photoz_benchmark_comparison_smoke.json"
    standard.write_text(
        json.dumps(
            {
                "artifact": "comparison_example_summary",
                "rows": [
                    {"Method": "MSE", "NMAD": 0.2, "CatastrophicRate": 0.1, "HighZ_MAE": 0.3},
                    {
                        "Method": "GaussianNLL",
                        "NMAD": 0.1,
                        "CatastrophicRate": 0.08,
                        "HighZ_MAE": 0.2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    ordered = output_dir / "photoz_nnc_crps_rail_comparison_smoke.json"
    ordered.write_text(
        json.dumps(
            {
                "artifact": "comparison_example_summary",
                "rows": [
                    {"Method": "BinnedCE", "CRPS": 0.2, "PDF_NLL": 1.1, "PITChi2": 2.0},
                    {"Method": "SoftBinnedCE", "CRPS": 0.1, "PDF_NLL": 0.9, "PITChi2": 1.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    ppi = output_dir / "ppi_photoz_inference_comparison_smoke.json"
    ppi.write_text(
        json.dumps(
            {
                "artifact": "comparison_example_summary",
                "rows": [
                    {
                        "Method": "PPIMeanCI",
                        "Target": "mean",
                        "Estimate": 0.1,
                        "AbsError": 0.02,
                        "CIWidth": 0.2,
                        "CoversTruth": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    conformal = output_dir / "photoz_transferz_conformal_comparison_smoke.json"
    conformal.write_text(
        json.dumps(
            {
                "artifact": "comparison_example_summary",
                "rows": [
                    {
                        "Method": "SplitConformal",
                        "Coverage90": 0.91,
                        "Width90": 0.4,
                        "IntervalScore90": 0.5,
                        "NMAD": 0.08,
                        "CatastrophicRate": 0.11,
                        "HighZ_MAE": 0.19,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    suite_report = output_dir / "photoz_benchmark_suite_latest.json"
    suite_report.write_text(
        json.dumps(
            {
                "artifact": "photoz_benchmark_suite_report",
                "profile": "smoke",
                "output_dir": str(output_dir),
                "real_data_only": False,
                "summary_paths": {
                    "photoz_benchmark_comparison": str(standard),
                    "photoz_nnc_crps_rail_comparison": str(ordered),
                    "ppi_photoz_inference_comparison": str(ppi),
                    "photoz_transferz_conformal_comparison": str(conformal),
                },
                "skipped_examples": [],
                "rail_merge": None,
            }
        ),
        encoding="utf-8",
    )

    out = render_photoz_benchmark_report.render_report(
        suite_report_path=suite_report,
        output_path=output_dir / "photoz_benchmark_suite_latest.md",
    )

    text = out.read_text(encoding="utf-8")
    assert "Photo-z Benchmark Suite Report" in text
    assert "Standard Regression Track" in text
    assert "Ordered-Bin / PDF Track" in text
    assert "Prediction-Powered Inference Track" in text
    assert "TransferZ Conformal Track" in text
    assert "GaussianNLL" in text
    assert "SoftBinnedCE" in text
    assert "SplitConformal" in text
