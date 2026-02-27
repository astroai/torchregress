import json
from pathlib import Path

from tools import benchmark_report_summary


def _sample_report() -> dict:
    return {
        "artifact": "benchmark_sweep",
        "config": {"device": "cpu"},
        "aggregate": {"n_cases": 3, "n_ok": 2, "n_skipped": 1, "n_error": 0},
        "cases": [
            {
                "name": "case_a",
                "status": "ok",
                "mean_ms": 1.0,
                "std_ms": 0.1,
                "params": {"batch": 16},
            },
            {
                "name": "case_a",
                "status": "ok",
                "mean_ms": 2.0,
                "std_ms": 0.2,
                "params": {"batch": 64},
            },
            {
                "name": "case_b",
                "status": "skipped",
                "mean_ms": None,
                "std_ms": None,
                "params": None,
            },
        ],
    }


def test_summarize_report_renders_case_and_grouped_tables(tmp_path: Path) -> None:
    report_path = tmp_path / "bench.json"
    report = _sample_report()
    report_path.write_text(json.dumps(report), encoding="utf-8")

    loaded = benchmark_report_summary.load_report(report_path)
    case_md = benchmark_report_summary.summarize_report(loaded, group_by_name=False)
    grouped_md = benchmark_report_summary.summarize_report(loaded, group_by_name=True)

    assert "# Benchmark Summary (benchmark_sweep)" in case_md
    assert "| Case | Status | Mean (ms) |" in case_md
    assert "case_a" in case_md and "batch=16" in case_md

    assert "| Case | Rows | OK | Mean of Means (ms) | Max Mean (ms) |" in grouped_md
    assert "| case_a | 2 | 2 | 1.500 | 2.000 |" in grouped_md
    assert "| case_b | 1 | 0 | - | - |" in grouped_md
