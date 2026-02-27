from __future__ import annotations

from pathlib import Path

from tools import benchmark_report_summary, render_benchmark_baselines

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DOC = REPO_ROOT / "docs" / "audits" / "benchmark_cpu_baselines_2026-02-26.md"
SMOKE_JSON = REPO_ROOT / "reports" / "benchmark_smoke_2026-02-26.json"
SWEEP_JSON = REPO_ROOT / "reports" / "benchmark_sweep_cpu_2026-02-26.json"


def test_render_generated_section_contains_expected_headers() -> None:
    smoke = benchmark_report_summary.load_report(SMOKE_JSON)
    sweep = benchmark_report_summary.load_report(SWEEP_JSON)
    section = render_benchmark_baselines.render_generated_section(smoke, sweep)

    assert "## Smoke Baseline Summary" in section
    assert "# Benchmark Summary (benchmark_smoke)" in section
    assert "## Sweep Baseline Summary (Grouped)" in section
    assert "| sweep_mdn_full_forward |" in section


def test_committed_benchmark_baseline_doc_is_in_sync() -> None:
    if not BASELINE_DOC.exists() or not SMOKE_JSON.exists() or not SWEEP_JSON.exists():
        import pytest
        pytest.skip("Benchmark baseline artifacts (ignored) are missing")

    text = BASELINE_DOC.read_text(encoding="utf-8")
    smoke = benchmark_report_summary.load_report(SMOKE_JSON)
    sweep = benchmark_report_summary.load_report(SWEEP_JSON)
    expected = render_benchmark_baselines.render_baseline_page(
        text,
        smoke_report=smoke,
        sweep_report=sweep,
    )
    assert text == expected
