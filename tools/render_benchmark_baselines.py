"""Render the committed CPU benchmark baseline docs page from JSON artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from tools import benchmark_report_summary
except ModuleNotFoundError:
    import benchmark_report_summary

START_MARKER = "<!-- BENCHMARK-BASELINE-GENERATED:START -->"
END_MARKER = "<!-- BENCHMARK-BASELINE-GENERATED:END -->"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "audits" / "benchmark_cpu_baselines_2026-02-26.md"
DEFAULT_SMOKE_JSON = REPO_ROOT / "reports" / "benchmark_smoke_2026-02-26.json"
DEFAULT_SWEEP_JSON = REPO_ROOT / "reports" / "benchmark_sweep_cpu_2026-02-26.json"


def _replace_marked_section(text: str, replacement: str) -> str:
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ValueError("Benchmark baseline doc markers are missing or out of order")
    before = text[: start + len(START_MARKER)]
    after = text[end:]
    body = "\n\n" + replacement.rstrip() + "\n"
    return before + body + after


def render_generated_section(smoke_report: dict[str, Any], sweep_report: dict[str, Any]) -> str:
    smoke_md = benchmark_report_summary.summarize_report(smoke_report, group_by_name=False).rstrip()
    sweep_md = benchmark_report_summary.summarize_report(sweep_report, group_by_name=True).rstrip()
    return (
        "## Smoke Baseline Summary\n\n"
        f"{smoke_md}\n\n"
        "## Sweep Baseline Summary (Grouped)\n\n"
        f"{sweep_md}"
    )


def render_baseline_page(
    doc_text: str,
    *,
    smoke_report: dict[str, Any],
    sweep_report: dict[str, Any],
) -> str:
    return _replace_marked_section(doc_text, render_generated_section(smoke_report, sweep_report))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate committed benchmark baseline docs page."
    )
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC_PATH)
    parser.add_argument("--smoke", type=Path, default=DEFAULT_SMOKE_JSON)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP_JSON)
    args = parser.parse_args()

    doc_text = args.doc.read_text(encoding="utf-8")
    smoke_report = benchmark_report_summary.load_report(args.smoke)
    sweep_report = benchmark_report_summary.load_report(args.sweep)
    rendered = render_baseline_page(doc_text, smoke_report=smoke_report, sweep_report=sweep_report)
    args.doc.write_text(rendered, encoding="utf-8")
    print(f"Updated benchmark baseline doc: {args.doc}")


if __name__ == "__main__":
    main()
