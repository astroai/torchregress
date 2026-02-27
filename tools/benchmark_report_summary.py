"""Render benchmark JSON reports as concise Markdown tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    """Load a benchmark report from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def _format_ms(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _format_params(params: Any) -> str:
    if not params:
        return ""
    return ", ".join(f"{k}={params[k]}" for k in sorted(params))


def render_case_table(report: dict[str, Any]) -> str:
    lines = [
        "| Case | Status | Mean (ms) | Std (ms) | Params |",
        "|---|---:|---:|---:|---|",
    ]
    for case in report.get("cases", []):
        lines.append(
            "| "
            f"{case.get('name', '-')}"
            " | "
            f"{case.get('status', '-')}"
            " | "
            f"{_format_ms(case.get('mean_ms'))}"
            " | "
            f"{_format_ms(case.get('std_ms'))}"
            " | "
            f"{_format_params(case.get('params'))}"
            " |"
        )
    return "\n".join(lines)


def render_grouped_table(report: dict[str, Any]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in report.get("cases", []):
        grouped[str(case.get("name", "-"))].append(case)

    lines = [
        "| Case | Rows | OK | Mean of Means (ms) | Max Mean (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in sorted(grouped):
        rows = grouped[name]
        ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("mean_ms") is not None]
        mean_of_means = (
            sum(float(r["mean_ms"]) for r in ok_rows) / len(ok_rows) if ok_rows else None
        )
        max_mean = max((float(r["mean_ms"]) for r in ok_rows), default=None)
        lines.append(
            "| "
            f"{name} | {len(rows)} | {len(ok_rows)} | "
            f"{_format_ms(mean_of_means)} | {_format_ms(max_mean)} |"
        )
    return "\n".join(lines)


def summarize_report(report: dict[str, Any], *, group_by_name: bool = False) -> str:
    artifact = report.get("artifact", "benchmark")
    device = report.get("config", {}).get("device", "unknown")
    aggregate = report.get("aggregate", {})
    header = [
        f"# Benchmark Summary ({artifact})",
        "",
        f"- Device: `{device}`",
        f"- Cases: `{aggregate.get('n_cases', len(report.get('cases', [])))}`",
        (
            "- OK/Skipped/Error: "
            f"`{aggregate.get('n_ok', '?')}/"
            f"{aggregate.get('n_skipped', '?')}/"
            f"{aggregate.get('n_error', '?')}`"
        ),
        "",
    ]
    table = render_grouped_table(report) if group_by_name else render_case_table(report)
    return "\n".join(header + [table, ""])


def main() -> None:
    parser = argparse.ArgumentParser(description="Render benchmark report JSON as Markdown.")
    parser.add_argument("report", type=Path, help="Path to benchmark JSON report")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output path")
    parser.add_argument(
        "--group-by-name",
        action="store_true",
        help="Aggregate repeated sweep rows by benchmark case name",
    )
    args = parser.parse_args()

    summary = summarize_report(load_report(args.report), group_by_name=args.group_by_name)
    if args.output is None:
        print(summary)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary, encoding="utf-8")
        print(f"Wrote benchmark summary: {args.output}")


if __name__ == "__main__":
    main()
