"""Render a compact markdown report for the photo-z benchmark suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_key(value: Any) -> float:
    if value is None:
        return float("inf")
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def _sort_rows(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: tuple(_metric_key(row.get(metric)) for metric in metrics))


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["(no rows)", ""]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(_format_cell(row.get(col)) for col in columns) + "|")
    lines.append("")
    return lines


def render_report(
    *,
    suite_report_path: Path,
    output_path: Path,
) -> Path:
    suite = _load_summary(suite_report_path)
    summary_paths = {name: Path(path) for name, path in suite["summary_paths"].items()}

    lines: list[str] = [
        "# Photo-z Benchmark Suite Report",
        "",
        f"- Profile: `{suite['profile']}`",
        f"- Output directory: `{suite['output_dir']}`",
        f"- Real-data only: `{bool(suite.get('real_data_only', False))}`",
        "",
    ]

    if suite.get("skipped_examples"):
        skipped = ", ".join(f"`{name}`" for name in suite["skipped_examples"])
        lines.append(f"- Skipped examples: {skipped}")
        lines.append("")

    if "photoz_benchmark_comparison" in summary_paths:
        payload = _load_summary(summary_paths["photoz_benchmark_comparison"])
        rows = _sort_rows(payload["rows"], ["NMAD", "CatastrophicRate", "HighZ_MAE"])
        lines.extend(
            [
                "## Standard Regression Track",
                "",
                "Sorted by `NMAD`, `CatastrophicRate`, `HighZ_MAE`.",
                "",
            ]
        )
        lines.extend(
            _render_table(
                rows,
                [
                    "Method",
                    "NMAD",
                    "CatastrophicRate",
                    "HighZ_MAE",
                    "Cov90",
                    "Width90",
                    "train_s",
                    "eval_s",
                    "DataSource",
                ],
            )
        )

    if "photoz_nnc_crps_rail_comparison" in summary_paths:
        payload = _load_summary(summary_paths["photoz_nnc_crps_rail_comparison"])
        rows = _sort_rows(payload["rows"], ["CRPS", "PDF_NLL", "PITChi2"])
        lines.extend(
            [
                "## Ordered-Bin / PDF Track",
                "",
                "Sorted by `CRPS`, `PDF_NLL`, `PITChi2`.",
                "",
            ]
        )
        lines.extend(
            _render_table(
                rows,
                [
                    "Method",
                    "CRPS",
                    "PDF_NLL",
                    "PITChi2",
                    "NMAD",
                    "CatastrophicRate",
                    "NativeCov90",
                    "NativeWidth90",
                    "train_s",
                    "eval_s",
                    "DataSource",
                ],
            )
        )

    if "ppi_photoz_inference_comparison" in summary_paths:
        payload = _load_summary(summary_paths["ppi_photoz_inference_comparison"])
        rows = sorted(
            payload["rows"],
            key=lambda row: (str(row.get("Target", "")), _metric_key(row.get("CIWidth"))),
        )
        lines.extend(
            [
                "## Prediction-Powered Inference Track",
                "",
                "Sorted by `Target`, `CIWidth`.",
                "",
            ]
        )
        lines.extend(
            _render_table(
                rows,
                [
                    "Method",
                    "Target",
                    "Estimate",
                    "AbsError",
                    "CIWidth",
                    "CoversTruth",
                ],
            )
        )

    rail_merge = suite.get("rail_merge")
    if isinstance(rail_merge, dict):
        merged_path = rail_merge.get("merged_output_path")
        if isinstance(merged_path, str) and Path(merged_path).exists():
            payload = _load_summary(Path(merged_path))
            rows = _sort_rows(payload["rows"], ["CRPS", "PDF_NLL", "NMAD"])
            lines.extend(
                [
                    "## RAIL Merge",
                    "",
                    "Sorted by `CRPS`, `PDF_NLL`, `NMAD` when present.",
                    "",
                ]
            )
            lines.extend(
                _render_table(
                    rows,
                    [
                        "Method",
                        "CRPS",
                        "PDF_NLL",
                        "NMAD",
                        "CatastrophicRate",
                        "NativeCov90",
                        "NativeWidth90",
                    ],
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a markdown photo-z benchmark suite report."
    )
    parser.add_argument("--suite-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    out = render_report(suite_report_path=args.suite_report, output_path=args.output)
    print(f"Wrote photo-z markdown report: {out}")


if __name__ == "__main__":
    main()
