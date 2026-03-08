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


def _best_row(
    rows: list[dict[str, Any]],
    metrics: list[str],
    *,
    predicate: Any = None,
) -> dict[str, Any] | None:
    filtered = rows if predicate is None else [row for row in rows if predicate(row)]
    if not filtered:
        return None
    return _sort_rows(filtered, metrics)[0]


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

    if suite.get("dataset_path"):
        lines.extend([f"- Shared dataset path: `{suite['dataset_path']}`", ""])
    elif (
        suite.get("train_dataset_path")
        and suite.get("cal_dataset_path")
        and suite.get("test_dataset_path")
    ):
        lines.extend(
            [
                f"- Train split: `{suite['train_dataset_path']}`",
                f"- Cal split: `{suite['cal_dataset_path']}`",
                f"- Test split: `{suite['test_dataset_path']}`",
                "",
            ]
        )

    if suite.get("skipped_examples"):
        skipped = ", ".join(f"`{name}`" for name in suite["skipped_examples"])
        lines.append(f"- Skipped examples: {skipped}")
        lines.append("")

    if "photoz_benchmark_comparison" in summary_paths:
        payload = _load_summary(summary_paths["photoz_benchmark_comparison"])
        rows = _sort_rows(payload["rows"], ["NMAD", "CatastrophicRate", "HighZ_MAE"])
        best_overall = rows[0] if rows else None
        best_tail = _best_row(rows, ["HighZ_MAE", "NMAD"])
        best_high_err = _best_row(rows, ["HighErr_NMAD", "HighErr_CatastrophicRate", "NMAD"])
        best_robust = _best_row(
            rows,
            ["CatastrophicRate", "NMAD", "HighZ_MAE"],
            predicate=lambda row: "Huber" in str(row.get("Method", "")),
        )
        best_noisy_label = _best_row(
            rows,
            ["NLL", "NMAD", "CatastrophicRate"],
            predicate=lambda row: "NoisyTarget" in str(row.get("Method", "")),
        )
        best_prob = _best_row(
            rows,
            ["NLL", "NMAD", "CatastrophicRate"],
            predicate=lambda row: row.get("NLL") is not None,
        )
        lines.extend(
            [
                "## Standard Regression Track",
                "",
                "Sorted by `NMAD`, `CatastrophicRate`, `HighZ_MAE`.",
                "",
            ]
        )
        if best_overall is not None:
            lines.append(
                f"- Best overall: `{best_overall['Method']}` "
                f"(NMAD {_format_cell(best_overall.get('NMAD'))}, "
                f"CatastrophicRate {_format_cell(best_overall.get('CatastrophicRate'))})"
            )
        if best_tail is not None:
            lines.append(
                f"- Best high-z tail: `{best_tail['Method']}` "
                f"(HighZ_MAE {_format_cell(best_tail.get('HighZ_MAE'))})"
            )
        if best_high_err is not None:
            lines.append(
                f"- Best high-feature-error row: `{best_high_err['Method']}` "
                "("
                f"HighErr_NMAD {_format_cell(best_high_err.get('HighErr_NMAD'))}, "
                "HighErr_CatastrophicRate "
                f"{_format_cell(best_high_err.get('HighErr_CatastrophicRate'))}"
                ")"
            )
        if best_robust is not None:
            lines.append(
                f"- Best robust row: `{best_robust['Method']}` "
                f"(CatastrophicRate {_format_cell(best_robust.get('CatastrophicRate'))})"
            )
        if best_noisy_label is not None:
            lines.append(
                f"- Best noisy-label-aware row: `{best_noisy_label['Method']}` "
                f"(NLL {_format_cell(best_noisy_label.get('NLL'))}, "
                f"NMAD {_format_cell(best_noisy_label.get('NMAD'))})"
            )
        if best_prob is not None:
            lines.append(
                f"- Best probabilistic row: `{best_prob['Method']}` "
                f"(NLL {_format_cell(best_prob.get('NLL'))}, "
                f"Cov90 {_format_cell(best_prob.get('Cov90'))})"
            )
        lines.append("")
        lines.extend(
            _render_table(
                rows,
                [
                    "Method",
                    "NMAD",
                    "CatastrophicRate",
                    "HighZ_MAE",
                    "HighErr_NMAD",
                    "HighErr_CatastrophicRate",
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

    if "photoz_transferz_conformal_comparison" in summary_paths:
        payload = _load_summary(summary_paths["photoz_transferz_conformal_comparison"])
        rows = _sort_rows(payload["rows"], ["IntervalScore90", "Coverage90", "Width90"])
        lines.extend(
            [
                "## TransferZ Conformal Track",
                "",
                "Sorted by `IntervalScore90`, `Coverage90`, `Width90`.",
                "",
            ]
        )
        lines.extend(
            _render_table(
                rows,
                [
                    "Method",
                    "Coverage90",
                    "Width90",
                    "IntervalScore90",
                    "NMAD",
                    "CatastrophicRate",
                    "HighZ_MAE",
                    "HighZCoverage90",
                    "HighZWidth90",
                    "train_s",
                    "eval_s",
                    "DataSource",
                ],
            )
        )

    if "photoz_transferz_semisupervised_comparison" in summary_paths:
        payload = _load_summary(summary_paths["photoz_transferz_semisupervised_comparison"])
        rows = sorted(
            payload["rows"],
            key=lambda row: (
                _metric_key(row.get("LabeledFraction")),
                _metric_key(row.get("NMAD")),
                _metric_key(row.get("CatastrophicRate")),
            ),
        )
        lines.extend(
            [
                "## TransferZ Semi-Supervised Track",
                "",
                "Sorted by `LabeledFraction`, `NMAD`, `CatastrophicRate`.",
                "",
            ]
        )
        fractions = sorted({float(row.get("LabeledFraction", 0.0)) for row in rows})
        for frac in fractions:
            best_frac = _best_row(
                rows,
                ["NMAD", "CatastrophicRate", "HighZ_MAE"],
                predicate=lambda row, frac=frac: abs(float(row.get("LabeledFraction", -1.0)) - frac)
                < 1e-8,
            )
            best_ssl = _best_row(
                rows,
                ["NMAD", "CatastrophicRate", "HighZ_MAE"],
                predicate=lambda row, frac=frac: (
                    abs(float(row.get("LabeledFraction", -1.0)) - frac) < 1e-8
                    and str(row.get("Method")) not in {"HuberLabeledOnly", "GaussianLabeledOnly"}
                ),
            )
            if best_frac is not None:
                lines.append(
                    f"- Best at labeled fraction `{frac:.4f}`: `{best_frac['Method']}` "
                    f"(NMAD {_format_cell(best_frac.get('NMAD'))}, "
                    f"PseudoAcceptRate {_format_cell(best_frac.get('PseudoAcceptRate'))})"
                )
            if best_ssl is not None:
                lines.append(
                    f"- Best SSL-only at labeled fraction `{frac:.4f}`: `{best_ssl['Method']}` "
                    f"(NMAD {_format_cell(best_ssl.get('NMAD'))}, "
                    f"PseudoAcceptRate {_format_cell(best_ssl.get('PseudoAcceptRate'))})"
                )
        lines.append("")
        lines.extend(
            _render_table(
                rows,
                [
                    "Method",
                    "LabeledFraction",
                    "NMAD",
                    "CatastrophicRate",
                    "HighZ_MAE",
                    "PseudoAcceptRate",
                    "PseudoMeanConfidence",
                    "AcceptedHighZShare",
                    "AcceptedLowErrShare",
                    "TeacherDisagreement",
                    "FeatureStability",
                    "LabeledHighZShare",
                    "TrainHighZShare",
                    "train_s",
                    "DataSource",
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
