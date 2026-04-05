"""Render the task-first method catalog into docs/report artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from torchregress import method_catalog

MATRIX_SECTION_START = "<!-- BEGIN:METHOD_CATALOG_GENERATED_SECTION -->"
MATRIX_SECTION_END = "<!-- END:METHOD_CATALOG_GENERATED_SECTION -->"
TASK_MATRIX_SECTION_START = "<!-- BEGIN:TASK_MATRIX_GENERATED -->"
TASK_MATRIX_SECTION_END = "<!-- END:TASK_MATRIX_GENERATED -->"
DECISION_WORKFLOW_SECTION_START = "<!-- BEGIN:DECISION_WORKFLOW_GENERATED -->"
DECISION_WORKFLOW_SECTION_END = "<!-- END:DECISION_WORKFLOW_GENERATED -->"
FAMILY_MATRIX_SECTION_START = "<!-- BEGIN:FAMILY_CAPABILITY_MATRIX_GENERATED -->"
FAMILY_MATRIX_SECTION_END = "<!-- END:FAMILY_CAPABILITY_MATRIX_GENERATED -->"
PROVENANCE_DATE = "2026-04-05"

CAPABILITY_COLUMNS = [
    "multi_target",
    "multimodal",
    "non_gaussian",
    "epistemic",
    "aleatoric",
    "decomposition",
    "calibration",
    "ood_support",
    "imbalance",
    "noisy_features_eiv",
]


def _provenance_lines(
    *,
    source_fn: str,
    source_artifacts: list[str] | tuple[str, ...],
) -> list[str]:
    return [
        f"_Generated provenance_: `{source_fn}`",
        f"_Source artifacts_: {', '.join(f'`{x}`' for x in source_artifacts)}",
        f"_Generated date_: `{PROVENANCE_DATE}`",
    ]


def load_rows() -> list[dict[str, Any]]:
    """Load catalog rows from the public discovery API."""
    return method_catalog.list_methods()


def load_task_recommendations() -> list[dict[str, Any]]:
    """Load task-first recommendation rows from the public catalog API."""
    return method_catalog.list_task_recommendations()


def load_decision_workflow_steps() -> list[dict[str, Any]]:
    """Load fast-path decision workflow rows from the public catalog API."""
    return method_catalog.list_decision_workflow_steps()


def load_comparative_evidence_rows() -> list[dict[str, Any]]:
    """Load comparative evidence coverage rows from the public catalog API."""
    return method_catalog.list_comparative_evidence_rows()


def _capability_marks(row: dict[str, Any]) -> str:
    caps: list[str] = []
    for key in ("epistemic", "aleatoric", "decomposition", "calibration", "ood_support"):
        value = row.get(key)
        if value == "yes":
            caps.append(key.replace("_support", "").replace("_", "-"))
        elif value == "partial":
            caps.append(f"{key.replace('_support', '').replace('_', '-')}*")
    return ", ".join(caps) if caps else "-"


def _family_counts(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counts = Counter(str(row["family"]) for row in rows)
    return sorted(counts.items(), key=lambda item: item[0])


def _maturity_counts(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counts = Counter(row["maturity"] for row in rows)
    order = {"Core": 0, "Strong": 1, "Available": 2, "Advanced": 3}
    return sorted(counts.items(), key=lambda item: (order.get(item[0], 99), item[0]))


def _aggregate_capability(values: list[str]) -> str:
    if any(v == "yes" for v in values):
        return "yes"
    if any(v == "partial" for v in values):
        return "partial"
    return "no"


def _family_capability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["family"], []).append(row)

    aggregated: list[dict[str, Any]] = []
    for family in sorted(grouped):
        fam_rows = grouped[family]
        agg_row: dict[str, Any] = {
            "family": family,
            "n_methods": len(fam_rows),
            "methods": [r["name"] for r in sorted(fam_rows, key=lambda x: x["name"])],
        }
        for key in CAPABILITY_COLUMNS:
            agg_row[key] = _aggregate_capability([str(r.get(key, "no")) for r in fam_rows])
        aggregated.append(agg_row)
    return aggregated


def render_family_capability_matrix_table(rows: list[dict[str, Any]]) -> str:
    """Render the family capability matrix table used in the main docs section."""
    lines: list[str] = []
    lines.extend(
        _provenance_lines(
            source_fn="tools/render_method_catalog.py:render_family_capability_matrix_table",
            source_artifacts=["reports/method_catalog_latest.json"],
        )
    )
    lines.append("")
    lines.append(
        "| Method Family | Multi-target | Multimodal | Non-Gaussian | Epistemic | Aleatoric | "
        "Decomposition | Calibration | OOD Support | Imbalance | Noisy Features (EIV) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in _family_capability_rows(rows):
        lines.append(
            "| "
            f"`{row['family']}` ({row['n_methods']}) | "
            f"{row['multi_target']} | {row['multimodal']} | {row['non_gaussian']} | "
            f"{row['epistemic']} | {row['aleatoric']} | {row['decomposition']} | "
            f"{row['calibration']} | {row['ood_support']} | {row['imbalance']} | "
            f"{row['noisy_features_eiv']} |"
        )
    return "\n".join(lines)


def _render_method_cell(text: str) -> str:
    parts = [p.strip() for p in text.split("/") if p.strip()]
    if not parts:
        return "-"
    return " / ".join(f"`{p}`" for p in parts)


def render_task_matrix_table(rows: list[dict[str, Any]]) -> str:
    """Render the top task matrix rows from catalog task recommendations."""
    lines: list[str] = []
    lines.extend(
        _provenance_lines(
            source_fn="tools/render_method_catalog.py:render_task_matrix_table",
            source_artifacts=["reports/method_catalog_latest.json"],
        )
    )
    lines.append("")
    lines.append("| Task / Constraint | Recommended Start | Strong Alternatives | Notes |")
    lines.append("|---|---|---|---|")
    for row in rows:
        recommended = _render_method_cell(str(row["recommended_start"]))
        alternatives = row.get("strong_alternatives") or []
        alt_cell = (
            ", ".join(_render_method_cell(str(x)) for x in alternatives) if alternatives else "-"
        )
        lines.append(f"| {row['task']} | {recommended} | {alt_cell} | {row['notes']} |")
    return "\n".join(lines)


def render_decision_workflow(rows: list[dict[str, Any]]) -> str:
    """Render the fast-path decision workflow section from catalog metadata."""
    ordered = sorted(rows, key=lambda r: int(r["order"]))
    lines: list[str] = []
    lines.extend(
        _provenance_lines(
            source_fn="tools/render_method_catalog.py:render_decision_workflow",
            source_artifacts=["reports/method_catalog_latest.json"],
        )
    )
    lines.append("")
    for row in ordered:
        primary = f"`{row['primary_recommendation']}`"
        lines.append(f"{row['order']}. {row['question']}")
        lines.append(f"   Use {primary}.")
        alternatives = row.get("alternatives") or []
        if alternatives:
            alt_text = ", ".join(f"`{x}`" for x in alternatives)
            lines.append(f"   Alternatives: {alt_text}.")
        caveat = str(row.get("caveat", "")).strip()
        if caveat:
            lines.append(f"   Caveat: {caveat}")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_comparative_evidence_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a JSON-friendly comparative evidence coverage report."""
    grade_counts = Counter(str(r["comparison_grade"]) for r in rows)
    task_count = len(rows)
    decision_grade = grade_counts.get("Decision-grade", 0)
    strong_or_better = decision_grade + grade_counts.get("Strong", 0)
    return {
        "artifact": "comparative_evidence_matrix",
        "version": 1,
        "summary": {
            "task_rows": task_count,
            "grade_counts": dict(sorted(grade_counts.items())),
            "decision_grade_tasks": decision_grade,
            "strong_or_better_tasks": strong_or_better,
        },
        "rows": json.loads(json.dumps(rows)),
    }


def render_comparative_evidence_markdown(rows: list[dict[str, Any]]) -> str:
    """Render the comparative evidence coverage page from metadata."""
    report = build_comparative_evidence_report(rows)
    lines: list[str] = []
    lines.append("# Comparative Evidence Matrix")
    lines.append("")
    lines.extend(
        _provenance_lines(
            source_fn="tools/render_method_catalog.py:render_comparative_evidence_markdown",
            source_artifacts=[
                "reports/comparative_evidence_matrix_latest.json",
                "reports/method_catalog_latest.json",
            ],
        )
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Task rows: `{report['summary']['task_rows']}`")
    lines.append(f"- Decision-grade rows: `{report['summary']['decision_grade_tasks']}`")
    lines.append(f"- Strong-or-better rows: `{report['summary']['strong_or_better_tasks']}`")
    lines.append("")
    lines.append("## Coverage Matrix")
    lines.append("")
    lines.append(
        "| Task | Evidence Grade | Examples | Fairness Controls | Metrics Coverage | "
        "Peer Methods Visible | Gaps |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        examples = ", ".join(f"`{x}`" for x in row["examples"]) if row["examples"] else "-"
        fairness = ", ".join(row["fairness_controls"]) if row["fairness_controls"] else "-"
        metrics = ", ".join(row["metrics_coverage"]) if row["metrics_coverage"] else "-"
        peers = (
            ", ".join(f"`{x}`" for x in row["peer_methods_visible"])
            if row["peer_methods_visible"]
            else "-"
        )
        lines.append(
            "| "
            f"{row['task']} | "
            f"`{row['comparison_grade']}` | "
            f"{examples} | "
            f"{fairness} | "
            f"{metrics} | "
            f"{peers} | "
            f"{row['gaps']} |"
        )
        notes = str(row.get("notes", "")).strip()
        if notes:
            lines.append(f"|  |  |  |  |  |  | _Note_: {notes} |")
    lines.append("")
    lines.append("## Programmatic Access")
    lines.append("")
    lines.append(
        "Use the public catalog metadata to inspect comparative evidence coverage from code:"
    )
    lines.append("")
    lines.append("```python")
    lines.append("import torchregress as tr")
    lines.append("")
    lines.append("rows = tr.method_catalog.list_comparative_evidence_rows()")
    lines.append("for row in rows:")
    lines.append("    print(row['task'], row['comparison_grade'])")
    lines.append("```")
    lines.append("")
    lines.append("## Usage")
    lines.append("")
    lines.append(
        "- Use this page with the [Task-First Method Selection Matrix](method_selection_matrix.md) "
        "to distinguish API support from decision-grade comparative evidence."
    )
    lines.append(
        "- Prioritize filling `Missing` and `Demo-only` rows before making strong performance or "
        "method-selection claims in docs."
    )
    lines.append("")
    return "\n".join(lines)


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a JSON-friendly report for audit artifacts."""
    normalized_rows = json.loads(json.dumps(rows))
    return {
        "artifact": "method_catalog",
        "version": 1,
        "summary": {
            "total_methods": len(rows),
            "families": {k: v for k, v in _family_counts(rows)},
            "maturity": {k: v for k, v in _maturity_counts(rows)},
            "peer_uq_methods_present": {
                name: any(row["name"] == name for row in rows)
                for name in ("SWAG", "BayesianNeuralNetwork", "MDNLoss")
            },
        },
        "rows": normalized_rows,
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    """Render a deterministic markdown snapshot from the catalog rows."""
    rows_sorted = sorted(rows, key=lambda r: (r["family"], r["name"]))

    lines: list[str] = []
    lines.append("# Generated Method Catalog Snapshot")
    lines.append("")
    lines.extend(
        _provenance_lines(
            source_fn="tools/render_method_catalog.py:render_markdown",
            source_artifacts=["reports/method_catalog_latest.json"],
        )
    )
    lines.append("")
    lines.append("It is a code-backed snapshot used to reduce docs drift in the task-first matrix.")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total methods: `{len(rows_sorted)}`")
    lines.append(
        "- Peer methods present: "
        + ", ".join(
            f"`{name}`"
            for name in ("SWAG", "BayesianNeuralNetwork", "MDNLoss")
            if any(row["name"] == name for row in rows_sorted)
        )
    )
    lines.append("")

    lines.append("### By Family")
    lines.append("")
    lines.append("| Family | Count |")
    lines.append("|---|---:|")
    for family, count in _family_counts(rows_sorted):
        lines.append(f"| `{family}` | {count} |")
    lines.append("")

    lines.append("### By Maturity")
    lines.append("")
    lines.append("| Maturity | Count |")
    lines.append("|---|---:|")
    for maturity, count in _maturity_counts(rows_sorted):
        lines.append(f"| `{maturity}` | {count} |")
    lines.append("")

    lines.append("## Method Rows")
    lines.append("")
    lines.append(
        "Legend: `*` suffix means partial support. This snapshot is for discovery and comparison, "
        "not absolute ranking."
    )
    lines.append("")
    lines.append("| Name | Family | Maturity | Task Tags | Core Capability Signals | Public Path |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows_sorted:
        task_tags = ", ".join(f"`{tag}`" for tag in row["task_tags"])
        capability_signals = _capability_marks(row)
        if capability_signals != "-":
            capability_signals = ", ".join(f"`{x}`" for x in capability_signals.split(", "))
        lines.append(
            "| "
            f"`{row['name']}` | "
            f"`{row['family']}` | "
            f"`{row['maturity']}` | "
            f"{task_tags or '-'} | "
            f"{capability_signals} | "
            f"`{row['public_path']}` |"
        )
    lines.append("")

    lines.append("## Capability-Focused Shortlists")
    lines.append("")
    lines.append("### Multimodal (`multimodal=yes`)")
    lines.append("")
    multimodal = [row for row in rows_sorted if row.get("multimodal") == "yes"]
    lines.append(", ".join(f"`{row['name']}`" for row in multimodal) if multimodal else "-")
    lines.append("")

    lines.append("### Noisy Features / EIV (`noisy_features_eiv=yes`)")
    lines.append("")
    eiv = [row for row in rows_sorted if row.get("noisy_features_eiv") == "yes"]
    lines.append(", ".join(f"`{row['name']}`" for row in eiv) if eiv else "-")
    lines.append("")

    lines.append("### Decomposition (`decomposition=yes`)")
    lines.append("")
    decomp = [row for row in rows_sorted if row.get("decomposition") == "yes"]
    lines.append(", ".join(f"`{row['name']}`" for row in decomp) if decomp else "-")
    lines.append("")

    return "\n".join(lines) + "\n"


def render_method_matrix_generated_section(rows: list[dict[str, Any]]) -> str:
    """Render a compact code-backed subsection for the task-first method matrix page."""
    rows_sorted = sorted(rows, key=lambda r: (r["family"], r["name"]))
    lines: list[str] = []
    lines.append("### Catalog-Backed Peer Method Snapshot (Generated)")
    lines.append("")
    lines.extend(
        _provenance_lines(
            source_fn="tools/render_method_catalog.py:render_method_matrix_generated_section",
            source_artifacts=[
                "docs/guides/method_catalog_generated.md",
                "reports/method_catalog_latest.json",
            ],
        )
    )
    lines.append("")
    lines.append(
        "| Method | Family | Maturity | Multi-target | Multimodal | Epistemic | Aleatoric | "
        "Decomposition | Calibration | OOD |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in rows_sorted:
        lines.append(
            "| "
            f"`{row['name']}` | "
            f"`{row['family']}` | "
            f"`{row['maturity']}` | "
            f"{row['multi_target']} | "
            f"{row['multimodal']} | "
            f"{row['epistemic']} | "
            f"{row['aleatoric']} | "
            f"{row['decomposition']} | "
            f"{row['calibration']} | "
            f"{row['ood_support']} |"
        )
    lines.append("")
    lines.append(
        "Peer-method check: "
        + ", ".join(
            f"`{name}`"
            for name in ("SWAG", "BayesianNeuralNetwork", "MDNLoss")
            if any(row["name"] == name for row in rows_sorted)
        )
    )
    lines.append("")
    lines.append("### Generated Family Capability Matrix")
    lines.append("")
    lines.append(
        "| Family | # Methods | Multi-target | Multimodal | Non-Gaussian | Epistemic | "
        "Aleatoric | Decomposition | Calibration | OOD | Imbalance | EIV |"
    )
    lines.append("|---|---:|---|---|---|---|---|---|---|---|---|---|")
    for row in _family_capability_rows(rows_sorted):
        lines.append(
            "| "
            f"`{row['family']}` | {row['n_methods']} | "
            f"{row['multi_target']} | {row['multimodal']} | {row['non_gaussian']} | "
            f"{row['epistemic']} | {row['aleatoric']} | {row['decomposition']} | "
            f"{row['calibration']} | {row['ood_support']} | {row['imbalance']} | "
            f"{row['noisy_features_eiv']} |"
        )
    lines.append("")
    lines.append("### Generated Hard-Task Shortlists")
    lines.append("")
    lines.append("| Need | Catalog Filter (conceptual) | Suggested Methods |")
    lines.append("|---|---|---|")
    shortlist_specs = [
        (
            "OOD + epistemic signals",
            "`task_tag='ood'` + `epistemic=yes`",
            [r for r in rows_sorted if "ood" in r["task_tags"] and r["epistemic"] == "yes"],
        ),
        (
            "Coverage / calibration",
            "`calibration=yes`",
            [r for r in rows_sorted if r["calibration"] == "yes"],
        ),
        (
            "Multimodal targets",
            "`multimodal=yes`",
            [r for r in rows_sorted if r["multimodal"] == "yes"],
        ),
        (
            "Imbalanced / rare targets",
            "`imbalance=yes`",
            [r for r in rows_sorted if r["imbalance"] == "yes"],
        ),
        (
            "Noisy features / EIV",
            "`noisy_features_eiv=yes`",
            [r for r in rows_sorted if r["noisy_features_eiv"] == "yes"],
        ),
    ]
    for title, filter_desc, matched in shortlist_specs:
        method_names = ", ".join(f"`{row['name']}`" for row in matched) if matched else "-"
        lines.append(f"| {title} | {filter_desc} | {method_names} |")
    lines.append("")
    return "\n".join(lines)


def _replace_marked_section(
    text: str,
    replacement: str,
    *,
    start_marker: str = MATRIX_SECTION_START,
    end_marker: str = MATRIX_SECTION_END,
) -> str:
    """Replace the marker-delimited section in the method matrix page."""
    if start_marker not in text or end_marker not in text:
        raise ValueError("Method matrix generation markers not found.")
    before, rest = text.split(start_marker, 1)
    _, after = rest.split(end_marker, 1)
    body = f"{start_marker}\n{replacement.rstrip()}\n{end_marker}"
    return before + body + after


def update_method_matrix_page(path: Path, rows: list[dict[str, Any]]) -> None:
    """Update the generated marker section in the task-first method matrix page."""
    text = path.read_text(encoding="utf-8")
    text = _replace_marked_section(
        text,
        render_task_matrix_table(load_task_recommendations()),
        start_marker=TASK_MATRIX_SECTION_START,
        end_marker=TASK_MATRIX_SECTION_END,
    )
    text = _replace_marked_section(
        text,
        render_family_capability_matrix_table(rows),
        start_marker=FAMILY_MATRIX_SECTION_START,
        end_marker=FAMILY_MATRIX_SECTION_END,
    )
    text = _replace_marked_section(
        text,
        render_decision_workflow(load_decision_workflow_steps()),
        start_marker=DECISION_WORKFLOW_SECTION_START,
        end_marker=DECISION_WORKFLOW_SECTION_END,
    )
    text = _replace_marked_section(text, render_method_matrix_generated_section(rows))
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render method catalog markdown/json artifacts.")
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument(
        "--update-method-matrix",
        type=Path,
        help="Update the marker-delimited generated subsection in method_selection_matrix.md",
    )
    parser.add_argument("--comparative-evidence-md-out", type=Path)
    parser.add_argument("--comparative-evidence-json-out", type=Path)
    args = parser.parse_args()

    rows = load_rows()
    markdown = render_markdown(rows)
    report = build_report(rows)

    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(markdown, encoding="utf-8")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.update_method_matrix is not None:
        update_method_matrix_page(args.update_method_matrix, rows)
        print(f"Updated method matrix section: {args.update_method_matrix}")

    if (
        args.comparative_evidence_md_out is not None
        or args.comparative_evidence_json_out is not None
    ):
        evidence_rows = load_comparative_evidence_rows()
        evidence_md = render_comparative_evidence_markdown(evidence_rows)
        evidence_report = build_comparative_evidence_report(evidence_rows)
        if args.comparative_evidence_md_out is not None:
            args.comparative_evidence_md_out.parent.mkdir(parents=True, exist_ok=True)
            args.comparative_evidence_md_out.write_text(evidence_md, encoding="utf-8")
            print(f"Wrote comparative evidence markdown: {args.comparative_evidence_md_out}")
        if args.comparative_evidence_json_out is not None:
            args.comparative_evidence_json_out.parent.mkdir(parents=True, exist_ok=True)
            args.comparative_evidence_json_out.write_text(
                json.dumps(evidence_report, indent=2), encoding="utf-8"
            )
            print(f"Wrote comparative evidence JSON: {args.comparative_evidence_json_out}")

    print(f"Wrote markdown snapshot: {args.markdown_out}")
    print(f"Wrote JSON report: {args.json_out}")
    print(f"Methods: {report['summary']['total_methods']}")


if __name__ == "__main__":
    main()
