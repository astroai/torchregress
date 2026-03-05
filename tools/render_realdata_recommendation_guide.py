"""Render a real-data-first recommendation guide from comparative evidence artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

START_MARKER = "<!-- REALDATA-RECOMMENDATION-GENERATED:START -->"
END_MARKER = "<!-- REALDATA-RECOMMENDATION-GENERATED:END -->"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC = REPO_ROOT / "docs" / "guides" / "real_data_recommendation_guide.md"
DEFAULT_COMPARATIVE_JSON = REPO_ROOT / "reports" / "comparative_evidence_matrix_latest.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_marked_section(text: str, replacement: str) -> str:
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        raise ValueError("Recommendation guide markers are missing or out of order")
    before = text[: start + len(START_MARKER)]
    after = text[end:]
    body = "\n\n" + replacement.rstrip() + "\n"
    return before + body + after


def _has_real_proxy_example(example_path: str) -> bool:
    token = example_path.lower()
    return any(
        marker in token
        for marker in (
            "_realdata_",
            "photoz_",
            "_rail_",
            "rail_",
        )
    )


def _evidence_band(row: dict[str, Any]) -> str:
    examples = row.get("examples", [])
    has_real_proxy = any(_has_real_proxy_example(str(path)) for path in examples)
    grade = str(row.get("comparison_grade", ""))
    gaps = str(row.get("gaps", "")).lower()
    needs_more_real_data = any(
        marker in gaps
        for marker in (
            "needs",
            "need ",
            "missing",
            "requires",
        )
    )
    if has_real_proxy and grade == "Decision-grade" and not needs_more_real_data:
        return "Decision-grade real-data"
    if has_real_proxy:
        return "Real proxy"
    return "Synthetic only"


def _claim_boundary(band: str) -> str:
    if band == "Decision-grade real-data":
        return "Deployment-facing recommendation allowed with explicit scope."
    if band == "Real proxy":
        return "Task-fit and transfer plausibility; avoid production-readiness claims."
    return "Algorithmic feasibility only; no domain transfer claims."


def _recommended_methods(row: dict[str, Any], *, max_methods: int = 3) -> str:
    methods = [str(name) for name in row.get("peer_methods_visible", []) if str(name).strip()]
    if not methods:
        return "n/a"
    return ", ".join(f"`{name}`" for name in methods[:max_methods])


def _render_recommendation_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Task | Start Methods | Evidence Band | Claim Boundary | Next Data Step |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        task = str(row.get("task", ""))
        band = _evidence_band(row)
        methods = _recommended_methods(row)
        gap = str(row.get("gaps", "")).strip() or "Maintain regression tracking."
        lines.append(
            f"| {task} | {methods} | `{band}` | {_claim_boundary(band)} | {gap} |"
        )
    return "\n".join(lines)


def render_generated_section(
    comparative_payload: dict[str, Any],
    *,
    source_json_path: Path = DEFAULT_COMPARATIVE_JSON,
) -> str:
    source_json_resolved = source_json_path.resolve()
    try:
        source_display = source_json_resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        source_display = source_json_resolved.as_posix()

    rows = list(comparative_payload.get("rows", []))
    band_counts = {"Synthetic only": 0, "Real proxy": 0, "Decision-grade real-data": 0}
    for row in rows:
        band_counts[_evidence_band(row)] += 1

    generated_header = "\n".join(
        [
            "_Generated provenance_: "
            "`tools/render_realdata_recommendation_guide.py:render_generated_section`",
            f"_Source artifact_: `{source_display}`",
            f"_Generated date_: `{date.today().isoformat()}`",
        ]
    )
    band_summary = "\n".join(
        [
            "## Evidence Band Summary",
            "",
            f"- Synthetic only: `{band_counts['Synthetic only']}`",
            f"- Real proxy: `{band_counts['Real proxy']}`",
            f"- Decision-grade real-data: `{band_counts['Decision-grade real-data']}`",
        ]
    )
    policy = "\n".join(
        [
            "## Claim Policy",
            "",
            "- `Synthetic only`: claim feasibility and relative behavior under controlled "
            "synthetic assumptions.",
            "- `Real proxy`: claim task-fit plausibility on at least one real-data "
            "proxy benchmark.",
            "- `Decision-grade real-data`: claim deployment-facing recommendation within "
            "validated scope.",
        ]
    )
    table = "\n".join(
        [
            "## Task-to-Method Recommendations",
            "",
            _render_recommendation_table(rows),
        ]
    )
    return "\n\n".join([generated_header, band_summary, policy, table])


def render_recommendation_guide(
    doc_text: str,
    *,
    comparative_payload: dict[str, Any],
    source_json_path: Path = DEFAULT_COMPARATIVE_JSON,
) -> str:
    generated = render_generated_section(comparative_payload, source_json_path=source_json_path)
    return _replace_marked_section(doc_text, generated)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate real-data-first recommendation guide from comparative evidence."
    )
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--comparative-json", type=Path, default=DEFAULT_COMPARATIVE_JSON)
    args = parser.parse_args()

    doc_text = args.doc.read_text(encoding="utf-8")
    comparative_payload = _load_json(args.comparative_json)
    rendered = render_recommendation_guide(
        doc_text,
        comparative_payload=comparative_payload,
        source_json_path=args.comparative_json,
    )
    args.doc.write_text(rendered, encoding="utf-8")
    print(f"Updated real-data recommendation guide: {args.doc}")


if __name__ == "__main__":
    main()
