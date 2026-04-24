"""Render a consolidated review-readiness packet from audit/evidence artifacts."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _find_latest(directory: Path, pattern: str) -> Path:
    """Find the file with the lexicographically largest name matching the pattern."""
    matches = list(directory.glob(pattern))
    if not matches:
        # Return a non-existent path so _load_optional_json handles it
        return directory / f"MISSING_{pattern}"
    return max(matches, key=lambda p: p.name)


REPO_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = REPO_ROOT / "reports"
AUDITS_DIR = REPO_ROOT / "docs" / "reports" / "audits"

ADOPTION_AUDIT_MD = _find_latest(AUDITS_DIR, "adoption_readiness_*.md")
ADOPTION_AUDIT_JSON = _find_latest(REPORTS_DIR, "adoption_readiness_*.json")
COMPARATIVE_EVIDENCE_JSON = REPORTS_DIR / "comparative_evidence_matrix_latest.json"
METHOD_CATALOG_JSON = REPORTS_DIR / "method_catalog_latest.json"
NATIVE_LEVERAGE_JSON = _find_latest(REPORTS_DIR, "native_pytorch_leverage_matrix_*.json")
MYPY_TRIAGE_JSON = REPORTS_DIR / "mypy_triage_latest.json"
EXAMPLE_PROFILE_COMPARE_JSON = (
    REPORTS_DIR / "example_summaries" / "profile_comparison_audit_vs_full.json"
)
EXAMPLE_THRESH_VERDICT_JSON = REPORTS_DIR / "example_summaries" / "threshold_check_full_latest.json"
EXAMPLE_THRESHOLDS_JSON = REPORTS_DIR / "example_summaries" / "thresholds_full.json"
EXAMPLE_THRESH_REVIEW_STRICT_VERDICT_JSON = (
    REPORTS_DIR / "example_summaries" / "threshold_check_full_review_strict_latest.json"
)
EXAMPLE_THRESHOLDS_REVIEW_STRICT_JSON = (
    REPORTS_DIR / "example_summaries" / "thresholds_full_review_strict.json"
)
BENCH_THRESH_SMOKE_JSON = REPORTS_DIR / "benchmark_thresholds" / "cpu" / "smoke.json"
BENCH_THRESH_SWEEP_JSON = REPORTS_DIR / "benchmark_thresholds" / "cpu" / "sweep.json"
BENCH_BASELINE_JSON = _find_latest(REPORTS_DIR, "benchmark_sweep_cpu_*.json")
REVIEW_PACKET_JSON = REPORTS_DIR / "review_readiness_packet_latest.json"


def _get_review_packet_md_path() -> Path:
    """Generate review packet path with current date."""
    import datetime

    date_str = datetime.date.today().strftime("%Y-%m-%d")
    return AUDITS_DIR / f"review_readiness_packet_{date_str}.md"


REVIEW_PACKET_MD = _get_review_packet_md_path()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _parse_adoption_scores(md_text: str) -> dict[str, float]:
    baseline_match = re.search(r"`([0-9]+(?:\.[0-9]+)?) / 100` score above remains", md_text)
    provisional_match = re.search(
        r"\*\*Total \(Provisional\)\*\*.*?\*\*([0-9]+(?:\.[0-9]+)?) / 100\*\*", md_text, re.S
    )
    out: dict[str, float] = {}
    if baseline_match:
        out["baseline"] = float(baseline_match.group(1))
    if provisional_match:
        out["provisional"] = float(provisional_match.group(1))
    return out


def _require_adoption_scores(scores: dict[str, float]) -> tuple[float, float]:
    baseline = scores.get("baseline")
    provisional = scores.get("provisional")
    if baseline is None or provisional is None:
        raise ValueError(
            "Could not parse baseline/provisional adoption audit scores from "
            f"{ADOPTION_AUDIT_MD}. Keep the score lines stable or move scores into JSON."
        )
    return baseline, provisional


def _native_decision_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in payload.get("decisions", []):
        if isinstance(row, dict) and isinstance(row.get("decision"), str):
            counts[row["decision"]] += 1
    return dict(sorted(counts.items()))


def _count_limits(payload: dict[str, Any]) -> int | None:
    limits = payload.get("limits")
    if isinstance(limits, dict):
        return len(limits)
    if isinstance(limits, list):
        return len(limits)
    return None


def _comparative_gaps(payload: dict[str, Any]) -> list[dict[str, str]]:
    rows = payload.get("rows", [])
    gaps: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        gap = row.get("gaps")
        task = row.get("task")
        grade = row.get("comparison_grade")
        if isinstance(task, str) and isinstance(gap, str) and gap.strip():
            gaps.append(
                {
                    "task": task,
                    "grade": str(grade),
                    "gap": gap.strip(),
                }
            )
    return gaps


def _review_focus_files() -> list[str]:
    return [
        str(ADOPTION_AUDIT_MD.relative_to(REPO_ROOT)),
        "docs/guide/method-selection.md",
        "docs/reports/comparative_evidence_matrix.md",
        str(NATIVE_LEVERAGE_JSON.relative_to(REPO_ROOT)),
        "tests/test_native_parity.py",
        "tests/test_loss_forward_signature_contracts.py",
        "reports/example_summaries/profile_comparison_audit_vs_full.json",
        "reports/example_summaries/threshold_check_full_latest.json",
        "tools/benchmark_smoke.py",
        "reports/benchmark_thresholds/cpu/sweep.json",
    ]


def build_review_packet() -> dict[str, Any]:
    audit_md_text = (
        ADOPTION_AUDIT_MD.read_text(encoding="utf-8") if ADOPTION_AUDIT_MD.exists() else ""
    )
    audit_scores = _parse_adoption_scores(audit_md_text)
    baseline_score = audit_scores.get("baseline", 0.0)
    provisional_score = audit_scores.get("provisional", 0.0)

    adoption_audit = _load_optional_json(ADOPTION_AUDIT_JSON) or {}
    comparative = _load_optional_json(COMPARATIVE_EVIDENCE_JSON) or {}
    method_catalog = _load_optional_json(METHOD_CATALOG_JSON) or {}
    native = _load_optional_json(NATIVE_LEVERAGE_JSON) or {}
    mypy = _load_optional_json(MYPY_TRIAGE_JSON) or {}
    ex_compare = _load_optional_json(EXAMPLE_PROFILE_COMPARE_JSON) or {}
    ex_thresh_verdict = _load_optional_json(EXAMPLE_THRESH_VERDICT_JSON) or {}
    ex_thresholds = _load_optional_json(EXAMPLE_THRESHOLDS_JSON) or {}
    ex_thresh_review_strict_verdict = _load_optional_json(EXAMPLE_THRESH_REVIEW_STRICT_VERDICT_JSON)
    ex_thresholds_review_strict = _load_optional_json(EXAMPLE_THRESHOLDS_REVIEW_STRICT_JSON)
    bench_smoke_thresh = _load_optional_json(BENCH_THRESH_SMOKE_JSON) or {}
    bench_sweep_thresh = _load_optional_json(BENCH_THRESH_SWEEP_JSON) or {}

    mypy_summary = mypy.get("summary", {})
    comparative_summary = comparative.get("summary", {})
    catalog_summary = method_catalog.get("summary", {})

    packet: dict[str, Any] = {
        "artifact": "review_readiness_packet",
        "version": 1,
        "date": REVIEW_PACKET_MD.stem.split("_")[-1],
        "audit_v1_status": {
            "closed_v1": True,
            "closeout_date": REVIEW_PACKET_MD.stem.split("_")[-1],
            "closed_actionables": [
                "docs_example_api_drift_zero",
                "full_repo_mypy_zero",
                "native_leverage_matrix_with_parity_contracts",
                "example_summary_profile_and_threshold_governance",
                "benchmark_threshold_governance_cpu",
                "review_packet_artifact_in_always_on_ci",
            ],
            "deferred_v2_backlog": [
                "additional_real_data_ood_selective_benchmarks",
                "domain_specific_multimodal_real_data_benchmarks",
                "broader_noisy_features_noisy_labels_external_validity",
                "zuko_flow_optional_ci_expansion",
            ],
        },
        "adoption_audit": {
            "baseline_score": baseline_score,
            "provisional_score": provisional_score,
            "docs_drift": adoption_audit.get("docs", {}),
            "examples_summary": adoption_audit.get("examples", {}),
            "docs_drift_counts": adoption_audit.get("docs", {}).get("counts", {}),
            "example_import_counts": adoption_audit.get("examples", {}).get("counts", {}),
        },
        "typing": {
            "total_errors": mypy_summary.get("total_errors"),
            "packages": mypy_summary.get("packages", {}),
            "top_files": mypy_summary.get("top_files", []),
        },
        "comparative_evidence": {
            "summary": comparative_summary,
            "open_gaps": _comparative_gaps(comparative),
        },
        "method_catalog": {
            "summary": catalog_summary,
        },
        "native_leverage": {
            "decision_counts": _native_decision_counts(native),
            "n_areas": len(native.get("decisions", [])),
        },
        "example_summary_governance": {
            "profile_compare_ok": ex_compare.get("ok"),
            "profile_compare_rows": (
                len(ex_compare.get("rows", []))
                if isinstance(ex_compare.get("rows"), list)
                else None
            ),
            "threshold_ok": ex_thresh_verdict.get("ok"),
            "checked_limits": ex_thresh_verdict.get("checked_limits"),
            "failed_limits": ex_thresh_verdict.get("failed_limits"),
            "missing_limits": ex_thresh_verdict.get("missing_limits"),
            "threshold_limit_count": ex_thresholds.get("n_limits"),
            "threshold_artifact_count": ex_thresholds.get("n_artifacts"),
            "ci_threshold_profile": ex_thresholds.get("threshold_profile", "ci_conservative"),
            "review_threshold_profile": (
                ex_thresholds_review_strict.get("threshold_profile")
                if isinstance(ex_thresholds_review_strict, dict)
                else None
            ),
            "review_threshold_ok": (
                ex_thresh_review_strict_verdict.get("ok")
                if isinstance(ex_thresh_review_strict_verdict, dict)
                else None
            ),
            "review_threshold_checked_limits": (
                ex_thresh_review_strict_verdict.get("checked_limits")
                if isinstance(ex_thresh_review_strict_verdict, dict)
                else None
            ),
            "review_threshold_failed_limits": (
                ex_thresh_review_strict_verdict.get("failed_limits")
                if isinstance(ex_thresh_review_strict_verdict, dict)
                else None
            ),
            "review_threshold_missing_limits": (
                ex_thresh_review_strict_verdict.get("missing_limits")
                if isinstance(ex_thresh_review_strict_verdict, dict)
                else None
            ),
        },
        "benchmark_governance": {
            "cpu_smoke_threshold_limits": _count_limits(bench_smoke_thresh),
            "cpu_sweep_threshold_limits": _count_limits(bench_sweep_thresh),
            # Keep review packet deterministic against committed threshold artifacts.
            # Raw benchmark sweep reports are optional local artifacts and may not exist
            # in clean checkouts, so we avoid environment-sensitive summary drift here.
            "cpu_sweep_baseline_summary": {},
        },
        "review_focus_files": _review_focus_files(),
    }
    return packet


def render_markdown(packet: dict[str, Any]) -> str:
    status = packet["audit_v1_status"]
    audit = packet["adoption_audit"]
    typing = packet["typing"]
    comp = packet["comparative_evidence"]
    cat = packet["method_catalog"]["summary"]
    native = packet["native_leverage"]
    exgov = packet["example_summary_governance"]
    bench = packet["benchmark_governance"]
    comp_summary = comp.get("summary", {})

    focus_lines = "\n".join(f"- `{path}`" for path in packet["review_focus_files"])

    gaps = comp.get("open_gaps", [])
    gap_lines = (
        "\n".join(
            f"- `{g['task']}` (`{g['grade']}`): {g['gap']}"
            for g in gaps[:10]
            if isinstance(g, dict)
        )
        if gaps
        else "- None"
    )

    decision_counts = native.get("decision_counts", {})
    decision_lines = (
        "\n".join(f"- `{k}`: {v}" for k, v in decision_counts.items())
        if isinstance(decision_counts, dict) and decision_counts
        else "- unavailable"
    )

    docs_counts = audit.get("docs_drift_counts", {})
    ex_counts = audit.get("example_import_counts", {})
    score_line = f"`{audit.get('baseline_score')} -> {audit.get('provisional_score')}`"
    drift_line = (
        f"`attr={docs_counts.get('invalid_attr_refs')}`, "
        f"`imports={docs_counts.get('invalid_python_imports')}`, "
        f"`extras={docs_counts.get('invalid_extras')}`, "
        f"`example_imports={ex_counts.get('invalid_imports')}`"
    )
    evidence_line = (
        f"`{comp_summary.get('strong_or_better_tasks')} / {comp_summary.get('task_rows')}`"
    )
    profile_line = (
        f"`ok={exgov.get('profile_compare_ok')}`, rows=`{exgov.get('profile_compare_rows')}`"
    )
    threshold_line = (
        f"`ok={exgov.get('threshold_ok')}`, "
        f"checked=`{exgov.get('checked_limits')}`, "
        f"failed=`{exgov.get('failed_limits')}`, "
        f"missing=`{exgov.get('missing_limits')}`"
    )
    review_threshold_line = (
        f"`ok={exgov.get('review_threshold_ok')}`, "
        f"checked=`{exgov.get('review_threshold_checked_limits')}`, "
        f"failed=`{exgov.get('review_threshold_failed_limits')}`, "
        f"missing=`{exgov.get('review_threshold_missing_limits')}`"
    )
    threshold_baseline_line = (
        f"limits=`{exgov.get('threshold_limit_count')}`, "
        f"artifacts=`{exgov.get('threshold_artifact_count')}`"
    )
    bench_threshold_line = (
        f"smoke limits=`{bench.get('cpu_smoke_threshold_limits')}`, "
        f"sweep limits=`{bench.get('cpu_sweep_threshold_limits')}`"
    )
    reviewer_q1 = (
        "Are the current real-data proxy tracks "
        "(OOD/noisy-label/EIV/multimodal on real covariates) "
        "sufficient for the near-term product claims?"
    )
    reviewer_q2 = (
        "Are benchmark/example thresholds conservative enough for CI stability "
        "but strict enough to catch real regressions?"
    )
    reviewer_q3 = (
        "Are wrap-native choices consistent with the matrix and parity tests, "
        "or are any remaining custom implementations accidental reinvention?"
    )
    reviewer_q4 = (
        "Are any generated docs/pages still too difficult to review because "
        "they hide important assumptions behind metadata?"
    )

    today = REVIEW_PACKET_MD.stem.split("_")[-1]
    return f"""# Review Readiness Packet ({today})

This page consolidates the highest-value audit and governance artifacts for a deep review pass.

_Generated provenance_: `tools/render_review_packet.py:render_markdown`
_Source artifacts_: `{ADOPTION_AUDIT_JSON.relative_to(REPO_ROOT)}`,
`{COMPARATIVE_EVIDENCE_JSON.relative_to(REPO_ROOT)}`,
`{METHOD_CATALOG_JSON.relative_to(REPO_ROOT)}`,
`{NATIVE_LEVERAGE_JSON.relative_to(REPO_ROOT)}`,
`{EXAMPLE_PROFILE_COMPARE_JSON.relative_to(REPO_ROOT)}`,
`{EXAMPLE_THRESH_VERDICT_JSON.relative_to(REPO_ROOT)}`
_Generated date_: `{today}`

## Audit v1 Status

- Audit v1 closed: `{status.get("closed_v1")}`
- Closeout date: `{status.get("closeout_date")}`
- Closed actionables: `{status.get("closed_actionables")}`
- Deferred v2 backlog: `{status.get("deferred_v2_backlog")}`

## Executive Snapshot

- Adoption audit score (baseline -> provisional): {score_line}
- Full repo mypy status: `{typing.get("total_errors")} errors`
- Docs/example drift checks: {drift_line}
- Examples tracked by audit: `{audit["examples_summary"].get("count")}`
- Comparative evidence coverage (strong-or-better): {evidence_line}
- Method catalog peer methods present (`SWAG`/`BNN`/`MDN`): `{cat.get("peer_uq_methods_present")}`

## Governance Status

- Example summary profile comparison (`audit -> full`): {profile_line}
- Example summary thresholds (full, CI conservative): {threshold_line}
- Example summary thresholds (full, review strict): {review_threshold_line}
- Example summary threshold baselines: {threshold_baseline_line}
- Benchmark threshold baselines (CPU): {bench_threshold_line}
- Benchmark sweep baseline summary (CPU): `{bench.get("cpu_sweep_baseline_summary")}`

## Native Leverage Decisions (Counts)

{decision_lines}

## Review Focus Files

{focus_lines}

## Remaining Evidence/External-Validity Gaps (from Comparative Evidence Matrix)

{gap_lines}

## Reviewer Questions (Suggested)

1. {reviewer_q1}
2. {reviewer_q2}
3. {reviewer_q3}
4. {reviewer_q4}
"""


def write_outputs(
    *,
    md_path: Path = REVIEW_PACKET_MD,
    json_path: Path = REVIEW_PACKET_JSON,
) -> None:
    packet = build_review_packet()
    md = render_markdown(packet)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    write_outputs()
    print(f"Wrote review packet markdown: {REVIEW_PACKET_MD}")
    print(f"Wrote review packet JSON: {REVIEW_PACKET_JSON}")


if __name__ == "__main__":
    main()
