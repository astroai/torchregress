"""Audit docs markdown for LaTeX rendering and paragraph/section cohesion."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
AUDIT_JSON = REPO_ROOT / "reports" / "docs_quality_audit.json"

DONE_FILES = {
    "docs/getting-started/concepts.md",
    "docs/javascripts/mathjax.js",
    "docs/losses/gaussian.md",
    "docs/metrics/decision.md",
    "docs/methods/ensemble/index.md",
    "docs/methods/conformal/index.md",
    "docs/examples/basic_usage.md",
    "docs/examples/loss_comparison.md",
    "docs/examples/evidential_regression.md",
    "docs/examples/ood_selective_prediction_comparison.md",
}

SKIPPED_FILES = {
    "docs/reports/method_catalog_generated.md",
    "docs/reports/comparative_evidence_matrix.md",
    "docs/reports/real_data_recommendation_guide.md",
    "docs/reports/docs_quality_audit.md",
}

HAS_MATH_RE = re.compile(r"\$[^$]+\$|\$\$")


@dataclass
class Issue:
    line: int
    rule: str
    severity: str
    message: str


@dataclass
class FileAudit:
    path: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warn_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warn")


def _strip_code_fences(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _has_math(text: str) -> bool:
    return bool(HAS_MATH_RE.search(text))


def _count_single_dollars(line: str) -> int:
    count = 0
    i = 0
    while i < len(line):
        if line[i] == "$":
            if i + 1 < len(line) and line[i + 1] == "$":
                i += 2
                continue
            if i > 0 and line[i - 1] == "\\":
                i += 1
                continue
            count += 1
        i += 1
    return count


def audit_file(path: Path) -> FileAudit:
    rel = path.relative_to(REPO_ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    result = FileAudit(path=rel)

    if path.suffix == ".js":
        return result

    prose = _strip_code_fences(text)
    dollar_count = 0
    i = 0
    while i < len(prose):
        if prose[i] == "$":
            if i + 1 < len(prose) and prose[i + 1] == "$":
                i += 2
                continue
            if i > 0 and prose[i - 1] == "\\":
                i += 1
                continue
            dollar_count += 1
        i += 1
    if dollar_count % 2 != 0:
        result.issues.append(
            Issue(
                line=0,
                rule="unbalanced_dollar",
                severity="error",
                message=f"Unbalanced single '$' delimiters in file ({dollar_count} singles)",
            )
        )

    display_delims = prose.count("$$")
    if display_delims % 2 != 0:
        result.issues.append(
            Issue(
                line=0,
                rule="display_math_layout",
                severity="error",
                message=f"Unpaired '$$' display-math delimiters ({display_delims} total)",
            )
        )

    in_code = False
    current_section_has_star = False
    current_section_has_dash = False
    section_start = 1

    for ln, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue

        if re.match(r"^#{1,6}\s+", line):
            if current_section_has_star and current_section_has_dash:
                result.issues.append(
                    Issue(
                        line=section_start,
                        rule="mixed_list_markers",
                        severity="warn",
                        message="Section mixes '*' and '-' list markers",
                    )
                )
            current_section_has_star = False
            current_section_has_dash = False
            section_start = ln

            title = re.sub(r"^#{1,6}\s+", "", line)
            if "$" in title and (re.search(r"[a-z]{2,}[A-Z][a-z]", title) or len(title) > 90):
                result.issues.append(
                    Issue(
                        line=ln,
                        rule="heading_glue",
                        severity="error",
                        message=f"Heading may contain glued body text: {title[:80]}",
                    )
                )

        if stripped.endswith((",", ".")) and stripped.startswith("$$") and stripped.endswith("$$"):
            result.issues.append(
                Issue(
                    line=ln,
                    rule="punctuation_in_display_math",
                    severity="warn",
                    message="Trailing punctuation inside display math",
                )
            )

        if "\\text{stop\\_gradient}" in line or "\\text{stop_gradient}" in line:
            result.issues.append(
                Issue(
                    line=ln,
                    rule="latex_style",
                    severity="warn",
                    message="Prefer \\operatorname{sg} over \\text{stop\\_gradient}",
                )
            )

        if re.search(r"(?<![\\$])\[[0-9]+\](?![(\[])", line):
            if not line.strip().startswith("|") and not re.search(r"\]\(", line):
                if "`" not in line or line.count("`") < 2:
                    result.issues.append(
                        Issue(
                            line=ln,
                            rule="citation_escape",
                            severity="warn",
                            message="Unescaped citation marker [N]; use \\[N\\] in prose",
                        )
                    )

        if ln >= 2 and (line.startswith("- ") or line.startswith("* ")):
            prev = lines[ln - 2]
            prev_stripped = prev.strip()
            if prev_stripped.endswith("$$") or (
                prev_stripped.count("$$") == 2 and "$$" in prev_stripped
            ):
                if prev.startswith((" ", "\t")) or prev.lstrip().startswith(("-", "*")):
                    pass
                elif not prev_stripped.endswith(":"):
                    result.issues.append(
                        Issue(
                            line=ln,
                            rule="list_after_math",
                            severity="warn",
                            message="List immediately after display math without bridge sentence",
                        )
                    )

        if line.startswith("* "):
            current_section_has_star = True
        if line.startswith("- "):
            current_section_has_dash = True

    return result


def discover_doc_files() -> list[Path]:
    md_files = sorted(DOCS_DIR.rglob("*.md"))
    extra = DOCS_DIR / "javascripts" / "mathjax.js"
    files = md_files[:]
    if extra.exists():
        files.append(extra)
    return sorted(set(files), key=lambda p: p.relative_to(REPO_ROOT).as_posix())


def load_tracker() -> dict[str, dict]:
    if AUDIT_JSON.exists():
        return json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    return {}


def save_tracker(data: dict[str, dict]) -> None:
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def init_tracker(files: list[Path]) -> dict[str, dict]:
    tracker = load_tracker()
    today = date.today().isoformat()
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in tracker:
            continue
        if rel in DONE_FILES:
            tracker[rel] = {
                "status": "done",
                "has_math": _has_math(path.read_text(encoding="utf-8"))
                if path.suffix == ".md"
                else False,
                "issues_found": 0,
                "issues_fixed": 0,
                "notes": "pre-reviewed",
                "reviewed_at": today,
            }
        elif rel in SKIPPED_FILES:
            tracker[rel] = {
                "status": "skipped",
                "has_math": _has_math(path.read_text(encoding="utf-8")),
                "issues_found": 0,
                "issues_fixed": 0,
                "notes": "auto-generated report",
                "reviewed_at": today,
            }
        else:
            tracker[rel] = {
                "status": "pending",
                "has_math": _has_math(path.read_text(encoding="utf-8"))
                if path.suffix == ".md"
                else False,
                "issues_found": 0,
                "issues_fixed": 0,
                "notes": "",
                "reviewed_at": "",
            }
    save_tracker(tracker)
    return tracker


def render_markdown(tracker: dict[str, dict], audits: list[FileAudit]) -> str:
    audit_by_path = {a.path: a for a in audits}
    done = sum(1 for v in tracker.values() if v["status"] == "done")
    skipped = sum(1 for v in tracker.values() if v["status"] == "skipped")
    pending = sum(1 for v in tracker.values() if v["status"] == "pending")
    total = len(tracker)
    errors = sum(a.error_count for a in audits)
    warns = sum(a.warn_count for a in audits)

    lines = [
        "# Docs Quality Audit",
        "",
        f"_Generated by `tools/audit_docs_quality.py` on {date.today().isoformat()}._",
        "",
        "## Summary",
        "",
        f"- **Tracked files**: {total}",
        f"- **Done**: {done}",
        f"- **Skipped**: {skipped}",
        f"- **Pending**: {pending}",
        f"- **Open errors**: {errors}",
        f"- **Open warnings**: {warns}",
        "",
        "## Per-file status",
        "",
        "| File | Status | Math | Errors | Warnings | Notes |",
        "|:-----|:-------|:-----|-------:|---------:|:------|",
    ]
    for rel in sorted(tracker):
        entry = tracker[rel]
        audit = audit_by_path.get(rel)
        err = audit.error_count if audit else 0
        warn = audit.warn_count if audit else 0
        notes = entry.get("notes", "").replace("|", "\\|")
        lines.append(
            f"| `{rel}` | {entry['status']} | "
            f"{'yes' if entry.get('has_math') else 'no'} | {err} | {warn} | {notes} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="Audit a single file under repo root")
    parser.add_argument("--init", action="store_true", help="Initialize tracker JSON")
    parser.add_argument("--mark-done", type=str, help="Mark a file done in tracker")
    parser.add_argument("--notes", type=str, default="", help="Notes for --mark-done")
    parser.add_argument("--mark-all-done", action="store_true", help="Mark all pending files done")
    parser.add_argument("--markdown-out", type=Path, help="Write human-readable audit report")
    parser.add_argument(
        "--errors-only", action="store_true", help="Only print error-severity issues"
    )
    args = parser.parse_args()

    files = discover_doc_files()
    if args.init or not AUDIT_JSON.exists():
        init_tracker(files)

    tracker = load_tracker()

    if args.mark_done:
        rel = args.mark_done.replace("\\", "/")
        if rel not in tracker:
            tracker[rel] = {
                "status": "done",
                "has_math": False,
                "issues_found": 0,
                "issues_fixed": 0,
                "notes": args.notes,
                "reviewed_at": date.today().isoformat(),
            }
        else:
            tracker[rel]["status"] = "done"
            tracker[rel]["notes"] = args.notes or tracker[rel].get("notes", "")
            tracker[rel]["reviewed_at"] = date.today().isoformat()
        save_tracker(tracker)
        print(f"Marked done: {rel}")
        return 0

    if args.file:
        target = args.file if args.file.is_absolute() else REPO_ROOT / args.file
        audit_paths = [target]
    else:
        audit_paths = files

    audits: list[FileAudit] = []
    total_errors = 0
    total_warns = 0
    for path in audit_paths:
        if not path.exists() or path.suffix not in {".md", ".js"}:
            continue
        audit = audit_file(path)
        audits.append(audit)
        total_errors += audit.error_count
        total_warns += audit.warn_count
        if audit.issues:
            print(f"\n{audit.path} ({audit.error_count} errors, {audit.warn_count} warnings)")
            for issue in audit.issues:
                if args.errors_only and issue.severity != "error":
                    continue
                print(f"  L{issue.line} [{issue.severity}] {issue.rule}: {issue.message}")

    print(f"\nAudited {len(audits)} file(s): {total_errors} error(s), {total_warns} warning(s)")

    if args.mark_all_done:
        today = date.today().isoformat()
        for rel, entry in tracker.items():
            if entry.get("status") in {"pending", "in_progress"}:
                entry["status"] = "done"
                entry["reviewed_at"] = today
                if not entry.get("notes"):
                    entry["notes"] = "batch review"
        save_tracker(tracker)
        print(f"Marked {sum(1 for v in tracker.values() if v['status'] == 'done')} files done")
        return 0

    if args.markdown_out:
        out = (
            args.markdown_out if args.markdown_out.is_absolute() else REPO_ROOT / args.markdown_out
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(tracker, audits), encoding="utf-8")
        print(f"Wrote {out.relative_to(REPO_ROOT)}")

    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
