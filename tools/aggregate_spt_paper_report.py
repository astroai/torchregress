"""Aggregate SPT-Reg NeurIPS full-run JSONs into spt_paper_report.json + METRICS.md.

Expected layout under ``run_root``::

    run_root/
      full/artifact_manifest.json
      full/year_competing_methods_full.json
      audit/artifact_manifest.json
      audit/year_competing_methods_audit.json
      stage_a_sweep/stage_a_sweep.json   # optional
      large_tabular/year_competing_methods_full.json  # optional (legacy: yolanda/)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import paper_report_common as prc  # noqa: E402


def _load_profile(
    run_root: Path, name: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    sub = run_root / name
    manifest_path = sub / "artifact_manifest.json"
    manifest = prc.read_json(manifest_path)
    if manifest is None:
        warnings.append(f"missing SPT manifest: {manifest_path}")
        return None, None, warnings
    profile = str(manifest.get("profile") or name)
    year_name = f"year_competing_methods_{profile}.json"
    year_path = sub / year_name
    if not year_path.is_file():
        year_path = sub / "year_competing_methods_full.json"
    year_raw = prc.read_json(year_path)
    if year_raw is None:
        warnings.append(f"missing SPT year summary: {sub}/{year_name}")
        return manifest, None, warnings
    return manifest, prc.summarize_spt_year(year_raw), warnings


def build_report(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    all_warnings: list[str] = []

    full_manifest, full_year, w1 = _load_profile(run_root, "full")
    all_warnings.extend(w1)
    audit_manifest, audit_year, w2 = _load_profile(run_root, "audit")
    all_warnings.extend(w2)

    stage_a = prc.read_json(run_root / "stage_a_sweep" / "stage_a_sweep.json")
    lt_raw = prc.read_json(run_root / "large_tabular" / "year_competing_methods_full.json")
    if lt_raw is None:
        lt_raw = prc.read_json(run_root / "yolanda" / "year_competing_methods_full.json")
    large_tabular_summary = prc.summarize_spt_year(lt_raw) if lt_raw else None
    run_manifest = prc.read_json(run_root / "neurips_spt_reg_full_manifest.json")

    spt_other: dict[str, Any] = {}
    if isinstance(full_manifest, dict):
        summaries = full_manifest.get("summaries")
        if isinstance(summaries, dict):
            spt_other["full_summaries"] = summaries
    if isinstance(audit_manifest, dict):
        summaries = audit_manifest.get("summaries")
        if isinstance(summaries, dict):
            spt_other["audit_summaries"] = summaries

    return {
        "artifact": "spt_paper_report",
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "paths": {
            "full_manifest": str(run_root / "full" / "artifact_manifest.json"),
            "audit_manifest": str(run_root / "audit" / "artifact_manifest.json"),
        },
        "spt_year_full": full_year,
        "spt_year_audit": audit_year,
        "spt_manifest_full": full_manifest,
        "spt_manifest_audit": audit_manifest,
        "spt_other_summaries": spt_other,
        "stage_a_sweep": stage_a,
        "large_tabular_year": large_tabular_summary,
        "yolanda_year": large_tabular_summary,
        "neurips_full_manifest": run_manifest,
        "warnings": all_warnings,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SPT-Reg full run — metric digest",
        "",
        f"- Generated: `{report.get('generated_at_utc')}`",
        f"- Run root: `{report.get('run_root')}`",
        "",
    ]
    for w in report.get("warnings", []):
        lines.append(f"- **Warning:** {w}")
    if report.get("warnings"):
        lines.append("")

    def _methods_section(title: str, spt: dict[str, Any] | None) -> None:
        if not spt or not spt.get("methods"):
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Method | NLL | CRPS | Cov90 | Width90 |")
        lines.append("|--------|----:|-----:|------:|--------:|")
        methods = spt["methods"]
        priority = (
            "SourceGaussian",
            "SPTRegGaussian",
            "TargetRefitSmallGaussian",
            "RawSplitConformalGaussian",
            "WeightedSplitConformalGaussian",
        )
        for name in list(priority) + sorted(m for m in methods if m not in priority):
            if name not in methods:
                continue
            r = methods[name]
            lines.append(
                f"| {name} | {r.get('NLL')} | {r.get('CRPS')} | "
                f"{r.get('Cov90')} | {r.get('Width90')} |"
            )
        lines.append("")

    _methods_section("Large-tabular track (profile full)", report.get("spt_year_full"))
    _methods_section("Large-tabular track (profile audit)", report.get("spt_year_audit"))

    lt = report.get("large_tabular_year") or report.get("yolanda_year")
    if lt and lt.get("methods"):
        _methods_section("Extra-large OpenML regression track", lt)

    nm = report.get("neurips_full_manifest")
    if isinstance(nm, dict) and nm.get("phases"):
        lines.append("## Run manifest (phase paths)")
        lines.append("")
        for k, v in nm["phases"].items():
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SPT NeurIPS full-run JSON summaries.")
    parser.add_argument(
        "--run-root", type=Path, required=True, help="Output root from run_neurips_spt_reg_full.py"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON (default: <run-root>/spt_paper_report.json)",
    )
    parser.add_argument("--write-markdown", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit 2 if full year track missing")
    args = parser.parse_args()
    report = build_report(args.run_root)
    out = args.output or (args.run_root / "spt_paper_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    if args.write_markdown:
        md = args.run_root / "METRICS.md"
        _write_markdown(report, md)
        print(f"Wrote {md}")
    if args.strict:
        if report.get("spt_year_full") is None:
            print("STRICT: missing full year track", file=sys.stderr)
            raise SystemExit(2)


if __name__ == "__main__":
    main()
