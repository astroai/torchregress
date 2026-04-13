"""Collect SAGE + SPT tabular benchmark JSONs into one assessment report.

Expected layout (written by ``scripts/run_tabular_paper_bundle.sh``)::

    bundle_dir/
      bundle_meta.json              # optional; env + timing from shell
      sage/year_direct/summary.json
      sage/multiseed/multiseed_summary.json
      spt/full/year_competing_methods_full.json
      spt/full/artifact_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools import paper_report_common as prc


def build_report(bundle_dir: Path) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    warnings: list[str] = []

    meta_path = bundle_dir / "bundle_meta.json"
    meta = prc.read_json(meta_path)
    spt_profile = "full"
    if isinstance(meta, dict):
        spt_profile = str(meta.get("spt_profile") or "full")

    sage_direct_path = bundle_dir / "sage" / "year_direct" / "summary.json"
    sage_direct_raw = prc.read_json(sage_direct_path)
    if sage_direct_raw is None:
        warnings.append(f"missing SAGE year direct summary: {sage_direct_path}")
        sage_direct_summary = None
    else:
        sage_direct_summary = prc.summarize_sage_year_direct(sage_direct_raw)

    multiseed_path = bundle_dir / "sage" / "multiseed" / "multiseed_summary.json"
    multiseed_raw = prc.read_json(multiseed_path)
    if multiseed_raw is None:
        warnings.append(f"missing SAGE multiseed summary: {multiseed_path}")
        multiseed_summary = None
    else:
        multiseed_summary = prc.summarize_multiseed(multiseed_raw)

    spt_path = bundle_dir / "spt" / "full" / f"year_competing_methods_{spt_profile}.json"
    spt_raw = prc.read_json(spt_path)
    if spt_raw is None:
        warnings.append(f"missing SPT year full summary: {spt_path}")
        spt_summary = None
    else:
        spt_summary = prc.summarize_spt_year(spt_raw)

    manifest_path = bundle_dir / "spt" / "full" / "artifact_manifest.json"
    manifest = prc.read_json(manifest_path)
    if manifest is None:
        warnings.append(f"missing SPT artifact manifest: {manifest_path}")

    spt_other_summaries: dict[str, str] = {}
    if isinstance(manifest, dict):
        summaries = manifest.get("summaries")
        if isinstance(summaries, dict):
            for key, val in summaries.items():
                if key != "tabular_large" and isinstance(val, str):
                    spt_other_summaries[str(key)] = val

    return {
        "artifact": "tabular_paper_bundle_report",
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle_dir),
        "bundle_meta": meta,
        "spt_other_summaries": spt_other_summaries,
        "paths": {
            "sage_year_direct": str(sage_direct_path),
            "sage_multiseed": str(multiseed_path),
            "spt_year_track": str(spt_path),
            "spt_profile": spt_profile,
            "spt_manifest": str(manifest_path),
        },
        "sage_year_direct": sage_direct_summary,
        "sage_multiseed": multiseed_summary,
        "spt_year_full": spt_summary,
        "spt_artifact_manifest": manifest,
        "warnings": warnings,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Tabular paper bundle — metric digest",
        "",
        f"- Generated: `{report.get('generated_at_utc')}`",
        f"- Bundle dir: `{report.get('bundle_dir')}`",
        "",
    ]
    for w in report.get("warnings", []):
        lines.append(f"- **Warning:** {w}")
    if report.get("warnings"):
        lines.append("")

    yd = report.get("sage_year_direct")
    if yd and yd.get("by_method"):
        lines.append("## SAGE year direct (by method × unlabeled fraction)")
        lines.append("")
        want = ("SupervisedOnly", "ConfidenceWeightedPseudoLabel", "SAGE-Reg")
        for m in want:
            rows = yd["by_method"].get(m, [])
            if not rows:
                continue
            lines.append(f"### {m}")
            lines.append("")
            lines.append("| UnlabeledFraction | NLL | CRPS | Cov90 | CalibMAE | RMSE |")
            lines.append("|------------------:|----:|-----:|------:|---------:|-----:|")
            for r in rows:
                lines.append(
                    f"| {r.get('UnlabeledFraction')} | {r.get('NLL')} | {r.get('CRPS')} | "
                    f"{r.get('Cov90')} | {r.get('CalibMAE')} | {r.get('RMSE')} |"
                )
            lines.append("")

    ms = report.get("sage_multiseed")
    if ms and ms.get("aggregate"):
        lines.append("## SAGE multiseed (supervised gap, aggregate over seeds)")
        lines.append("")
        lines.append("| Benchmark | Seeds | SAGE−Sup (mean) | SAGE−Sup (std) | Conf−Sup (mean) |")
        lines.append("|-----------|------:|----------------:|---------------:|----------------:|")
        for row in ms["aggregate"]:
            lines.append(
                f"| {row.get('Benchmark')} | {row.get('Seeds')} | "
                f"{row.get('SAGEMinusSupervisedMean')} | {row.get('SAGEMinusSupervisedStd')} | "
                f"{row.get('ConfidenceMinusSupervisedMean')} |"
            )
        lines.append("")

    extras = report.get("spt_other_summaries") or {}
    if extras:
        lines.append("## Other SPT summaries (paths from artifact manifest)")
        lines.append("")
        for k, v in sorted(extras.items()):
            lines.append(f"- **{k}:** `{v}`")
        lines.append("")

    spt = report.get("spt_year_full")
    if spt and spt.get("methods"):
        lines.append("## SPT large-tabular track (Gaussian family, key methods)")
        lines.append("")
        lines.append("| Method | NLL | CRPS | Cov90 | Width90 |")
        lines.append("|--------|----:|-----:|------:|--------:|")
        priority = (
            "SourceGaussian",
            "SPTRegGaussian",
            "TargetRefitSmallGaussian",
            "RawSplitConformalGaussian",
        )
        methods = spt["methods"]
        for name in priority:
            if name not in methods:
                continue
            r = methods[name]
            lines.append(
                f"| {name} | {r.get('NLL')} | {r.get('CRPS')} | "
                f"{r.get('Cov90')} | {r.get('Width90')} |"
            )
        for name in sorted(methods):
            if name in priority:
                continue
            r = methods[name]
            lines.append(
                f"| {name} | {r.get('NLL')} | {r.get('CRPS')} | "
                f"{r.get('Cov90')} | {r.get('Width90')} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate SAGE + SPT tabular JSON summaries under bundle_dir."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Output root from run_tabular_paper_bundle.sh",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write combined JSON (default: <bundle-dir>/tabular_paper_bundle_report.json)",
    )
    parser.add_argument(
        "--write-markdown",
        action="store_true",
        help="Also write <bundle-dir>/METRICS.md (human-readable digest).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if any expected summary file is missing.",
    )
    args = parser.parse_args()
    bundle_dir: Path = args.bundle_dir
    report = build_report(bundle_dir)

    out_json = args.output or (bundle_dir / "tabular_paper_bundle_report.json")
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")

    if args.write_markdown:
        md_path = bundle_dir / "METRICS.md"
        _write_markdown(report, md_path)
        print(f"Wrote {md_path}")

    if args.strict and report.get("warnings"):
        for w in report["warnings"]:
            print(f"STRICT: {w}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
