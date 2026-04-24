"""Aggregate SAGE-Reg NeurIPS full-run JSONs into sage_paper_report.json + METRICS.md.

Expected layout under ``run_root`` (from ``scripts/run_neurips_sage_reg_full.py``)::

    run_root/
      sage/year_direct/summary.json
      sage/multiseed/multiseed_summary.json
      openml_diamonds/multiseed_summary.json                 # optional (OpenML id 42225)
      year_labeled_sweep/year_labeled_sweep_collated.json   # optional
      multiseed_year_nl2048/multiseed_summary.json          # optional
      catboost/year_catboost_labeled_only.json              # optional
      synthetic/summary.json                                 # optional
      backbone/summary.json                                  # optional
      tabred/bundle_summary.json                             # optional
      ablations/summary.json                                 # optional
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


def build_report(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    warnings: list[str] = []

    sage_direct_path = run_root / "sage" / "year_direct" / "summary.json"
    sage_direct_raw = prc.read_json(sage_direct_path)
    if sage_direct_raw is None:
        warnings.append(f"missing SAGE year direct summary: {sage_direct_path}")
        sage_direct_summary = None
    else:
        sage_direct_summary = prc.summarize_sage_year_direct(sage_direct_raw)

    multiseed_path = run_root / "sage" / "multiseed" / "multiseed_summary.json"
    multiseed_raw = prc.read_json(multiseed_path)
    if multiseed_raw is None:
        warnings.append(f"missing SAGE multiseed summary: {multiseed_path}")
        multiseed_summary = None
    else:
        multiseed_summary = prc.summarize_multiseed(multiseed_raw)

    diamonds_multiseed_path = run_root / "openml_diamonds" / "multiseed_summary.json"
    diamonds_multiseed_raw = prc.read_json(diamonds_multiseed_path)
    if diamonds_multiseed_raw is None:
        diamonds_multiseed_summary = None
    else:
        diamonds_multiseed_summary = prc.summarize_multiseed(diamonds_multiseed_raw)

    collated = prc.read_json(run_root / "year_labeled_sweep" / "year_labeled_sweep_collated.json")
    if collated is None:
        collated = None
    else:
        collated = {"rows": collated.get("rows"), "source_files": collated.get("source_files")}

    nl2048 = prc.read_json(run_root / "multiseed_year_nl2048" / "multiseed_summary.json")
    catboost_year = prc.read_json(run_root / "catboost" / "year_catboost_labeled_only.json")
    catboost_higgs = prc.read_json(run_root / "catboost" / "higgs_catboost_baselines.json")
    synthetic = prc.read_json(run_root / "synthetic" / "summary.json")
    backbone = prc.read_json(run_root / "backbone" / "summary.json")
    tabred = prc.read_json(run_root / "tabred" / "bundle_summary.json")
    ablations = prc.read_json(run_root / "ablations" / "summary.json")
    image_rebuttal = prc.read_json(run_root / "image_rebuttal" / "summary.json")
    run_manifest = prc.read_json(run_root / "neurips_sage_reg_full_manifest.json")

    optional_paths = {
        "openml_diamonds_multiseed": run_root / "openml_diamonds" / "multiseed_summary.json",
        "year_labeled_sweep_collated": run_root
        / "year_labeled_sweep"
        / "year_labeled_sweep_collated.json",
        "multiseed_year_nl2048": run_root / "multiseed_year_nl2048" / "multiseed_summary.json",
        "catboost_year": run_root / "catboost" / "year_catboost_labeled_only.json",
        "catboost_higgs": run_root / "catboost" / "higgs_catboost_baselines.json",
        "synthetic": run_root / "synthetic" / "summary.json",
        "backbone": run_root / "backbone" / "summary.json",
        "tabred": run_root / "tabred" / "bundle_summary.json",
        "ablations": run_root / "ablations" / "summary.json",
        "image_rebuttal": run_root / "image_rebuttal" / "summary.json",
    }
    paths_present = {k: str(p) for k, p in optional_paths.items() if p.is_file()}

    return {
        "artifact": "sage_paper_report",
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "paths": {
            "sage_year_direct": str(sage_direct_path),
            "sage_multiseed": str(multiseed_path),
            **paths_present,
        },
        "sage_year_direct": sage_direct_summary,
        "sage_multiseed": multiseed_summary,
        "openml_diamonds_multiseed": diamonds_multiseed_summary,
        "year_labeled_sweep_collated": collated,
        "multiseed_year_nl2048": nl2048,
        "catboost_year": catboost_year,
        "catboost_higgs": catboost_higgs,
        "synthetic_summary": synthetic,
        "backbone_summary": backbone,
        "tabred_bundle": tabred,
        "ablations": ablations,
        "image_rebuttal": image_rebuttal,
        "neurips_full_manifest": run_manifest,
        "warnings": warnings,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SAGE-Reg full run — metric digest",
        "",
        f"- Generated: `{report.get('generated_at_utc')}`",
        f"- Run root: `{report.get('run_root')}`",
        "",
    ]
    for w in report.get("warnings", []):
        lines.append(f"- **Warning:** {w}")
    if report.get("warnings"):
        lines.append("")

    yd = report.get("sage_year_direct")
    if yd and yd.get("by_method"):
        lines.append("## Year direct (by method × unlabeled fraction)")
        lines.append("")
        for m in ("SupervisedOnly", "MeanTeacher", "ConfidenceWeightedPseudoLabel", "SAGE-Reg"):
            rows = yd["by_method"].get(m, [])
            if not rows:
                continue
            lines.append(f"### {m}")
            lines.append("")
            lines.append("| UnlabeledFraction | NLL | CRPS | Cov90 | RMSE |")
            lines.append("|------------------:|----:|-----:|------:|-----:|")
            for r in rows:
                lines.append(
                    f"| {r.get('UnlabeledFraction')} | {r.get('NLL')} | {r.get('CRPS')} | "
                    f"{r.get('Cov90')} | {r.get('RMSE')} |"
                )
            lines.append("")
        for m in sorted(yd["by_method"]):
            if m in ("SupervisedOnly", "MeanTeacher", "ConfidenceWeightedPseudoLabel", "SAGE-Reg"):
                continue
            rows = yd["by_method"][m]
            lines.append(f"### {m}")
            lines.append("")
            lines.append("| UnlabeledFraction | NLL | CRPS | Cov90 | RMSE |")
            lines.append("|------------------:|----:|-----:|------:|-----:|")
            for r in rows:
                lines.append(
                    f"| {r.get('UnlabeledFraction')} | {r.get('NLL')} | {r.get('CRPS')} | "
                    f"{r.get('Cov90')} | {r.get('RMSE')} |"
                )
            lines.append("")

    ms = report.get("sage_multiseed")
    if ms and ms.get("aggregate"):
        lines.append("## Multiseed (tuned row, aggregate)")
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

    dm = report.get("openml_diamonds_multiseed")
    if dm and dm.get("aggregate"):
        lines.append("## OpenML diamonds multiseed (tuned Year row, aggregate)")
        lines.append("")
        lines.append("| Benchmark | Seeds | SAGE−Sup (mean) | SAGE−Sup (std) | Conf−Sup (mean) |")
        lines.append("|-----------|------:|----------------:|---------------:|----------------:|")
        for row in dm["aggregate"]:
            lines.append(
                f"| {row.get('Benchmark')} | {row.get('Seeds')} | "
                f"{row.get('SAGEMinusSupervisedMean')} | {row.get('SAGEMinusSupervisedStd')} | "
                f"{row.get('ConfidenceMinusSupervisedMean')} |"
            )
        lines.append("")

    coll = report.get("year_labeled_sweep_collated")
    if coll and coll.get("rows"):
        lines.append("## Labeled budget sweep (collated NLL)")
        lines.append("")
        lines.append("| n_labeled | NLL sup | NLL SAGE | Δ(SAGE−sup) |")
        lines.append("|----------:|--------:|---------:|------------:|")
        for row in coll["rows"]:
            lines.append(
                f"| {row.get('n_labeled')} | {row.get('NLL_SupervisedOnly')} | "
                f"{row.get('NLL_SAGE-Reg')} | {row.get('NLL_SAGEMinusSupervised')} |"
            )
        lines.append("")

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
    parser = argparse.ArgumentParser(description="Aggregate SAGE NeurIPS full-run JSON summaries.")
    parser.add_argument(
        "--run-root", type=Path, required=True, help="Output root from run_neurips_sage_reg_full.py"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON (default: <run-root>/sage_paper_report.json)",
    )
    parser.add_argument("--write-markdown", action="store_true", help="Write <run-root>/METRICS.md")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 if year direct or multiseed summaries missing.",
    )
    args = parser.parse_args()
    report = build_report(args.run_root)
    out = args.output or (args.run_root / "sage_paper_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    if args.write_markdown:
        md = args.run_root / "METRICS.md"
        _write_markdown(report, md)
        print(f"Wrote {md}")
    if args.strict and report.get("warnings"):
        for w in report["warnings"]:
            print(f"STRICT: {w}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
