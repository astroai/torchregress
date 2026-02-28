"""End-to-end photo-z RAIL pipeline: collect assets, render summary, merge baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import photoz_rail_assets, render_example_summaries


def _parse_overrides(items: list[str], *, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid {label} override `{item}`. Expected KEY=VALUE format.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid {label} override `{item}`. Empty key/value is not allowed.")
        parsed[key] = value
    return parsed


def run_pipeline(
    *,
    manifest_path: Path,
    output_dir: Path,
    profile: str,
    torchregress_summary_path: Path | None = None,
    merged_output_path: Path | None = None,
    render_torchregress_summary: bool = True,
    allow_download: bool = True,
    overwrite_downloads: bool = False,
    strict_checksums: bool = False,
    paper_parity: bool = True,
    write_manifest_path: Path | None = None,
    materialization_report_path: Path | None = None,
    dataset_urls: dict[str, str] | None = None,
    baseline_urls: dict[str, str] | None = None,
    dataset_paths: dict[str, str] | None = None,
    baseline_paths: dict[str, str] | None = None,
    preset: str = "rail",
    template_path: Path | None = None,
) -> dict[str, Any]:
    manifest = photoz_rail_assets.load_manifest(
        manifest_path,
        preset=preset,
        template_path=template_path,
    )
    photoz_rail_assets.apply_manifest_overrides(
        manifest,
        dataset_urls=dataset_urls,
        baseline_urls=baseline_urls,
        dataset_paths=dataset_paths,
        baseline_paths=baseline_paths,
    )
    materialization = photoz_rail_assets.materialize_manifest_assets(
        manifest,
        allow_download=allow_download,
        overwrite=overwrite_downloads,
        strict_checksums=strict_checksums,
        fail_on_missing=True,
    )

    out_manifest = write_manifest_path or manifest_path
    photoz_rail_assets.write_manifest(out_manifest, manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    if materialization_report_path is not None:
        materialization_report_path.parent.mkdir(parents=True, exist_ok=True)
        materialization_report_path.write_text(
            json.dumps(materialization, indent=2), encoding="utf-8"
        )

    if torchregress_summary_path is None:
        torchregress_summary_path = output_dir / f"photoz_nnc_crps_rail_comparison_{profile}.json"

    if render_torchregress_summary:
        render_example_summaries.render_all(
            profile=profile,
            output_dir=output_dir,
            examples=["photoz_nnc_crps_rail_comparison"],
        )
    elif not torchregress_summary_path.exists():
        raise FileNotFoundError(f"Missing torchregress summary: {torchregress_summary_path}")

    rail_inputs = photoz_rail_assets.collect_baseline_input_paths(manifest, require_required=True)
    if not rail_inputs:
        raise FileNotFoundError("No baseline payload files found in manifest baseline_payloads.")

    merged_output = merged_output_path or (
        output_dir / f"photoz_rail_baseline_comparison_{profile}.json"
    )
    merged_path = render_example_summaries.render_photoz_rail_merge(
        manifest_path=out_manifest,
        torchregress_summary_path=torchregress_summary_path,
        rail_input_paths=rail_inputs,
        output_path=merged_output,
        paper_parity=paper_parity,
    )
    merged_payload = json.loads(merged_path.read_text(encoding="utf-8"))
    merge_analysis = merged_payload.get("analysis", {})

    return {
        "artifact": "photoz_rail_pipeline_report",
        "version": 1,
        "manifest_path": str(out_manifest),
        "torchregress_summary_path": str(torchregress_summary_path),
        "merged_output_path": str(merged_path),
        "baseline_inputs": [str(p) for p in rail_inputs],
        "merge_analysis": merge_analysis,
        "materialization": materialization,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end photo-z RAIL collection + merge.")
    parser.add_argument(
        "--preset",
        choices=["rail", "nnc_crps"],
        default="rail",
        help="Manifest preset used if --manifest does not exist.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/rail/rail_photoz_manifest.json"),
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Optional template path used when --manifest does not exist.",
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        default=None,
        help="Path for updated manifest (defaults to --manifest).",
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "audit", "full"],
        default="full",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/example_summaries"),
    )
    parser.add_argument(
        "--torchregress-summary",
        type=Path,
        default=None,
        help="Existing torchregress summary path. Defaults to output-dir profile path.",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=None,
        help="Optional explicit path for merged output artifact.",
    )
    parser.add_argument(
        "--skip-render-torchregress",
        action="store_true",
        help="Use existing torchregress summary instead of rendering.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download missing assets from manifest URLs.",
    )
    parser.add_argument(
        "--overwrite-downloads",
        action="store_true",
        help="Overwrite existing manifest assets when URLs are available.",
    )
    parser.add_argument(
        "--strict-checksums",
        action="store_true",
        help="Fail if existing checksum_policy values differ from computed values.",
    )
    parser.add_argument(
        "--no-paper-parity",
        action="store_true",
        help="Disable strict dataset/split/core-method parity checks during merge.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/example_summaries/photoz_rail_pipeline_latest.json"),
    )
    parser.add_argument(
        "--materialization-report",
        type=Path,
        default=Path("reports/example_summaries/photoz_rail_materialization_latest.json"),
    )
    parser.add_argument(
        "--dataset-url",
        action="append",
        default=[],
        metavar="KEY=URL",
        help="Override a dataset_files URL without editing manifest (repeatable).",
    )
    parser.add_argument(
        "--baseline-url",
        action="append",
        default=[],
        metavar="METHOD=URL",
        help="Override a baseline_payloads URL without editing manifest (repeatable).",
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Override a dataset_files path without editing manifest (repeatable).",
    )
    parser.add_argument(
        "--baseline-path",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Override a baseline_payloads path without editing manifest (repeatable).",
    )
    args = parser.parse_args()

    report = run_pipeline(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        profile=args.profile,
        torchregress_summary_path=args.torchregress_summary,
        merged_output_path=args.merged_output,
        render_torchregress_summary=not args.skip_render_torchregress,
        allow_download=not args.no_download,
        overwrite_downloads=args.overwrite_downloads,
        strict_checksums=args.strict_checksums,
        paper_parity=not args.no_paper_parity,
        write_manifest_path=args.write_manifest,
        materialization_report_path=args.materialization_report,
        dataset_urls=_parse_overrides(args.dataset_url, label="dataset-url"),
        baseline_urls=_parse_overrides(args.baseline_url, label="baseline-url"),
        dataset_paths=_parse_overrides(args.dataset_path, label="dataset-path"),
        baseline_paths=_parse_overrides(args.baseline_path, label="baseline-path"),
        preset=args.preset,
        template_path=args.template,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote pipeline report: {args.report}")
    print(f"Merged output: {report['merged_output_path']}")
    print(f"Baseline inputs: {len(report['baseline_inputs'])}")


if __name__ == "__main__":
    main()
