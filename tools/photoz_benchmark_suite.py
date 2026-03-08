"""Run the local photo-z benchmark suite and optional RAIL merge."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

try:
    from tools import photoz_rail_pipeline, render_example_summaries, render_photoz_benchmark_report
except ModuleNotFoundError:  # pragma: no cover - script execution path
    import photoz_rail_pipeline  # type: ignore[no-redef]
    import render_example_summaries  # type: ignore[no-redef]
    import render_photoz_benchmark_report  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
DEFAULT_OUTPUT_DIR = Path("reports/example_summaries")
DEFAULT_MANIFEST = Path("data/rail/rail_photoz_manifest.json")
DEFAULT_MARKDOWN_REPORT = DEFAULT_OUTPUT_DIR / "photoz_benchmark_suite_latest.md"

CORE_EXAMPLES = [
    "photoz_benchmark_comparison",
    "photoz_nnc_crps_rail_comparison",
    "ppi_photoz_inference_comparison",
]
REAL_DATA_EXAMPLES = [
    "photoz_benchmark_comparison",
    "photoz_nnc_crps_rail_comparison",
]


def _import_example_module(module_name: str) -> Any:
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.pop(0)


def _example_config(
    module_name: str,
    profile: str,
    *,
    real_data_only: bool,
    dataset_path: Path | None = None,
) -> Any:
    if module_name == "photoz_benchmark_comparison":
        module = _import_example_module(module_name)
        cfg = render_example_summaries._photoz_benchmark_config(module, profile)
        if dataset_path is not None:
            return replace(
                cfg,
                dataset_path=str(dataset_path),
                force_simulated=False,
                require_real_data=True,
            )
        if real_data_only:
            return replace(cfg, force_simulated=False, require_real_data=True)
        return cfg
    if module_name == "photoz_nnc_crps_rail_comparison":
        module = _import_example_module(module_name)
        cfg = render_example_summaries._photoz_nnc_config(module, profile)
        if dataset_path is not None:
            return replace(
                cfg,
                dataset_path=str(dataset_path),
                force_simulated=False,
                require_real_data=True,
            )
        if real_data_only:
            return replace(cfg, force_simulated=False, require_real_data=True)
        return cfg
    if module_name == "ppi_photoz_inference_comparison":
        module = _import_example_module(module_name)
        return render_example_summaries._ppi_photoz_config(module, profile)
    raise ValueError(f"Unsupported photo-z suite example: {module_name}")


def _run_example(
    module_name: str,
    *,
    profile: str,
    output_dir: Path,
    real_data_only: bool,
    dataset_path: Path | None = None,
) -> Path:
    module = _import_example_module(module_name)
    cfg = _example_config(
        module_name,
        profile,
        real_data_only=real_data_only,
        dataset_path=dataset_path,
    )
    output_path = output_dir / f"{module_name}_{profile}.json"
    module.main(cfg, summary_json_path=str(output_path))
    return output_path


def run_suite(
    *,
    profile: str,
    output_dir: Path,
    include_rail_merge: bool = False,
    rail_preset: str = "nnc_crps",
    manifest_path: Path = DEFAULT_MANIFEST,
    allow_download: bool = False,
    overwrite_downloads: bool = False,
    strict_checksums: bool = False,
    paper_parity: bool = True,
    real_data_only: bool = False,
    dataset_path: Path | None = None,
    markdown_report_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    example_names = REAL_DATA_EXAMPLES if real_data_only else CORE_EXAMPLES
    summary_paths = [
        _run_example(
            name,
            profile=profile,
            output_dir=output_dir,
            real_data_only=real_data_only,
            dataset_path=dataset_path,
        )
        for name in example_names
    ]
    suite_rows = {path.stem.replace(f"_{profile}", ""): str(path) for path in summary_paths}
    skipped_examples = [name for name in CORE_EXAMPLES if name not in example_names]

    report: dict[str, Any] = {
        "artifact": "photoz_benchmark_suite_report",
        "version": 1,
        "profile": profile,
        "output_dir": str(output_dir),
        "core_examples": list(example_names),
        "real_data_only": real_data_only,
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "skipped_examples": skipped_examples,
        "summary_paths": suite_rows,
        "recommended_read_order": list(example_names),
        "rail_merge": None,
        "markdown_report_path": None,
    }

    if include_rail_merge:
        rail_report_path = output_dir / "photoz_rail_pipeline_suite_latest.json"
        rail_materialization_path = output_dir / "photoz_rail_materialization_suite_latest.json"
        rail_report = photoz_rail_pipeline.run_pipeline(
            manifest_path=manifest_path,
            output_dir=output_dir,
            profile=profile,
            torchregress_summary_path=(
                output_dir / f"photoz_nnc_crps_rail_comparison_{profile}.json"
            ),
            render_torchregress_summary=False,
            allow_download=allow_download,
            overwrite_downloads=overwrite_downloads,
            strict_checksums=strict_checksums,
            paper_parity=paper_parity,
            write_manifest_path=manifest_path,
            materialization_report_path=rail_materialization_path,
            preset=rail_preset,
        )
        rail_report_path.write_text(json.dumps(rail_report, indent=2), encoding="utf-8")
        report["rail_merge"] = {
            "preset": rail_preset,
            "manifest_path": str(manifest_path),
            "report_path": str(rail_report_path),
            "materialization_report_path": str(rail_materialization_path),
            "merged_output_path": rail_report["merged_output_path"],
        }

    markdown_path = markdown_report_path or (output_dir / DEFAULT_MARKDOWN_REPORT.name)
    suite_report_path = output_dir / "photoz_benchmark_suite_latest.json"
    rendered = render_photoz_benchmark_report.render_report(
        suite_report_path=_write_suite_report(report, suite_report_path),
        output_path=markdown_path,
    )
    report["markdown_report_path"] = str(rendered)

    return report


def _write_suite_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local photo-z benchmark suite.")
    parser.add_argument("--profile", choices=["smoke", "audit", "full"], default="full")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-rail-merge",
        action="store_true",
        help="Also run the manifest-driven RAIL merge against the ordered-bin summary.",
    )
    parser.add_argument(
        "--rail-preset",
        choices=["rail", "nnc_crps"],
        default="nnc_crps",
        help="Preset used when bootstrapping a missing RAIL manifest.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow the optional RAIL merge to download missing manifest assets.",
    )
    parser.add_argument("--overwrite-downloads", action="store_true")
    parser.add_argument("--strict-checksums", action="store_true")
    parser.add_argument("--no-paper-parity", action="store_true")
    parser.add_argument(
        "--real-data-only",
        action="store_true",
        help=(
            "Require real photo-z data for the real-data-capable tracks and "
            "skip synthetic-only PPI."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "photoz_benchmark_suite_latest.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=DEFAULT_MARKDOWN_REPORT,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Explicit photo-z dataset file for the standard and ordered-bin tracks.",
    )
    args = parser.parse_args()

    report = run_suite(
        profile=args.profile,
        output_dir=args.output_dir,
        include_rail_merge=args.include_rail_merge,
        rail_preset=args.rail_preset,
        manifest_path=args.manifest,
        allow_download=args.allow_download,
        overwrite_downloads=args.overwrite_downloads,
        strict_checksums=args.strict_checksums,
        paper_parity=not args.no_paper_parity,
        real_data_only=args.real_data_only,
        dataset_path=args.dataset_path,
        markdown_report_path=args.markdown_report,
    )
    _write_suite_report(report, args.report)
    print(f"Wrote photo-z suite report: {args.report}")
    for name, path in report["summary_paths"].items():
        print(f"- {name}: {path}")
    if report["markdown_report_path"] is not None:
        print(f"- markdown_report: {report['markdown_report_path']}")
    if report["rail_merge"] is not None:
        print(f"- rail_merge: {report['rail_merge']['merged_output_path']}")


if __name__ == "__main__":
    main()
