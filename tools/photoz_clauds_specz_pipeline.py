"""One-command pipeline: (optional) build CLAUDS+specz catalog, then run benchmark example."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "clauds_specz"
DEFAULT_CATALOG = DEFAULT_DATA_DIR / "clauds_specz_catalog.parquet"
DEFAULT_REPORT = (
    REPO_ROOT / "reports" / "example_summaries" / "photoz_clauds_specz_pipeline_latest.json"
)
EXAMPLES_DIR = REPO_ROOT / "examples"

PROFILE_SIZES = {
    "smoke": {"n_train": 64, "n_cal": 24, "n_test": 24, "labeled_fractions": (0.1, 0.25, 0.5)},
    "audit": {"n_train": 192, "n_cal": 64, "n_test": 64, "labeled_fractions": (0.1, 0.25, 0.5)},
    "full": {"n_train": 512, "n_cal": 256, "n_test": 256, "labeled_fractions": (0.1, 0.25, 0.5)},
    "ssl_full": {
        "n_train": 2048,
        "n_cal": 512,
        "n_test": 512,
        "labeled_fractions": (0.05, 0.1, 0.25, 0.5, 0.75, 1.0),
    },
}


def _run_clauds_example(
    *,
    profile: str,
    catalog_path: Path,
    output_dir: Path,
) -> Path:
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        module = importlib.import_module("photoz_clauds_specz_comparison")
        from comparison_utils import write_comparison_summary_json
    finally:
        sys.path.pop(0)

    sizes = PROFILE_SIZES.get(profile, PROFILE_SIZES["smoke"])
    labeled_fractions = sizes.get("labeled_fractions", (0.1, 0.25, 0.5))
    summary_path = output_dir / f"photoz_clauds_specz_comparison_{profile}.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, notes = module.run_comparison(
        catalog_path=catalog_path,
        n_train=sizes["n_train"],
        n_cal=sizes["n_cal"],
        n_test=sizes["n_test"],
        seed=240226,
        labeled_fractions=labeled_fractions,
        label_policy="highz_scarce",
        split_policy="stratified_redshift",
        report_counts=True,
        repo_root=REPO_ROOT,
    )
    write_comparison_summary_json(
        str(summary_path),
        example="examples/photoz_clauds_specz_comparison.py",
        task="CLAUDS+Spec-z semi-supervised photometric redshift comparison",
        config={"profile": profile, **sizes},
        rows=rows,
        notes=notes,
    )
    return summary_path


def run_pipeline(
    *,
    profile: str = "smoke",
    catalog_path: Path = DEFAULT_CATALOG,
    output_dir: Path | None = None,
    report_path: Path = DEFAULT_REPORT,
    build_if_missing: bool = False,
    raw_dir: Path | None = None,
    merged_path: Path | None = None,
    specz_fits: Path | None = None,
) -> dict[str, Any]:
    """
    Run CLAUDS+specz benchmark. If catalog_path does not exist and build_if_missing,
    run download -> merge -> crossmatch (requires CLAUDS FITS and spec-z FITS).
    """
    catalog_path = Path(catalog_path)
    output_dir = output_dir or REPO_ROOT / "reports" / "example_summaries"
    output_dir = Path(output_dir)
    raw_dir = raw_dir or DEFAULT_DATA_DIR / "raw"
    merged_path = merged_path or DEFAULT_DATA_DIR / "clauds_merged_photometry.parquet"
    specz_fits = (
        specz_fits
        or DEFAULT_DATA_DIR / "specz_compilation" / "specz_compilation_COSMOS_DR1.1_unique.fits"
    )

    if not catalog_path.exists() and build_if_missing:
        from tools import clauds_download, clauds_merge, clauds_specz_crossmatch

        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        clauds_download.download_clauds(output_dir=raw_dir, hscpipe=True, overwrite=False)
        merged_out = clauds_merge.merge_clauds(
            raw_dir=raw_dir,
            output_dir=catalog_path.parent,
            merged_filename=Path(merged_path).name,
        )
        if not Path(specz_fits).exists():
            raise FileNotFoundError(
                f"Spec-z FITS required for crossmatch: {specz_fits}. "
                "Run tools/specz_compilation_download.py with --url or --local-path."
            )
        clauds_specz_crossmatch.crossmatch_clauds_specz(
            merged_parquet=merged_out,
            specz_fits=Path(specz_fits),
            output_path=catalog_path,
        )
        # Validate the built catalog for consistency
        from tools import clauds_catalog_validate

        validation = clauds_catalog_validate.validate_clauds_specz_catalog(catalog_path)
        if not validation["ok"]:
            raise RuntimeError(
                "CLAUDS+spec-z catalog validation failed after build: "
                + "; ".join(validation["errors"])
            )
    elif not catalog_path.exists():
        raise FileNotFoundError(
            f"CLAUDS+specz catalog not found: {catalog_path}. "
            "Build it (run tools/clauds_download.py, clauds_merge.py, clauds_specz_crossmatch.py) "
            "or use --build-if-missing (requires CLAUDS and spec-z FITS)."
        )

    summary_path = _run_clauds_example(
        profile=profile, catalog_path=catalog_path, output_dir=output_dir
    )

    report = {
        "artifact": "photoz_clauds_specz_pipeline_report",
        "version": 1,
        "profile": profile,
        "catalog_path": str(catalog_path),
        "summary_path": str(summary_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CLAUDS+Spec-z pipeline: optional build, then benchmark."
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "audit", "full", "ssl_full"],
        default="smoke",
        help="ssl_full: 2048/512/512 train/cal/test, 6 labeled fractions, all SSL variants.",
    )
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--build-if-missing",
        action="store_true",
        help="Run download/merge/crossmatch if catalog missing.",
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--merged-path", type=Path, default=None)
    parser.add_argument("--specz-fits", type=Path, default=None)
    args = parser.parse_args()

    report = run_pipeline(
        profile=args.profile,
        catalog_path=args.catalog_path,
        output_dir=args.output_dir,
        report_path=args.report,
        build_if_missing=args.build_if_missing,
        raw_dir=args.raw_dir,
        merged_path=args.merged_path,
        specz_fits=args.specz_fits,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
