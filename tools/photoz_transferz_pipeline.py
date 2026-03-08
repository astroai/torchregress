"""Download, normalize, and benchmark the TransferZ tabular photo-z release."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools import photoz_benchmark_suite, photoz_collect_real_data
except ModuleNotFoundError:  # pragma: no cover - script execution path
    import photoz_benchmark_suite  # type: ignore[no-redef]
    import photoz_collect_real_data  # type: ignore[no-redef]

DEFAULT_RAW_DIR = Path("data/transferz/raw")
DEFAULT_NORMALIZED_DIR = Path("data/transferz/normalized")
DEFAULT_SUITE_OUTPUT_DIR = Path("reports/example_summaries/transferz")
DEFAULT_SUITE_REPORT = DEFAULT_SUITE_OUTPUT_DIR / "photoz_transferz_suite_latest.json"
DEFAULT_MARKDOWN_REPORT = DEFAULT_SUITE_OUTPUT_DIR / "photoz_transferz_suite_latest.md"
DEFAULT_REPORT = DEFAULT_SUITE_OUTPUT_DIR / "photoz_transferz_pipeline_latest.json"
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _expected_normalized_paths(normalized_dir: Path) -> dict[str, Path]:
    return {
        "train": normalized_dir / "transferz_train_photoz.csv",
        "cal": normalized_dir / "transferz_cal_photoz.csv",
        "test": normalized_dir / "transferz_test_photoz.csv",
        "conformal": normalized_dir / "transferz_conformal_photoz.csv",
    }


def _existing_normalized_paths(normalized_dir: Path) -> dict[str, Path] | None:
    paths = _expected_normalized_paths(normalized_dir)
    if all(path.exists() for path in paths.values()):
        return paths
    return None


def _import_example_module(module_name: str) -> Any:
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.pop(0)


def _run_conformal_example(
    *,
    profile: str,
    output_dir: Path,
    train_path: Path,
    cal_path: Path,
    conformal_path: Path,
    test_path: Path,
) -> Path:
    module = _import_example_module("photoz_transferz_conformal_comparison")
    if profile == "smoke":
        cfg = module.PhotoZTransferZConformalConfig(
            n_train=64,
            n_cal=24,
            n_conformal=24,
            n_test=24,
            batch_size=16,
            epochs=1,
            hidden=16,
            n_mc_samples=8,
            n_bins=24,
            train_dataset_path=str(train_path),
            cal_dataset_path=str(cal_path),
            conformal_dataset_path=str(conformal_path),
            test_dataset_path=str(test_path),
            require_real_data=True,
        )
    elif profile == "audit":
        cfg = module.PhotoZTransferZConformalConfig(
            n_train=192,
            n_cal=64,
            n_conformal=64,
            n_test=64,
            batch_size=32,
            epochs=6,
            hidden=32,
            n_mc_samples=16,
            n_bins=32,
            train_dataset_path=str(train_path),
            cal_dataset_path=str(cal_path),
            conformal_dataset_path=str(conformal_path),
            test_dataset_path=str(test_path),
            require_real_data=True,
        )
    else:
        cfg = module.PhotoZTransferZConformalConfig(
            train_dataset_path=str(train_path),
            cal_dataset_path=str(cal_path),
            conformal_dataset_path=str(conformal_path),
            test_dataset_path=str(test_path),
            require_real_data=True,
        )
    output_path = output_dir / f"photoz_transferz_conformal_comparison_{profile}.json"
    module.main(cfg, summary_json_path=str(output_path))
    return output_path


def _run_semisupervised_example(
    *,
    profile: str,
    output_dir: Path,
    train_path: Path,
    cal_path: Path,
    test_path: Path,
) -> Path:
    module = _import_example_module("photoz_transferz_semisupervised_comparison")
    if profile == "smoke":
        cfg = module.PhotoZTransferZSemiSupervisedConfig(
            n_train=96,
            n_cal=32,
            n_test=48,
            batch_size=16,
            epochs=1,
            teacher_epochs=2,
            hidden=16,
            labeled_fractions=(0.15, 0.4),
            train_dataset_path=str(train_path),
            cal_dataset_path=str(cal_path),
            test_dataset_path=str(test_path),
            require_real_data=True,
        )
    elif profile == "audit":
        cfg = module.PhotoZTransferZSemiSupervisedConfig(
            n_train=256,
            n_cal=96,
            n_test=128,
            batch_size=32,
            epochs=5,
            teacher_epochs=6,
            hidden=32,
            labeled_fractions=(0.1, 0.25, 0.5),
            train_dataset_path=str(train_path),
            cal_dataset_path=str(cal_path),
            test_dataset_path=str(test_path),
            require_real_data=True,
        )
    else:
        cfg = module.PhotoZTransferZSemiSupervisedConfig(
            train_dataset_path=str(train_path),
            cal_dataset_path=str(cal_path),
            test_dataset_path=str(test_path),
            require_real_data=True,
        )
    output_path = output_dir / f"photoz_transferz_semisupervised_comparison_{profile}.json"
    module.main(cfg, summary_json_path=str(output_path))
    return output_path


def run_pipeline(
    *,
    profile: str,
    raw_output_dir: Path = DEFAULT_RAW_DIR,
    normalized_output_dir: Path = DEFAULT_NORMALIZED_DIR,
    suite_output_dir: Path = DEFAULT_SUITE_OUTPUT_DIR,
    suite_report_path: Path = DEFAULT_SUITE_REPORT,
    markdown_report_path: Path = DEFAULT_MARKDOWN_REPORT,
    report_path: Path = DEFAULT_REPORT,
    record_id: int = photoz_collect_real_data.DEFAULT_TRANSFERZ_ZENODO_RECORD,
    default_target_err: float = 0.01,
    download_if_missing: bool = False,
    force_download: bool = False,
) -> dict[str, Any]:
    collection_report: dict[str, Any] | None = None
    normalized_paths = None if force_download else _existing_normalized_paths(normalized_output_dir)
    if normalized_paths is None:
        if not download_if_missing and not force_download:
            raise FileNotFoundError(
                "TransferZ normalized splits not found. Re-run with --download-if-missing "
                "or provide pre-populated data/transferz/normalized files."
            )
        collection_report = photoz_collect_real_data.collect_transferz_splits(
            record_id=record_id,
            raw_output_dir=raw_output_dir,
            normalized_output_dir=normalized_output_dir,
            default_target_err=default_target_err,
        )
        normalized_paths = {
            name: Path(path_str) for name, path_str in collection_report["normalized_paths"].items()
        }

    suite_report = photoz_benchmark_suite.run_suite(
        profile=profile,
        output_dir=suite_output_dir,
        real_data_only=True,
        train_dataset_path=normalized_paths["train"],
        cal_dataset_path=normalized_paths["cal"],
        test_dataset_path=normalized_paths["test"],
        markdown_report_path=markdown_report_path,
    )
    conformal_summary_path = _run_conformal_example(
        profile=profile,
        output_dir=suite_output_dir,
        train_path=normalized_paths["train"],
        cal_path=normalized_paths["cal"],
        conformal_path=normalized_paths["conformal"],
        test_path=normalized_paths["test"],
    )
    semisup_summary_path = _run_semisupervised_example(
        profile=profile,
        output_dir=suite_output_dir,
        train_path=normalized_paths["train"],
        cal_path=normalized_paths["cal"],
        test_path=normalized_paths["test"],
    )
    suite_report["summary_paths"]["photoz_transferz_conformal_comparison"] = str(
        conformal_summary_path
    )
    suite_report["summary_paths"]["photoz_transferz_semisupervised_comparison"] = str(
        semisup_summary_path
    )
    suite_report["recommended_read_order"] = [
        *suite_report.get("recommended_read_order", []),
        "photoz_transferz_semisupervised_comparison",
        "photoz_transferz_conformal_comparison",
    ]
    suite_report_path.parent.mkdir(parents=True, exist_ok=True)
    suite_report_path.write_text(json.dumps(suite_report, indent=2), encoding="utf-8")

    report = {
        "artifact": "photoz_transferz_pipeline_report",
        "version": 1,
        "profile": profile,
        "record_id": int(record_id),
        "default_target_err": float(default_target_err),
        "split_policy": {
            "train": str(normalized_paths["train"]),
            "cal": str(normalized_paths["cal"]),
            "test": str(normalized_paths["test"]),
            "conformal_reserved": str(normalized_paths["conformal"]),
        },
        "collection_report": collection_report,
        "suite_output_dir": str(suite_output_dir),
        "suite_report_path": str(suite_report_path),
        "markdown_report_path": str(markdown_report_path),
        "suite_summary_paths": suite_report["summary_paths"],
        "semisupervised_summary_path": str(semisup_summary_path),
        "conformal_summary_path": str(conformal_summary_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TransferZ tabular photo-z benchmark.")
    parser.add_argument("--profile", choices=["smoke", "audit", "full"], default="full")
    parser.add_argument("--raw-output-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--normalized-output-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    parser.add_argument(
        "--suite-output-dir",
        type=Path,
        default=DEFAULT_SUITE_OUTPUT_DIR,
    )
    parser.add_argument("--suite-report", type=Path, default=DEFAULT_SUITE_REPORT)
    parser.add_argument("--markdown-report", type=Path, default=DEFAULT_MARKDOWN_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--record-id",
        type=int,
        default=photoz_collect_real_data.DEFAULT_TRANSFERZ_ZENODO_RECORD,
    )
    parser.add_argument("--default-target-err", type=float, default=0.01)
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    report = run_pipeline(
        profile=args.profile,
        raw_output_dir=args.raw_output_dir,
        normalized_output_dir=args.normalized_output_dir,
        suite_output_dir=args.suite_output_dir,
        suite_report_path=args.suite_report,
        markdown_report_path=args.markdown_report,
        report_path=args.report,
        record_id=args.record_id,
        default_target_err=args.default_target_err,
        download_if_missing=args.download_if_missing,
        force_download=args.force_download,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
