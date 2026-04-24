"""Render SPT-Reg benchmark summaries for the NeurIPS paper workspace."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "neurips_spt_reg"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "artifact_manifest_latest.json"

PROFILE_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "smoke": {
        "synthetic": {
            "n_source": 96,
            "n_target_unlabeled": 48,
            "n_target_cal": 24,
            "n_target_test": 24,
            "n_support": 64,
            "n_bins": 12,
            "n_samples_eval": 16,
            "target_label_budget": 16,
        },
        "realdata": {
            "n_source": 160,
            "n_target_unlabeled": 32,
            "n_target_cal": 24,
            "n_target_test": 24,
            "n_support": 64,
            "n_bins": 12,
            "n_samples_eval": 16,
            "target_label_budget": 16,
        },
        "year": {
            "n_source": 160,
            "n_target_unlabeled": 48,
            "n_target_cal": 24,
            "n_target_test": 24,
            "n_support": 64,
            "n_bins": 12,
            "n_samples_eval": 16,
            "target_label_budget": 16,
        },
    },
    "audit": {
        "synthetic": {
            "n_source": 224,
            "n_target_unlabeled": 96,
            "n_target_cal": 48,
            "n_target_test": 64,
            "n_support": 96,
            "n_bins": 16,
            "n_samples_eval": 32,
            "target_label_budget": 24,
        },
        "realdata": {
            "n_source": 220,
            "n_target_unlabeled": 64,
            "n_target_cal": 40,
            "n_target_test": 40,
            "n_support": 96,
            "n_bins": 16,
            "n_samples_eval": 32,
            "target_label_budget": 24,
        },
        "year": {
            "n_source": 224,
            "n_target_unlabeled": 96,
            "n_target_cal": 48,
            "n_target_test": 48,
            "n_support": 96,
            "n_bins": 16,
            "n_samples_eval": 32,
            "target_label_budget": 24,
        },
    },
    "full": {
        "synthetic": {
            "n_source": 320,
            "n_target_unlabeled": 192,
            "n_target_cal": 64,
            "n_target_test": 128,
            "n_support": 128,
            "n_bins": 20,
            "n_samples_eval": 64,
            "target_label_budget": 64,
        },
        "realdata": {
            "n_source": 220,
            "n_target_unlabeled": 72,
            "n_target_cal": 48,
            "n_target_test": 48,
            "n_support": 128,
            "n_bins": 20,
            "n_samples_eval": 64,
            "target_label_budget": 32,
        },
        "year": {
            "n_source": 4096,
            "n_target_unlabeled": 8192,
            "n_target_cal": 2048,
            "n_target_test": 4096,
            "n_support": 128,
            "n_bins": 20,
            "n_samples_eval": 64,
            "target_label_budget": 512,
        },
    },
}


def _load_example_module(name: str) -> Any:
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.pop(0)


def _render_example(
    *,
    module_name: str,
    config_name: str,
    config_kwargs: dict[str, Any],
    summary_path: Path,
) -> Path:
    module = _load_example_module(module_name)
    config_cls = getattr(module, config_name)
    cfg = config_cls(**config_kwargs)
    module.main(cfg, summary_json_path=str(summary_path))
    return summary_path


def _write_local_year_dataset(
    path: Path, *, n_rows: int, n_features: int = 8, seed: int = 260410
) -> Path:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    weights = rng.normal(size=n_features).astype(np.float32)
    y = (x @ weights + 0.1 * rng.normal(size=n_rows)).astype(np.float32)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(n_features)])
    frame["target"] = y
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def run_render(
    *,
    profile: str,
    include_photoz: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    year_cache_path: Path | None = None,
    year_dataset_path: Path | None = None,
    year_allow_download: bool = False,
    year_openml_data_id: int | None = None,
    year_openml_dataset_name: str | None = None,
    year_max_dataset_rows: int | None = None,
) -> dict[str, Any]:
    if profile not in PROFILE_CONFIGS:
        raise ValueError(f"Unknown profile {profile!r}. Expected one of {sorted(PROFILE_CONFIGS)}.")
    if year_cache_path is not None and year_dataset_path is not None:
        raise ValueError("Pass at most one of year_cache_path and year_dataset_path.")
    if year_dataset_path is not None and (
        year_openml_data_id is not None or year_openml_dataset_name is not None
    ):
        raise ValueError("year_dataset_path cannot be combined with OpenML fetch parameters.")

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_cfg = PROFILE_CONFIGS[profile]

    synthetic_path = output_dir / f"synthetic_competing_methods_{profile}.json"
    realdata_path = output_dir / f"tabular_competing_methods_{profile}.json"
    year_path = output_dir / f"year_competing_methods_{profile}.json"

    _render_example(
        module_name="spt_reg_synthetic_comparison",
        config_name="SPTRegSyntheticConfig",
        config_kwargs=profile_cfg["synthetic"],
        summary_path=synthetic_path,
    )
    _render_example(
        module_name="spt_reg_realdata_comparison",
        config_name="SPTRegRealDataConfig",
        config_kwargs=profile_cfg["realdata"],
        summary_path=realdata_path,
    )
    year_cfg = dict(profile_cfg["year"])
    year_data_note: str
    if year_dataset_path is not None:
        year_cfg["dataset_path"] = str(year_dataset_path.resolve())
        year_cfg["cache_path"] = None
        year_cfg["allow_download"] = False
        year_data_note = f"dataset_path={year_cfg['dataset_path']}"
    elif (
        year_cache_path is not None
        or year_openml_data_id is not None
        or year_openml_dataset_name is not None
    ):
        year_cfg["dataset_path"] = None
        year_cfg["cache_path"] = (
            str(year_cache_path.resolve()) if year_cache_path is not None else None
        )
        year_cfg["allow_download"] = bool(year_allow_download)
        year_cfg["openml_data_id"] = year_openml_data_id
        year_cfg["openml_dataset_name"] = year_openml_dataset_name
        if year_max_dataset_rows is not None:
            year_cfg["max_dataset_rows"] = year_max_dataset_rows
        parts = []
        if year_cfg["cache_path"]:
            parts.append(f"cache_path={year_cfg['cache_path']}")
        if year_openml_data_id is not None:
            parts.append(f"openml_data_id={year_openml_data_id}")
        if year_openml_dataset_name is not None:
            parts.append(f"openml_dataset_name={year_openml_dataset_name}")
        if year_max_dataset_rows is not None:
            parts.append(f"max_dataset_rows={year_max_dataset_rows}")
        parts.append(f"allow_download={year_allow_download}")
        year_data_note = ", ".join(parts)
    else:
        year_rows = (
            int(year_cfg["n_source"])
            + int(year_cfg["n_target_unlabeled"])
            + int(year_cfg["n_target_cal"])
            + int(year_cfg["n_target_test"])
        )
        local_year_dataset = _write_local_year_dataset(
            output_dir / f"year_local_dataset_{profile}.csv",
            n_rows=year_rows + 64,
        )
        year_cfg["dataset_path"] = str(local_year_dataset)
        year_cfg["cache_path"] = None
        year_cfg["allow_download"] = False
        year_data_note = f"synthetic_local={year_cfg['dataset_path']}"
    _render_example(
        module_name="spt_reg_year_comparison",
        config_name="SPTRegYearConfig",
        config_kwargs=year_cfg,
        summary_path=year_path,
    )
    summaries: dict[str, str] = {
        "synthetic": str(synthetic_path),
        "tabular_small": str(realdata_path),
        "tabular_large": str(year_path),
    }

    report = {
        "artifact": "spt_reg_paper_artifact_manifest",
        "version": 1,
        "profile": profile,
        "include_photoz": False,
        "output_dir": str(output_dir),
        "summaries": summaries,
        "year_track_data": year_data_note,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render SPT-Reg summary artifacts for the NeurIPS paper workspace."
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="smoke")

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--year-cache-path",
        type=Path,
        default=None,
        help=(
            "OpenML Year cache CSV/Parquet for the large-tabular track (real data). "
            "Skips synthetic year_local_dataset_<profile>.csv. "
            "Use with --year-allow-download if the file may need to be created."
        ),
    )
    parser.add_argument(
        "--year-dataset-path",
        type=Path,
        default=None,
        help="Explicit Year table path (must not be combined with --year-cache-path).",
    )
    parser.add_argument(
        "--year-allow-download",
        action="store_true",
        help="Allow fetch_openml when the cache file is missing or when fetching by OpenML id/name.",
    )
    parser.add_argument(
        "--year-openml-data-id",
        type=int,
        default=None,
        help="OpenML data_id for the large-tabular track (default runner: 42225 diamonds).",
    )
    parser.add_argument(
        "--year-openml-dataset-name",
        type=str,
        default="",
        help="OpenML name=... alternative to --year-openml-data-id (with sklearn version).",
    )
    parser.add_argument(
        "--year-max-dataset-rows",
        type=int,
        default=None,
        help="Cap rows before the covariate-shift split (RAM-friendly for huge OpenML dumps).",
    )
    args = parser.parse_args()

    report = run_render(
        profile=args.profile,
        include_photoz=False,
        output_dir=args.output_dir,
        report_path=args.report,
        year_cache_path=args.year_cache_path,
        year_dataset_path=args.year_dataset_path,
        year_allow_download=args.year_allow_download,
        year_openml_data_id=args.year_openml_data_id,
        year_openml_dataset_name=args.year_openml_dataset_name or None,
        year_max_dataset_rows=args.year_max_dataset_rows,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
