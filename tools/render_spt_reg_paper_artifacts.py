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
        "photoz": {
            "n_train": 96,
            "n_target_unlabeled": 24,
            "n_target_cal": 24,
            "n_target_test": 24,
            "epochs": 2,
            "hidden": 16,
            "n_support": 64,
            "n_bins": 12,
            "n_samples_eval": 16,
            "target_label_budget": 16,
            "sample_size_if_generate": 256,
            "force_simulated": True,
            "allow_download": False,
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
        "photoz": {
            "n_train": 160,
            "n_target_unlabeled": 48,
            "n_target_cal": 40,
            "n_target_test": 40,
            "epochs": 3,
            "hidden": 24,
            "n_support": 96,
            "n_bins": 16,
            "n_samples_eval": 24,
            "target_label_budget": 24,
            "sample_size_if_generate": 512,
            "force_simulated": True,
            "allow_download": False,
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
            "n_source": 320,
            "n_target_unlabeled": 192,
            "n_target_cal": 64,
            "n_target_test": 96,
            "n_support": 128,
            "n_bins": 20,
            "n_samples_eval": 64,
            "target_label_budget": 32,
        },
        "photoz": {
            "n_train": 256,
            "n_target_unlabeled": 96,
            "n_target_cal": 64,
            "n_target_test": 96,
            "epochs": 8,
            "hidden": 64,
            "n_support": 128,
            "n_bins": 24,
            "n_samples_eval": 64,
            "target_label_budget": 32,
            "sample_size_if_generate": 5000,
            "force_simulated": True,
            "allow_download": False,
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
) -> dict[str, Any]:
    if profile not in PROFILE_CONFIGS:
        raise ValueError(f"Unknown profile {profile!r}. Expected one of {sorted(PROFILE_CONFIGS)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    profile_cfg = PROFILE_CONFIGS[profile]

    synthetic_path = output_dir / f"synthetic_competing_methods_{profile}.json"
    realdata_path = output_dir / f"tabular_competing_methods_{profile}.json"
    year_path = output_dir / f"year_competing_methods_{profile}.json"
    photoz_path = output_dir / f"photoz_competing_methods_{profile}.json"

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
    if include_photoz:
        _render_example(
            module_name="spt_reg_photoz_comparison",
            config_name="SPTRegPhotoZConfig",
            config_kwargs=profile_cfg["photoz"],
            summary_path=photoz_path,
        )
        summaries["photoz"] = str(photoz_path)

    report = {
        "artifact": "spt_reg_paper_artifact_manifest",
        "version": 1,
        "profile": profile,
        "include_photoz": include_photoz,
        "output_dir": str(output_dir),
        "summaries": summaries,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render SPT-Reg summary artifacts for the NeurIPS paper workspace."
    )
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default="smoke")
    parser.add_argument(
        "--include-photoz",
        action="store_true",
        help="Render the optional photo-z compatibility benchmark from torchregress.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    report = run_render(
        profile=args.profile,
        include_photoz=args.include_photoz,
        output_dir=args.output_dir,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
