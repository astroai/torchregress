"""Generate machine-readable summary artifacts for comparison examples."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools import photoz_rail_compare
except ModuleNotFoundError:  # pragma: no cover - script execution path
    import photoz_rail_compare  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "example_summaries"
DEFAULT_PHOTOZ_RAIL_MANIFEST = REPO_ROOT / "data" / "rail" / "rail_photoz_manifest.json"


def _import_example_module(module_name: str) -> Any:
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.pop(0)


def _ood_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.OODConfig(
            n_train=48,
            n_id_test=32,
            n_ood_test=32,
            epochs=2,
            ensemble_size=2,
            mc_samples=6,
            swag_samples=4,
            bnn_samples=6,
        )
    if profile == "audit":
        return module.OODConfig(
            n_train=160,
            n_id_test=80,
            n_ood_test=80,
            epochs=12,
            ensemble_size=3,
            mc_samples=12,
            swag_samples=10,
            bnn_samples=12,
        )
    return module.OODConfig()


def _ood_realdata_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.OODRealDataConfig(
            n_train=96,
            n_id_test=32,
            n_ood_test=32,
            epochs=2,
            ensemble_size=2,
            mc_samples=6,
            swag_samples=4,
            bnn_samples=6,
        )
    if profile == "audit":
        return module.OODRealDataConfig(
            n_train=192,
            n_id_test=64,
            n_ood_test=64,
            epochs=10,
            ensemble_size=3,
            mc_samples=10,
            swag_samples=8,
            bnn_samples=10,
        )
    return module.OODRealDataConfig()


def _eiv_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.EIVConfig(n_train=48, n_test=48, epochs=2, hidden=8)
    if profile == "audit":
        return module.EIVConfig(n_train=96, n_test=96, epochs=12, hidden=16)
    return module.EIVConfig()


def _eiv_realdata_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.EIVRealDataConfig(n_train=96, n_test=48, epochs=2, hidden=16)
    if profile == "audit":
        return module.EIVRealDataConfig(n_train=192, n_test=96, epochs=10, hidden=24)
    return module.EIVRealDataConfig()


def _multimodal_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.MultimodalComparisonConfig(
            n_train=64,
            n_test=48,
            batch_size=16,
            epochs=2,
            hidden=16,
            mdn_components=2,
            eval_samples=16,
            flow_context_dim=8,
            flow_transforms=2,
        )
    if profile == "audit":
        return module.MultimodalComparisonConfig(
            n_train=256,
            n_test=160,
            batch_size=64,
            epochs=12,
            hidden=64,
            mdn_components=4,
            eval_samples=64,
            flow_context_dim=16,
            flow_transforms=4,
        )
    return module.MultimodalComparisonConfig()


def _multimodal_realdata_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.MultimodalRealDataConfig(
            n_train=96,
            n_test=48,
            batch_size=16,
            epochs=2,
            hidden=16,
            mdn_components=2,
            eval_samples=16,
            flow_context_dim=8,
            flow_transforms=2,
        )
    if profile == "audit":
        return module.MultimodalRealDataConfig(
            n_train=192,
            n_test=96,
            batch_size=32,
            epochs=12,
            hidden=48,
            mdn_components=4,
            eval_samples=48,
            flow_context_dim=16,
            flow_transforms=4,
        )
    return module.MultimodalRealDataConfig()


def _noisy_label_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.NoisyLabelComparisonConfig(
            n_train=64,
            n_cal=24,
            n_test=48,
            epochs=2,
            batch_size=16,
            hidden=16,
        )
    if profile == "audit":
        return module.NoisyLabelComparisonConfig(
            n_train=256,
            n_cal=96,
            n_test=160,
            epochs=12,
            batch_size=32,
            hidden=64,
        )
    return module.NoisyLabelComparisonConfig()


def _noisy_label_realdata_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.NoisyLabelRealDataConfig(
            n_train=96,
            n_cal=32,
            n_test=32,
            epochs=2,
            batch_size=16,
            hidden=16,
        )
    if profile == "audit":
        return module.NoisyLabelRealDataConfig(
            n_train=192,
            n_cal=80,
            n_test=80,
            epochs=12,
            batch_size=32,
            hidden=32,
        )
    return module.NoisyLabelRealDataConfig()


def _photoz_benchmark_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.PhotoZBenchmarkConfig(
            n_train=64,
            n_cal=24,
            n_test=24,
            batch_size=16,
            epochs=1,
            hidden=16,
            sample_size_if_generate=256,
            force_simulated=True,
            allow_download=False,
        )
    if profile == "audit":
        return module.PhotoZBenchmarkConfig(
            n_train=192,
            n_cal=64,
            n_test=64,
            batch_size=32,
            epochs=8,
            hidden=32,
            sample_size_if_generate=800,
            force_simulated=True,
            allow_download=False,
        )
    return module.PhotoZBenchmarkConfig(
        force_simulated=True,
        allow_download=False,
        sample_size_if_generate=1600,
    )


def _photoz_nnc_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.PhotoZNNCConfig(
            n_train=64,
            n_cal=24,
            n_test=24,
            batch_size=16,
            epochs=1,
            hidden=16,
            n_bins=24,
            sample_size_if_generate=256,
            force_simulated=True,
            allow_download=False,
            temperature_max_iter=50,
        )
    if profile == "audit":
        return module.PhotoZNNCConfig(
            n_train=192,
            n_cal=64,
            n_test=64,
            batch_size=32,
            epochs=8,
            hidden=32,
            n_bins=32,
            sample_size_if_generate=800,
            force_simulated=True,
            allow_download=False,
            temperature_max_iter=120,
        )
    return module.PhotoZNNCConfig(
        force_simulated=True,
        allow_download=False,
        sample_size_if_generate=1600,
        n_bins=48,
    )


def _ppi_photoz_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.PPIPhotoZConfig(
            n_labeled=64,
            n_unlabeled=320,
            n_boot=120,
        )
    if profile == "audit":
        return module.PPIPhotoZConfig(
            n_labeled=160,
            n_unlabeled=1200,
            n_boot=320,
        )
    return module.PPIPhotoZConfig(
        n_labeled=256,
        n_unlabeled=3000,
        n_boot=600,
    )


def _ordinal_comparison_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.OrdinalComparisonConfig(
            n_train=128,
            n_test=64,
            hidden=16,
            epochs=4,
            batch_size=32,
        )
    if profile == "audit":
        return module.OrdinalComparisonConfig(
            n_train=512,
            n_test=256,
            hidden=32,
            epochs=20,
            batch_size=64,
        )
    return module.OrdinalComparisonConfig()


def _censored_comparison_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.CensoredComparisonConfig(
            n_train=192,
            n_test=96,
            hidden=16,
            epochs=4,
            batch_size=32,
        )
    if profile == "audit":
        return module.CensoredComparisonConfig(
            n_train=768,
            n_test=256,
            hidden=32,
            epochs=20,
            batch_size=64,
        )
    return module.CensoredComparisonConfig()


def _propensity_tail_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.PropensityTailConfig(
            n_train_pool=384,
            n_test=128,
            hidden=16,
            epochs=4,
            batch_size=32,
        )
    if profile == "audit":
        return module.PropensityTailConfig(
            n_train_pool=1200,
            n_test=400,
            hidden=32,
            epochs=20,
            batch_size=64,
        )
    return module.PropensityTailConfig()


def _constraints_calibration_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.ConstraintCalibrationConfig(
            n_cal=256,
            n_test=128,
        )
    if profile == "audit":
        return module.ConstraintCalibrationConfig(
            n_cal=512,
            n_test=256,
        )
    return module.ConstraintCalibrationConfig()


def _uncertain_gt_density_conformal_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.UncertainGTConformalConfig(
            n_cal=192,
            n_test=128,
            n_mc_samples=12,
        )
    if profile == "audit":
        return module.UncertainGTConformalConfig(
            n_cal=400,
            n_test=300,
            n_mc_samples=24,
        )
    return module.UncertainGTConformalConfig()


def _causal_dr_uplift_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.CausalDRConfig(
            n_samples=600,
            folds=2,
        )
    if profile == "audit":
        return module.CausalDRConfig(
            n_samples=1200,
            folds=3,
        )
    return module.CausalDRConfig()


EXAMPLE_SPECS: dict[str, dict[str, Any]] = {
    "ood_selective_prediction_comparison": {
        "filename": "ood_selective_prediction_comparison",
        "config_factory": _ood_config,
    },
    "ood_selective_prediction_realdata_comparison": {
        "filename": "ood_selective_prediction_realdata_comparison",
        "config_factory": _ood_realdata_config,
    },
    "eiv_method_comparison": {
        "filename": "eiv_method_comparison",
        "config_factory": _eiv_config,
    },
    "eiv_method_realdata_comparison": {
        "filename": "eiv_method_realdata_comparison",
        "config_factory": _eiv_realdata_config,
    },
    "multimodal_method_comparison": {
        "filename": "multimodal_method_comparison",
        "config_factory": _multimodal_config,
    },
    "multimodal_method_realdata_comparison": {
        "filename": "multimodal_method_realdata_comparison",
        "config_factory": _multimodal_realdata_config,
    },
    "noisy_label_comparison": {
        "filename": "noisy_label_comparison",
        "config_factory": _noisy_label_config,
    },
    "noisy_label_realdata_comparison": {
        "filename": "noisy_label_realdata_comparison",
        "config_factory": _noisy_label_realdata_config,
    },
    "photoz_benchmark_comparison": {
        "filename": "photoz_benchmark_comparison",
        "config_factory": _photoz_benchmark_config,
    },
    "photoz_nnc_crps_rail_comparison": {
        "filename": "photoz_nnc_crps_rail_comparison",
        "config_factory": _photoz_nnc_config,
    },
    "ppi_photoz_inference_comparison": {
        "filename": "ppi_photoz_inference_comparison",
        "config_factory": _ppi_photoz_config,
    },
    "ordinal_regression_comparison": {
        "filename": "ordinal_regression_comparison",
        "config_factory": _ordinal_comparison_config,
    },
    "censored_regression_comparison": {
        "filename": "censored_regression_comparison",
        "config_factory": _censored_comparison_config,
    },
    "propensity_tail_regression_comparison": {
        "filename": "propensity_tail_regression_comparison",
        "config_factory": _propensity_tail_config,
    },
    "constraints_calibration_comparison": {
        "filename": "constraints_calibration_comparison",
        "config_factory": _constraints_calibration_config,
    },
    "uncertain_gt_density_conformal_comparison": {
        "filename": "uncertain_gt_density_conformal_comparison",
        "config_factory": _uncertain_gt_density_conformal_config,
    },
    "causal_dr_uplift_comparison": {
        "filename": "causal_dr_uplift_comparison",
        "config_factory": _causal_dr_uplift_config,
    },
}


def render_example_summary(
    example_name: str,
    *,
    profile: str,
    output_dir: Path,
) -> Path:
    if example_name not in EXAMPLE_SPECS:
        raise KeyError(f"Unknown example {example_name!r}. Expected one of {sorted(EXAMPLE_SPECS)}")

    spec = EXAMPLE_SPECS[example_name]
    module = _import_example_module(example_name)
    config_factory = spec["config_factory"]
    cfg = config_factory(module, profile)
    output_path = output_dir / f"{spec['filename']}_{profile}.json"
    module.main(cfg, summary_json_path=str(output_path))
    return output_path


def render_all(
    *,
    profile: str,
    output_dir: Path,
    examples: list[str] | None = None,
    photoz_rail_inputs: list[Path] | None = None,
    photoz_rail_manifest: Path | None = None,
    photoz_rail_output: Path | None = None,
    photoz_rail_paper_parity: bool = True,
) -> list[Path]:
    names = examples or list(EXAMPLE_SPECS.keys())
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in names:
        path = render_example_summary(name, profile=profile, output_dir=output_dir)
        print(f"Wrote {name} summary -> {path}")
        paths.append(path)

    if photoz_rail_inputs:
        manifest_path = photoz_rail_manifest or DEFAULT_PHOTOZ_RAIL_MANIFEST
        tr_summary_path = output_dir / f"photoz_nnc_crps_rail_comparison_{profile}.json"
        if not tr_summary_path.exists():
            raise FileNotFoundError(
                "RAIL merge requested but torchregress photo-z summary is missing: "
                f"{tr_summary_path}. Include `photoz_nnc_crps_rail_comparison` in --examples "
                "or render it first."
            )
        out_path = photoz_rail_output or (
            output_dir / f"photoz_rail_baseline_comparison_{profile}.json"
        )
        merged_path = render_photoz_rail_merge(
            manifest_path=manifest_path,
            torchregress_summary_path=tr_summary_path,
            rail_input_paths=photoz_rail_inputs,
            output_path=out_path,
            paper_parity=photoz_rail_paper_parity,
        )
        print(f"Wrote photo-z RAIL merged summary -> {merged_path}")
        paths.append(merged_path)
    return paths


def render_photoz_rail_merge(
    *,
    manifest_path: Path,
    torchregress_summary_path: Path,
    rail_input_paths: list[Path],
    output_path: Path,
    paper_parity: bool,
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tr_summary = json.loads(torchregress_summary_path.read_text(encoding="utf-8"))
    rail_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in rail_input_paths]
    merged = photoz_rail_compare.merge_summaries(
        manifest=manifest,
        torchregress_summary=tr_summary,
        rail_payloads=rail_payloads,
        paper_parity=paper_parity,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render comparison-example summary JSON artifacts."
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "audit", "full"],
        default="audit",
        help="Budget profile for generated artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSON summary artifacts",
    )
    parser.add_argument(
        "--examples",
        nargs="*",
        choices=sorted(EXAMPLE_SPECS.keys()),
        help="Optional subset of examples to render",
    )
    parser.add_argument(
        "--photoz-rail-inputs",
        type=Path,
        nargs="+",
        help=(
            "Optional RAIL payload JSON files. When set, also emits merged "
            "photo-z comparison artifact."
        ),
    )
    parser.add_argument(
        "--photoz-rail-manifest",
        type=Path,
        default=DEFAULT_PHOTOZ_RAIL_MANIFEST,
        help="Manifest used for paper-parity checks during RAIL merge.",
    )
    parser.add_argument(
        "--photoz-rail-output",
        type=Path,
        default=None,
        help="Optional explicit output path for merged photo-z RAIL artifact.",
    )
    parser.add_argument(
        "--no-photoz-rail-parity",
        action="store_true",
        help="Disable strict manifest parity checks for the optional RAIL merge.",
    )
    args = parser.parse_args()
    render_all(
        profile=args.profile,
        output_dir=args.output_dir,
        examples=args.examples,
        photoz_rail_inputs=args.photoz_rail_inputs,
        photoz_rail_manifest=args.photoz_rail_manifest,
        photoz_rail_output=args.photoz_rail_output,
        photoz_rail_paper_parity=not args.no_photoz_rail_parity,
    )


if __name__ == "__main__":
    main()
