from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "example_summaries"


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
            n_cal=16,
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
            n_cal=48,
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
            n_cal=32,
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
            n_cal=64,
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


def _contrastive_flow_synth_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.ContrastiveFlowComparisonConfig(
            n_train=64,
            n_test=24,
            events_per_experiment=48,
            batch_size=16,
            epochs=2,
            hidden=16,
            flow_context_dim=8,
            flow_transforms=2,
            n_negatives=2,
            mu_grid_size=11,
            nuisance_grid_size=9,
        )
    if profile == "audit":
        return module.ContrastiveFlowComparisonConfig(
            n_train=192,
            n_test=64,
            events_per_experiment=96,
            batch_size=32,
            epochs=12,
            hidden=48,
            flow_context_dim=16,
            flow_transforms=4,
            n_negatives=4,
            mu_grid_size=21,
            nuisance_grid_size=17,
        )
    return module.ContrastiveFlowComparisonConfig()


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


def _ordinal_realdata_comparison_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.OrdinalRealDataConfig(
            n_train=128,
            n_test=64,
            hidden=16,
            epochs=4,
            batch_size=32,
        )
    if profile == "audit":
        return module.OrdinalRealDataConfig(
            n_train=280,
            n_test=120,
            hidden=32,
            epochs=20,
            batch_size=64,
        )
    return module.OrdinalRealDataConfig()


def _ordinal_ugt_comparison_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.OrdinalUGTComparisonConfig(
            n_train=128,
            n_test=64,
            hidden=16,
            epochs=4,
            teacher_epochs=4,
            batch_size=32,
        )
    if profile == "audit":
        return module.OrdinalUGTComparisonConfig(
            n_train=512,
            n_test=256,
            hidden=32,
            epochs=20,
            teacher_epochs=16,
            batch_size=64,
        )
    return module.OrdinalUGTComparisonConfig()


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


def _censored_realdata_comparison_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.CensoredRealDataConfig(
            n_train=128,
            n_test=64,
            hidden=16,
            epochs=4,
            batch_size=32,
        )
    if profile == "audit":
        return module.CensoredRealDataConfig(
            n_train=280,
            n_test=120,
            hidden=32,
            epochs=20,
            batch_size=64,
        )
    return module.CensoredRealDataConfig()


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


def _transformed_target_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.TransformComparisonConfig(
            n_train=128,
            n_test=64,
            hidden=16,
            epochs=4,
        )
    if profile == "audit":
        return module.TransformComparisonConfig(
            n_train=384,
            n_test=192,
            hidden=24,
            epochs=18,
        )
    return module.TransformComparisonConfig()


def _semi_supervised_regression_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.SemiSupervisedRegressionConfig(
            n_labeled=48,
            n_unlabeled=96,
            n_test=64,
            hidden=16,
            teacher_epochs=6,
            student_epochs=8,
        )
    if profile == "audit":
        return module.SemiSupervisedRegressionConfig(
            n_labeled=80,
            n_unlabeled=180,
            n_test=90,
            hidden=24,
            teacher_epochs=20,
            student_epochs=24,
        )
    return module.SemiSupervisedRegressionConfig()


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


def _uncertain_gt_density_conformal_realdata_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.UncertainGTConformalRealDataConfig(
            n_train=128,
            n_cal=64,
            n_test=64,
            hidden=16,
            epochs=4,
            n_mc_samples=12,
        )
    if profile == "audit":
        return module.UncertainGTConformalRealDataConfig(
            n_train=220,
            n_cal=110,
            n_test=90,
            hidden=24,
            epochs=16,
            n_mc_samples=18,
        )
    return module.UncertainGTConformalRealDataConfig()


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


def _causal_dr_realdata_config(module: Any, profile: str) -> Any:
    if profile == "smoke":
        return module.CausalDRRealDataConfig(
            n_samples=280,
            folds=2,
        )
    if profile == "audit":
        return module.CausalDRRealDataConfig(
            n_samples=360,
            folds=3,
        )
    return module.CausalDRRealDataConfig()


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
    "contrastive_flow_parameter_estimation_comparison": {
        "filename": "contrastive_flow_parameter_estimation_comparison",
        "config_factory": _contrastive_flow_synth_config,
    },
    "noisy_label_comparison": {
        "filename": "noisy_label_comparison",
        "config_factory": _noisy_label_config,
    },
    "noisy_label_realdata_comparison": {
        "filename": "noisy_label_realdata_comparison",
        "config_factory": _noisy_label_realdata_config,
    },
    "ordinal_regression_comparison": {
        "filename": "ordinal_regression_comparison",
        "config_factory": _ordinal_comparison_config,
    },
    "ordinal_regression_realdata_comparison": {
        "filename": "ordinal_regression_realdata_comparison",
        "config_factory": _ordinal_realdata_comparison_config,
    },
    "ordinal_uncertain_ground_truth_comparison": {
        "filename": "ordinal_uncertain_ground_truth_comparison",
        "config_factory": _ordinal_ugt_comparison_config,
    },
    "censored_regression_comparison": {
        "filename": "censored_regression_comparison",
        "config_factory": _censored_comparison_config,
    },
    "censored_regression_realdata_comparison": {
        "filename": "censored_regression_realdata_comparison",
        "config_factory": _censored_realdata_comparison_config,
    },
    "propensity_tail_regression_comparison": {
        "filename": "propensity_tail_regression_comparison",
        "config_factory": _propensity_tail_config,
    },
    "constraints_calibration_comparison": {
        "filename": "constraints_calibration_comparison",
        "config_factory": _constraints_calibration_config,
    },
    "transformed_target_regression_comparison": {
        "filename": "transformed_target_regression_comparison",
        "config_factory": _transformed_target_config,
    },
    "semi_supervised_regression_comparison": {
        "filename": "semi_supervised_regression_comparison",
        "config_factory": _semi_supervised_regression_config,
    },
    "uncertain_gt_density_conformal_comparison": {
        "filename": "uncertain_gt_density_conformal_comparison",
        "config_factory": _uncertain_gt_density_conformal_config,
    },
    "uncertain_gt_density_conformal_realdata_comparison": {
        "filename": "uncertain_gt_density_conformal_realdata_comparison",
        "config_factory": _uncertain_gt_density_conformal_realdata_config,
    },
    "causal_dr_uplift_comparison": {
        "filename": "causal_dr_uplift_comparison",
        "config_factory": _causal_dr_uplift_config,
    },
    "causal_dr_realdata_comparison": {
        "filename": "causal_dr_realdata_comparison",
        "config_factory": _causal_dr_realdata_config,
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
) -> list[Path]:
    names = examples or list(EXAMPLE_SPECS.keys())
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in names:
        path = render_example_summary(name, profile=profile, output_dir=output_dir)
        print(f"Wrote {name} summary -> {path}")
        paths.append(path)

    return paths


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
    args = parser.parse_args()
    render_all(
        profile=args.profile,
        output_dir=args.output_dir,
        examples=args.examples,
    )


if __name__ == "__main__":
    main()
