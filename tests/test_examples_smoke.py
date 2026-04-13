from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

matplotlib.use("Agg", force=True)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_example_module(stem: str) -> ModuleType:
    path = EXAMPLES_DIR / f"{stem}.py"
    module_name = f"_example_smoke_{stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load example module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _write_tiny_tabular_csv(
    path: Path,
    *,
    n_rows: int,
    n_features: int,
    seed: int,
    target_column: str = "target",
) -> None:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, n_features)).astype("float32")
    weights = rng.normal(size=n_features).astype("float32")
    y = (x @ weights + 0.1 * rng.normal(size=n_rows)).astype("float32")
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(n_features)])
    frame[target_column] = y
    frame.to_csv(path, index=False)


def test_hard_problem_examples_import_smoke() -> None:
    # Import-only smoke checks for examples touched in the audit-priority pass.
    _load_example_module("imbalanced_regression")
    _load_example_module("evaluate_conformal_methods")
    _load_example_module("ood_selective_prediction_comparison")
    _load_example_module("ood_selective_prediction_realdata_comparison")
    _load_example_module("eiv_method_comparison")
    _load_example_module("eiv_method_realdata_comparison")
    _load_example_module("noisy_label_comparison")
    _load_example_module("noisy_label_realdata_comparison")
    _load_example_module("multimodal_method_comparison")
    _load_example_module("multimodal_method_realdata_comparison")
    _load_example_module("contrastive_flow_parameter_estimation_comparison")
    _load_example_module("contrastive_flow_photoz_proxy_comparison")
    _load_example_module("photoz_benchmark_comparison")
    _load_example_module("photoz_nnc_crps_rail_comparison")
    _load_example_module("photoz_transferz_semisupervised_comparison")
    _load_example_module("photoz_transferz_conformal_comparison")
    _load_example_module("ppi_photoz_inference_comparison")
    _load_example_module("spt_reg_photoz_comparison")
    _load_example_module("ordinal_regression_comparison")
    _load_example_module("ordinal_regression_realdata_comparison")
    _load_example_module("ordinal_uncertain_ground_truth_comparison")
    _load_example_module("censored_regression_comparison")
    _load_example_module("censored_regression_realdata_comparison")
    _load_example_module("propensity_tail_regression_comparison")
    _load_example_module("constraints_calibration_comparison")
    _load_example_module("transformed_target_regression_comparison")
    _load_example_module("spt_reg_synthetic_comparison")
    _load_example_module("spt_reg_realdata_comparison")
    _load_example_module("spt_reg_year_comparison")
    _load_example_module("semi_supervised_regression_comparison")
    _load_example_module("uncertain_gt_density_conformal_comparison")
    _load_example_module("uncertain_gt_density_conformal_realdata_comparison")
    _load_example_module("causal_dr_uplift_comparison")
    _load_example_module("causal_dr_realdata_comparison")

    # Optional dependency path (zuko/flow backend) may not be present in all environments.
    try:
        _load_example_module("normalizing_flows_multitarget")
    except ImportError as exc:
        if "zuko" not in str(exc).lower():
            raise


def test_comprehensive_comparison_main_smoke(monkeypatch) -> None:
    mod = _load_example_module("comprehensive_comparison")

    def _quick_train_model(model, *args, **kwargs):
        return model

    def _quick_train_ensemble(n_models, dataloader, model_fn, loss_fn, *args, **kwargs):
        return [model_fn() for _ in range(n_models)]

    monkeypatch.setattr(mod, "train_model", _quick_train_model)
    monkeypatch.setattr(mod, "train_ensemble", _quick_train_ensemble)
    monkeypatch.setattr(mod, "plot_comparison", lambda *args, **kwargs: None)

    mod.main()


def test_comprehensive_loss_comparison_main_smoke(monkeypatch) -> None:
    mod = _load_example_module("comprehensive_loss_comparison")

    monkeypatch.setattr(mod, "train_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.plt, "show", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.DensityWeightedLoss, "fit_density", lambda self, y: None)

    mod.main()


def test_imbalanced_regression_main_smoke(monkeypatch) -> None:
    mod = _load_example_module("imbalanced_regression")

    class _TinyDataset:
        def __init__(self, n_samples: int = 16):
            xs = torch.linspace(-2, 2, n_samples).numpy().astype("float32")
            self.x = xs
            self.y = (xs**3).astype("float32")
            self.y_clean = self.y.copy()

        def __len__(self) -> int:
            return len(self.x)

        def __getitem__(self, idx: int):
            return (
                torch.tensor([self.x[idx]], dtype=torch.float32),
                torch.tensor([self.y[idx]], dtype=torch.float32),
                idx,
            )

    class _TinyHeteroscedastic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)

        def forward(self, x, return_var=False):
            mean = self.linear(x)
            if return_var:
                return mean, torch.zeros_like(mean)
            return mean

    monkeypatch.setattr(mod, "ImbalancedRegressionDataset", _TinyDataset)
    monkeypatch.setattr(mod, "HeteroscedasticRegressor", _TinyHeteroscedastic)
    monkeypatch.setattr(mod, "train_baseline", lambda dataset, n_epochs=100: _TinyHeteroscedastic())
    monkeypatch.setattr(
        mod, "train_density_weighted", lambda dataset, n_epochs=100: _TinyHeteroscedastic()
    )
    monkeypatch.setattr(mod, "train_lds", lambda dataset, n_epochs=100: _TinyHeteroscedastic())
    monkeypatch.setattr(mod, "evaluate_on_regions", lambda *args, **kwargs: (0.1, 0.2))
    monkeypatch.setattr(mod, "visualize_predictions", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mod,
        "compute_calibration_error",
        lambda *args, **kwargs: (0.05, [0.0, 0.5, 1.0], [0.0, 0.45, 0.95]),
    )
    monkeypatch.setattr(mod, "plot_calibration_curve", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "DataLoader", lambda *args, **kwargs: [])
    monkeypatch.setattr(mod.plt, "show", lambda *args, **kwargs: None)

    mod.main()


def test_evaluate_conformal_methods_main_smoke(monkeypatch) -> None:
    mod = _load_example_module("evaluate_conformal_methods")

    class _StubConformalLoss:
        def __init__(self, method="split", alpha=0.1, **kwargs):
            self.method = method
            self.alpha = alpha

        def __call__(self, y_pred, y_true):
            return ((y_pred[..., :1] - y_true) ** 2).mean()

        def calibrate(self, y_pred_cal, y_cal):
            return None

        def predict_interval(self, y_pred_test):
            if y_pred_test.shape[-1] >= 2:
                lower = y_pred_test[:, :1] - 0.05
                upper = y_pred_test[:, 1:2] + 0.05
            else:
                lower = y_pred_test - 0.1
                upper = y_pred_test + 0.1
            return lower, upper

    monkeypatch.setattr(
        mod,
        "generate_synthetic_data",
        lambda n_samples=2000, n_features=1: (
            torch.randn(60, n_features),
            torch.randn(60, 1),
            torch.randn(60, 1),
        ),
    )
    monkeypatch.setattr(mod, "ConformalLoss", _StubConformalLoss)
    monkeypatch.setattr(mod, "train_model", lambda model, *args, **kwargs: model.eval())
    monkeypatch.setattr(mod, "timed_call", lambda fn, *args, **kwargs: (fn(*args, **kwargs), 0.0))
    monkeypatch.setattr(mod.plt, "savefig", lambda *args, **kwargs: None)

    mod.main()


def test_normalizing_flows_multitarget_main_smoke(monkeypatch) -> None:
    try:
        mod = _load_example_module("normalizing_flows_multitarget")
    except ImportError as exc:
        if "zuko" in str(exc).lower():
            pytest.skip("optional zuko dependency not available")
        raise

    class _StubLoss:
        pass

    def _stub_model_and_flow(*args, **kwargs):
        return torch.nn.Linear(1, 4), torch.nn.Linear(16, 4), _StubLoss()

    def _stub_generate_data(n_samples=2000, noise_level=0.1):
        x = torch.linspace(-1, 1, 24).unsqueeze(1)
        y = torch.stack([x[:, 0], x[:, 0] ** 2], dim=1)
        return x, y

    def _stub_predict_with_flow(model, flow, loss_fn, x_test, n_samples=100):
        n = x_test.shape[0]
        mean = torch.zeros(n, 2)
        std = torch.ones(n, 2) * 0.1
        samples = torch.zeros(n, min(n_samples, 5), 2)
        return samples, mean, std

    monkeypatch.setattr(mod, "generate_complex_multitarget_data", _stub_generate_data)
    monkeypatch.setattr(mod, "create_multitarget_flow_model", _stub_model_and_flow)
    monkeypatch.setattr(mod, "train_flow_model", lambda *args, **kwargs: [0.0])
    monkeypatch.setattr(mod, "predict_with_flow", _stub_predict_with_flow)
    monkeypatch.setattr(mod, "visualize_predictions", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "timed_call", lambda fn, *args, **kwargs: (fn(*args, **kwargs), 0.0))

    mod.main()


def test_ood_selective_prediction_comparison_main_smoke() -> None:
    mod = _load_example_module("ood_selective_prediction_comparison")
    cfg = mod.OODConfig(
        n_train=24,
        n_cal=12,
        n_id_test=16,
        n_ood_test=16,
        epochs=1,
        ensemble_size=2,
        mc_samples=4,
        swag_samples=3,
        bnn_samples=4,
    )
    mod.main(cfg)


def test_ood_selective_prediction_realdata_comparison_main_smoke() -> None:
    mod = _load_example_module("ood_selective_prediction_realdata_comparison")
    cfg = mod.OODRealDataConfig(
        n_train=64,
        n_cal=24,
        n_id_test=24,
        n_ood_test=24,
        epochs=1,
        ensemble_size=2,
        mc_samples=4,
        swag_samples=3,
        bnn_samples=4,
    )
    mod.main(cfg)


def test_transformed_target_regression_comparison_main_smoke() -> None:
    mod = _load_example_module("transformed_target_regression_comparison")
    cfg = mod.TransformComparisonConfig(
        n_train=64,
        n_test=32,
        hidden=8,
        epochs=2,
    )
    mod.main(cfg)


def test_spt_reg_synthetic_comparison_main_smoke() -> None:
    mod = _load_example_module("spt_reg_synthetic_comparison")
    cfg = mod.SPTRegSyntheticConfig(
        n_source=96,
        n_target_unlabeled=48,
        n_target_cal=24,
        n_target_test=24,
        n_support=64,
        n_bins=12,
        n_samples_eval=16,
        target_label_budget=16,
    )
    mod.main(cfg)


def test_spt_reg_realdata_comparison_main_smoke() -> None:
    mod = _load_example_module("spt_reg_realdata_comparison")
    cfg = mod.SPTRegRealDataConfig(
        n_source=160,
        n_target_unlabeled=32,
        n_target_cal=24,
        n_target_test=24,
        n_support=64,
        n_bins=12,
        n_samples_eval=16,
        target_label_budget=16,
    )
    mod.main(cfg)


def test_spt_reg_year_comparison_main_smoke(tmp_path: Path) -> None:
    mod = _load_example_module("spt_reg_year_comparison")
    data_path = tmp_path / "year_like.csv"
    _write_tiny_tabular_csv(data_path, n_rows=320, n_features=10, seed=42)
    cfg = mod.SPTRegYearConfig(
        dataset_path=str(data_path),
        allow_download=False,
        n_source=160,
        n_target_unlabeled=32,
        n_target_cal=24,
        n_target_test=24,
        n_support=64,
        n_bins=12,
        n_samples_eval=16,
        target_label_budget=16,
    )
    mod.main(cfg)


def test_spt_reg_year_scale_split_sizes() -> None:
    mod = _load_example_module("spt_reg_year_comparison")
    base = mod.SPTRegYearConfig()
    scaled = mod.spt_year_scale_split_sizes(base, 3)
    assert scaled.n_source == base.n_source * 3
    assert scaled.n_target_unlabeled == base.n_target_unlabeled * 3
    assert scaled.target_label_budget == base.target_label_budget * 3
    assert mod.spt_year_split_row_budget(scaled) == mod.spt_year_split_row_budget(base) * 3


def test_spt_reg_year_max_dataset_rows_subsamples(tmp_path: Path) -> None:
    mod = _load_example_module("spt_reg_year_comparison")
    data_path = tmp_path / "year_many.csv"
    _write_tiny_tabular_csv(data_path, n_rows=500, n_features=10, seed=43)
    cfg = mod.SPTRegYearConfig(
        dataset_path=str(data_path),
        allow_download=False,
        max_dataset_rows=200,
        n_source=80,
        n_target_unlabeled=32,
        n_target_cal=24,
        n_target_test=24,
        n_support=64,
        n_bins=12,
        n_samples_eval=16,
        target_label_budget=16,
        seed=99,
    )
    splits, _ = mod._make_year_split(cfg)
    assert splits["source_x"].shape[0] == 80
    assert splits["target_pool_x"].shape[0] == 32 + 24 + 24


def test_spt_reg_photoz_comparison_main_smoke() -> None:
    mod = _load_example_module("spt_reg_photoz_comparison")
    cfg = mod.SPTRegPhotoZConfig(
        n_train=96,
        n_target_unlabeled=24,
        n_target_cal=24,
        n_target_test=24,
        epochs=2,
        hidden=16,
        n_support=64,
        n_bins=12,
        n_samples_eval=16,
        target_label_budget=16,
        sample_size_if_generate=256,
        force_simulated=True,
    )
    mod.main(cfg)


def test_semi_supervised_regression_comparison_main_smoke() -> None:
    mod = _load_example_module("semi_supervised_regression_comparison")
    cfg = mod.SemiSupervisedRegressionConfig(
        n_labeled=32,
        n_unlabeled=64,
        n_test=32,
        hidden=8,
        teacher_epochs=2,
        student_epochs=3,
    )
    mod.main(cfg)


def test_eiv_method_comparison_main_smoke() -> None:
    mod = _load_example_module("eiv_method_comparison")
    cfg = mod.EIVConfig(n_train=24, n_test=24, epochs=1, hidden=8)
    mod.main(cfg)


def test_eiv_method_realdata_comparison_main_smoke() -> None:
    mod = _load_example_module("eiv_method_realdata_comparison")
    cfg = mod.EIVRealDataConfig(n_train=64, n_test=32, epochs=1, hidden=8)
    mod.main(cfg)


def test_noisy_label_comparison_main_smoke() -> None:
    mod = _load_example_module("noisy_label_comparison")
    cfg = mod.NoisyLabelComparisonConfig(
        n_train=32,
        n_cal=16,
        n_test=24,
        epochs=1,
        batch_size=8,
        hidden=8,
    )
    mod.main(cfg)


def test_noisy_label_realdata_comparison_main_smoke() -> None:
    mod = _load_example_module("noisy_label_realdata_comparison")
    cfg = mod.NoisyLabelRealDataConfig(
        n_train=64,
        n_cal=24,
        n_test=24,
        epochs=1,
        batch_size=16,
        hidden=8,
    )
    mod.main(cfg)


def test_multimodal_method_comparison_main_smoke() -> None:
    mod = _load_example_module("multimodal_method_comparison")
    cfg = mod.MultimodalComparisonConfig(
        n_train=32,
        n_test=24,
        batch_size=8,
        epochs=1,
        hidden=8,
        mdn_components=2,
        eval_samples=8,
        flow_context_dim=4,
        flow_transforms=2,
    )
    mod.main(cfg)


def test_photoz_benchmark_comparison_main_smoke() -> None:
    mod = _load_example_module("photoz_benchmark_comparison")
    cfg = mod.PhotoZBenchmarkConfig(
        n_train=48,
        n_cal=16,
        n_test=16,
        batch_size=16,
        epochs=1,
        hidden=8,
        sample_size_if_generate=160,
        force_simulated=True,
        allow_download=False,
    )
    mod.main(cfg)


def test_photoz_benchmark_comparison_accepts_grizy_external_data(tmp_path: Path) -> None:
    mod = _load_example_module("photoz_benchmark_comparison")
    dataset = tmp_path / "hsc_like.csv"
    rows = 96
    pd.DataFrame(
        {
            "objid": list(range(rows)),
            "spec_z": [0.05 + 0.002 * i for i in range(rows)],
            "spec_z_err": [0.01] * rows,
            "g_r": [0.3 + 0.001 * i for i in range(rows)],
            "r_i": [0.2 + 0.001 * i for i in range(rows)],
            "i_z": [0.15 + 0.001 * i for i in range(rows)],
            "z_y": [0.1 + 0.001 * i for i in range(rows)],
            "g_r_err": [0.02] * rows,
            "r_i_err": [0.02] * rows,
            "i_z_err": [0.02] * rows,
            "z_y_err": [0.02] * rows,
        }
    ).to_csv(dataset, index=False)

    cfg = mod.PhotoZBenchmarkConfig(
        n_train=48,
        n_cal=16,
        n_test=16,
        batch_size=16,
        epochs=1,
        hidden=8,
        dataset_path=str(dataset),
        require_real_data=True,
        allow_download=False,
    )
    mod.main(cfg)


def test_photoz_nnc_crps_rail_comparison_main_smoke() -> None:
    mod = _load_example_module("photoz_nnc_crps_rail_comparison")
    cfg = mod.PhotoZNNCConfig(
        n_train=48,
        n_cal=16,
        n_test=16,
        batch_size=16,
        epochs=1,
        hidden=8,
        n_bins=16,
        sample_size_if_generate=160,
        force_simulated=True,
        allow_download=False,
        temperature_max_iter=30,
    )
    mod.main(cfg)


def test_photoz_transferz_conformal_comparison_main_smoke() -> None:
    mod = _load_example_module("photoz_transferz_conformal_comparison")
    cfg = mod.PhotoZTransferZConformalConfig(
        n_train=48,
        n_cal=16,
        n_conformal=16,
        n_test=16,
        batch_size=16,
        epochs=1,
        hidden=8,
        n_mc_samples=6,
        n_bins=16,
        sample_size_if_generate=160,
        force_simulated=True,
    )
    mod.main(cfg)


def test_multimodal_method_realdata_comparison_main_smoke() -> None:
    mod = _load_example_module("multimodal_method_realdata_comparison")
    cfg = mod.MultimodalRealDataConfig(
        n_train=64,
        n_test=24,
        batch_size=8,
        epochs=1,
        hidden=8,
        mdn_components=2,
        eval_samples=8,
        flow_context_dim=4,
        flow_transforms=2,
    )
    mod.main(cfg)


def test_contrastive_flow_parameter_estimation_comparison_main_smoke() -> None:
    mod = _load_example_module("contrastive_flow_parameter_estimation_comparison")
    cfg = mod.ContrastiveFlowComparisonConfig(
        n_train=32,
        n_test=16,
        events_per_experiment=24,
        batch_size=8,
        epochs=1,
        hidden=8,
        flow_context_dim=4,
        flow_transforms=2,
        n_negatives=2,
        mu_grid_size=9,
        nuisance_grid_size=7,
    )
    mod.main(cfg)


def test_contrastive_flow_photoz_proxy_comparison_main_smoke() -> None:
    mod = _load_example_module("contrastive_flow_photoz_proxy_comparison")
    cfg = mod.ContrastivePhotoZProxyConfig(
        n_train=48,
        n_cal=16,
        n_test=16,
        batch_size=8,
        epochs=1,
        hidden=8,
        flow_context_dim=4,
        flow_transforms=2,
        n_negatives=2,
        n_train_experiments=16,
        n_test_experiments=8,
        catalog_size=12,
        force_simulated=True,
        allow_download=False,
        sample_size_if_generate=160,
    )
    mod.main(cfg)


def test_ppi_photoz_inference_comparison_main_smoke() -> None:
    mod = _load_example_module("ppi_photoz_inference_comparison")
    cfg = mod.PPIPhotoZConfig(
        n_labeled=64,
        n_unlabeled=320,
        n_boot=120,
    )
    mod.main(cfg)


def test_ordinal_regression_comparison_main_smoke() -> None:
    mod = _load_example_module("ordinal_regression_comparison")
    cfg = mod.OrdinalComparisonConfig(
        n_train=96,
        n_test=48,
        hidden=12,
        epochs=2,
        batch_size=24,
    )
    mod.main(cfg)


def test_ordinal_regression_realdata_comparison_main_smoke() -> None:
    mod = _load_example_module("ordinal_regression_realdata_comparison")
    cfg = mod.OrdinalRealDataConfig(
        n_train=128,
        n_test=64,
        hidden=12,
        epochs=2,
        batch_size=24,
    )
    mod.main(cfg)


def test_ordinal_uncertain_ground_truth_comparison_main_smoke() -> None:
    mod = _load_example_module("ordinal_uncertain_ground_truth_comparison")
    cfg = mod.OrdinalUGTComparisonConfig(
        n_train=96,
        n_test=48,
        hidden=12,
        epochs=2,
        teacher_epochs=2,
        batch_size=24,
    )
    mod.main(cfg)


def test_censored_regression_comparison_main_smoke() -> None:
    mod = _load_example_module("censored_regression_comparison")
    cfg = mod.CensoredComparisonConfig(
        n_train=192,
        n_test=96,
        hidden=12,
        epochs=2,
        batch_size=32,
    )
    mod.main(cfg)


def test_censored_regression_realdata_comparison_main_smoke() -> None:
    mod = _load_example_module("censored_regression_realdata_comparison")
    cfg = mod.CensoredRealDataConfig(
        n_train=128,
        n_test=64,
        hidden=12,
        epochs=2,
        batch_size=24,
    )
    mod.main(cfg)


def test_propensity_tail_regression_comparison_main_smoke() -> None:
    mod = _load_example_module("propensity_tail_regression_comparison")
    cfg = mod.PropensityTailConfig(
        n_train_pool=256,
        n_test=96,
        hidden=12,
        epochs=2,
        batch_size=32,
    )
    mod.main(cfg)


def test_constraints_calibration_comparison_main_smoke() -> None:
    mod = _load_example_module("constraints_calibration_comparison")
    cfg = mod.ConstraintCalibrationConfig(
        n_cal=192,
        n_test=96,
    )
    mod.main(cfg)


def test_uncertain_gt_density_conformal_comparison_main_smoke() -> None:
    mod = _load_example_module("uncertain_gt_density_conformal_comparison")
    cfg = mod.UncertainGTConformalConfig(
        n_cal=160,
        n_test=96,
        n_mc_samples=10,
    )
    mod.main(cfg)


def test_uncertain_gt_density_conformal_realdata_comparison_main_smoke() -> None:
    mod = _load_example_module("uncertain_gt_density_conformal_realdata_comparison")
    cfg = mod.UncertainGTConformalRealDataConfig(
        n_train=120,
        n_cal=64,
        n_test=64,
        hidden=12,
        epochs=2,
        n_mc_samples=8,
    )
    mod.main(cfg)


def test_causal_dr_uplift_comparison_main_smoke() -> None:
    mod = _load_example_module("causal_dr_uplift_comparison")
    cfg = mod.CausalDRConfig(
        n_samples=400,
        folds=2,
    )
    mod.main(cfg)


def test_causal_dr_realdata_comparison_main_smoke() -> None:
    mod = _load_example_module("causal_dr_realdata_comparison")
    cfg = mod.CausalDRRealDataConfig(
        n_samples=280,
        folds=2,
    )
    mod.main(cfg)
