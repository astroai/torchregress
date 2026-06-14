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
    # Register in sys.modules so the dataclasses decorator can resolve
    # ``cls.__module__`` via ``sys.modules.get(cls.__module__).__dict__``
    # (required for ``@dataclass(frozen=True)`` which triggers the
    # ``_is_type`` check during __hash__ generation).
    sys.modules[module.__name__] = module
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
    _load_example_module("ot_shift_conformal_demo")
    _load_example_module("test_time_bayesian_linear_head_demo")
    _load_example_module("test_time_blr_predictive_adapter_demo")
    _load_example_module("gaussian_wasserstein_bound_demo")
    _load_example_module("wasserstein_bound_hybrid_pretrain_demo")
    _load_example_module("heteroscedastic_beta_nll_demo")
    _load_example_module("imbalanced_regression")
    _load_example_module("balanced_mse_demo")
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
    _load_example_module("poisson_regression_demo")
    _load_example_module("tweedie_regression_demo")
    _load_example_module("poisson_gaussian_mixture_demo")
    _load_example_module("expectile_regression_demo")
    _load_example_module("conformal_mondrian_demo")
    _load_example_module("eiv_algorithms_demo")
    _load_example_module("heteroscedastic_laplace_demo")
    _load_example_module("viz_diagnostic_gallery")
    _load_example_module("metrics_suite_showcase")
    _load_example_module("test_time_adaptation_suite")
    _load_example_module("bnn_and_batch_ensemble_demo")
    _load_example_module("transforms_and_augmentations_demo")
    _load_example_module("bayesian_learning_rule_demo")

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


def test_balanced_mse_demo_main_smoke(monkeypatch) -> None:
    import sys

    mod = _load_example_module("balanced_mse_demo")
    monkeypatch.setattr(
        sys,
        "argv",
        ["balanced_mse_demo.py", "--steps", "5", "--n", "64", "--seed", "0"],
    )
    mod.main()


def test_ot_shift_conformal_demo_main_smoke(monkeypatch) -> None:
    import sys

    mod = _load_example_module("ot_shift_conformal_demo")
    monkeypatch.setattr(sys, "argv", ["ot_shift_conformal_demo.py", "--seed", "1"])
    mod.main()


def test_test_time_bayesian_linear_head_demo_main_smoke(monkeypatch) -> None:
    import sys

    mod = _load_example_module("test_time_bayesian_linear_head_demo")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_time_bayesian_linear_head_demo.py",
            "--n-train",
            "40",
            "--n-test",
            "30",
            "--dim",
            "3",
            "--seed",
            "2",
        ],
    )
    mod.main()


def test_test_time_blr_predictive_adapter_demo_main_smoke(monkeypatch) -> None:
    import sys

    mod = _load_example_module("test_time_blr_predictive_adapter_demo")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_time_blr_predictive_adapter_demo.py",
            "--n-train",
            "32",
            "--n-test",
            "16",
            "--dim",
            "3",
            "--seed",
            "1",
        ],
    )
    mod.main()


def test_gaussian_wasserstein_bound_demo_main_smoke(monkeypatch) -> None:
    import sys

    mod = _load_example_module("gaussian_wasserstein_bound_demo")
    monkeypatch.setattr(
        sys, "argv", ["gaussian_wasserstein_bound_demo.py", "--batch", "2", "--dim", "2"]
    )
    mod.main()


def test_wasserstein_bound_hybrid_pretrain_demo_main_smoke(monkeypatch) -> None:
    import sys

    mod = _load_example_module("wasserstein_bound_hybrid_pretrain_demo")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wasserstein_bound_hybrid_pretrain_demo.py",
            "--pretrain-steps",
            "3",
            "--finetune-steps",
            "3",
            "--n",
            "48",
            "--seed",
            "1",
        ],
    )
    mod.main()


def test_heteroscedastic_beta_nll_demo_main_smoke(monkeypatch) -> None:
    import sys

    mod = _load_example_module("heteroscedastic_beta_nll_demo")
    monkeypatch.setattr(
        sys,
        "argv",
        ["heteroscedastic_beta_nll_demo.py", "--epochs", "1", "--seed", "0"],
    )
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


def test_external_comparison_conformal_vs_mapie_main_smoke() -> None:
    import json
    import tempfile

    mod = _load_example_module("external_comparison_conformal_vs_mapie")
    cfg = mod.ConformalExternalConfig(
        n_train=80,
        n_cal=40,
        n_test=80,
        epochs=2,
        batch_size=32,
        hidden=8,
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "summary.json")
        mod.main(cfg, summary_json_path=out)
        assert Path(out).exists()
        payload = json.loads(Path(out).read_text())
        rows = payload.get("rows") or []
        # 3 torchregress rows (always) + up to 5 external rows (2 MAPIE + 2 crepes + 1 torchcp).
        # Skipped external libraries still emit rows with null metrics, so the
        # total row count is invariant across environments (this is the schema-
        # stability contract the docs promise).
        assert len(rows) == 8, f"expected 8 rows (3 tr + 5 ext), got {len(rows)}: {rows}"
        libraries = sorted({r["Library"] for r in rows})
        assert libraries == ["MAPIE", "crepes", "torchcp", "torchregress"], libraries


@pytest.mark.skipif(
    not _load_example_module("external_comparison_conformal_vs_mapie")._CREPES_AVAILABLE,
    reason="crepes not installed (install via `uv pip install torchregress[external]`)",
)
def test_external_comparison_conformal_vs_mapie_crepes_paths_run() -> None:
    """Exercise the crepes split + CQR code paths when crepes is installed.

    The schema-stability smoke above only asserts row count + library set; it
    does not validate the actual crepes wrappers because the soft-import
    guard short-circuits when crepes is missing. This test forces the crepes
    branches to run on a tiny synthetic split.
    """
    mod = _load_example_module("external_comparison_conformal_vs_mapie")
    cfg = mod.ConformalExternalConfig(
        n_train=80,
        n_cal=40,
        n_test=80,
        epochs=2,
        batch_size=32,
        hidden=8,
    )
    splits = mod._simulate(cfg)
    # Direct calls into the crepes helpers; raises if the API contract breaks.
    lo_split, hi_split = mod._crepes_split_intervals(splits, alpha=cfg.alpha)
    assert lo_split.shape == (cfg.n_test,)
    assert hi_split.shape == (cfg.n_test,)
    lo_cqr, hi_cqr = mod._crepes_cqr_intervals(splits, alpha=cfg.alpha, seed=cfg.seed)
    assert lo_cqr.shape == (cfg.n_test,)
    assert hi_cqr.shape == (cfg.n_test,)
    # Sanity: intervals are well-ordered.
    assert (lo_split <= hi_split).all()
    assert (lo_cqr <= hi_cqr).all()


def test_external_comparison_bayesian_linear_vs_botorch_main_smoke() -> None:
    import tempfile

    mod = _load_example_module("external_comparison_bayesian_linear_vs_botorch")
    cfg = mod.BayesianLinearExternalConfig(
        n_train=12,
        n_test=32,
        dim=3,
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "summary.json")
        mod.main(cfg, summary_json_path=out)
        assert Path(out).exists()


def test_external_comparison_tweedie_vs_sklego_main_smoke() -> None:
    import tempfile

    mod = _load_example_module("external_comparison_tweedie_vs_sklego")
    cfg = mod.TweedieExternalConfig(
        n_train=200,
        n_test=80,
        epochs=2,
        batch_size=32,
        hidden=8,
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "summary.json")
        mod.main(cfg, summary_json_path=out)
        assert Path(out).exists()


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


def test_poisson_regression_demo_main_smoke(monkeypatch) -> None:
    mod = _load_example_module("poisson_regression_demo")
    orig_train_and_eval = mod.train_and_eval
    monkeypatch.setattr(
        mod,
        "train_and_eval",
        lambda *args, **kwargs: orig_train_and_eval(*args, epochs=1, **kwargs),
    )
    mod.main()


def test_tweedie_regression_demo_main_smoke(monkeypatch) -> None:
    mod = _load_example_module("tweedie_regression_demo")
    orig_train_tweedie = mod.train_tweedie
    monkeypatch.setattr(
        mod,
        "train_tweedie",
        lambda *args, **kwargs: orig_train_tweedie(*args, epochs=1, **kwargs),
    )
    mod.main()


def test_poisson_gaussian_mixture_demo_main_smoke(monkeypatch) -> None:
    mod = _load_example_module("poisson_gaussian_mixture_demo")
    orig_train_and_evaluate = mod.train_and_evaluate
    monkeypatch.setattr(
        mod,
        "train_and_evaluate",
        lambda *args, **kwargs: orig_train_and_evaluate(*args, epochs=1, **kwargs),
    )
    mod.main()


def test_expectile_regression_demo_main_smoke(monkeypatch) -> None:
    mod = _load_example_module("expectile_regression_demo")
    orig_train_model = mod.train_model
    monkeypatch.setattr(
        mod,
        "train_model",
        lambda *args, **kwargs: orig_train_model(*args, epochs=1, **kwargs),
    )
    mod.main()


def test_conformal_mondrian_demo_main_smoke() -> None:
    mod = _load_example_module("conformal_mondrian_demo")
    mod.main()


def test_eiv_algorithms_demo_main_smoke(monkeypatch) -> None:
    mod = _load_example_module("eiv_algorithms_demo")
    orig_train_baseline_model = mod.train_baseline_model
    orig_simex_train_wrapper = mod.simex_train_wrapper
    monkeypatch.setattr(
        mod,
        "train_baseline_model",
        lambda *args, **kwargs: orig_train_baseline_model(*args, epochs=1, **kwargs),
    )
    monkeypatch.setattr(
        mod,
        "simex_train_wrapper",
        lambda *args, **kwargs: orig_simex_train_wrapper(*args, epochs=1, **kwargs),
    )

    from torchregress.algorithms import SIMEX as OriginalSIMEX

    class FastSIMEX(OriginalSIMEX):
        def __init__(self, *args, **kwargs):
            kwargs["lambdas"] = [1.0]
            kwargs["n_simulations"] = 1
            super().__init__(*args, **kwargs)

    from torchregress.algorithms import LatentNN as OriginalLatentNN

    class FastLatentNN(OriginalLatentNN):
        def __init__(self, *args, **kwargs):
            kwargs["epochs"] = 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(mod, "SIMEX", FastSIMEX)
    monkeypatch.setattr(mod, "LatentNN", FastLatentNN)
    mod.main()


def test_heteroscedastic_laplace_demo_main_smoke(monkeypatch) -> None:
    mod = _load_example_module("heteroscedastic_laplace_demo")
    from torchregress.algorithms import HeteroscedasticLaplaceRegressor

    orig_fit = HeteroscedasticLaplaceRegressor.fit

    def fast_fit(self, *args, **kwargs):
        kwargs["epochs"] = 1
        return orig_fit(self, *args, **kwargs)

    monkeypatch.setattr(HeteroscedasticLaplaceRegressor, "fit", fast_fit)
    mod.main()
