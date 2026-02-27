from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg", force=True)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _load_example_module(stem: str) -> ModuleType:
    path = EXAMPLES_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_example_summary_{stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXAMPLES_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _assert_summary_schema(path: Path, *, task_substring: str, required_methods: set[str]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "comparison_example_summary"
    assert payload["version"] == 1
    assert task_substring.lower() in payload["task"].lower()
    assert isinstance(payload["config"], dict)
    assert isinstance(payload["rows"], list)
    assert payload["rows"]
    methods = {str(row.get("Method")) for row in payload["rows"]}
    assert required_methods <= methods
    for row in payload["rows"]:
        assert "Method" in row


def _load_payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_method(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = payload["rows"]
    assert isinstance(rows, list)
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        assert isinstance(row, dict)
        method = row.get("Method")
        assert isinstance(method, str)
        out[method] = row
    return out


def _assert_row_has_keys(row: dict[str, object], keys: list[str]) -> None:
    for key in keys:
        assert key in row, f"Missing key {key!r} in row {row.get('Method')!r}"


def _assert_finite_numeric(value: object) -> None:
    assert isinstance(value, (int, float))
    if isinstance(value, float):
        assert value == value  # not NaN


def _assert_non_negative(value: object) -> None:
    _assert_finite_numeric(value)
    assert float(value) >= 0.0


def _assert_probability(value: object) -> None:
    _assert_finite_numeric(value)
    v = float(value)
    assert 0.0 <= v <= 1.0


def test_ood_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("ood_selective_prediction_comparison")
    out = tmp_path / "ood_summary.json"
    cfg = mod.OODConfig(
        n_train=24,
        n_id_test=16,
        n_ood_test=16,
        epochs=1,
        ensemble_size=2,
        mc_samples=4,
        swag_samples=3,
        bnn_samples=4,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="ood",
        required_methods={
            "DeepEnsemble",
            "HeteroscedasticEnsemble",
            "MCDropoutWrapper (proxy)",
            "SWAG",
            "BayesianNeuralNetwork",
        },
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in (
        "DeepEnsemble",
        "HeteroscedasticEnsemble",
        "MCDropoutWrapper (proxy)",
        "SWAG",
        "BayesianNeuralNetwork",
    ):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "MSE_ID",
                "MSE_OOD",
                "AURC",
                "rej20_risk",
                "rej20_cov",
                "ood_unc_gap",
                "train_s",
                "eval_s",
            ],
        )
        _assert_finite_numeric(row["MSE_ID"])
        _assert_finite_numeric(row["AURC"])
        _assert_probability(row["rej20_cov"])
        _assert_non_negative(row["rej20_risk"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_ood_realdata_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("ood_selective_prediction_realdata_comparison")
    out = tmp_path / "ood_realdata_summary.json"
    cfg = mod.OODRealDataConfig(
        n_train=64,
        n_id_test=24,
        n_ood_test=24,
        epochs=1,
        ensemble_size=2,
        mc_samples=4,
        swag_samples=3,
        bnn_samples=4,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="real-data",
        required_methods={
            "DeepEnsemble",
            "HeteroscedasticEnsemble",
            "MCDropoutWrapper (proxy)",
            "SWAG",
            "BayesianNeuralNetwork",
        },
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in (
        "DeepEnsemble",
        "HeteroscedasticEnsemble",
        "MCDropoutWrapper (proxy)",
        "SWAG",
        "BayesianNeuralNetwork",
    ):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "MSE_ID",
                "MSE_OOD",
                "AURC",
                "rej20_risk",
                "rej20_cov",
                "ood_unc_gap",
                "train_s",
                "eval_s",
            ],
        )
        _assert_non_negative(row["MSE_ID"])
        _assert_non_negative(row["MSE_OOD"])
        _assert_non_negative(row["AURC"])
        _assert_probability(row["rej20_cov"])
        _assert_non_negative(row["rej20_risk"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_eiv_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("eiv_method_comparison")
    out = tmp_path / "eiv_summary.json"
    cfg = mod.EIVConfig(n_train=24, n_test=24, epochs=1, hidden=8)
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="eiv",
        required_methods={"Baseline MSE", "FunctionalEIV (analytic)", "ODR"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("Baseline MSE", "FunctionalEIV (analytic)", "ODR"):
        row = rows[method]
        _assert_row_has_keys(
            row,
            ["clean_mse", "obs_mse", "obs_input_clean_target_mse", "train_s", "eval_s"],
        )
        _assert_non_negative(row["clean_mse"])
        _assert_non_negative(row["obs_mse"])
        _assert_non_negative(row["obs_input_clean_target_mse"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_eiv_realdata_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("eiv_method_realdata_comparison")
    out = tmp_path / "eiv_realdata_summary.json"
    cfg = mod.EIVRealDataConfig(n_train=64, n_test=32, epochs=1, hidden=8)
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="real-data",
        required_methods={"Baseline MSE", "FunctionalEIV (analytic)", "ODR"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("Baseline MSE", "FunctionalEIV (analytic)", "ODR"):
        row = rows[method]
        _assert_row_has_keys(
            row,
            ["clean_mse", "obs_mse", "obs_input_clean_target_mse", "train_s", "eval_s"],
        )
        _assert_non_negative(row["clean_mse"])
        _assert_non_negative(row["obs_mse"])
        _assert_non_negative(row["obs_input_clean_target_mse"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_multimodal_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("multimodal_method_comparison")
    out = tmp_path / "multimodal_summary.json"
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
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="multimodal",
        required_methods={"GaussianNLL", "MDN", "NormalizingFlow"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("GaussianNLL", "MDN", "NormalizingFlow"):
        row = rows[method]
        _assert_row_has_keys(row, ["train_s", "eval_s", "Notes"])
        train_s = row["train_s"]
        eval_s = row["eval_s"]
        if train_s is not None:
            _assert_non_negative(train_s)
        if eval_s is not None:
            _assert_non_negative(eval_s)
    _assert_row_has_keys(rows["GaussianNLL"], ["NLL", "Energy", "MCE"])
    _assert_row_has_keys(rows["MDN"], ["NLL", "Energy", "MCE"])
    for method in ("GaussianNLL", "MDN"):
        _assert_non_negative(rows[method]["NLL"])
        _assert_non_negative(rows[method]["Energy"])
        _assert_non_negative(rows[method]["MCE"])


def test_multimodal_realdata_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("multimodal_method_realdata_comparison")
    out = tmp_path / "multimodal_realdata_summary.json"
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
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="real-data",
        required_methods={"GaussianNLL", "MDN", "NormalizingFlow"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("GaussianNLL", "MDN", "NormalizingFlow"):
        row = rows[method]
        _assert_row_has_keys(row, ["train_s", "eval_s", "Notes"])
        if row["train_s"] is not None:
            _assert_non_negative(row["train_s"])
        if row["eval_s"] is not None:
            _assert_non_negative(row["eval_s"])
    _assert_row_has_keys(rows["GaussianNLL"], ["NLL", "Energy", "MCE"])
    _assert_row_has_keys(rows["MDN"], ["NLL", "Energy", "MCE"])
    for method in ("GaussianNLL", "MDN"):
        _assert_non_negative(rows[method]["NLL"])
        _assert_non_negative(rows[method]["Energy"])
        _assert_non_negative(rows[method]["MCE"])


def test_photoz_benchmark_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("photoz_benchmark_comparison")
    out = tmp_path / "photoz_benchmark_summary.json"
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
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="photo-z",
        required_methods={"MSE", "Huber", "GaussianNLL", "Quantile90", "FunctionalEIV"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("MSE", "Huber", "GaussianNLL", "Quantile90", "FunctionalEIV"):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "RMSE",
                "MAE",
                "NMAD",
                "CatastrophicRate",
                "HighZ_MAE",
                "train_s",
                "eval_s",
            ],
        )
        _assert_non_negative(row["RMSE"])
        _assert_non_negative(row["MAE"])
        _assert_non_negative(row["NMAD"])
        _assert_probability(row["CatastrophicRate"])
        _assert_non_negative(row["HighZ_MAE"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])
    for method in ("GaussianNLL", "Quantile90"):
        row = rows[method]
        _assert_row_has_keys(row, ["Cov90", "Width90"])
        _assert_probability(row["Cov90"])
        _assert_non_negative(row["Width90"])
    _assert_non_negative(rows["GaussianNLL"]["NLL"])


def test_photoz_nnc_crps_rail_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("photoz_nnc_crps_rail_comparison")
    out = tmp_path / "photoz_nnc_summary.json"
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
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="ordered-bin",
        required_methods={
            "BinnedCE",
            "BinnedCE+TempScaling",
            "OrderedBinCRPS",
            "OrderedBinCRPS+TempScaling",
            "GaussianNLL",
            "MultiQuantileLoss",
        },
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in (
        "BinnedCE",
        "BinnedCE+TempScaling",
        "OrderedBinCRPS",
        "OrderedBinCRPS+TempScaling",
        "GaussianNLL",
        "MultiQuantileLoss",
    ):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "RMSE",
                "MAE",
                "NMAD",
                "CatastrophicRate",
                "HighZ_MAE",
                "CRPS",
                "PDF_NLL",
                "PITChi2",
                "NativeCov90",
                "NativeWidth90",
                "train_s",
                "eval_s",
                "calibrate_s",
            ],
        )
        _assert_non_negative(row["RMSE"])
        _assert_non_negative(row["MAE"])
        _assert_non_negative(row["NMAD"])
        _assert_probability(row["CatastrophicRate"])
        _assert_non_negative(row["HighZ_MAE"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])
        _assert_non_negative(row["calibrate_s"])
        _assert_probability(row["NativeCov90"])
        _assert_non_negative(row["NativeWidth90"])
    for method in (
        "BinnedCE",
        "BinnedCE+TempScaling",
        "OrderedBinCRPS",
        "OrderedBinCRPS+TempScaling",
    ):
        _assert_non_negative(rows[method]["CRPS"])
        _assert_non_negative(rows[method]["PDF_NLL"])
        _assert_non_negative(rows[method]["PITChi2"])


def test_ppi_photoz_inference_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("ppi_photoz_inference_comparison")
    out = tmp_path / "ppi_photoz_summary.json"
    cfg = mod.PPIPhotoZConfig(
        n_labeled=64,
        n_unlabeled=320,
        n_boot=120,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="inference",
        required_methods={
            "LabeledOnlyMeanCI",
            "PPIMeanCI",
            "LabeledOnlyQuantileCI",
            "PPIQuantileCI",
        },
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in (
        "LabeledOnlyMeanCI",
        "PPIMeanCI",
        "LabeledOnlyQuantileCI",
        "PPIQuantileCI",
    ):
        row = rows[method]
        _assert_row_has_keys(
            row,
            ["Estimate", "AbsError", "CIWidth", "CoversTruth", "train_s", "eval_s", "Notes"],
        )
        _assert_non_negative(row["AbsError"])
        _assert_non_negative(row["CIWidth"])
        _assert_probability(row["CoversTruth"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_ordinal_regression_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("ordinal_regression_comparison")
    out = tmp_path / "ordinal_comparison_summary.json"
    cfg = mod.OrdinalComparisonConfig(
        n_train=96,
        n_test=48,
        hidden=12,
        epochs=2,
        batch_size=24,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="ordinal",
        required_methods={"OrdinalCrossEntropy", "CumulativeLink", "CORAL"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("OrdinalCrossEntropy", "CumulativeLink", "CORAL"):
        row = rows[method]
        _assert_row_has_keys(row, ["Accuracy", "OrdinalMAE", "QWK", "train_s", "eval_s", "Notes"])
        _assert_probability(row["Accuracy"])
        _assert_non_negative(row["OrdinalMAE"])
        _assert_finite_numeric(row["QWK"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_censored_regression_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("censored_regression_comparison")
    out = tmp_path / "censored_comparison_summary.json"
    cfg = mod.CensoredComparisonConfig(
        n_train=192,
        n_test=96,
        hidden=12,
        epochs=2,
        batch_size=32,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="censored",
        required_methods={"CensoredGaussianNLL", "CensoredQuantile", "AFT"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("CensoredGaussianNLL", "CensoredQuantile", "AFT"):
        row = rows[method]
        _assert_row_has_keys(
            row,
            ["MAE_true", "ObsMAE", "CIndex", "CensorRate", "train_s", "eval_s", "Notes"],
        )
        _assert_non_negative(row["MAE_true"])
        _assert_finite_numeric(row["ObsMAE"])
        _assert_finite_numeric(row["CIndex"])
        _assert_probability(row["CensorRate"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_propensity_tail_regression_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("propensity_tail_regression_comparison")
    out = tmp_path / "propensity_tail_summary.json"
    cfg = mod.PropensityTailConfig(
        n_train_pool=256,
        n_test=96,
        hidden=12,
        epochs=2,
        batch_size=32,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="selection bias",
        required_methods={"MSE", "DensityWeighted", "PropensityWeighted"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("MSE", "DensityWeighted", "PropensityWeighted"):
        row = rows[method]
        _assert_row_has_keys(
            row,
            ["MAE", "TailMAE90", "TailRMSE90", "ObservedRate", "train_s", "eval_s", "Notes"],
        )
        _assert_non_negative(row["MAE"])
        _assert_non_negative(row["TailMAE90"])
        _assert_non_negative(row["TailRMSE90"])
        _assert_probability(row["ObservedRate"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_constraints_calibration_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("constraints_calibration_comparison")
    out = tmp_path / "constraints_calibration_summary.json"
    cfg = mod.ConstraintCalibrationConfig(n_cal=192, n_test=96)
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="calibration transforms",
        required_methods={"Raw", "Calibrated+Constrained"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("Raw", "Calibrated+Constrained"):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "MAE",
                "NLL",
                "PITChi2",
                "CrossingRate",
                "BoundViolation",
                "train_s",
                "eval_s",
                "Notes",
            ],
        )
        _assert_non_negative(row["MAE"])
        _assert_non_negative(row["NLL"])
        _assert_non_negative(row["PITChi2"])
        _assert_probability(row["CrossingRate"])
        _assert_probability(row["BoundViolation"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_uncertain_gt_density_conformal_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("uncertain_gt_density_conformal_comparison")
    out = tmp_path / "uncertain_gt_density_conformal_summary.json"
    cfg = mod.UncertainGTConformalConfig(
        n_cal=192,
        n_test=96,
        n_mc_samples=12,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="uncertain ground-truth",
        required_methods={
            "SplitConformal",
            "DensityConformal",
            "PrevalenceAdjustedCP",
            "MonteCarloConformal",
        },
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in (
        "SplitConformal",
        "DensityConformal",
        "PrevalenceAdjustedCP",
        "MonteCarloConformal",
    ):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "Coverage90",
                "Width90",
                "NoisyTargetNLL",
                "ConsistencyLoss",
                "PseudoLabelNLL",
                "train_s",
                "eval_s",
                "Notes",
            ],
        )
        _assert_probability(row["Coverage90"])
        _assert_non_negative(row["Width90"])
        _assert_finite_numeric(row["NoisyTargetNLL"])
        _assert_non_negative(row["ConsistencyLoss"])
        _assert_finite_numeric(row["PseudoLabelNLL"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_causal_dr_uplift_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("causal_dr_uplift_comparison")
    out = tmp_path / "causal_dr_uplift_summary.json"
    cfg = mod.CausalDRConfig(n_samples=600, folds=2)
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="causal",
        required_methods={
            "Uplift-NaiveDiff",
            "Uplift-DRATE",
            "Uplift-DRCATE",
            "AstronomyBias-NaiveDiff",
            "AstronomyBias-DRATE",
            "AstronomyBias-DRCATE",
        },
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in (
        "Uplift-NaiveDiff",
        "Uplift-DRATE",
        "Uplift-DRCATE",
        "AstronomyBias-NaiveDiff",
        "AstronomyBias-DRATE",
        "AstronomyBias-DRCATE",
    ):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "ATE_true",
                "ATE_hat",
                "ATE_abs_error",
                "CI_contains_true",
                "CI_width",
                "OverlapRate",
                "MinESS",
                "train_s",
                "Notes",
            ],
        )
        _assert_non_negative(row["ATE_abs_error"])
        _assert_probability(row["CI_contains_true"])
        _assert_non_negative(row["CI_width"])
        _assert_probability(row["OverlapRate"])
        _assert_non_negative(row["MinESS"])
        _assert_non_negative(row["train_s"])


def test_noisy_label_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("noisy_label_comparison")
    out = tmp_path / "noisy_label_summary.json"
    cfg = mod.NoisyLabelComparisonConfig(
        n_train=32,
        n_cal=16,
        n_test=24,
        epochs=1,
        batch_size=8,
        hidden=8,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="noisy labels",
        required_methods={"MSE", "Huber", "GaussianNLL", "Quantile90"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("MSE", "Huber", "GaussianNLL", "Quantile90"):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "CleanMSE",
                "ObsMSE",
                "ConformalCov90",
                "ConformalWidth90",
                "ConformalIS90",
                "train_s",
                "eval_s",
            ],
        )
        _assert_probability(row["ConformalCov90"])
        _assert_non_negative(row["ConformalWidth90"])
        _assert_non_negative(row["ConformalIS90"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])
    _assert_row_has_keys(rows["GaussianNLL"], ["NativeCov90", "NativeWidth90"])
    _assert_row_has_keys(rows["Quantile90"], ["NativeCov90", "NativeWidth90"])
    _assert_probability(rows["GaussianNLL"]["NativeCov90"])
    _assert_non_negative(rows["GaussianNLL"]["NativeWidth90"])
    _assert_probability(rows["Quantile90"]["NativeCov90"])
    _assert_non_negative(rows["Quantile90"]["NativeWidth90"])


def test_noisy_label_realdata_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("noisy_label_realdata_comparison")
    out = tmp_path / "noisy_label_realdata_summary.json"
    cfg = mod.NoisyLabelRealDataConfig(
        n_train=64,
        n_cal=24,
        n_test=24,
        epochs=1,
        batch_size=16,
        hidden=8,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="real-data",
        required_methods={"MSE", "Huber", "GaussianNLL", "Quantile90"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("MSE", "Huber", "GaussianNLL", "Quantile90"):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "CleanMSE",
                "ObsMSE",
                "ConformalCov90",
                "ConformalWidth90",
                "ConformalIS90",
                "train_s",
                "eval_s",
            ],
        )
        _assert_probability(row["ConformalCov90"])
        _assert_non_negative(row["ConformalWidth90"])
        _assert_non_negative(row["ConformalIS90"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])
    _assert_row_has_keys(rows["GaussianNLL"], ["NativeCov90", "NativeWidth90"])
    _assert_row_has_keys(rows["Quantile90"], ["NativeCov90", "NativeWidth90"])
