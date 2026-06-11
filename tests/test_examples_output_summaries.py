from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import pandas as pd

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
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


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
    assert isinstance(value, (int, float))
    assert float(value) >= 0.0


def _assert_probability(value: object) -> None:
    _assert_finite_numeric(value)
    assert isinstance(value, (int, float))
    v = float(value)
    assert 0.0 <= v <= 1.0


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


def test_ood_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("ood_selective_prediction_comparison")
    out = tmp_path / "ood_summary.json"
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
                "ConformalCov90_ID",
                "ConformalCov90_OOD",
                "ConformalWidth90_ID",
                "ood_unc_gap",
                "train_s",
                "eval_s",
            ],
        )
        _assert_finite_numeric(row["MSE_ID"])
        _assert_finite_numeric(row["AURC"])
        _assert_probability(row["rej20_cov"])
        _assert_probability(row["ConformalCov90_ID"])
        _assert_probability(row["ConformalCov90_OOD"])
        _assert_non_negative(row["ConformalWidth90_ID"])
        _assert_non_negative(row["rej20_risk"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_ood_realdata_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("ood_selective_prediction_realdata_comparison")
    out = tmp_path / "ood_realdata_summary.json"
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
                "ConformalCov90_ID",
                "ConformalCov90_OOD",
                "ConformalWidth90_ID",
                "ood_unc_gap",
                "train_s",
                "eval_s",
            ],
        )
        _assert_non_negative(row["MSE_ID"])
        _assert_non_negative(row["MSE_OOD"])
        _assert_non_negative(row["AURC"])
        _assert_probability(row["rej20_cov"])
        _assert_probability(row["ConformalCov90_ID"])
        _assert_probability(row["ConformalCov90_OOD"])
        _assert_non_negative(row["ConformalWidth90_ID"])
        _assert_non_negative(row["rej20_risk"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_transformed_target_regression_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("transformed_target_regression_comparison")
    out = tmp_path / "transformed_target_summary.json"
    cfg = mod.TransformComparisonConfig(
        n_train=64,
        n_test=32,
        hidden=8,
        epochs=2,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="target transforms",
        required_methods={"MSE", "LogTransform", "BoxCox(0.25)", "SqrtTransform"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("MSE", "LogTransform", "BoxCox(0.25)", "SqrtTransform"):
        row = rows[method]
        _assert_row_has_keys(row, ["MSE", "MAE", "R2", "MAPE", "TailMAE80", "train_s", "eval_s"])
        _assert_non_negative(row["MSE"])
        _assert_non_negative(row["MAE"])
        _assert_non_negative(row["MAPE"])
        _assert_non_negative(row["TailMAE80"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_spt_reg_synthetic_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("spt_reg_synthetic_comparison")
    out = tmp_path / "spt_reg_synthetic_summary.json"
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
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="spt-reg",
        required_methods={
            "SourceGaussian",
            "WeightedSplitConformalGaussian",
            "SPTRegGaussian",
            "TargetRefitSmallGaussian",
            "SourceBinnedPDF",
            "SPTRegBinnedPDF",
            "SourceMDN",
            "SPTRegMDN",
        },
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in (
        "SourceGaussian",
        "WeightedSplitConformalGaussian",
        "SPTRegGaussian",
        "TargetRefitSmallGaussian",
        "SourceBinnedPDF",
        "SPTRegBinnedPDF",
        "SourceMDN",
        "SPTRegMDN",
    ):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "Family",
                "MSE",
                "MAE",
                "TailRMSE90",
                "NLL",
                "CRPS",
                "Cov90",
                "Width90",
                "AURC",
                "PPIMeanCIWidth",
                "PPIMeanCICovers",
                "PPIQuantileCIWidth",
                "PPIQuantileCICovers",
                "train_s",
                "eval_s",
            ],
        )
        _assert_non_negative(row["MSE"])
        _assert_non_negative(row["MAE"])
        _assert_non_negative(row["TailRMSE90"])
        _assert_finite_numeric(row["NLL"])
        _assert_non_negative(row["CRPS"])
        _assert_probability(row["Cov90"])
        _assert_non_negative(row["Width90"])
        _assert_non_negative(row["AURC"])
        _assert_non_negative(row["PPIMeanCIWidth"])
        _assert_probability(row["PPIMeanCICovers"])
        _assert_non_negative(row["PPIQuantileCIWidth"])
        _assert_probability(row["PPIQuantileCICovers"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_spt_reg_realdata_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("spt_reg_realdata_comparison")
    out = tmp_path / "spt_reg_realdata_summary.json"
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
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="real-data",
        required_methods={
            "SourceGaussian",
            "WeightedSplitConformalGaussian",
            "SPTRegGaussian",
            "TargetRefitSmallGaussian",
            "SourceBinnedPDF",
            "SPTRegBinnedPDF",
        },
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in (
        "SourceGaussian",
        "WeightedSplitConformalGaussian",
        "SPTRegGaussian",
        "TargetRefitSmallGaussian",
        "SourceBinnedPDF",
        "SPTRegBinnedPDF",
    ):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "Family",
                "MSE",
                "MAE",
                "TailRMSE90",
                "NLL",
                "CRPS",
                "Cov90",
                "Width90",
                "AURC",
                "PPIMeanCIWidth",
                "PPIMeanCICovers",
                "PPIQuantileCIWidth",
                "PPIQuantileCICovers",
                "train_s",
                "eval_s",
            ],
        )
        _assert_non_negative(row["MSE"])
        _assert_non_negative(row["MAE"])
        _assert_non_negative(row["TailRMSE90"])
        _assert_finite_numeric(row["NLL"])
        _assert_non_negative(row["CRPS"])
        _assert_probability(row["Cov90"])
        _assert_non_negative(row["Width90"])
        _assert_non_negative(row["AURC"])
        _assert_non_negative(row["PPIMeanCIWidth"])
        _assert_probability(row["PPIMeanCICovers"])
        _assert_non_negative(row["PPIQuantileCIWidth"])
        _assert_probability(row["PPIQuantileCICovers"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])


def test_semi_supervised_regression_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("semi_supervised_regression_comparison")
    out = tmp_path / "semi_supervised_summary.json"
    cfg = mod.SemiSupervisedRegressionConfig(
        n_labeled=32,
        n_unlabeled=64,
        n_test=32,
        hidden=8,
        teacher_epochs=2,
        student_epochs=3,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="semi-supervised",
        required_methods={"SupervisedMSE", "PseudoLabelConsistency", "PseudoLabelNLL"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("SupervisedMSE", "PseudoLabelConsistency", "PseudoLabelNLL"):
        row = rows[method]
        _assert_row_has_keys(
            row,
            ["MSE", "MAE", "R2", "PseudoAcceptRate", "PseudoMeanConf", "train_s", "eval_s"],
        )
        _assert_non_negative(row["MSE"])
        _assert_non_negative(row["MAE"])
        _assert_probability(row["PseudoAcceptRate"])
        _assert_probability(row["PseudoMeanConf"])
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


def test_spt_reg_year_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("spt_reg_year_comparison")
    out = tmp_path / "spt_reg_year_summary.json"
    data_path = tmp_path / "year_like.csv"
    _write_tiny_tabular_csv(data_path, n_rows=320, n_features=10, seed=41)
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
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="yearpredictionmsd-style",
        required_methods={
            "SourceGaussian",
            "WeightedSplitConformalGaussian",
            "SPTRegGaussian",
            "TargetRefitSmallGaussian",
            "SourceBinnedPDF",
            "SPTRegBinnedPDF",
        },
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in (
        "SourceGaussian",
        "WeightedSplitConformalGaussian",
        "SPTRegGaussian",
        "TargetRefitSmallGaussian",
        "SourceBinnedPDF",
        "SPTRegBinnedPDF",
    ):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "Family",
                "MSE",
                "MAE",
                "TailRMSE90",
                "NLL",
                "CRPS",
                "Cov90",
                "Width90",
                "AURC",
                "PPIMeanCIWidth",
                "PPIMeanCICovers",
                "PPIQuantileCIWidth",
                "PPIQuantileCICovers",
                "train_s",
                "eval_s",
            ],
        )
        _assert_non_negative(row["MSE"])
        _assert_non_negative(row["MAE"])
        _assert_non_negative(row["TailRMSE90"])
        _assert_finite_numeric(row["NLL"])
        _assert_non_negative(row["CRPS"])
        _assert_probability(row["Cov90"])
        _assert_non_negative(row["Width90"])
        _assert_non_negative(row["AURC"])
        _assert_non_negative(row["PPIMeanCIWidth"])
        _assert_probability(row["PPIMeanCICovers"])
        _assert_non_negative(row["PPIQuantileCIWidth"])
        _assert_probability(row["PPIQuantileCICovers"])
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


def test_contrastive_flow_parameter_estimation_comparison_writes_summary_json(
    tmp_path: Path,
) -> None:
    mod = _load_example_module("contrastive_flow_parameter_estimation_comparison")
    out = tmp_path / "contrastive_flow_synth_summary.json"
    cfg = mod.ContrastiveFlowComparisonConfig(
        n_train=48,
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
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="parameter estimation",
        required_methods={"GaussianSummary", "NormalizingFlow", "ContrastiveFlow"},
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("GaussianSummary", "NormalizingFlow", "ContrastiveFlow"):
        row = rows[method]
        _assert_row_has_keys(row, ["train_s", "eval_s", "Notes"])
        if row["train_s"] is not None:
            _assert_non_negative(row["train_s"])
        if row["eval_s"] is not None:
            _assert_non_negative(row["eval_s"])
    _assert_non_negative(rows["GaussianSummary"]["ParamMAE"])
    _assert_non_negative(rows["GaussianSummary"]["Dim0_MAE"])
    _assert_non_negative(rows["GaussianSummary"]["Dim1_MAE"])


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


def test_ordinal_regression_realdata_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("ordinal_regression_realdata_comparison")
    out = tmp_path / "ordinal_realdata_comparison_summary.json"
    cfg = mod.OrdinalRealDataConfig(
        n_train=128,
        n_test=64,
        hidden=12,
        epochs=2,
        batch_size=24,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="real-data",
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


def test_ordinal_uncertain_ground_truth_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("ordinal_uncertain_ground_truth_comparison")
    out = tmp_path / "ordinal_uncertain_ground_truth_comparison_summary.json"
    cfg = mod.OrdinalUGTComparisonConfig(
        n_train=96,
        n_test=48,
        hidden=12,
        epochs=2,
        teacher_epochs=2,
        batch_size=24,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="ordinal",
        required_methods={
            "HardOrdinalCE",
            "SoftOrdinalCE",
            "SoftOrdinalCE+Pseudo",
            "SoftCumulativeLink",
        },
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("HardOrdinalCE", "SoftOrdinalCE", "SoftOrdinalCE+Pseudo", "SoftCumulativeLink"):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "Accuracy",
                "OrdinalMAE",
                "QWK",
                "TrueNLL",
                "PlausibilityCE",
                "train_s",
                "eval_s",
                "Notes",
            ],
        )
        _assert_probability(row["Accuracy"])
        _assert_non_negative(row["OrdinalMAE"])
        _assert_finite_numeric(row["QWK"])
        _assert_non_negative(row["TrueNLL"])
        _assert_non_negative(row["PlausibilityCE"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])

    _assert_probability(rows["SoftOrdinalCE+Pseudo"]["PseudoAcceptRate"])


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


def test_censored_regression_realdata_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("censored_regression_realdata_comparison")
    out = tmp_path / "censored_realdata_comparison_summary.json"
    cfg = mod.CensoredRealDataConfig(
        n_train=128,
        n_test=64,
        hidden=12,
        epochs=2,
        batch_size=24,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="real-data",
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
        required_methods={
            "MSE",
            "DensityWeighted",
            "PropensityWeighted",
            "GaussianNLL",
            "Quantile90",
        },
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in ("MSE", "DensityWeighted", "PropensityWeighted", "GaussianNLL", "Quantile90"):
        row = rows[method]
        _assert_row_has_keys(
            row,
            [
                "MAE",
                "TailMAE90",
                "TailRMSE90",
                "NativeCov90",
                "NativeWidth90",
                "TailCov90",
                "ObservedRate",
                "train_s",
                "eval_s",
                "Notes",
            ],
        )
        _assert_non_negative(row["MAE"])
        _assert_non_negative(row["TailMAE90"])
        _assert_non_negative(row["TailRMSE90"])
        _assert_probability(row["ObservedRate"])
        _assert_non_negative(row["train_s"])
        _assert_non_negative(row["eval_s"])
        if method in {"GaussianNLL", "Quantile90"}:
            _assert_probability(row["NativeCov90"])
            _assert_non_negative(row["NativeWidth90"])
            _assert_probability(row["TailCov90"])


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


def test_uncertain_gt_density_conformal_realdata_comparison_writes_summary_json(
    tmp_path: Path,
) -> None:
    mod = _load_example_module("uncertain_gt_density_conformal_realdata_comparison")
    out = tmp_path / "uncertain_gt_density_conformal_realdata_summary.json"
    cfg = mod.UncertainGTConformalRealDataConfig(
        n_train=128,
        n_cal=64,
        n_test=64,
        hidden=12,
        epochs=2,
        n_mc_samples=8,
    )
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="real-data",
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
            "SelectionBias-NaiveDiff",
            "SelectionBias-DRATE",
            "SelectionBias-DRCATE",
        },
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in (
        "Uplift-NaiveDiff",
        "Uplift-DRATE",
        "Uplift-DRCATE",
        "SelectionBias-NaiveDiff",
        "SelectionBias-DRATE",
        "SelectionBias-DRCATE",
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


def test_causal_dr_realdata_comparison_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_example_module("causal_dr_realdata_comparison")
    out = tmp_path / "causal_dr_realdata_summary.json"
    cfg = mod.CausalDRRealDataConfig(n_samples=320, folds=2)
    mod.main(cfg, summary_json_path=str(out))
    _assert_summary_schema(
        out,
        task_substring="real covariates",
        required_methods={
            "DiabetesProxy-NaiveDiff",
            "DiabetesProxy-DRATE",
            "DiabetesProxy-DRCATE",
            "DiabetesSelectionBias-NaiveDiff",
            "DiabetesSelectionBias-DRATE",
            "DiabetesSelectionBias-DRCATE",
        },
    )
    payload = _load_payload(out)
    rows = _rows_by_method(payload)
    for method in (
        "DiabetesProxy-NaiveDiff",
        "DiabetesProxy-DRATE",
        "DiabetesProxy-DRCATE",
        "DiabetesSelectionBias-NaiveDiff",
        "DiabetesSelectionBias-DRATE",
        "DiabetesSelectionBias-DRCATE",
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
