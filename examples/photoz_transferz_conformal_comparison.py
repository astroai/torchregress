"""Dedicated real-data conformal photo-z benchmark using TransferZ split semantics."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import photoz_benchmark_comparison as pzbase
import torch
import torch.nn as nn
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from photoz_binned_utils import (
    apply_temperature,
    fit_temperature_scaler,
    make_bins_from_train_targets,
)
from torch.utils.data import DataLoader, TensorDataset

from torchregress.calibration import VarianceTemperatureScaler
from torchregress.losses import (
    CQR,
    DensityConformal,
    GaussianNLLLoss,
    MonteCarloConformal,
    MultiQuantileLoss,
    PrevalenceAdjustedCP,
    R2CConformal,
    SplitConformal,
    WeightedCrossEntropyLoss,
    WeightedHuberLoss,
)


@dataclass(frozen=True)
class PhotoZTransferZConformalConfig:
    seed: int = 260308
    n_train: int = 1024
    n_cal: int = 512
    n_conformal: int = 512
    n_test: int = 512
    batch_size: int = 64
    epochs: int = 10
    lr: float = 2e-3
    hidden: int = 64
    alpha: float = 0.1
    n_mc_samples: int = 32
    n_bins: int = 48
    dataset_path: str | None = None
    train_dataset_path: str | None = None
    cal_dataset_path: str | None = None
    conformal_dataset_path: str | None = None
    test_dataset_path: str | None = None
    force_simulated: bool = False
    require_real_data: bool = False
    sample_size_if_generate: int = 4096
    temperature_max_iter: int = 200
    temperature_lr: float = 0.05
    variance_temperature_max_iter: int = 200
    variance_temperature_lr: float = 0.05
    density_bandwidth_grid: tuple[float, ...] = (0.15, 0.25, 0.35, 0.5, 0.75)
    prevalence_n_bins_grid: tuple[int, ...] = (4, 5, 6, 8)
    prevalence_min_group_grid: tuple[int, ...] = (8, 16, 32)
    tuning_highz_quantile: float = 0.8
    tuning_coverage_floor: float = 0.88


def _default_transferz_paths() -> dict[str, Path]:
    base = Path("data/transferz/normalized")
    return {
        "train": base / "transferz_train_photoz.csv",
        "cal": base / "transferz_cal_photoz.csv",
        "conformal": base / "transferz_conformal_photoz.csv",
        "test": base / "transferz_test_photoz.csv",
    }


def _resolve_split_paths(cfg: PhotoZTransferZConformalConfig) -> dict[str, Path] | None:
    explicit = {
        "train": cfg.train_dataset_path,
        "cal": cfg.cal_dataset_path,
        "conformal": cfg.conformal_dataset_path,
        "test": cfg.test_dataset_path,
    }
    provided = {name: value for name, value in explicit.items() if value is not None}
    if provided and len(provided) != 4:
        raise ValueError(
            "Conformal photo-z example requires all of train/cal/conformal/test split paths."
        )
    if len(provided) == 4:
        return {name: Path(path or "") for name, path in explicit.items()}

    default_paths = _default_transferz_paths()
    if not cfg.force_simulated and all(path.exists() for path in default_paths.values()):
        return default_paths
    if cfg.require_real_data:
        raise FileNotFoundError(
            "Real TransferZ split files were requested but not found. Expected:\n"
            + "\n".join(f"- {name}: {path}" for name, path in default_paths.items())
        )
    return None


def _load_single_or_simulated(cfg: PhotoZTransferZConformalConfig) -> torch.Tensor | None:
    if cfg.dataset_path is None:
        return None
    return pzbase._load_photoz_table(Path(cfg.dataset_path))  # type: ignore[return-value]


def _simulate_dataframe(cfg: PhotoZTransferZConformalConfig):
    out_path = Path("data/sdss/sdss_photoz_simulated.csv")
    return pzbase._create_simulated_sdss_data(
        n_galaxies=max(
            cfg.sample_size_if_generate,
            cfg.n_train + cfg.n_cal + cfg.n_conformal + cfg.n_test,
        ),
        out_path=out_path,
        seed=cfg.seed,
    )


def _data_source_name(cfg: PhotoZTransferZConformalConfig) -> str:
    resolved = _resolve_split_paths(cfg)
    if resolved is not None:
        train_stem = resolved["train"].stem
        if train_stem.startswith("transferz_"):
            return "external_splits:transferz"
        return f"external_splits:{train_stem}"
    if cfg.dataset_path is not None:
        return f"external:{Path(cfg.dataset_path).stem}"
    return "simulated_photoz_conformal"


def _make_splits(cfg: PhotoZTransferZConformalConfig) -> dict[str, torch.Tensor]:
    resolved = _resolve_split_paths(cfg)
    target_col = "spec_z"
    target_err_col = "spec_z_err"
    if resolved is not None:
        frames = {name: pzbase._load_photoz_table(path) for name, path in resolved.items()}
        feature_cols, error_cols = pzbase._infer_feature_columns(frames["train"])
        for name in ("cal", "conformal", "test"):
            cols, errs = pzbase._infer_feature_columns(frames[name])
            if cols != feature_cols or errs != error_cols:
                raise ValueError(
                    f"Feature columns for split {name} do not match train split. "
                    f"train={feature_cols}, {name}={cols}"
                )
        limits = {
            "train": cfg.n_train,
            "cal": cfg.n_cal,
            "conformal": cfg.n_conformal,
            "test": cfg.n_test,
        }
        rng = np_random(cfg.seed)
        split_frames = {}
        for offset, name in enumerate(("train", "cal", "conformal", "test")):
            frame = frames[name]
            limit = limits[name]
            if len(frame) < limit:
                raise ValueError(
                    f"Requested {limit} rows for {name}, but split only has {len(frame)}."
                )
            idx = rng.permutation(len(frame))[:limit]
            split_frames[name] = frame.iloc[idx].reset_index(drop=True)
    else:
        df = _simulate_dataframe(cfg) if cfg.dataset_path is None else _load_single_or_simulated(cfg)
        assert df is not None
        feature_cols, error_cols = pzbase._infer_feature_columns(df)
        need = cfg.n_train + cfg.n_cal + cfg.n_conformal + cfg.n_test
        if len(df) < need:
            raise ValueError(f"Need {need} rows but dataset only has {len(df)}.")
        rng = np_random(cfg.seed)
        idx = rng.permutation(len(df))[:need]
        df = df.iloc[idx].reset_index(drop=True)
        split_frames = {
            "train": df.iloc[: cfg.n_train].reset_index(drop=True),
            "cal": df.iloc[cfg.n_train : cfg.n_train + cfg.n_cal].reset_index(drop=True),
            "conformal": df.iloc[
                cfg.n_train + cfg.n_cal : cfg.n_train + cfg.n_cal + cfg.n_conformal
            ].reset_index(drop=True),
            "test": df.iloc[cfg.n_train + cfg.n_cal + cfg.n_conformal :].reset_index(drop=True),
        }

    def _frame_to_tensors(frame):
        x = torch.tensor(frame[feature_cols].to_numpy(dtype="float32"))
        x_err = torch.tensor(frame[error_cols].to_numpy(dtype="float32"))
        y = torch.tensor(frame[target_col].to_numpy(dtype="float32")).unsqueeze(1)
        y_err = torch.tensor(frame[target_err_col].to_numpy(dtype="float32")).unsqueeze(1)
        return x, x_err, y, y_err

    x_train, xerr_train, y_train, yerr_train = _frame_to_tensors(split_frames["train"])
    x_cal, xerr_cal, y_cal, yerr_cal = _frame_to_tensors(split_frames["cal"])
    x_conf, xerr_conf, y_conf, yerr_conf = _frame_to_tensors(split_frames["conformal"])
    x_test, xerr_test, y_test, yerr_test = _frame_to_tensors(split_frames["test"])

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)

    return {
        "x_train": (x_train - x_mean) / x_std,
        "x_cal": (x_cal - x_mean) / x_std,
        "x_conf": (x_conf - x_mean) / x_std,
        "x_test": (x_test - x_mean) / x_std,
        "xerr_train": xerr_train / x_std,
        "xerr_cal": xerr_cal / x_std,
        "xerr_conf": xerr_conf / x_std,
        "xerr_test": xerr_test / x_std,
        "y_train": (y_train - y_mean) / y_std,
        "y_cal": (y_cal - y_mean) / y_std,
        "y_conf": (y_conf - y_mean) / y_std,
        "y_test": (y_test - y_mean) / y_std,
        "y_train_raw": y_train,
        "y_cal_raw": y_cal,
        "y_conf_raw": y_conf,
        "y_test_raw": y_test,
        "yerr_train": yerr_train / y_std,
        "yerr_cal": yerr_cal / y_std,
        "yerr_conf": yerr_conf / y_std,
        "yerr_test": yerr_test / y_std,
        "y_scale": y_std,
        "y_shift": y_mean,
        "data_source": _data_source_name(cfg),
    }


def np_random(seed: int):
    import numpy as np

    return np.random.default_rng(seed)


def _train_supervised(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs: int,
    lr: float,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def _train_supervised_tuple(
    model: nn.Module,
    loss_fn: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs: int,
    lr: float,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            mean, logvar = out[:, :1], out[:, 1:2]
            loss = loss_fn((mean, logvar), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def _train_binned(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    bin_edges: torch.Tensor,
    epochs: int,
    lr: float,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = WeightedCrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            y_bin = torch.bucketize(yb.view(-1), bin_edges[1:-1])
            loss = ce(logits, y_bin)
            loss.backward()
            opt.step()
    model.eval()
    return model


def _to_raw_y(y_scaled: torch.Tensor, splits: dict[str, torch.Tensor]) -> torch.Tensor:
    return pzbase._to_raw_y(y_scaled, splits)


def _interval_metrics(
    *,
    name: str,
    lower: torch.Tensor,
    upper: torch.Tensor,
    center: torch.Tensor,
    splits: dict[str, torch.Tensor],
    train_s: float,
    eval_s: float,
    notes: str,
) -> dict[str, object]:
    lower_raw = _to_raw_y(lower, splits)
    upper_raw = _to_raw_y(upper, splits)
    y_test_raw = splits["y_test_raw"]

    cov90, width90 = pzbase._coverage_width(lower_raw, upper_raw, y_test_raw)
    under = (lower_raw - y_test_raw).clamp_min(0.0)
    over = (y_test_raw - upper_raw).clamp_min(0.0)
    interval_score = torch.mean((upper_raw - lower_raw) + (2.0 / 0.1) * (under + over)).item()

    q80 = torch.quantile(y_test_raw[:, 0], 0.80)
    high_mask = y_test_raw[:, 0] >= q80
    if bool(high_mask.any().item()):
        high_cov, high_width = pzbase._coverage_width(
            lower_raw[high_mask],
            upper_raw[high_mask],
            y_test_raw[high_mask],
        )
    else:
        high_cov, high_width = float("nan"), float("nan")

    point = pzbase._point_metrics(center, splits["y_test"])
    pz = pzbase._photoz_metrics(center, splits["y_test"], splits)
    return {
        "Method": name,
        **point,
        **pz,
        "Coverage90": cov90,
        "Width90": width90,
        "HighZCoverage90": float(high_cov),
        "HighZWidth90": float(high_width),
        "IntervalScore90": float(interval_score),
        "train_s": float(train_s),
        "eval_s": float(eval_s),
        "DataSource": splits["data_source"],
        "Notes": notes,
    }


def _interval_summary_raw(
    *,
    lower_raw: torch.Tensor,
    upper_raw: torch.Tensor,
    y_raw: torch.Tensor,
    alpha: float,
    highz_quantile: float,
) -> dict[str, float]:
    cov, width = pzbase._coverage_width(lower_raw, upper_raw, y_raw)
    under = (lower_raw - y_raw).clamp_min(0.0)
    over = (y_raw - upper_raw).clamp_min(0.0)
    interval_score = torch.mean((upper_raw - lower_raw) + (2.0 / alpha) * (under + over)).item()
    q_hi = torch.quantile(y_raw[:, 0], highz_quantile)
    high_mask = y_raw[:, 0] >= q_hi
    if bool(high_mask.any().item()):
        high_cov, high_width = pzbase._coverage_width(
            lower_raw[high_mask],
            upper_raw[high_mask],
            y_raw[high_mask],
        )
        high_under = (lower_raw[high_mask] - y_raw[high_mask]).clamp_min(0.0)
        high_over = (y_raw[high_mask] - upper_raw[high_mask]).clamp_min(0.0)
        high_is = torch.mean(
            (upper_raw[high_mask] - lower_raw[high_mask]) + (2.0 / alpha) * (high_under + high_over)
        ).item()
    else:
        high_cov, high_width, high_is = float("nan"), float("nan"), float("nan")
    return {
        "Coverage90": float(cov),
        "Width90": float(width),
        "IntervalScore90": float(interval_score),
        "HighZCoverage90": float(high_cov),
        "HighZWidth90": float(high_width),
        "HighZIntervalScore90": float(high_is),
    }


def _selection_objective(
    summary: dict[str, float],
    *,
    coverage_floor: float,
) -> float:
    coverage = summary["Coverage90"]
    high_cov = summary["HighZCoverage90"]
    interval_score = summary["IntervalScore90"]
    high_interval_score = summary["HighZIntervalScore90"]
    width = summary["Width90"]
    high_width = summary["HighZWidth90"]

    penalty = 0.0
    penalty += 30.0 * max(coverage_floor - coverage, 0.0)
    if high_cov == high_cov:
        penalty += 45.0 * max(coverage_floor - high_cov, 0.0)
    if high_interval_score == high_interval_score:
        return float(high_interval_score + 0.35 * interval_score + 0.05 * width + 0.05 * high_width + penalty)
    return float(interval_score + 0.1 * width + penalty)


def _summarize_candidate(
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    y_eval: torch.Tensor,
    splits: dict[str, torch.Tensor],
    alpha: float,
    highz_quantile: float,
) -> dict[str, float]:
    lower_raw = _to_raw_y(lower, splits)
    upper_raw = _to_raw_y(upper, splits)
    y_raw = _to_raw_y(y_eval, splits)
    return _interval_summary_raw(
        lower_raw=lower_raw,
        upper_raw=upper_raw,
        y_raw=y_raw,
        alpha=alpha,
        highz_quantile=highz_quantile,
    )


def _tune_density_conformal(
    *,
    pred_conf: torch.Tensor,
    y_conf: torch.Tensor,
    pred_eval: torch.Tensor,
    y_eval: torch.Tensor,
    splits: dict[str, torch.Tensor],
    cfg: PhotoZTransferZConformalConfig,
) -> tuple[DensityConformal, float, dict[str, float]]:
    best_model: DensityConformal | None = None
    best_score = float("inf")
    best_meta: dict[str, float] = {}
    for bandwidth in cfg.density_bandwidth_grid:
        candidate = DensityConformal(alpha=cfg.alpha, bandwidth=bandwidth)
        candidate.calibrate(pred_conf, y_conf)
        lower, upper = candidate.predict_interval(pred_eval)
        summary = _summarize_candidate(
            lower=lower,
            upper=upper,
            y_eval=y_eval,
            splits=splits,
            alpha=cfg.alpha,
            highz_quantile=cfg.tuning_highz_quantile,
        )
        score = _selection_objective(summary, coverage_floor=cfg.tuning_coverage_floor)
        if score < best_score:
            best_model = candidate
            best_score = score
            best_meta = {"bandwidth": float(bandwidth), **summary, "selection_score": float(score)}
    assert best_model is not None
    return best_model, best_score, best_meta


def _tune_prevalence_adjusted(
    *,
    pred_conf: torch.Tensor,
    y_conf: torch.Tensor,
    pred_eval: torch.Tensor,
    y_eval: torch.Tensor,
    splits: dict[str, torch.Tensor],
    cfg: PhotoZTransferZConformalConfig,
) -> tuple[PrevalenceAdjustedCP, float, dict[str, float]]:
    best_model: PrevalenceAdjustedCP | None = None
    best_score = float("inf")
    best_meta: dict[str, float] = {}
    for n_bins in cfg.prevalence_n_bins_grid:
        for min_group_size in cfg.prevalence_min_group_grid:
            candidate = PrevalenceAdjustedCP(
                alpha=cfg.alpha,
                n_bins=n_bins,
                min_group_size=min_group_size,
            )
            candidate.calibrate(pred_conf, y_conf)
            lower, upper = candidate.predict_interval(pred_eval)
            summary = _summarize_candidate(
                lower=lower,
                upper=upper,
                y_eval=y_eval,
                splits=splits,
                alpha=cfg.alpha,
                highz_quantile=cfg.tuning_highz_quantile,
            )
            score = _selection_objective(summary, coverage_floor=cfg.tuning_coverage_floor)
            if score < best_score:
                best_model = candidate
                best_score = score
                best_meta = {
                    "n_bins": float(n_bins),
                    "min_group_size": float(min_group_size),
                    **summary,
                    "selection_score": float(score),
                }
    assert best_model is not None
    return best_model, best_score, best_meta


def _mc_samples(
    mean: torch.Tensor,
    var: torch.Tensor,
    *,
    n_samples: int,
    seed: int,
) -> torch.Tensor:
    std = var.clamp_min(1e-8).sqrt()
    g = torch.Generator(device=mean.device).manual_seed(seed)
    noise = torch.randn((n_samples, *mean.shape), generator=g, device=mean.device, dtype=mean.dtype)
    return mean.unsqueeze(0) + std.unsqueeze(0) * noise


def run_comparison(
    cfg: PhotoZTransferZConformalConfig,
) -> tuple[list[dict[str, object]], list[str]]:
    splits = _make_splits(cfg)
    set_comparison_seed(cfg.seed)

    train_loader = DataLoader(
        TensorDataset(splits["x_train"], splits["y_train"]),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    d_in = int(splits["x_train"].shape[1])
    rows: list[dict[str, object]] = []

    # Shared predictors.
    huber_model = pzbase.PhotoZRegressor(d_in, hidden=cfg.hidden)
    _, huber_train_s = timed_call(
        _train_supervised,
        huber_model,
        WeightedHuberLoss(delta=1.0),
        train_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )
    with torch.no_grad():
        pred_cal = huber_model(splits["x_cal"])
        pred_conf = huber_model(splits["x_conf"])
        pred_test = huber_model(splits["x_test"])

    quant_model = pzbase.PhotoZRegressor(d_in, out_dim=3, hidden=cfg.hidden)
    _, quant_train_s = timed_call(
        _train_supervised,
        quant_model,
        MultiQuantileLoss(quantiles=[0.05, 0.5, 0.95], joint_prediction=True),
        train_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )
    with torch.no_grad():
        quant_conf = quant_model(splits["x_conf"])
        quant_test = quant_model(splits["x_test"])

    gauss_model = pzbase.PhotoZRegressor(d_in, out_dim=2, hidden=cfg.hidden)
    _, gauss_train_s = timed_call(
        _train_supervised_tuple,
        gauss_model,
        GaussianNLLLoss(),
        train_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )
    with torch.no_grad():
        gauss_cal = gauss_model(splits["x_cal"])
        gauss_conf = gauss_model(splits["x_conf"])
        gauss_test = gauss_model(splits["x_test"])
    mean_cal, logvar_cal = gauss_cal[:, :1], gauss_cal[:, 1:2].clamp(-8.0, 6.0)
    mean_conf, logvar_conf = gauss_conf[:, :1], gauss_conf[:, 1:2].clamp(-8.0, 6.0)
    mean_test, logvar_test = gauss_test[:, :1], gauss_test[:, 1:2].clamp(-8.0, 6.0)
    var_scaler, varcal_s = timed_call(
        VarianceTemperatureScaler().fit,
        mean_cal,
        torch.exp(logvar_cal),
        splits["y_cal"],
        max_iter=cfg.variance_temperature_max_iter,
        lr=cfg.variance_temperature_lr,
    )
    var_conf = var_scaler.transform(torch.exp(logvar_conf))
    var_test = var_scaler.transform(torch.exp(logvar_test))

    bin_edges = make_bins_from_train_targets(splits["y_train"], n_bins=cfg.n_bins, strategy="quantile")
    binned_model = pzbase.PhotoZRegressor(d_in, out_dim=cfg.n_bins, hidden=cfg.hidden)
    _, binned_train_s = timed_call(
        _train_binned,
        binned_model,
        train_loader,
        bin_edges=bin_edges,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )
    with torch.no_grad():
        logits_cal = binned_model(splits["x_cal"])
        logits_conf = binned_model(splits["x_conf"])
        logits_test = binned_model(splits["x_test"])
    temp, temps_s = timed_call(
        fit_temperature_scaler,
        logits_cal,
        splits["y_cal"],
        bin_edges,
        max_iter=cfg.temperature_max_iter,
        lr=cfg.temperature_lr,
    )
    probs_conf = torch.softmax(apply_temperature(logits_conf, float(temp)), dim=-1)
    probs_test = torch.softmax(apply_temperature(logits_test, float(temp)), dim=-1)

    # Native baselines.
    q05 = torch.minimum(quant_test[:, 0:1], quant_test[:, 2:3])
    q95 = torch.maximum(quant_test[:, 0:1], quant_test[:, 2:3])
    q50 = quant_test[:, 1:2]
    rows.append(
        _interval_metrics(
            name="NativeQuantile90",
            lower=q05,
            upper=q95,
            center=q50,
            splits=splits,
            train_s=quant_train_s,
            eval_s=0.0,
            notes="native quantile intervals before conformalization",
        )
    )
    native_std = var_test.sqrt()
    rows.append(
        _interval_metrics(
            name="NativeGaussian90",
            lower=mean_test - 1.645 * native_std,
            upper=mean_test + 1.645 * native_std,
            center=mean_test,
            splits=splits,
            train_s=gauss_train_s + varcal_s,
            eval_s=0.0,
            notes="Gaussian intervals after variance-temperature scaling on validation split",
        )
    )

    # Conformal methods.
    split_cp = SplitConformal(alpha=cfg.alpha)
    _, split_cal_s = timed_call(split_cp.calibrate, pred_conf, splits["y_conf"])
    (split_l, split_u), split_eval_s = timed_call(split_cp.predict_interval, pred_test)
    rows.append(
        _interval_metrics(
            name="SplitConformal",
            lower=split_l,
            upper=split_u,
            center=pred_test,
            splits=splits,
            train_s=huber_train_s + split_cal_s,
            eval_s=split_eval_s,
            notes="absolute-residual split conformal on Huber predictor",
        )
    )

    cqr = CQR(alpha=cfg.alpha, debias=True)
    _, cqr_cal_s = timed_call(
        cqr.calibrate,
        torch.cat([quant_conf[:, 0:1], quant_conf[:, 2:3]], dim=1),
        splits["y_conf"],
    )
    (cqr_l, cqr_u), cqr_eval_s = timed_call(
        cqr.predict_interval,
        torch.cat([quant_test[:, 0:1], quant_test[:, 2:3]], dim=1),
    )
    rows.append(
        _interval_metrics(
            name="CQR",
            lower=cqr_l,
            upper=cqr_u,
            center=q50,
            splits=splits,
            train_s=quant_train_s + cqr_cal_s,
            eval_s=cqr_eval_s,
            notes="conformalized quantile regression using reserved conformal split",
        )
    )

    density_cp, _, density_meta = _tune_density_conformal(
        pred_conf=pred_conf,
        y_conf=splits["y_conf"],
        pred_eval=pred_cal,
        y_eval=splits["y_cal"],
        splits=splits,
        cfg=cfg,
    )
    _, density_cal_s = timed_call(density_cp.calibrate, pred_conf, splits["y_conf"])
    (density_l, density_u), density_eval_s = timed_call(density_cp.predict_interval, pred_test)
    rows.append(
        _interval_metrics(
            name="DensityConformal",
            lower=density_l,
            upper=density_u,
            center=pred_test,
            splits=splits,
            train_s=huber_train_s + density_cal_s,
            eval_s=density_eval_s,
            notes=(
                "density-aware split conformal tuned on validation high-z interval score; "
                f"bandwidth={density_meta['bandwidth']:.2f}"
            ),
        )
    )

    prev_cp, _, prev_meta = _tune_prevalence_adjusted(
        pred_conf=pred_conf,
        y_conf=splits["y_conf"],
        pred_eval=pred_cal,
        y_eval=splits["y_cal"],
        splits=splits,
        cfg=cfg,
    )
    _, prev_cal_s = timed_call(prev_cp.calibrate, pred_conf, splits["y_conf"])
    (prev_l, prev_u), prev_eval_s = timed_call(prev_cp.predict_interval, pred_test)
    rows.append(
        _interval_metrics(
            name="PrevalenceAdjustedCP",
            lower=prev_l,
            upper=prev_u,
            center=pred_test,
            splits=splits,
            train_s=huber_train_s + prev_cal_s,
            eval_s=prev_eval_s,
            notes=(
                "group-prevalence-adjusted conformal tuned on validation high-z interval score; "
                f"n_bins={prev_meta['n_bins']:.0f}, min_group={prev_meta['min_group_size']:.0f}"
            ),
        )
    )

    mc_cp = MonteCarloConformal(alpha=cfg.alpha)
    mc_conf = _mc_samples(mean_conf, var_conf, n_samples=cfg.n_mc_samples, seed=cfg.seed + 31)
    mc_test = _mc_samples(mean_test, var_test, n_samples=cfg.n_mc_samples, seed=cfg.seed + 32)
    _, mc_cal_s = timed_call(mc_cp.calibrate, mc_conf, splits["y_conf"])
    (mc_l, mc_u), mc_eval_s = timed_call(mc_cp.predict_interval, mc_test)
    rows.append(
        _interval_metrics(
            name="MonteCarloConformal",
            lower=mc_l,
            upper=mc_u,
            center=mc_test.mean(dim=0),
            splits=splits,
            train_s=gauss_train_s + varcal_s + mc_cal_s,
            eval_s=mc_eval_s,
            notes="MC-sample conformal with variance-calibrated Gaussian predictor",
        )
    )

    r2c = R2CConformal(alpha=cfg.alpha, bin_edges=bin_edges)
    _, r2c_cal_s = timed_call(r2c.calibrate, probs_conf, splits["y_conf"])
    (r2c_l, r2c_u), r2c_eval_s = timed_call(r2c.predict_interval, probs_test)
    rows.append(
        _interval_metrics(
            name="R2CConformal",
            lower=r2c_l,
            upper=r2c_u,
            center=0.5 * (r2c_l + r2c_u),
            splits=splits,
            train_s=binned_train_s + temps_s + r2c_cal_s,
            eval_s=r2c_eval_s,
            notes="regression-as-classification conformal with validation temperature scaling",
        )
    )

    notes = [
        "TransferZ split semantics are preserved: TRAINING fit, VALIDATION post-hoc calibration, CONFORMAL conformal calibration, TESTING evaluation.",
        "Coverage and width are reported on the raw photo-z scale; point metrics use interval centers.",
        "DensityConformal and PrevalenceAdjustedCP are tuned on the validation split for high-z interval efficiency before final conformal calibration on the reserved conformal split.",
        "MonteCarloConformal and R2CConformal test richer uncertainty structures.",
    ]
    return rows, notes


def main(
    cfg: PhotoZTransferZConformalConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or PhotoZTransferZConformalConfig()
    rows, notes = run_comparison(cfg)
    print_fairness_notes(
        title="TransferZ Conformal Photo-z Comparison",
        seed_policy="fixed seed; released TransferZ split semantics preserved when real data is used",
        train_budget=(
            f"{cfg.epochs} epochs shared across predictors; validation used only for post-hoc "
            "temperature/variance scaling; conformal split reserved for conformal calibration"
        ),
        metric_policy=(
            "Coverage/width/interval score on raw photo-z scale plus NMAD/catastrophic/high-z MAE "
            "from interval centers"
        ),
    )
    print_comparison_summary(
        "TransferZ Conformal Photo-z Summary",
        rows,
        metric_order=[
            "Coverage90",
            "Width90",
            "IntervalScore90",
            "NMAD",
            "CatastrophicRate",
            "HighZ_MAE",
            "HighZCoverage90",
            "HighZWidth90",
            "train_s",
            "eval_s",
        ],
    )
    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/photoz_transferz_conformal_comparison.py",
            task="TransferZ real-data conformal photometric redshift comparison",
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run conformal photo-z comparison on TransferZ.")
    parser.add_argument("--summary-json-path", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--train-dataset-path", type=str, default=None)
    parser.add_argument("--cal-dataset-path", type=str, default=None)
    parser.add_argument("--conformal-dataset-path", type=str, default=None)
    parser.add_argument("--test-dataset-path", type=str, default=None)
    parser.add_argument("--force-simulated", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    args = parser.parse_args()
    main(
        PhotoZTransferZConformalConfig(
            dataset_path=args.dataset_path,
            train_dataset_path=args.train_dataset_path,
            cal_dataset_path=args.cal_dataset_path,
            conformal_dataset_path=args.conformal_dataset_path,
            test_dataset_path=args.test_dataset_path,
            force_simulated=args.force_simulated,
            require_real_data=args.require_real_data,
        ),
        summary_json_path=args.summary_json_path,
    )
