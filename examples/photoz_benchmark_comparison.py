"""
Photo-z benchmark comparison with shared-budget metrics and summary artifacts.

This example focuses on a domain-realistic photometric redshift workload using
SDSS-style features (colors + measurement errors). It supports:
- local cached real SDSS data if present
- deterministic simulated fallback otherwise
- shared-budget comparison across point, robust, probabilistic, and EIV methods
- machine-readable summary JSON output for audit/review pipelines
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import (
    FunctionalEIVLoss,
    GaussianNLLLoss,
    LogTransformLoss,
    MultiQuantileLoss,
    PseudoLabelConsistencyLoss,
    PseudoLabelNLL,
    WeightedHuberLoss,
    WeightedMSELoss,
)
from torchregress.utils import generate_pseudo_labels


@dataclass(frozen=True)
class PhotoZBenchmarkConfig:
    seed: int = 240226
    n_train: int = 512
    n_cal: int = 256
    n_test: int = 256
    batch_size: int = 64
    epochs: int = 12
    lr: float = 2e-3
    hidden: int = 64
    labeled_fraction: float = 0.35
    teacher_epochs: int = 8
    pseudo_confidence_threshold: float = 0.35
    dataset_path: str | None = None
    force_simulated: bool = False
    require_real_data: bool = False
    allow_download: bool = False
    sample_size_if_generate: int = 5000


class PhotoZRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int = 1,
        hidden: int = 64,
        positive_output: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        ]
        if positive_output:
            layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_photoz_df(cfg: PhotoZBenchmarkConfig):
    data_dir = Path("data/sdss")
    real_path = data_dir / "sdss_photoz_real.csv"
    sim_path = data_dir / "sdss_photoz_simulated.csv"

    if cfg.dataset_path is not None:
        dataset_path = Path(cfg.dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Configured photo-z dataset path does not exist: {dataset_path}"
            )
        suffix = dataset_path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(dataset_path)
        if suffix in {".json", ".jsonl"}:
            return pd.read_json(dataset_path, lines=(suffix == ".jsonl"))
        if suffix in {".pkl", ".pickle"}:
            return pd.read_pickle(dataset_path)
        raise ValueError(
            f"Unsupported photo-z dataset format `{suffix}` for {dataset_path}. "
            "Use CSV, JSON, JSONL, or pickle."
        )

    if not cfg.force_simulated and real_path.exists():
        return pd.read_csv(real_path)

    if cfg.require_real_data:
        raise FileNotFoundError(
            f"Real photo-z dataset required but not found at {real_path}. "
            "Run tools.photoz_collect_real_data first or disable require_real_data."
        )

    # Check if existing simulated data meets the size requirements for the current profile.
    # This prevents CI failures where a small 'audit' file is reused by a larger 'full' profile.
    if sim_path.exists():
        df = pd.read_csv(sim_path)
        need = max(cfg.sample_size_if_generate, cfg.n_train + cfg.n_cal + cfg.n_test)
        if len(df) >= need:
            return df

    return _create_simulated_sdss_data(
        n_galaxies=max(cfg.sample_size_if_generate, cfg.n_train + cfg.n_cal + cfg.n_test),
        out_path=sim_path,
        seed=cfg.seed,
    )


def _create_simulated_sdss_data(
    *,
    n_galaxies: int,
    out_path: Path,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    z_spec = rng.lognormal(mean=-1.3, sigma=0.5, size=n_galaxies)
    z_spec = np.clip(z_spec, 0.01, 1.0)

    u_true = 20.0 + 2.0 * z_spec + rng.normal(0, 0.10, n_galaxies)
    g_true = 19.0 + 1.8 * z_spec + rng.normal(0, 0.06, n_galaxies)
    r_true = 18.5 + 1.6 * z_spec + rng.normal(0, 0.04, n_galaxies)
    i_true = 18.0 + 1.4 * z_spec + rng.normal(0, 0.03, n_galaxies)
    z_mag_true = 17.5 + 1.2 * z_spec + rng.normal(0, 0.05, n_galaxies)

    u_err = 0.02 + 0.08 * np.exp((u_true - 18) / 4)
    g_err = 0.015 + 0.05 * np.exp((g_true - 17) / 5)
    r_err = 0.01 + 0.03 * np.exp((r_true - 16) / 6)
    i_err = 0.01 + 0.03 * np.exp((i_true - 16) / 6)
    z_mag_err = 0.015 + 0.04 * np.exp((z_mag_true - 15) / 5)

    df = pd.DataFrame(
        {
            "objid": np.arange(n_galaxies),
            "spec_z": z_spec,
            "spec_z_err": 0.0005 + 0.001 * z_spec,
            "u": u_true + rng.normal(0, 1, n_galaxies) * u_err,
            "g": g_true + rng.normal(0, 1, n_galaxies) * g_err,
            "r": r_true + rng.normal(0, 1, n_galaxies) * r_err,
            "i": i_true + rng.normal(0, 1, n_galaxies) * i_err,
            "z_mag": z_mag_true + rng.normal(0, 1, n_galaxies) * z_mag_err,
            "u_err": u_err,
            "g_err": g_err,
            "r_err": r_err,
            "i_err": i_err,
            "z_mag_err": z_mag_err,
        }
    )
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z_mag"]
    df["u_g_err"] = np.sqrt(df["u_err"] ** 2 + df["g_err"] ** 2)
    df["g_r_err"] = np.sqrt(df["g_err"] ** 2 + df["r_err"] ** 2)
    df["r_i_err"] = np.sqrt(df["r_err"] ** 2 + df["i_err"] ** 2)
    df["i_z_err"] = np.sqrt(df["i_err"] ** 2 + df["z_mag_err"] ** 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def _make_splits(cfg: PhotoZBenchmarkConfig) -> dict[str, torch.Tensor]:
    df = _load_photoz_df(cfg)
    feature_cols = ["u_g", "g_r", "r_i", "i_z"]
    error_cols = ["u_g_err", "g_r_err", "r_i_err", "i_z_err"]
    target_col = "spec_z"
    target_err_col = "spec_z_err"

    need = cfg.n_train + cfg.n_cal + cfg.n_test
    if len(df) < need:
        raise ValueError(f"Need {need} rows but dataset only has {len(df)}.")

    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(df))[:need]
    df = df.iloc[idx].reset_index(drop=True)

    x = torch.tensor(df[feature_cols].to_numpy(dtype="float32"))
    x_err = torch.tensor(df[error_cols].to_numpy(dtype="float32"))
    y = torch.tensor(df[target_col].to_numpy(dtype="float32")).unsqueeze(1)
    y_err = torch.tensor(df[target_err_col].to_numpy(dtype="float32")).unsqueeze(1)

    x_train = x[: cfg.n_train]
    x_cal = x[cfg.n_train : cfg.n_train + cfg.n_cal]
    x_test = x[cfg.n_train + cfg.n_cal :]
    xerr_train = x_err[: cfg.n_train]
    xerr_cal = x_err[cfg.n_train : cfg.n_train + cfg.n_cal]
    xerr_test = x_err[cfg.n_train + cfg.n_cal :]
    y_train = y[: cfg.n_train]
    y_cal = y[cfg.n_train : cfg.n_train + cfg.n_cal]
    y_test = y[cfg.n_train + cfg.n_cal :]
    yerr_train = y_err[: cfg.n_train]
    yerr_cal = y_err[cfg.n_train : cfg.n_train + cfg.n_cal]
    yerr_test = y_err[cfg.n_train + cfg.n_cal :]

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    x_train_s = (x_train - x_mean) / x_std
    x_cal_s = (x_cal - x_mean) / x_std
    x_test_s = (x_test - x_mean) / x_std
    xerr_train_s = xerr_train / x_std
    xerr_cal_s = xerr_cal / x_std
    xerr_test_s = xerr_test / x_std

    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    y_train_s = (y_train - y_mean) / y_std
    y_cal_s = (y_cal - y_mean) / y_std
    y_test_s = (y_test - y_mean) / y_std
    yerr_train_s = yerr_train / y_std
    yerr_cal_s = yerr_cal / y_std
    yerr_test_s = yerr_test / y_std

    return {
        "x_train": x_train_s,
        "x_cal": x_cal_s,
        "x_test": x_test_s,
        "xerr_train": xerr_train_s,
        "xerr_cal": xerr_cal_s,
        "xerr_test": xerr_test_s,
        "y_train": y_train_s,
        "y_cal": y_cal_s,
        "y_test": y_test_s,
        "y_train_raw": y_train,
        "y_cal_raw": y_cal,
        "yerr_train": yerr_train_s,
        "yerr_cal": yerr_cal_s,
        "yerr_test": yerr_test_s,
        "y_test_raw": y_test,
        "y_scale": y_std,
        "y_shift": y_mean,
        "data_source": _data_source_name(cfg),
    }


def _data_source_name(cfg: PhotoZBenchmarkConfig) -> str:
    if cfg.dataset_path is not None:
        return f"external:{Path(cfg.dataset_path).stem}"
    if Path("data/sdss/sdss_photoz_real.csv").exists() and not cfg.force_simulated:
        return "real_sdss_cache"
    return "simulated_sdss"


def _to_raw_y(y_scaled: torch.Tensor, splits: dict[str, torch.Tensor]) -> torch.Tensor:
    return y_scaled * splits["y_scale"] + splits["y_shift"]


def _point_metrics(y_pred_scaled: torch.Tensor, y_true_scaled: torch.Tensor) -> dict[str, float]:
    err = y_pred_scaled - y_true_scaled
    mse = torch.mean(err**2).item()
    mae = torch.mean(torch.abs(err)).item()
    rmse = mse**0.5
    return {"MSE": float(mse), "MAE": float(mae), "RMSE": float(rmse)}


def _photoz_metrics(
    y_pred_scaled: torch.Tensor, y_true_scaled: torch.Tensor, splits: dict[str, torch.Tensor]
) -> dict[str, float]:
    y_pred = _to_raw_y(y_pred_scaled, splits)
    y_true = _to_raw_y(y_true_scaled, splits)
    residual = (y_pred - y_true) / (1.0 + y_true.clamp_min(0.0))
    med = torch.median(residual)
    nmad = 1.48 * torch.median(torch.abs(residual - med)).item()
    catastrophic = (torch.abs(y_pred - y_true) > (0.15 * (1.0 + y_true))).float().mean().item()
    q80 = torch.quantile(y_true[:, 0], 0.80)
    high_mask = y_true[:, 0] >= q80
    high_mae = (
        torch.abs(y_pred[high_mask] - y_true[high_mask]).mean().item()
        if high_mask.any()
        else float("nan")
    )
    return {
        "NMAD": float(nmad),
        "CatastrophicRate": float(catastrophic),
        "HighZ_MAE": float(high_mae),
    }


def _coverage_width(
    lower: torch.Tensor, upper: torch.Tensor, y_true: torch.Tensor
) -> tuple[float, float]:
    lower_s = torch.minimum(lower, upper)
    upper_s = torch.maximum(lower, upper)
    cov = ((y_true >= lower_s) & (y_true <= upper_s)).float().mean().item()
    width = torch.mean(upper_s - lower_s).item()
    return float(cov), float(width)


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


def _train_eiv(
    model: nn.Module,
    splits: dict[str, torch.Tensor],
    *,
    epochs: int,
    lr: float,
) -> nn.Module:
    sigma_x = splits["xerr_train"].pow(2).mean(dim=0).sqrt().clamp_min(1e-4)
    sigma_y = splits["yerr_train"].pow(2).mean(dim=0).sqrt().clamp_min(1e-4)
    loss_fn = FunctionalEIVLoss(model, sigma_x=sigma_x, sigma_y=sigma_y)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    x_obs = splits["x_train"]
    y_obs = splits["y_train"]
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(x_obs, y_obs)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _make_label_mask(cfg: PhotoZBenchmarkConfig, n_train: int) -> torch.Tensor:
    n_labeled = max(16, int(round(cfg.labeled_fraction * n_train)))
    n_labeled = min(max(n_labeled, 1), n_train - 1)
    perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(cfg.seed + 17))
    mask = torch.zeros(n_train, 1, dtype=torch.bool)
    mask[perm[:n_labeled]] = True
    return mask


def _bootstrap_teacher(
    cfg: PhotoZBenchmarkConfig,
    splits: dict[str, torch.Tensor],
    label_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labeled_x = splits["x_train"][label_mask[:, 0]]
    labeled_y = splits["y_train"][label_mask[:, 0]]
    teacher = PhotoZRegressor(splits["x_train"].shape[1], out_dim=2, hidden=cfg.hidden)
    loader = DataLoader(
        TensorDataset(labeled_x, labeled_y),
        batch_size=min(cfg.batch_size, labeled_x.shape[0]),
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed + 23),
    )
    teacher = _train_supervised_tuple(
        teacher,
        GaussianNLLLoss(),
        loader,
        epochs=max(cfg.teacher_epochs, cfg.epochs),
        lr=cfg.lr,
    )
    with torch.no_grad():
        out_all = teacher(splits["x_train"])
        mean_all = out_all[:, :1]
        logvar_all = out_all[:, 1:2].clamp(-8.0, 6.0)
    unlabeled_mask = ~label_mask
    pseudo_target = splits["y_train"].clone()
    pseudo_confidence = torch.zeros_like(pseudo_target)
    accepted = torch.zeros_like(label_mask)
    if bool(unlabeled_mask.any().item()):
        pseudo_u, conf_u, accepted_u = generate_pseudo_labels(
            mean_all[unlabeled_mask[:, 0]],
            log_variance=logvar_all[unlabeled_mask[:, 0]],
            confidence_threshold=cfg.pseudo_confidence_threshold,
        )
        if not bool(accepted_u.any().item()):
            accepted_u = torch.ones_like(accepted_u, dtype=torch.bool)
            conf_u = torch.full_like(conf_u, 0.5)
        pseudo_target[unlabeled_mask[:, 0]] = pseudo_u
        pseudo_confidence[unlabeled_mask[:, 0]] = conf_u * accepted_u.to(conf_u.dtype)
        accepted[unlabeled_mask[:, 0]] = accepted_u
    return mean_all, pseudo_target, pseudo_confidence


def _train_pseudo_label_gaussian(
    model: nn.Module,
    splits: dict[str, torch.Tensor],
    *,
    target_all: torch.Tensor,
    pseudo_target: torch.Tensor,
    pseudo_confidence: torch.Tensor,
    label_mask: torch.Tensor,
    epochs: int,
    lr: float,
) -> nn.Module:
    loss_fn = PseudoLabelNLL(pseudo_weight=0.8)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = model(splits["x_train"])
        mean, logvar = out[:, :1], out[:, 1:2].clamp(-8.0, 6.0)
        loss = loss_fn(
            (mean, logvar),
            target_all,
            pseudo_target=pseudo_target,
            pseudo_confidence=pseudo_confidence,
            label_mask=label_mask,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
    model.eval()
    return model


def _train_pseudo_label_consistency(
    model: nn.Module,
    splits: dict[str, torch.Tensor],
    *,
    target_all: torch.Tensor,
    pseudo_target: torch.Tensor,
    pseudo_confidence: torch.Tensor,
    teacher_pred: torch.Tensor,
    label_mask: torch.Tensor,
    epochs: int,
    lr: float,
) -> nn.Module:
    loss_fn = PseudoLabelConsistencyLoss(
        pseudo_weight=0.8,
        consistency_weight=0.25,
        confidence_threshold=0.1,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(splits["x_train"])
        loss = loss_fn(
            pred,
            target_all,
            pseudo_target=pseudo_target,
            pseudo_confidence=pseudo_confidence,
            teacher_pred=teacher_pred,
            label_mask=label_mask,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
    model.eval()
    return model


def _evaluate_point_method(
    name: str,
    model: nn.Module,
    splits: dict[str, torch.Tensor],
    *,
    interval_from: Optional[str] = None,
) -> dict[str, object]:
    with torch.no_grad():
        out = model(splits["x_test"])

    pred = out
    q05 = q95 = None
    nll90 = None
    if interval_from == "quantile":
        q05 = out[:, 0:1]
        pred = out[:, 1:2]
        q95 = out[:, 2:3]
        q05, q95 = torch.minimum(q05, q95), torch.maximum(q05, q95)

    point = _point_metrics(pred, splits["y_test"])
    pz = _photoz_metrics(pred, splits["y_test"], splits)
    row: dict[str, object] = {
        "Method": name,
        **point,
        **pz,
        "NLL": nll90,
        "Cov90": None,
        "Width90": None,
    }
    if q05 is not None and q95 is not None:
        cov90, width90 = _coverage_width(q05, q95, splits["y_test"])
        row["Cov90"] = cov90
        row["Width90"] = width90
    return row


def _evaluate_point_predictions(
    name: str,
    pred: torch.Tensor,
    splits: dict[str, torch.Tensor],
) -> dict[str, object]:
    point = _point_metrics(pred, splits["y_test"])
    pz = _photoz_metrics(pred, splits["y_test"], splits)
    return {
        "Method": name,
        **point,
        **pz,
        "NLL": None,
        "Cov90": None,
        "Width90": None,
    }


def _evaluate_gaussian_method(
    model: nn.Module,
    splits: dict[str, torch.Tensor],
) -> dict[str, object]:
    with torch.no_grad():
        out = model(splits["x_test"])
        mean = out[:, :1]
        logvar = out[:, 1:2].clamp(-8.0, 6.0)
        var = torch.exp(logvar)
        std = var.sqrt()

    point = _point_metrics(mean, splits["y_test"])
    pz = _photoz_metrics(mean, splits["y_test"], splits)
    resid2 = (splits["y_test"] - mean) ** 2
    nll = (0.5 * (logvar + resid2 / var)).mean().item()
    lower90 = mean - 1.645 * std
    upper90 = mean + 1.645 * std
    cov90, width90 = _coverage_width(lower90, upper90, splits["y_test"])
    return {
        "Method": "GaussianNLL",
        **point,
        **pz,
        "NLL": float(nll),
        "Cov90": cov90,
        "Width90": width90,
    }


def run_benchmark(cfg: PhotoZBenchmarkConfig) -> tuple[list[dict[str, object]], list[str]]:
    splits = _make_splits(cfg)
    train_loader = DataLoader(
        TensorDataset(splits["x_train"], splits["y_train"]),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader_raw = DataLoader(
        TensorDataset(splits["x_train"], splits["y_train_raw"]),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed + 1),
    )
    d_in = int(splits["x_train"].shape[1])
    label_mask = _make_label_mask(cfg, int(splits["x_train"].shape[0]))
    teacher_mean_all, pseudo_target_all, pseudo_confidence_all = _bootstrap_teacher(
        cfg, splits, label_mask
    )
    target_all = splits["y_train"].clone()
    target_all[~label_mask[:, 0]] = pseudo_target_all[~label_mask[:, 0]]
    pseudo_accept_rate = float((pseudo_confidence_all > 0).float().mean().item())
    accepted_conf = pseudo_confidence_all[pseudo_confidence_all > 0]
    pseudo_mean_conf = float(accepted_conf.mean().item()) if accepted_conf.numel() > 0 else 0.0

    specs: list[tuple[str, nn.Module, str]] = [
        ("MSE", PhotoZRegressor(d_in, hidden=cfg.hidden), "mse"),
        ("Huber", PhotoZRegressor(d_in, hidden=cfg.hidden), "huber"),
        ("LogTransform", PhotoZRegressor(d_in, hidden=cfg.hidden, positive_output=True), "log"),
        ("Quantile90", PhotoZRegressor(d_in, out_dim=3, hidden=cfg.hidden), "quantile"),
        ("GaussianNLL", PhotoZRegressor(d_in, out_dim=2, hidden=cfg.hidden), "gaussian"),
        ("PseudoLabelNLL", PhotoZRegressor(d_in, out_dim=2, hidden=cfg.hidden), "pseudo_gaussian"),
        (
            "PseudoLabelConsistency",
            PhotoZRegressor(d_in, hidden=cfg.hidden),
            "pseudo_consistency",
        ),
        ("FunctionalEIV", PhotoZRegressor(d_in, hidden=cfg.hidden), "eiv"),
    ]

    rows: list[dict[str, object]] = []
    notes = [
        "SDSS-style photo-z benchmark with cached-real or deterministic simulated fallback",
        "Shared train/cal/test splits and shared training budget across methods",
        "Photo-z metrics include NMAD, catastrophic outlier rate, and high-z MAE",
        "Pseudo-label rows use a partial-spec-z track: only part of the train split is treated as labeled.",
    ]
    for idx, (name, model, kind) in enumerate(specs):
        set_comparison_seed(cfg.seed + idx)
        if kind == "mse":
            _, train_s = timed_call(
                _train_supervised,
                model,
                WeightedMSELoss(),
                train_loader,
                epochs=cfg.epochs,
                lr=cfg.lr,
            )
            result, eval_s = timed_call(_evaluate_point_method, name, model, splits)
            result["Notes"] = "point baseline"
        elif kind == "huber":
            _, train_s = timed_call(
                _train_supervised,
                model,
                WeightedHuberLoss(delta=1.0),
                train_loader,
                epochs=cfg.epochs,
                lr=cfg.lr,
            )
            result, eval_s = timed_call(_evaluate_point_method, name, model, splits)
            result["Notes"] = "robust point loss"
        elif kind == "log":
            _, train_s = timed_call(
                _train_supervised,
                model,
                LogTransformLoss(),
                train_loader_raw,
                epochs=cfg.epochs,
                lr=cfg.lr,
            )
            with torch.no_grad():
                pred_raw = model(splits["x_test"])
            pred_scaled = (pred_raw - splits["y_shift"]) / splits["y_scale"]
            result, eval_s = timed_call(_evaluate_point_predictions, name, pred_scaled, splits)
            result["Notes"] = "positive-support log-transform target loss"
        elif kind == "quantile":
            _, train_s = timed_call(
                _train_supervised,
                model,
                MultiQuantileLoss(quantiles=[0.05, 0.5, 0.95], joint_prediction=True),
                train_loader,
                epochs=cfg.epochs,
                lr=cfg.lr,
            )
            result, eval_s = timed_call(
                _evaluate_point_method,
                name,
                model,
                splits,
                interval_from="quantile",
            )
            result["Notes"] = "quantile intervals"
        elif kind == "gaussian":
            _, train_s = timed_call(
                _train_supervised_tuple,
                model,
                GaussianNLLLoss(),
                train_loader,
                epochs=cfg.epochs,
                lr=cfg.lr,
            )
            result, eval_s = timed_call(_evaluate_gaussian_method, model, splits)
            result["Notes"] = "heteroscedastic Gaussian"
        elif kind == "pseudo_gaussian":
            _, train_s = timed_call(
                _train_pseudo_label_gaussian,
                model,
                splits,
                target_all=target_all,
                pseudo_target=pseudo_target_all,
                pseudo_confidence=pseudo_confidence_all,
                label_mask=label_mask,
                epochs=cfg.epochs,
                lr=cfg.lr,
            )
            result, eval_s = timed_call(_evaluate_gaussian_method, model, splits)
            result["Method"] = name
            result["Notes"] = "Gaussian student with bootstrap teacher pseudo labels"
        elif kind == "pseudo_consistency":
            _, train_s = timed_call(
                _train_pseudo_label_consistency,
                model,
                splits,
                target_all=target_all,
                pseudo_target=pseudo_target_all,
                pseudo_confidence=pseudo_confidence_all,
                teacher_pred=teacher_mean_all,
                label_mask=label_mask,
                epochs=cfg.epochs,
                lr=cfg.lr,
            )
            result, eval_s = timed_call(_evaluate_point_method, name, model, splits)
            result["Notes"] = "point student with pseudo labels + bootstrap-teacher consistency"
        elif kind == "eiv":
            _, train_s = timed_call(
                _train_eiv,
                model,
                splits,
                epochs=cfg.epochs,
                lr=cfg.lr,
            )
            result, eval_s = timed_call(_evaluate_point_method, name, model, splits)
            result["Notes"] = "uses feature-error-aware EIV loss"
        else:
            raise AssertionError(kind)

        result["train_s"] = float(train_s)
        result["eval_s"] = float(eval_s)
        result["DataSource"] = splits["data_source"]
        result["LabeledFraction"] = float(label_mask.float().mean().item())
        result["PseudoAcceptRate"] = pseudo_accept_rate if "PseudoLabel" in name else None
        result["PseudoMeanConf"] = pseudo_mean_conf if "PseudoLabel" in name else None
        rows.append(result)

    return rows, notes


def main(
    cfg: Optional[PhotoZBenchmarkConfig] = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or PhotoZBenchmarkConfig()
    set_comparison_seed(cfg.seed)
    rows, notes = run_benchmark(cfg)

    print_fairness_notes(
        title="Photo-z Benchmark Comparison",
        seed_policy="fixed seed; shared SDSS-style split and model init seeds",
        train_budget=f"{cfg.epochs} epochs, batch_size={cfg.batch_size}, lr={cfg.lr}",
        metric_policy=(
            "Shared point metrics + photo-z metrics (NMAD/catastrophic/high-z MAE) and "
            "native interval metrics for Gaussian/quantile methods; pseudo-label diagnostics "
            "for partial-label SSL rows"
        ),
    )
    print_comparison_summary(
        "Photo-z Benchmark Summary (SDSS-style)",
        rows,
        metric_order=[
            "RMSE",
            "MAE",
            "NMAD",
            "CatastrophicRate",
            "HighZ_MAE",
            "NLL",
            "Cov90",
            "Width90",
            "train_s",
            "eval_s",
        ],
    )
    data_sources = sorted({str(row.get("DataSource")) for row in rows})
    print("\nDataset notes:")
    print(f"- Data source used: {', '.join(data_sources)}")
    print(
        "- Real SDSS cache is used if present; otherwise deterministic simulated SDSS-style data is used."
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/photoz_benchmark_comparison.py",
            task="Photometric redshift benchmark (photo-z, SDSS-style)",
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the standard photo-z benchmark.")
    parser.add_argument("--summary-json-path", type=str, default=None)
    parser.add_argument("--force-simulated", action="store_true")
    parser.add_argument("--require-real-data", action="store_true")
    parser.add_argument("--dataset-path", type=str, default=None)
    args = parser.parse_args()
    cfg = PhotoZBenchmarkConfig(
        dataset_path=args.dataset_path,
        force_simulated=args.force_simulated,
        require_real_data=args.require_real_data,
    )
    main(cfg, summary_json_path=args.summary_json_path)
