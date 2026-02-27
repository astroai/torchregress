"""
Photo-z benchmark comparison with shared-budget metrics and summary artifacts.

This example focuses on a domain-realistic photometric redshift workload using
SDSS-style features (colors + measurement errors). It supports:
- local cached real SDSS data if present
- deterministic simulated fallback otherwise
- shared-budget comparison across point, robust, probabilistic, and EIV methods
- machine-readable summary JSON output for audit/review pipelines
"""

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
    MultiQuantileLoss,
    WeightedHuberLoss,
    WeightedMSELoss,
)


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
    force_simulated: bool = False
    allow_download: bool = False
    sample_size_if_generate: int = 5000


class PhotoZRegressor(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 1, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_photoz_df(cfg: PhotoZBenchmarkConfig):
    data_dir = Path("data/sdss")
    real_path = data_dir / "sdss_photoz_real.csv"
    sim_path = data_dir / "sdss_photoz_simulated.csv"
    if not cfg.force_simulated and real_path.exists():
        return pd.read_csv(real_path)
    if sim_path.exists():
        return pd.read_csv(sim_path)
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
        "yerr_train": yerr_train_s,
        "yerr_cal": yerr_cal_s,
        "yerr_test": yerr_test_s,
        "y_test_raw": y_test,
        "y_scale": y_std,
        "y_shift": y_mean,
        "data_source": "real_sdss_cache"
        if (Path("data/sdss/sdss_photoz_real.csv").exists() and not cfg.force_simulated)
        else "simulated_sdss",
    }


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
    d_in = int(splits["x_train"].shape[1])

    specs: list[tuple[str, nn.Module, str]] = [
        ("MSE", PhotoZRegressor(d_in, hidden=cfg.hidden), "mse"),
        ("Huber", PhotoZRegressor(d_in, hidden=cfg.hidden), "huber"),
        ("Quantile90", PhotoZRegressor(d_in, out_dim=3, hidden=cfg.hidden), "quantile"),
        ("GaussianNLL", PhotoZRegressor(d_in, out_dim=2, hidden=cfg.hidden), "gaussian"),
        ("FunctionalEIV", PhotoZRegressor(d_in, hidden=cfg.hidden), "eiv"),
    ]

    rows: list[dict[str, object]] = []
    notes = [
        "SDSS-style photo-z benchmark with cached-real or deterministic simulated fallback",
        "Shared train/cal/test splits and shared training budget across methods",
        "Photo-z metrics include NMAD, catastrophic outlier rate, and high-z MAE",
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
            "native interval metrics for Gaussian/quantile methods"
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
    main()
