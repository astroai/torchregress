"""Focused photo-z comparison using ordered bins (example-local NNC-CRPS style).

This example keeps ordered-bin CRPS and temperature scaling logic inside `examples/`
to avoid expanding the core public API surface for a highly specialized setup.
"""

import argparse
from dataclasses import dataclass
from typing import Optional

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
    bin_targets,
    fit_temperature_scaler,
    logits_to_pdf,
    make_bins_from_train_targets,
    ordered_bin_crps_loss,
    pdf_quantiles,
    pdf_to_point_estimate,
)
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import GaussianNLLLoss, MultiQuantileLoss, WeightedCrossEntropyLoss
from torchregress.metrics import continuous_ranked_probability_score, crps_gaussian, gaussian_nll


@dataclass(frozen=True)
class PhotoZNNCConfig:
    seed: int = 260226
    n_train: int = 512
    n_cal: int = 256
    n_test: int = 256
    batch_size: int = 64
    epochs: int = 12
    lr: float = 2e-3
    hidden: int = 64
    n_bins: int = 48
    binning_strategy: str = "quantile"
    force_simulated: bool = False
    allow_download: bool = False
    sample_size_if_generate: int = 5000
    temperature_max_iter: int = 200
    temperature_lr: float = 0.05


def _pit_chi2_from_pdf(pdf: torch.Tensor, targets: torch.Tensor, bin_edges: torch.Tensor) -> float:
    target_bins = bin_targets(targets, bin_edges)
    cdf = torch.cumsum(pdf, dim=-1)
    pit = cdf[torch.arange(cdf.shape[0], device=cdf.device), target_bins].clamp(0.0, 1.0)
    hist_edges = torch.linspace(0.0, 1.0, 21, device=pit.device)
    counts = torch.histogram(pit, hist_edges)[0].float()
    expected = pit.numel() / 20.0
    chi2 = torch.sum((counts - expected) ** 2 / max(expected, 1.0))
    return float(chi2.item())


def _train_binned(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    bin_edges: torch.Tensor,
    objective: str,
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
            if objective == "ce":
                y_bin = bin_targets(yb, bin_edges)
                loss = ce(logits, y_bin)
            elif objective == "crps":
                loss = ordered_bin_crps_loss(logits, yb, bin_edges, reduction="mean")
            else:
                raise ValueError(f"Unknown binned objective: {objective}")
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


def _evaluate_binned_row(
    *,
    name: str,
    model: nn.Module,
    splits: dict[str, torch.Tensor],
    bin_edges: torch.Tensor,
    temperature: float = 1.0,
) -> dict[str, object]:
    with torch.no_grad():
        logits = model(splits["x_test"])
    logits_s = apply_temperature(logits, temperature)
    pdf = logits_to_pdf(logits_s)
    point = pdf_to_point_estimate(pdf, bin_edges)
    q = pdf_quantiles(pdf, bin_edges, quantiles=[0.05, 0.95])
    q05, q95 = q[0.05], q[0.95]
    q05, q95 = torch.minimum(q05, q95), torch.maximum(q05, q95)
    cov90, width90 = pzbase._coverage_width(q05, q95, splits["y_test"])

    y_bin = bin_targets(splits["y_test"], bin_edges)
    pdf_nll = float(nn.CrossEntropyLoss()(logits_s, y_bin).item())

    point_metrics = pzbase._point_metrics(point, splits["y_test"])
    photoz_metrics = pzbase._photoz_metrics(point, splits["y_test"], splits)
    crps = float(
        ordered_bin_crps_loss(logits_s, splits["y_test"], bin_edges, reduction="mean").item()
    )
    pit_chi2 = _pit_chi2_from_pdf(pdf, splits["y_test"], bin_edges)
    return {
        "Method": name,
        **point_metrics,
        **photoz_metrics,
        "CRPS": crps,
        "PDF_NLL": pdf_nll,
        "PITChi2": pit_chi2,
        "NativeCov90": cov90,
        "NativeWidth90": width90,
    }


def _evaluate_gaussian_row(model: nn.Module, splits: dict[str, torch.Tensor]) -> dict[str, object]:
    with torch.no_grad():
        out = model(splits["x_test"])
        mean = out[:, :1]
        logvar = out[:, 1:2].clamp(-8.0, 6.0)
        var = torch.exp(logvar)
        std = var.sqrt()
    point_metrics = pzbase._point_metrics(mean, splits["y_test"])
    photoz_metrics = pzbase._photoz_metrics(mean, splits["y_test"], splits)
    lower90 = mean - 1.645 * std
    upper90 = mean + 1.645 * std
    cov90, width90 = pzbase._coverage_width(lower90, upper90, splits["y_test"])
    return {
        "Method": "GaussianNLL",
        **point_metrics,
        **photoz_metrics,
        "CRPS": float(crps_gaussian(mean, splits["y_test"], std, reduction="mean")),
        "PDF_NLL": float(gaussian_nll(mean, splits["y_test"], var, reduction="mean")),
        "PITChi2": None,
        "NativeCov90": cov90,
        "NativeWidth90": width90,
    }


def _evaluate_quantile_row(model: nn.Module, splits: dict[str, torch.Tensor]) -> dict[str, object]:
    with torch.no_grad():
        out = model(splits["x_test"])
    q05 = out[:, 0:1]
    q50 = out[:, 1:2]
    q95 = out[:, 2:3]
    q05, q95 = torch.minimum(q05, q95), torch.maximum(q05, q95)
    cov90, width90 = pzbase._coverage_width(q05, q95, splits["y_test"])
    point_metrics = pzbase._point_metrics(q50, splits["y_test"])
    photoz_metrics = pzbase._photoz_metrics(q50, splits["y_test"], splits)
    crps = continuous_ranked_probability_score(
        {0.05: q05, 0.5: q50, 0.95: q95},
        splits["y_test"],
        reduction="mean",
    )
    return {
        "Method": "MultiQuantileLoss",
        **point_metrics,
        **photoz_metrics,
        "CRPS": float(crps),
        "PDF_NLL": None,
        "PITChi2": None,
        "NativeCov90": cov90,
        "NativeWidth90": width90,
    }


def run_comparison(cfg: PhotoZNNCConfig) -> tuple[list[dict[str, object]], list[str]]:
    splits = pzbase._make_splits(
        pzbase.PhotoZBenchmarkConfig(
            seed=cfg.seed,
            n_train=cfg.n_train,
            n_cal=cfg.n_cal,
            n_test=cfg.n_test,
            batch_size=cfg.batch_size,
            epochs=cfg.epochs,
            lr=cfg.lr,
            hidden=cfg.hidden,
            force_simulated=cfg.force_simulated,
            allow_download=cfg.allow_download,
            sample_size_if_generate=cfg.sample_size_if_generate,
        )
    )
    bin_edges = make_bins_from_train_targets(
        splits["y_train"],
        n_bins=cfg.n_bins,
        strategy=cfg.binning_strategy,
    )
    train_loader = DataLoader(
        TensorDataset(splits["x_train"], splits["y_train"]),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    d_in = int(splits["x_train"].shape[1])

    rows: list[dict[str, object]] = []
    notes = [
        "Ordered-bin CRPS setup is examples-only and aligned with regression-as-classification flow.",
        "RAIL baseline adapter is handled by tools/photoz_rail_compare.py for cross-framework comparison.",
        "Shared train/cal/test split and training budget across methods.",
    ]

    # Binned CE
    set_comparison_seed(cfg.seed + 1)
    ce_model = pzbase.PhotoZRegressor(d_in, out_dim=cfg.n_bins, hidden=cfg.hidden)
    _, ce_train_s = timed_call(
        _train_binned,
        ce_model,
        train_loader,
        bin_edges=bin_edges,
        objective="ce",
        epochs=cfg.epochs,
        lr=cfg.lr,
    )
    ce_row, ce_eval_s = timed_call(
        _evaluate_binned_row,
        name="BinnedCE",
        model=ce_model,
        splits=splits,
        bin_edges=bin_edges,
    )
    ce_row["train_s"] = float(ce_train_s)
    ce_row["eval_s"] = float(ce_eval_s)
    ce_row["calibrate_s"] = 0.0
    ce_row["DataSource"] = splits["data_source"]
    ce_row["Notes"] = "classification baseline on ordered bins"
    rows.append(ce_row)

    # Binned CE + temperature scaling
    with torch.no_grad():
        cal_logits = ce_model(splits["x_cal"])
    temp, cal_s = timed_call(
        fit_temperature_scaler,
        cal_logits,
        splits["y_cal"],
        bin_edges,
        max_iter=cfg.temperature_max_iter,
        lr=cfg.temperature_lr,
    )
    ce_temp_row, ce_temp_eval_s = timed_call(
        _evaluate_binned_row,
        name="BinnedCE+TempScaling",
        model=ce_model,
        splits=splits,
        bin_edges=bin_edges,
        temperature=float(temp),
    )
    ce_temp_row["train_s"] = float(ce_train_s)
    ce_temp_row["eval_s"] = float(ce_temp_eval_s)
    ce_temp_row["calibrate_s"] = float(cal_s)
    ce_temp_row["DataSource"] = splits["data_source"]
    ce_temp_row["Notes"] = "post-hoc scalar temperature on calibration split"
    rows.append(ce_temp_row)

    # Ordered-bin CRPS
    set_comparison_seed(cfg.seed + 2)
    crps_model = pzbase.PhotoZRegressor(d_in, out_dim=cfg.n_bins, hidden=cfg.hidden)
    _, crps_train_s = timed_call(
        _train_binned,
        crps_model,
        train_loader,
        bin_edges=bin_edges,
        objective="crps",
        epochs=cfg.epochs,
        lr=cfg.lr,
    )
    crps_row, crps_eval_s = timed_call(
        _evaluate_binned_row,
        name="OrderedBinCRPS",
        model=crps_model,
        splits=splits,
        bin_edges=bin_edges,
    )
    crps_row["train_s"] = float(crps_train_s)
    crps_row["eval_s"] = float(crps_eval_s)
    crps_row["calibrate_s"] = 0.0
    crps_row["DataSource"] = splits["data_source"]
    crps_row["Notes"] = "ordered-bin CRPS objective (example-local)"
    rows.append(crps_row)

    # Ordered-bin CRPS + temperature scaling
    with torch.no_grad():
        crps_cal_logits = crps_model(splits["x_cal"])
    crps_temp, crps_cal_s = timed_call(
        fit_temperature_scaler,
        crps_cal_logits,
        splits["y_cal"],
        bin_edges,
        max_iter=cfg.temperature_max_iter,
        lr=cfg.temperature_lr,
    )
    crps_temp_row, crps_temp_eval_s = timed_call(
        _evaluate_binned_row,
        name="OrderedBinCRPS+TempScaling",
        model=crps_model,
        splits=splits,
        bin_edges=bin_edges,
        temperature=float(crps_temp),
    )
    crps_temp_row["train_s"] = float(crps_train_s)
    crps_temp_row["eval_s"] = float(crps_temp_eval_s)
    crps_temp_row["calibrate_s"] = float(crps_cal_s)
    crps_temp_row["DataSource"] = splits["data_source"]
    crps_temp_row["Notes"] = "ordered-bin CRPS with post-hoc temperature scaling"
    rows.append(crps_temp_row)

    # Gaussian NLL anchor
    set_comparison_seed(cfg.seed + 3)
    g_model = pzbase.PhotoZRegressor(d_in, out_dim=2, hidden=cfg.hidden)
    _, g_train_s = timed_call(
        _train_supervised_tuple,
        g_model,
        GaussianNLLLoss(),
        train_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )
    g_row, g_eval_s = timed_call(_evaluate_gaussian_row, g_model, splits)
    g_row["train_s"] = float(g_train_s)
    g_row["eval_s"] = float(g_eval_s)
    g_row["calibrate_s"] = 0.0
    g_row["DataSource"] = splits["data_source"]
    g_row["Notes"] = "heteroscedastic Gaussian anchor baseline"
    rows.append(g_row)

    # Quantile interval anchor
    set_comparison_seed(cfg.seed + 4)
    q_model = pzbase.PhotoZRegressor(d_in, out_dim=3, hidden=cfg.hidden)
    _, q_train_s = timed_call(
        _train_supervised,
        q_model,
        MultiQuantileLoss(quantiles=[0.05, 0.5, 0.95], joint_prediction=True),
        train_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )
    q_row, q_eval_s = timed_call(_evaluate_quantile_row, q_model, splits)
    q_row["train_s"] = float(q_train_s)
    q_row["eval_s"] = float(q_eval_s)
    q_row["calibrate_s"] = 0.0
    q_row["DataSource"] = splits["data_source"]
    q_row["Notes"] = "interval baseline without explicit PDF bins"
    rows.append(q_row)

    return rows, notes


def main(cfg: Optional[PhotoZNNCConfig] = None, summary_json_path: str | None = None) -> None:
    cfg = cfg or PhotoZNNCConfig()
    set_comparison_seed(cfg.seed)
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="Photo-z NNC-CRPS + RAIL-ready Comparison",
        seed_policy="fixed seed; shared split and shared method init seeds",
        train_budget=f"{cfg.epochs} epochs, batch_size={cfg.batch_size}, lr={cfg.lr}",
        metric_policy=(
            "Photo-z domain metrics + PDF metrics (CRPS/PDF_NLL/PITChi2) + interval coverage/width + runtime"
        ),
    )
    print_comparison_summary(
        "Photo-z Ordered-Bin Summary (NNC-CRPS style, example-local)",
        rows,
        metric_order=[
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

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/photoz_nnc_crps_rail_comparison.py",
            task="Photometric redshift benchmark (photo-z, ordered-bin NNC-CRPS style)",
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo-z ordered-bin NNC-CRPS style comparison.")
    parser.add_argument("--summary-json-path", type=str, default=None)
    parser.add_argument("--force-simulated", action="store_true")
    args = parser.parse_args()
    cfg = PhotoZNNCConfig(force_simulated=args.force_simulated)
    main(cfg, summary_json_path=args.summary_json_path)
