"""Photo-z competing-method benchmark for Shift-Factored Predictive Transport."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import photoz_benchmark_comparison as pzbase
import spt_reg_synthetic_comparison as sptbase
import torch
import torch.nn as nn
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    timed_call,
    write_comparison_summary_json,
)
from photoz_binned_utils import (
    bin_targets,
    make_bins_from_train_targets,
)
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import GaussianNLLLoss, WeightedCrossEntropyLoss
from torchregress.test_time import (
    FeatureStatNormalizer,
    ShiftFactoredPredictiveTransport,
    ShiftFactoredTransportConfig,
    SignificantSubspaceAligner,
)


@dataclass(frozen=True)
class SPTRegPhotoZConfig:
    seed: int = 260409
    n_train: int = 256
    n_target_unlabeled: int = 96
    n_target_cal: int = 64
    n_target_test: int = 96
    batch_size: int = 64
    epochs: int = 8
    lr: float = 2e-3
    hidden: int = 64
    n_bins: int = 24
    alpha: float = 0.1
    ppi_quantile: float = 0.9
    n_support: int = 128
    n_samples_eval: int = 64
    target_label_budget: int = 32
    dataset_path: str | None = None
    train_dataset_path: str | None = None
    cal_dataset_path: str | None = None
    test_dataset_path: str | None = None
    force_simulated: bool = False
    require_real_data: bool = False
    allow_download: bool = False
    sample_size_if_generate: int = 5000


class _GaussianPhotoZPredictor:
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def predict_distribution(self, X: np.ndarray, **kwargs: object):
        del kwargs
        x_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(x_t)
        mean = out[:, :1].cpu().numpy().reshape(-1)
        logvar = out[:, 1:2].clamp(-8.0, 6.0).cpu().numpy().reshape(-1)
        std = np.exp(0.5 * logvar).astype(np.float32)
        return sptbase.PredictiveBatch(mean=mean.astype(np.float32), std=std)


class _BinnedPhotoZPredictor:
    def __init__(self, model: nn.Module, bin_edges: torch.Tensor) -> None:
        self.model = model
        self.bin_edges = bin_edges.detach().cpu().numpy().astype(np.float32)

    def predict_distribution(self, X: np.ndarray, **kwargs: object):
        del kwargs
        x_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x_t).cpu().numpy()
        return sptbase.PredictiveBatch(
            bar_logits=logits.astype(np.float32),
            bin_edges=self.bin_edges,
        )


def _has_explicit_splits(cfg: SPTRegPhotoZConfig) -> bool:
    provided = [
        cfg.train_dataset_path is not None,
        cfg.cal_dataset_path is not None,
        cfg.test_dataset_path is not None,
    ]
    if any(provided) and not all(provided):
        raise ValueError(
            "Explicit split mode requires train_dataset_path, cal_dataset_path, and test_dataset_path."
        )
    return all(provided)


def _frame_to_splits(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    cfg: SPTRegPhotoZConfig,
) -> dict[str, torch.Tensor]:
    feature_cols, error_cols = pzbase._infer_feature_columns(train_df)
    target_col = "spec_z"
    target_err_col = "spec_z_err"

    def _to_tensor(frame: pd.DataFrame, cols: list[str]) -> torch.Tensor:
        return torch.tensor(frame[cols].to_numpy(dtype="float32"))

    x_train = _to_tensor(train_df, feature_cols)
    xerr_train = _to_tensor(train_df, error_cols)
    y_train_raw = torch.tensor(train_df[target_col].to_numpy(dtype="float32")).unsqueeze(1)
    yerr_train = torch.tensor(train_df[target_err_col].to_numpy(dtype="float32")).unsqueeze(1)

    x_target = _to_tensor(target_df, feature_cols)
    xerr_target = _to_tensor(target_df, error_cols)
    y_target_raw = torch.tensor(target_df[target_col].to_numpy(dtype="float32")).unsqueeze(1)
    yerr_target = torch.tensor(target_df[target_err_col].to_numpy(dtype="float32")).unsqueeze(1)

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    y_mean = y_train_raw.mean(dim=0, keepdim=True)
    y_std = y_train_raw.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)

    target_perm = torch.randperm(
        x_target.shape[0], generator=torch.Generator().manual_seed(cfg.seed)
    )
    x_target = x_target[target_perm]
    xerr_target = xerr_target[target_perm]
    y_target_raw = y_target_raw[target_perm]
    yerr_target = yerr_target[target_perm]

    unlabeled_stop = cfg.n_target_unlabeled
    cal_stop = unlabeled_stop + cfg.n_target_cal

    splits = {
        "x_train": (x_train - x_mean) / x_std,
        "xerr_train": xerr_train / x_std,
        "y_train": (y_train_raw - y_mean) / y_std,
        "y_train_raw": y_train_raw,
        "yerr_train": yerr_train / y_std,
        "x_target_pool": (x_target - x_mean) / x_std,
        "xerr_target_pool": xerr_target / x_std,
        "y_target_pool": (y_target_raw - y_mean) / y_std,
        "y_target_pool_raw": y_target_raw,
        "yerr_target_pool": yerr_target / y_std,
        "x_cal": ((x_target - x_mean) / x_std)[unlabeled_stop:cal_stop],
        "x_test": ((x_target - x_mean) / x_std)[cal_stop:],
        "xerr_test": (xerr_target / x_std)[cal_stop:],
        "y_cal": ((y_target_raw - y_mean) / y_std)[unlabeled_stop:cal_stop],
        "y_test": ((y_target_raw - y_mean) / y_std)[cal_stop:],
        "y_test_raw": y_target_raw[cal_stop:],
        "y_scale": y_std,
        "y_shift": y_mean,
        "data_source": "photoz_explicit" if _has_explicit_splits(cfg) else "photoz_shifted_pool",
        "unlabeled_stop": torch.tensor([unlabeled_stop]),
        "cal_stop": torch.tensor([cal_stop]),
    }
    return splits


def _make_photoz_shift_splits(cfg: SPTRegPhotoZConfig) -> dict[str, torch.Tensor]:
    if _has_explicit_splits(cfg):
        train_df = pzbase._load_photoz_table(Path(cfg.train_dataset_path or ""))
        cal_df = pzbase._load_photoz_table(Path(cfg.cal_dataset_path or ""))
        test_df = pzbase._load_photoz_table(Path(cfg.test_dataset_path or ""))
        train_df = train_df.sample(n=cfg.n_train, random_state=cfg.seed).reset_index(drop=True)
        target_df = pd.concat([cal_df, test_df], ignore_index=True)
        need = cfg.n_target_unlabeled + cfg.n_target_cal + cfg.n_target_test
        target_df = target_df.sample(n=need, random_state=cfg.seed + 1).reset_index(drop=True)
        return _frame_to_splits(train_df, target_df, cfg=cfg)

    base_cfg = pzbase.PhotoZBenchmarkConfig(
        seed=cfg.seed,
        n_train=max(
            cfg.n_train + cfg.n_target_unlabeled + cfg.n_target_cal + cfg.n_target_test, 512
        ),
        n_cal=8,
        n_test=8,
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        hidden=cfg.hidden,
        dataset_path=cfg.dataset_path,
        force_simulated=cfg.force_simulated,
        require_real_data=cfg.require_real_data,
        allow_download=cfg.allow_download,
        sample_size_if_generate=cfg.sample_size_if_generate,
    )
    df = pzbase._load_photoz_df(base_cfg)
    feature_cols, error_cols = pzbase._infer_feature_columns(df)
    need = cfg.n_train + cfg.n_target_unlabeled + cfg.n_target_cal + cfg.n_target_test
    if len(df) < need:
        raise ValueError(f"Need {need} photo-z rows but dataset only has {len(df)}.")
    df = df.copy()
    df["_shift_score"] = np.sqrt(np.mean(df[error_cols].to_numpy(dtype=float) ** 2, axis=1))
    df = df.sort_values("_shift_score", ascending=False).reset_index(drop=True)
    target_need = cfg.n_target_unlabeled + cfg.n_target_cal + cfg.n_target_test
    target_df = df.iloc[:target_need].copy().reset_index(drop=True)
    source_pool = df.iloc[target_need:].copy().reset_index(drop=True)
    source_df = source_pool.sample(n=cfg.n_train, random_state=cfg.seed).reset_index(drop=True)
    return _frame_to_splits(source_df, target_df, cfg=cfg)


def _build_loader(x: torch.Tensor, y: torch.Tensor, *, batch_size: int, seed: int):
    return DataLoader(
        TensorDataset(x, y),
        batch_size=min(batch_size, int(x.shape[0])),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )


def _train_binned(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    bin_edges: torch.Tensor,
    epochs: int,
    lr: float,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = WeightedCrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, bin_targets(yb, bin_edges))
            loss.backward()
            opt.step()
    model.eval()
    return model


def _evaluate_photoz_row(
    *,
    method: str,
    family: str,
    batch_cal: sptbase.PredictiveBatch,
    batch_test: sptbase.PredictiveBatch,
    y_cal: np.ndarray,
    y_test: np.ndarray,
    cfg: SPTRegPhotoZConfig,
    splits: dict[str, torch.Tensor],
    train_s: float,
    eval_s: float,
    notes: str,
) -> dict[str, object]:
    row = sptbase._evaluate_row(
        method=method,
        family=family,
        batch_cal=batch_cal,
        batch_test=batch_test,
        y_cal=y_cal,
        y_test=y_test,
        cfg=cfg,
        train_s=train_s,
        eval_s=eval_s,
        notes=notes,
    )
    mean, _ = sptbase._batch_mean_std(batch_test)
    point_t = torch.tensor(mean[:, None], dtype=torch.float32)
    row.update(pzbase._photoz_metrics(point_t, splits["y_test"], splits))
    return row


def run_comparison(cfg: SPTRegPhotoZConfig) -> tuple[list[dict[str, object]], list[str]]:
    splits = _make_photoz_shift_splits(cfg)
    source_x = splits["x_train"]
    source_y = splits["y_train"]
    target_pool_x = splits["x_target_pool"]
    target_pool_y = splits["y_target_pool"]
    unlabeled_stop = int(splits["unlabeled_stop"][0])
    cal_stop = int(splits["cal_stop"][0])
    y_cal = target_pool_y[unlabeled_stop:cal_stop].cpu().numpy().reshape(-1)
    y_test = target_pool_y[cal_stop:].cpu().numpy().reshape(-1)

    gauss_model = pzbase.PhotoZRegressor(int(source_x.shape[1]), out_dim=2, hidden=cfg.hidden)
    gauss_loader = _build_loader(source_x, source_y, batch_size=cfg.batch_size, seed=cfg.seed)
    _, gauss_train_s = timed_call(
        pzbase._train_supervised_tuple,
        gauss_model,
        GaussianNLLLoss(),
        gauss_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )
    gauss_predictor = _GaussianPhotoZPredictor(gauss_model)

    bin_edges = make_bins_from_train_targets(source_y, n_bins=cfg.n_bins)
    binned_model = pzbase.PhotoZRegressor(
        int(source_x.shape[1]), out_dim=cfg.n_bins, hidden=cfg.hidden
    )
    _, binned_train_s = timed_call(
        _train_binned,
        binned_model,
        gauss_loader,
        bin_edges=bin_edges,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )
    binned_predictor = _BinnedPhotoZPredictor(binned_model, bin_edges)

    target_pool_np = target_pool_x.cpu().numpy()
    source_gaussian_pool = gauss_predictor.predict_distribution(target_pool_np)
    source_binned_pool = binned_predictor.predict_distribution(target_pool_np)
    n_pool = target_pool_np.shape[0]

    rows: list[dict[str, object]] = []
    source_gaussian_cal = sptbase._slice_batch(source_gaussian_pool, unlabeled_stop, cal_stop)
    source_gaussian_test = sptbase._slice_batch(source_gaussian_pool, cal_stop, n_pool)
    rows.append(
        _evaluate_photoz_row(
            method="SourceGaussian",
            family="Gaussian",
            batch_cal=source_gaussian_cal,
            batch_test=source_gaussian_test,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            splits=splits,
            train_s=gauss_train_s,
            eval_s=0.0,
            notes="source Gaussian photo-z predictor without target adaptation",
        )
    )

    normalizer = FeatureStatNormalizer().fit(source_x.cpu().numpy())
    norm_pool = gauss_predictor.predict_distribution(normalizer.transform(target_pool_np))
    rows.append(
        _evaluate_photoz_row(
            method="FeatureStatNormGaussian",
            family="Gaussian",
            batch_cal=sptbase._slice_batch(norm_pool, unlabeled_stop, cal_stop),
            batch_test=sptbase._slice_batch(norm_pool, cal_stop, n_pool),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            splits=splits,
            train_s=gauss_train_s,
            eval_s=0.0,
            notes="feature-stat normalization before rerunning the Gaussian photo-z model",
        )
    )

    aligner = SignificantSubspaceAligner(rank=2, random_state=cfg.seed).fit(
        source_x.cpu().numpy(),
        source_y.cpu().numpy().reshape(-1),
    )
    aligned_pool = gauss_predictor.predict_distribution(aligner.transform(target_pool_np))
    rows.append(
        _evaluate_photoz_row(
            method="SignificantSubspaceGaussian",
            family="Gaussian",
            batch_cal=sptbase._slice_batch(aligned_pool, unlabeled_stop, cal_stop),
            batch_test=sptbase._slice_batch(aligned_pool, cal_stop, n_pool),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            splits=splits,
            train_s=gauss_train_s,
            eval_s=0.0,
            notes="significant-subspace alignment only on photo-z features",
        )
    )

    raw_split = sptbase._manual_split_conformal(
        source_gaussian_cal, y_cal, source_gaussian_test, cfg.alpha
    )
    rows.append(
        _evaluate_photoz_row(
            method="RawSplitConformalGaussian",
            family="Gaussian",
            batch_cal=source_gaussian_cal,
            batch_test=raw_split,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            splits=splits,
            train_s=gauss_train_s,
            eval_s=0.0,
            notes="raw Gaussian source predictions with target split conformal",
        )
    )

    spt = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(
            n_support=cfg.n_support, alpha=cfg.alpha, random_state=cfg.seed
        )
    ).fit_source(
        gauss_predictor.predict_distribution(source_x.cpu().numpy()),
        source_y.cpu().numpy().reshape(-1),
        source_inputs=source_x.cpu().numpy(),
    )
    spt_pool, spt_eval_s = timed_call(
        spt.adapt_unlabeled_target,
        target_predictions=source_gaussian_pool,
        target_inputs=target_pool_np,
        predictor=gauss_predictor,
    )
    spt_cal = sptbase._slice_batch(spt_pool, unlabeled_stop, cal_stop)
    spt_test = sptbase._slice_batch(spt_pool, cal_stop, n_pool)
    _, spt_cal_s = timed_call(spt.calibrate_target, spt_cal, y_cal)
    rows.append(
        _evaluate_photoz_row(
            method="SPTTransportGaussian",
            family="Gaussian",
            batch_cal=spt_cal,
            batch_test=spt_test,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            splits=splits,
            train_s=gauss_train_s,
            eval_s=spt_eval_s,
            notes="transported Gaussian photo-z predictions without conformal wrapping",
        )
    )
    rows.append(
        _evaluate_photoz_row(
            method="SPTRegGaussian",
            family="Gaussian",
            batch_cal=spt_cal,
            batch_test=spt.apply_conformal(spt_test),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            splits=splits,
            train_s=gauss_train_s + spt_cal_s,
            eval_s=spt_eval_s,
            notes="full SPT-Reg on Gaussian photo-z predictions",
        )
    )

    small_batch, small_s = timed_call(
        sptbase._refit_batch,
        splits["x_cal"][: cfg.target_label_budget].cpu().numpy(),
        splits["y_cal"][: cfg.target_label_budget].cpu().numpy().reshape(-1),
        splits["x_test"].cpu().numpy(),
    )
    rows.append(
        _evaluate_photoz_row(
            method="TargetRefitSmallGaussian",
            family="Gaussian",
            batch_cal=sptbase._refit_batch(
                splits["x_cal"][: cfg.target_label_budget].cpu().numpy(),
                splits["y_cal"][: cfg.target_label_budget].cpu().numpy().reshape(-1),
                splits["x_cal"].cpu().numpy(),
            ),
            batch_test=small_batch,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            splits=splits,
            train_s=small_s,
            eval_s=0.0,
            notes="small target-label refit baseline on photo-z calibration labels",
        )
    )

    source_binned_cal = sptbase._slice_batch(source_binned_pool, unlabeled_stop, cal_stop)
    source_binned_test = sptbase._slice_batch(source_binned_pool, cal_stop, n_pool)
    rows.append(
        _evaluate_photoz_row(
            method="SourceBinnedPDF",
            family="BinnedPDF",
            batch_cal=source_binned_cal,
            batch_test=source_binned_test,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            splits=splits,
            train_s=binned_train_s,
            eval_s=0.0,
            notes="ordered-bin photo-z predictive law without target adaptation",
        )
    )

    spt_binned = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(
            n_support=cfg.n_support, alpha=cfg.alpha, random_state=cfg.seed
        )
    ).fit_source(
        binned_predictor.predict_distribution(source_x.cpu().numpy()),
        source_y.cpu().numpy().reshape(-1),
        source_inputs=source_x.cpu().numpy(),
    )
    spt_binned_pool, spt_binned_eval_s = timed_call(
        spt_binned.adapt_unlabeled_target,
        target_predictions=source_binned_pool,
        target_inputs=target_pool_np,
    )
    spt_binned_cal = sptbase._slice_batch(spt_binned_pool, unlabeled_stop, cal_stop)
    spt_binned_test = sptbase._slice_batch(spt_binned_pool, cal_stop, n_pool)
    _, spt_binned_cal_s = timed_call(spt_binned.calibrate_target, spt_binned_cal, y_cal)
    rows.append(
        _evaluate_photoz_row(
            method="SPTTransportBinnedPDF",
            family="BinnedPDF",
            batch_cal=spt_binned_cal,
            batch_test=spt_binned_test,
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            splits=splits,
            train_s=binned_train_s,
            eval_s=spt_binned_eval_s,
            notes="transported ordered-bin photo-z predictions without conformal wrapping",
        )
    )
    rows.append(
        _evaluate_photoz_row(
            method="SPTRegBinnedPDF",
            family="BinnedPDF",
            batch_cal=spt_binned_cal,
            batch_test=spt_binned.apply_conformal(spt_binned_test),
            y_cal=y_cal,
            y_test=y_test,
            cfg=cfg,
            splits=splits,
            train_s=binned_train_s + spt_binned_cal_s,
            eval_s=spt_binned_eval_s,
            notes="SPT-Reg applied to ordered-bin photo-z predictions",
        )
    )

    notes = [
        "The photo-z benchmark uses existing photo-z feature engineering and metrics from the benchmark helpers.",
        "When explicit splits are not provided, target rows are taken from the highest photometric-error region.",
        "BinnedPDF rows use ordered-bin classification heads with source-derived target bins.",
    ]
    return rows, notes


def main(
    cfg: SPTRegPhotoZConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or SPTRegPhotoZConfig()
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="SPT-Reg photo-z comparison",
        seed_policy=f"fixed seed = {cfg.seed}",
        train_budget="shared photo-z source training budget and matched target-label budget",
        metric_policy="photo-z, probabilistic, interval, selective, and PPI summaries",
    )
    print_comparison_summary(
        "SPT-Reg photo-z summary",
        rows,
        metric_order=[
            "RMSE",
            "NMAD",
            "CatastrophicRate",
            "HighZ_MAE",
            "NLL",
            "CRPS",
            "Cov90",
            "Width90",
            "AURC",
            "PPIMeanCIWidth",
            "PPIQuantileCIWidth",
            "train_s",
            "eval_s",
        ],
    )

    if summary_json_path is not None:
        write_comparison_summary_json(
            summary_json_path,
            example="examples/spt_reg_photoz_comparison.py",
            task="SPT-Reg photo-z competing-method benchmark with BinnedPDF",
            config=cfg,
            rows=rows,
            notes=notes,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the photo-z SPT-Reg competing-method benchmark."
    )
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
