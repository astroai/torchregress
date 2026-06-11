"""Real-data SSL benchmark for SAGE-Reg on OpenML/UCI YearPredictionMSD."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.datasets import fetch_openml  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from torchregress.comparison import (  # noqa: E402
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.metrics import (  # noqa: E402
    calibration_score,
    crps_gaussian,
    gaussian_nll,
    prediction_interval_coverage_probability,
)
from torchregress.prediction import PredictiveBatch  # noqa: E402
from torchregress.semi_supervised import (  # noqa: E402
    SelfAgreementTrainer,
    disagreement_to_weight,
    predictive_agreement_score,
)
from torchregress.utils.openml_relaxed import (  # noqa: E402
    fetch_openml_regression_with_sklearn_fallback,
)


@dataclass(frozen=True)
class YearRealDataConfig:
    seed: int = 260408
    dataset_path: str | None = None
    allow_download: bool = True
    cache_path: str | None = None
    target_column: str = "target"
    openml_data_id: int | None = None
    openml_dataset_name: str | None = None
    openml_version: int = 1
    max_dataset_rows: int | None = None
    weight_decay: float = 0.0
    sage_batch_relative_mode: str | None = None
    sage_batch_trust_top_k: int | None = None
    canonical_test_size: int = 51_630
    n_labeled: int = 4_096
    n_unlabeled: int = 65_536
    n_test: int = 32_768
    hidden: int = 128
    teacher_epochs: int = 24
    student_epochs: int = 24
    batch_size: int = 512
    lr: float = 2e-3
    lr_schedule: str = "constant"  # "constant" | "cosine"
    lr_min: float = 1e-5
    dropout: float = 0.10
    unlabeled_noise: float = 0.03
    feature_drop_prob: float = 0.0
    feature_mix_prob: float = 0.0
    tau: float = 0.18
    agreement_weight: float = 0.5
    pseudo_weight: float = 0.8
    ema_decay: float = 0.96
    n_views: int = 4
    weight_power: float = 1.0
    hard_weight_threshold: float | None = None
    unlabeled_fractions: tuple[float, ...] = (0.25, 0.5, 1.0)
    dataloader_num_workers: int = 0
    # RankUp (Huang, Fu, Tsao; NeurIPS 2024; arXiv:2410.22124): auxiliary quantile buckets.
    rankup_n_buckets: int = 32
    rankup_aux_weight: float = 0.35
    rankup_min_teacher_precision: float = 0.12
    # PabLO-SSL (Harit et al.; ICML 2025): batchwise precision quantile self-training gate.
    pablo_precision_quantile: float = 0.35
    # EMA smoothing of the batch quantile threshold (0 = disabled). Stabilizes early training.
    pablo_tau_ema_momentum: float = 0.0


@dataclass(frozen=True)
class YearSplit:
    x_labeled: Tensor
    y_labeled: Tensor
    x_unlabeled: Tensor
    y_unlabeled_true: Tensor
    x_test: Tensor
    y_test: Tensor
    dataset_name: str
    n_features: int


class TabularGaussianRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_var_head = nn.Linear(hidden, 1)
        nn.init.constant_(self.log_var_head.bias, -1.2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.backbone(x)
        return self.mean_head(h), self.log_var_head(h).clamp(min=-6.0, max=3.0)


class TabularGaussianRankUpRegressor(nn.Module):
    """Gaussian regressor + auxiliary bucket logits (RankUp-style, NeurIPS 2024)."""

    def __init__(self, input_dim: int, hidden: int, dropout: float, n_buckets: int) -> None:
        super().__init__()
        self.n_buckets = int(n_buckets)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_var_head = nn.Linear(hidden, 1)
        self.bucket_head = nn.Linear(hidden, self.n_buckets)
        nn.init.constant_(self.log_var_head.bias, -1.2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.backbone(x)
        return (
            self.mean_head(h),
            self.log_var_head(h).clamp(min=-6.0, max=3.0),
            self.bucket_head(h),
        )


def _training_seed(seed: int, offset: int) -> None:
    set_comparison_seed(seed + offset)


def _set_epoch_lr(
    optimizer: torch.optim.Optimizer,
    *,
    cfg: YearRealDataConfig,
    epoch_idx: int,
    total_epochs: int,
) -> None:
    """Set per-epoch learning rate for simple Adam loops (teacher / baselines)."""
    base_lr = float(cfg.lr)
    if cfg.lr_schedule == "constant":
        lr = base_lr
    elif cfg.lr_schedule == "cosine":
        if total_epochs <= 1:
            mult = 1.0
        else:
            mult = 0.5 * (1.0 + math.cos(math.pi * float(epoch_idx) / float(total_epochs - 1)))
        lr = max(base_lr * mult, float(cfg.lr_min))
    else:
        raise ValueError(
            f"unknown lr_schedule: {cfg.lr_schedule!r} (expected 'constant' or 'cosine')"
        )
    for g in optimizer.param_groups:
        g["lr"] = lr


def _write_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _rows_by_method(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["Method"]), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda item: float(item["UnlabeledFraction"]))
    return grouped


def _plot_performance(path: str | Path, rows: list[dict[str, object]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = _rows_by_method(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method, values in grouped.items():
        x = [float(v["UnlabeledFraction"]) for v in values]
        axes[0].plot(x, [float(v["CRPS"]) for v in values], marker="o", label=method)
        axes[1].plot(x, [float(v["RMSE"]) for v in values], marker="o", label=method)
    axes[0].set_title("CRPS vs unlabeled fraction")
    axes[0].set_xlabel("Unlabeled fraction")
    axes[0].set_ylabel("CRPS")
    axes[1].set_title("RMSE vs unlabeled fraction")
    axes[1].set_xlabel("Unlabeled fraction")
    axes[1].set_ylabel("RMSE")
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_calibration(path: str | Path, rows: list[dict[str, object]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = _rows_by_method(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method, values in grouped.items():
        x = [float(v["UnlabeledFraction"]) for v in values]
        axes[0].plot(x, [float(v["CoverageGap90"]) for v in values], marker="o", label=method)
        axes[1].plot(x, [float(v["CalibMAE"]) for v in values], marker="o", label=method)
    axes[0].set_title("Coverage gap vs unlabeled fraction")
    axes[0].set_xlabel("Unlabeled fraction")
    axes[0].set_ylabel("|Cov90 - 0.90|")
    axes[1].set_title("Calibration MAE vs unlabeled fraction")
    axes[1].set_xlabel("Unlabeled fraction")
    axes[1].set_ylabel("CalibMAE")
    axes[1].legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _rankdata(values: Tensor) -> Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.shape[0], device=values.device, dtype=torch.float32)
    return ranks


def _spearman_corr(x: Tensor, y: Tensor) -> float:
    xr = _rankdata(x.reshape(-1))
    yr = _rankdata(y.reshape(-1))
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = xr.std(unbiased=False) * yr.std(unbiased=False)
    if float(denom.item()) <= 0.0:
        return 0.0
    return float(((xr * yr).mean() / denom).item())


def _top_bottom_error_ratio(score: Tensor, abs_error: Tensor) -> float:
    score = score.reshape(-1)
    abs_error = abs_error.reshape(-1)
    k = max(1, score.shape[0] // 4)
    order = torch.argsort(score, descending=True)
    top_error = abs_error[order[:k]].mean()
    bottom_error = abs_error[order[-k:]].mean()
    return float((top_error / bottom_error.clamp_min(1.0e-8)).item())


def _diagnostic_summary(diagnostics: dict[str, Tensor]) -> dict[str, float]:
    abs_error = diagnostics["abs_error"]
    confidence = diagnostics["pseudo_confidence"]
    weight = diagnostics["agreement_weight"]
    disagreement = diagnostics["disagreement"]
    neg_error = -abs_error
    return {
        "ConfidenceVsNegErrorSpearman": _spearman_corr(confidence, neg_error),
        "AgreementWeightVsNegErrorSpearman": _spearman_corr(weight, neg_error),
        "NegDisagreementVsNegErrorSpearman": _spearman_corr(-disagreement, neg_error),
        "ConfidenceTopBottomErrorRatio": _top_bottom_error_ratio(confidence, abs_error),
        "AgreementWeightTopBottomErrorRatio": _top_bottom_error_ratio(weight, abs_error),
        "NegDisagreementTopBottomErrorRatio": _top_bottom_error_ratio(-disagreement, abs_error),
    }


def _format_diagnostic_summary(summary: dict[str, float], *, fraction: float) -> str:
    return (
        "\nUnlabeled ranking summary\n"
        f"  fraction: {fraction:.4f}\n"
        f"  confidence_vs_neg_error_spearman: {summary['ConfidenceVsNegErrorSpearman']:.4f}\n"
        f"  agreement_weight_vs_neg_error_spearman: {summary['AgreementWeightVsNegErrorSpearman']:.4f}\n"
        f"  neg_disagreement_vs_neg_error_spearman: {summary['NegDisagreementVsNegErrorSpearman']:.4f}\n"
        f"  confidence_top_bottom_error_ratio: {summary['ConfidenceTopBottomErrorRatio']:.4f}\n"
        f"  agreement_weight_top_bottom_error_ratio: {summary['AgreementWeightTopBottomErrorRatio']:.4f}\n"
        f"  neg_disagreement_top_bottom_error_ratio: {summary['NegDisagreementTopBottomErrorRatio']:.4f}"
    )


def _plot_unlabeled_diagnostics(
    path: str | Path,
    diagnostics: dict[str, Tensor],
    *,
    fraction: float,
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    abs_error = diagnostics["abs_error"]
    confidence = diagnostics["pseudo_confidence"]
    weight = diagnostics["agreement_weight"]
    disagreement = diagnostics["disagreement"]

    threshold = torch.quantile(abs_error, 0.75)
    high_error = abs_error >= threshold
    low_error = ~high_error

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    bins = 16

    axes[0].hist(confidence[low_error].cpu().numpy(), bins=bins, alpha=0.65, density=True)
    axes[0].hist(confidence[high_error].cpu().numpy(), bins=bins, alpha=0.65, density=True)
    axes[0].set_title("Confidence vs pseudo-label error")
    axes[0].set_xlabel("Teacher confidence")
    axes[0].set_ylabel("Density")

    axes[1].hist(weight[low_error].cpu().numpy(), bins=bins, alpha=0.65, density=True)
    axes[1].hist(weight[high_error].cpu().numpy(), bins=bins, alpha=0.65, density=True)
    axes[1].set_title("SAGE weight vs pseudo-label error")
    axes[1].set_xlabel("Agreement weight")

    axes[2].hist(disagreement[low_error].cpu().numpy(), bins=bins, alpha=0.65, density=True)
    axes[2].hist(disagreement[high_error].cpu().numpy(), bins=bins, alpha=0.65, density=True)
    axes[2].set_title("Disagreement vs pseudo-label error")
    axes[2].set_xlabel("Mean disagreement")

    axes[2].legend(["low error", "high error"], loc="best", fontsize=8)
    fig.suptitle(f"Year unlabeled diagnostic at fraction = {fraction:.2f}", fontsize=11)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _openml_regression_frame(cfg: YearRealDataConfig) -> tuple[pd.DataFrame, str]:
    """Load a generic OpenML regression dataset (numeric features + single target)."""
    if cfg.openml_data_id is None and cfg.openml_dataset_name is None:
        raise ValueError("openml regression fetch requires openml_data_id or openml_dataset_name")
    return fetch_openml_regression_with_sklearn_fallback(
        data_id=cfg.openml_data_id,
        name=cfg.openml_dataset_name,
        version=int(cfg.openml_version),
        target_column=cfg.target_column,
    )


def _load_dataset_frame(cfg: YearRealDataConfig) -> tuple[pd.DataFrame, str]:
    if cfg.dataset_path and (cfg.openml_data_id is not None or cfg.openml_dataset_name is not None):
        raise ValueError("dataset_path cannot be combined with openml_data_id/openml_dataset_name")

    if cfg.dataset_path:
        path = Path(cfg.dataset_path)
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        return frame, str(path)

    cache_path = Path(cfg.cache_path) if cfg.cache_path else None
    if cache_path is not None and cache_path.exists():
        if cache_path.suffix.lower() == ".parquet":
            try:
                return pd.read_parquet(cache_path), str(cache_path)
            except ImportError:
                csv_fallback = cache_path.with_suffix(".csv")
                if csv_fallback.exists():
                    return pd.read_csv(csv_fallback), str(csv_fallback)
                raise
        return pd.read_csv(cache_path), str(cache_path)

    if cache_path is not None and cache_path.suffix.lower() == ".parquet":
        csv_fallback = cache_path.with_suffix(".csv")
        if csv_fallback.exists():
            return pd.read_csv(csv_fallback), str(csv_fallback)

    if cfg.openml_data_id is not None or cfg.openml_dataset_name is not None:
        if not cfg.allow_download:
            raise FileNotFoundError(
                "OpenML fetch requested but allow_download is False and cache is missing"
            )
        frame, tag = _openml_regression_frame(cfg)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.suffix.lower() == ".parquet":
                try:
                    frame.to_parquet(cache_path, index=False)
                except ImportError:
                    csv_fallback = cache_path.with_suffix(".csv")
                    frame.to_csv(csv_fallback, index=False)
                    return frame, str(csv_fallback)
            else:
                frame.to_csv(cache_path, index=False)
        return frame, tag

    if not cfg.allow_download:
        raise FileNotFoundError("dataset_path/cache_path missing and allow_download is False")

    bunch = fetch_openml(name="year", version=1, as_frame=True)
    features = cast(pd.DataFrame, bunch.data).copy()
    target = pd.to_numeric(cast(pd.Series, bunch.target), errors="raise")
    frame = features.copy()
    frame[cfg.target_column] = target.to_numpy()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.suffix.lower() == ".parquet":
            try:
                frame.to_parquet(cache_path, index=False)
            except ImportError:
                csv_fallback = cache_path.with_suffix(".csv")
                frame.to_csv(csv_fallback, index=False)
                return frame, str(csv_fallback)
        else:
            frame.to_csv(cache_path, index=False)
    return frame, "OpenML:year"


def _load_year_xy_for_split(
    cfg: YearRealDataConfig,
) -> tuple[Tensor, Tensor, str, torch.Generator, int]:
    set_comparison_seed(cfg.seed)
    frame, dataset_name = _load_dataset_frame(cfg)
    if cfg.target_column not in frame.columns:
        raise ValueError(f"target column {cfg.target_column!r} not found in dataset")

    feature_frame = frame.drop(columns=[cfg.target_column]).astype("float32")
    target = frame[cfg.target_column].astype("float32")
    x_all = torch.tensor(feature_frame.to_numpy(copy=True), dtype=torch.float32)
    y_all = torch.tensor(target.to_numpy(copy=True), dtype=torch.float32).unsqueeze(1)

    n_total = int(x_all.shape[0])
    g = torch.Generator().manual_seed(cfg.seed)
    if cfg.max_dataset_rows is not None and n_total > cfg.max_dataset_rows:
        perm = torch.randperm(n_total, generator=g)
        pick = perm[: cfg.max_dataset_rows]
        x_all = x_all[pick]
        y_all = y_all[pick]
        n_total = int(x_all.shape[0])

    need = cfg.n_labeled + cfg.n_unlabeled + cfg.n_test
    if need > n_total:
        raise ValueError(f"Requested {need} rows but dataset has {n_total}")
    return x_all, y_all, dataset_name, g, n_total


def _year_train_test_indices(
    cfg: YearRealDataConfig, *, n_total: int, g: torch.Generator
) -> tuple[Tensor, Tensor]:
    canonical_like = n_total >= cfg.canonical_test_size + cfg.n_labeled + cfg.n_unlabeled
    if canonical_like:
        train_pool_end = n_total - cfg.canonical_test_size
        train_pool_idx = torch.arange(train_pool_end)
        test_pool_idx = torch.arange(train_pool_end, n_total)
        test_idx = test_pool_idx[: cfg.n_test]
    else:
        perm = torch.randperm(n_total, generator=g)
        test_idx = perm[: cfg.n_test]
        train_pool_idx = perm[cfg.n_test :]
    return train_pool_idx, test_idx


def _year_split_from_indices(
    x_all: Tensor,
    y_all: Tensor,
    labeled_idx: Tensor,
    unlabeled_idx: Tensor,
    test_idx: Tensor,
    *,
    dataset_name: str,
) -> YearSplit:
    x_train_pool = x_all[torch.cat([labeled_idx, unlabeled_idx], dim=0)]
    x_mean = x_train_pool.mean(dim=0, keepdim=True)
    x_std = x_train_pool.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_all[labeled_idx].mean(dim=0, keepdim=True)
    y_std = y_all[labeled_idx].std(dim=0, keepdim=True).clamp_min(1e-6)

    return YearSplit(
        x_labeled=(x_all[labeled_idx] - x_mean) / x_std,
        y_labeled=(y_all[labeled_idx] - y_mean) / y_std,
        x_unlabeled=(x_all[unlabeled_idx] - x_mean) / x_std,
        y_unlabeled_true=(y_all[unlabeled_idx] - y_mean) / y_std,
        x_test=(x_all[test_idx] - x_mean) / x_std,
        y_test=(y_all[test_idx] - y_mean) / y_std,
        dataset_name=dataset_name,
        n_features=int(x_all.shape[1]),
    )


def make_year_split_label_pool_fraction(
    cfg: YearRealDataConfig,
    *,
    label_pool_percent: float,
    shift_mode: str = "none",
    min_unlabeled: int = 1,
) -> YearSplit:
    """Split the full train pool into labeled vs unlabeled by labeled fraction.

    Normalization matches :func:`_make_split`: ``x`` uses labeled+unlabeled train
    moments; ``y`` uses labeled moments only.

    ``label_pool_percent`` is interpreted as a percent of the **train pool** size
    ``T`` (all rows not in the held-out test set). The number of labeled rows is
    ``max(1, min(round(p/100*T), T - max(min_unlabeled, 1)))`` so SSL trainers
    always retain at least one unlabeled row unless ``min_unlabeled`` is 0 (not
    recommended for this benchmark's SSL code paths).

    ``shift_mode``:
    - ``none``: random permutation of the train pool (same generator seed policy
      as the tail of :func:`_make_split` after train/test indices are fixed).
    - ``covariate_high_labeled``: sort the train pool by descending mean raw feature.
    - ``label_high_labeled``: sort the train pool by descending raw target.
    """
    if not 0.0 < float(label_pool_percent) <= 100.0:
        raise ValueError("label_pool_percent must lie in (0, 100]")
    mode = str(shift_mode).strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "covariate_high_labeled": "covariate_high_labeled",
        "covariate": "covariate_high_labeled",
        "label_high_labeled": "label_high_labeled",
        "label": "label_high_labeled",
    }
    if mode not in aliases:
        raise ValueError(
            f"unknown shift_mode: {shift_mode!r} (expected one of {sorted(set(aliases))})"
        )
    resolved_mode = aliases[mode]

    x_all, y_all, dataset_name, g, n_total = _load_year_xy_for_split(cfg)
    train_pool_idx, test_idx = _year_train_test_indices(cfg, n_total=n_total, g=g)
    t_pool = int(train_pool_idx.shape[0])
    nu_floor = max(int(min_unlabeled), 1)
    if t_pool < 1 + nu_floor:
        raise ValueError(
            f"train pool ({t_pool} rows) too small for min_unlabeled={min_unlabeled} "
            f"(need at least {1 + nu_floor} rows)"
        )
    raw_nl = int(round(float(label_pool_percent) / 100.0 * float(t_pool)))
    n_labeled = max(1, min(raw_nl, t_pool - nu_floor))
    n_unlabeled = t_pool - n_labeled
    if n_unlabeled < 1:
        raise ValueError("internal split error: empty unlabeled pool")

    if resolved_mode == "none":
        ordered = train_pool_idx[torch.randperm(t_pool, generator=g)]
    elif resolved_mode == "covariate_high_labeled":
        pool_x = x_all[train_pool_idx]
        score = pool_x.mean(dim=1)
        order = torch.argsort(score, descending=True)
        ordered = train_pool_idx[order]
    else:
        pool_y = y_all[train_pool_idx].reshape(-1)
        order = torch.argsort(pool_y, descending=True)
        ordered = train_pool_idx[order]

    labeled_idx = ordered[:n_labeled]
    unlabeled_idx = ordered[n_labeled : n_labeled + n_unlabeled]
    return _year_split_from_indices(
        x_all,
        y_all,
        labeled_idx,
        unlabeled_idx,
        test_idx,
        dataset_name=dataset_name,
    )


def _make_split(cfg: YearRealDataConfig) -> YearSplit:
    x_all, y_all, dataset_name, g, n_total = _load_year_xy_for_split(cfg)
    train_pool_idx, test_idx = _year_train_test_indices(cfg, n_total=n_total, g=g)

    train_perm = train_pool_idx[torch.randperm(train_pool_idx.shape[0], generator=g)]
    labeled_idx = train_perm[: cfg.n_labeled]
    unlabeled_idx = train_perm[cfg.n_labeled : cfg.n_labeled + cfg.n_unlabeled]

    return _year_split_from_indices(
        x_all,
        y_all,
        labeled_idx,
        unlabeled_idx,
        test_idx,
        dataset_name=dataset_name,
    )


def _subsample_pair(
    x: Tensor,
    y: Tensor,
    fraction: float,
    *,
    subsample_seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Subsample the first dimension without replacement.

    Uses a random subset when ``fraction < 1`` so unlabeled-fraction sweeps are not
    biased toward a fixed prefix of the (already permuted) unlabeled tensor order.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    n = int(x.shape[0])
    count = max(1, int(round(fraction * n)))
    if count >= n:
        return x, y
    if subsample_seed is not None:
        g = torch.Generator(device=x.device)
        g.manual_seed(int(subsample_seed) & 0x7FFF_FFFF)
        perm = torch.randperm(n, generator=g, device=x.device)
    else:
        perm = torch.randperm(n, device=x.device)
    pick = perm[:count]
    return x[pick], y[pick]


def _augment_fn(scale: float, feature_drop_prob: float, feature_mix_prob: float):
    def _augment(x: Tensor) -> Tensor:
        noisy = x + scale * torch.randn_like(x)
        if feature_mix_prob > 0.0 and noisy.shape[0] > 1:
            perm = torch.randperm(noisy.shape[0], device=noisy.device)
            mixed = noisy[perm]
            mix_mask = torch.rand_like(noisy).lt(feature_mix_prob)
            noisy = torch.where(mix_mask, mixed, noisy)
        if feature_drop_prob <= 0.0:
            return noisy
        keep = torch.rand_like(noisy).ge(feature_drop_prob).to(noisy.dtype)
        return noisy * keep

    return _augment


def _supervised_loss(model: TabularGaussianRegressor, x: Tensor, y: Tensor) -> Tensor:
    mean, log_var = model(x)
    return torch.nn.functional.gaussian_nll_loss(mean, y, torch.exp(log_var).clamp_min(1e-6))


def _sorted_y_reference(y_labeled: Tensor) -> Tensor:
    return torch.sort(y_labeled.reshape(-1).float())[0]


def _y_to_rank_buckets(y_query: Tensor, sorted_ref: Tensor, n_buckets: int) -> Tensor:
    """Map scalar targets to ``[0, n_buckets-1]`` via normalized rank in ``sorted_ref``."""
    s = sorted_ref
    n = int(s.numel())
    k = max(2, int(n_buckets))
    if n <= 0:
        raise ValueError("sorted_ref must be non-empty")
    pos = torch.searchsorted(s, y_query.reshape(-1).float(), right=True).clamp(0, n)
    pos = pos.clamp(min=1)
    return ((pos.float() - 1.0) / float(max(n - 1, 1)) * float(k)).long().clamp(0, k - 1)


def _build_rankup_from_teacher(
    teacher: TabularGaussianRegressor,
    *,
    n_buckets: int,
) -> TabularGaussianRankUpRegressor:
    in_dim = int(teacher.backbone[0].in_features)
    hidden = int(teacher.backbone[0].out_features)
    drop_p = float(teacher.backbone[2].p)
    student = TabularGaussianRankUpRegressor(in_dim, hidden, drop_p, n_buckets)
    student.backbone.load_state_dict(teacher.backbone.state_dict())
    student.mean_head.load_state_dict(teacher.mean_head.state_dict())
    student.log_var_head.load_state_dict(teacher.log_var_head.state_dict())
    return student


def _predictive_batch(model_: nn.Module, x: Tensor) -> PredictiveBatch:
    mean, log_var = cast(TabularGaussianRegressor, model_)(x)
    return PredictiveBatch(mean=mean, std=torch.exp(0.5 * log_var))


def _build_loaders(
    x_labeled: Tensor,
    y_labeled: Tensor,
    x_unlabeled: Tensor,
    *,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor]]]:
    nw = max(0, int(num_workers))
    labeled_loader = DataLoader(
        TensorDataset(x_labeled, y_labeled),
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        persistent_workers=nw > 0,
    )
    unlabeled_loader = DataLoader(
        TensorDataset(x_unlabeled),
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        persistent_workers=nw > 0,
    )
    return labeled_loader, unlabeled_loader


def _train_supervised_teacher(
    cfg: YearRealDataConfig,
    x_labeled: Tensor,
    y_labeled: Tensor,
    *,
    input_dim: int,
) -> TabularGaussianRegressor:
    _training_seed(cfg.seed, 0)
    model = TabularGaussianRegressor(input_dim, cfg.hidden, cfg.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    nw = int(cfg.dataloader_num_workers)
    loader = DataLoader(
        TensorDataset(x_labeled, y_labeled),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=nw,
        persistent_workers=nw > 0,
    )
    model.train()
    for epoch_idx in range(cfg.teacher_epochs):
        _set_epoch_lr(optimizer, cfg=cfg, epoch_idx=epoch_idx, total_epochs=cfg.teacher_epochs)
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = _supervised_loss(model, xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    return model.eval()


def _train_confidence_weighted_student(
    cfg: YearRealDataConfig,
    teacher: TabularGaussianRegressor,
    x_labeled: Tensor,
    y_labeled: Tensor,
    x_unlabeled: Tensor,
) -> tuple[TabularGaussianRegressor, dict[str, float]]:
    _training_seed(cfg.seed, 1)
    student = copy.deepcopy(teacher).train()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    with torch.no_grad():
        teacher_mean, teacher_log_var = teacher(x_unlabeled)
        pseudo_confidence = torch.exp(-0.5 * teacher_log_var).clamp(max=1.0)

    x_all = torch.cat([x_labeled, x_unlabeled], dim=0)
    y_all = torch.cat([y_labeled, teacher_mean.detach()], dim=0)
    weights_all = torch.cat([torch.zeros_like(y_labeled), pseudo_confidence], dim=0)

    nw = int(cfg.dataloader_num_workers)
    train_loader = DataLoader(
        TensorDataset(x_all, y_all, weights_all),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=nw,
        persistent_workers=nw > 0,
    )
    n_labeled = x_labeled.shape[0]

    for epoch_idx in range(cfg.student_epochs):
        _set_epoch_lr(optimizer, cfg=cfg, epoch_idx=epoch_idx, total_epochs=cfg.student_epochs)
        for xb, yb, wb in train_loader:
            optimizer.zero_grad()
            mean, log_var = student(xb)
            var = torch.exp(log_var).clamp_min(1e-6)
            labeled_mask = wb.reshape(-1) == 0.0
            unlabeled_mask = ~labeled_mask

            labeled_loss = torch.nn.functional.gaussian_nll_loss(
                mean[labeled_mask],
                yb[labeled_mask],
                var[labeled_mask],
            )
            if bool(unlabeled_mask.any().item()):
                pseudo_loss = torch.nn.functional.gaussian_nll_loss(
                    mean[unlabeled_mask],
                    yb[unlabeled_mask],
                    var[unlabeled_mask],
                    reduction="none",
                )
                weight = wb[unlabeled_mask].reshape(-1, 1).clamp_min(0.0)
                blended = (pseudo_loss * weight).sum() / weight.sum().clamp_min(1e-8)
            else:
                blended = torch.zeros((), device=xb.device, dtype=xb.dtype)
            loss = labeled_loss + cfg.pseudo_weight * blended
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

    return student.eval(), {
        "mean_weight": float(pseudo_confidence.mean().item()),
        "mean_disagreement": 0.0,
        "n_labeled": float(n_labeled),
    }


def _train_sage_student(
    cfg: YearRealDataConfig,
    teacher: TabularGaussianRegressor,
    x_labeled: Tensor,
    y_labeled: Tensor,
    x_unlabeled: Tensor,
) -> tuple[TabularGaussianRegressor, dict[str, float]]:
    _training_seed(cfg.seed, 2)
    student = copy.deepcopy(teacher)
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    labeled_loader, unlabeled_loader = _build_loaders(
        x_labeled,
        y_labeled,
        x_unlabeled,
        batch_size=cfg.batch_size,
        num_workers=int(cfg.dataloader_num_workers),
    )
    trainer = SelfAgreementTrainer(
        optimizer=optimizer,
        supervised_loss_fn=lambda model_, x, y: _supervised_loss(
            cast(TabularGaussianRegressor, model_), x, y
        ),
        predictive_batch_fn=_predictive_batch,
        augment_fn=_augment_fn(
            cfg.unlabeled_noise,
            cfg.feature_drop_prob,
            cfg.feature_mix_prob,
        ),
        n_views=cfg.n_views,
        tau=cfg.tau,
        agreement_weight=cfg.agreement_weight,
        ema_decay=cfg.ema_decay,
        weight_power=cfg.weight_power,
        hard_weight_threshold=cfg.hard_weight_threshold,
        batch_relative_mode=cfg.sage_batch_relative_mode,
        batch_trust_top_k=cfg.sage_batch_trust_top_k,
    )
    history = trainer.fit(
        student,
        labeled_loader,
        unlabeled_loader,
        epochs=cfg.student_epochs,
        lr_schedule=cfg.lr_schedule,
        lr_min=cfg.lr_min,
    )
    return student.eval(), {
        "mean_weight": float(history["mean_weight"][-1]),
        "mean_disagreement": float(history["mean_disagreement"][-1]),
        "n_labeled": float(x_labeled.shape[0]),
    }


def _train_rankup_reg_student(
    cfg: YearRealDataConfig,
    teacher: TabularGaussianRegressor,
    x_labeled: Tensor,
    y_labeled: Tensor,
    x_unlabeled: Tensor,
) -> tuple[TabularGaussianRankUpRegressor, dict[str, float]]:
    """RankUp-style SSL: auxiliary quantile buckets + self-training (Huang, Fu, Tsao; NeurIPS 2024).

    Reference: https://arxiv.org/abs/2410.22124 — joint Gaussian NLL with an auxiliary bucket
    classifier; unlabeled terms use frozen-teacher Gaussian pseudo-targets and buckets induced
    from the teacher mean under the labeled marginal (rank bins), with a fixed minimum precision
    mask (self-training gate).
    """
    _training_seed(cfg.seed, 11)
    k0 = int(cfg.rankup_n_buckets)
    k = min(max(2, k0), max(2, int(x_labeled.shape[0]) // 2))
    sorted_ref = _sorted_y_reference(y_labeled)
    student = _build_rankup_from_teacher(teacher, n_buckets=k).train()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    labeled_loader, unlabeled_loader = _build_loaders(
        x_labeled,
        y_labeled,
        x_unlabeled,
        batch_size=cfg.batch_size,
        num_workers=int(cfg.dataloader_num_workers),
    )
    weak_scale = 0.35 * float(cfg.unlabeled_noise)
    strong_aug = _augment_fn(
        cfg.unlabeled_noise,
        cfg.feature_drop_prob,
        cfg.feature_mix_prob,
    )
    lambda_u = float(cfg.pseudo_weight)
    lambda_aux = float(cfg.rankup_aux_weight)
    tau_prec = float(cfg.rankup_min_teacher_precision)
    accept_sum = 0.0
    accept_steps = 0

    for epoch_idx in range(cfg.student_epochs):
        _set_epoch_lr(optimizer, cfg=cfg, epoch_idx=epoch_idx, total_epochs=cfg.student_epochs)
        ul_iter = iter(unlabeled_loader)
        for xb_l, yb_l in labeled_loader:
            try:
                (xb_u,) = next(ul_iter)
            except StopIteration:
                ul_iter = iter(unlabeled_loader)
                (xb_u,) = next(ul_iter)
            optimizer.zero_grad()
            mean_l, lv_l, blogits_l = student(xb_l)
            loss_sup = F.gaussian_nll_loss(
                mean_l,
                yb_l,
                torch.exp(lv_l).clamp_min(1e-6),
            )
            b_tgt = _y_to_rank_buckets(yb_l, sorted_ref, k)
            loss_aux = F.cross_entropy(blogits_l, b_tgt)

            x_w = xb_u + weak_scale * torch.randn_like(xb_u)
            x_s = strong_aug(xb_u)
            with torch.no_grad():
                mu_t, lv_t = teacher(x_w)
                prec = torch.exp(-0.5 * lv_t).squeeze(-1)
                pseudo_bucket = _y_to_rank_buckets(mu_t, sorted_ref, k)
            mask = prec >= tau_prec
            accept_sum += float(mask.float().mean().item())
            accept_steps += 1
            mu_s, lv_s, blogits_s = student(x_s)
            nll_u = F.gaussian_nll_loss(
                mu_s,
                mu_t.detach(),
                torch.exp(lv_t).detach().clamp_min(1e-6),
                reduction="none",
            ).squeeze(-1)
            ce_u = F.cross_entropy(blogits_s, pseudo_bucket, reduction="none")
            if bool(mask.any().item()):
                w = mask.float()
                loss_u = (nll_u * w).sum() / w.sum().clamp_min(1.0)
                loss_b = (ce_u * w).sum() / w.sum().clamp_min(1.0)
            else:
                loss_u = nll_u.mean() * 0.0
                loss_b = ce_u.mean() * 0.0
            loss = loss_sup + lambda_aux * loss_aux + lambda_u * (loss_u + lambda_aux * loss_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

    return student.eval(), {
        "mean_weight": float(accept_sum / max(accept_steps, 1)),
        "mean_disagreement": 0.0,
        "n_labeled": float(x_labeled.shape[0]),
    }


def _train_padaptive_pseudolabel_student(
    cfg: YearRealDataConfig,
    teacher: TabularGaussianRegressor,
    x_labeled: Tensor,
    y_labeled: Tensor,
    x_unlabeled: Tensor,
) -> tuple[TabularGaussianRegressor, dict[str, float]]:
    """PabLO-style batchwise self-training on Gaussian precision (Harit et al.; ICML 2025).

    Reference: https://openreview.net/forum?id=w4c5bLkhsz — we adapt the *idea* of learning a
    pseudo-label acceptance gate from batch statistics: accept teacher pseudo-labels whose
    precision exceeds the ``pablo_precision_quantile`` empirical quantile of the batch, then
    apply Gaussian NLL on strong views (tabular noise / feature dropout augmentations).
    """
    _training_seed(cfg.seed, 12)
    student = copy.deepcopy(teacher).train()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    labeled_loader, unlabeled_loader = _build_loaders(
        x_labeled,
        y_labeled,
        x_unlabeled,
        batch_size=cfg.batch_size,
        num_workers=int(cfg.dataloader_num_workers),
    )
    weak_scale = 0.45 * float(cfg.unlabeled_noise)
    strong_aug = _augment_fn(
        cfg.unlabeled_noise,
        cfg.feature_drop_prob,
        cfg.feature_mix_prob,
    )
    lambda_u = float(cfg.pseudo_weight)
    q = float(cfg.pablo_precision_quantile)
    q = min(max(q, 0.05), 0.95)
    ema_m = float(cfg.pablo_tau_ema_momentum)
    ema_m = min(max(ema_m, 0.0), 0.999)
    tau_state: Tensor | None = None
    accept_sum = 0.0
    accept_steps = 0

    for epoch_idx in range(cfg.student_epochs):
        _set_epoch_lr(optimizer, cfg=cfg, epoch_idx=epoch_idx, total_epochs=cfg.student_epochs)
        ul_iter = iter(unlabeled_loader)
        for xb_l, yb_l in labeled_loader:
            try:
                (xb_u,) = next(ul_iter)
            except StopIteration:
                ul_iter = iter(unlabeled_loader)
                (xb_u,) = next(ul_iter)
            optimizer.zero_grad()
            loss_sup = _supervised_loss(student, xb_l, yb_l)
            x_w = xb_u + weak_scale * torch.randn_like(xb_u)
            x_s = strong_aug(xb_u)
            with torch.no_grad():
                mu_w, lv_w = teacher(x_w)
                prec = torch.exp(-0.5 * lv_w).reshape(-1)
                tau_b = torch.quantile(prec, q)
                if 0.0 < ema_m < 1.0:
                    if tau_state is None:
                        tau_state = tau_b.detach()
                    else:
                        tau_state = ema_m * tau_state + (1.0 - ema_m) * tau_b.detach()
                    tau_eff = tau_state
                else:
                    tau_eff = tau_b
                mask = prec >= tau_eff
            accept_sum += float(mask.float().mean().item())
            accept_steps += 1
            mu_s, lv_s = student(x_s)
            nll_u = F.gaussian_nll_loss(
                mu_s,
                mu_w.detach(),
                torch.exp(lv_w).detach().clamp_min(1e-6),
                reduction="none",
            ).squeeze(-1)
            if bool(mask.any().item()):
                loss_u = (nll_u * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            else:
                loss_u = nll_u.mean() * 0.0
            loss = loss_sup + lambda_u * loss_u
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

    return student.eval(), {
        "mean_weight": float(accept_sum / max(accept_steps, 1)),
        "mean_disagreement": 0.0,
        "n_labeled": float(x_labeled.shape[0]),
    }


def _train_mean_teacher_student(
    cfg: YearRealDataConfig,
    teacher: TabularGaussianRegressor,
    x_labeled: Tensor,
    y_labeled: Tensor,
    x_unlabeled: Tensor,
) -> tuple[TabularGaussianRegressor, dict[str, float]]:
    """EMA teacher + consistency on Gaussian mean (standard Mean Teacher for regression)."""
    _training_seed(cfg.seed, 3)
    student = copy.deepcopy(teacher).train()
    teacher_ema = copy.deepcopy(teacher).eval()
    for param in teacher_ema.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    labeled_loader, unlabeled_loader = _build_loaders(
        x_labeled,
        y_labeled,
        x_unlabeled,
        batch_size=cfg.batch_size,
        num_workers=int(cfg.dataloader_num_workers),
    )
    augment = _augment_fn(
        cfg.unlabeled_noise,
        cfg.feature_drop_prob,
        cfg.feature_mix_prob,
    )
    decay = float(cfg.ema_decay)
    lambda_u = float(cfg.pseudo_weight)

    for epoch_idx in range(cfg.student_epochs):
        _set_epoch_lr(optimizer, cfg=cfg, epoch_idx=epoch_idx, total_epochs=cfg.student_epochs)
        ul_iter = iter(unlabeled_loader)
        for xb_l, yb_l in labeled_loader:
            try:
                (xb_u,) = next(ul_iter)
            except StopIteration:
                ul_iter = iter(unlabeled_loader)
                (xb_u,) = next(ul_iter)
            optimizer.zero_grad()
            loss_sup = _supervised_loss(student, xb_l, yb_l)
            mean_s, _ = student(xb_u)
            with torch.no_grad():
                mean_t, _ = teacher_ema(augment(xb_u))
            loss_u = F.mse_loss(mean_s, mean_t)
            loss = loss_sup + lambda_u * loss_u
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()
            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher_ema.parameters(), strict=True):
                    pt.data.mul_(decay).add_(ps.data, alpha=1.0 - decay)

    return student.eval(), {
        "mean_weight": lambda_u,
        "mean_disagreement": 0.0,
        "n_labeled": float(x_labeled.shape[0]),
    }


def _train_pi_model_student(
    cfg: YearRealDataConfig,
    teacher: TabularGaussianRegressor,
    x_labeled: Tensor,
    y_labeled: Tensor,
    x_unlabeled: Tensor,
) -> tuple[TabularGaussianRegressor, dict[str, float]]:
    """Simple consistency baseline (Pi-model style, no EMA teacher)."""
    _training_seed(cfg.seed, 4)
    student = copy.deepcopy(teacher).train()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    labeled_loader, unlabeled_loader = _build_loaders(
        x_labeled,
        y_labeled,
        x_unlabeled,
        batch_size=cfg.batch_size,
        num_workers=int(cfg.dataloader_num_workers),
    )
    augment = _augment_fn(
        cfg.unlabeled_noise,
        cfg.feature_drop_prob,
        cfg.feature_mix_prob,
    )
    lambda_u = float(cfg.pseudo_weight)

    for epoch_idx in range(cfg.student_epochs):
        _set_epoch_lr(optimizer, cfg=cfg, epoch_idx=epoch_idx, total_epochs=cfg.student_epochs)
        ul_iter = iter(unlabeled_loader)
        for xb_l, yb_l in labeled_loader:
            try:
                (xb_u,) = next(ul_iter)
            except StopIteration:
                ul_iter = iter(unlabeled_loader)
                (xb_u,) = next(ul_iter)
            optimizer.zero_grad()
            loss_sup = _supervised_loss(student, xb_l, yb_l)
            mean_a, _ = student(augment(xb_u))
            mean_b, _ = student(augment(xb_u))
            loss_u = F.mse_loss(mean_a, mean_b)
            loss = loss_sup + lambda_u * loss_u
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

    return student.eval(), {
        "mean_weight": lambda_u,
        "mean_disagreement": 0.0,
        "n_labeled": float(x_labeled.shape[0]),
    }


def _evaluate_model(model: nn.Module, x_test: Tensor, y_test: Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        raw = model(x_test)
        if isinstance(raw, tuple) and len(raw) == 3:
            mean, log_var, _ = raw
        else:
            mean, log_var = cast(tuple[Tensor, Tensor], raw)
        std = torch.exp(0.5 * log_var).clamp_min(1e-4)
        var = std.square()

    # High-dimensional tabular (e.g. TabReD) can sporadically yield non-finite Gaussian
    # parameters; calibration_score rejects NaN quantiles. Keep metrics on finite rows only.
    ok = (torch.isfinite(mean) & torch.isfinite(std) & torch.isfinite(y_test)).squeeze(-1)
    if not bool(ok.any().item()):
        raise ValueError(
            "Gaussian evaluation produced no finite (mean, std, y) rows; "
            "check inputs for NaN/Inf or reduce LR / stabilize features."
        )
    if not bool(ok.all().item()):
        mean = mean[ok]
        std = std[ok]
        var = var[ok]
        y_test = y_test[ok]

    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=mean.device, dtype=mean.dtype),
        torch.tensor(1.0, device=mean.device, dtype=mean.dtype),
    )
    z = normal.icdf(torch.tensor(0.95, device=mean.device, dtype=mean.dtype))
    lower = mean - z * std
    upper = mean + z * std
    cov90 = prediction_interval_coverage_probability(lower, upper, y_test, alpha=0.1)
    calibration = calibration_score(y_test, mean, std)
    calib_mae = calibration["mean_absolute_calibration_error"]
    calib_mae_f = float(calib_mae.item()) if torch.is_tensor(calib_mae) else float(calib_mae)

    return {
        "RMSE": float(torch.sqrt(torch.mean((mean - y_test).square())).item()),
        "NLL": gaussian_nll(mean, y_test, var, reduction="mean"),
        "CRPS": crps_gaussian(mean, y_test, std, reduction="mean"),
        "Cov90": float(cov90),
        "Width90": float(torch.mean(upper - lower).item()),
        "CoverageGap90": abs(float(cov90) - 0.9),
        "CalibMAE": calib_mae_f,
    }


def _collect_unlabeled_diagnostics(
    cfg: YearRealDataConfig,
    split: YearSplit,
    fraction: float,
) -> dict[str, Tensor]:
    x_unlabeled, y_unlabeled_true = _subsample_pair(
        split.x_unlabeled,
        split.y_unlabeled_true,
        fraction,
        subsample_seed=cfg.seed * 1_000_003 + int(round(fraction * 1_000_000)),
    )
    teacher = _train_supervised_teacher(
        cfg,
        split.x_labeled,
        split.y_labeled,
        input_dim=split.n_features,
    )

    with torch.no_grad():
        teacher_mean, teacher_log_var = teacher(x_unlabeled)
        pseudo_confidence = torch.exp(-0.5 * teacher_log_var).clamp(max=1.0)
        views = [
            _predictive_batch(
                teacher,
                (
                    x_unlabeled
                    if idx == 0
                    else _augment_fn(
                        cfg.unlabeled_noise,
                        cfg.feature_drop_prob,
                        cfg.feature_mix_prob,
                    )(x_unlabeled)
                ),
            )
            for idx in range(cfg.n_views)
        ]
        disagreement = cast(
            Tensor,
            torch.as_tensor(predictive_agreement_score(views, n_support=96, reduction="none")),
        )
        agreement_weight = disagreement_to_weight(
            disagreement,
            cfg.tau,
            power=cfg.weight_power,
            hard_weight_threshold=cfg.hard_weight_threshold,
        )

    abs_error = (teacher_mean - y_unlabeled_true).abs().reshape(-1)
    return {
        "abs_error": abs_error,
        "pseudo_confidence": pseudo_confidence.reshape(-1),
        "agreement_weight": agreement_weight.reshape(-1),
        "disagreement": disagreement.reshape(-1),
    }


def _run_fraction(
    cfg: YearRealDataConfig, split: YearSplit, fraction: float
) -> list[dict[str, object]]:
    x_unlabeled, y_unlabeled_true = _subsample_pair(
        split.x_unlabeled,
        split.y_unlabeled_true,
        fraction,
        subsample_seed=cfg.seed * 1_000_003 + int(round(fraction * 1_000_000)),
    )
    teacher, teacher_s = timed_call(
        _train_supervised_teacher,
        cfg,
        split.x_labeled,
        split.y_labeled,
        input_dim=split.n_features,
    )
    rows: list[dict[str, object]] = []

    specs = [
        (
            "SupervisedOnly",
            lambda: (teacher, {"mean_weight": 0.0, "mean_disagreement": 0.0}, 0.0),
        ),
        (
            "MeanTeacher",
            lambda: (
                *timed_call(
                    _train_mean_teacher_student,
                    cfg,
                    teacher,
                    split.x_labeled,
                    split.y_labeled,
                    x_unlabeled,
                ),
            ),
        ),
        (
            "PiModelConsistency",
            lambda: (
                *timed_call(
                    _train_pi_model_student,
                    cfg,
                    teacher,
                    split.x_labeled,
                    split.y_labeled,
                    x_unlabeled,
                ),
            ),
        ),
        (
            "ConfidenceWeightedPseudoLabel",
            lambda: (
                *timed_call(
                    _train_confidence_weighted_student,
                    cfg,
                    teacher,
                    split.x_labeled,
                    split.y_labeled,
                    x_unlabeled,
                ),
            ),
        ),
        (
            "RankUp",
            lambda: (
                *timed_call(
                    _train_rankup_reg_student,
                    cfg,
                    teacher,
                    split.x_labeled,
                    split.y_labeled,
                    x_unlabeled,
                ),
            ),
        ),
        (
            "PabLOPseudo",
            lambda: (
                *timed_call(
                    _train_padaptive_pseudolabel_student,
                    cfg,
                    teacher,
                    split.x_labeled,
                    split.y_labeled,
                    x_unlabeled,
                ),
            ),
        ),
        (
            "SAGE-Reg",
            lambda: (
                *timed_call(
                    _train_sage_student,
                    cfg,
                    teacher,
                    split.x_labeled,
                    split.y_labeled,
                    x_unlabeled,
                ),
            ),
        ),
    ]

    for method, builder in specs:
        if method == "SupervisedOnly":
            model, meta, train_s = builder()
        else:
            (model, meta), train_s = builder()
        metrics, eval_s = timed_call(_evaluate_model, model, split.x_test, split.y_test)
        rows.append(
            {
                "Seed": int(cfg.seed),
                "Method": method,
                "Dataset": split.dataset_name,
                "UnlabeledFraction": float(fraction),
                **metrics,
                "MeanWeight": float(meta["mean_weight"]),
                "MeanDisagreement": float(meta["mean_disagreement"]),
                "train_s": float(teacher_s + train_s if method != "SupervisedOnly" else teacher_s),
                "eval_s": float(eval_s),
            }
        )
    return rows


def run_benchmark_on_split(cfg: YearRealDataConfig, split: YearSplit) -> list[dict[str, object]]:
    print(
        "[self_agreement_realdata_year] run_benchmark_on_split "
        f"seed={cfg.seed} dataset={split.dataset_name} "
        f"n_labeled={int(split.x_labeled.shape[0])} n_unlabeled={int(split.x_unlabeled.shape[0])} "
        f"n_test={int(split.x_test.shape[0])} unlabeled_fractions={cfg.unlabeled_fractions}",
        flush=True,
    )
    rows: list[dict[str, object]] = []
    for fraction in cfg.unlabeled_fractions:
        print(
            f"[self_agreement_realdata_year]   fraction={float(fraction):.4f} (training all methods)",
            flush=True,
        )
        rows.extend(_run_fraction(cfg, split, fraction))
    print(f"[self_agreement_realdata_year] done; {len(rows)} rows", flush=True)
    return rows


def run_benchmark(cfg: YearRealDataConfig) -> list[dict[str, object]]:
    return run_benchmark_on_split(cfg, _make_split(cfg))


def main(
    cfg: YearRealDataConfig | None = None,
    *,
    output_csv: str | None = None,
    figure_path: str | None = None,
    performance_figure_path: str | None = None,
    calibration_figure_path: str | None = None,
    diagnostic_figure_path: str | None = None,
    summary_json_path: str | None = None,
) -> list[dict[str, object]]:
    resolved = YearRealDataConfig() if cfg is None else cfg
    print(
        "[self_agreement_realdata_year] main() "
        f"seed={resolved.seed} nl/nu/nt={resolved.n_labeled}/{resolved.n_unlabeled}/{resolved.n_test} "
        f"epochs={resolved.teacher_epochs}/{resolved.student_epochs} batch={resolved.batch_size}",
        flush=True,
    )
    rows = run_benchmark(resolved)

    perf_path = performance_figure_path or figure_path
    calib_path = calibration_figure_path
    if perf_path is not None and calib_path is None:
        perf = Path(perf_path)
        calib_path = str(perf.with_name(f"{perf.stem}_calibration{perf.suffix or '.png'}"))

    print_fairness_notes(
        title="SAGE-Reg Real-Data Benchmark (YearPredictionMSD)",
        seed_policy=f"single fixed seed ({resolved.seed}) across all methods/fractions",
        train_budget=(
            f"shared Gaussian tabular backbone with {resolved.teacher_epochs} teacher epochs "
            f"and {resolved.student_epochs} student epochs"
        ),
        metric_policy="RMSE, NLL, CRPS, 90% interval quality, and Gaussian calibration summary",
    )
    print_comparison_summary(
        "SAGE-Reg Real-Data Benchmark (YearPredictionMSD)",
        rows,
        metric_order=[
            "Seed",
            "Method",
            "UnlabeledFraction",
            "RMSE",
            "NLL",
            "CRPS",
            "Cov90",
            "CalibMAE",
            "MeanWeight",
            "MeanDisagreement",
            "train_s",
        ],
    )

    if output_csv:
        out = _write_csv(output_csv, rows)
        print(f"\nWrote CSV: {out}")
    if perf_path:
        out = _plot_performance(perf_path, rows)
        print(f"Wrote performance figure: {out}")
    if calib_path:
        out = _plot_calibration(calib_path, rows)
        print(f"Wrote calibration figure: {out}")
    if diagnostic_figure_path:
        split = _make_split(resolved)
        diagnostic_fraction = max(resolved.unlabeled_fractions)
        diagnostics = _collect_unlabeled_diagnostics(resolved, split, float(diagnostic_fraction))
        diagnostic_summary = _diagnostic_summary(diagnostics)
        out = _plot_unlabeled_diagnostics(
            diagnostic_figure_path,
            diagnostics,
            fraction=float(diagnostic_fraction),
        )
        print(f"Wrote diagnostic figure: {out}")
        print(_format_diagnostic_summary(diagnostic_summary, fraction=float(diagnostic_fraction)))
    if summary_json_path:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/benchmarks/self_agreement_realdata_year.py",
            task="self-agreement real-data benchmark on OpenML/UCI YearPredictionMSD",
            config=resolved,
            rows=rows,
            notes=[
                "Uses OpenML/UCI YearPredictionMSD when no local dataset_path is provided.",
                "Gaussian head: supervised, Mean Teacher (EMA consistency), confidence-weighted pseudo-labeling, and SAGE-Reg.",
                "Includes a Pi-model consistency baseline (no EMA teacher) for external SSL comparison breadth.",
                "Use dataset_path for offline or smoke-test runs; local CSV/Parquet paths bypass network access.",
                "Optional diagnostic figure compares confidence and SAGE agreement-weight rankings against pseudo-label error on the unlabeled pool.",
            ],
        )
        print(f"Wrote summary JSON: {out}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the real-data SAGE-Reg benchmark on year.")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--figure-path", type=str, default="")
    parser.add_argument("--performance-figure-path", type=str, default="")
    parser.add_argument("--calibration-figure-path", type=str, default="")
    parser.add_argument("--diagnostic-figure-path", type=str, default="")
    parser.add_argument("--summary-json-path", type=str, default="")
    parser.add_argument(
        "--seed", type=int, default=YearRealDataConfig.seed, help="Global split/train seed."
    )
    parser.add_argument("--n-labeled", type=int, default=YearRealDataConfig.n_labeled)
    parser.add_argument("--n-unlabeled", type=int, default=YearRealDataConfig.n_unlabeled)
    parser.add_argument("--n-test", type=int, default=YearRealDataConfig.n_test)
    parser.add_argument("--batch-size", type=int, default=YearRealDataConfig.batch_size)
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=YearRealDataConfig.dataloader_num_workers,
        help="DataLoader worker processes (0 = main process only).",
    )
    parser.add_argument("--teacher-epochs", type=int, default=YearRealDataConfig.teacher_epochs)
    parser.add_argument("--student-epochs", type=int, default=YearRealDataConfig.student_epochs)
    parser.add_argument(
        "--openml-data-id",
        type=int,
        default=None,
        help="Fetch this OpenML regression dataset (e.g. 42225 diamonds) when no dataset/cache.",
    )
    parser.add_argument(
        "--openml-dataset-name",
        type=str,
        default="",
        help="Alternatively fetch by OpenML name; ignored if --openml-data-id is set.",
    )
    parser.add_argument("--openml-version", type=int, default=YearRealDataConfig.openml_version)
    parser.add_argument(
        "--max-dataset-rows",
        type=int,
        default=None,
        help="Random subsample cap before splitting (large OpenML dumps).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=YearRealDataConfig.weight_decay,
        help="Adam L2 penalty on all training phases (teacher, student, baselines).",
    )
    parser.add_argument("--lr", type=float, default=YearRealDataConfig.lr, help="Base Adam LR.")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default=YearRealDataConfig.lr_schedule,
        help="Per-epoch schedule: constant or cosine (cosine uses --lr-min floor).",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=YearRealDataConfig.lr_min,
        help="Minimum LR when --lr-schedule cosine.",
    )
    parser.add_argument(
        "--sage-batch-relative-mode",
        type=str,
        default="",
        help="Optional SAGE disagreement mode, e.g. zscore (see SelfAgreementTrainer).",
    )
    parser.add_argument(
        "--sage-batch-trust-top-k",
        type=int,
        default=None,
        help="Optional SAGE batch trust top-k (positive int).",
    )
    parser.add_argument(
        "--unlabeled-fractions",
        type=float,
        nargs="+",
        default=list(YearRealDataConfig.unlabeled_fractions),
        help=(
            "Fractions of the unlabeled pool to use per sub-run (e.g. 0.25 0.5 1.0). "
            "Useful for semi-sup curves without changing n_labeled/n_unlabeled splits."
        ),
    )
    parser.add_argument("--rankup-n-buckets", type=int, default=YearRealDataConfig.rankup_n_buckets)
    parser.add_argument(
        "--rankup-aux-weight",
        type=float,
        default=YearRealDataConfig.rankup_aux_weight,
        help="RankUp auxiliary CE weight (labeled + partial unlabeled).",
    )
    parser.add_argument(
        "--rankup-min-teacher-precision",
        type=float,
        default=YearRealDataConfig.rankup_min_teacher_precision,
        help="Min teacher precision to keep an unlabeled pseudo-label for RankUp.",
    )
    parser.add_argument(
        "--pablo-precision-quantile",
        type=float,
        default=YearRealDataConfig.pablo_precision_quantile,
        help="Batch quantile q for tau=quantile(precision,q); accept prec>=tau (PabLO-style).",
    )
    parser.add_argument(
        "--pablo-tau-ema-momentum",
        type=float,
        default=YearRealDataConfig.pablo_tau_ema_momentum,
        help="EMA momentum for tau (0 disables).",
    )
    args = parser.parse_args()

    lr_schedule = str(args.lr_schedule).strip().lower()
    if lr_schedule not in {"constant", "cosine"}:
        raise SystemExit("--lr-schedule must be 'constant' or 'cosine'")
    cfg = YearRealDataConfig(
        dataset_path=args.dataset_path or None,
        cache_path=args.cache_path or None,
        allow_download=not args.no_download,
        seed=int(args.seed),
        n_labeled=args.n_labeled,
        n_unlabeled=args.n_unlabeled,
        n_test=args.n_test,
        batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
        openml_data_id=args.openml_data_id,
        openml_dataset_name=args.openml_dataset_name or None,
        openml_version=args.openml_version,
        max_dataset_rows=args.max_dataset_rows,
        weight_decay=args.weight_decay,
        lr=args.lr,
        lr_schedule=lr_schedule,
        lr_min=args.lr_min,
        sage_batch_relative_mode=args.sage_batch_relative_mode or None,
        sage_batch_trust_top_k=args.sage_batch_trust_top_k,
        unlabeled_fractions=tuple(args.unlabeled_fractions),
        rankup_n_buckets=int(args.rankup_n_buckets),
        rankup_aux_weight=float(args.rankup_aux_weight),
        rankup_min_teacher_precision=float(args.rankup_min_teacher_precision),
        pablo_precision_quantile=float(args.pablo_precision_quantile),
        pablo_tau_ema_momentum=float(args.pablo_tau_ema_momentum),
    )
    main(
        cfg,
        output_csv=args.output_csv or None,
        figure_path=args.figure_path or None,
        performance_figure_path=args.performance_figure_path or None,
        calibration_figure_path=args.calibration_figure_path or None,
        diagnostic_figure_path=args.diagnostic_figure_path or None,
        summary_json_path=args.summary_json_path or None,
    )
