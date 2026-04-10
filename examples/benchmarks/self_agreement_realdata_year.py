"""Real-data SSL benchmark for SAGE-Reg on OpenML/UCI YearPredictionMSD."""

from __future__ import annotations

import argparse
import copy
import csv
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
from comparison_utils import (  # noqa: E402
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from sklearn.datasets import fetch_openml  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

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


@dataclass(frozen=True)
class YearRealDataConfig:
    seed: int = 260408
    dataset_path: str | None = None
    allow_download: bool = True
    cache_path: str | None = None
    target_column: str = "target"
    canonical_test_size: int = 51_630
    n_labeled: int = 4_096
    n_unlabeled: int = 65_536
    n_test: int = 32_768
    hidden: int = 128
    teacher_epochs: int = 24
    student_epochs: int = 24
    batch_size: int = 512
    lr: float = 2e-3
    dropout: float = 0.10
    unlabeled_noise: float = 0.03
    feature_drop_prob: float = 0.0
    tau: float = 0.18
    agreement_weight: float = 0.5
    pseudo_weight: float = 0.8
    ema_decay: float = 0.96
    n_views: int = 4
    weight_power: float = 1.0
    hard_weight_threshold: float | None = None
    unlabeled_fractions: tuple[float, ...] = (0.25, 0.5, 1.0)


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


def _training_seed(seed: int, offset: int) -> None:
    set_comparison_seed(seed + offset)


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


def _load_dataset_frame(cfg: YearRealDataConfig) -> tuple[pd.DataFrame, str]:
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


def _make_split(cfg: YearRealDataConfig) -> YearSplit:
    set_comparison_seed(cfg.seed)
    frame, dataset_name = _load_dataset_frame(cfg)
    if cfg.target_column not in frame.columns:
        raise ValueError(f"target column {cfg.target_column!r} not found in dataset")

    feature_frame = frame.drop(columns=[cfg.target_column]).astype("float32")
    target = frame[cfg.target_column].astype("float32")
    x_all = torch.tensor(feature_frame.to_numpy(copy=True), dtype=torch.float32)
    y_all = torch.tensor(target.to_numpy(copy=True), dtype=torch.float32).unsqueeze(1)

    n_total = x_all.shape[0]
    need = cfg.n_labeled + cfg.n_unlabeled + cfg.n_test
    if need > n_total:
        raise ValueError(f"Requested {need} rows but dataset has {n_total}")

    canonical_like = n_total >= cfg.canonical_test_size + cfg.n_labeled + cfg.n_unlabeled
    g = torch.Generator().manual_seed(cfg.seed)
    if canonical_like:
        train_pool_end = n_total - cfg.canonical_test_size
        train_pool_idx = torch.arange(train_pool_end)
        test_pool_idx = torch.arange(train_pool_end, n_total)
    else:
        perm = torch.randperm(n_total, generator=g)
        test_pool_idx = perm[: cfg.n_test]
        train_pool_idx = perm[cfg.n_test :]

    train_perm = train_pool_idx[torch.randperm(train_pool_idx.shape[0], generator=g)]
    labeled_idx = train_perm[: cfg.n_labeled]
    unlabeled_idx = train_perm[cfg.n_labeled : cfg.n_labeled + cfg.n_unlabeled]

    if canonical_like:
        test_idx = test_pool_idx[: cfg.n_test]
    else:
        test_idx = test_pool_idx

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
        n_features=x_all.shape[1],
    )


def _subsample_pair(x: Tensor, y: Tensor, fraction: float) -> tuple[Tensor, Tensor]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    count = max(1, int(round(fraction * x.shape[0])))
    return x[:count], y[:count]


def _augment_fn(scale: float, feature_drop_prob: float):
    def _augment(x: Tensor) -> Tensor:
        noisy = x + scale * torch.randn_like(x)
        if feature_drop_prob <= 0.0:
            return noisy
        keep = torch.rand_like(noisy).ge(feature_drop_prob).to(noisy.dtype)
        return noisy * keep

    return _augment


def _supervised_loss(model: TabularGaussianRegressor, x: Tensor, y: Tensor) -> Tensor:
    mean, log_var = model(x)
    return torch.nn.functional.gaussian_nll_loss(mean, y, torch.exp(log_var).clamp_min(1e-6))


def _predictive_batch(model_: nn.Module, x: Tensor) -> PredictiveBatch:
    mean, log_var = cast(TabularGaussianRegressor, model_)(x)
    return PredictiveBatch(mean=mean, std=torch.exp(0.5 * log_var))


def _build_loaders(
    x_labeled: Tensor,
    y_labeled: Tensor,
    x_unlabeled: Tensor,
    *,
    batch_size: int,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor]]]:
    labeled_loader = DataLoader(
        TensorDataset(x_labeled, y_labeled),
        batch_size=batch_size,
        shuffle=True,
    )
    unlabeled_loader = DataLoader(
        TensorDataset(x_unlabeled),
        batch_size=batch_size,
        shuffle=True,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(
        TensorDataset(x_labeled, y_labeled),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    model.train()
    for _ in range(cfg.teacher_epochs):
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
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr)

    with torch.no_grad():
        teacher_mean, teacher_log_var = teacher(x_unlabeled)
        pseudo_confidence = torch.exp(-0.5 * teacher_log_var).clamp(max=1.0)

    x_all = torch.cat([x_labeled, x_unlabeled], dim=0)
    y_all = torch.cat([y_labeled, teacher_mean.detach()], dim=0)
    weights_all = torch.cat([torch.zeros_like(y_labeled), pseudo_confidence], dim=0)

    train_loader = DataLoader(
        TensorDataset(x_all, y_all, weights_all),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    n_labeled = x_labeled.shape[0]

    for _ in range(cfg.student_epochs):
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
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader = _build_loaders(
        x_labeled,
        y_labeled,
        x_unlabeled,
        batch_size=cfg.batch_size,
    )
    trainer = SelfAgreementTrainer(
        optimizer=optimizer,
        supervised_loss_fn=lambda model_, x, y: _supervised_loss(
            cast(TabularGaussianRegressor, model_), x, y
        ),
        predictive_batch_fn=_predictive_batch,
        augment_fn=_augment_fn(cfg.unlabeled_noise, cfg.feature_drop_prob),
        n_views=cfg.n_views,
        tau=cfg.tau,
        agreement_weight=cfg.agreement_weight,
        ema_decay=cfg.ema_decay,
        weight_power=cfg.weight_power,
        hard_weight_threshold=cfg.hard_weight_threshold,
    )
    history = trainer.fit(student, labeled_loader, unlabeled_loader, epochs=cfg.student_epochs)
    return student.eval(), {
        "mean_weight": float(history["mean_weight"][-1]),
        "mean_disagreement": float(history["mean_disagreement"][-1]),
        "n_labeled": float(x_labeled.shape[0]),
    }


def _evaluate_model(
    model: TabularGaussianRegressor, x_test: Tensor, y_test: Tensor
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        mean, log_var = model(x_test)
        std = torch.exp(0.5 * log_var).clamp_min(1e-4)
        var = std.square()

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
        split.x_unlabeled, split.y_unlabeled_true, fraction
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
                    else _augment_fn(cfg.unlabeled_noise, cfg.feature_drop_prob)(x_unlabeled)
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
        split.x_unlabeled, split.y_unlabeled_true, fraction
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


def run_benchmark(cfg: YearRealDataConfig) -> list[dict[str, object]]:
    split = _make_split(cfg)
    rows: list[dict[str, object]] = []
    for fraction in cfg.unlabeled_fractions:
        rows.extend(_run_fraction(cfg, split, fraction))
    return rows


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
                "The first real-data pass stays narrow: Gaussian head only, with supervised, confidence-weighted pseudo-labeling, and SAGE-Reg.",
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
    parser.add_argument("--n-labeled", type=int, default=YearRealDataConfig.n_labeled)
    parser.add_argument("--n-unlabeled", type=int, default=YearRealDataConfig.n_unlabeled)
    parser.add_argument("--n-test", type=int, default=YearRealDataConfig.n_test)
    parser.add_argument("--teacher-epochs", type=int, default=YearRealDataConfig.teacher_epochs)
    parser.add_argument("--student-epochs", type=int, default=YearRealDataConfig.student_epochs)
    args = parser.parse_args()

    cfg = YearRealDataConfig(
        dataset_path=args.dataset_path or None,
        cache_path=args.cache_path or None,
        allow_download=not args.no_download,
        n_labeled=args.n_labeled,
        n_unlabeled=args.n_unlabeled,
        n_test=args.n_test,
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
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
