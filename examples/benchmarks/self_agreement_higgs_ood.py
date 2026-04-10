"""Higgs-inspired OOD benchmark for SAGE-Reg.

This benchmark is intentionally narrow:
- Gaussian head only
- supervised vs confidence-weighted pseudo-labeling vs SAGE-Reg
- ID/OOD evaluation with unlabeled-pool weighting diagnostics

If ``dataset_path`` is provided, the script loads a local CSV/Parquet table.
Otherwise it runs on a built-in Higgs-like proxy with systematic covariate shift.
"""

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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from comparison_utils import (  # noqa: E402
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
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
class HiggsOODConfig:
    seed: int = 260409
    dataset_path: str | None = None
    target_column: str = "target"
    ood_score_column: str | None = None
    drop_columns: tuple[str, ...] = ()
    shift_feature_idx: int = 0
    n_train: int = 4_096
    n_unlabeled_id: int = 16_384
    n_unlabeled_ood: int = 16_384
    n_id_test: int = 8_192
    n_ood_test: int = 8_192
    parquet_sample_factor: int = 8
    parquet_max_sample_rows: int = 250_000
    proxy_dim: int = 24
    hidden: int = 128
    teacher_epochs: int = 24
    student_epochs: int = 24
    batch_size: int = 512
    lr: float = 2e-3
    dropout: float = 0.10
    unlabeled_noise: float = 0.04
    feature_drop_prob: float = 0.0
    feature_mix_prob: float = 0.0
    ood_perturb_boost: float = 2.0
    tau: float = 0.18
    agreement_weight: float = 0.5
    pseudo_weight: float = 0.8
    ema_decay: float = 0.96
    n_views: int = 4
    weight_power: float = 1.0
    hard_weight_threshold: float | None = None


@dataclass(frozen=True)
class HiggsOODSplit:
    x_train: Tensor
    y_train: Tensor
    x_unlabeled: Tensor
    y_unlabeled_true: Tensor
    unlabeled_is_ood: Tensor
    x_id_test: Tensor
    y_id_test: Tensor
    x_ood_test: Tensor
    y_ood_test: Tensor
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


def _plot_performance(path: str | Path, rows: list[dict[str, object]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = [str(row["Method"]) for row in rows]
    x = list(range(len(rows)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(x, [float(row["NLL_ID"]) for row in rows], label="ID", alpha=0.85)
    axes[0].bar(x, [float(row["NLL_OOD"]) for row in rows], label="OOD", alpha=0.55)
    axes[0].set_title("NLL by regime")
    axes[0].set_xticks(x, methods, rotation=20, ha="right")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].bar(x, [float(row["RMSE_ID"]) for row in rows], label="ID", alpha=0.85)
    axes[1].bar(x, [float(row["RMSE_OOD"]) for row in rows], label="OOD", alpha=0.55)
    axes[1].set_title("RMSE by regime")
    axes[1].set_xticks(x, methods, rotation=20, ha="right")
    axes[1].legend(loc="best", fontsize=8)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_calibration(path: str | Path, rows: list[dict[str, object]]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = [str(row["Method"]) for row in rows]
    x = list(range(len(rows)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(x, [abs(float(row["Cov90_ID"]) - 0.9) for row in rows], alpha=0.8, label="ID gap")
    axes[0].bar(
        x,
        [abs(float(row["Cov90_OOD"]) - 0.9) for row in rows],
        alpha=0.55,
        label="OOD gap",
    )
    axes[0].set_title("Coverage gap by regime")
    axes[0].set_xticks(x, methods, rotation=20, ha="right")
    axes[0].legend(loc="best", fontsize=8)

    axes[1].bar(x, [float(row["OODUncGap"]) for row in rows], alpha=0.8)
    axes[1].set_title("Mean std(OOD) - std(ID)")
    axes[1].set_xticks(x, methods, rotation=20, ha="right")
    axes[1].set_ylabel("OOD uncertainty gap")

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_local_frame(cfg: HiggsOODConfig) -> tuple[pd.DataFrame, str]:
    if cfg.dataset_path is None:
        raise FileNotFoundError("dataset_path is required for local dataset loading")
    path = Path(cfg.dataset_path)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        total_rows = parquet.metadata.num_rows
        need = (
            cfg.n_train + cfg.n_unlabeled_id + cfg.n_unlabeled_ood + cfg.n_id_test + cfg.n_ood_test
        )
        sample_rows = min(
            total_rows,
            max(
                need,
                min(cfg.parquet_max_sample_rows, need * cfg.parquet_sample_factor),
            ),
        )
        if sample_rows >= total_rows:
            frame = parquet.read().to_pandas()
        else:
            rng = np.random.default_rng(cfg.seed)
            selected_indices = np.sort(rng.choice(total_rows, size=sample_rows, replace=False))
            parts: list[pd.DataFrame] = []
            current_row = 0
            for row_group_index in range(parquet.num_row_groups):
                row_group_rows = parquet.metadata.row_group(row_group_index).num_rows
                in_group = (
                    selected_indices[
                        (selected_indices >= current_row)
                        & (selected_indices < current_row + row_group_rows)
                    ]
                    - current_row
                )
                if in_group.size > 0:
                    row_group = parquet.read_row_group(row_group_index).to_pandas()
                    parts.append(row_group.iloc[in_group])
                current_row += row_group_rows
            frame = pd.concat(parts, ignore_index=True)
    else:
        frame = pd.read_csv(path)
    return frame, str(path)


def _make_proxy_split(cfg: HiggsOODConfig) -> HiggsOODSplit:
    g = torch.Generator().manual_seed(cfg.seed)
    d = cfg.proxy_dim

    n_id_pool = cfg.n_train + cfg.n_unlabeled_id + cfg.n_id_test
    n_ood_pool = cfg.n_unlabeled_ood + cfg.n_ood_test

    x_id = torch.randn(n_id_pool, d, generator=g)
    x_ood = torch.randn(n_ood_pool, d, generator=g)
    x_ood[:, :4] = x_ood[:, :4] + 2.0
    x_ood[:, 4:8] = x_ood[:, 4:8] * 1.5

    weights = torch.linspace(-1.0, 1.0, d).unsqueeze(1)

    def _target(x: Tensor, *, shifted: bool) -> Tensor:
        base = x @ weights + 0.30 * torch.sin(x[:, :1]) - 0.20 * torch.cos(x[:, 1:2])
        if shifted:
            base = base + 0.45 * x[:, :1] - 0.30 * x[:, 2:3]
        noise = (0.12 + (0.18 if shifted else 0.06) * x[:, :1].abs()) * torch.randn(
            x.shape[0], 1, generator=g
        )
        return base + noise

    y_id = _target(x_id, shifted=False)
    y_ood = _target(x_ood, shifted=True)

    x_train = x_id[: cfg.n_train]
    y_train = y_id[: cfg.n_train]
    unlabeled_id_start = cfg.n_train
    unlabeled_id_end = cfg.n_train + cfg.n_unlabeled_id
    x_unlabeled_id = x_id[unlabeled_id_start:unlabeled_id_end]
    y_unlabeled_id = y_id[unlabeled_id_start:unlabeled_id_end]
    x_id_test = x_id[unlabeled_id_end : unlabeled_id_end + cfg.n_id_test]
    y_id_test = y_id[unlabeled_id_end : unlabeled_id_end + cfg.n_id_test]

    x_unlabeled_ood = x_ood[: cfg.n_unlabeled_ood]
    y_unlabeled_ood = y_ood[: cfg.n_unlabeled_ood]
    x_ood_test = x_ood[cfg.n_unlabeled_ood : cfg.n_unlabeled_ood + cfg.n_ood_test]
    y_ood_test = y_ood[cfg.n_unlabeled_ood : cfg.n_unlabeled_ood + cfg.n_ood_test]

    x_unlabeled = torch.cat([x_unlabeled_id, x_unlabeled_ood], dim=0)
    y_unlabeled = torch.cat([y_unlabeled_id, y_unlabeled_ood], dim=0)
    unlabeled_is_ood = torch.cat(
        [
            torch.zeros(cfg.n_unlabeled_id, dtype=torch.bool),
            torch.ones(cfg.n_unlabeled_ood, dtype=torch.bool),
        ]
    )

    x_pool = torch.cat([x_train, x_unlabeled], dim=0)
    x_mean = x_pool.mean(dim=0, keepdim=True)
    x_std = x_pool.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    return HiggsOODSplit(
        x_train=(x_train - x_mean) / x_std,
        y_train=(y_train - y_mean) / y_std,
        x_unlabeled=(x_unlabeled - x_mean) / x_std,
        y_unlabeled_true=(y_unlabeled - y_mean) / y_std,
        unlabeled_is_ood=unlabeled_is_ood,
        x_id_test=(x_id_test - x_mean) / x_std,
        y_id_test=(y_id_test - y_mean) / y_std,
        x_ood_test=(x_ood_test - x_mean) / x_std,
        y_ood_test=(y_ood_test - y_mean) / y_std,
        dataset_name="higgs_proxy_systematic_shift",
        n_features=d,
    )


def _make_local_split(cfg: HiggsOODConfig) -> HiggsOODSplit:
    set_comparison_seed(cfg.seed)
    frame, dataset_name = _load_local_frame(cfg)
    if cfg.target_column not in frame.columns:
        raise ValueError(f"target column {cfg.target_column!r} not found in dataset")

    excluded = [cfg.target_column, *cfg.drop_columns]
    present_excluded = [column for column in excluded if column in frame.columns]
    feature_frame = frame.drop(columns=present_excluded).select_dtypes(include=["number"])
    target = frame[cfg.target_column].astype("float32")
    if feature_frame.empty:
        raise ValueError("dataset must contain numeric feature columns")

    x_all = torch.tensor(feature_frame.to_numpy(copy=True), dtype=torch.float32)
    y_all = torch.tensor(target.to_numpy(copy=True), dtype=torch.float32).unsqueeze(1)
    n_total = x_all.shape[0]
    need = cfg.n_train + cfg.n_unlabeled_id + cfg.n_unlabeled_ood + cfg.n_id_test + cfg.n_ood_test
    if need > n_total:
        raise ValueError(f"Requested {need} rows but dataset has {n_total}")

    if cfg.ood_score_column is not None:
        if cfg.ood_score_column not in frame.columns:
            raise ValueError(f"ood_score_column {cfg.ood_score_column!r} not found in dataset")
        score = torch.tensor(
            frame[cfg.ood_score_column].to_numpy(copy=True), dtype=torch.float32
        ).abs()
    else:
        score = x_all[:, cfg.shift_feature_idx].abs()

    order = torch.argsort(score)
    id_pool = order[: cfg.n_train + cfg.n_unlabeled_id + cfg.n_id_test]
    ood_pool = torch.flip(order[-(cfg.n_unlabeled_ood + cfg.n_ood_test) :], dims=[0])

    x_train = x_all[id_pool[: cfg.n_train]]
    y_train = y_all[id_pool[: cfg.n_train]]
    id_unl_start = cfg.n_train
    id_unl_end = cfg.n_train + cfg.n_unlabeled_id
    x_unlabeled_id = x_all[id_pool[id_unl_start:id_unl_end]]
    y_unlabeled_id = y_all[id_pool[id_unl_start:id_unl_end]]
    x_id_test = x_all[id_pool[id_unl_end : id_unl_end + cfg.n_id_test]]
    y_id_test = y_all[id_pool[id_unl_end : id_unl_end + cfg.n_id_test]]

    x_unlabeled_ood = x_all[ood_pool[: cfg.n_unlabeled_ood]]
    y_unlabeled_ood = y_all[ood_pool[: cfg.n_unlabeled_ood]]
    x_ood_test = x_all[ood_pool[cfg.n_unlabeled_ood : cfg.n_unlabeled_ood + cfg.n_ood_test]]
    y_ood_test = y_all[ood_pool[cfg.n_unlabeled_ood : cfg.n_unlabeled_ood + cfg.n_ood_test]]

    x_unlabeled = torch.cat([x_unlabeled_id, x_unlabeled_ood], dim=0)
    y_unlabeled = torch.cat([y_unlabeled_id, y_unlabeled_ood], dim=0)
    unlabeled_is_ood = torch.cat(
        [
            torch.zeros(cfg.n_unlabeled_id, dtype=torch.bool),
            torch.ones(cfg.n_unlabeled_ood, dtype=torch.bool),
        ]
    )

    x_pool = torch.cat([x_train, x_unlabeled], dim=0)
    x_mean = x_pool.mean(dim=0, keepdim=True)
    x_std = x_pool.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    return HiggsOODSplit(
        x_train=(x_train - x_mean) / x_std,
        y_train=(y_train - y_mean) / y_std,
        x_unlabeled=(x_unlabeled - x_mean) / x_std,
        y_unlabeled_true=(y_unlabeled - y_mean) / y_std,
        unlabeled_is_ood=unlabeled_is_ood,
        x_id_test=(x_id_test - x_mean) / x_std,
        y_id_test=(y_id_test - y_mean) / y_std,
        x_ood_test=(x_ood_test - x_mean) / x_std,
        y_ood_test=(y_ood_test - y_mean) / y_std,
        dataset_name=dataset_name,
        n_features=x_all.shape[1],
    )


def make_split(cfg: HiggsOODConfig) -> HiggsOODSplit:
    set_comparison_seed(cfg.seed)
    if cfg.dataset_path:
        return _make_local_split(cfg)
    return _make_proxy_split(cfg)


def _supervised_loss(model: TabularGaussianRegressor, x: Tensor, y: Tensor) -> Tensor:
    mean, log_var = model(x)
    return torch.nn.functional.gaussian_nll_loss(mean, y, torch.exp(log_var).clamp_min(1e-6))


def _predictive_batch(model_: nn.Module, x: Tensor) -> PredictiveBatch:
    mean, log_var = cast(TabularGaussianRegressor, model_)(x)
    return PredictiveBatch(mean=mean, std=torch.exp(0.5 * log_var))


def _augment_batch(
    x: Tensor,
    is_ood: Tensor,
    base_scale: float,
    boost: float,
    feature_drop_prob: float,
    feature_mix_prob: float,
) -> Tensor:
    scale = torch.full_like(x, base_scale)
    scale = scale * (1.0 + boost * is_ood.reshape(-1, 1).to(dtype=x.dtype))
    augmented = x + scale * torch.randn_like(x)
    if feature_mix_prob > 0.0 and augmented.shape[0] > 1:
        perm = torch.randperm(augmented.shape[0], device=augmented.device)
        mixed = augmented[perm]
        mix_prob = feature_mix_prob * (1.0 + boost * is_ood.reshape(-1, 1).to(dtype=x.dtype))
        mix_mask = torch.rand_like(augmented).lt(mix_prob.clamp(max=1.0))
        augmented = torch.where(mix_mask, mixed, augmented)
    if feature_drop_prob <= 0.0:
        return augmented
    keep_prob = 1.0 - feature_drop_prob * (1.0 + boost * is_ood.reshape(-1, 1).to(dtype=x.dtype))
    keep = torch.rand_like(augmented).le(keep_prob.clamp_min(0.0)).to(augmented.dtype)
    return augmented * keep


def _build_loaders(
    split: HiggsOODSplit,
    *,
    batch_size: int,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    labeled_loader = DataLoader(
        TensorDataset(split.x_train, split.y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    unlabeled_loader = DataLoader(
        TensorDataset(split.x_unlabeled, split.unlabeled_is_ood.to(torch.float32).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=True,
    )
    return labeled_loader, unlabeled_loader


def _train_supervised_teacher(
    cfg: HiggsOODConfig, split: HiggsOODSplit
) -> TabularGaussianRegressor:
    _training_seed(cfg.seed, 0)
    model = TabularGaussianRegressor(split.n_features, cfg.hidden, cfg.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(
        TensorDataset(split.x_train, split.y_train),
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


def _batch_weight_stats(
    weight: Tensor,
    disagreement: Tensor,
    is_ood: Tensor,
) -> dict[str, float]:
    id_mask = ~is_ood
    ood_mask = is_ood

    def _mean_or_zero(values: Tensor, mask: Tensor) -> float:
        if not bool(mask.any().item()):
            return 0.0
        return float(values[mask].mean().item())

    return {
        "mean_weight": float(weight.mean().item()),
        "mean_disagreement": float(disagreement.mean().item()),
        "mean_weight_id": _mean_or_zero(weight, id_mask),
        "mean_weight_ood": _mean_or_zero(weight, ood_mask),
        "mean_disagreement_id": _mean_or_zero(disagreement, id_mask),
        "mean_disagreement_ood": _mean_or_zero(disagreement, ood_mask),
    }


def _train_confidence_student(
    cfg: HiggsOODConfig,
    split: HiggsOODSplit,
    teacher: TabularGaussianRegressor,
) -> tuple[TabularGaussianRegressor, dict[str, float]]:
    _training_seed(cfg.seed, 1)
    student = copy.deepcopy(teacher).train()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr)

    with torch.no_grad():
        teacher_mean, teacher_log_var = teacher(split.x_unlabeled)
        pseudo_confidence = torch.exp(-0.5 * teacher_log_var).clamp(max=1.0).reshape(-1)

    x_all = torch.cat([split.x_train, split.x_unlabeled], dim=0)
    y_all = torch.cat([split.y_train, teacher_mean.detach()], dim=0)
    weight_all = torch.cat(
        [torch.zeros(split.x_train.shape[0]), pseudo_confidence], dim=0
    ).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(x_all, y_all, weight_all),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    for _ in range(cfg.student_epochs):
        for xb, yb, wb in train_loader:
            optimizer.zero_grad()
            mean, log_var = student(xb)
            var = torch.exp(log_var).clamp_min(1e-6)
            labeled_mask = wb.reshape(-1) == 0.0
            unlabeled_mask = ~labeled_mask
            labeled_loss = torch.nn.functional.gaussian_nll_loss(
                mean[labeled_mask], yb[labeled_mask], var[labeled_mask]
            )
            pseudo_loss = torch.zeros((), device=xb.device, dtype=xb.dtype)
            if bool(unlabeled_mask.any().item()):
                per_row = torch.nn.functional.gaussian_nll_loss(
                    mean[unlabeled_mask],
                    yb[unlabeled_mask],
                    var[unlabeled_mask],
                    reduction="none",
                )
                pseudo_weight = wb[unlabeled_mask].clamp_min(0.0)
                pseudo_loss = (per_row * pseudo_weight).sum() / pseudo_weight.sum().clamp_min(1e-8)
            loss = labeled_loss + cfg.pseudo_weight * pseudo_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

    stats = _batch_weight_stats(
        pseudo_confidence,
        torch.zeros_like(pseudo_confidence),
        split.unlabeled_is_ood,
    )
    return student.eval(), stats


def _train_sage_student(
    cfg: HiggsOODConfig,
    split: HiggsOODSplit,
    teacher: TabularGaussianRegressor,
) -> tuple[TabularGaussianRegressor, dict[str, float]]:
    _training_seed(cfg.seed, 2)
    student = copy.deepcopy(teacher)
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader = _build_loaders(split, batch_size=cfg.batch_size)

    def _augment(x: Tensor) -> Tensor:
        return _augment_batch(
            x,
            torch.zeros(x.shape[0], dtype=torch.bool, device=x.device),
            cfg.unlabeled_noise,
            0.0,
            cfg.feature_drop_prob,
            cfg.feature_mix_prob,
        )

    trainer = SelfAgreementTrainer(
        optimizer=optimizer,
        supervised_loss_fn=lambda model_, x, y: _supervised_loss(
            cast(TabularGaussianRegressor, model_), x, y
        ),
        predictive_batch_fn=_predictive_batch,
        augment_fn=_augment,
        n_views=cfg.n_views,
        tau=cfg.tau,
        agreement_weight=cfg.agreement_weight,
        ema_decay=cfg.ema_decay,
        weight_power=cfg.weight_power,
        hard_weight_threshold=cfg.hard_weight_threshold,
    )
    history = trainer.fit(student, labeled_loader, unlabeled_loader, epochs=cfg.student_epochs)

    with torch.no_grad():
        views = [
            _predictive_batch(
                teacher,
                (
                    split.x_unlabeled
                    if idx == 0
                    else _augment_batch(
                        split.x_unlabeled,
                        split.unlabeled_is_ood,
                        cfg.unlabeled_noise,
                        cfg.ood_perturb_boost,
                        cfg.feature_drop_prob,
                        cfg.feature_mix_prob,
                    )
                ),
            )
            for idx in range(cfg.n_views)
        ]
        disagreement = cast(
            Tensor, predictive_agreement_score(views, n_support=96, reduction="none")
        )
        weight = disagreement_to_weight(
            disagreement,
            cfg.tau,
            power=cfg.weight_power,
            hard_weight_threshold=cfg.hard_weight_threshold,
        )

    stats = _batch_weight_stats(weight, disagreement, split.unlabeled_is_ood)
    stats["history_mean_weight"] = float(history["mean_weight"][-1])
    return student.eval(), stats


def _evaluate_regime(model: TabularGaussianRegressor, x: Tensor, y: Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        mean, log_var = model(x)
        std = torch.exp(0.5 * log_var).clamp_min(1e-4)
        var = std.square()

    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=mean.device, dtype=mean.dtype),
        torch.tensor(1.0, device=mean.device, dtype=mean.dtype),
    )
    z = normal.icdf(torch.tensor(0.95, device=mean.device, dtype=mean.dtype))
    lower = mean - z * std
    upper = mean + z * std
    cov90 = prediction_interval_coverage_probability(lower, upper, y, alpha=0.1)
    calibration = calibration_score(y, mean, std)
    calib_mae = calibration["mean_absolute_calibration_error"]
    calib_mae_f = float(calib_mae.item()) if torch.is_tensor(calib_mae) else float(calib_mae)
    return {
        "RMSE": float(torch.sqrt(torch.mean((mean - y).square())).item()),
        "NLL": gaussian_nll(mean, y, var, reduction="mean"),
        "CRPS": crps_gaussian(mean, y, std, reduction="mean"),
        "Cov90": float(cov90),
        "Width90": float(torch.mean(upper - lower).item()),
        "CalibMAE": calib_mae_f,
        "MeanStd": float(std.mean().item()),
    }


def run_benchmark(cfg: HiggsOODConfig) -> list[dict[str, object]]:
    split = make_split(cfg)
    teacher, teacher_s = timed_call(_train_supervised_teacher, cfg, split)
    rows: list[dict[str, object]] = []

    specs = [
        (
            "SupervisedOnly",
            lambda: (
                teacher,
                _batch_weight_stats(
                    torch.zeros(split.x_unlabeled.shape[0]),
                    torch.zeros(split.x_unlabeled.shape[0]),
                    split.unlabeled_is_ood,
                ),
                0.0,
            ),
        ),
        (
            "ConfidenceWeightedPseudoLabel",
            lambda: (*timed_call(_train_confidence_student, cfg, split, teacher),),
        ),
        ("SAGE-Reg", lambda: (*timed_call(_train_sage_student, cfg, split, teacher),)),
    ]

    for method, builder in specs:
        if method == "SupervisedOnly":
            model, meta, train_s = builder()
        else:
            (model, meta), train_s = builder()
        id_metrics, eval_id_s = timed_call(
            _evaluate_regime, model, split.x_id_test, split.y_id_test
        )
        ood_metrics, eval_ood_s = timed_call(
            _evaluate_regime, model, split.x_ood_test, split.y_ood_test
        )
        rows.append(
            {
                "Method": method,
                "Dataset": split.dataset_name,
                "RMSE_ID": float(id_metrics["RMSE"]),
                "RMSE_OOD": float(ood_metrics["RMSE"]),
                "NLL_ID": float(id_metrics["NLL"]),
                "NLL_OOD": float(ood_metrics["NLL"]),
                "CRPS_ID": float(id_metrics["CRPS"]),
                "CRPS_OOD": float(ood_metrics["CRPS"]),
                "Cov90_ID": float(id_metrics["Cov90"]),
                "Cov90_OOD": float(ood_metrics["Cov90"]),
                "Width90_ID": float(id_metrics["Width90"]),
                "Width90_OOD": float(ood_metrics["Width90"]),
                "CalibMAE_ID": float(id_metrics["CalibMAE"]),
                "CalibMAE_OOD": float(ood_metrics["CalibMAE"]),
                "OODUncGap": float(ood_metrics["MeanStd"] - id_metrics["MeanStd"]),
                "MeanWeight": float(meta["mean_weight"]),
                "MeanDisagreement": float(meta["mean_disagreement"]),
                "MeanWeightID": float(meta["mean_weight_id"]),
                "MeanWeightOOD": float(meta["mean_weight_ood"]),
                "MeanDisagreementID": float(meta["mean_disagreement_id"]),
                "MeanDisagreementOOD": float(meta["mean_disagreement_ood"]),
                "train_s": float(teacher_s + train_s if method != "SupervisedOnly" else teacher_s),
                "eval_s": float(eval_id_s + eval_ood_s),
            }
        )
    return rows


def main(
    cfg: HiggsOODConfig | None = None,
    *,
    output_csv: str | None = None,
    performance_figure_path: str | None = None,
    calibration_figure_path: str | None = None,
    summary_json_path: str | None = None,
) -> list[dict[str, object]]:
    resolved = HiggsOODConfig() if cfg is None else cfg
    rows = run_benchmark(resolved)
    print_fairness_notes(
        title="SAGE-Reg Higgs-Inspired OOD Benchmark",
        seed_policy=f"single fixed seed ({resolved.seed}) across all methods",
        train_budget=(
            f"shared Gaussian tabular backbone with {resolved.teacher_epochs} teacher epochs "
            f"and {resolved.student_epochs} student epochs"
        ),
        metric_policy="ID/OOD RMSE, NLL, CRPS, interval quality, and unlabeled weight diagnostics",
    )
    print_comparison_summary(
        "SAGE-Reg Higgs-Inspired OOD Benchmark",
        rows,
        metric_order=[
            "RMSE_ID",
            "RMSE_OOD",
            "NLL_ID",
            "NLL_OOD",
            "Cov90_ID",
            "Cov90_OOD",
            "OODUncGap",
            "MeanWeightID",
            "MeanWeightOOD",
            "MeanDisagreementID",
            "MeanDisagreementOOD",
            "train_s",
        ],
    )

    if output_csv:
        out = _write_csv(output_csv, rows)
        print(f"\nWrote CSV: {out}")
    if performance_figure_path:
        out = _plot_performance(performance_figure_path, rows)
        print(f"Wrote performance figure: {out}")
    if calibration_figure_path:
        out = _plot_calibration(calibration_figure_path, rows)
        print(f"Wrote calibration figure: {out}")
    if summary_json_path:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/benchmarks/self_agreement_higgs_ood.py",
            task="Higgs-inspired OOD benchmark for self-agreement regression",
            config=resolved,
            rows=rows,
            notes=[
                "Uses a local tabular dataset when dataset_path is provided; otherwise falls back to a Higgs-like systematic-shift proxy.",
                "OOD diagnostics focus on whether SAGE-Reg downweights OOD unlabeled samples more than ID unlabeled samples.",
                "The FAIR Universe Higgs Uncertainty Challenge motivates the shift/interval focus: https://fair-universe.lbl.gov/Higgs-Uncertainty-Challenge.html",
            ],
        )
        print(f"Wrote summary JSON: {out}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Higgs-inspired OOD SAGE-Reg benchmark.")
    parser.add_argument("--dataset-path", type=str, default="")
    parser.add_argument("--target-column", type=str, default=HiggsOODConfig.target_column)
    parser.add_argument("--ood-score-column", type=str, default="")
    parser.add_argument("--drop-column", action="append", default=[])
    parser.add_argument("--shift-feature-idx", type=int, default=HiggsOODConfig.shift_feature_idx)
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--performance-figure-path", type=str, default="")
    parser.add_argument("--calibration-figure-path", type=str, default="")
    parser.add_argument("--summary-json-path", type=str, default="")
    parser.add_argument("--n-train", type=int, default=HiggsOODConfig.n_train)
    parser.add_argument("--n-unlabeled-id", type=int, default=HiggsOODConfig.n_unlabeled_id)
    parser.add_argument("--n-unlabeled-ood", type=int, default=HiggsOODConfig.n_unlabeled_ood)
    parser.add_argument("--n-id-test", type=int, default=HiggsOODConfig.n_id_test)
    parser.add_argument("--n-ood-test", type=int, default=HiggsOODConfig.n_ood_test)
    parser.add_argument("--teacher-epochs", type=int, default=HiggsOODConfig.teacher_epochs)
    parser.add_argument("--student-epochs", type=int, default=HiggsOODConfig.student_epochs)
    args = parser.parse_args()

    main(
        HiggsOODConfig(
            dataset_path=args.dataset_path or None,
            target_column=args.target_column,
            ood_score_column=args.ood_score_column or None,
            drop_columns=tuple(args.drop_column),
            shift_feature_idx=args.shift_feature_idx,
            n_train=args.n_train,
            n_unlabeled_id=args.n_unlabeled_id,
            n_unlabeled_ood=args.n_unlabeled_ood,
            n_id_test=args.n_id_test,
            n_ood_test=args.n_ood_test,
            teacher_epochs=args.teacher_epochs,
            student_epochs=args.student_epochs,
        ),
        output_csv=args.output_csv or None,
        performance_figure_path=args.performance_figure_path or None,
        calibration_figure_path=args.calibration_figure_path or None,
        summary_json_path=args.summary_json_path or None,
    )
