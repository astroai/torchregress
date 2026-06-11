"""Synthetic semi-supervised benchmark for SAGE-Reg."""

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
import torch  # noqa: E402
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
    build_consensus_predictive_batch,
    distributional_pseudo_loss,
    predictive_agreement_score,
)
from torchregress.utils import generate_pseudo_labels, update_ema_teacher_  # noqa: E402


@dataclass(frozen=True)
class SyntheticRegressionGeneratorConfig:
    seed: int = 260408
    n_labeled: int = 72
    n_unlabeled: int = 320
    n_test: int = 240
    x_range: float = 3.0
    hole_radius: float = 0.65
    label_edge_jitter: float = 0.25
    noise_scale: float = 0.18
    input_noise_std: float = 0.08
    multimodal_prob: float = 0.0
    multimodal_shift: float = 0.75
    imbalance_strength: float = 0.0
    confidence_trap_strength: float = 0.0


@dataclass(frozen=True)
class SelfAgreementSyntheticConfig:
    data: SyntheticRegressionGeneratorConfig = SyntheticRegressionGeneratorConfig()
    hidden: int = 48
    teacher_epochs: int = 36
    student_epochs: int = 40
    batch_size: int = 32
    lr: float = 5e-3
    dropout: float = 0.12
    unlabeled_noise: float = 0.08
    center_perturb_boost: float = 1.5
    center_perturb_radius: float = 0.75
    tau: float = 0.18
    agreement_weight: float = 0.7
    ema_decay: float = 0.96
    pseudo_weight: float = 0.8
    n_views: int = 4
    unlabeled_fractions: tuple[float, ...] = (0.25, 0.5, 1.0)
    run_ablations: bool = True


@dataclass(frozen=True)
class SAGESyntheticSplit:
    x_labeled: Tensor
    y_labeled: Tensor
    x_unlabeled: Tensor
    y_unlabeled_true: Tensor
    x_test: Tensor
    y_test: Tensor
    hole_mask: Tensor
    x_train_raw: Tensor
    y_train_raw: Tensor
    x_mean: Tensor
    x_std: Tensor
    y_mean: Tensor
    y_std: Tensor


class GaussianDropoutRegressor(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_var_head = nn.Linear(hidden, 1)
        nn.init.constant_(self.log_var_head.bias, -1.4)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.backbone(x)
        return self.mean_head(h), self.log_var_head(h).clamp(min=-4.5, max=2.0)


def _sample_x(generator_cfg: SyntheticRegressionGeneratorConfig, *, n_samples: int) -> Tensor:
    generator = torch.Generator().manual_seed(generator_cfg.seed + n_samples)
    base = torch.empty(n_samples, 1).uniform_(
        -generator_cfg.x_range, generator_cfg.x_range, generator=generator
    )
    if generator_cfg.imbalance_strength <= 0.0:
        return base

    skew_prob = 0.5 + 0.35 * min(generator_cfg.imbalance_strength, 1.0)
    selector = torch.rand(n_samples, 1, generator=generator) < skew_prob
    left = torch.randn(n_samples, 1, generator=generator) * 0.55 - 1.55
    right = torch.randn(n_samples, 1, generator=generator) * 0.95 + 1.10
    skewed = torch.where(selector, left, right).clamp(-generator_cfg.x_range, generator_cfg.x_range)
    mix = min(0.9, generator_cfg.imbalance_strength)
    return (1.0 - mix) * base + mix * skewed


def _latent_mean(x: Tensor) -> Tensor:
    return torch.sin(1.8 * x) + 0.35 * x + 0.20 * torch.sin(4.0 * x)


def _aleatoric_std(x: Tensor, generator_cfg: SyntheticRegressionGeneratorConfig) -> Tensor:
    base = generator_cfg.noise_scale * (0.55 + 0.35 * torch.sigmoid(1.2 * x) + 0.25 * x.abs())
    if generator_cfg.confidence_trap_strength <= 0.0:
        return base

    trap_width = max(generator_cfg.hole_radius, 1.0e-3)
    trap_profile = torch.exp(-0.5 * (x / trap_width).square())
    scaled = base * (1.0 - min(generator_cfg.confidence_trap_strength, 0.95) * trap_profile)
    floor = 0.15 * generator_cfg.noise_scale
    return scaled.clamp_min(floor)


def _sample_targets(
    x_true: Tensor,
    *,
    generator_cfg: SyntheticRegressionGeneratorConfig,
    generator: torch.Generator,
) -> Tensor:
    mean = _latent_mean(x_true)
    base_std = _aleatoric_std(x_true, generator_cfg)
    y = mean + base_std * torch.randn(x_true.shape, generator=generator)

    if generator_cfg.multimodal_prob > 0.0:
        selector = torch.rand(x_true.shape, generator=generator) < generator_cfg.multimodal_prob
        branch = torch.where(
            torch.rand(x_true.shape, generator=generator) < 0.5,
            -torch.ones_like(x_true),
            torch.ones_like(x_true),
        )
        shift = generator_cfg.multimodal_shift * (0.6 + 0.25 * torch.cos(1.5 * x_true))
        multimodal_sample = (
            mean + branch * shift + 0.7 * base_std * torch.randn(x_true.shape, generator=generator)
        )
        y = torch.where(selector, multimodal_sample, y)
    return y


def generate_self_agreement_regression_split(
    generator_cfg: SyntheticRegressionGeneratorConfig,
) -> SAGESyntheticSplit:
    """Reusable synthetic split generator for SAGE-Reg experiments."""

    set_comparison_seed(generator_cfg.seed)
    train_generator = torch.Generator().manual_seed(generator_cfg.seed)
    test_generator = torch.Generator().manual_seed(generator_cfg.seed + 1)

    n_train = generator_cfg.n_labeled + generator_cfg.n_unlabeled
    x_true_train = _sample_x(generator_cfg, n_samples=n_train)
    x_true_test = _sample_x(
        SyntheticRegressionGeneratorConfig(
            **{**generator_cfg.__dict__, "seed": generator_cfg.seed + 17}
        ),
        n_samples=generator_cfg.n_test,
    )

    x_obs_train = x_true_train + generator_cfg.input_noise_std * torch.randn(
        x_true_train.shape,
        generator=train_generator,
    )
    x_obs_test = x_true_test + generator_cfg.input_noise_std * torch.randn(
        x_true_test.shape,
        generator=test_generator,
    )

    y_train = _sample_targets(x_true_train, generator_cfg=generator_cfg, generator=train_generator)
    y_test = _sample_targets(x_true_test, generator_cfg=generator_cfg, generator=test_generator)

    selection_score = x_true_train.abs().view(-1) + generator_cfg.label_edge_jitter * torch.rand(
        n_train,
        generator=train_generator,
    )
    labeled_idx = torch.topk(selection_score, k=generator_cfg.n_labeled).indices
    labeled_mask = torch.zeros(n_train, dtype=torch.bool)
    labeled_mask[labeled_idx] = True
    unlabeled_mask = ~labeled_mask

    x_labeled = x_obs_train[labeled_mask]
    y_labeled = y_train[labeled_mask]
    x_unlabeled = x_obs_train[unlabeled_mask]
    y_unlabeled_true = y_train[unlabeled_mask]

    x_mean = x_obs_train.mean(dim=0, keepdim=True)
    x_std = x_obs_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_labeled.mean(dim=0, keepdim=True)
    y_std = y_labeled.std(dim=0, keepdim=True).clamp_min(1e-6)

    hole_mask = x_true_test.abs().view(-1) < generator_cfg.hole_radius
    return SAGESyntheticSplit(
        x_labeled=(x_labeled - x_mean) / x_std,
        y_labeled=(y_labeled - y_mean) / y_std,
        x_unlabeled=(x_unlabeled - x_mean) / x_std,
        y_unlabeled_true=(y_unlabeled_true - y_mean) / y_std,
        x_test=(x_obs_test - x_mean) / x_std,
        y_test=(y_test - y_mean) / y_std,
        hole_mask=hole_mask,
        x_train_raw=x_obs_train,
        y_train_raw=y_train,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )


def _forward_with_mode(
    model: GaussianDropoutRegressor,
    x: Tensor,
    *,
    stochastic: bool,
) -> tuple[Tensor, Tensor]:
    was_training = model.training
    if stochastic and not was_training:
        model.train()
    elif not stochastic and was_training:
        model.eval()
    try:
        mean, log_var = model(x)
    finally:
        model.train(was_training)
    return mean, log_var


def _predictive_batch(
    model: GaussianDropoutRegressor,
    x: Tensor,
    *,
    stochastic: bool,
) -> PredictiveBatch:
    mean, log_var = _forward_with_mode(model, x, stochastic=stochastic)
    return PredictiveBatch(mean=mean, std=torch.exp(0.5 * log_var))


def _gaussian_supervised_loss(model: GaussianDropoutRegressor, x: Tensor, y: Tensor) -> Tensor:
    mean, log_var = model(x)
    return torch.nn.functional.gaussian_nll_loss(mean, y, torch.exp(log_var).clamp_min(1e-6))


def _train_supervised_teacher(
    cfg: SelfAgreementSyntheticConfig,
    x_labeled: Tensor,
    y_labeled: Tensor,
) -> GaussianDropoutRegressor:
    model = GaussianDropoutRegressor(cfg.hidden, cfg.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.teacher_epochs):
        model.train()
        optimizer.zero_grad()
        loss = _gaussian_supervised_loss(model, x_labeled, y_labeled)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    return model.eval()


def _subsample_tensor(x: Tensor, fraction: float) -> Tensor:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    count = max(1, int(round(fraction * x.shape[0])))
    return x[:count]


def _train_pseudo_student(
    cfg: SelfAgreementSyntheticConfig,
    teacher: GaussianDropoutRegressor,
    x_labeled: Tensor,
    y_labeled: Tensor,
    x_unlabeled: Tensor,
    *,
    weighted: bool,
) -> tuple[GaussianDropoutRegressor, dict[str, float]]:
    student = copy.deepcopy(teacher).train()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr)

    with torch.no_grad():
        teacher_mean, teacher_log_var = _forward_with_mode(teacher, x_unlabeled, stochastic=False)
    if weighted:
        _, pseudo_confidence, _ = generate_pseudo_labels(
            teacher_mean,
            log_variance=teacher_log_var,
            confidence_threshold=0.0,
        )
    else:
        pseudo_confidence = torch.ones_like(teacher_mean)

    x_all = torch.cat([x_labeled, x_unlabeled], dim=0)
    y_all = torch.cat([y_labeled, teacher_mean.detach()], dim=0)
    weights_all = torch.cat([torch.zeros_like(y_labeled), pseudo_confidence], dim=0)

    for _ in range(cfg.student_epochs):
        optimizer.zero_grad()
        mean, log_var = student(x_all)
        var = torch.exp(log_var).clamp_min(1e-6)
        labeled_loss = torch.nn.functional.gaussian_nll_loss(
            mean[: y_labeled.shape[0]], y_labeled, var[: y_labeled.shape[0]]
        )
        pseudo_loss = torch.nn.functional.gaussian_nll_loss(
            mean[y_labeled.shape[0] :],
            y_all[y_labeled.shape[0] :],
            var[y_labeled.shape[0] :],
            reduction="none",
        )
        weight = weights_all[y_labeled.shape[0] :].clamp_min(0.0)
        blended = (pseudo_loss * weight).sum() / weight.sum().clamp_min(1e-8)
        loss = labeled_loss + cfg.pseudo_weight * blended
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        optimizer.step()

    return student.eval(), {
        "mean_weight": float(pseudo_confidence.mean().item()),
        "mean_disagreement": 0.0,
    }


def _teacher_views(
    teacher: GaussianDropoutRegressor,
    x_unlabeled: Tensor,
    *,
    n_views: int,
    augment_scale: float,
    center_boost: float,
    center_radius: float,
) -> list[PredictiveBatch]:
    views: list[PredictiveBatch] = []
    with torch.no_grad():
        for idx in range(n_views):
            if idx == 0:
                x_view = x_unlabeled
            else:
                scale = torch.full_like(x_unlabeled, augment_scale)
                if center_boost > 0.0:
                    radius = max(center_radius, 1.0e-3)
                    center_profile = torch.exp(-0.5 * (x_unlabeled / radius).square())
                    scale = scale * (1.0 + center_boost * center_profile)
                x_view = x_unlabeled + scale * torch.randn_like(x_unlabeled)
            views.append(_predictive_batch(teacher, x_view, stochastic=True))
    return views


def _train_sage_variant(
    cfg: SelfAgreementSyntheticConfig,
    bootstrap_teacher: GaussianDropoutRegressor,
    x_labeled: Tensor,
    y_labeled: Tensor,
    x_unlabeled: Tensor,
    *,
    use_disagreement_weighting: bool,
    use_multi_view_consensus: bool,
    ema_decay: float | None,
) -> tuple[GaussianDropoutRegressor, dict[str, float]]:
    student = copy.deepcopy(bootstrap_teacher)
    teacher = copy.deepcopy(student).eval() if ema_decay is not None else bootstrap_teacher.eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr)

    labeled_loader = DataLoader(
        TensorDataset(x_labeled, y_labeled), batch_size=cfg.batch_size, shuffle=True
    )
    unlabeled_loader = DataLoader(
        TensorDataset(x_unlabeled), batch_size=cfg.batch_size, shuffle=True
    )

    latest_weight = 1.0
    latest_disagreement = 0.0
    for _ in range(cfg.student_epochs):
        unlabeled_iter = iter(unlabeled_loader)
        for xb, yb in labeled_loader:
            try:
                (xu,) = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                (xu,) = next(unlabeled_iter)

            student.train()
            optimizer.zero_grad()
            supervised_loss = _gaussian_supervised_loss(student, xb, yb)
            student_pred = _predictive_batch(student, xu, stochastic=False)

            if use_multi_view_consensus:
                views = _teacher_views(
                    cast(GaussianDropoutRegressor, teacher),
                    xu,
                    n_views=cfg.n_views,
                    augment_scale=cfg.unlabeled_noise,
                    center_boost=cfg.center_perturb_boost,
                    center_radius=cfg.center_perturb_radius,
                )
                consensus = build_consensus_predictive_batch(views, n_support=96)
                disagreement = predictive_agreement_score(views, n_support=96, reduction="none")
                weights = (
                    torch.exp(-disagreement / cfg.tau)
                    if use_disagreement_weighting
                    else torch.ones_like(disagreement)
                )
            else:
                with torch.no_grad():
                    consensus = _predictive_batch(
                        cast(GaussianDropoutRegressor, teacher), xu, stochastic=True
                    )
                disagreement = torch.zeros(xu.shape[0], device=xu.device, dtype=xu.dtype)
                weights = torch.ones_like(disagreement)

            unsup_loss = distributional_pseudo_loss(
                student_pred, consensus, sample_weights=weights, n_support=96
            )
            total_loss = supervised_loss + cfg.agreement_weight * unsup_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()
            if ema_decay is not None:
                update_ema_teacher_(cast(nn.Module, teacher), student, momentum=ema_decay)

            latest_weight = float(weights.mean().detach().item())
            latest_disagreement = float(disagreement.mean().detach().item())

    return student.eval(), {
        "mean_weight": latest_weight,
        "mean_disagreement": latest_disagreement,
    }


def _evaluate_model(
    model: GaussianDropoutRegressor,
    x_test: Tensor,
    y_test: Tensor,
    hole_mask: Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        mean, log_var = _forward_with_mode(model, x_test, stochastic=False)
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

    rmse = float(torch.sqrt(torch.mean((mean - y_test).square())).item())
    hole_rmse = float(torch.sqrt(torch.mean((mean[hole_mask] - y_test[hole_mask]).square())).item())
    return {
        "RMSE": rmse,
        "HoleRMSE": hole_rmse,
        "NLL": gaussian_nll(mean, y_test, var, reduction="mean"),
        "CRPS": crps_gaussian(mean, y_test, std, reduction="mean"),
        "Cov90": float(cov90),
        "Width90": float(torch.mean(upper - lower).item()),
        "CoverageGap90": abs(float(cov90) - 0.9),
        "CalibMAE": calib_mae_f,
    }


def _run_fraction(
    cfg: SelfAgreementSyntheticConfig,
    split: SAGESyntheticSplit,
    fraction: float,
) -> list[dict[str, object]]:
    x_labeled = split.x_labeled
    y_labeled = split.y_labeled
    x_unlabeled = _subsample_tensor(split.x_unlabeled, fraction)
    x_test = split.x_test
    y_test = split.y_test
    hole_mask = split.hole_mask

    teacher, teacher_s = timed_call(_train_supervised_teacher, cfg, x_labeled, y_labeled)
    rows: list[dict[str, object]] = []

    method_specs: list[tuple[str, callable]] = [
        ("SupervisedOnly", lambda: (teacher, {"mean_weight": 0.0, "mean_disagreement": 0.0}, 0.0)),
        (
            "PointPseudoLabel",
            lambda: (
                *timed_call(
                    _train_pseudo_student,
                    cfg,
                    teacher,
                    x_labeled,
                    y_labeled,
                    x_unlabeled,
                    weighted=False,
                ),
            ),
        ),
        (
            "ConfidenceWeightedPseudoLabel",
            lambda: (
                *timed_call(
                    _train_pseudo_student,
                    cfg,
                    teacher,
                    x_labeled,
                    y_labeled,
                    x_unlabeled,
                    weighted=True,
                ),
            ),
        ),
        (
            "SAGE-Reg",
            lambda: (
                *timed_call(
                    _train_sage_variant,
                    cfg,
                    teacher,
                    x_labeled,
                    y_labeled,
                    x_unlabeled,
                    use_disagreement_weighting=True,
                    use_multi_view_consensus=True,
                    ema_decay=cfg.ema_decay,
                ),
            ),
        ),
    ]
    if cfg.run_ablations:
        method_specs.extend(
            [
                (
                    "SAGE-Reg (No Weighting)",
                    lambda: (
                        *timed_call(
                            _train_sage_variant,
                            cfg,
                            teacher,
                            x_labeled,
                            y_labeled,
                            x_unlabeled,
                            use_disagreement_weighting=False,
                            use_multi_view_consensus=True,
                            ema_decay=cfg.ema_decay,
                        ),
                    ),
                ),
                (
                    "SAGE-Reg (Single View)",
                    lambda: (
                        *timed_call(
                            _train_sage_variant,
                            cfg,
                            teacher,
                            x_labeled,
                            y_labeled,
                            x_unlabeled,
                            use_disagreement_weighting=False,
                            use_multi_view_consensus=False,
                            ema_decay=cfg.ema_decay,
                        ),
                    ),
                ),
                (
                    "SAGE-Reg (No EMA)",
                    lambda: (
                        *timed_call(
                            _train_sage_variant,
                            cfg,
                            teacher,
                            x_labeled,
                            y_labeled,
                            x_unlabeled,
                            use_disagreement_weighting=True,
                            use_multi_view_consensus=True,
                            ema_decay=None,
                        ),
                    ),
                ),
            ]
        )

    for method, builder in method_specs:
        if method == "SupervisedOnly":
            model, meta, train_s = builder()
        else:
            (model, meta), train_s = builder()
        metrics, eval_s = timed_call(_evaluate_model, model, x_test, y_test, hole_mask)
        rows.append(
            {
                "Method": method,
                "UnlabeledFraction": float(fraction),
                **metrics,
                "MeanWeight": float(meta["mean_weight"]),
                "MeanDisagreement": float(meta["mean_disagreement"]),
                "train_s": float(teacher_s + train_s if method != "SupervisedOnly" else teacher_s),
                "eval_s": float(eval_s),
            }
        )
    return rows


def run_benchmark(cfg: SelfAgreementSyntheticConfig) -> list[dict[str, object]]:
    split = generate_self_agreement_regression_split(cfg.data)
    rows: list[dict[str, object]] = []
    for fraction in cfg.unlabeled_fractions:
        rows.extend(_run_fraction(cfg, split, fraction))
    return rows


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
    axes[1].legend(loc="best", fontsize=7)
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
    axes[1].legend(loc="best", fontsize=7)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _collect_unlabeled_diagnostics(
    cfg: SelfAgreementSyntheticConfig,
    split: SAGESyntheticSplit,
    fraction: float,
) -> dict[str, Tensor]:
    x_labeled = split.x_labeled
    y_labeled = split.y_labeled
    x_unlabeled = _subsample_tensor(split.x_unlabeled, fraction)
    y_unlabeled_true = _subsample_tensor(split.y_unlabeled_true, fraction)
    teacher = _train_supervised_teacher(cfg, x_labeled, y_labeled)

    with torch.no_grad():
        teacher_mean, teacher_log_var = _forward_with_mode(teacher, x_unlabeled, stochastic=False)
        _, pseudo_confidence, _ = generate_pseudo_labels(
            teacher_mean,
            log_variance=teacher_log_var,
            confidence_threshold=0.0,
        )
        views = _teacher_views(
            teacher,
            x_unlabeled,
            n_views=cfg.n_views,
            augment_scale=cfg.unlabeled_noise,
            center_boost=cfg.center_perturb_boost,
            center_radius=cfg.center_perturb_radius,
        )
        disagreement = predictive_agreement_score(views, n_support=96, reduction="none")
        agreement_weight = torch.exp(-disagreement / cfg.tau)

    abs_error = (teacher_mean - y_unlabeled_true).abs().reshape(-1)
    return {
        "fraction": torch.full_like(abs_error, float(fraction)),
        "abs_error": abs_error,
        "pseudo_confidence": pseudo_confidence.reshape(-1),
        "agreement_weight": agreement_weight.reshape(-1),
        "disagreement": disagreement.reshape(-1),
    }


def _plot_stress_diagnostics(
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

    axes[0].hist(
        confidence[low_error].cpu().numpy(),
        bins=bins,
        alpha=0.65,
        label="low error",
        density=True,
    )
    axes[0].hist(
        confidence[high_error].cpu().numpy(),
        bins=bins,
        alpha=0.65,
        label="high error",
        density=True,
    )
    axes[0].set_title("Confidence vs pseudo-label error")
    axes[0].set_xlabel("Teacher confidence")
    axes[0].set_ylabel("Density")

    axes[1].hist(
        weight[low_error].cpu().numpy(),
        bins=bins,
        alpha=0.65,
        label="low error",
        density=True,
    )
    axes[1].hist(
        weight[high_error].cpu().numpy(),
        bins=bins,
        alpha=0.65,
        label="high error",
        density=True,
    )
    axes[1].set_title("SAGE weight vs pseudo-label error")
    axes[1].set_xlabel("Agreement weight")

    axes[2].hist(
        disagreement[low_error].cpu().numpy(),
        bins=bins,
        alpha=0.65,
        label="low error",
        density=True,
    )
    axes[2].hist(
        disagreement[high_error].cpu().numpy(),
        bins=bins,
        alpha=0.65,
        label="high error",
        density=True,
    )
    axes[2].set_title("Disagreement vs pseudo-label error")
    axes[2].set_xlabel("Mean disagreement")

    axes[2].legend(loc="best", fontsize=8)
    fig.suptitle(f"Stress diagnostic at unlabeled fraction = {fraction:.2f}", fontsize=11)
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
        "\nStress ranking summary\n"
        f"  fraction: {fraction:.4f}\n"
        f"  confidence_vs_neg_error_spearman: {summary['ConfidenceVsNegErrorSpearman']:.4f}\n"
        f"  agreement_weight_vs_neg_error_spearman: {summary['AgreementWeightVsNegErrorSpearman']:.4f}\n"
        f"  neg_disagreement_vs_neg_error_spearman: {summary['NegDisagreementVsNegErrorSpearman']:.4f}\n"
        f"  confidence_top_bottom_error_ratio: {summary['ConfidenceTopBottomErrorRatio']:.4f}\n"
        f"  agreement_weight_top_bottom_error_ratio: {summary['AgreementWeightTopBottomErrorRatio']:.4f}\n"
        f"  neg_disagreement_top_bottom_error_ratio: {summary['NegDisagreementTopBottomErrorRatio']:.4f}"
    )


def main(
    cfg: SelfAgreementSyntheticConfig | None = None,
    *,
    output_csv: str | None = None,
    figure_path: str | None = None,
    performance_figure_path: str | None = None,
    calibration_figure_path: str | None = None,
    diagnostic_figure_path: str | None = None,
    summary_json_path: str | None = None,
) -> list[dict[str, object]]:
    resolved = SelfAgreementSyntheticConfig() if cfg is None else cfg
    rows = run_benchmark(resolved)

    perf_path = performance_figure_path or figure_path
    calib_path = calibration_figure_path
    if perf_path is not None and calib_path is None:
        perf = Path(perf_path)
        calib_path = str(perf.with_name(f"{perf.stem}_calibration{perf.suffix or '.png'}"))

    print_fairness_notes(
        title="SAGE-Reg Synthetic Benchmark",
        seed_policy=f"single fixed seed ({resolved.data.seed}) across all methods/fractions",
        train_budget=(
            "shared Gaussian backbone and teacher bootstrap; methods differ only in unlabeled objective "
            f"under {resolved.student_epochs} student epochs"
        ),
        metric_policy="RMSE, NLL, CRPS, 90% interval quality, and Gaussian calibration summary",
    )
    print_comparison_summary(
        "SAGE-Reg Synthetic Benchmark",
        rows,
        metric_order=[
            "UnlabeledFraction",
            "RMSE",
            "HoleRMSE",
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
        diagnostic_fraction = max(resolved.unlabeled_fractions)
        diagnostics = _collect_unlabeled_diagnostics(
            resolved, generate_self_agreement_regression_split(resolved.data), diagnostic_fraction
        )
        diagnostic_summary = _diagnostic_summary(diagnostics)
        out = _plot_stress_diagnostics(
            diagnostic_figure_path,
            diagnostics,
            fraction=float(diagnostic_fraction),
        )
        print(f"Wrote diagnostic figure: {out}")
        print(_format_diagnostic_summary(diagnostic_summary, fraction=float(diagnostic_fraction)))
    if summary_json_path:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/benchmarks/self_agreement_synthetic.py",
            task="self-agreement synthetic semi-supervised regression benchmark",
            config=resolved,
            rows=rows,
            notes=[
                "Generator supports heteroscedastic noise, multimodality, epistemic holes, target imbalance, and input noise.",
                "Ablations cover no disagreement weighting, no multi-view consensus, and no EMA teacher.",
                "Optional diagnostic figure compares confidence and agreement-weight distributions against pseudo-label error on unlabeled data.",
            ],
        )
        print(f"Wrote summary JSON: {out}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the synthetic SAGE-Reg benchmark.")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--figure-path", type=str, default="")
    parser.add_argument("--performance-figure-path", type=str, default="")
    parser.add_argument("--calibration-figure-path", type=str, default="")
    parser.add_argument("--diagnostic-figure-path", type=str, default="")
    parser.add_argument("--summary-json-path", type=str, default="")
    parser.add_argument(
        "--teacher-epochs", type=int, default=SelfAgreementSyntheticConfig.teacher_epochs
    )
    parser.add_argument(
        "--student-epochs", type=int, default=SelfAgreementSyntheticConfig.student_epochs
    )
    parser.add_argument(
        "--multimodal-prob", type=float, default=SyntheticRegressionGeneratorConfig.multimodal_prob
    )
    parser.add_argument(
        "--imbalance-strength",
        type=float,
        default=SyntheticRegressionGeneratorConfig.imbalance_strength,
    )
    parser.add_argument(
        "--input-noise-std", type=float, default=SyntheticRegressionGeneratorConfig.input_noise_std
    )
    parser.add_argument(
        "--confidence-trap-strength",
        type=float,
        default=SyntheticRegressionGeneratorConfig.confidence_trap_strength,
    )
    args = parser.parse_args()
    cfg = SelfAgreementSyntheticConfig(
        teacher_epochs=args.teacher_epochs,
        student_epochs=args.student_epochs,
        data=SyntheticRegressionGeneratorConfig(
            multimodal_prob=args.multimodal_prob,
            imbalance_strength=args.imbalance_strength,
            input_noise_std=args.input_noise_std,
            confidence_trap_strength=args.confidence_trap_strength,
        ),
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
