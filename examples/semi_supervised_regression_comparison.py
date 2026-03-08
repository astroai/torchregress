"""Real-data proxy comparison for semi-supervised regression."""

import argparse
import copy
from dataclasses import dataclass
from typing import cast

import torch
from comparison_utils import (
    compute_point_metrics,
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from sklearn.datasets import load_diabetes
from torch import Tensor, nn

from torchregress.losses import GaussianNLLLoss, PseudoLabelConsistencyLoss, PseudoLabelNLL
from torchregress.utils import generate_pseudo_labels, update_ema_teacher_


@dataclass(frozen=True)
class SemiSupervisedRegressionConfig:
    seed: int = 260305
    n_labeled: int = 96
    n_unlabeled: int = 220
    n_test: int = 100
    hidden: int = 32
    teacher_epochs: int = 32
    student_epochs: int = 40
    lr: float = 5e-3
    pseudo_confidence_threshold: float = 0.35
    ema_momentum: float = 0.95


class PointRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.net(x))


class GaussianRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_var_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.constant_(self.log_var_head.bias, -1.0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.backbone(x)
        mean = self.mean_head(h)
        log_var = self.log_var_head(h).clamp(min=-5.0, max=2.0)
        return mean, log_var


def _load_data(cfg: SemiSupervisedRegressionConfig) -> dict[str, Tensor]:
    set_comparison_seed(cfg.seed)
    ds = load_diabetes()
    x = torch.tensor(ds.data, dtype=torch.float32)
    y = torch.tensor(ds.target, dtype=torch.float32).unsqueeze(-1)
    perm = torch.randperm(x.shape[0])
    x = x[perm]
    y = y[perm]

    n_total = cfg.n_labeled + cfg.n_unlabeled + cfg.n_test
    x = x[:n_total]
    y = y[:n_total]

    x_train = x[: cfg.n_labeled + cfg.n_unlabeled]
    y_train = y[: cfg.n_labeled + cfg.n_unlabeled]
    x_test = x[cfg.n_labeled + cfg.n_unlabeled :]
    y_test = y[cfg.n_labeled + cfg.n_unlabeled :]

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_train_std = (y_train - y_mean) / y_std
    y_test_std = (y_test - y_mean) / y_std

    return {
        "x_labeled": x_train[: cfg.n_labeled],
        "y_labeled": y_train_std[: cfg.n_labeled],
        "x_unlabeled": x_train[cfg.n_labeled :],
        "y_unlabeled_true": y_train_std[cfg.n_labeled :],
        "x_test": x_test,
        "y_test": y_test_std,
    }


def _train_supervised_point(
    model: PointRegressor,
    x: Tensor,
    y: Tensor,
    *,
    epochs: int,
    lr: float,
) -> PointRegressor:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    return model.eval()


def _train_supervised_gaussian(
    model: GaussianRegressor,
    x: Tensor,
    y: Tensor,
    *,
    epochs: int,
    lr: float,
) -> GaussianRegressor:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = GaussianNLLLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite Gaussian teacher loss during semi-supervised bootstrap")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    return model.eval()


def _bootstrap_teacher(
    cfg: SemiSupervisedRegressionConfig,
    x_labeled: Tensor,
    y_labeled: Tensor,
    x_unlabeled: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    teacher = GaussianRegressor(x_labeled.shape[1], cfg.hidden)
    teacher = _train_supervised_gaussian(
        teacher,
        x_labeled,
        y_labeled,
        epochs=cfg.teacher_epochs,
        lr=cfg.lr,
    )
    with torch.no_grad():
        pseudo_mean, pseudo_log_var = teacher(x_unlabeled)
    pseudo_target, pseudo_confidence, accepted = generate_pseudo_labels(
        pseudo_mean,
        log_variance=pseudo_log_var,
        confidence_threshold=cfg.pseudo_confidence_threshold,
    )
    if not bool(accepted.any().item()):
        flat_conf = pseudo_confidence.flatten()
        fallback_threshold = torch.quantile(flat_conf, 0.75)
        accepted = pseudo_confidence >= fallback_threshold
        pseudo_confidence = torch.where(accepted, pseudo_confidence.clamp_min(0.5), 0.0)
    pseudo_confidence = pseudo_confidence * accepted.to(pseudo_confidence.dtype)
    return pseudo_target, pseudo_confidence, pseudo_mean


def _train_ssl_student(
    cfg: SemiSupervisedRegressionConfig,
    x_all: Tensor,
    target_all: Tensor,
    label_mask: Tensor,
    pseudo_target: Tensor,
    pseudo_confidence: Tensor,
) -> PointRegressor:
    student = PointRegressor(x_all.shape[1], cfg.hidden)
    ema_teacher = copy.deepcopy(student).eval()
    loss_fn = PseudoLabelConsistencyLoss(
        pseudo_weight=0.8,
        consistency_weight=0.25,
        confidence_threshold=cfg.pseudo_confidence_threshold,
    )
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr)

    for _ in range(cfg.student_epochs):
        optimizer.zero_grad()
        student_pred = student(x_all)
        with torch.no_grad():
            teacher_pred = ema_teacher(x_all)
        loss = loss_fn(
            student_pred,
            target_all,
            pseudo_target=pseudo_target,
            pseudo_confidence=pseudo_confidence,
            teacher_pred=teacher_pred,
            label_mask=label_mask,
        )
        loss.backward()
        optimizer.step()
        update_ema_teacher_(ema_teacher, student, momentum=cfg.ema_momentum)

    return student.eval()


def _train_pseudo_label_student(
    cfg: SemiSupervisedRegressionConfig,
    x_all: Tensor,
    target_all: Tensor,
    label_mask: Tensor,
    pseudo_target: Tensor,
    pseudo_confidence: Tensor,
) -> GaussianRegressor:
    model = GaussianRegressor(x_all.shape[1], cfg.hidden)
    loss_fn = PseudoLabelNLL(pseudo_weight=0.8)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.student_epochs):
        optimizer.zero_grad()
        loss = loss_fn(
            model(x_all),
            target_all,
            pseudo_target=pseudo_target,
            pseudo_confidence=pseudo_confidence,
            label_mask=label_mask,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite pseudo-label student loss in semi-supervised example")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    return model.eval()


def run_comparison(
    cfg: SemiSupervisedRegressionConfig,
) -> tuple[list[dict[str, object]], list[str]]:
    data = _load_data(cfg)
    x_labeled = data["x_labeled"]
    y_labeled = data["y_labeled"]
    x_unlabeled = data["x_unlabeled"]
    x_all = torch.cat([x_labeled, x_unlabeled], dim=0)

    pseudo_u, pseudo_conf_u, _ = _bootstrap_teacher(cfg, x_labeled, y_labeled, x_unlabeled)
    target_all = torch.cat([y_labeled, pseudo_u], dim=0)
    label_mask = torch.zeros_like(target_all, dtype=torch.bool)
    label_mask[: y_labeled.shape[0]] = True
    pseudo_target = torch.cat([y_labeled, pseudo_u], dim=0)
    pseudo_confidence = torch.cat([torch.zeros_like(y_labeled), pseudo_conf_u], dim=0)
    accepted_rate = float((pseudo_conf_u > 0).float().mean().item())
    mean_conf = float(pseudo_conf_u[pseudo_conf_u > 0].mean().item()) if accepted_rate > 0 else 0.0

    rows: list[dict[str, object]] = []

    set_comparison_seed(cfg.seed)
    baseline_model = PointRegressor(x_all.shape[1], cfg.hidden)
    baseline_model, train_s = timed_call(
        _train_supervised_point,
        baseline_model,
        x_labeled,
        y_labeled,
        epochs=cfg.student_epochs,
        lr=cfg.lr,
    )
    baseline_pred, eval_s = timed_call(baseline_model, data["x_test"])
    rows.append(
        {
            "Method": "SupervisedMSE",
            **compute_point_metrics(baseline_pred, data["y_test"]),
            "PseudoAcceptRate": accepted_rate,
            "PseudoMeanConf": mean_conf,
            "train_s": float(train_s),
            "eval_s": float(eval_s),
        }
    )

    set_comparison_seed(cfg.seed)
    ssl_model, train_s = timed_call(
        _train_ssl_student,
        cfg,
        x_all,
        target_all,
        label_mask,
        pseudo_target,
        pseudo_confidence,
    )
    ssl_pred, eval_s = timed_call(ssl_model, data["x_test"])
    rows.append(
        {
            "Method": "PseudoLabelConsistency",
            **compute_point_metrics(ssl_pred, data["y_test"]),
            "PseudoAcceptRate": accepted_rate,
            "PseudoMeanConf": mean_conf,
            "train_s": float(train_s),
            "eval_s": float(eval_s),
        }
    )

    set_comparison_seed(cfg.seed)
    pseudo_model, train_s = timed_call(
        _train_pseudo_label_student,
        cfg,
        x_all,
        target_all,
        label_mask,
        pseudo_target,
        pseudo_confidence,
    )
    (pseudo_mean, _), eval_s = timed_call(pseudo_model, data["x_test"])
    rows.append(
        {
            "Method": "PseudoLabelNLL",
            **compute_point_metrics(pseudo_mean, data["y_test"]),
            "PseudoAcceptRate": accepted_rate,
            "PseudoMeanConf": mean_conf,
            "train_s": float(train_s),
            "eval_s": float(eval_s),
        }
    )

    notes = [
        "Uses sklearn Diabetes as a real-data proxy with train-label masking to create unlabeled pool.",
        "Pseudo labels come from a Gaussian teacher trained only on labeled data.",
        "PseudoLabelConsistency combines supervised, pseudo-label, and EMA-teacher consistency terms.",
    ]
    return rows, notes


def main(
    cfg: SemiSupervisedRegressionConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or SemiSupervisedRegressionConfig()
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="Semi-supervised regression comparison",
        seed_policy=f"fixed seed = {cfg.seed}",
        train_budget=(
            f"shared MLP width = {cfg.hidden}, teacher epochs = {cfg.teacher_epochs}, "
            f"student epochs = {cfg.student_epochs}"
        ),
        metric_policy="MSE, MAE, R2, pseudo-label acceptance/confidence, runtime",
    )
    print_comparison_summary(
        "Semi-supervised regression comparison",
        rows,
        metric_order=[
            "MSE",
            "MAE",
            "R2",
            "PseudoAcceptRate",
            "PseudoMeanConf",
            "train_s",
            "eval_s",
        ],
    )

    if summary_json_path is not None:
        write_comparison_summary_json(
            summary_json_path,
            example="examples/semi_supervised_regression_comparison.py",
            task="Semi-supervised regression with pseudo labels and EMA teacher",
            config=cfg,
            rows=rows,
            notes=notes,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare semi-supervised regression methods.")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
