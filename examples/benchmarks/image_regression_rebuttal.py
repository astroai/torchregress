"""Optional image-regression rebuttal benchmark for SAGE-Reg.

This track is intentionally lightweight and synthetic so it can run in CI-sized
budgets while exercising a non-tabular backbone.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from torchregress.comparison import (  # noqa: E402
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.metrics import (  # noqa: E402
    crps_gaussian,
    gaussian_nll,
    prediction_interval_coverage_probability,
)
from torchregress.prediction import PredictiveBatch  # noqa: E402
from torchregress.semi_supervised import SelfAgreementTrainer  # noqa: E402

os.environ.setdefault("MPLBACKEND", "Agg")


@dataclass(frozen=True)
class ImageRebuttalConfig:
    seed: int = 260417
    image_size: int = 16
    n_labeled: int = 256
    n_unlabeled: int = 1024
    n_test: int = 512
    hidden: int = 32
    teacher_epochs: int = 5
    student_epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    tau: float = 0.2
    agreement_weight: float = 0.7
    pseudo_weight: float = 0.8
    ema_decay: float = 0.96
    n_views: int = 3
    unlabeled_noise: float = 0.08


class TinyImageGaussianRegressor(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_var_head = nn.Linear(hidden, 1)
        nn.init.constant_(self.log_var_head.bias, -1.0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        mean = self.mean_head(h)
        log_var = self.log_var_head(h).clamp(min=-6.0, max=3.0)
        return mean, log_var


def _make_image_dataset(cfg: ImageRebuttalConfig) -> tuple[Tensor, Tensor]:
    n_total = cfg.n_labeled + cfg.n_unlabeled + cfg.n_test
    g = torch.Generator().manual_seed(cfg.seed)
    images = torch.zeros((n_total, 1, cfg.image_size, cfg.image_size), dtype=torch.float32)
    targets = torch.zeros((n_total, 1), dtype=torch.float32)
    grid = torch.linspace(-1.0, 1.0, cfg.image_size)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    for i in range(n_total):
        cx = torch.rand(1, generator=g).item() * 1.4 - 0.7
        cy = torch.rand(1, generator=g).item() * 1.4 - 0.7
        amp = 0.6 + 0.8 * torch.rand(1, generator=g).item()
        sigma = 0.15 + 0.25 * torch.rand(1, generator=g).item()
        blob = amp * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
        img = blob + 0.05 * torch.randn_like(blob, generator=g)
        images[i, 0] = img
        targets[i, 0] = float(1.2 * cx - 0.8 * cy + 0.4 * amp + 0.1 * torch.randn(1, generator=g))
    return images, targets


def _split_and_standardize(
    cfg: ImageRebuttalConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    x_all, y_all = _make_image_dataset(cfg)
    x_l = x_all[: cfg.n_labeled]
    y_l = y_all[: cfg.n_labeled]
    x_u = x_all[cfg.n_labeled : cfg.n_labeled + cfg.n_unlabeled]
    y_u = y_all[cfg.n_labeled : cfg.n_labeled + cfg.n_unlabeled]
    x_t = x_all[cfg.n_labeled + cfg.n_unlabeled :]
    y_t = y_all[cfg.n_labeled + cfg.n_unlabeled :]
    y_mean = y_l.mean(dim=0, keepdim=True)
    y_std = y_l.std(dim=0, keepdim=True).clamp_min(1e-6)
    return x_l, (y_l - y_mean) / y_std, x_u, (y_u - y_mean) / y_std, x_t, (y_t - y_mean) / y_std


def _supervised_loss(model: TinyImageGaussianRegressor, x: Tensor, y: Tensor) -> Tensor:
    mean, log_var = model(x)
    return F.gaussian_nll_loss(mean, y, torch.exp(log_var).clamp_min(1e-6))


def _predictive_batch(model_: nn.Module, x: Tensor) -> PredictiveBatch:
    mean, log_var = model_(x)
    return PredictiveBatch(mean=mean, std=torch.exp(0.5 * log_var))


def _augment(x: Tensor, noise: float) -> Tensor:
    return x + noise * torch.randn_like(x)


def _train_teacher(
    cfg: ImageRebuttalConfig, x_l: Tensor, y_l: Tensor
) -> TinyImageGaussianRegressor:
    set_comparison_seed(cfg.seed)
    model = TinyImageGaussianRegressor(cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(TensorDataset(x_l, y_l), batch_size=cfg.batch_size, shuffle=True)
    for _ in range(cfg.teacher_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = _supervised_loss(model, xb, yb)
            loss.backward()
            opt.step()
    return model.eval()


def _train_confidence_student(
    cfg: ImageRebuttalConfig,
    teacher: TinyImageGaussianRegressor,
    x_l: Tensor,
    y_l: Tensor,
    x_u: Tensor,
) -> tuple[TinyImageGaussianRegressor, dict[str, float]]:
    set_comparison_seed(cfg.seed + 1)
    student = TinyImageGaussianRegressor(cfg.hidden)
    student.load_state_dict(teacher.state_dict())
    opt = torch.optim.Adam(student.parameters(), lr=cfg.lr)
    with torch.no_grad():
        m_u, lv_u = teacher(x_u)
        w_u = torch.exp(-0.5 * lv_u).clamp(max=1.0)
    x_all = torch.cat([x_l, x_u], dim=0)
    y_all = torch.cat([y_l, m_u], dim=0)
    w_all = torch.cat([torch.zeros_like(y_l), w_u], dim=0)
    loader = DataLoader(TensorDataset(x_all, y_all, w_all), batch_size=cfg.batch_size, shuffle=True)
    for _ in range(cfg.student_epochs):
        for xb, yb, wb in loader:
            opt.zero_grad()
            m, lv = student(xb)
            var = torch.exp(lv).clamp_min(1e-6)
            is_l = wb.reshape(-1) == 0.0
            is_u = ~is_l
            loss_l = F.gaussian_nll_loss(m[is_l], yb[is_l], var[is_l])
            loss_u = torch.zeros((), dtype=xb.dtype)
            if bool(is_u.any().item()):
                per = F.gaussian_nll_loss(m[is_u], yb[is_u], var[is_u], reduction="none")
                w = wb[is_u].reshape(-1, 1).clamp_min(0.0)
                loss_u = (per * w).sum() / w.sum().clamp_min(1e-8)
            loss = loss_l + cfg.pseudo_weight * loss_u
            loss.backward()
            opt.step()
    return student.eval(), {"mean_weight": float(w_u.mean().item()), "mean_disagreement": 0.0}


def _train_sage_student(
    cfg: ImageRebuttalConfig,
    teacher: TinyImageGaussianRegressor,
    x_l: Tensor,
    y_l: Tensor,
    x_u: Tensor,
) -> tuple[TinyImageGaussianRegressor, dict[str, float]]:
    set_comparison_seed(cfg.seed + 2)
    student = TinyImageGaussianRegressor(cfg.hidden)
    student.load_state_dict(teacher.state_dict())
    opt = torch.optim.Adam(student.parameters(), lr=cfg.lr)
    l_loader = DataLoader(TensorDataset(x_l, y_l), batch_size=cfg.batch_size, shuffle=True)
    u_loader = DataLoader(TensorDataset(x_u), batch_size=cfg.batch_size, shuffle=True)
    trainer = SelfAgreementTrainer(
        optimizer=opt,
        supervised_loss_fn=lambda model_, x, y: _supervised_loss(model_, x, y),
        predictive_batch_fn=_predictive_batch,
        augment_fn=lambda x: _augment(x, cfg.unlabeled_noise),
        n_views=cfg.n_views,
        tau=cfg.tau,
        agreement_weight=cfg.agreement_weight,
        ema_decay=cfg.ema_decay,
    )
    history = trainer.fit(student, l_loader, u_loader, epochs=cfg.student_epochs)
    return student.eval(), {
        "mean_weight": float(history["mean_weight"][-1]),
        "mean_disagreement": float(history["mean_disagreement"][-1]),
    }


def _eval(model: TinyImageGaussianRegressor, x: Tensor, y: Tensor) -> dict[str, float]:
    with torch.no_grad():
        m, lv = model(x)
        std = torch.exp(0.5 * lv).clamp_min(1e-4)
        var = std.square()
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.95))
    lo = m - z * std
    hi = m + z * std
    cov90 = prediction_interval_coverage_probability(lo, hi, y, alpha=0.1)
    return {
        "RMSE": float(torch.sqrt(torch.mean((m - y).square())).item()),
        "NLL": gaussian_nll(m, y, var, reduction="mean"),
        "CRPS": crps_gaussian(m, y, std, reduction="mean"),
        "Cov90": float(cov90),
        "Width90": float(torch.mean(hi - lo).item()),
    }


def run_benchmark(cfg: ImageRebuttalConfig) -> list[dict[str, object]]:
    x_l, y_l, x_u, _y_u, x_t, y_t = _split_and_standardize(cfg)
    teacher, t_s = timed_call(_train_teacher, cfg, x_l, y_l)
    rows: list[dict[str, object]] = []
    specs = [
        ("SupervisedOnly", lambda: (teacher, {"mean_weight": 0.0, "mean_disagreement": 0.0}, 0.0)),
        (
            "ConfidenceWeightedPseudoLabel",
            lambda: (*timed_call(_train_confidence_student, cfg, teacher, x_l, y_l, x_u),),
        ),
        ("SAGE-Reg", lambda: (*timed_call(_train_sage_student, cfg, teacher, x_l, y_l, x_u),)),
    ]
    for method, builder in specs:
        if method == "SupervisedOnly":
            model, meta, train_s = builder()
        else:
            (model, meta), train_s = builder()
        metrics, eval_s = timed_call(_eval, model, x_t, y_t)
        rows.append(
            {
                "Method": method,
                **metrics,
                "MeanWeight": float(meta["mean_weight"]),
                "MeanDisagreement": float(meta["mean_disagreement"]),
                "train_s": float(t_s + train_s if method != "SupervisedOnly" else t_s),
                "eval_s": float(eval_s),
            }
        )
    return rows


def main(
    cfg: ImageRebuttalConfig | None = None, *, summary_json_path: str | None = None
) -> list[dict[str, object]]:
    resolved = ImageRebuttalConfig() if cfg is None else cfg
    rows = run_benchmark(resolved)
    print_fairness_notes(
        title="Image Rebuttal Benchmark (Synthetic Vision Regression)",
        seed_policy=f"single fixed seed ({resolved.seed})",
        train_budget=(
            f"shared tiny CNN + Gaussian head with {resolved.teacher_epochs} teacher and "
            f"{resolved.student_epochs} student epochs"
        ),
        metric_policy="NLL, CRPS, Cov90, Width90, RMSE",
    )
    print_comparison_summary(
        "Image Rebuttal Benchmark",
        rows,
        metric_order=["RMSE", "NLL", "CRPS", "Cov90", "Width90", "MeanWeight", "train_s"],
    )
    if summary_json_path:
        write_comparison_summary_json(
            summary_json_path,
            example="examples/benchmarks/image_regression_rebuttal.py",
            task="optional image regression rebuttal benchmark for SAGE-Reg",
            config=resolved,
            rows=rows,
            notes=[
                "Synthetic image benchmark for fast cross-modality sanity checks.",
                "Not a replacement for real vision benchmarks; intended as optional rebuttal support.",
            ],
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run optional image-regression rebuttal benchmark."
    )
    parser.add_argument("--summary-json-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=ImageRebuttalConfig.seed)
    parser.add_argument("--n-labeled", type=int, default=ImageRebuttalConfig.n_labeled)
    parser.add_argument("--n-unlabeled", type=int, default=ImageRebuttalConfig.n_unlabeled)
    parser.add_argument("--n-test", type=int, default=ImageRebuttalConfig.n_test)
    parser.add_argument("--teacher-epochs", type=int, default=ImageRebuttalConfig.teacher_epochs)
    parser.add_argument("--student-epochs", type=int, default=ImageRebuttalConfig.student_epochs)
    args = parser.parse_args()
    main(
        ImageRebuttalConfig(
            seed=args.seed,
            n_labeled=args.n_labeled,
            n_unlabeled=args.n_unlabeled,
            n_test=args.n_test,
            teacher_epochs=args.teacher_epochs,
            student_epochs=args.student_epochs,
        ),
        summary_json_path=args.summary_json_path or None,
    )
