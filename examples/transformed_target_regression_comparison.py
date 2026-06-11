"""Shared-budget comparison of transformed-target regression losses."""

import argparse
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from torchregress.comparison import (
    compute_point_metrics,
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.losses import (
    BoxCoxTransformLoss,
    LogTransformLoss,
    SqrtTransformLoss,
    WeightedMSELoss,
)


@dataclass(frozen=True)
class TransformComparisonConfig:
    seed: int = 260305
    n_train: int = 512
    n_test: int = 256
    n_features: int = 3
    hidden: int = 32
    epochs: int = 24
    lr: float = 5e-3


def _simulate(cfg: TransformComparisonConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    set_comparison_seed(cfg.seed)
    n_total = cfg.n_train + cfg.n_test
    x = torch.rand(n_total, cfg.n_features)
    linear = 1.2 * x[:, :1] + 0.8 * x[:, 1:2].square() + 0.4 * torch.sin(3.0 * x[:, 2:3])
    y_true = torch.exp(linear)
    y = y_true * torch.exp(0.35 * torch.randn_like(y_true))
    return x[: cfg.n_train], y[: cfg.n_train], x[cfg.n_train :], y[cfg.n_train :]


def _make_model(cfg: TransformComparisonConfig) -> nn.Module:
    return nn.Sequential(
        nn.Linear(cfg.n_features, cfg.hidden),
        nn.ReLU(),
        nn.Linear(cfg.hidden, cfg.hidden),
        nn.ReLU(),
        nn.Linear(cfg.hidden, 1),
        nn.Softplus(),
    )


def _train_model(
    model: nn.Module,
    loss_fn: nn.Module,
    x_train: Tensor,
    y_train: Tensor,
    *,
    epochs: int,
    lr: float,
) -> nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()
    return model.eval()


def _tail_mae(y_pred: Tensor, y_true: Tensor) -> float:
    threshold = torch.quantile(y_true, 0.8)
    mask = y_true >= threshold
    return float(torch.mean(torch.abs(y_pred[mask] - y_true[mask])).item())


def _mape(y_pred: Tensor, y_true: Tensor) -> float:
    return float(torch.mean(torch.abs(y_pred - y_true) / y_true.clamp_min(1e-6)).item())


def run_comparison(cfg: TransformComparisonConfig) -> tuple[list[dict[str, object]], list[str]]:
    x_train, y_train, x_test, y_test = _simulate(cfg)
    methods: list[tuple[str, nn.Module]] = [
        ("MSE", WeightedMSELoss()),
        ("LogTransform", LogTransformLoss()),
        ("BoxCox(0.25)", BoxCoxTransformLoss(lam=0.25)),
        ("SqrtTransform", SqrtTransformLoss()),
    ]
    rows: list[dict[str, object]] = []
    for method_name, loss_fn in methods:
        set_comparison_seed(cfg.seed)
        model = _make_model(cfg)
        trained_model, train_s = timed_call(
            _train_model,
            model,
            loss_fn,
            x_train,
            y_train,
            epochs=cfg.epochs,
            lr=cfg.lr,
        )
        y_pred, eval_s = timed_call(trained_model, x_test)
        metrics = compute_point_metrics(y_pred, y_test)
        rows.append(
            {
                "Method": method_name,
                **metrics,
                "MAPE": _mape(y_pred, y_test),
                "TailMAE80": _tail_mae(y_pred, y_test),
                "train_s": float(train_s),
                "eval_s": float(eval_s),
            }
        )

    notes = [
        "Synthetic task uses multiplicative noise and strong right-skew to stress target transforms.",
        "All methods share seed, architecture, optimizer, and epoch budget.",
        "Log/Box-Cox/Sqrt transforms require positive-support targets and positive model outputs.",
    ]
    return rows, notes


def main(
    cfg: TransformComparisonConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or TransformComparisonConfig()
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="Transformed-target comparison",
        seed_policy=f"fixed seed = {cfg.seed}",
        train_budget=f"shared MLP and {cfg.epochs} epochs",
        metric_policy="MSE, MAE, R2, MAPE, TailMAE80, runtime",
    )
    print_comparison_summary(
        "Transformed-target regression comparison",
        rows,
        metric_order=["MSE", "MAE", "R2", "MAPE", "TailMAE80", "train_s", "eval_s"],
    )

    if summary_json_path is not None:
        write_comparison_summary_json(
            summary_json_path,
            example="examples/transformed_target_regression_comparison.py",
            task="Target transforms for skewed / multiplicative-noise regression",
            config=cfg,
            rows=rows,
            notes=notes,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare transformed-target losses on skewed positive regression."
    )
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
