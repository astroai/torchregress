"""Shared-budget comparison for ordinal regression methods."""

import argparse
from dataclasses import dataclass

import torch
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    timed_call,
    write_comparison_summary_json,
)
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import CORALLoss, CumulativeLinkLoss, OrdinalCrossEntropyLoss
from torchregress.metrics import (
    mean_absolute_class_error,
    ordinal_accuracy,
    quadratic_weighted_kappa,
)
from torchregress.utils import ordinal_predict


@dataclass(frozen=True)
class OrdinalComparisonConfig:
    seed: int = 260227
    n_train: int = 512
    n_test: int = 256
    n_features: int = 6
    num_classes: int = 5
    hidden: int = 32
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-2


class _MLP(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _make_data(cfg: OrdinalComparisonConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(cfg.seed)
    x = torch.randn(cfg.n_train + cfg.n_test, cfg.n_features)

    w = torch.tensor([0.9, -0.7, 0.4, 0.2, -0.3, 0.1])[: cfg.n_features]
    latent = x @ w
    latent = latent + 0.35 * x[:, 0] * x[:, 1] - 0.2 * x[:, 2] ** 2
    latent = latent + 0.4 * torch.randn_like(latent)

    cutpoints = torch.tensor([-1.0, -0.25, 0.4, 1.1], dtype=latent.dtype)
    y = torch.bucketize(latent, cutpoints).long()

    x_train = x[: cfg.n_train]
    y_train = y[: cfg.n_train]
    x_test = x[cfg.n_train :]
    y_test = y[cfg.n_train :]
    return x_train, y_train, x_test, y_test


def _train_model(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    x_train: Tensor,
    y_train: Tensor,
    cfg: OrdinalComparisonConfig,
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()


def _evaluate(
    model: torch.nn.Module,
    x_test: Tensor,
    y_test: Tensor,
    *,
    encoding: str,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x_test)

    pred = ordinal_predict(logits, encoding=encoding)
    return {
        "Accuracy": float(ordinal_accuracy(pred, y_test, encoding="labels").item()),
        "OrdinalMAE": float(mean_absolute_class_error(pred, y_test, encoding="labels").item()),
        "QWK": float(quadratic_weighted_kappa(pred, y_test, encoding="labels").item()),
    }


def run_comparison(cfg: OrdinalComparisonConfig) -> list[dict[str, object]]:
    torch.manual_seed(cfg.seed)
    x_train, y_train, x_test, y_test = _make_data(cfg)

    rows: list[dict[str, object]] = []
    methods = [
        (
            "OrdinalCrossEntropy",
            _MLP(cfg.n_features, cfg.hidden, cfg.num_classes),
            OrdinalCrossEntropyLoss(),
            "class_logits",
            "class-logit baseline",
        ),
        (
            "CumulativeLink",
            _MLP(cfg.n_features, cfg.hidden, cfg.num_classes - 1),
            CumulativeLinkLoss(),
            "cumulative_logits",
            "cumulative-threshold ordinal objective",
        ),
        (
            "CORAL",
            _MLP(cfg.n_features, cfg.hidden, cfg.num_classes - 1),
            CORALLoss(),
            "cumulative_logits",
            "CORAL-style cumulative objective",
        ),
    ]

    for name, model, loss_fn, encoding, notes in methods:
        _, train_s = timed_call(_train_model, model, loss_fn, x_train, y_train, cfg)
        metrics, eval_s = timed_call(_evaluate, model, x_test, y_test, encoding=encoding)
        rows.append(
            {
                "Method": name,
                **metrics,
                "train_s": float(train_s),
                "eval_s": float(eval_s),
                "Notes": notes,
            }
        )

    return rows


def main(cfg: OrdinalComparisonConfig | None = None, summary_json_path: str | None = None) -> None:
    cfg = cfg or OrdinalComparisonConfig()
    rows = run_comparison(cfg)

    print_fairness_notes(
        title="Ordinal Regression Comparison",
        seed_policy="fixed seed and shared synthetic split",
        train_budget="same MLP depth/width and epochs across methods",
        metric_policy="accuracy, ordinal class-MAE, QWK, runtime",
    )
    print_comparison_summary(
        "Ordinal method summary",
        rows,
        metric_order=["Accuracy", "OrdinalMAE", "QWK", "train_s", "eval_s"],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/ordinal_regression_comparison.py",
            task="Ordinal regression / ordered targets",
            config=cfg,
            rows=rows,
            notes=[
                "All methods share architecture and training budget.",
                "CumulativeLink/CORAL use K-1 cumulative logits; CE uses K class logits.",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ordinal regression comparison example")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
