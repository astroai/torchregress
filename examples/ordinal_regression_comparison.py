"""Shared-budget comparison for ordinal regression methods.

Run on a synthetic latent-variable task (default) or on quantile-binned Diabetes
targets::

    python examples/ordinal_regression_comparison.py --dataset synthetic
    python examples/ordinal_regression_comparison.py --dataset diabetes

Each dataset keeps its own fixed seed and split sizes; the models, training loop,
and metrics are shared.
"""

import argparse
from dataclasses import dataclass

import torch
from sklearn.datasets import load_diabetes
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.losses import CumulativeLinkLoss, OrdinalCrossEntropyLoss
from torchregress.metrics import (
    mean_absolute_class_error,
    ordinal_accuracy,
    quadratic_weighted_kappa,
)
from torchregress.utils import ordinal_predict

DATASETS = ("synthetic", "diabetes")
EXAMPLE = "examples/ordinal_regression_comparison.py"


@dataclass(frozen=True)
class OrdinalComparisonConfig:
    """Synthetic latent-variable task (``--dataset synthetic``)."""

    seed: int = 260227
    n_train: int = 512
    n_test: int = 256
    n_features: int = 6
    num_classes: int = 5
    hidden: int = 32
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-2


@dataclass(frozen=True)
class OrdinalRealDataConfig:
    """sklearn Diabetes targets binned at train quantiles (``--dataset diabetes``)."""

    seed: int = 260305
    n_train: int = 320
    n_test: int = 120
    num_classes: int = 5
    hidden: int = 32
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-2


OrdinalConfig = OrdinalComparisonConfig | OrdinalRealDataConfig


def default_config(dataset: str) -> OrdinalConfig:
    if dataset == "synthetic":
        return OrdinalComparisonConfig()
    if dataset == "diabetes":
        return OrdinalRealDataConfig()
    raise ValueError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}.")


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


def _make_synthetic_data(cfg: OrdinalComparisonConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(cfg.seed)
    x = torch.randn(cfg.n_train + cfg.n_test, cfg.n_features)

    w = torch.tensor([0.9, -0.7, 0.4, 0.2, -0.3, 0.1])[: cfg.n_features]
    latent = x @ w
    latent = latent + 0.35 * x[:, 0] * x[:, 1] - 0.2 * x[:, 2] ** 2
    latent = latent + 0.4 * torch.randn_like(latent)

    cutpoints = torch.tensor([-1.0, -0.25, 0.4, 1.1], dtype=latent.dtype)
    y = torch.bucketize(latent, cutpoints).long()
    return x[: cfg.n_train], y[: cfg.n_train], x[cfg.n_train :], y[cfg.n_train :]


def _make_diabetes_data(cfg: OrdinalRealDataConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    x_np, y_np = load_diabetes(return_X_y=True)
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    need = cfg.n_train + cfg.n_test
    if need > x.shape[0]:
        raise ValueError(f"Requested {need} samples but diabetes dataset has {x.shape[0]}.")

    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(x.shape[0], generator=g)[:need]
    x = x[perm]
    y = y[perm]

    # Ordinal classes from train-only quantile cutpoints (no test leakage).
    quantiles = torch.linspace(0.0, 1.0, cfg.num_classes + 1, dtype=y.dtype)
    cutpoints = torch.quantile(y[: cfg.n_train], quantiles[1:-1])
    y_ord = torch.bucketize(y, cutpoints).long()
    return x[: cfg.n_train], y_ord[: cfg.n_train], x[cfg.n_train :], y_ord[cfg.n_train :]


def _make_data(cfg: OrdinalConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if isinstance(cfg, OrdinalRealDataConfig):
        return _make_diabetes_data(cfg)
    return _make_synthetic_data(cfg)


def _train_model(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    x_train: Tensor,
    y_train: Tensor,
    cfg: OrdinalConfig,
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # The synthetic run shuffles from the global RNG seeded in run_comparison; the
    # Diabetes run uses its own generator. Both are deterministic for a fixed seed.
    generator = (
        torch.Generator().manual_seed(cfg.seed) if isinstance(cfg, OrdinalRealDataConfig) else None
    )
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
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


def run_comparison(cfg: OrdinalConfig) -> list[dict[str, object]]:
    torch.manual_seed(cfg.seed)
    x_train, y_train, x_test, y_test = _make_data(cfg)
    n_features = int(x_train.shape[1])
    is_real = isinstance(cfg, OrdinalRealDataConfig)

    methods = [
        (
            "OrdinalCrossEntropy",
            _MLP(n_features, cfg.hidden, cfg.num_classes),
            OrdinalCrossEntropyLoss(),
            "class_logits",
            "class-logit baseline on quantile-binned Diabetes targets"
            if is_real
            else "class-logit baseline",
        ),
        (
            "CumulativeLink",
            _MLP(n_features, cfg.hidden, cfg.num_classes - 1),
            CumulativeLinkLoss(),
            "cumulative_logits",
            "cumulative-threshold ordinal objective",
        ),
    ]

    rows: list[dict[str, object]] = []
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


def main(
    cfg: OrdinalConfig | None = None,
    summary_json_path: str | None = None,
    *,
    dataset: str = "synthetic",
) -> None:
    """Run the comparison; ``cfg`` (when given) determines the dataset."""
    cfg = cfg or default_config(dataset)
    is_real = isinstance(cfg, OrdinalRealDataConfig)
    rows = run_comparison(cfg)

    suffix = " (Real Data)" if is_real else ""
    print_fairness_notes(
        title=f"Ordinal Regression Comparison{suffix}",
        seed_policy=(
            "fixed seed and shared Diabetes split with train-quantile binning"
            if is_real
            else "fixed seed and shared synthetic split"
        ),
        train_budget="same MLP depth/width and epochs across methods",
        metric_policy="accuracy, ordinal class-MAE, QWK, runtime",
    )
    print_comparison_summary(
        "Ordinal method summary (real data)" if is_real else "Ordinal method summary",
        rows,
        metric_order=["Accuracy", "OrdinalMAE", "QWK", "train_s", "eval_s"],
    )

    if summary_json_path is not None:
        if is_real:
            notes = [
                "Targets are quantile-binned from Diabetes continuous labels using train-only cutpoints.",
                "All methods share architecture and training budget.",
            ]
        else:
            notes = [
                "All methods share architecture and training budget.",
                "CumulativeLink/CORAL use K-1 cumulative logits; CE uses K class logits.",
            ]
        out = write_comparison_summary_json(
            summary_json_path,
            example=f"{EXAMPLE} --dataset diabetes" if is_real else EXAMPLE,
            task=(
                "Ordinal regression / ordered targets (real-data)"
                if is_real
                else "Ordinal regression / ordered targets"
            ),
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ordinal regression comparison example")
    parser.add_argument("--dataset", choices=DATASETS, default="synthetic")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path, dataset=args.dataset)
