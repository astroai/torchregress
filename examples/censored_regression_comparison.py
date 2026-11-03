"""Shared-budget comparison for censored regression losses.

Two datasets share the models, losses, and metrics; each keeps its own fixed seed and size:

- ``--dataset synthetic`` (default): simulated log-normal event times with random
  left/right censoring limits and a small interval-censored subset.
- ``--dataset diabetes``: real ``sklearn`` Diabetes covariates/targets with synthetic
  quantile-based left/right censoring and an interval-censored subset.
"""

import argparse
from collections.abc import Callable
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
from torchregress.losses import AFTLoss, CensoredGaussianNLLLoss, CensoredQuantileLoss
from torchregress.metrics import censoring_rate, concordance_index, observed_mae

EXAMPLE_PATH = "examples/censored_regression_comparison.py"


@dataclass(frozen=True)
class CensoredComparisonConfig:
    """Synthetic configuration (``--dataset synthetic``)."""

    seed: int = 260227
    n_train: int = 768
    n_test: int = 256
    n_features: int = 6
    hidden: int = 32
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-2


@dataclass(frozen=True)
class CensoredRealDataConfig:
    """Diabetes configuration (``--dataset diabetes``)."""

    seed: int = 260305
    n_train: int = 320
    n_test: int = 120
    hidden: int = 32
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-2


AnyConfig = CensoredComparisonConfig | CensoredRealDataConfig


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


# ---------------------------------------------------------------------------
# Datasets: both return the same train/test dictionary layout.
# ---------------------------------------------------------------------------


def _censor_and_split(
    y_true: Tensor,
    right_limit: Tensor,
    left_limit: Tensor,
    interval_mask_draw: Callable[[Tensor], Tensor],
    *,
    interval_factors: tuple[float, float],
    split: int,
) -> dict[str, Tensor]:
    """Apply left/right censoring limits plus an interval-censored subset, then split."""
    right_mask = y_true > right_limit
    left_mask = (~right_mask) & (y_true < left_limit)
    observed_mask = ~(right_mask | left_mask)

    y_obs = y_true.clone()
    y_obs[right_mask] = right_limit[right_mask]
    y_obs[left_mask] = left_limit[left_mask]

    censor = torch.zeros_like(y_true, dtype=torch.int64)
    censor[right_mask] = 1
    censor[left_mask] = -1

    # Small interval-censored subset for explicit bound-path coverage.
    interval_mask = observed_mask & interval_mask_draw(y_true)
    lb = torch.full_like(y_true, float("nan"))
    ub = torch.full_like(y_true, float("nan"))
    lb[interval_mask] = y_true[interval_mask] * interval_factors[0]
    ub[interval_mask] = y_true[interval_mask] * interval_factors[1]

    return {
        "y_true_train": y_true[:split],
        "y_true_test": y_true[split:],
        "y_obs_train": y_obs[:split],
        "y_obs_test": y_obs[split:],
        "c_train": censor[:split],
        "c_test": censor[split:],
        "lb_train": lb[:split],
        "ub_train": ub[:split],
        "lb_test": lb[split:],
        "ub_test": ub[split:],
    }


def _simulate(cfg: CensoredComparisonConfig) -> dict[str, Tensor]:
    torch.manual_seed(cfg.seed)
    n = cfg.n_train + cfg.n_test
    x = torch.randn(n, cfg.n_features)

    w = torch.tensor([0.9, -0.6, 0.35, 0.2, -0.25, 0.15])[: cfg.n_features]
    latent = x @ w + 0.3 * x[:, 0] * x[:, 1] + 0.4 * torch.randn(n)
    true_t = torch.exp(latent)

    right_limit = torch.exp(0.6 * torch.randn(n) + 0.6)
    left_limit = torch.exp(0.6 * torch.randn(n) - 0.8)

    split = cfg.n_train
    data = _censor_and_split(
        true_t,
        right_limit,
        left_limit,
        lambda y: torch.rand(n) < 0.12,
        interval_factors=(0.85, 1.15),
        split=split,
    )
    return {"x_train": x[:split], "x_test": x[split:], **data}


def _make_realdata(cfg: CensoredRealDataConfig) -> dict[str, Tensor]:
    # Seeds the global RNG used by censoring draws and model init, as _simulate does.
    torch.manual_seed(cfg.seed)
    x_np, y_np = load_diabetes(return_X_y=True)
    x_all = torch.tensor(x_np, dtype=torch.float32)
    y_all_raw = torch.tensor(y_np, dtype=torch.float32)

    need = cfg.n_train + cfg.n_test
    if need > x_all.shape[0]:
        raise ValueError(f"Requested {need} samples but diabetes dataset has {x_all.shape[0]}.")

    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(x_all.shape[0], generator=g)[:need]
    x = x_all[perm]
    y_raw = y_all_raw[perm]

    # Shift to positive support for AFT while preserving ordering.
    y_std = (y_raw - y_raw.mean()) / y_raw.std(unbiased=False).clamp_min(1e-6)
    y_true = (y_std + 3.5).clamp_min(1e-3)

    right_q = float(torch.quantile(y_true[: cfg.n_train], 0.78).item())
    left_q = float(torch.quantile(y_true[: cfg.n_train], 0.22).item())

    feature_r = torch.abs(x[:, 0])
    feature_l = torch.abs(x[:, 1])
    right_limit = (right_q + 0.35 * feature_r).clamp_min(1e-3)
    left_limit = (left_q - 0.25 * feature_l).clamp_min(1e-3)

    split = cfg.n_train
    data = _censor_and_split(
        y_true,
        right_limit,
        left_limit,
        lambda y: torch.rand(y.shape, generator=g) < 0.10,
        interval_factors=(0.9, 1.1),
        split=split,
    )
    return {"x_train": x[:split], "x_test": x[split:], **data}


@dataclass(frozen=True)
class DatasetSpec:
    config_cls: type[AnyConfig]
    make_data: Callable[..., dict[str, Tensor]]
    # Synthetic shuffles from the global RNG seeded in ``_simulate``; the Diabetes
    # track gives every method an identically seeded loader generator.
    seeded_loader: bool
    title: str
    seed_policy: str
    summary_title: str
    task: str
    row_note: str
    notes: tuple[str, ...]


DATASETS: dict[str, DatasetSpec] = {
    "synthetic": DatasetSpec(
        config_cls=CensoredComparisonConfig,
        make_data=_simulate,
        seeded_loader=False,
        title="Censored Regression Comparison",
        seed_policy="fixed seed and shared censoring split",
        summary_title="Censored method summary",
        task="Censored regression / interval-censored targets",
        row_note="shared architecture and censoring split",
        notes=(
            "Censor coding uses 0 observed / 1 right / -1 left.",
            "Interval-censored subset uses explicit lower/upper bounds.",
        ),
    ),
    "diabetes": DatasetSpec(
        config_cls=CensoredRealDataConfig,
        make_data=_make_realdata,
        seeded_loader=True,
        title="Censored Regression Comparison (Real Data)",
        seed_policy="fixed seed and shared Diabetes split with identical censoring masks",
        summary_title="Censored method summary (real data)",
        task="Censored regression / interval-censored targets (real-data)",
        row_note="real data targets with synthetic left/right + interval censoring",
        notes=(
            "Uses Diabetes covariates/targets with synthetic censoring overlays.",
            "Censor coding: 0 observed / 1 right / -1 left; interval subset uses bounds.",
        ),
    ),
}


def _dataset_for_config(cfg: AnyConfig) -> DatasetSpec:
    for spec in DATASETS.values():
        if type(cfg) is spec.config_cls:
            return spec
    raise TypeError(f"Unsupported config type: {type(cfg).__name__}")


# ---------------------------------------------------------------------------
# Shared training and evaluation
# ---------------------------------------------------------------------------


def _make_loss(loss_name: str) -> torch.nn.Module:
    if loss_name == "gaussian":
        return CensoredGaussianNLLLoss()
    if loss_name == "quantile":
        return CensoredQuantileLoss(quantile=0.5)
    if loss_name == "aft":
        return AFTLoss()
    raise ValueError(f"Unknown loss_name: {loss_name}")


def _loss_input(out: Tensor, loss_name: str) -> Tensor | tuple[Tensor, Tensor]:
    """Gaussian: (mean, log_var); AFT: (loc, log_scale); quantile: point prediction."""
    if loss_name == "quantile":
        return out[:, 0]
    return out[:, 0], out[:, 1].clamp(min=-8.0, max=8.0)


def _train(
    model: torch.nn.Module,
    loss_name: str,
    data: dict[str, Tensor],
    cfg: AnyConfig,
    *,
    seeded_loader: bool,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ds = TensorDataset(
        data["x_train"],
        data["y_obs_train"],
        data["c_train"],
        data["lb_train"],
        data["ub_train"],
    )
    generator = torch.Generator().manual_seed(cfg.seed) if seeded_loader else None
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    loss_fn = _make_loss(loss_name)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb, cb, lb, ub in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(
                _loss_input(model(xb), loss_name),
                yb,
                censoring=cb,
                lower_bound=lb,
                upper_bound=ub,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()


def _predict(model: torch.nn.Module, x: Tensor, *, loss_name: str) -> Tensor:
    model.eval()
    with torch.no_grad():
        out = model(x)
    if loss_name in {"gaussian", "quantile"}:
        return torch.nan_to_num(out[:, 0], nan=0.0, posinf=1e3, neginf=-1e3)
    # AFT: E[time] proxy from median exp(loc)
    return torch.exp(torch.nan_to_num(out[:, 0], nan=0.0, posinf=7.0, neginf=-7.0)).clamp(max=1e3)


def run_comparison(cfg: AnyConfig) -> list[dict[str, object]]:
    spec = _dataset_for_config(cfg)
    data = spec.make_data(cfg)
    n_features = int(data["x_train"].shape[1])

    methods = [
        ("CensoredGaussianNLL", _MLP(n_features, cfg.hidden, 2), "gaussian"),
        ("CensoredQuantile", _MLP(n_features, cfg.hidden, 1), "quantile"),
        ("AFT", _MLP(n_features, cfg.hidden, 2), "aft"),
    ]

    rows: list[dict[str, object]] = []
    for name, model, loss_name in methods:
        _, train_s = timed_call(
            _train, model, loss_name, data, cfg, seeded_loader=spec.seeded_loader
        )
        pred, eval_s = timed_call(_predict, model, data["x_test"], loss_name=loss_name)

        mae_true = torch.mean(torch.abs(pred - data["y_true_test"]))
        obs_mae = observed_mae(pred, data["y_obs_test"], data["c_test"])
        c_idx = concordance_index(pred, data["y_obs_test"], data["c_test"])

        rows.append(
            {
                "Method": name,
                "MAE_true": float(mae_true.item()),
                "ObsMAE": float(obs_mae.item()) if obs_mae == obs_mae else float("nan"),
                "CIndex": float(c_idx.item()) if c_idx == c_idx else float("nan"),
                "CensorRate": float(censoring_rate(data["c_test"]).item()),
                "train_s": float(train_s),
                "eval_s": float(eval_s),
                "Notes": spec.row_note,
            }
        )

    return rows


def main(
    cfg: AnyConfig | None = None,
    summary_json_path: str | None = None,
    *,
    dataset: str = "synthetic",
) -> None:
    """Run the comparison; ``cfg`` (if given) selects the dataset by its type."""
    cfg = cfg or DATASETS[dataset].config_cls()
    spec = _dataset_for_config(cfg)
    rows = run_comparison(cfg)

    print_fairness_notes(
        title=spec.title,
        seed_policy=spec.seed_policy,
        train_budget="same MLP capacity and epochs across methods",
        metric_policy="true-target MAE, observed MAE, concordance index, runtime",
    )
    print_comparison_summary(
        spec.summary_title,
        rows,
        metric_order=["MAE_true", "ObsMAE", "CIndex", "CensorRate", "train_s", "eval_s"],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example=EXAMPLE_PATH,
            task=spec.task,
            config=cfg,
            rows=rows,
            notes=list(spec.notes),
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run censored regression comparison")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="synthetic")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(dataset=args.dataset, summary_json_path=args.summary_json_path)
