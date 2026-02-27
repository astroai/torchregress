"""Shared-budget comparison for selection bias and long-tail regression."""

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

from torchregress.losses import DensityWeightedLoss, PropensityWeightedLoss, WeightedMSELoss
from torchregress.metrics import mean_absolute_error, tail_mae, tail_rmse
from torchregress.utils import PropensityEstimator


@dataclass(frozen=True)
class PropensityTailConfig:
    seed: int = 260227
    n_train_pool: int = 1200
    n_test: int = 400
    n_features: int = 6
    hidden: int = 32
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-2


class _MLP(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def _simulate(cfg: PropensityTailConfig) -> dict[str, Tensor]:
    torch.manual_seed(cfg.seed)
    n = cfg.n_train_pool + cfg.n_test
    x = torch.randn(n, cfg.n_features)

    w = torch.tensor([1.0, -0.8, 0.5, 0.3, -0.2, 0.15])[: cfg.n_features]
    y = x @ w + 0.5 * x[:, 0] * x[:, 1] + 0.2 * x[:, 2] ** 2 + 0.4 * torch.randn(n)

    # Selection depends on covariates -> biased observed training labels.
    logits = 1.4 * x[:, 0] - 1.0 * x[:, 1] - 0.4
    p_obs = torch.sigmoid(logits).clamp(0.05, 0.95)
    observed = torch.bernoulli(p_obs).long()

    split = cfg.n_train_pool
    x_pool = x[:split]
    y_pool = y[:split]
    obs_pool = observed[:split]

    x_test = x[split:]
    y_test = y[split:]

    obs_idx = torch.nonzero(obs_pool == 1, as_tuple=False).reshape(-1)
    x_obs = x_pool[obs_idx]
    y_obs = y_pool[obs_idx]

    return {
        "x_pool": x_pool,
        "y_pool": y_pool,
        "obs_pool": obs_pool,
        "x_obs": x_obs,
        "y_obs": y_obs,
        "obs_idx": obs_idx,
        "x_test": x_test,
        "y_test": y_test,
    }


def _train_mse(model: torch.nn.Module, data: dict[str, Tensor], cfg: PropensityTailConfig) -> None:
    loss_fn = WeightedMSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ds = TensorDataset(data["x_obs"], data["y_obs"])
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()


def _train_density(model: torch.nn.Module, data: dict[str, Tensor], cfg: PropensityTailConfig) -> None:
    loss_fn = DensityWeightedLoss(kernel_width=0.5, reweight_factor=1.0)
    loss_fn.fit_density(data["y_obs"])

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    local_idx = torch.arange(data["x_obs"].shape[0])
    ds = TensorDataset(data["x_obs"], data["y_obs"], local_idx)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb, ib in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb, sample_indices=ib)
            loss.backward()
            opt.step()


def _train_propensity(
    model: torch.nn.Module,
    data: dict[str, Tensor],
    cfg: PropensityTailConfig,
) -> None:
    estimator = PropensityEstimator()
    estimator.fit(data["x_pool"], data["obs_pool"])
    p_obs = estimator.predict_proba(data["x_obs"])

    loss_fn = PropensityWeightedLoss(base_loss="mse")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ds = TensorDataset(data["x_obs"], data["y_obs"], p_obs)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb, pb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb, propensity=pb)
            loss.backward()
            opt.step()


def _evaluate(model: torch.nn.Module, x_test: Tensor, y_test: Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred = model(x_test)

    mae = mean_absolute_error(pred, y_test)
    tail_mae_90 = tail_mae(pred, y_test, quantile=0.9, tail="upper")
    tail_rmse_90 = tail_rmse(pred, y_test, quantile=0.9, tail="upper")
    return {
        "MAE": float(mae),
        "TailMAE90": float(tail_mae_90),
        "TailRMSE90": float(tail_rmse_90),
    }


def run_comparison(cfg: PropensityTailConfig) -> tuple[list[dict[str, object]], list[str]]:
    data = _simulate(cfg)
    obs_rate = float((data["obs_pool"] == 1).float().mean().item())

    methods = [
        ("MSE", _train_mse, "naive observed-only baseline"),
        ("DensityWeighted", _train_density, "target-density reweighting"),
        ("PropensityWeighted", _train_propensity, "IPW using p(observed|x)"),
    ]

    rows: list[dict[str, object]] = []
    for name, train_fn, notes in methods:
        model = _MLP(cfg.n_features, cfg.hidden)
        _, train_s = timed_call(train_fn, model, data, cfg)
        metrics, eval_s = timed_call(_evaluate, model, data["x_test"], data["y_test"])
        rows.append(
            {
                "Method": name,
                **metrics,
                "ObservedRate": obs_rate,
                "train_s": float(train_s),
                "eval_s": float(eval_s),
                "Notes": notes,
            }
        )

    notes = [
        "Training labels are observed with covariate-dependent probability.",
        "Tail metrics evaluate upper 10% target regime on unbiased test data.",
    ]
    return rows, notes


def main(cfg: PropensityTailConfig | None = None, summary_json_path: str | None = None) -> None:
    cfg = cfg or PropensityTailConfig()
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="Propensity + Tail Regression Comparison",
        seed_policy="fixed seed and shared selection process",
        train_budget="same MLP capacity and epochs across methods",
        metric_policy="MAE + upper-tail MAE/RMSE + runtime",
    )
    print_comparison_summary(
        "Propensity/tail summary",
        rows,
        metric_order=["MAE", "TailMAE90", "TailRMSE90", "ObservedRate", "train_s", "eval_s"],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/propensity_tail_regression_comparison.py",
            task="Selection bias + long-tail regression",
            config=cfg,
            rows=rows,
            notes=notes,
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run propensity/tail regression comparison")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
