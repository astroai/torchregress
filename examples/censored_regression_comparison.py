"""Shared-budget comparison for censored regression losses."""

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

from torchregress.losses import AFTLoss, CensoredGaussianNLLLoss, CensoredQuantileLoss
from torchregress.metrics import censoring_rate, concordance_index, observed_mae


@dataclass(frozen=True)
class CensoredComparisonConfig:
    seed: int = 260227
    n_train: int = 768
    n_test: int = 256
    n_features: int = 6
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


def _simulate(cfg: CensoredComparisonConfig) -> dict[str, Tensor]:
    torch.manual_seed(cfg.seed)
    n = cfg.n_train + cfg.n_test
    x = torch.randn(n, cfg.n_features)

    w = torch.tensor([0.9, -0.6, 0.35, 0.2, -0.25, 0.15])[: cfg.n_features]
    latent = x @ w + 0.3 * x[:, 0] * x[:, 1] + 0.4 * torch.randn(n)
    true_t = torch.exp(latent)

    right_limit = torch.exp(0.6 * torch.randn(n) + 0.6)
    left_limit = torch.exp(0.6 * torch.randn(n) - 0.8)

    right_mask = true_t > right_limit
    left_mask = (~right_mask) & (true_t < left_limit)
    observed_mask = ~(right_mask | left_mask)

    observed_t = true_t.clone()
    observed_t[right_mask] = right_limit[right_mask]
    observed_t[left_mask] = left_limit[left_mask]

    censoring = torch.zeros(n, dtype=torch.int64)
    censoring[right_mask] = 1
    censoring[left_mask] = -1

    # Small interval-censored subset for explicit bound-path coverage.
    interval_mask = observed_mask & (torch.rand(n) < 0.12)
    lower_bound = torch.full_like(true_t, float("nan"))
    upper_bound = torch.full_like(true_t, float("nan"))
    lower_bound[interval_mask] = true_t[interval_mask] * 0.85
    upper_bound[interval_mask] = true_t[interval_mask] * 1.15

    split = cfg.n_train
    return {
        "x_train": x[:split],
        "x_test": x[split:],
        "y_true_train": true_t[:split],
        "y_true_test": true_t[split:],
        "y_obs_train": observed_t[:split],
        "y_obs_test": observed_t[split:],
        "c_train": censoring[:split],
        "c_test": censoring[split:],
        "lb_train": lower_bound[:split],
        "ub_train": upper_bound[:split],
        "lb_test": lower_bound[split:],
        "ub_test": upper_bound[split:],
    }


def _train(
    model: torch.nn.Module,
    loss_name: str,
    data: dict[str, Tensor],
    cfg: CensoredComparisonConfig,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ds = TensorDataset(
        data["x_train"],
        data["y_obs_train"],
        data["c_train"],
        data["lb_train"],
        data["ub_train"],
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    if loss_name == "gaussian":
        loss_fn = CensoredGaussianNLLLoss()
    elif loss_name == "quantile":
        loss_fn = CensoredQuantileLoss(quantile=0.5)
    elif loss_name == "aft":
        loss_fn = AFTLoss()
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb, cb, lb, ub in loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            if loss_name == "gaussian":
                mean = out[:, 0]
                log_var = out[:, 1].clamp(min=-8.0, max=8.0)
                loss = loss_fn(
                    (mean, log_var),
                    yb,
                    censoring=cb,
                    lower_bound=lb,
                    upper_bound=ub,
                )
            elif loss_name == "quantile":
                pred = out[:, 0]
                loss = loss_fn(
                    pred,
                    yb,
                    censoring=cb,
                    lower_bound=lb,
                    upper_bound=ub,
                )
            else:
                loc = out[:, 0]
                log_scale = out[:, 1].clamp(min=-8.0, max=8.0)
                loss = loss_fn(
                    (loc, log_scale),
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
    if loss_name == "gaussian":
        return torch.nan_to_num(out[:, 0], nan=0.0, posinf=1e3, neginf=-1e3)
    if loss_name == "quantile":
        return torch.nan_to_num(out[:, 0], nan=0.0, posinf=1e3, neginf=-1e3)
    # AFT: E[time] proxy from median exp(loc)
    return torch.exp(torch.nan_to_num(out[:, 0], nan=0.0, posinf=7.0, neginf=-7.0)).clamp(
        max=1e3
    )


def run_comparison(cfg: CensoredComparisonConfig) -> list[dict[str, object]]:
    data = _simulate(cfg)

    methods = [
        ("CensoredGaussianNLL", _MLP(cfg.n_features, cfg.hidden, 2), "gaussian"),
        ("CensoredQuantile", _MLP(cfg.n_features, cfg.hidden, 1), "quantile"),
        ("AFT", _MLP(cfg.n_features, cfg.hidden, 2), "aft"),
    ]

    rows: list[dict[str, object]] = []
    for name, model, loss_name in methods:
        _, train_s = timed_call(_train, model, loss_name, data, cfg)
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
                "Notes": "shared architecture and censoring split",
            }
        )

    return rows


def main(cfg: CensoredComparisonConfig | None = None, summary_json_path: str | None = None) -> None:
    cfg = cfg or CensoredComparisonConfig()
    rows = run_comparison(cfg)

    print_fairness_notes(
        title="Censored Regression Comparison",
        seed_policy="fixed seed and shared censoring split",
        train_budget="same MLP capacity and epochs across methods",
        metric_policy="true-target MAE, observed MAE, concordance index, runtime",
    )
    print_comparison_summary(
        "Censored method summary",
        rows,
        metric_order=["MAE_true", "ObsMAE", "CIndex", "CensorRate", "train_s", "eval_s"],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/censored_regression_comparison.py",
            task="Censored regression / interval-censored targets",
            config=cfg,
            rows=rows,
            notes=[
                "Censor coding uses 0 observed / 1 right / -1 left.",
                "Interval-censored subset uses explicit lower/upper bounds.",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run censored regression comparison")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
