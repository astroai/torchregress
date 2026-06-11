"""Shared-budget comparison for censored regression losses on real tabular data."""

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
from torchregress.losses import AFTLoss, CensoredGaussianNLLLoss, CensoredQuantileLoss
from torchregress.metrics import censoring_rate, concordance_index, observed_mae


@dataclass(frozen=True)
class CensoredRealDataConfig:
    seed: int = 260305
    n_train: int = 320
    n_test: int = 120
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


def _make_realdata(cfg: CensoredRealDataConfig) -> dict[str, Tensor]:
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

    right_mask = y_true > right_limit
    left_mask = (~right_mask) & (y_true < left_limit)
    observed_mask = ~(right_mask | left_mask)

    y_obs = y_true.clone()
    y_obs[right_mask] = right_limit[right_mask]
    y_obs[left_mask] = left_limit[left_mask]

    censor = torch.zeros_like(y_true, dtype=torch.int64)
    censor[right_mask] = 1
    censor[left_mask] = -1

    interval_mask = observed_mask & (torch.rand(y_true.shape, generator=g) < 0.10)
    lb = torch.full_like(y_true, float("nan"))
    ub = torch.full_like(y_true, float("nan"))
    lb[interval_mask] = y_true[interval_mask] * 0.9
    ub[interval_mask] = y_true[interval_mask] * 1.1

    split = cfg.n_train
    return {
        "x_train": x[:split],
        "x_test": x[split:],
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


def _train(
    model: torch.nn.Module,
    loss_name: str,
    data: dict[str, Tensor],
    cfg: CensoredRealDataConfig,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ds = TensorDataset(
        data["x_train"],
        data["y_obs_train"],
        data["c_train"],
        data["lb_train"],
        data["ub_train"],
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

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
    if loss_name in {"gaussian", "quantile"}:
        return torch.nan_to_num(out[:, 0], nan=0.0, posinf=1e3, neginf=-1e3)
    return torch.exp(torch.nan_to_num(out[:, 0], nan=0.0, posinf=7.0, neginf=-7.0)).clamp(max=1e3)


def run_comparison(cfg: CensoredRealDataConfig) -> list[dict[str, object]]:
    data = _make_realdata(cfg)
    n_features = int(data["x_train"].shape[1])

    methods = [
        ("CensoredGaussianNLL", _MLP(n_features, cfg.hidden, 2), "gaussian"),
        ("CensoredQuantile", _MLP(n_features, cfg.hidden, 1), "quantile"),
        ("AFT", _MLP(n_features, cfg.hidden, 2), "aft"),
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
                "Notes": "real data targets with synthetic left/right + interval censoring",
            }
        )

    return rows


def main(
    cfg: CensoredRealDataConfig | None = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or CensoredRealDataConfig()
    rows = run_comparison(cfg)

    print_fairness_notes(
        title="Censored Regression Comparison (Real Data)",
        seed_policy="fixed seed and shared Diabetes split with identical censoring masks",
        train_budget="same MLP capacity and epochs across methods",
        metric_policy="true-target MAE, observed MAE, concordance index, runtime",
    )
    print_comparison_summary(
        "Censored method summary (real data)",
        rows,
        metric_order=["MAE_true", "ObsMAE", "CIndex", "CensorRate", "train_s", "eval_s"],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/censored_regression_realdata_comparison.py",
            task="Censored regression / interval-censored targets (real-data)",
            config=cfg,
            rows=rows,
            notes=[
                "Uses Diabetes covariates/targets with synthetic censoring overlays.",
                "Censor coding: 0 observed / 1 right / -1 left; interval subset uses bounds.",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run censored regression real-data comparison")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(summary_json_path=args.summary_json_path)
