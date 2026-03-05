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

from torchregress.losses import (
    DensityWeightedLoss,
    GaussianNLLLoss,
    MultiQuantileLoss,
    PropensityWeightedLoss,
    WeightedMSELoss,
)
from torchregress.metrics import mean_absolute_error, prediction_interval_coverage, tail_mae, tail_rmse
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
    alpha: float = 0.1


class _MLP(torch.nn.Module):
    def __init__(self, n_features: int, hidden: int, out_dim: int = 1) -> None:
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
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()


def _train_density(
    model: torch.nn.Module, data: dict[str, Tensor], cfg: PropensityTailConfig
) -> None:
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
            pred = model(xb).squeeze(-1)
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
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb, propensity=pb)
            loss.backward()
            opt.step()


def _train_gaussian(
    model: torch.nn.Module,
    data: dict[str, Tensor],
    cfg: PropensityTailConfig,
) -> None:
    loss_fn = GaussianNLLLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ds = TensorDataset(data["x_obs"], data["y_obs"])
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            raw = model(xb)
            mean = raw[:, :1]
            log_var = raw[:, 1:2].clamp(min=-8.0, max=4.0)
            pred = torch.cat([mean, log_var], dim=-1)
            loss = loss_fn(pred, yb.unsqueeze(-1))
            loss.backward()
            opt.step()


def _train_quantile(
    model: torch.nn.Module,
    data: dict[str, Tensor],
    cfg: PropensityTailConfig,
) -> None:
    loss_fn = MultiQuantileLoss(quantiles=[0.05, 0.5, 0.95], joint_prediction=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ds = TensorDataset(data["x_obs"], data["y_obs"])
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb.unsqueeze(-1))
            loss.backward()
            opt.step()


def _tail_coverage(
    lower: Tensor,
    upper: Tensor,
    y_true: Tensor,
    *,
    quantile: float = 0.9,
) -> float:
    threshold = torch.quantile(y_true, quantile)
    mask = y_true >= threshold
    if int(mask.sum().item()) == 0:
        return float("nan")
    covered = ((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])).float().mean()
    return float(covered.item())


def _evaluate(
    model: torch.nn.Module,
    x_test: Tensor,
    y_test: Tensor,
    *,
    method: str,
    alpha: float,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        raw = model(x_test)

    native_cov = float("nan")
    native_width = float("nan")
    tail_cov = float("nan")

    if method == "GaussianNLL":
        mean = raw[:, 0]
        log_var = raw[:, 1].clamp(min=-8.0, max=4.0)
        std = torch.exp(0.5 * log_var).clamp_min(1e-6)
        pred = mean
        z = torch.distributions.Normal(0.0, 1.0).icdf(
            torch.tensor(1.0 - alpha / 2.0, dtype=std.dtype, device=std.device)
        )
        lower = mean - z * std
        upper = mean + z * std
        native_cov = float(prediction_interval_coverage(lower, upper, y_test).item())
        native_width = float(torch.mean(upper - lower).item())
        tail_cov = _tail_coverage(lower, upper, y_test, quantile=0.9)
    elif method == "Quantile90":
        q = raw.reshape(raw.shape[0], 3)
        q_sorted, _ = torch.sort(q, dim=-1)
        lower = q_sorted[:, 0]
        pred = q_sorted[:, 1]
        upper = q_sorted[:, 2]
        native_cov = float(prediction_interval_coverage(lower, upper, y_test).item())
        native_width = float(torch.mean(upper - lower).item())
        tail_cov = _tail_coverage(lower, upper, y_test, quantile=0.9)
    else:
        pred = raw.squeeze(-1)

    mae = mean_absolute_error(pred, y_test)
    tail_mae_90 = tail_mae(pred, y_test, quantile=0.9, tail="upper")
    tail_rmse_90 = tail_rmse(pred, y_test, quantile=0.9, tail="upper")
    return {
        "MAE": float(mae),
        "TailMAE90": float(tail_mae_90),
        "TailRMSE90": float(tail_rmse_90),
        "NativeCov90": native_cov,
        "NativeWidth90": native_width,
        "TailCov90": tail_cov,
    }


def run_comparison(cfg: PropensityTailConfig) -> tuple[list[dict[str, object]], list[str]]:
    data = _simulate(cfg)
    obs_rate = float((data["obs_pool"] == 1).float().mean().item())

    methods = [
        (
            "MSE",
            lambda: _MLP(cfg.n_features, cfg.hidden, out_dim=1),
            _train_mse,
            "naive observed-only baseline; can underperform on upper tail under severe selection bias",
        ),
        (
            "DensityWeighted",
            lambda: _MLP(cfg.n_features, cfg.hidden, out_dim=1),
            _train_density,
            "target-density reweighting; can overfocus sparse tails when density fit is unstable",
        ),
        (
            "PropensityWeighted",
            lambda: _MLP(cfg.n_features, cfg.hidden, out_dim=1),
            _train_propensity,
            "IPW using p(observed|x); can be high-variance when propensities are near zero",
        ),
        (
            "GaussianNLL",
            lambda: _MLP(cfg.n_features, cfg.hidden, out_dim=2),
            _train_gaussian,
            "distributional baseline with native intervals; can under-cover under multimodal tails",
        ),
        (
            "Quantile90",
            lambda: _MLP(cfg.n_features, cfg.hidden, out_dim=3),
            _train_quantile,
            "quantile-family baseline with native 90% interval; can widen intervals in low-signal regions",
        ),
    ]

    rows: list[dict[str, object]] = []
    for name, model_factory, train_fn, notes in methods:
        model = model_factory()
        _, train_s = timed_call(train_fn, model, data, cfg)
        metrics, eval_s = timed_call(
            _evaluate,
            model,
            data["x_test"],
            data["y_test"],
            method=name,
            alpha=cfg.alpha,
        )
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
        "Gaussian/quantile methods add interval quality signals beyond reweighting-only comparisons.",
    ]
    return rows, notes


def main(cfg: PropensityTailConfig | None = None, summary_json_path: str | None = None) -> None:
    cfg = cfg or PropensityTailConfig()
    rows, notes = run_comparison(cfg)

    print_fairness_notes(
        title="Propensity + Tail Regression Comparison",
        seed_policy="fixed seed and shared selection process",
        train_budget="same MLP capacity and epochs across methods",
        metric_policy="MAE + upper-tail MAE/RMSE + native interval coverage/width + runtime",
    )
    print_comparison_summary(
        "Propensity/tail summary",
        rows,
        metric_order=[
            "MAE",
            "TailMAE90",
            "TailRMSE90",
            "NativeCov90",
            "NativeWidth90",
            "TailCov90",
            "ObservedRate",
            "train_s",
            "eval_s",
        ],
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
