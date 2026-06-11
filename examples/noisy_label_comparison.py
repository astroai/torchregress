"""
Dedicated noisy-label comparison with calibration metrics.

This example compares robust and probabilistic regression approaches under
label corruption using a shared training budget and common metrics.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.comparison import (
    compute_point_metrics,
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.losses import (
    CauchyLoss,
    GaussianNLLLoss,
    MultiQuantileLoss,
    TukeyBiweightLoss,
    WeightedHuberLoss,
    WeightedMSELoss,
)
from torchregress.metrics import interval_score, prediction_interval_coverage


@dataclass
class NoisyLabelComparisonConfig:
    n_train: int = 512
    n_cal: int = 128
    n_test: int = 256
    noise_ratio: float = 0.2
    noise_scale: float = 2.5
    base_noise: float = 0.15
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    seed: int = 7
    hidden: int = 64
    alpha: float = 0.1


def _make_clean_targets(x: torch.Tensor) -> torch.Tensor:
    return (torch.sin(1.6 * x) + 0.4 * x + 0.15 * x.square()).float()


def _corrupt_labels(
    y_clean: torch.Tensor,
    *,
    noise_ratio: float,
    noise_scale: float,
    base_noise: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    y_obs = y_clean + base_noise * torch.randn(y_clean.shape, generator=g, device=y_clean.device)
    n = y_clean.shape[0]
    n_bad = int(noise_ratio * n)
    bad_idx = torch.randperm(n, generator=g)[:n_bad]
    y_obs[bad_idx] = y_obs[bad_idx] + noise_scale * torch.randn(
        n_bad, 1, generator=g, device=y_clean.device
    )
    noisy_mask = torch.zeros(n, dtype=torch.bool)
    noisy_mask[bad_idx] = True
    return y_obs, noisy_mask


def generate_noisy_label_splits(
    cfg: NoisyLabelComparisonConfig,
) -> dict[str, torch.Tensor]:
    total = cfg.n_train + cfg.n_cal + cfg.n_test
    x = torch.linspace(-3.0, 3.0, total).unsqueeze(1)
    y_clean = _make_clean_targets(x)
    y_obs, noisy_mask = _corrupt_labels(
        y_clean,
        noise_ratio=cfg.noise_ratio,
        noise_scale=cfg.noise_scale,
        base_noise=cfg.base_noise,
        seed=cfg.seed,
    )

    train_end = cfg.n_train
    cal_end = train_end + cfg.n_cal
    return {
        "x_train": x[:train_end],
        "y_train_obs": y_obs[:train_end],
        "y_train_clean": y_clean[:train_end],
        "train_noisy_mask": noisy_mask[:train_end],
        "x_cal": x[train_end:cal_end],
        "y_cal_obs": y_obs[train_end:cal_end],
        "y_cal_clean": y_clean[train_end:cal_end],
        "x_test": x[cal_end:],
        "y_test_obs": y_obs[cal_end:],
        "y_test_clean": y_clean[cal_end:],
        "test_noisy_mask": noisy_mask[cal_end:],
    }


class Regressor(nn.Module):
    def __init__(self, out_dim: int = 1, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train(model: nn.Module, loss_fn, loader: DataLoader, *, epochs: int, lr: float) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()


def _conformal_interval(
    pred_cal: torch.Tensor,
    y_cal_obs: torch.Tensor,
    pred_test: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residuals = (y_cal_obs - pred_cal).abs().view(-1)
    if residuals.numel() == 0:
        raise ValueError("Calibration residuals are empty")
    q = torch.quantile(residuals, 1.0 - alpha)
    return pred_test - q, pred_test + q


def _gaussian_interval(raw: torch.Tensor, *, alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    mean, log_var = torch.chunk(raw, 2, dim=-1)
    std = torch.exp(0.5 * log_var).clamp_min(1e-4)
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor([1 - alpha / 2])).to(raw.device)
    return mean - z * std, mean + z * std


def _quantile_outputs(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = raw.reshape(raw.shape[0], 3, 1)
    q_sorted, _ = torch.sort(q, dim=1)
    lower = q_sorted[:, 0]
    median = q_sorted[:, 1]
    upper = q_sorted[:, 2]
    return lower, median, upper


def _interval_metrics(
    lower: torch.Tensor,
    upper: torch.Tensor,
    y_true: torch.Tensor,
    *,
    alpha: float,
) -> tuple[float, float, float]:
    cov = prediction_interval_coverage(lower, upper, y_true, alpha=alpha)
    width = float(torch.mean(upper - lower).item())
    score = interval_score(lower, upper, y_true, alpha=alpha)
    score_f = float(score.item()) if isinstance(score, torch.Tensor) else float(score)
    return float(cov), width, score_f


def _evaluate_method(
    name: str,
    model: nn.Module,
    splits: dict[str, torch.Tensor],
    *,
    alpha: float,
) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        raw_cal = model(splits["x_cal"])
        raw_test = model(splits["x_test"])

        native_cov = None
        native_width = None

        if name == "GaussianNLL":
            pred_cal = torch.chunk(raw_cal, 2, dim=-1)[0]
            pred_test = torch.chunk(raw_test, 2, dim=-1)[0]
            nll_loss = GaussianNLLLoss()
            loss_value = float(nll_loss(raw_test, splits["y_test_obs"]).item())
            native_lower, native_upper = _gaussian_interval(raw_test, alpha=alpha)
            native_cov, native_width, _ = _interval_metrics(
                native_lower, native_upper, splits["y_test_clean"], alpha=alpha
            )
        elif name == "Quantile90":
            q_cal_l, q_cal_m, _ = _quantile_outputs(raw_cal)
            q_test_l, pred_test, q_test_u = _quantile_outputs(raw_test)
            pred_cal = q_cal_m
            loss_value = float("nan")
            native_cov, native_width, _ = _interval_metrics(
                q_test_l, q_test_u, splits["y_test_clean"], alpha=alpha
            )
        else:
            pred_cal = raw_cal
            pred_test = raw_test
            loss_value = float("nan")

        point_clean = compute_point_metrics(pred_test, splits["y_test_clean"])
        point_obs = compute_point_metrics(pred_test, splits["y_test_obs"])
        conf_lower, conf_upper = _conformal_interval(
            pred_cal, splits["y_cal_obs"], pred_test, alpha=alpha
        )
        cov90, width90, int_score = _interval_metrics(
            conf_lower, conf_upper, splits["y_test_clean"], alpha=alpha
        )

    return {
        "Method": name,
        "CleanMSE": point_clean["MSE"],
        "CleanMAE": point_clean["MAE"],
        "ObsMSE": point_obs["MSE"],
        "ConformalCov90": cov90,
        "ConformalWidth90": width90,
        "ConformalIS90": int_score,
        "NativeCov90": native_cov,
        "NativeWidth90": native_width,
        "EvalLossOnObs": loss_value,
    }


def main(
    cfg: Optional[NoisyLabelComparisonConfig] = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or NoisyLabelComparisonConfig()
    set_comparison_seed(cfg.seed)
    splits = generate_noisy_label_splits(cfg)
    train_loader = DataLoader(
        TensorDataset(splits["x_train"], splits["y_train_obs"]),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    methods: list[tuple[str, nn.Module, object]] = [
        ("MSE", Regressor(hidden=cfg.hidden), WeightedMSELoss()),
        ("Huber", Regressor(hidden=cfg.hidden), WeightedHuberLoss(delta=1.0)),
        ("Cauchy", Regressor(hidden=cfg.hidden), CauchyLoss(c=1.0)),
        ("Tukey", Regressor(hidden=cfg.hidden), TukeyBiweightLoss(c=4.685)),
        ("GaussianNLL", Regressor(out_dim=2, hidden=cfg.hidden), GaussianNLLLoss()),
        (
            "Quantile90",
            Regressor(out_dim=3, hidden=cfg.hidden),
            MultiQuantileLoss(quantiles=[0.05, 0.5, 0.95], joint_prediction=True),
        ),
    ]

    summary_rows: list[dict[str, object]] = []
    for idx, (name, model, loss_fn) in enumerate(methods):
        set_comparison_seed(cfg.seed + 10 + idx)
        _, train_s = timed_call(_train, model, loss_fn, train_loader, epochs=cfg.epochs, lr=cfg.lr)
        result, eval_s = timed_call(_evaluate_method, name, model, splits, alpha=cfg.alpha)
        result["train_s"] = train_s
        result["eval_s"] = eval_s
        if name in {"GaussianNLL", "Quantile90"}:
            result["Notes"] = "native interval + shared split-conformal metrics"
        else:
            result["Notes"] = "shared split-conformal intervals from point predictions"
        summary_rows.append(result)

    print_fairness_notes(
        title="Noisy Label Comparison",
        seed_policy="fixed seed; shared corrupted train/cal/test splits",
        train_budget=f"{cfg.epochs} epochs, batch_size={cfg.batch_size}, lr={cfg.lr}",
        metric_policy=(
            "Clean/observed point metrics + shared split-conformal coverage/width/interval score; "
            "native interval metrics for Gaussian/quantile methods"
        ),
    )
    print_comparison_summary(
        "Noisy Label Robustness + Calibration Summary",
        summary_rows,
        metric_order=[
            "CleanMSE",
            "CleanMAE",
            "ObsMSE",
            "ConformalCov90",
            "ConformalWidth90",
            "ConformalIS90",
            "NativeCov90",
            "NativeWidth90",
            "train_s",
            "eval_s",
        ],
    )
    noisy_frac = float(splits["train_noisy_mask"].float().mean().item())
    print("\nDataset notes:")
    print(f"- Train label corruption fraction: {noisy_frac:.2%}")
    print("- Conformal intervals use noisy calibration labels (stressful but realistic).")
    print("- Coverage is evaluated against clean test labels to show robustness to corruption.")

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/noisy_label_comparison.py",
            task="Noisy labels / corruption",
            config=cfg,
            rows=summary_rows,
            notes=[
                "Shared split-conformal calibration metrics for all methods",
                "Native interval metrics included for Gaussian and quantile heads",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    main()
