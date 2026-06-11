"""
Compare EIV loss variants on a shared synthetic measurement-error regression task.

This example is designed as decision-grade comparative evidence for EIV methods:
- shared synthetic data and initialization policy
- fixed epochs and optimizer per method
- common metrics (clean/observed test MSE, train runtime)
- explicit caveats about toy scale and ODR compute cost
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.losses import (
    EnsembleEIVLoss,
    FunctionalEIVLoss,
    OrthogonalDistanceRegressionLoss,
    StructuralEIVLoss,
)


@dataclass(frozen=True)
class EIVConfig:
    seed: int = 321
    n_train: int = 96
    n_test: int = 96
    epochs: int = 20
    lr: float = 0.03
    hidden: int = 16
    sigma_x: float = 0.20
    sigma_y: float = 0.10


class SmallRegressor(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def _true_fn(x: torch.Tensor) -> torch.Tensor:
    return 1.2 * x[:, :1] - 0.8 * x[:, 1:2] + 0.15 * x[:, :1] * x[:, 1:2]


def make_dataset(cfg: EIVConfig) -> dict[str, torch.Tensor]:
    set_seed(cfg.seed)
    x_train_true = torch.randn(cfg.n_train, 2)
    x_test_true = torch.randn(cfg.n_test, 2)
    y_train_true = _true_fn(x_train_true)
    y_test_true = _true_fn(x_test_true)

    x_train_obs = x_train_true + cfg.sigma_x * torch.randn_like(x_train_true)
    x_test_obs = x_test_true + cfg.sigma_x * 1.5 * torch.randn_like(x_test_true)
    y_train_obs = y_train_true + cfg.sigma_y * torch.randn_like(y_train_true)
    y_test_obs = y_test_true + cfg.sigma_y * torch.randn_like(y_test_true)

    return {
        "x_train_true": x_train_true,
        "x_test_true": x_test_true,
        "y_train_true": y_train_true,
        "y_test_true": y_test_true,
        "x_train_obs": x_train_obs,
        "x_test_obs": x_test_obs,
        "y_train_obs": y_train_obs,
        "y_test_obs": y_test_obs,
    }


def train_with_loss(
    model: nn.Module,
    loss_fn: nn.Module,
    x_obs: torch.Tensor,
    y_obs: torch.Tensor,
    *,
    epochs: int,
    lr: float,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(x_obs, y_obs)
        loss.backward()
        opt.step()
    model.eval()
    return model


def train_baseline_mse(
    model: nn.Module,
    x_obs: torch.Tensor,
    y_obs: torch.Tensor,
    *,
    epochs: int,
    lr: float,
) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_obs)
        loss = mse(pred, y_obs)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _evaluate_model(
    model: nn.Module,
    x_test_true: torch.Tensor,
    x_test_obs: torch.Tensor,
    y_test_true: torch.Tensor,
    y_test_obs: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        pred_clean = model(x_test_true)
        pred_obs = model(x_test_obs)
    clean_mse = torch.mean((pred_clean - y_test_true) ** 2).item()
    noisy_mse = torch.mean((pred_obs - y_test_obs) ** 2).item()
    shift_mse = torch.mean((pred_obs - y_test_true) ** 2).item()
    return {
        "clean_mse": float(clean_mse),
        "obs_mse": float(noisy_mse),
        "obs_input_clean_target_mse": float(shift_mse),
    }


def run_comparison(cfg: EIVConfig) -> list[dict[str, float | str]]:
    data = make_dataset(cfg)
    rows: list[dict[str, float | str]] = []

    sigma_x_diag = torch.tensor([cfg.sigma_x, cfg.sigma_x])
    sigma_y_diag = torch.tensor([cfg.sigma_y])
    sigma_xy_zero = torch.zeros(1, 2)

    method_specs: list[tuple[str, callable]] = [
        (
            "Baseline MSE",
            lambda model: ("baseline", train_baseline_mse, {"loss": None}),
        ),
        (
            "FunctionalEIV (analytic)",
            lambda model: (
                "eiv",
                train_with_loss,
                {"loss": FunctionalEIVLoss(model, sigma_x=sigma_x_diag, sigma_y=sigma_y_diag)},
            ),
        ),
        (
            "FunctionalEIV (MC)",
            lambda model: (
                "eiv",
                train_with_loss,
                {
                    "loss": FunctionalEIVLoss(
                        model,
                        sigma_x=sigma_x_diag,
                        sigma_y=sigma_y_diag,
                        monte_carlo=True,
                        n_samples=6,
                    )
                },
            ),
        ),
        (
            "StructuralEIV",
            lambda model: (
                "eiv",
                train_with_loss,
                {
                    "loss": StructuralEIVLoss(
                        model,
                        sigma_x=sigma_x_diag,
                        sigma_y=sigma_y_diag,
                        sigma_xy=sigma_xy_zero,
                    )
                },
            ),
        ),
        (
            "ODR",
            lambda model: (
                "eiv",
                train_with_loss,
                {
                    "loss": OrthogonalDistanceRegressionLoss(
                        model,
                        sigma_x=sigma_x_diag,
                        sigma_y=sigma_y_diag,
                        max_iterations=2,
                        learning_rate=0.03,
                    )
                },
            ),
        ),
        (
            "EnsembleEIV",
            lambda model: (
                "eiv",
                train_with_loss,
                {"loss": EnsembleEIVLoss(model, sigma_x=sigma_x_diag, n_samples=6)},
            ),
        ),
    ]

    for idx, (name, builder) in enumerate(method_specs):
        set_seed(cfg.seed + idx)
        model = SmallRegressor(cfg.hidden)
        mode, trainer, params = builder(model)

        if mode == "baseline":
            _, train_s = timed_call(
                trainer,
                model,
                data["x_train_obs"],
                data["y_train_obs"],
                epochs=cfg.epochs,
                lr=cfg.lr,
            )
        else:
            loss_fn = params["loss"]
            _, train_s = timed_call(
                trainer,
                model,
                loss_fn,
                data["x_train_obs"],
                data["y_train_obs"],
                epochs=cfg.epochs,
                lr=cfg.lr,
            )

        metrics, eval_s = timed_call(
            _evaluate_model,
            model,
            data["x_test_true"],
            data["x_test_obs"],
            data["y_test_true"],
            data["y_test_obs"],
        )
        rows.append(
            {
                "Method": name,
                **metrics,
                "train_s": train_s,
                "eval_s": eval_s,
                "Notes": "ODR inner optimization" if name == "ODR" else "",
            }
        )
    return rows


def main(cfg: EIVConfig | None = None, summary_json_path: str | None = None) -> None:
    cfg = cfg or EIVConfig()
    rows = run_comparison(cfg)

    print("EIV Method Comparison (Synthetic Measurement Error)")
    print("=" * 58)
    print_fairness_notes(
        title="EIV Method Comparison",
        seed_policy="fixed torch seed; shared synthetic x/y measurement-error splits",
        train_budget=(
            f"same small MLP init/width; {cfg.epochs} epochs each; Adam lr={cfg.lr}; "
            "ODR uses max_iterations=2 per forward for tractable demo runtime"
        ),
        metric_policy=(
            "clean-target test MSE, observed-data test MSE, observed-input/clean-target stress MSE, "
            "train/eval runtime"
        ),
    )
    print_comparison_summary(
        "EIV Comparison Summary",
        rows,
        metric_order=[
            "clean_mse",
            "obs_mse",
            "obs_input_clean_target_mse",
            "train_s",
            "eval_s",
        ],
    )
    print("\nCaveats:")
    print(
        "- Synthetic low-dimensional problem; use as a comparison template, not a production benchmark."
    )
    print("- ODR is compute-heavier per step; compare quality/runtime tradeoffs on your data.")

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/eiv_method_comparison.py",
            task="Noisy features / EIV",
            config=cfg,
            rows=rows,
            notes=[
                "Synthetic measurement-error benchmark",
                "Includes analytic/MC/structural/ODR/ensemble EIV variants",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    main()
