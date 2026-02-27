"""
Compare EIV loss variants on a real tabular dataset with synthetic measurement error.

This example uses the sklearn Diabetes regression dataset as the clean underlying
data and injects synthetic feature/label measurement error to compare EIV methods
under shared budgets with common metrics and runtime tracking.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from comparison_utils import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from sklearn.datasets import load_diabetes

from torchregress.losses import (
    EnsembleEIVLoss,
    FunctionalEIVLoss,
    OrthogonalDistanceRegressionLoss,
    StructuralEIVLoss,
)


@dataclass(frozen=True)
class EIVRealDataConfig:
    seed: int = 4321
    n_train: int = 256
    n_test: int = 120
    epochs: int = 15
    lr: float = 0.02
    hidden: int = 32
    sigma_x: float = 0.08
    sigma_y: float = 0.08


class TabularRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_dataset(cfg: EIVRealDataConfig) -> dict[str, torch.Tensor]:
    x_np, y_np = load_diabetes(return_X_y=True)
    x_all = torch.tensor(x_np, dtype=torch.float32)
    y_all = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    need = cfg.n_train + cfg.n_test
    if need > x_all.shape[0]:
        raise ValueError(f"Requested {need} samples but diabetes dataset has {x_all.shape[0]}.")

    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(x_all.shape[0], generator=g)[:need]
    x_all = x_all[perm]
    y_all = y_all[perm]

    x_train_true = x_all[: cfg.n_train]
    x_test_true = x_all[cfg.n_train :]
    y_train_true_raw = y_all[: cfg.n_train]
    y_test_true_raw = y_all[cfg.n_train :]

    # Standardize targets on clean train labels for optimization stability.
    y_mean = y_train_true_raw.mean()
    y_std = y_train_true_raw.std(unbiased=False).clamp_min(1e-6)
    y_train_true = (y_train_true_raw - y_mean) / y_std
    y_test_true = (y_test_true_raw - y_mean) / y_std

    x_train_noise = torch.randn(x_train_true.shape, generator=g, device=x_train_true.device)
    x_test_noise = torch.randn(x_test_true.shape, generator=g, device=x_test_true.device)
    y_train_noise = torch.randn(y_train_true.shape, generator=g, device=y_train_true.device)
    y_test_noise = torch.randn(y_test_true.shape, generator=g, device=y_test_true.device)

    x_train_obs = x_train_true + cfg.sigma_x * x_train_noise
    x_test_obs = x_test_true + 1.5 * cfg.sigma_x * x_test_noise
    y_train_obs = y_train_true + cfg.sigma_y * y_train_noise
    y_test_obs = y_test_true + cfg.sigma_y * y_test_noise

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
    obs_mse = torch.mean((pred_obs - y_test_obs) ** 2).item()
    stress_mse = torch.mean((pred_obs - y_test_true) ** 2).item()
    return {
        "clean_mse": float(clean_mse),
        "obs_mse": float(obs_mse),
        "obs_input_clean_target_mse": float(stress_mse),
    }


def run_comparison(cfg: EIVRealDataConfig) -> list[dict[str, float | str]]:
    data = make_dataset(cfg)
    d_in = int(data["x_train_obs"].shape[1])
    rows: list[dict[str, float | str]] = []

    sigma_x_diag = torch.full((d_in,), float(cfg.sigma_x))
    sigma_y_diag = torch.tensor([cfg.sigma_y], dtype=torch.float32)
    sigma_xy_zero = torch.zeros(1, d_in, dtype=torch.float32)

    method_specs = [
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
                        learning_rate=0.02,
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
        set_comparison_seed(cfg.seed + idx)
        model = TabularRegressor(d_in, cfg.hidden)
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
            _, train_s = timed_call(
                trainer,
                model,
                params["loss"],
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


def main(cfg: EIVRealDataConfig | None = None, summary_json_path: str | None = None) -> None:
    cfg = cfg or EIVRealDataConfig()
    rows = run_comparison(cfg)

    print("EIV Method Comparison (Real Data + Synthetic Measurement Error)")
    print("=" * 64)
    print_fairness_notes(
        title="EIV Method Comparison (real-data)",
        seed_policy="fixed seed; shared Diabetes split + shared synthetic measurement-error injection",
        train_budget=(
            f"same tabular MLP init/width; {cfg.epochs} epochs each; Adam lr={cfg.lr}; "
            "ODR uses max_iterations=2 per forward"
        ),
        metric_policy=(
            "clean-target test MSE, observed-data test MSE, observed-input/clean-target stress MSE, "
            "train/eval runtime"
        ),
    )
    print_comparison_summary(
        "EIV Comparison Summary (Real Data)",
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
        "- Real-data example uses synthetic measurement-error injection on Diabetes features/targets."
    )
    print(
        "- This improves external validity vs pure synthetic data but is not a domain benchmark suite."
    )
    print(
        "- ODR runtime can dominate on larger problems; compare quality/runtime tradeoffs on your data."
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/eiv_method_realdata_comparison.py",
            task="Noisy features / EIV (real-data)",
            config=cfg,
            rows=rows,
            notes=[
                "Real-data EIV comparison on sklearn Diabetes with synthetic measurement-error injection",
                "Includes analytic/MC/structural/ODR/ensemble EIV variants",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    main()
