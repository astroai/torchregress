"""
Compare EIV loss variants on a shared measurement-error regression task.

This example is designed as comparative evidence for EIV methods:
- shared data and initialization policy per dataset
- fixed epochs and optimizer per method
- common metrics (clean/observed test MSE, train runtime)
- explicit caveats about scale and ODR compute cost

Datasets (each keeps its own fixed seed, size, and model width):

- ``--dataset synthetic`` (default): 2-feature synthetic regression with injected
  feature/label measurement error.
- ``--dataset diabetes``: the ``sklearn`` Diabetes regression dataset as the clean
  underlying data, with injected synthetic feature/label measurement error.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes

from torchregress.comparison import (
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.losses import (
    EnsembleEIVLoss,
    FunctionalEIVLoss,
    OrthogonalDistanceRegressionLoss,
    StructuralEIVLoss,
)

EXAMPLE_PATH = "examples/eiv_method_comparison.py"


@dataclass(frozen=True)
class EIVConfig:
    """Synthetic configuration (``--dataset synthetic``)."""

    seed: int = 321
    n_train: int = 96
    n_test: int = 96
    epochs: int = 20
    lr: float = 0.03
    hidden: int = 16
    sigma_x: float = 0.20
    sigma_y: float = 0.10


@dataclass(frozen=True)
class EIVRealDataConfig:
    """Diabetes configuration (``--dataset diabetes``)."""

    seed: int = 4321
    n_train: int = 256
    n_test: int = 120
    epochs: int = 15
    lr: float = 0.02
    hidden: int = 32
    sigma_x: float = 0.08
    sigma_y: float = 0.08


AnyConfig = EIVConfig | EIVRealDataConfig


class Regressor(nn.Module):
    """Tanh MLP with ``n_hidden_layers`` hidden layers of width ``hidden``."""

    def __init__(self, input_dim: int, hidden: int, n_hidden_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Datasets: both return clean/observed train/test tensors with the same keys.
# ---------------------------------------------------------------------------


def _true_fn(x: torch.Tensor) -> torch.Tensor:
    return 1.2 * x[:, :1] - 0.8 * x[:, 1:2] + 0.15 * x[:, :1] * x[:, 1:2]


def _make_synthetic(cfg: EIVConfig) -> dict[str, torch.Tensor]:
    torch.manual_seed(cfg.seed)
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


def _make_diabetes(cfg: EIVRealDataConfig) -> dict[str, torch.Tensor]:
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

    x_train_noise = torch.randn(x_train_true.shape, generator=g)
    x_test_noise = torch.randn(x_test_true.shape, generator=g)
    y_train_noise = torch.randn(y_train_true.shape, generator=g)
    y_test_noise = torch.randn(y_test_true.shape, generator=g)

    return {
        "x_train_true": x_train_true,
        "x_test_true": x_test_true,
        "y_train_true": y_train_true,
        "y_test_true": y_test_true,
        "x_train_obs": x_train_true + cfg.sigma_x * x_train_noise,
        "x_test_obs": x_test_true + 1.5 * cfg.sigma_x * x_test_noise,
        "y_train_obs": y_train_true + cfg.sigma_y * y_train_noise,
        "y_test_obs": y_test_true + cfg.sigma_y * y_test_noise,
    }


@dataclass(frozen=True)
class DatasetSpec:
    config_cls: type[AnyConfig]
    make_data: Callable[..., dict[str, torch.Tensor]]
    n_hidden_layers: int
    odr_learning_rate: float
    header: str
    title: str
    seed_policy: str
    model_desc: str
    odr_desc: str
    summary_title: str
    caveats: tuple[str, ...]
    task: str
    notes: tuple[str, ...]


DATASETS: dict[str, DatasetSpec] = {
    "synthetic": DatasetSpec(
        config_cls=EIVConfig,
        make_data=_make_synthetic,
        n_hidden_layers=1,
        odr_learning_rate=0.03,
        header="EIV Method Comparison (Synthetic Measurement Error)",
        title="EIV Method Comparison",
        seed_policy="fixed torch seed; shared synthetic x/y measurement-error splits",
        model_desc="small MLP",
        odr_desc="ODR uses max_iterations=2 per forward for tractable demo runtime",
        summary_title="EIV Comparison Summary",
        caveats=(
            "Synthetic low-dimensional problem; use as a comparison template, "
            "not a production benchmark.",
            "ODR is compute-heavier per step; compare quality/runtime tradeoffs on your data.",
        ),
        task="Noisy features / EIV",
        notes=(
            "Synthetic measurement-error benchmark",
            "Includes analytic/MC/structural/ODR/ensemble EIV variants",
        ),
    ),
    "diabetes": DatasetSpec(
        config_cls=EIVRealDataConfig,
        make_data=_make_diabetes,
        n_hidden_layers=2,
        odr_learning_rate=0.02,
        header="EIV Method Comparison (Real Data + Synthetic Measurement Error)",
        title="EIV Method Comparison (real-data)",
        seed_policy="fixed seed; shared Diabetes split + shared synthetic measurement-error injection",
        model_desc="tabular MLP",
        odr_desc="ODR uses max_iterations=2 per forward",
        summary_title="EIV Comparison Summary (Real Data)",
        caveats=(
            "Real-data example uses synthetic measurement-error injection on Diabetes "
            "features/targets.",
            "This improves external validity vs pure synthetic data but is not a domain "
            "benchmark suite.",
            "ODR runtime can dominate on larger problems; compare quality/runtime tradeoffs "
            "on your data.",
        ),
        task="Noisy features / EIV (real-data)",
        notes=(
            "Real-data EIV comparison on sklearn Diabetes with synthetic measurement-error "
            "injection",
            "Includes analytic/MC/structural/ODR/ensemble EIV variants",
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


def _eiv_loss_builders(
    d_in: int, cfg: AnyConfig, odr_learning_rate: float
) -> list[tuple[str, Callable[[nn.Module], nn.Module | None]]]:
    """Method name -> loss factory bound to the model (``None`` = plain MSE baseline)."""
    sigma_x = torch.full((d_in,), float(cfg.sigma_x))
    sigma_y = torch.tensor([cfg.sigma_y], dtype=torch.float32)
    sigma_xy_zero = torch.zeros(1, d_in, dtype=torch.float32)
    return [
        ("Baseline MSE", lambda model: None),
        (
            "FunctionalEIV (analytic)",
            lambda model: FunctionalEIVLoss(model, sigma_x=sigma_x, sigma_y=sigma_y),
        ),
        (
            "FunctionalEIV (MC)",
            lambda model: FunctionalEIVLoss(
                model, sigma_x=sigma_x, sigma_y=sigma_y, monte_carlo=True, n_samples=6
            ),
        ),
        (
            "StructuralEIV",
            lambda model: StructuralEIVLoss(
                model, sigma_x=sigma_x, sigma_y=sigma_y, sigma_xy=sigma_xy_zero
            ),
        ),
        (
            "ODR",
            lambda model: OrthogonalDistanceRegressionLoss(
                model,
                sigma_x=sigma_x,
                sigma_y=sigma_y,
                max_iterations=2,
                learning_rate=odr_learning_rate,
            ),
        ),
        (
            "EnsembleEIV",
            lambda model: EnsembleEIVLoss(model, sigma_x=sigma_x, n_samples=6),
        ),
    ]


def run_comparison(cfg: AnyConfig) -> list[dict[str, float | str]]:
    spec = _dataset_for_config(cfg)
    data = spec.make_data(cfg)
    d_in = int(data["x_train_obs"].shape[1])
    rows: list[dict[str, float | str]] = []

    builders = _eiv_loss_builders(d_in, cfg, spec.odr_learning_rate)
    for idx, (name, build_loss) in enumerate(builders):
        set_comparison_seed(cfg.seed + idx)
        model = Regressor(d_in, cfg.hidden, spec.n_hidden_layers)
        loss_fn = build_loss(model)
        train_args = (data["x_train_obs"], data["y_train_obs"])
        if loss_fn is None:
            _, train_s = timed_call(
                train_baseline_mse, model, *train_args, epochs=cfg.epochs, lr=cfg.lr
            )
        else:
            _, train_s = timed_call(
                train_with_loss, model, loss_fn, *train_args, epochs=cfg.epochs, lr=cfg.lr
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

    print(spec.header)
    print("=" * len(spec.header))
    print_fairness_notes(
        title=spec.title,
        seed_policy=spec.seed_policy,
        train_budget=(
            f"same {spec.model_desc} init/width; {cfg.epochs} epochs each; Adam lr={cfg.lr}; "
            f"{spec.odr_desc}"
        ),
        metric_policy=(
            "clean-target test MSE, observed-data test MSE, observed-input/clean-target stress MSE, "
            "train/eval runtime"
        ),
    )
    print_comparison_summary(
        spec.summary_title,
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
    for caveat in spec.caveats:
        print(f"- {caveat}")

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
    parser = argparse.ArgumentParser(description="Run EIV method comparison")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="synthetic")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(dataset=args.dataset, summary_json_path=args.summary_json_path)
