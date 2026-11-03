"""
Shared-budget comparison for multimodal / multi-target regression methods.

Compares:
- Diagonal Gaussian NLL (unimodal baseline)
- MDN (multimodal baseline)
- Normalizing flow (optional, if zuko is installed)

Datasets (each keeps its own fixed seed, size, and training budget):

- ``--dataset synthetic`` (default): 1D input with an input-dependent bimodal
  region and a 2D target.
- ``--dataset diabetes``: ``sklearn`` Diabetes features as real covariates and a
  synthetic conditional multimodal 2D target built from them.

The goal is decision support, not SOTA benchmarking.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes
from torch.utils.data import DataLoader, TensorDataset

from torchregress.calibration import marginal_calibration_error
from torchregress.comparison import (
    compute_point_metrics,
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torchregress.losses import MixtureDensityLoss, create_gaussian_nll
from torchregress.metrics import energy_score

EXAMPLE_PATH = "examples/multimodal_method_comparison.py"
N_TARGETS = 2


@dataclass
class MultimodalComparisonConfig:
    """Synthetic configuration (``--dataset synthetic``)."""

    n_train: int = 384
    n_test: int = 256
    batch_size: int = 64
    epochs: int = 40
    lr: float = 1e-3
    seed: int = 42
    hidden: int = 64
    mdn_components: int = 4
    flow_context_dim: int = 16
    flow_transforms: int = 4
    eval_samples: int = 128


@dataclass
class MultimodalRealDataConfig:
    """Diabetes configuration (``--dataset diabetes``)."""

    n_train: int = 256
    n_test: int = 128
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    seed: int = 1234
    hidden: int = 64
    mdn_components: int = 4
    flow_context_dim: int = 16
    flow_transforms: int = 4
    eval_samples: int = 96
    mode_shift_y1: float = 0.55
    mode_shift_y2: float = 0.35
    base_noise: float = 0.08


AnyConfig = MultimodalComparisonConfig | MultimodalRealDataConfig
Split = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Datasets: each returns (x_train, y_train, x_test, y_test) with 2D targets.
# ---------------------------------------------------------------------------


def generate_multimodal_multitarget_data(
    n_samples: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 2D targets with input-dependent multimodal structure."""
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(-3.0, 3.0, n_samples).unsqueeze(1)

    base1 = torch.sin(2.0 * x) * (1.0 + 0.15 * x.square())
    base2 = 0.4 * base1.square() + 0.5 * torch.cos(2.5 * x)

    # Input-dependent bimodal region (middle of the domain).
    multimodal_region = (x.abs() < 1.4).float()
    mode = torch.where(
        torch.rand(n_samples, 1, generator=g) < 0.5,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )
    mode_shift = mode * multimodal_region

    noise_scale = 0.08 + 0.05 * x.abs()
    y1 = base1 + 0.45 * mode_shift + torch.randn(n_samples, 1, generator=g) * noise_scale
    y2 = base2 + 0.25 * mode_shift + torch.randn(n_samples, 1, generator=g) * (noise_scale * 1.1)
    y = torch.cat([y1, y2], dim=1)
    return x.float(), y.float()


def _synthetic_split(cfg: MultimodalComparisonConfig) -> Split:
    x_train, y_train = generate_multimodal_multitarget_data(cfg.n_train, seed=cfg.seed)
    x_test, y_test = generate_multimodal_multitarget_data(cfg.n_test, seed=cfg.seed + 1)
    return x_train, y_train, x_test, y_test


def generate_multimodal_realdata_targets(
    cfg: MultimodalRealDataConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diabetes covariates with a synthetic conditional multimodal 2D target."""
    x_np, y_np = load_diabetes(return_X_y=True)
    x_all = torch.tensor(x_np, dtype=torch.float32)
    y_all = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    total = cfg.n_train + cfg.n_test
    if total > x_all.shape[0]:
        raise ValueError(f"Requested {total} samples but diabetes dataset has {x_all.shape[0]}.")

    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(x_all.shape[0], generator=g)[:total]
    x = x_all[perm]
    y = y_all[perm]

    # Standardize target from selected subset for stable training.
    y = (y - y.mean()) / y.std(unbiased=False).clamp_min(1e-6)

    # Construct multimodal conditional 2D target using real covariates.
    x0 = x[:, 0:1]
    x1 = x[:, 1:2]
    x2 = x[:, 2:3]
    x3 = x[:, 3:4]
    region = ((x2.abs() < x2.abs().median()) | (x3 > x3.median())).float()
    mode = torch.where(
        torch.rand(total, 1, generator=g) < 0.5,
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(-1.0, dtype=torch.float32),
    )
    mode_shift = region * mode

    noise1 = torch.randn(total, 1, generator=g) * (cfg.base_noise + 0.03 * x0.abs())
    noise2 = torch.randn(total, 1, generator=g) * (cfg.base_noise + 0.03 * x1.abs())

    base1 = 0.9 * y + 0.35 * x0 - 0.25 * x1 + 0.10 * x2 * x3
    base2 = 0.55 * y - 0.30 * x0 + 0.45 * x1 + 0.15 * x0.square()

    y1 = base1 + cfg.mode_shift_y1 * mode_shift + noise1
    y2 = base2 + cfg.mode_shift_y2 * mode_shift + noise2
    y_out = torch.cat([y1, y2], dim=1)
    return x.float(), y_out.float()


def _diabetes_split(cfg: MultimodalRealDataConfig) -> Split:
    x_all, y_all = generate_multimodal_realdata_targets(cfg)
    n = cfg.n_train
    return x_all[:n], y_all[:n], x_all[n:], y_all[n:]


@dataclass(frozen=True)
class DatasetSpec:
    config_cls: type[AnyConfig]
    make_split: Callable[..., Split]
    title: str
    seed_policy: str
    summary_title: str
    caveats: tuple[str, ...]
    task: str
    notes: tuple[str, ...]


DATASETS: dict[str, DatasetSpec] = {
    "synthetic": DatasetSpec(
        config_cls=MultimodalComparisonConfig,
        make_split=_synthetic_split,
        title="Multimodal Method Comparison",
        seed_policy="fixed global seed; deterministic synthetic train/test generation",
        summary_title="Multimodal / Multi-Target Comparison Summary",
        caveats=(
            "Synthetic 2D data does not establish real-world ranking.",
            "Gaussian baseline is intentionally unimodal and may average across modes.",
            "Flow row is optional and may be skipped when zuko is not installed.",
        ),
        task="Multimodal / multi-target non-Gaussian",
        notes=(
            "Flow row is optional when zuko is unavailable",
            "Common metrics include NLL, energy score, and marginal calibration error",
        ),
    ),
    "diabetes": DatasetSpec(
        config_cls=MultimodalRealDataConfig,
        make_split=_diabetes_split,
        title="Multimodal Method Comparison (real-data)",
        seed_policy=(
            "fixed seed; shared Diabetes feature split and synthetic multimodal target construction"
        ),
        summary_title="Multimodal / Multi-Target Comparison Summary (Real Data Features)",
        caveats=(
            "Real covariates are used, but multimodal targets are synthetically constructed.",
            "Flow row is optional and may be skipped when zuko is not installed.",
            "This reduces synthetic-only risk but is not a domain benchmark for multimodal "
            "regression.",
        ),
        task="Multimodal / multi-target non-Gaussian (real-data)",
        notes=(
            "Real covariates from sklearn Diabetes; synthetic conditional multimodal target "
            "construction",
            "Flow row is optional when zuko is unavailable",
            "Common metrics include NLL, energy score, and marginal calibration error",
        ),
    ),
}


def _dataset_for_config(cfg: AnyConfig) -> DatasetSpec:
    for spec in DATASETS.values():
        if type(cfg) is spec.config_cls:
            return spec
    raise TypeError(f"Unsupported config type: {type(cfg).__name__}")


# ---------------------------------------------------------------------------
# Shared models, training, and evaluation
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_model(
    model: nn.Module,
    loss_fn,
    train_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()


def _train_flow(
    context_model: nn.Module,
    flow: nn.Module,
    loss_fn,
    train_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(list(context_model.parameters()) + list(flow.parameters()), lr=lr)
    for _ in range(epochs):
        context_model.train()
        flow.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            context = context_model(xb)
            loss = loss_fn(context, yb)
            loss.backward()
            optimizer.step()


def _sample_metrics(
    point_pred: torch.Tensor,
    samples: torch.Tensor,
    y_test: torch.Tensor,
    nll: float,
    *,
    n_samples: int,
) -> dict[str, float]:
    """Shared metric block; ``samples`` is ``[S, B, D]``."""
    point = compute_point_metrics(point_pred, y_test)
    mce = marginal_calibration_error(samples, y_test)["marginal_calibration_error"]
    es = energy_score(samples, y_test, max_pairs=min(32, n_samples))
    return {**point, "NLL": nll, "Energy": float(es), "MCE": float(mce)}


def _gaussian_predict_and_metrics(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    n_samples: int,
    seed: int | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    loss_fn = create_gaussian_nll(covariance_type="diagonal")
    model.eval()
    with torch.no_grad():
        raw = model(x_test)
        mean, log_var = torch.chunk(raw, 2, dim=-1)
        std = torch.exp(0.5 * log_var).clamp_min(1e-4)
        dist = torch.distributions.Normal(mean, std)
        if seed is not None:
            torch.manual_seed(seed)
        samples = dist.sample((n_samples,))  # [S, B, D]
        nll = float(loss_fn(raw, y_test).item())
        metrics = _sample_metrics(mean, samples, y_test, nll, n_samples=n_samples)
    return metrics, {"Notes": "diag Gaussian baseline (unimodal)"}


def _mdn_predict_and_metrics(
    model: nn.Module,
    loss_fn: MixtureDensityLoss,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    n_samples: int,
    seed: int | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    model.eval()
    with torch.no_grad():
        raw = model(x_test)
        if seed is not None:
            torch.manual_seed(seed)
        samples = loss_fn.sample(raw, n_samples=n_samples)  # [S, B, D]
        nll = float(loss_fn(raw, y_test).item())
        metrics = _sample_metrics(samples.mean(dim=0), samples, y_test, nll, n_samples=n_samples)
    return metrics, {"Notes": f"{loss_fn.n_components}-component MDN"}


def _try_make_flow(
    *,
    in_dim: int,
    n_targets: int,
    context_dim: int,
    n_transforms: int,
) -> tuple[Optional[nn.Module], Optional[object], Optional[str]]:
    try:
        from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

        context_model = MLP(in_dim, context_dim, hidden=64)
        flow = create_flow_model(
            n_features=n_targets,
            context_dim=context_dim,
            flow_type="nsf",
            n_transforms=n_transforms,
            hidden_features=[64, 64],
        )
        loss_fn = NormalizingFlowLoss(flow=flow, reduction="mean")
        return context_model, loss_fn, None
    except ImportError as exc:
        return None, None, str(exc)


def _flow_predict_and_metrics(
    context_model: nn.Module,
    loss_fn,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    n_samples: int,
    seed: int | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    context_model.eval()
    with torch.no_grad():
        context = context_model(x_test)
        nll = float(loss_fn(context, y_test).item())
        if seed is not None:
            torch.manual_seed(seed)
        flow_samples = loss_fn.sample(context, n_samples=n_samples)  # [B, S, D]
        samples = flow_samples.transpose(0, 1)  # [S, B, D]
        metrics = _sample_metrics(samples.mean(dim=0), samples, y_test, nll, n_samples=n_samples)
    return metrics, {"Notes": "conditional NSF flow (zuko optional)"}


def run_comparison(cfg: AnyConfig) -> list[dict[str, object]]:
    spec = _dataset_for_config(cfg)
    set_comparison_seed(cfg.seed)

    x_train, y_train, x_test, y_test = spec.make_split(cfg)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    d_in = int(x_train.shape[1])

    summary_rows: list[dict[str, object]] = []

    # Gaussian baseline
    gaussian_model = MLP(d_in, 2 * N_TARGETS, hidden=cfg.hidden)
    gaussian_loss = create_gaussian_nll(covariance_type="diagonal")
    _, train_s = timed_call(
        _train_model, gaussian_model, gaussian_loss, loader, epochs=cfg.epochs, lr=cfg.lr
    )
    (gaussian_metrics, gaussian_meta), eval_s = timed_call(
        _gaussian_predict_and_metrics,
        gaussian_model,
        x_test,
        y_test,
        n_samples=cfg.eval_samples,
        seed=2025 + 0,
    )
    summary_rows.append(
        {
            "Method": "GaussianNLL",
            **gaussian_metrics,
            "train_s": train_s,
            "eval_s": eval_s,
            **gaussian_meta,
        }
    )

    # MDN
    mdn_loss = MixtureDensityLoss(
        n_components=cfg.mdn_components, n_features=N_TARGETS, covariance_type="diagonal"
    )
    mdn_model = MLP(d_in, mdn_loss.expected_output_size, hidden=cfg.hidden)
    _, train_s = timed_call(_train_model, mdn_model, mdn_loss, loader, epochs=cfg.epochs, lr=cfg.lr)
    (mdn_metrics, mdn_meta), eval_s = timed_call(
        _mdn_predict_and_metrics,
        mdn_model,
        mdn_loss,
        x_test,
        y_test,
        n_samples=cfg.eval_samples,
        seed=2025 + 1,
    )
    summary_rows.append(
        {"Method": "MDN", **mdn_metrics, "train_s": train_s, "eval_s": eval_s, **mdn_meta}
    )

    # Optional flow
    flow_context_model, flow_loss, flow_err = _try_make_flow(
        in_dim=d_in,
        n_targets=N_TARGETS,
        context_dim=cfg.flow_context_dim,
        n_transforms=cfg.flow_transforms,
    )
    if flow_context_model is not None and flow_loss is not None:
        flow_module = getattr(flow_loss, "flow")
        _, train_s = timed_call(
            _train_flow,
            flow_context_model,
            flow_module,
            flow_loss,
            loader,
            epochs=cfg.epochs,
            lr=cfg.lr,
        )
        (flow_metrics, flow_meta), eval_s = timed_call(
            _flow_predict_and_metrics,
            flow_context_model,
            flow_loss,
            x_test,
            y_test,
            n_samples=cfg.eval_samples,
            seed=2025 + 2,
        )
        summary_rows.append(
            {
                "Method": "NormalizingFlow",
                **flow_metrics,
                "train_s": train_s,
                "eval_s": eval_s,
                **flow_meta,
            }
        )
    else:
        summary_rows.append(
            {
                "Method": "NormalizingFlow",
                "NLL": None,
                "Energy": None,
                "MCE": None,
                "train_s": None,
                "eval_s": None,
                "Notes": f"skipped (optional dependency unavailable: {flow_err})",
            }
        )
    return summary_rows


def main(
    cfg: AnyConfig | None = None,
    summary_json_path: str | None = None,
    *,
    dataset: str = "synthetic",
) -> None:
    """Run the comparison; ``cfg`` (if given) selects the dataset by its type."""
    cfg = cfg or DATASETS[dataset].config_cls()
    spec = _dataset_for_config(cfg)
    summary_rows = run_comparison(cfg)

    print_fairness_notes(
        title=spec.title,
        seed_policy=spec.seed_policy,
        train_budget=f"{cfg.epochs} epochs, batch_size={cfg.batch_size}, lr={cfg.lr}",
        metric_policy="Shared point metrics + NLL + energy score + marginal calibration error + runtime",
    )
    print_comparison_summary(
        spec.summary_title,
        summary_rows,
        metric_order=["MSE", "MAE", "R2", "NLL", "Energy", "MCE", "train_s", "eval_s"],
    )
    print("\nFailure modes / caveats:")
    for caveat in spec.caveats:
        print(f"- {caveat}")

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example=EXAMPLE_PATH,
            task=spec.task,
            config=cfg,
            rows=summary_rows,
            notes=list(spec.notes),
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multimodal method comparison")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="synthetic")
    parser.add_argument("--summary-json-path", type=str, default=None)
    args = parser.parse_args()
    main(dataset=args.dataset, summary_json_path=args.summary_json_path)
