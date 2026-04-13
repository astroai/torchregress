"""
Shared-budget comparison for multimodal / multi-target regression methods.

Compares:
- Diagonal Gaussian NLL (unimodal baseline)
- MDN (multimodal baseline)
- Normalizing flow (optional, if zuko is installed)

The goal is decision support, not SOTA benchmarking.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from comparison_utils import (
    compute_point_metrics,
    print_comparison_summary,
    print_fairness_notes,
    set_comparison_seed,
    timed_call,
    write_comparison_summary_json,
)
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import MDNLoss, create_gaussian_nll
from torchregress.metrics import energy_score, marginal_calibration_error


@dataclass
class MultimodalComparisonConfig:
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
        point = compute_point_metrics(mean, y_test)
        nll = float(loss_fn(raw, y_test).item())
        mce = marginal_calibration_error(samples, y_test)["marginal_calibration_error"]
        es = energy_score(samples, y_test, max_pairs=min(32, n_samples))
    metrics = {
        **point,
        "NLL": nll,
        "Energy": float(es),
        "MCE": float(mce),
    }
    return metrics, {"Notes": "diag Gaussian baseline (unimodal)"}


def _mdn_predict_and_metrics(
    model: nn.Module,
    loss_fn: MDNLoss,
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
        mean = samples.mean(dim=0)
        point = compute_point_metrics(mean, y_test)
        nll = float(loss_fn(raw, y_test).item())
        mce = marginal_calibration_error(samples, y_test)["marginal_calibration_error"]
        es = energy_score(samples, y_test, max_pairs=min(32, n_samples))
    metrics = {
        **point,
        "NLL": nll,
        "Energy": float(es),
        "MCE": float(mce),
    }
    return metrics, {"Notes": f"{loss_fn.n_components}-component MDN"}


def _try_make_flow(
    *,
    n_targets: int,
    context_dim: int,
    n_transforms: int,
) -> tuple[Optional[nn.Module], Optional[object], Optional[str]]:
    try:
        from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

        context_model = MLP(1, context_dim, hidden=64)
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
        mean = samples.mean(dim=0)
        point = compute_point_metrics(mean, y_test)
        mce = marginal_calibration_error(samples, y_test)["marginal_calibration_error"]
        es = energy_score(samples, y_test, max_pairs=min(32, n_samples))
    metrics = {
        **point,
        "NLL": nll,
        "Energy": float(es),
        "MCE": float(mce),
    }
    return metrics, {"Notes": "conditional NSF flow (zuko optional)"}


def main(
    cfg: Optional[MultimodalComparisonConfig] = None,
    summary_json_path: str | None = None,
) -> None:
    cfg = cfg or MultimodalComparisonConfig()
    set_comparison_seed(cfg.seed)

    x_train, y_train = generate_multimodal_multitarget_data(cfg.n_train, seed=cfg.seed)
    x_test, y_test = generate_multimodal_multitarget_data(cfg.n_test, seed=cfg.seed + 1)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)

    summary_rows: list[dict[str, object]] = []

    # Gaussian baseline
    gaussian_model = MLP(1, 4, hidden=cfg.hidden)
    gaussian_loss = create_gaussian_nll(covariance_type="diagonal")
    _, train_s = timed_call(
        _train_model,
        gaussian_model,
        gaussian_loss,
        loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
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
    mdn_loss = MDNLoss(n_components=cfg.mdn_components, n_features=2, covariance_type="diagonal")
    mdn_model = MLP(1, mdn_loss.expected_output_size, hidden=cfg.hidden)
    _, train_s = timed_call(
        _train_model,
        mdn_model,
        mdn_loss,
        loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )
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
        n_targets=2,
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

    print_fairness_notes(
        title="Multimodal Method Comparison",
        seed_policy="fixed global seed; deterministic synthetic train/test generation",
        train_budget=f"{cfg.epochs} epochs, batch_size={cfg.batch_size}, lr={cfg.lr}",
        metric_policy="Shared point metrics + NLL + energy score + marginal calibration error + runtime",
    )
    print_comparison_summary(
        "Multimodal / Multi-Target Comparison Summary",
        summary_rows,
        metric_order=["MSE", "MAE", "R2", "NLL", "Energy", "MCE", "train_s", "eval_s"],
    )
    print("\nFailure modes / caveats:")
    print("- Synthetic 2D data does not establish real-world ranking.")
    print("- Gaussian baseline is intentionally unimodal and may average across modes.")
    print("- Flow row is optional and may be skipped when zuko is not installed.")

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/multimodal_method_comparison.py",
            task="Multimodal / multi-target non-Gaussian",
            config=cfg,
            rows=summary_rows,
            notes=[
                "Flow row is optional when zuko is unavailable",
                "Common metrics include NLL, energy score, and marginal calibration error",
            ],
        )
        print(f"\nWrote summary JSON: {out}")


if __name__ == "__main__":
    main()
