"""
Contrastive normalizing flows for nuisance-aware parameter estimation.

This example keeps the workflow PyTorch-native:

1. simulate pseudoexperiments under a signal-strength parameter and nuisance shift
2. summarize each pseudoexperiment into a fixed-width feature vector
3. train a conditional normalizing flow with a contrastive objective
4. recover parameters by scanning the learned likelihood over a small grid
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchregress.losses import ContrastiveFlowLoss, create_flow_model

MU_RANGE = (0.6, 1.4)
NUISANCE_RANGE = (-0.7, 0.7)


@dataclass
class ContrastiveFlowConfig:
    n_train: int = 512
    n_test: int = 48
    events_per_experiment: int = 128
    batch_size: int = 64
    n_epochs: int = 40
    n_negatives: int = 4
    context_dim: int = 32
    flow_type: str = "nsf"
    n_transforms: int = 4
    mu_grid_size: int = 31
    nuisance_grid_size: int = 21
    lr: float = 2e-3
    seed: int = 7
    make_plot: bool = True


def _rand_uniform(
    n: int,
    low: float,
    high: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    return low + (high - low) * torch.rand(n, generator=generator)


def _sample_experiment_events(
    mu: float,
    nuisance: float,
    *,
    n_events: int,
    generator: torch.Generator,
) -> torch.Tensor:
    signal_prob = min(max(0.18 + 0.24 * (mu - 1.0), 0.05), 0.55)
    is_signal = torch.rand(n_events, generator=generator) < signal_prob

    background = torch.randn(n_events, generator=generator) * (1.05 + 0.15 * abs(nuisance))
    background = background + 0.8 * nuisance

    signal = torch.randn(n_events, generator=generator) * 0.7
    signal = signal + 1.6 * mu + 0.35 * nuisance

    events = torch.where(is_signal, signal, background)
    tail = (events > (0.6 + 0.4 * nuisance)).float().mean().unsqueeze(0)
    return torch.cat([events, tail], dim=0)


def _summarize_experiment(events_with_tail: torch.Tensor) -> torch.Tensor:
    events = events_with_tail[:-1]
    tail_rate = events_with_tail[-1]
    mean = events.mean()
    centered = events - mean
    std = events.std(unbiased=False).clamp_min(1e-4)
    skew = (centered.pow(3).mean() / std.pow(3)).clamp(-8.0, 8.0)
    quantiles = torch.quantile(events, torch.tensor([0.1, 0.5, 0.9], dtype=events.dtype))
    high_score = (events > 1.5).float().mean()
    return torch.stack(
        [
            mean,
            std,
            skew,
            quantiles[0],
            quantiles[1],
            quantiles[2],
            high_score,
            tail_rate,
        ]
    )


def generate_parameterized_pseudoexperiments(
    n_experiments: int,
    *,
    n_events: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    mu = _rand_uniform(n_experiments, MU_RANGE[0], MU_RANGE[1], generator=generator)
    nuisance = _rand_uniform(
        n_experiments,
        NUISANCE_RANGE[0],
        NUISANCE_RANGE[1],
        generator=generator,
    )

    summaries = []
    for mu_i, nuisance_i in zip(mu.tolist(), nuisance.tolist()):
        events = _sample_experiment_events(
            mu_i,
            nuisance_i,
            n_events=n_events,
            generator=generator,
        )
        summaries.append(_summarize_experiment(events))

    params = torch.stack([mu, nuisance], dim=1)
    return params.float(), torch.stack(summaries).float()


class ParameterContextNet(nn.Module):
    """Map hypothesis parameters to flow conditioning vectors."""

    def __init__(self, input_dim: int = 2, context_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, context_dim),
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.net(params)


def sample_negative_parameters(
    batch_size: int,
    *,
    n_negatives: int,
    generator: torch.Generator,
) -> torch.Tensor:
    mu = _rand_uniform(batch_size * n_negatives, MU_RANGE[0], MU_RANGE[1], generator=generator)
    nuisance = _rand_uniform(
        batch_size * n_negatives,
        NUISANCE_RANGE[0],
        NUISANCE_RANGE[1],
        generator=generator,
    )
    return torch.stack([mu, nuisance], dim=1).reshape(batch_size, n_negatives, 2)


def train_contrastive_flow_model(
    model: nn.Module,
    loss_fn: ContrastiveFlowLoss,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    cfg: ContrastiveFlowConfig,
) -> list[float]:
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=cfg.lr,
    )
    generator = torch.Generator().manual_seed(cfg.seed + 17)
    history: list[float] = []

    for epoch in range(cfg.n_epochs):
        total = 0.0
        count = 0
        model.train()
        for params, summaries in loader:
            negative_params = sample_negative_parameters(
                params.shape[0],
                n_negatives=cfg.n_negatives,
                generator=generator,
            )
            optimizer.zero_grad(set_to_none=True)
            positive_context = model(params)
            negative_context = model(negative_params.reshape(-1, params.shape[-1])).reshape(
                params.shape[0],
                cfg.n_negatives,
                -1,
            )
            loss = loss_fn(positive_context, summaries, negative_context=negative_context)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * params.shape[0]
            count += params.shape[0]
        history.append(total / max(count, 1))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{cfg.n_epochs} contrastive loss: {history[-1]:.4f}")

    return history


def score_parameter_grid(
    model: nn.Module,
    loss_fn: ContrastiveFlowLoss,
    summary: torch.Tensor,
    mu_values: torch.Tensor,
    nuisance_values: torch.Tensor,
) -> torch.Tensor:
    grid = torch.cartesian_prod(mu_values, nuisance_values)
    with torch.no_grad():
        context = model(grid)
        repeated_summary = summary.unsqueeze(0).expand(grid.shape[0], -1)
        score = loss_fn.log_prob(context, repeated_summary)
    return score.reshape(mu_values.numel(), nuisance_values.numel())


def estimate_parameters_from_grid(
    model: nn.Module,
    loss_fn: ContrastiveFlowLoss,
    summary: torch.Tensor,
    *,
    mu_grid_size: int,
    nuisance_grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mu_values = torch.linspace(MU_RANGE[0], MU_RANGE[1], mu_grid_size)
    nuisance_values = torch.linspace(NUISANCE_RANGE[0], NUISANCE_RANGE[1], nuisance_grid_size)
    score_grid = score_parameter_grid(model, loss_fn, summary, mu_values, nuisance_values)
    best_index = torch.argmax(score_grid)
    mu_idx = best_index // nuisance_values.numel()
    nuisance_idx = best_index % nuisance_values.numel()
    estimate = torch.tensor([mu_values[mu_idx], nuisance_values[nuisance_idx]])
    return estimate, score_grid


def evaluate_parameter_scan(
    model: nn.Module,
    loss_fn: ContrastiveFlowLoss,
    params: torch.Tensor,
    summaries: torch.Tensor,
    *,
    cfg: ContrastiveFlowConfig,
) -> dict[str, float]:
    estimates = []
    for summary in summaries:
        estimate, _ = estimate_parameters_from_grid(
            model,
            loss_fn,
            summary,
            mu_grid_size=cfg.mu_grid_size,
            nuisance_grid_size=cfg.nuisance_grid_size,
        )
        estimates.append(estimate)

    estimate_tensor = torch.stack(estimates)
    abs_error = (estimate_tensor - params).abs()
    return {
        "mu_mae": float(abs_error[:, 0].mean().item()),
        "nuisance_mae": float(abs_error[:, 1].mean().item()),
    }


def plot_parameter_scan(
    score_grid: torch.Tensor,
    truth: torch.Tensor,
    estimate: torch.Tensor,
    *,
    cfg: ContrastiveFlowConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(
        score_grid.T,
        origin="lower",
        aspect="auto",
        extent=[MU_RANGE[0], MU_RANGE[1], NUISANCE_RANGE[0], NUISANCE_RANGE[1]],
    )
    ax.scatter(truth[0], truth[1], color="white", marker="x", s=80, label="truth")
    ax.scatter(estimate[0], estimate[1], color="red", marker="o", s=45, label="estimate")
    ax.set_xlabel("signal strength")
    ax.set_ylabel("nuisance shift")
    ax.set_title("Learned parameter scan")
    ax.legend(loc="upper left")
    fig.colorbar(image, ax=ax, label="flow log-likelihood")
    fig.tight_layout()
    plt.show()


def main(cfg: ContrastiveFlowConfig | None = None) -> dict[str, float]:
    cfg = cfg or ContrastiveFlowConfig()
    torch.manual_seed(cfg.seed)

    train_params, train_summaries = generate_parameterized_pseudoexperiments(
        cfg.n_train,
        n_events=cfg.events_per_experiment,
        seed=cfg.seed,
    )
    test_params, test_summaries = generate_parameterized_pseudoexperiments(
        cfg.n_test,
        n_events=cfg.events_per_experiment,
        seed=cfg.seed + 1,
    )

    model = ParameterContextNet(context_dim=cfg.context_dim)
    flow = create_flow_model(
        n_features=train_summaries.shape[1],
        context_dim=cfg.context_dim,
        flow_type=cfg.flow_type,
        n_transforms=cfg.n_transforms,
        hidden_features=64,
        n_hidden_layers=2,
    )
    loss_fn = ContrastiveFlowLoss(flow=flow, temperature=0.7, margin=0.2)

    loader = DataLoader(
        TensorDataset(train_params, train_summaries),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    train_contrastive_flow_model(model, loss_fn, loader, cfg=cfg)

    metrics = evaluate_parameter_scan(model, loss_fn, test_params, test_summaries, cfg=cfg)
    print(
        "Held-out scan metrics:",
        f"mu_mae={metrics['mu_mae']:.3f}",
        f"nuisance_mae={metrics['nuisance_mae']:.3f}",
    )

    example_estimate, score_grid = estimate_parameters_from_grid(
        model,
        loss_fn,
        test_summaries[0],
        mu_grid_size=cfg.mu_grid_size,
        nuisance_grid_size=cfg.nuisance_grid_size,
    )
    print(
        "Single pseudoexperiment:",
        f"truth={test_params[0].tolist()}",
        f"estimate={example_estimate.tolist()}",
    )
    if cfg.make_plot:
        plot_parameter_scan(score_grid, test_params[0], example_estimate, cfg=cfg)

    return metrics


if __name__ == "__main__":
    main()
