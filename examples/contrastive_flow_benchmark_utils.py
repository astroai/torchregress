"""Shared helpers for contrastive-flow parameter-estimation comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from comparison_utils import compute_point_metrics
from torch.utils.data import DataLoader

from torchregress.losses import create_gaussian_nll


@dataclass(frozen=True)
class ParameterGrid:
    lows: tuple[float, ...]
    highs: tuple[float, ...]
    steps: tuple[int, ...]

    def axes(self) -> list[torch.Tensor]:
        return [
            torch.linspace(low, high, steps)
            for low, high, steps in zip(self.lows, self.highs, self.steps, strict=True)
        ]

    def tensor(self) -> torch.Tensor:
        axes = self.axes()
        return torch.cartesian_prod(*axes)


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


def train_supervised_density_model(
    model: nn.Module,
    loss_fn,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for params, summaries in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(params), summaries)
            loss.backward()
            optimizer.step()


def train_flow_density_model(
    context_model: nn.Module,
    flow: nn.Module,
    loss_fn,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(list(context_model.parameters()) + list(flow.parameters()), lr=lr)
    for _ in range(epochs):
        context_model.train()
        for params, summaries in loader:
            optimizer.zero_grad(set_to_none=True)
            context = context_model(params)
            loss = loss_fn(context, summaries)
            loss.backward()
            optimizer.step()


def train_contrastive_density_model(
    context_model: nn.Module,
    flow: nn.Module,
    loss_fn,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    negative_sampler: Callable[[int], torch.Tensor],
) -> None:
    optimizer = torch.optim.Adam(list(context_model.parameters()) + list(flow.parameters()), lr=lr)
    for _ in range(epochs):
        context_model.train()
        for params, summaries in loader:
            negative_params = negative_sampler(params.shape[0])
            optimizer.zero_grad(set_to_none=True)
            positive_context = context_model(params)
            negative_context = context_model(negative_params.reshape(-1, params.shape[-1])).reshape(
                params.shape[0],
                negative_params.shape[1],
                -1,
            )
            loss = loss_fn(positive_context, summaries, negative_context=negative_context)
            loss.backward()
            optimizer.step()


def score_gaussian_summary_model(
    model: nn.Module,
    summary: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        raw = model(grid)
        mean, log_var = torch.chunk(raw, 2, dim=-1)
        std = torch.exp(0.5 * log_var).clamp_min(1e-4)
        dist = torch.distributions.Normal(mean, std)
        repeated_summary = summary.unsqueeze(0).expand(grid.shape[0], -1)
        return dist.log_prob(repeated_summary).sum(dim=-1)


def score_flow_summary_model(
    context_model: nn.Module,
    loss_fn,
    summary: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        context = context_model(grid)
        repeated_summary = summary.unsqueeze(0).expand(grid.shape[0], -1)
        return loss_fn.log_prob(context, repeated_summary)


def estimate_parameters_from_scores(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    summaries: torch.Tensor,
    params_true: torch.Tensor,
    grid_spec: ParameterGrid,
) -> tuple[dict[str, float], torch.Tensor]:
    grid = grid_spec.tensor()
    estimates: list[torch.Tensor] = []
    for summary in summaries:
        scores = score_fn(summary, grid)
        best_idx = int(torch.argmax(scores).item())
        estimates.append(grid[best_idx])
    estimate_tensor = torch.stack(estimates)
    point = compute_point_metrics(estimate_tensor, params_true)
    abs_error = (estimate_tensor - params_true).abs()
    metrics = {
        **point,
        "ParamMAE": float(abs_error.mean().item()),
    }
    for dim in range(params_true.shape[1]):
        metrics[f"Dim{dim}_MAE"] = float(abs_error[:, dim].mean().item())
    return metrics, estimate_tensor


def gaussian_summary_loss(summary_dim: int):
    del summary_dim
    return create_gaussian_nll(covariance_type="diagonal")


def try_make_flow_losses(
    *,
    param_dim: int,
    summary_dim: int,
    hidden: int,
    context_dim: int,
    n_transforms: int,
) -> tuple[
    Optional[nn.Module], Optional[object], Optional[nn.Module], Optional[object], Optional[str]
]:
    try:
        from torchregress.losses import ContrastiveFlowLoss
        from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model
    except ImportError as exc:
        return None, None, None, None, str(exc)

    flow_context_model = MLP(param_dim, context_dim, hidden=hidden)
    flow = create_flow_model(
        n_features=summary_dim,
        context_dim=context_dim,
        flow_type="nsf",
        n_transforms=n_transforms,
        hidden_features=[hidden, hidden],
    )
    flow_loss = NormalizingFlowLoss(flow=flow, reduction="mean")

    contrastive_context_model = MLP(param_dim, context_dim, hidden=hidden)
    contrastive_flow = create_flow_model(
        n_features=summary_dim,
        context_dim=context_dim,
        flow_type="nsf",
        n_transforms=n_transforms,
        hidden_features=[hidden, hidden],
    )
    contrastive_loss = ContrastiveFlowLoss(
        flow=contrastive_flow,
        reduction="mean",
        temperature=0.7,
        margin=0.2,
    )
    return (
        flow_context_model,
        flow_loss,
        contrastive_context_model,
        contrastive_loss,
        None,
    )
