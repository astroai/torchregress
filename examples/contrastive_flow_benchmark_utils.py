"""Shared helpers for contrastive-flow parameter-estimation comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchregress.comparison import compute_point_metrics
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
    except ImportError as exc:
        return None, None, None, None, str(exc)


def train_nre_model(
    classifier: nn.Module,
    loss_fn,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    negative_sampler: Callable[[int], torch.Tensor],
    generator: torch.Generator,
    negative_target_mode: str = "mismatched",
) -> None:
    """Train a classifier-based neural ratio estimator.

    ``negative_target_mode="mismatched"`` forms canonical joint-vs-product NRE
    pairs by matching each negative context with an independently permuted
    observation; ``"paired"`` scores the negative contexts against the same
    observation (likelihood ratios between hypotheses).
    """
    if negative_target_mode not in {"mismatched", "paired"}:
        raise ValueError(
            f"negative_target_mode must be 'mismatched' or 'paired', got {negative_target_mode!r}"
        )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    for _ in range(epochs):
        classifier.train()
        for params, summaries in loader:
            negative_params = negative_sampler(params.shape[0])
            negative_summaries = None
            if negative_target_mode == "mismatched":
                permutation = torch.randperm(summaries.shape[0], generator=generator)
                negative_summaries = (
                    summaries[permutation].unsqueeze(1).expand(-1, negative_params.shape[1], -1)
                )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(
                params,
                summaries,
                negative_context=negative_params,
                negative_target=negative_summaries,
            )
            loss.backward()
            optimizer.step()


def score_nre_summary_model(
    classifier: nn.Module,
    summary: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the NRE classifier logit for every grid hypothesis."""
    from torchregress.losses import neural_ratio_logits

    with torch.no_grad():
        repeated_summary = summary.unsqueeze(0).expand(grid.shape[0], -1)
        return neural_ratio_logits(classifier, grid, repeated_summary)


def train_posterior_flow_model(
    summary_encoder: nn.Module,
    flow_loss,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
) -> None:
    """Train a conditional flow p(theta | summary) with plain NLL (NPE)."""
    optimizer = torch.optim.Adam(
        list(summary_encoder.parameters()) + list(flow_loss.parameters()),
        lr=lr,
    )
    for _ in range(epochs):
        summary_encoder.train()
        for params, summaries in loader:
            optimizer.zero_grad(set_to_none=True)
            context = summary_encoder(summaries)
            loss = flow_loss(context, params)
            loss.backward()
            optimizer.step()


def sample_posterior_flow_model(
    summary_encoder: nn.Module,
    flow_loss,
    summaries: torch.Tensor,
    *,
    n_samples: int,
) -> torch.Tensor:
    """Draw posterior samples [N, n_samples, D] for each summary."""
    with torch.no_grad():
        context = summary_encoder(summaries)
        return flow_loss.sample(context, n_samples=n_samples)


def _resample_grid_posterior(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    summaries: torch.Tensor,
    grid: torch.Tensor,
    *,
    n_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Resample candidate hypotheses from the grid posterior weights."""
    from torchregress.inference import posterior_log_weights

    samples: list[torch.Tensor] = []
    for summary in summaries:
        scores = score_fn(summary, grid)
        log_weights = posterior_log_weights(scores)
        index = torch.multinomial(
            log_weights.exp(), n_samples, replacement=True, generator=generator
        )
        samples.append(grid[index])
    return torch.stack(samples)


def calibration_metrics_from_samples(
    samples: torch.Tensor,
    params_true: torch.Tensor,
    reference_params: torch.Tensor,
) -> dict[str, float]:
    """SBC/TARP posterior-calibration diagnostics for [N, S, D] samples."""
    from torchregress.metrics import (
        sbc_uniformity_pvalue,
        simulation_based_calibration_ranks,
        tarp_coverage,
    )

    ranks = simulation_based_calibration_ranks(samples, params_true)
    # Keep the chi-square approximation valid for the number of test observations.
    n_ranks = int(ranks.shape[0])
    n_bins = max(2, min(10, n_ranks // 10))
    try:
        uniformity = sbc_uniformity_pvalue(ranks, n_bins=n_bins)
        sbc_p = float(uniformity["min_p_value"])
    except ValueError:
        sbc_p = float("nan")
    tarp = tarp_coverage(samples, params_true, reference_params)
    return {
        "SBC_min_p": sbc_p,
        "TARP_err": float(tarp["tarp_error"]),
    }


def posterior_point_metrics(
    samples: torch.Tensor,
    params_true: torch.Tensor,
    reference_params: torch.Tensor,
) -> dict[str, float]:
    """Posterior-mean point metrics plus SBC/TARP calibration diagnostics."""
    posterior_mean = samples.mean(dim=1)
    point = compute_point_metrics(posterior_mean, params_true)
    abs_error = (posterior_mean - params_true).abs()
    metrics = {
        **point,
        "ParamMAE": float(abs_error.mean().item()),
        **calibration_metrics_from_samples(samples, params_true, reference_params),
    }
    for dim in range(params_true.shape[1]):
        metrics[f"Dim{dim}_MAE"] = float(abs_error[:, dim].mean().item())
    return metrics


def evaluate_scan_with_calibration(
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    summaries: torch.Tensor,
    params_true: torch.Tensor,
    grid_spec: ParameterGrid,
    *,
    n_posterior_samples: int,
    reference_params: torch.Tensor,
    generator: torch.Generator,
) -> tuple[dict[str, float], torch.Tensor]:
    """MLE-from-grid metrics plus SBC/TARP on the resampled grid posterior."""
    metrics, estimate_tensor = estimate_parameters_from_scores(
        score_fn, summaries, params_true, grid_spec
    )
    grid = grid_spec.tensor()
    samples = _resample_grid_posterior(
        score_fn,
        summaries,
        grid,
        n_samples=n_posterior_samples,
        generator=generator,
    )
    metrics.update(calibration_metrics_from_samples(samples, params_true, reference_params))
    return metrics, estimate_tensor


def try_make_ratio_and_posterior_models(
    *,
    param_dim: int,
    summary_dim: int,
    hidden: int,
    context_dim: int,
    n_transforms: int,
) -> tuple[
    Optional[nn.Module], Optional[object], Optional[nn.Module], Optional[object], Optional[str]
]:
    """Build the classifier-based NRE and the NPE posterior-flow baselines."""
    try:
        from torchregress.losses import NeuralRatioLoss, build_ratio_classifier
        from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model

        classifier = build_ratio_classifier(
            context_dim=param_dim,
            observable_dim=summary_dim,
            hidden_dims=(hidden, hidden),
        )
        nre_loss = NeuralRatioLoss(classifier, criterion="bce", reduction="mean")

        summary_encoder = MLP(summary_dim, context_dim, hidden=hidden)
        posterior_flow = create_flow_model(
            n_features=param_dim,
            context_dim=context_dim,
            flow_type="nsf",
            n_transforms=n_transforms,
            hidden_features=[hidden, hidden],
        )
        posterior_loss = NormalizingFlowLoss(flow=posterior_flow, reduction="mean")
        return nre_loss, summary_encoder, posterior_loss, None
    except ImportError as exc:  # pragma: no cover - optional zuko dependency
        return None, None, None, str(exc)
