"""
Contrastive normalizing flows for nuisance-aware parameter estimation.

Modes (``--mode``):

``demo``
    The PyTorch-native ``ContrastiveFlowLoss`` workflow:

    1. simulate pseudoexperiments under a signal-strength parameter and nuisance shift
    2. summarize each pseudoexperiment into a fixed-width feature vector
    3. train a conditional normalizing flow with a contrastive objective
    4. recover parameters by scanning the learned likelihood over a small grid

    Prints held-out scan MAE and shows the likelihood scan for one pseudoexperiment
    (``--no-plot`` to skip the figure).

``compare``
    Shared-budget comparison on the same pseudoexperiment generator: diagonal
    Gaussian summary density, plain conditional flow NLL, contrastive flow,
    classifier neural ratio estimation (NRE) and an NPE-style posterior flow. Reports
    parameter MAE on a shared scan grid, SBC/TARP posterior-calibration diagnostics and
    runtime; ``--summary-json-path`` writes the machine-readable summary.

``all`` (default)
    Runs ``demo`` then ``compare``. Each mode reseeds itself, so its output matches a
    standalone run.

Requires the optional ``zuko`` dependency (``pip install 'torchregress[flows]'``).
"""

import argparse
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.pyplot as plt
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
from torchregress.losses import ContrastiveFlowLoss, create_flow_model, create_gaussian_nll

MODES = ("demo", "compare")

MU_RANGE = (0.6, 1.4)
NUISANCE_RANGE = (-0.7, 0.7)


# ============================================================================
# Mode "demo": contrastive flow + likelihood scan
# ============================================================================


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


def run_demo(cfg: ContrastiveFlowConfig | None = None) -> dict[str, float]:
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


# ============================================================================
# Mode "compare": shared-budget baselines (Gaussian summary, flow NLL, contrastive
# flow, NRE, NPE) on the same pseudoexperiments
# ============================================================================


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


@dataclass(frozen=True)
class ContrastiveFlowComparisonConfig:
    n_train: int = 512
    n_test: int = 96
    events_per_experiment: int = 128
    batch_size: int = 64
    epochs: int = 30
    lr: float = 2e-3
    seed: int = 17
    hidden: int = 64
    flow_context_dim: int = 32
    flow_transforms: int = 4
    n_negatives: int = 4
    mu_grid_size: int = 31
    nuisance_grid_size: int = 21
    n_posterior_samples: int = 256


def _sample_negative_params(
    cfg: ContrastiveFlowComparisonConfig,
    batch_size: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    mu = torch.empty(batch_size, cfg.n_negatives)
    nuisance = torch.empty(batch_size, cfg.n_negatives)
    if generator is None:
        mu.uniform_(MU_RANGE[0], MU_RANGE[1])
        nuisance.uniform_(NUISANCE_RANGE[0], NUISANCE_RANGE[1])
    else:
        mu.uniform_(MU_RANGE[0], MU_RANGE[1], generator=generator)
        nuisance.uniform_(NUISANCE_RANGE[0], NUISANCE_RANGE[1], generator=generator)
    return torch.stack([mu, nuisance], dim=-1)


def run_comparison(
    cfg: Optional[ContrastiveFlowComparisonConfig] = None,
    summary_json_path: str | None = None,
) -> list[dict[str, object]]:
    cfg = cfg or ContrastiveFlowComparisonConfig()
    set_comparison_seed(cfg.seed)

    params_train, summaries_train = generate_parameterized_pseudoexperiments(
        cfg.n_train,
        n_events=cfg.events_per_experiment,
        seed=cfg.seed,
    )
    params_test, summaries_test = generate_parameterized_pseudoexperiments(
        cfg.n_test,
        n_events=cfg.events_per_experiment,
        seed=cfg.seed + 1,
    )
    loader = DataLoader(
        TensorDataset(params_train, summaries_train),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    grid = ParameterGrid(
        lows=(MU_RANGE[0], NUISANCE_RANGE[0]),
        highs=(MU_RANGE[1], NUISANCE_RANGE[1]),
        steps=(cfg.mu_grid_size, cfg.nuisance_grid_size),
    )

    summary_rows: list[dict[str, object]] = []
    calibration_generator = torch.Generator().manual_seed(cfg.seed + 7)
    reference_params = torch.stack(
        [
            torch.empty(cfg.n_test).uniform_(
                MU_RANGE[0], MU_RANGE[1], generator=calibration_generator
            ),
            torch.empty(cfg.n_test).uniform_(
                NUISANCE_RANGE[0], NUISANCE_RANGE[1], generator=calibration_generator
            ),
        ],
        dim=-1,
    )

    def scan_metrics(score_fn):
        return evaluate_scan_with_calibration(
            score_fn,
            summaries_test,
            params_test,
            grid,
            n_posterior_samples=cfg.n_posterior_samples,
            reference_params=reference_params,
            generator=calibration_generator,
        )[0]

    gaussian_model = MLP(2, summaries_train.shape[1] * 2, hidden=cfg.hidden)
    gaussian_loss = gaussian_summary_loss(summaries_train.shape[1])
    _, train_s = timed_call(
        train_supervised_density_model,
        gaussian_model,
        gaussian_loss,
        loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
    )
    gaussian_metrics, eval_s = timed_call(
        scan_metrics,
        lambda summary, grid_tensor: score_gaussian_summary_model(
            gaussian_model, summary, grid_tensor
        ),
    )
    summary_rows.append(
        {
            "Method": "GaussianSummary",
            **gaussian_metrics,
            "train_s": train_s,
            "eval_s": eval_s,
            "Notes": "diagonal Gaussian density over pseudoexperiment summaries",
        }
    )

    flow_context_model, flow_loss, contrastive_context_model, contrastive_loss, flow_err = (
        try_make_flow_losses(
            param_dim=2,
            summary_dim=summaries_train.shape[1],
            hidden=cfg.hidden,
            context_dim=cfg.flow_context_dim,
            n_transforms=cfg.flow_transforms,
        )
    )

    if flow_context_model is not None and flow_loss is not None:
        _, train_s = timed_call(
            train_flow_density_model,
            flow_context_model,
            getattr(flow_loss, "flow"),
            flow_loss,
            loader,
            epochs=cfg.epochs,
            lr=cfg.lr,
        )
        flow_metrics, eval_s = timed_call(
            scan_metrics,
            lambda summary, grid_tensor: score_flow_summary_model(
                flow_context_model, flow_loss, summary, grid_tensor
            ),
        )
        summary_rows.append(
            {
                "Method": "NormalizingFlow",
                **flow_metrics,
                "train_s": train_s,
                "eval_s": eval_s,
                "Notes": "conditional flow NLL over pseudoexperiment summaries",
            }
        )
    else:
        summary_rows.append(
            {
                "Method": "NormalizingFlow",
                "ParamMAE": None,
                "Dim0_MAE": None,
                "Dim1_MAE": None,
                "train_s": None,
                "eval_s": None,
                "Notes": f"skipped (optional dependency unavailable: {flow_err})",
            }
        )

    if contrastive_context_model is not None and contrastive_loss is not None:
        contrastive_generator = torch.Generator().manual_seed(cfg.seed + 11)
        _, train_s = timed_call(
            train_contrastive_density_model,
            contrastive_context_model,
            getattr(contrastive_loss, "flow"),
            contrastive_loss,
            loader,
            epochs=cfg.epochs,
            lr=cfg.lr,
            negative_sampler=lambda batch_size: _sample_negative_params(
                cfg, batch_size, generator=contrastive_generator
            ),
        )
        contrastive_metrics, eval_s = timed_call(
            scan_metrics,
            lambda summary, grid_tensor: score_flow_summary_model(
                contrastive_context_model, contrastive_loss, summary, grid_tensor
            ),
        )
        summary_rows.append(
            {
                "Method": "ContrastiveFlow",
                **contrastive_metrics,
                "train_s": train_s,
                "eval_s": eval_s,
                "Notes": "contrastive likelihood-ratio flow over positive vs alternate hypotheses",
            }
        )
    else:
        summary_rows.append(
            {
                "Method": "ContrastiveFlow",
                "ParamMAE": None,
                "Dim0_MAE": None,
                "Dim1_MAE": None,
                "train_s": None,
                "eval_s": None,
                "Notes": f"skipped (optional dependency unavailable: {flow_err})",
            }
        )

    nre_loss, npe_summary_encoder, npe_loss, baseline_err = try_make_ratio_and_posterior_models(
        param_dim=2,
        summary_dim=summaries_train.shape[1],
        hidden=cfg.hidden,
        context_dim=cfg.flow_context_dim,
        n_transforms=cfg.flow_transforms,
    )

    if nre_loss is not None:
        nre_generator = torch.Generator().manual_seed(cfg.seed + 13)
        _, train_s = timed_call(
            train_nre_model,
            nre_loss.classifier,
            nre_loss,
            loader,
            epochs=cfg.epochs,
            lr=cfg.lr,
            negative_sampler=lambda batch_size: _sample_negative_params(
                cfg, batch_size, generator=nre_generator
            ),
            generator=nre_generator,
        )
        nre_metrics, eval_s = timed_call(
            scan_metrics,
            lambda summary, grid_tensor: score_nre_summary_model(
                nre_loss.classifier, summary, grid_tensor
            ),
        )
        summary_rows.append(
            {
                "Method": "NeuralRatio",
                **nre_metrics,
                "train_s": train_s,
                "eval_s": eval_s,
                "Notes": "classifier NRE (mismatched joint-vs-product pairs) on the scan grid",
            }
        )
    else:
        summary_rows.append(
            {
                "Method": "NeuralRatio",
                "ParamMAE": None,
                "Dim0_MAE": None,
                "Dim1_MAE": None,
                "train_s": None,
                "eval_s": None,
                "Notes": f"skipped ({baseline_err})",
            }
        )

    if npe_loss is not None and npe_summary_encoder is not None:
        _, train_s = timed_call(
            train_posterior_flow_model,
            npe_summary_encoder,
            npe_loss,
            loader,
            epochs=cfg.epochs,
            lr=cfg.lr,
        )

        def npe_evaluate() -> dict[str, float]:
            samples = sample_posterior_flow_model(
                npe_summary_encoder,
                npe_loss,
                summaries_test,
                n_samples=cfg.n_posterior_samples,
            )
            return posterior_point_metrics(samples, params_test, reference_params)

        npe_metrics, eval_s = timed_call(npe_evaluate)
        summary_rows.append(
            {
                "Method": "NormalizingFlowPosterior",
                **npe_metrics,
                "train_s": train_s,
                "eval_s": eval_s,
                "Notes": "NPE-style conditional flow p(theta|summary) trained with NLL",
            }
        )
    else:
        summary_rows.append(
            {
                "Method": "NormalizingFlowPosterior",
                "ParamMAE": None,
                "Dim0_MAE": None,
                "Dim1_MAE": None,
                "train_s": None,
                "eval_s": None,
                "Notes": f"skipped ({baseline_err})",
            }
        )

    print_fairness_notes(
        title="Contrastive Flow Parameter Estimation Comparison",
        seed_policy="fixed synthetic generator and shared train/test pseudoexperiments",
        train_budget=f"{cfg.epochs} epochs, batch_size={cfg.batch_size}, lr={cfg.lr}",
        metric_policy="parameter MAE on shared scan grid + runtime",
    )
    print_comparison_summary(
        "Synthetic Parameter-Estimation Comparison Summary",
        summary_rows,
        metric_order=[
            "MSE",
            "MAE",
            "R2",
            "ParamMAE",
            "Dim0_MAE",
            "Dim1_MAE",
            "SBC_min_p",
            "TARP_err",
            "train_s",
            "eval_s",
        ],
    )

    if summary_json_path is not None:
        out = write_comparison_summary_json(
            summary_json_path,
            example="examples/contrastive_flow_parameter_estimation.py --mode compare",
            task="Synthetic parameter estimation under nuisance shift",
            config=cfg,
            rows=summary_rows,
            notes=[
                "Shared pseudoexperiment generator and shared parameter scan grid",
                "Flow/NRE/NPE rows are optional when zuko is unavailable",
                "SBC/TARP are computed from posterior weights resampled on the scan grid "
                "(scan methods) or from NPE posterior samples",
            ],
        )
        print(f"\nWrote summary JSON: {out}")

    return summary_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument(
        "--no-plot", action="store_true", help="demo mode: skip the likelihood-scan figure"
    )
    parser.add_argument(
        "--summary-json-path",
        default=None,
        help="compare mode: write the comparison summary JSON here",
    )
    args = parser.parse_args(argv)

    modes = MODES if args.mode == "all" else (args.mode,)
    for i, mode in enumerate(modes):
        if i:
            print()
        if mode == "demo":
            run_demo(ContrastiveFlowConfig(make_plot=not args.no_plot))
        else:
            run_comparison(summary_json_path=args.summary_json_path)


if __name__ == "__main__":
    main()
