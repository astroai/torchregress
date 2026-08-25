"""
Distribution metrics for evaluating probabilistic regression models.
"""

import math
import warnings
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import torch
from torchmetrics import Metric

from torchregress.metrics.utils import convert_to_tensor, metric_state_tensor

_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


class ContinuousRankedProbabilityScore(Metric):
    """
    Calculate Continuous Ranked Probability Score (CRPS) for probabilistic forecasts.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("crps_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred_quantiles: Dict[float, torch.Tensor], y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        y_true = convert_to_tensor(y_true)

        quantiles = sorted(y_pred_quantiles.keys())
        if len(quantiles) < 2:
            raise ValueError("At least 2 quantile levels are required for CRPS calculation")

        # Vectorized validation
        if torch.isnan(y_true).any() or torch.isinf(y_true).any():
            raise ValueError("y_true contains NaN or infinite values")

        forecasts = [convert_to_tensor(y_pred_quantiles[q]) for q in quantiles]
        forecast_tensor = torch.stack(forecasts)

        if torch.isnan(forecast_tensor).any() or torch.isinf(forecast_tensor).any():
            raise ValueError("y_pred contains NaN or infinite values")

        # Check shapes using first quantile
        y_pred_sample = forecasts[0]
        if y_pred_sample.dim() == 0 or y_true.dim() == 0:
            raise ValueError("Inputs cannot be scalars, must have at least one dimension")

        if y_pred_sample.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"y_pred and y_true must have same batch size. "
                f"Got y_pred: {y_pred_sample.shape}, y_true: {y_true.shape}"
            )

        if y_pred_sample.shape != y_true.shape:
            try:
                _ = y_pred_sample + y_true
            except RuntimeError:
                raise ValueError(
                    f"y_pred shape {y_pred_sample.shape} and y_true shape {y_true.shape} "
                    "are not compatible"
                )
        quantile_tensor = torch.tensor(quantiles, device=forecast_tensor.device)

        zero_tensor = torch.tensor([0.0], device=quantile_tensor.device)
        one_tensor = torch.tensor([1.0], device=quantile_tensor.device)
        weights = torch.diff(torch.cat([zero_tensor, quantile_tensor, one_tensor]))

        diff = y_true.unsqueeze(0) - forecast_tensor
        view_shape = (-1,) + (1,) * y_true.ndim
        q_tensor = quantile_tensor.view(*view_shape)
        quantile_loss = torch.max(q_tensor * diff, (q_tensor - 1) * diff)
        # ponytail: trapezoidal rule for CRPS = 2∫QL dτ ≈ 2 Σ_i w_i QL_i,
        # w_i = (τ_{i+1} - τ_{i-1}) / 2 (TR-MET-01: the factor 2 was missing)
        weights_tensor = (0.5 * (weights[:-1] + weights[1:])).view(*view_shape)
        crps_values = 2.0 * torch.sum(weights_tensor * quantile_loss, dim=0)

        metric_state_tensor(self.crps_sum).add_(torch.sum(crps_values))
        metric_state_tensor(self.total).add_(torch.as_tensor(y_true.numel(), device=y_true.device))

    def compute(self) -> torch.Tensor:
        """Compute CRPS."""
        return metric_state_tensor(self.crps_sum) / metric_state_tensor(self.total)


class EnergyScore(Metric):
    """
    Calculate Energy Score for multivariate probabilistic forecasts.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, beta: float = 1.0, max_pairs: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.max_pairs = max_pairs
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_samples: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        y_true = convert_to_tensor(y_true)
        y_samples = convert_to_tensor(y_samples)

        n_samples = y_samples.shape[0]
        batch_size = y_true.shape[0]

        if self.max_pairs is not None and n_samples > self.max_pairs:
            indices = torch.randperm(n_samples, device=y_samples.device)[: self.max_pairs]
            y_samples = y_samples[indices]
            n_samples = self.max_pairs
        diff = y_samples - y_true.unsqueeze(0)

        # TR-MET-10: both terms must use the same geometry —
        # term1 = E||X - y||^beta (Euclidean norm to the beta), term2 = ||X - X'||^beta.
        norms = (diff.abs().pow(2).sum(-1)).pow(self.beta / 2.0)

        term1 = torch.mean(norms, dim=0)

        y_samples_p = y_samples.permute(1, 0, 2)
        dists = torch.cdist(y_samples_p, y_samples_p, p=2)

        if self.beta != 1.0:
            dists = torch.pow(dists, self.beta)

        term2 = torch.sum(dists, dim=(1, 2))
        n_pairs = n_samples * (n_samples - 1) // 2

        if n_pairs > 0:
            term2 = term2 / (4.0 * n_pairs)
        else:
            term2 = torch.zeros_like(term2)

        energy_scores = term1 - term2
        metric_state_tensor(self.score_sum).add_(torch.sum(energy_scores))
        metric_state_tensor(self.total).add_(torch.as_tensor(batch_size, device=y_true.device))

    def compute(self) -> torch.Tensor:
        """Compute energy score."""
        return metric_state_tensor(self.score_sum) / metric_state_tensor(self.total)


def probability_integral_transform(
    cdf_fn: Any,
    y_true: Union[torch.Tensor, np.ndarray],
    n_bins: int = 10,
    return_histogram: bool = False,
    as_numpy: bool = False,
) -> Union[torch.Tensor, np.ndarray, Dict[str, Union[torch.Tensor, float]]]:
    y_true_t = convert_to_tensor(y_true)
    pit_values = convert_to_tensor(cdf_fn(y_true_t))

    _to_np = as_numpy or isinstance(y_true, np.ndarray)

    if not return_histogram:
        if _to_np:
            return pit_values.numpy()
        return pit_values

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=pit_values.device)
    counts = torch.histogram(pit_values, bin_edges)[0]
    expected = pit_values.numel() / n_bins
    uniformity_chi2 = torch.sum((counts - expected) ** 2 / max(expected, 1.0))
    ks_statistic = kolmogorov_smirnov_uniform_statistic(pit_values)

    out: Dict[str, Any] = {
        "pit_values": pit_values,
        "histogram_counts": counts,
        "bin_edges": bin_edges,
        "uniformity_chi2": uniformity_chi2,
        "uniformity_ks": ks_statistic,
    }
    if _to_np:
        out = {
            k: (float(v) if v.ndim == 0 else v.numpy()) if torch.is_tensor(v) else v
            for k, v in out.items()
        }
    return out


def kolmogorov_smirnov_uniform_statistic(
    pit_values: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Kolmogorov-Smirnov statistic against Uniform(0, 1)."""
    values = convert_to_tensor(pit_values).flatten()
    if values.numel() == 0:
        raise ValueError("pit_values must contain at least one value")
    values = torch.sort(values.clamp(0.0, 1.0))[0]
    n = values.numel()
    ranks = torch.arange(1, n + 1, device=values.device, dtype=values.dtype)
    cdf_upper = ranks / n
    cdf_lower = (ranks - 1) / n
    d_plus = torch.max(cdf_upper - values)
    d_minus = torch.max(values - cdf_lower)
    return torch.maximum(d_plus, d_minus)


def conditional_density_estimation_loss(
    support: Union[torch.Tensor, np.ndarray],
    density: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    reduction: str = "mean",
) -> Union[torch.Tensor, float]:
    """Empirical conditional density estimation loss on a 1D support grid.

    This implements the standard CDE benchmark score up to an additive constant:

    ``L = ∫ f_hat(y | x)^2 dy - 2 f_hat(y_true | x)``

    The support is treated as an ordered 1D grid and the integral is approximated
    with a trapezoidal rule. ``density`` must therefore represent density values,
    not probabilities.
    """
    support_t = convert_to_tensor(support).flatten()
    density_t = convert_to_tensor(density)
    y_true_t = convert_to_tensor(y_true).to(density_t.device).flatten()

    if support_t.ndim != 1 or support_t.numel() < 2:
        raise ValueError("support must be a 1D tensor with at least two points")
    if density_t.ndim != 2:
        raise ValueError("density must have shape [batch, support]")
    if density_t.shape[1] != support_t.numel():
        raise ValueError(
            f"density support mismatch: got {density_t.shape[1]} support values for "
            f"{support_t.numel()} support points"
        )
    if density_t.shape[0] != y_true_t.shape[0]:
        raise ValueError(
            f"batch mismatch: density has {density_t.shape[0]} rows but y_true has "
            f"{y_true_t.shape[0]} values"
        )

    support_t = support_t.to(device=density_t.device, dtype=density_t.dtype)
    if not torch.all(support_t[1:] > support_t[:-1]):
        raise ValueError("support must be strictly increasing")

    integral_term = torch.trapezoid(density_t.square(), support_t, dim=-1)
    density_at_y = _interp1d(support_t, density_t, y_true_t)
    loss = integral_term - 2.0 * density_at_y

    if reduction == "none":
        return loss
    if reduction == "sum":
        return float(loss.sum().item())
    return float(loss.mean().item())


def highest_posterior_density_level(
    support: Union[torch.Tensor, np.ndarray],
    density: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Return HPD calibration levels for 1D predictive densities.

    For a calibrated predictive density, these levels should be uniformly distributed
    on ``[0, 1]``. Lower values indicate that the observed target lies in a denser,
    more central region of the predictive distribution.
    """
    support_t = convert_to_tensor(support).flatten()
    density_t = convert_to_tensor(density)
    y_true_t = convert_to_tensor(y_true).to(density_t.device).flatten()

    if support_t.ndim != 1 or support_t.numel() < 2:
        raise ValueError("support must be a 1D tensor with at least two points")
    if not torch.all(support_t[1:] > support_t[:-1]):
        raise ValueError("support must be strictly increasing")
    if density_t.ndim != 2 or density_t.shape[1] != support_t.numel():
        raise ValueError("density must have shape [batch, support]")
    if density_t.shape[0] != y_true_t.shape[0]:
        raise ValueError("density and y_true must share the batch dimension")

    support_t = support_t.to(device=density_t.device, dtype=density_t.dtype)
    delta = _support_widths(support_t)
    normalized_density = density_t / torch.trapezoid(density_t, support_t, dim=-1).clamp_min(
        1.0e-8
    ).unsqueeze(-1)
    density_at_y = _interp1d(support_t, normalized_density, y_true_t)
    mask = normalized_density >= density_at_y.unsqueeze(-1)
    mass = torch.sum(
        normalized_density * delta.unsqueeze(0) * mask.to(normalized_density.dtype), dim=-1
    )
    return mass.clamp(0.0, 1.0)


def highest_posterior_density_coverage(
    support: Union[torch.Tensor, np.ndarray],
    density: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.1,
) -> float:
    """Coverage of the HPD region with nominal mass ``alpha``."""
    levels = highest_posterior_density_level(support, density, y_true)
    return float((levels <= float(alpha)).to(levels.dtype).mean().item())


def gaussian_nll(
    mean: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    var: Union[torch.Tensor, np.ndarray],
    reduction: str = "mean",
) -> Union[torch.Tensor, float]:
    """
    Functional Gaussian negative log-likelihood for diagonal Gaussian predictions.
    """
    mean_t = convert_to_tensor(mean)
    y_true_t = convert_to_tensor(y_true).to(mean_t.device)
    var_t = convert_to_tensor(var).to(mean_t.device).clamp(min=1e-8)

    nll = 0.5 * (torch.log(2 * torch.pi * var_t) + (y_true_t - mean_t) ** 2 / var_t)
    if reduction == "none":
        return nll
    if reduction == "sum":
        return float(torch.sum(nll).item())
    return float(torch.mean(nll).item())


def crps_gaussian(
    mean: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    std: Union[torch.Tensor, np.ndarray],
    reduction: str = "mean",
) -> Union[torch.Tensor, float]:
    """Analytic CRPS for a univariate Gaussian predictive distribution.

    Args:
        mean: Predicted mean.
        y_true: Ground truth values.
        std: Predicted standard deviation.
        reduction: Reduction method ('none', 'mean', 'sum').

    Returns:
        CRPS value (float with 'mean'/'sum', Tensor with 'none').
    """
    mean_t = convert_to_tensor(mean)
    y_true_t = convert_to_tensor(y_true).to(mean_t.device)
    std_t = convert_to_tensor(std).to(mean_t.device).clamp(min=1e-8)

    z = (y_true_t - mean_t) / std_t
    standard = torch.distributions.Normal(
        torch.tensor(0.0, device=mean_t.device),
        torch.tensor(1.0, device=mean_t.device),
    )
    pdf = torch.exp(standard.log_prob(z))
    cdf = standard.cdf(z)
    crps = std_t * (z * (2 * cdf - 1) + 2 * pdf - _INV_SQRT_PI)

    if reduction == "none":
        return cast(torch.Tensor, crps)
    if reduction == "sum":
        return float(torch.sum(crps).item())
    return float(torch.mean(crps).item())


def _support_widths(support: torch.Tensor) -> torch.Tensor:
    widths = torch.empty_like(support)
    widths[1:-1] = 0.5 * (support[2:] - support[:-2])
    widths[0] = support[1] - support[0]
    widths[-1] = support[-1] - support[-2]
    return widths.clamp_min(1.0e-8)


def _interp1d(
    support: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    support = support.to(device=values.device, dtype=values.dtype)
    query = query.to(device=values.device, dtype=values.dtype)
    idx = torch.searchsorted(support, query, right=False)
    idx = idx.clamp(1, support.numel() - 1)
    left_idx = idx - 1
    right_idx = idx
    x0 = support[left_idx]
    x1 = support[right_idx]
    y0 = values.gather(1, left_idx.unsqueeze(-1)).squeeze(-1)
    y1 = values.gather(1, right_idx.unsqueeze(-1)).squeeze(-1)
    weight = ((query - x0) / (x1 - x0).clamp_min(1.0e-8)).clamp(0.0, 1.0)
    return y0 + weight * (y1 - y0)


def _pit_from_samples(
    samples: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    samples_t = convert_to_tensor(samples)
    y_true_t = convert_to_tensor(y_true)
    if samples_t.ndim == 3 and samples_t.shape[-1] == 1:
        samples_t = samples_t.squeeze(-1)
    if y_true_t.ndim == 2 and y_true_t.shape[-1] == 1:
        y_true_t = y_true_t.squeeze(-1)
    if samples_t.ndim != 2 or y_true_t.ndim != 1:
        raise ValueError("PIT from samples currently supports scalar targets only.")
    return (samples_t <= y_true_t.unsqueeze(0)).to(torch.float32).mean(dim=0)


def _pit_from_density(
    support: Union[torch.Tensor, np.ndarray],
    density: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    support_t = convert_to_tensor(support).flatten()
    density_t = convert_to_tensor(density)
    y_true_t = convert_to_tensor(y_true).flatten()
    if density_t.ndim != 2:
        raise ValueError("density must have shape [batch, support]")
    if density_t.shape[1] != support_t.numel():
        raise ValueError("density support mismatch")
    support_t = support_t.to(device=density_t.device, dtype=density_t.dtype)
    if not torch.all(support_t[1:] > support_t[:-1]):
        raise ValueError("support must be strictly increasing")
    cdf_grid = _cdf_from_density(support_t, density_t)
    return _interp1d(support_t, cdf_grid, y_true_t).clamp(0.0, 1.0)


def _pit_from_quantiles(
    y_pred_quantiles: Dict[float, Union[torch.Tensor, np.ndarray]],
    y_true: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    quantile_levels = sorted(y_pred_quantiles.keys())
    if len(quantile_levels) < 2:
        raise ValueError("At least two quantile levels are required for PIT from quantiles.")

    quantile_values = [
        convert_to_tensor(y_pred_quantiles[level]).reshape(-1).to(torch.float32)
        for level in quantile_levels
    ]
    quantile_matrix = torch.stack(quantile_values, dim=1)
    # ponytail: cummax enforces monotonicity but silently collapses crossed
    # quantiles into flat plateaus, making PIT interpolation inaccurate for
    # any target above the crossing point. We now warn when this repair fires.
    n_rows = quantile_matrix.shape[0]
    # Rows whose quantiles are not already non-decreasing get repaired by cummax.
    n_crossed = int((torch.diff(quantile_matrix, dim=1) < 0).any(dim=1).sum().item())
    if n_crossed > 0:
        warnings.warn(
            f"PIT: quantile crossings detected and repaired in {n_crossed}/{n_rows} rows",
            RuntimeWarning,
        )
    y_true_t = convert_to_tensor(y_true).reshape(-1).to(torch.float32)
    if quantile_matrix.shape[0] != y_true_t.shape[0]:
        raise ValueError("Quantile predictions and targets must have matching batch size.")
    quantile_matrix = torch.cummax(quantile_matrix, dim=1).values

    level_tensor = torch.tensor(quantile_levels, device=quantile_matrix.device, dtype=torch.float32)

    y_expanded = y_true_t.unsqueeze(1)

    # Find insertion index for each element
    upper_idx = torch.searchsorted(quantile_matrix, y_expanded, right=False).squeeze(1)

    # Clamp to valid range [1, num_levels - 1] to get lower and upper bounds
    num_levels = len(quantile_levels)
    clamped_upper_idx = torch.clamp(upper_idx, 1, num_levels - 1)
    lower_idx = clamped_upper_idx - 1

    # Gather bounding quantiles and their corresponding levels
    q0 = torch.gather(quantile_matrix, 1, lower_idx.unsqueeze(1)).squeeze(1)
    q1 = torch.gather(quantile_matrix, 1, clamped_upper_idx.unsqueeze(1)).squeeze(1)
    p0 = level_tensor[lower_idx]
    p1 = level_tensor[clamped_upper_idx]

    # Compute interpolation
    weight = ((y_true_t - q0) / (q1 - q0).clamp_min(1.0e-8)).clamp(0.0, 1.0)
    pit = p0 + weight * (p1 - p0)

    # Handle out-of-bounds: Below lowest quantile
    q_min = quantile_matrix[:, 0]
    is_below = y_true_t <= q_min
    pit_below = torch.where(torch.isclose(y_true_t, q_min), level_tensor[0], torch.zeros_like(pit))
    pit = torch.where(is_below, pit_below, pit)

    # Handle out-of-bounds: Above highest quantile
    q_max = quantile_matrix[:, -1]
    is_above = y_true_t >= q_max
    pit_above = torch.where(torch.isclose(y_true_t, q_max), level_tensor[-1], torch.ones_like(pit))
    pit = torch.where(is_above, pit_above, pit)

    return pit.clamp(0.0, 1.0)


def _cdf_from_density(support: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
    trapezoids = (
        0.5 * (density[:, 1:] + density[:, :-1]) * (support[1:] - support[:-1]).unsqueeze(0)
    )
    cdf_grid = torch.cat(
        [
            torch.zeros(density.shape[0], 1, device=density.device, dtype=density.dtype),
            torch.cumsum(trapezoids, dim=-1),
        ],
        dim=-1,
    )
    return cdf_grid / cdf_grid[:, -1:].clamp_min(1.0e-8)


def _quantiles_from_density(
    support: Union[torch.Tensor, np.ndarray],
    density: Union[torch.Tensor, np.ndarray],
    probs: list[float],
) -> torch.Tensor:
    support_t = convert_to_tensor(support).flatten()
    density_t = convert_to_tensor(density)
    if density_t.ndim != 2:
        raise ValueError("density must have shape [batch, support]")
    if not torch.all(support_t[1:] > support_t[:-1]):
        raise ValueError("support must be strictly increasing")
    support_t = support_t.to(device=density_t.device, dtype=density_t.dtype)
    cdf_grid = _cdf_from_density(support_t, density_t)
    prob_t = torch.tensor(probs, device=density_t.device, dtype=density_t.dtype)
    batch_size = density_t.shape[0]
    probs_expanded = prob_t.unsqueeze(0).expand(batch_size, -1).contiguous()

    idx = torch.searchsorted(cdf_grid, probs_expanded, right=False)
    idx = torch.clamp(idx, 1, support_t.numel() - 1)

    left = idx - 1
    right = idx

    c0 = torch.gather(cdf_grid, 1, left)
    c1 = torch.gather(cdf_grid, 1, right)

    s0 = support_t[left]
    s1 = support_t[right]

    weight = ((probs_expanded - c0) / (c1 - c0).clamp_min(1.0e-8)).clamp(0.0, 1.0)
    quantiles = s0 + weight * (s1 - s0)

    return quantiles


def continuous_ranked_probability_score(
    y_pred_quantiles: Dict[float, Union[torch.Tensor, np.ndarray]],
    y_true: Union[torch.Tensor, np.ndarray],
    reduction: str = "mean",
) -> Union[torch.Tensor, float]:
    """
    Functional CRPS using quantile predictions.
    """
    y_true_t = convert_to_tensor(y_true)
    quantiles = sorted(y_pred_quantiles.keys())
    if len(quantiles) < 2:
        raise ValueError("At least 2 quantile levels are required for CRPS calculation")

    # Vectorized validation
    if torch.isnan(y_true_t).any() or torch.isinf(y_true_t).any():
        raise ValueError("y_true contains NaN or infinite values")

    forecasts = [convert_to_tensor(y_pred_quantiles[q]) for q in quantiles]
    forecast_tensor = torch.stack(forecasts)

    if torch.isnan(forecast_tensor).any() or torch.isinf(forecast_tensor).any():
        raise ValueError("y_pred contains NaN or infinite values")

    # Check shapes using first quantile
    y_pred_sample = forecasts[0]
    if y_pred_sample.dim() == 0 or y_true_t.dim() == 0:
        raise ValueError("Inputs cannot be scalars, must have at least one dimension")

    if y_pred_sample.shape[0] != y_true_t.shape[0]:
        raise ValueError(
            f"y_pred and y_true must have same batch size. "
            f"Got y_pred: {y_pred_sample.shape}, y_true: {y_true_t.shape}"
        )

    if y_pred_sample.shape != y_true_t.shape:
        try:
            _ = y_pred_sample + y_true_t
        except RuntimeError:
            raise ValueError(
                f"y_pred shape {y_pred_sample.shape} and y_true shape {y_true_t.shape} "
                "are not compatible"
            )
    quantile_tensor = torch.tensor(quantiles, device=forecast_tensor.device)

    zero_tensor = torch.tensor([0.0], device=quantile_tensor.device)
    one_tensor = torch.tensor([1.0], device=quantile_tensor.device)
    weights = torch.diff(torch.cat([zero_tensor, quantile_tensor, one_tensor]))

    diff = y_true_t.unsqueeze(0) - forecast_tensor
    view_shape = (-1,) + (1,) * y_true_t.ndim
    q_tensor = quantile_tensor.view(*view_shape)
    quantile_loss = torch.max(q_tensor * diff, (q_tensor - 1) * diff)
    # ponytail: trapezoidal rule for CRPS = 2∫QL dτ ≈ 2 Σ_i w_i QL_i,
    # w_i = (τ_{i+1} - τ_{i-1}) / 2 (TR-MET-01: the factor 2 was missing)
    weights_tensor = (0.5 * (weights[:-1] + weights[1:])).view(*view_shape)
    crps_values = 2.0 * torch.sum(weights_tensor * quantile_loss, dim=0)

    if reduction == "none":
        return crps_values
    if reduction == "sum":
        return float(torch.sum(crps_values).item())
    return float(torch.mean(crps_values).item())


def crps_from_samples(
    y_samples: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    reduction: str = "mean",
) -> Union[torch.Tensor, float]:
    """Empirical CRPS from predictive samples for univariate regression.

    Uses the standard sample approximation:

    ``CRPS = E|X - y| - 0.5 E|X - X'|``
    """
    y_true_t = convert_to_tensor(y_true)
    samples_t = convert_to_tensor(y_samples)
    if samples_t.dim() == y_true_t.dim():
        samples_t = samples_t.unsqueeze(0)
    if samples_t.dim() != y_true_t.dim() + 1:
        raise ValueError(
            "y_samples must have shape [n_samples, batch, ...] matching y_true shape [batch, ...]"
        )
    if samples_t.shape[1:] != y_true_t.shape:
        raise ValueError(
            f"y_samples trailing shape {tuple(samples_t.shape[1:])} must match y_true "
            f"shape {tuple(y_true_t.shape)}"
        )

    term1 = torch.mean(torch.abs(samples_t - y_true_t.unsqueeze(0)), dim=0)

    # Sort samples along sample dimension
    sorted_samples, _ = torch.sort(samples_t, dim=0)
    n = samples_t.shape[0]

    # Vectorized computation of E|X - X'| using L-moments approach
    j = torch.arange(n, device=samples_t.device, dtype=samples_t.dtype)
    weights = (2 * j - n + 1).view(n, *([1] * (samples_t.dim() - 1)))

    # ponytail: unbiased E|X-X'| estimator divides by n(n-1), not n²
    term2 = torch.sum(weights * sorted_samples, dim=0) / (n * (n - 1))

    crps = term1 - term2

    if reduction == "none":
        return crps
    if reduction == "sum":
        return float(torch.sum(crps).item())
    return float(torch.mean(crps).item())


def energy_score(
    y_samples: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    beta: float = 1.0,
    max_pairs: Optional[int] = None,
    reduction: str = "mean",
) -> Union[torch.Tensor, float]:
    """
    Functional energy score for multivariate probabilistic forecasts.
    """
    y_true_t = convert_to_tensor(y_true)
    y_samples_t = convert_to_tensor(y_samples)

    n_samples = y_samples_t.shape[0]

    if max_pairs is not None and n_samples > max_pairs:
        indices = torch.randperm(n_samples, device=y_samples_t.device)[:max_pairs]
        y_samples_t = y_samples_t[indices]
        n_samples = max_pairs

    diff = y_samples_t - y_true_t.unsqueeze(0)

    # TR-MET-10: both terms must use the same geometry —
    # term1 = E||X - y||^beta (Euclidean norm to the beta), term2 = ||X - X'||^beta.
    norms = (diff.abs().pow(2).sum(-1)).pow(beta / 2.0)

    term1 = torch.mean(norms, dim=0)

    y_samples_p = y_samples_t.permute(1, 0, 2)
    dists = torch.cdist(y_samples_p, y_samples_p, p=2)

    if beta != 1.0:
        dists = torch.pow(dists, beta)

    term2 = torch.sum(dists, dim=(1, 2))
    n_pairs = n_samples * (n_samples - 1) // 2

    if n_pairs > 0:
        term2 = term2 / (4.0 * n_pairs)
    else:
        term2 = torch.zeros_like(term2)
    scores = term1 - term2

    if reduction == "none":
        return scores
    if reduction == "sum":
        return float(torch.sum(scores).item())
    return float(torch.mean(scores).item())


def _process_distribution_metrics(
    dist_obj: torch.distributions.Distribution,
    y_true_t: torch.Tensor,
    samples: Optional[Union[torch.Tensor, np.ndarray]],
    n_samples: int,
    results: Dict[str, Union[torch.Tensor, float, np.ndarray]],
) -> Tuple[Optional[Union[torch.Tensor, np.ndarray]], Any]:
    """Process distribution-based metrics (log-prob, samples, PIT)."""
    pit_values = None

    # Log-probability
    try:
        log_prob = dist_obj.log_prob(y_true_t)
    except RuntimeError:
        if y_true_t.ndim == 2 and y_true_t.shape[-1] == 1:
            log_prob = dist_obj.log_prob(y_true_t.squeeze(-1))
        elif y_true_t.ndim == 1:
            log_prob = dist_obj.log_prob(y_true_t.unsqueeze(-1))
        else:
            raise

    if log_prob.dim() > 1:
        event_dims = list(range(1, log_prob.dim()))
        log_prob = log_prob.sum(dim=event_dims)
    results["log_prob"] = float(torch.mean(log_prob).item())

    # Draw samples
    if samples is None:
        samples = dist_obj.sample((n_samples,))

    # PIT from distribution
    try:
        pit_res = probability_integral_transform(dist_obj.cdf, y_true_t, return_histogram=True)
        if isinstance(pit_res, dict):
            # cast: ty cannot narrow through the Tensor|dict union (isinstance
            # against torch.Tensor produces bogus intersections).
            hist = cast(Dict[str, Union[torch.Tensor, float]], pit_res)
            results["pit_chi2"] = hist["uniformity_chi2"]
            results["pit_ks"] = hist["uniformity_ks"]
            pit_values = hist["pit_values"]
    except (AttributeError, NotImplementedError):
        pass

    return samples, pit_values


def _process_distance_metrics(
    y_true_t: torch.Tensor,
    y_pred_quantiles: Optional[Dict[float, Union[torch.Tensor, np.ndarray]]],
    samples: Optional[Union[torch.Tensor, np.ndarray]],
    results: Dict[str, Union[torch.Tensor, float, np.ndarray]],
) -> Optional[Dict[float, Union[torch.Tensor, np.ndarray]]]:
    """Process distance-based metrics (CRPS, Energy, Coverage)."""
    if y_pred_quantiles is None and samples is not None:
        q_levels = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
        samples_t = convert_to_tensor(samples)
        y_pred_quantiles = {q: torch.quantile(samples_t, q, dim=0) for q in q_levels}

    if y_pred_quantiles is not None:
        results["crps"] = continuous_ranked_probability_score(
            y_pred_quantiles, y_true_t, reduction="mean"
        )
        if 0.05 in y_pred_quantiles and 0.95 in y_pred_quantiles:
            q05 = convert_to_tensor(y_pred_quantiles[0.05])
            q95 = convert_to_tensor(y_pred_quantiles[0.95])
            within = (y_true_t >= q05) & (y_true_t <= q95)
            results["coverage_90"] = float(within.to(torch.float32).mean().item())
            results["interval_width_90"] = float((q95 - q05).mean().item())

    elif samples is not None and (y_true_t.dim() == 1 or y_true_t.shape[-1] == 1):
        results["crps"] = crps_from_samples(samples, y_true_t, reduction="mean")

    if samples is not None and y_true_t.dim() > 1:
        results["energy_score"] = energy_score(samples, y_true_t, reduction="mean")

    return y_pred_quantiles


def _process_density_metrics(
    support: Union[torch.Tensor, np.ndarray],
    density: Union[torch.Tensor, np.ndarray],
    y_true_t: torch.Tensor,
    results: Dict[str, Union[torch.Tensor, float, np.ndarray]],
) -> None:
    """Process density-based metrics (CDE loss, coverage from density)."""
    support_t = convert_to_tensor(support).flatten()
    density_t = convert_to_tensor(density)
    results["cde_loss"] = conditional_density_estimation_loss(
        support_t, density_t, y_true_t, reduction="mean"
    )
    if "log_prob" not in results:
        density_at_y = _interp1d(
            support_t.to(device=density_t.device, dtype=density_t.dtype),
            density_t,
            y_true_t.reshape(-1).to(device=density_t.device, dtype=density_t.dtype),
        )
        results["log_prob"] = float(torch.log(density_at_y.clamp_min(1.0e-8)).mean().item())
    if "coverage_90" not in results and (y_true_t.dim() == 1 or y_true_t.shape[-1] == 1):
        q05_q95 = _quantiles_from_density(support_t, density_t, [0.05, 0.95])
        lower = q05_q95[:, 0]
        upper = q05_q95[:, 1]
        y_flat = y_true_t.reshape(-1).to(device=lower.device, dtype=lower.dtype)
        within = (y_flat >= lower) & (y_flat <= upper)
        results["coverage_90"] = float(within.to(torch.float32).mean().item())
        results["interval_width_90"] = float((upper - lower).mean().item())


def _process_fallback_pit(
    pit_values: Optional[torch.Tensor],
    support: Optional[Union[torch.Tensor, np.ndarray]],
    density: Optional[Union[torch.Tensor, np.ndarray]],
    samples: Optional[Union[torch.Tensor, np.ndarray]],
    y_pred_quantiles: Optional[Dict[float, Union[torch.Tensor, np.ndarray]]],
    y_true_t: torch.Tensor,
    results: Dict[str, Union[torch.Tensor, float, np.ndarray]],
) -> None:
    """Compute PIT values and uniformity statistics if not already computed."""
    if pit_values is None:
        try:
            if support is not None and density is not None:
                pit_values = _pit_from_density(support, density, y_true_t)
            elif samples is not None and (y_true_t.dim() == 1 or y_true_t.shape[-1] == 1):
                pit_values = _pit_from_samples(samples, y_true_t)
            elif y_pred_quantiles is not None and (y_true_t.dim() == 1 or y_true_t.shape[-1] == 1):
                pit_values = _pit_from_quantiles(y_pred_quantiles, y_true_t)
        except ValueError:
            pit_values = None

    if pit_values is not None and "pit_chi2" not in results:
        pit_res = probability_integral_transform(
            lambda _: pit_values,
            pit_values,
            return_histogram=True,
        )
        if isinstance(pit_res, dict):
            # cast: ty cannot narrow through the Tensor|dict union.
            hist = cast(Dict[str, Union[torch.Tensor, float]], pit_res)
            results["pit_chi2"] = hist["uniformity_chi2"]
            results["pit_ks"] = hist["uniformity_ks"]


def distribution_metrics_report(
    dist: Optional[Union[torch.distributions.Distribution, Dict[str, Any]]] = None,
    y_true: Optional[Union[torch.Tensor, np.ndarray]] = None,
    y_pred_quantiles: Optional[Dict[float, Union[torch.Tensor, np.ndarray]]] = None,
    samples: Optional[Union[torch.Tensor, np.ndarray]] = None,
    support: Optional[Union[torch.Tensor, np.ndarray]] = None,
    density: Optional[Union[torch.Tensor, np.ndarray]] = None,
    n_samples: int = 100,
) -> Dict[str, Union[torch.Tensor, float, np.ndarray]]:
    """
    Generate a universal distribution metrics report for probabilistic regression.

    This helper consolidates density estimation (NLL, CDE), distance-based (CRPS, Energy),
    and calibration (PIT, Coverage) metrics into a single dictionary.

    Args:
        dist: Optional torch Distribution or dict with 'loc'/'scale' keys.
        y_true: Target values.
        y_pred_quantiles: Optional dict mapping quantile levels to predictions.
        samples: Optional predictive samples [n_samples, batch, ...].
        support: Optional 1D support grid for density-based metrics.
        density: Optional predictive densities [batch, support].
        n_samples: Number of samples to draw from `dist` if `samples` is None.

    Returns:
        Dict of results:
            - log_prob: Mean log-likelihood (if dist is available)
            - crps: Continuous Ranked Probability Score
            - energy_score: Energy Score (for multivariate targets)
            - pit_chi2: PIT uniformity chi-square statistic
            - pit_ks: PIT uniformity Kolmogorov-Smirnov statistic
            - cde_loss: Conditional Density Estimation loss (if support/density available)
            - coverage_90: Fraction of targets within [q0.05, q0.95]
            - interval_width_90: Mean width of the [q0.05, q0.95] interval
    """
    results: Dict[str, Union[torch.Tensor, float, np.ndarray]] = {}
    if y_true is None:
        raise ValueError("y_true must be provided either as a Tensor or ndarray.")
    y_true_t = convert_to_tensor(y_true)
    pit_values: Optional[torch.Tensor] = None

    # 1. Log-likelihood and Sampling
    if dist is not None:
        dist_obj: torch.distributions.Distribution
        if isinstance(dist, dict):
            loc_val: Any = dist.get("loc")
            if loc_val is None:
                loc_val = dist.get("mean")
            scale_val: Any = dist.get("scale")
            if scale_val is None:
                scale_val = dist.get("std")
            if loc_val is None or scale_val is None:
                raise ValueError("dist dict must provide loc/mean and scale/std")
            loc = convert_to_tensor(cast(Any, loc_val))
            scale = convert_to_tensor(cast(Any, scale_val))
            dist_obj = torch.distributions.Normal(loc, scale)
        else:
            dist_obj = dist

        samples, pit_values = _process_distribution_metrics(
            dist_obj, y_true_t, samples, n_samples, results
        )

    # 2. Distance-based metrics (CRPS / Energy)
    y_pred_quantiles = _process_distance_metrics(y_true_t, y_pred_quantiles, samples, results)

    # 3. Density-based metrics (CDE loss)
    if support is not None and density is not None:
        _process_density_metrics(support, density, y_true_t, results)

    # 4. PIT fallback for non-cdf predictive representations
    _process_fallback_pit(
        pit_values, support, density, samples, y_pred_quantiles, y_true_t, results
    )

    return results


class DawidSebastianiScore(Metric):
    """
    Dawid-Sebastiani score for diagonal-Gaussian predictive distributions (§6 F1).

    ``DS = (y - mu)^2 / sigma^2 + 2 * log(sigma)``
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("ds_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, y_pred_mean: torch.Tensor, y_pred_std: torch.Tensor, y_true: torch.Tensor
    ) -> None:
        """Update state with predictive mean/std and targets."""
        mean = convert_to_tensor(y_pred_mean)
        std = convert_to_tensor(y_pred_std)
        y = convert_to_tensor(y_true)
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            raise ValueError("y_pred_mean contains NaN or infinite values")
        if torch.isnan(std).any() or torch.isinf(std).any() or bool((std <= 0).any()):
            raise ValueError("y_pred_std must be finite and strictly positive")
        var = std.pow(2).clamp_min(1.0e-12)
        ds = (y - mean) ** 2 / var + 2.0 * torch.log(std.clamp_min(1.0e-12))
        metric_state_tensor(self.ds_sum).add_(torch.sum(ds))
        metric_state_tensor(self.total).add_(torch.as_tensor(y.numel(), device=y.device))

    def compute(self) -> torch.Tensor:
        """Compute mean DS."""
        return metric_state_tensor(self.ds_sum) / metric_state_tensor(self.total)


def dss_score(
    y_pred_mean: Union[torch.Tensor, np.ndarray],
    y_pred_std: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Functional Dawid-Sebastiani score: ``(y - mu)^2 / sigma^2 + 2 * ln(sigma)``.
    """
    metric = DawidSebastianiScore()
    metric.update(
        convert_to_tensor(y_pred_mean),  # ty: ignore[invalid-argument-type]  # torchmetrics update/compute overrides confuse ty
        convert_to_tensor(y_pred_std),
        convert_to_tensor(y_true),
    )
    return float(metric.compute().item())  # ty: ignore[missing-argument]  # torchmetrics update/compute overrides confuse ty


class VarioScore(Metric):
    """
    Vario-score of Zamo & Naveau for ensemble forecasts (§6 F1).

    ``nu_rho = 0.5 * E|X - X'|^rho - E|X - y|^rho``; fair, strictly proper for
    the CRPS family when rho in (0, 2].
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, rho: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not 0 < rho <= 2:
            raise ValueError(f"rho must lie in (0, 2], got {rho}")
        self.rho = rho
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_samples: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictive samples [n_samples, batch] and targets."""
        samples = convert_to_tensor(y_samples)
        y = convert_to_tensor(y_true)
        # _gini_mean_abs_diff returns the unbiased half-Gini estimator of
        # E|X - X'|^rho (same trick as crps_from_samples), i.e. the first term.
        term1 = _gini_mean_abs_diff(samples, rho=self.rho)
        term2 = torch.mean((samples - y.unsqueeze(0)).abs().pow(self.rho), dim=0)
        scores = term1 - term2
        metric_state_tensor(self.score_sum).add_(torch.sum(scores))
        metric_state_tensor(self.total).add_(torch.as_tensor(y.numel(), device=y.device))

    def compute(self) -> torch.Tensor:
        """Compute mean vario score."""
        return metric_state_tensor(self.score_sum) / metric_state_tensor(self.total)


def _gini_mean_abs_diff(samples: torch.Tensor, rho: float) -> torch.Tensor:
    """Unbiased ``E|X - X'|^rho / 2`` — the first vario-score term.

    For ``rho == 1`` this uses the same sorted-rank (Gini/L-moment) trick as
    :func:`crps_from_samples`; other exponents use exact pairwise distances.
    """
    n = samples.shape[0]
    if rho == 1.0:
        j = torch.arange(n, device=samples.device, dtype=samples.dtype)
        weights = (2 * j - n + 1).view(n, *([1] * (samples.dim() - 1)))
        sorted_samples, _ = torch.sort(samples, dim=0)
        return torch.sum(weights * sorted_samples, dim=0) / (n * (n - 1))
    flat = samples.reshape(n, -1)
    diffs = (flat.unsqueeze(1) - flat.unsqueeze(0)).abs().pow(rho)  # [n, n, D]
    off_diag = diffs.sum(dim=(0, 1)) - torch.diagonal(diffs, dim1=0, dim2=1).sum(dim=-1)
    half_mean = off_diag / (n * (n - 1)) / 2.0
    return half_mean.reshape(samples.shape[1:])


def vario_score(
    y_samples: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    rho: float = 1.0,
) -> float:
    """
    Functional vario score: ``0.5 * E|X - X'|^rho - E|X - y|^rho``.
    """
    metric = VarioScore(rho=rho)
    metric.update(convert_to_tensor(y_samples), convert_to_tensor(y_true))  # ty: ignore[invalid-argument-type]  # torchmetrics update/compute overrides confuse ty
    return float(metric.compute().item())  # ty: ignore[missing-argument]  # torchmetrics update/compute overrides confuse ty


def pinball_loss(level: float, quantile_value: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Pinball (quantile) loss at a single level."""
    diff = y_true - quantile_value
    return torch.maximum(level * diff, (level - 1.0) * diff)


class PinballMetric(Metric):
    """
    Mean pinball loss over quantile levels — the CRPS integrand standalone (§6 F1).
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred_quantiles: Dict[float, torch.Tensor], y_true: torch.Tensor) -> None:
        """Update state with {level: predictions} and targets."""
        levels = sorted(y_pred_quantiles.keys())
        if len(levels) < 1:
            raise ValueError("At least one quantile level is required")
        y = convert_to_tensor(y_true)
        total = torch.zeros_like(y)
        for level in levels:
            q = convert_to_tensor(y_pred_quantiles[level])
            total = total + pinball_loss(float(level), q, y)
        mean_pinball = total / len(levels)
        metric_state_tensor(self.loss_sum).add_(torch.sum(mean_pinball))
        metric_state_tensor(self.total).add_(torch.as_tensor(y.numel(), device=y.device))

    def compute(self) -> torch.Tensor:
        """Compute mean pinball."""
        return metric_state_tensor(self.loss_sum) / metric_state_tensor(self.total)


def pinball_metric(
    quantiles_dict: Dict[float, Union[torch.Tensor, np.ndarray]],
    y_true: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Functional mean pinball across quantile levels.
    """
    metric = PinballMetric()
    metric.update(
        {level: convert_to_tensor(q) for level, q in quantiles_dict.items()},  # ty: ignore[invalid-argument-type]  # torchmetrics update/compute overrides confuse ty
        convert_to_tensor(y_true),
    )
    return float(metric.compute().item())  # ty: ignore[missing-argument]  # torchmetrics update/compute overrides confuse ty


def wasserstein_gaussian_p2(
    loc1: Union[torch.Tensor, np.ndarray, float],
    scale1: Union[torch.Tensor, np.ndarray, float],
    loc2: Union[torch.Tensor, np.ndarray, float],
    scale2: Union[torch.Tensor, np.ndarray, float],
) -> float:
    """Exact 2-Wasserstein distance between two Gaussians (§6 F1).

    Scales may be per-dimension standard deviations (diag form) or full
    covariance matrices; the cross term uses :func:`torch.linalg.eigh`.
    """
    m1 = convert_to_tensor(loc1).reshape(-1).to(torch.float64)
    m2 = convert_to_tensor(loc2).reshape(-1).to(torch.float64)
    s1 = convert_to_tensor(scale1).to(torch.float64)
    s2 = convert_to_tensor(scale2).to(torch.float64)
    if m1.shape != m2.shape:
        raise ValueError("loc1 and loc2 must have matching shapes")

    if s1.dim() == 0 or (s1.dim() == 1 and s1.numel() == m1.numel()):
        # Diagonal (independent-dims) closed form.
        sd1 = s1.reshape(-1)
        sd2 = s2.reshape(-1)
        w2_sq = torch.sum((m1 - m2) ** 2 + (sd1 - sd2) ** 2)
        return float(torch.sqrt(w2_sq.clamp_min(0.0)).item())

    # Full-covariance form: W2^2 = ||m1-m2||^2 + tr(S1 + S2 - 2 (S1^{1/2} S2 S1^{1/2})^{1/2})
    d = m1.numel()
    cov1 = s1.reshape(d, d) if s1.dim() != 2 else s1
    cov2 = s2.reshape(d, d) if s2.dim() != 2 else s2
    ev1, evec1 = torch.linalg.eigh(cov1)
    root1 = evec1 @ torch.diag(ev1.clamp_min(0.0).sqrt()) @ evec1.T
    cross = root1 @ cov2 @ root1
    ev_c, _ = torch.linalg.eigh(cross)
    trace_cross = ev_c.clamp_min(0.0).sqrt().sum()
    w2_sq = torch.sum((m1 - m2) ** 2) + torch.trace(cov1 + cov2) - 2.0 * trace_cross
    return float(torch.sqrt(w2_sq.clamp_min(0.0)).item())


class WassersteinGaussian(Metric):
    """
    Mean exact 2-Wasserstein distance between predicted Gaussians and targets.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("dist_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        y_true: torch.Tensor,
    ) -> None:
        """Update with predicted mean/std (diag) and observed targets.

        W2 between N(mean, diag(std^2)) and the Dirac delta at the target is
        sqrt((mean - y)^2 + std^2).
        """
        mean = convert_to_tensor(loc)
        std = convert_to_tensor(scale).clamp_min(1.0e-12)
        y = convert_to_tensor(y_true)
        dists = torch.sqrt((mean - y) ** 2 + std.pow(2))
        metric_state_tensor(self.dist_sum).add_(torch.sum(dists))
        metric_state_tensor(self.total).add_(torch.as_tensor(y.numel(), device=y.device))

    def compute(self) -> torch.Tensor:
        """Compute mean W2 distance to a Dirac at the target."""
        return metric_state_tensor(self.dist_sum) / metric_state_tensor(self.total)
