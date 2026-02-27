"""
Distribution metrics for evaluating probabilistic regression models.
"""

from typing import Any, Dict, Optional, Union, cast

import numpy as np
import torch
from torchmetrics import Metric

from torchregress.metrics.utils import convert_to_tensor, metric_state_tensor


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
        weights_tensor = weights[1:].view(*view_shape)
        crps_values = torch.sum(weights_tensor * quantile_loss, dim=0)

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

        if self.beta == 1.0:
            norms = torch.norm(diff, dim=-1)
        else:
            norms = torch.pow(
                torch.sum(torch.pow(torch.abs(diff), self.beta), dim=-1), 1 / self.beta
            )

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
) -> Union[torch.Tensor, np.ndarray, Dict[str, Union[torch.Tensor, np.ndarray, float]]]:
    """
    Compute Probability Integral Transform (PIT) values for a given CDF.
    """
    y_true_t = convert_to_tensor(y_true)
    pit_values = convert_to_tensor(cdf_fn(y_true_t))

    if not return_histogram:
        if as_numpy or isinstance(y_true, np.ndarray):
            return pit_values.cpu().numpy()
        return pit_values

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=pit_values.device)
    counts = torch.histogram(pit_values, bin_edges)[0]
    expected = pit_values.numel() / n_bins
    uniformity_chi2 = torch.sum((counts - expected) ** 2 / max(expected, 1.0))

    result: Dict[str, Union[torch.Tensor, np.ndarray, float]] = {
        "pit_values": pit_values,
        "histogram_counts": counts,
        "bin_edges": bin_edges,
        "uniformity_chi2": uniformity_chi2,
    }

    if as_numpy or isinstance(y_true, np.ndarray):
        return {
            "pit_values": pit_values.cpu().numpy(),
            "histogram_counts": counts.cpu().numpy(),
            "bin_edges": bin_edges.cpu().numpy(),
            "uniformity_chi2": float(uniformity_chi2.item()),
        }
    return result


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
    """
    Analytic CRPS for a univariate Gaussian predictive distribution.
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
    crps = std_t * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))

    if reduction == "none":
        return cast(torch.Tensor, crps)
    if reduction == "sum":
        return float(torch.sum(crps).item())
    return float(torch.mean(crps).item())


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
    weights_tensor = weights[1:].view(*view_shape)
    crps_values = torch.sum(weights_tensor * quantile_loss, dim=0)

    if reduction == "none":
        return crps_values
    if reduction == "sum":
        return float(torch.sum(crps_values).item())
    return float(torch.mean(crps_values).item())


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

    if beta == 1.0:
        norms = torch.norm(diff, dim=-1)
    else:
        norms = torch.pow(torch.sum(torch.pow(torch.abs(diff), beta), dim=-1), 1 / beta)

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


def distribution_metrics_report(
    dist: Optional[Union[torch.distributions.Distribution, Dict[str, Any]]],
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred_quantiles: Optional[Dict[float, Union[torch.Tensor, np.ndarray]]] = None,
    samples: Optional[Union[torch.Tensor, np.ndarray]] = None,
    n_samples: int = 100,
) -> Dict[str, Union[torch.Tensor, float, np.ndarray]]:
    """
    Generate distribution metrics for probabilistic regression outputs.
    """
    results: Dict[str, Union[torch.Tensor, float, np.ndarray]] = {}
    y_true_t = convert_to_tensor(y_true)

    if dist is not None:
        dist_obj: torch.distributions.Distribution
        if isinstance(dist, dict):
            loc_raw = dist.get("loc", dist.get("mean"))
            scale_raw = dist.get("scale", dist.get("std"))
            if loc_raw is None or scale_raw is None:
                raise ValueError("dist dict must provide loc/mean and scale/std")
            loc = convert_to_tensor(loc_raw)
            scale = convert_to_tensor(scale_raw)
            dist_obj = torch.distributions.Normal(loc, scale)
        else:
            dist_obj = dist

        log_prob = dist_obj.log_prob(y_true_t)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(dim=-1)
        results["log_prob"] = float(torch.mean(log_prob).item())

        if samples is None:
            samples = dist_obj.sample((n_samples,))

    if y_pred_quantiles is None and samples is not None:
        q_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        samples_t = convert_to_tensor(samples)
        y_pred_quantiles = {q: torch.quantile(samples_t, q, dim=0) for q in q_levels}

    if y_pred_quantiles is not None:
        results["crps"] = continuous_ranked_probability_score(
            y_pred_quantiles, y_true_t, reduction="mean"
        )

    if samples is not None and y_true_t.dim() > 1:
        results["energy_score"] = energy_score(samples, y_true_t, reduction="mean")

    return results
