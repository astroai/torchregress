"""
Distribution metrics for evaluating probabilistic regression models.
"""

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.distributions import Distribution

from torchregress.metrics.utils import apply_reduction, convert_to_tensor, validate_inputs
from torchregress.utils.histogram import histogram_bins


def probability_integral_transform(
    cdf_fn: Callable,
    y_true: Union[torch.Tensor, np.ndarray],
    n_bins: int = 20,
    return_histogram: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Calculate Probability Integral Transform (PIT) values and optionally their histogram.

    The PIT is the value of the predictive CDF at the observed value.
    For a well-calibrated forecast, PIT values should be uniformly distributed.

    Args:
        cdf_fn: Function that takes y_true and returns CDF values at those points
        y_true: Ground truth values
        n_bins: Number of bins for PIT histogram
        return_histogram: Whether to return histogram counts and edges

    Returns:
        PIT values or dictionary with PIT values and histogram
    """
    y_true_tensor = convert_to_tensor(y_true)
    pit_values = cdf_fn(y_true_tensor)

    # Optionally calculate histogram for uniformity assessment
    if return_histogram:
        counts, bin_edges = histogram_bins(pit_values, n_bins, range=(0, 1))
        bin_counts = counts
        # Normalize histogram
        normalized_counts = bin_counts / torch.sum(bin_counts) * n_bins

        return {
            "pit_values": pit_values,
            "histogram_counts": normalized_counts,
            "bin_edges": bin_edges,
            "uniformity_chi2": torch.sum((normalized_counts - 1.0) ** 2),
        }

    return pit_values


def continuous_ranked_probability_score(
    y_pred_quantiles: Dict[float, Union[torch.Tensor, np.ndarray]],
    y_true: Union[torch.Tensor, np.ndarray],
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calculate Continuous Ranked Probability Score (CRPS) for probabilistic forecasts.

    CRPS is a proper scoring rule that measures the quality of probabilistic forecasts.
    Lower values indicate better performance.

    Args:
        y_pred_quantiles: Dictionary mapping quantile levels to predictions
            each with shape [n_samples]
        y_true: Ground truth values [n_samples]
        reduction: How to reduce the score ("none", "mean", "sum")

    Returns:
        CRPS score (reduced as specified)
    """
    y_true_tensor = convert_to_tensor(y_true)

    # Sort quantiles and extract predictions
    quantiles = sorted(y_pred_quantiles.keys())

    # Validate that we have at least 2 quantiles
    if len(quantiles) < 2:
        raise ValueError("At least 2 quantile levels are required for CRPS calculation")

    forecasts = []

    for q in quantiles:
        q_pred = convert_to_tensor(y_pred_quantiles[q])
        validate_inputs(q_pred, y_true_tensor)
        forecasts.append(q_pred)

    # Stack forecasts along a new dimension [n_quantiles, n_samples]
    forecast_tensor = torch.stack(forecasts)
    quantile_tensor = torch.tensor(quantiles, device=forecast_tensor.device)

    # Calculate weights (differences between consecutive quantiles)
    # Using torch.cat for better device handling
    zero_tensor = torch.tensor([0.0], device=quantile_tensor.device)
    one_tensor = torch.tensor([1.0], device=quantile_tensor.device)
    weights = torch.diff(torch.cat([zero_tensor, quantile_tensor, one_tensor]))

    # Calculate quantile loss for each level
    crps_values = torch.zeros_like(y_true_tensor, device=y_true_tensor.device)
    for i, q in enumerate(quantiles):
        diff = y_true_tensor - forecast_tensor[i]
        q_tensor = torch.tensor(q, device=diff.device)
        quantile_loss = torch.max(q_tensor * diff, (q_tensor - 1) * diff)
        crps_values = crps_values + weights[i + 1] * quantile_loss

    # Apply reduction
    return apply_reduction(crps_values, reduction)


def energy_score(
    y_samples: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    beta: float = 1.0,
    reduction: str = "mean",
    max_pairs: Optional[int] = None,
) -> torch.Tensor:
    """
    Calculate Energy Score for multivariate probabilistic forecasts.

    Energy Score is a multivariate generalization of the CRPS.

    Args:
        y_samples: Ensemble of forecast samples [n_samples, batch_size, n_dims]
        y_true: Ground truth values [batch_size, n_dims]
        beta: Parameter of the energy score (typically 1.0 or 0.5)
        reduction: How to reduce the score ("none", "mean", "sum")
        max_pairs: Maximum number of sample pairs to use (for better scalability)

    Returns:
        Energy Score (reduced as specified)
    """
    y_true = convert_to_tensor(y_true)
    y_samples = convert_to_tensor(y_samples)

    n_samples = y_samples.shape[0]
    batch_size = y_true.shape[0]

    # For very large sample counts, limit computation using random sampling
    if max_pairs is not None and n_samples > max_pairs:
        # Randomly select samples to use
        indices = torch.randperm(n_samples)[:max_pairs]
        y_samples = y_samples[indices]
        n_samples = max_pairs

    # Term 1: Expected distance between forecasts and observations
    norms = torch.zeros(batch_size, n_samples, device=y_true.device)
    for i in range(n_samples):
        # More numerically stable norm calculation, especially with beta!=1
        diff = y_samples[i] - y_true
        # Use safe power function to avoid numerical issues with negative numbers
        if beta == 1.0:
            norms[:, i] = torch.norm(diff, dim=1)
        elif beta == 0.5:
            norms[:, i] = torch.sqrt(torch.sum(torch.abs(diff), dim=1))
        else:
            norms[:, i] = torch.pow(torch.sum(torch.pow(torch.abs(diff), beta), dim=1), 1 / beta)

    term1 = torch.mean(norms, dim=1)  # [batch_size]

    # Term 2: Expected distance between pairs of forecasts (more efficient)
    # Compute pairwise distances using broadcasting
    term2 = torch.zeros(batch_size, device=y_true.device)
    n_pairs = 0

    # Calculate pairwise distances more efficiently using batch operations
    for i in range(n_samples):
        # Extract the current sample [batch_size, n_dims]
        sample_i = y_samples[i]

        # Calculate distances to all subsequent samples
        for j in range(i + 1, n_samples):
            sample_j = y_samples[j]
            term2 += torch.norm(sample_i - sample_j, dim=1) ** beta
            n_pairs += 1

    term2 /= 2.0 * n_pairs  # Normalize by number of pairs

    # Energy score = Term1 - Term2
    energy_scores = term1 - term2

    # Apply reduction
    return apply_reduction(energy_scores, reduction)


def distribution_metrics_report(
    distribution: Optional[Distribution],
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred_quantiles: Optional[Dict[float, Union[torch.Tensor, np.ndarray]]] = None,
    samples: Optional[Union[torch.Tensor, np.ndarray]] = None,
    quantiles_to_check: List[float] = [0.1, 0.5, 0.9],
) -> Dict[str, torch.Tensor]:
    """
    Generate a comprehensive report on distribution prediction quality.

    Args:
        distribution: PyTorch distribution object (optional)
        y_true: Ground truth values
        y_pred_quantiles: Dictionary mapping quantile levels to predictions (optional)
        samples: Samples from predictive distribution [n_samples, batch_size, ...] (optional)
        quantiles_to_check: Quantiles to evaluate for calibration

    Returns:
        Dictionary with multiple distribution metrics
    """
    y_true = convert_to_tensor(y_true)
    metrics = {}

    # Check we have at least one input
    if distribution is None and y_pred_quantiles is None and samples is None:
        raise ValueError("Must provide at least one of: distribution, y_pred_quantiles, or samples")

    # If we have a distribution, compute likelihood metrics
    if distribution is not None:
        metrics["log_prob"] = distribution.log_prob(y_true).mean()

        # Generate quantiles if not provided
        if y_pred_quantiles is None:
            y_pred_quantiles = {}
            for q in quantiles_to_check:
                y_pred_quantiles[q] = distribution.icdf(torch.tensor(q)).detach()

    # If we have samples but no distribution
    if samples is not None and distribution is None:
        samples = convert_to_tensor(samples)

        # Generate quantiles if not provided
        if y_pred_quantiles is None:
            y_pred_quantiles = {}
            for q in quantiles_to_check:
                y_pred_quantiles[q] = torch.quantile(samples, q, dim=0)

    # Calculate CRPS if we have quantiles
    if y_pred_quantiles is not None:
        metrics["crps"] = continuous_ranked_probability_score(y_pred_quantiles, y_true)

    # Calculate energy score for multivariate predictions
    if samples is not None and y_true.dim() > 1 and y_true.shape[1] > 1:
        metrics["energy_score"] = energy_score(samples, y_true)

    return metrics
