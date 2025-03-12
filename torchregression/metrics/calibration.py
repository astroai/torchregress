"""
Calibration metrics for evaluating probabilistic regression models.
"""

import torch
import numpy as np
from typing import Union, Optional, Dict, List, Any

from torchregression.metrics.utils import convert_to_tensor, validate_inputs


def expected_calibration_error(
    y_pred_quantiles: Dict[float, Union[torch.Tensor, np.ndarray]],
    y_true: Union[torch.Tensor, np.ndarray],
    n_bins: int = 10,
    return_diagnostics: bool = False,
) -> Dict[str, Union[float, List]]:
    """
    Calculate Expected Calibration Error (ECE) for quantile regression.

    Args:
        y_pred_quantiles: Dictionary mapping quantile levels to predictions
        y_true: Ground truth values
        n_bins: Number of bins for discretizing predictions
        return_diagnostics: Whether to return additional diagnostic information

    Returns:
        Dictionary with calibration metrics
    """
    y_true = convert_to_tensor(y_true)

    quantiles = sorted(y_pred_quantiles.keys())
    expected_proportions = torch.tensor(quantiles, device=y_true.device)
    actual_proportions = []

    for q in quantiles:
        q_pred = convert_to_tensor(y_pred_quantiles[q])
        validate_inputs(q_pred, y_true)
        proportion_below = torch.mean((y_true <= q_pred).float())
        actual_proportions.append(proportion_below)

    actual_proportions = torch.stack(actual_proportions)

    # Calculate calibration errors
    abs_errors = torch.abs(actual_proportions - expected_proportions)
    mace = torch.mean(abs_errors)  # Mean Absolute Calibration Error
    rmsce = torch.sqrt(
        torch.mean((actual_proportions - expected_proportions) ** 2)
    )  # Root Mean Square Calibration Error

    # Calculate maximum calibration error
    max_ce = torch.max(abs_errors)

    result = {
        "mean_absolute_calibration_error": mace.item(),
        "root_mean_squared_calibration_error": rmsce.item(),
        "maximum_calibration_error": max_ce.item(),
    }

    if return_diagnostics:
        # Bin the errors
        bin_edges = torch.linspace(0, 1, n_bins + 1)
        bin_indices = torch.bucketize(expected_proportions, bin_edges) - 1

        bin_errors = []
        for i in range(n_bins):
            mask = bin_indices == i
            if torch.any(mask):
                bin_error = torch.mean(abs_errors[mask])
                bin_errors.append(bin_error.item())
            else:
                bin_errors.append(0.0)

        # Add diagnostic information
        result.update(
            {
                "bin_errors": bin_errors,
                "expected_proportions": expected_proportions.tolist(),
                "actual_proportions": actual_proportions.tolist(),
            }
        )

    return result


def marginal_calibration_error(
    y_pred_samples: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    n_bins: int = 20,
    return_diagnostics: bool = False,
) -> Dict[str, Union[float, List]]:
    """
    Calculate Marginal Calibration Error (MCE) for probabilistic regression.

    MCE measures how well the predictive distribution's marginals match
    the empirical distribution of observations.

    Args:
        y_pred_samples: Samples from predictive distribution [n_samples, batch_size]
        y_true: Ground truth values [batch_size]
        n_bins: Number of bins for histogram
        return_diagnostics: Whether to return additional diagnostic information

    Returns:
        Dictionary with calibration metrics
    """
    y_true = convert_to_tensor(y_true)
    y_pred_samples = convert_to_tensor(y_pred_samples)

    # Calculate empirical CDFs
    # First, determine range from combined predictions and observations
    all_values = torch.cat([y_true, y_pred_samples.view(-1)])
    min_val = torch.min(all_values)
    max_val = torch.max(all_values)

    # Add small epsilon to ensure min != max and prevent histogram issues
    if torch.isclose(min_val, max_val, rtol=1e-5):
        min_val = min_val - 1e-5
        max_val = max_val + 1e-5

    # Create bin edges - ensure we're on CPU for histogram
    bin_edges = torch.linspace(min_val, max_val, n_bins + 1, device="cpu")

    # Move tensors to CPU for histogram calculation
    y_true_cpu = y_true.cpu()

    # Calculate empirical CDF for observations
    obs_hist = torch.histogram(y_true_cpu, bin_edges)[0]
    obs_cdf = torch.cumsum(obs_hist, dim=0) / max(1, len(y_true))  # Avoid division by zero

    # Calculate empirical CDF for predictions (average across samples)
    pred_cdfs = []
    for i in range(y_pred_samples.shape[0]):  # Loop through samples
        pred_samples_cpu = y_pred_samples[i].cpu()
        pred_hist = torch.histogram(pred_samples_cpu, bin_edges)[0]
        pred_cdf = torch.cumsum(pred_hist, dim=0) / max(
            1, len(y_pred_samples[i])
        )  # Avoid division by zero
        pred_cdfs.append(pred_cdf)

    pred_cdf_mean = torch.stack(pred_cdfs).mean(dim=0)

    # Calculate calibration errors
    abs_errors = torch.abs(obs_cdf - pred_cdf_mean)
    mce = torch.mean(abs_errors)  # Mean Marginal Calibration Error
    rmsce = torch.sqrt(torch.mean((obs_cdf - pred_cdf_mean) ** 2))  # Root Mean Square MCE
    max_mce = torch.max(abs_errors)  # Maximum MCE

    # Move back to original device if needed
    device = y_true.device
    if device.type != "cpu":
        mce = mce.to(device)
        rmsce = rmsce.to(device)
        max_mce = max_mce.to(device)

    result = {
        "marginal_calibration_error": mce.item(),
        "root_mean_squared_mce": rmsce.item(),
        "maximum_marginal_calibration_error": max_mce.item(),
    }

    if return_diagnostics:
        # Add diagnostic information
        result.update(
            {
                "bin_centers": (bin_edges[:-1] + bin_edges[1:]).tolist(),
                "observed_cdf": obs_cdf.tolist(),
                "predicted_cdf": pred_cdf_mean.tolist(),
                "abs_errors": abs_errors.tolist(),
            }
        )

    return result


def calibration_metrics_report(
    distribution_or_samples: Union[
        torch.distributions.Distribution, Dict[str, torch.Tensor], torch.Tensor
    ],
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred_quantiles: Optional[Dict[float, Union[torch.Tensor, np.ndarray]]] = None,
    quantile_levels: List[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
) -> Dict[str, float]:
    """
    Generate a comprehensive report on calibration quality.

    This function calculates multiple calibration metrics depending on available inputs.

    Args:
        distribution_or_samples: Either a PyTorch distribution, samples [n_samples, batch_size, ...],
                              or dict with distribution parameters
        y_true: Ground truth values
        y_pred_quantiles: Dictionary mapping quantile levels to predictions (optional)
        quantile_levels: Quantile levels to check if using distribution or samples

    Returns:
        Dictionary with calibration metrics
    """
    y_true = convert_to_tensor(y_true)
    metrics = {}

    # Handle different input types
    distribution = None
    samples = None

    if isinstance(distribution_or_samples, torch.distributions.Distribution):
        distribution = distribution_or_samples
    elif isinstance(distribution_or_samples, torch.Tensor):
        samples = distribution_or_samples
    elif isinstance(distribution_or_samples, dict):
        # Handle dictionary of distribution parameters
        if "loc" in distribution_or_samples and "scale" in distribution_or_samples:
            from torch.distributions import Normal

            loc = distribution_or_samples["loc"]
            scale = distribution_or_samples["scale"]
            distribution = Normal(loc, scale)

    # Generate quantiles from distribution if available
    if y_pred_quantiles is None and distribution is not None:
        y_pred_quantiles = {}
        for q in quantile_levels:
            try:
                y_pred_quantiles[q] = distribution.icdf(torch.tensor(q, device=y_true.device))
            except (NotImplementedError, RuntimeError):
                # Some distributions don't support icdf
                break

    # Generate quantiles from samples if available
    if y_pred_quantiles is None and samples is not None:
        y_pred_quantiles = {}
        for q in quantile_levels:
            y_pred_quantiles[q] = torch.quantile(samples, q, dim=0)

    # Calculate ECE if quantiles are available
    if y_pred_quantiles is not None:
        ece_metrics = expected_calibration_error(y_pred_quantiles, y_true)
        metrics.update(ece_metrics)

    # Calculate Marginal Calibration Error if samples are available
    if samples is not None:
        mce_metrics = marginal_calibration_error(samples, y_true)
        metrics.update(mce_metrics)

    return metrics
