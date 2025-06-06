"""
Interval metrics for evaluating prediction intervals in regression.
"""

import torch
import numpy as np
from typing import Union, Dict

from torchregress.metrics.utils import convert_to_tensor, apply_reduction, validate_inputs


def interval_score(
    lower_bound: Union[torch.Tensor, np.ndarray],
    upper_bound: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.1,
    reduction: str = "mean",
) -> Union[Dict[str, Union[torch.Tensor]], torch.Tensor]:
    """
    Calculate prediction interval score (Winkler score).

    This score rewards narrow intervals but penalizes when the true value
    falls outside the interval.

    Args:
        lower_bound: Lower bound of prediction interval (e.g., q=0.1)
        upper_bound: Upper bound of prediction interval (e.g., q=0.9)
        y_true: Ground truth values
        alpha: Significance level (e.g., 0.1 for 90% interval)
        reduction: How to reduce the score ("none", "mean", "sum", or "full")
            If "full", return a dictionary with detailed metrics

    Returns:
        If reduction="full": Dictionary with interval scores and components
        Otherwise: Tensor of interval scores (reduced as specified)
    """
    y_true = convert_to_tensor(y_true)
    lower_bound = convert_to_tensor(lower_bound)
    upper_bound = convert_to_tensor(upper_bound)

    validate_inputs(lower_bound, y_true)
    validate_inputs(upper_bound, y_true)

    # Calculate interval width
    interval_width = upper_bound - lower_bound

    # Validate interval bounds
    if torch.any(interval_width < 0):
        raise ValueError("Upper bounds must be greater than or equal to lower bounds")

    # Calculate penalties for observations outside the interval
    below_lower = torch.clamp(lower_bound - y_true, min=0)
    above_upper = torch.clamp(y_true - upper_bound, min=0)

    # Calculate the interval score: width + penalties
    score = interval_width + (2 / alpha) * (below_lower + above_upper)

    # Calculate coverage (1 if y is in the interval, 0 otherwise)
    coverage = ((y_true >= lower_bound) & (y_true <= upper_bound)).float()

    if reduction == "full":
        return {
            "score": torch.mean(score),
            "mean_width": torch.mean(interval_width),
            "mean_coverage": torch.mean(coverage),
            "expected_coverage": torch.tensor(1 - alpha),
            "coverage_error": torch.abs(torch.mean(coverage) - torch.tensor(1 - alpha)),
            "penalty_below": torch.mean(below_lower),
            "penalty_above": torch.mean(above_upper),
        }
    else:
        return apply_reduction(score, reduction)


def prediction_interval_coverage_probability(
    lower_bound: Union[torch.Tensor, np.ndarray],
    upper_bound: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    expected_coverage: float = 0.9,
    return_diagnostics: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Calculate Prediction Interval Coverage Probability (PICP) and related metrics.

    PICP measures the proportion of observations falling within the prediction interval.

    Args:
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
        y_true: Ground truth values
        expected_coverage: Expected coverage probability (e.g., 0.9 for 90% interval)
        return_diagnostics: Whether to return detailed diagnostics

    Returns:
        PICP value or dictionary with coverage metrics
    """
    y_true = convert_to_tensor(y_true)
    lower_bound = convert_to_tensor(lower_bound)
    upper_bound = convert_to_tensor(upper_bound)

    # Calculate coverage (1 if y is in the interval, 0 otherwise)
    coverage = ((y_true >= lower_bound) & (y_true <= upper_bound)).float()
    observed_coverage = torch.mean(coverage)

    if not return_diagnostics:
        return observed_coverage

    # Calculate Mean Prediction Interval Width (MPIW)
    interval_width = upper_bound - lower_bound
    mpiw = torch.mean(interval_width)

    # Calculate Normalized MPIW
    range_y = torch.max(y_true) - torch.min(y_true)
    nmpiw = mpiw / range_y

    # Calculate Coverage Error
    coverage_error = torch.abs(observed_coverage - torch.tensor(expected_coverage))

    # Calculate miss rates on both sides
    too_low = (y_true < lower_bound).float().mean()
    too_high = (y_true > upper_bound).float().mean()

    return {
        "picp": observed_coverage,
        "expected_coverage": torch.tensor(expected_coverage),
        "coverage_error": coverage_error,
        "mpiw": mpiw,
        "nmpiw": nmpiw,
        "miss_rate_low": too_low,
        "miss_rate_high": too_high,
        "miss_rate_ratio": too_high / too_low,
    }


def interval_metrics_report(
    predictions: Dict[str, Dict[str, Union[torch.Tensor, np.ndarray]]],
    y_true: Union[torch.Tensor, np.ndarray],
    alpha: float = 0.1,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Generate a comprehensive report on prediction interval quality for multiple models.

    Args:
        predictions: Dictionary mapping model names to dictionaries with 'lower' and 'upper' bounds
        y_true: Ground truth values
        alpha: Significance level (e.g., 0.1 for 90% interval)

    Returns:
        Dictionary with interval metrics for each model
    """
    y_true = convert_to_tensor(y_true)
    results = {}

    for model_name, pred in predictions.items():
        lower_bound = convert_to_tensor(pred["lower"])
        upper_bound = convert_to_tensor(pred["upper"])

        # Calculate interval score
        int_score = interval_score(lower_bound, upper_bound, y_true, alpha, reduction="full")

        # Calculate coverage probability
        coverage = prediction_interval_coverage_probability(
            lower_bound, upper_bound, y_true, expected_coverage=1 - alpha, return_diagnostics=True
        )

        # Combine metrics
        results[model_name] = {**int_score, **coverage}

    return results
