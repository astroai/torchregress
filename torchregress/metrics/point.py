"""
Point prediction metrics for regression evaluation.
"""

from typing import Dict, Optional, Union

import numpy as np
import torch
from torchmetrics import (
    ExplainedVariance,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogError,
    R2Score,
)

from torchregress.metrics.utils import convert_to_tensor, validate_inputs


def median_absolute_error(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    multioutput: str = "uniform_average",
) -> Union[float, np.ndarray]:
    """
    Median absolute error regression loss.

    Robust to outliers. Not available in torchmetrics functional API.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        multioutput: How to aggregate multiple outputs

    Returns:
        Median absolute error value(s)
    """
    y_pred = convert_to_tensor(y_pred)
    y_true = convert_to_tensor(y_true)
    validate_inputs(y_pred, y_true)

    # Calculate absolute errors
    abs_errors = torch.abs(y_pred - y_true)

    # Handle multioutput case
    if multioutput == "raw_values" and y_true.ndim > 1 and y_true.shape[1] > 1:
        result = torch.median(abs_errors, dim=0)[0]
        return result.cpu().numpy() if isinstance(y_true, np.ndarray) else result
    else:
        if abs_errors.ndim > 1 and abs_errors.shape[1] > 1:
            # Average across features first
            errors = torch.mean(abs_errors, dim=1)
        else:
            errors = abs_errors.view(-1)

        result = torch.median(errors)
        return result.item() if isinstance(y_true, np.ndarray) else result


def normalized_rmse(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    normalization: str = "std",
) -> float:
    """
    Normalized Root Mean Square Error.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        normalization: How to normalize ('std', 'range', 'mean', or 'iqr')

    Returns:
        Normalized RMSE value
    """
    y_pred = convert_to_tensor(y_pred)
    y_true = convert_to_tensor(y_true)
    validate_inputs(y_pred, y_true)

    # Calculate RMSE using torchmetrics
    rmse = torch.sqrt(MeanSquaredError()(y_pred, y_true))

    # Normalize based on specified method
    if normalization == "std":
        norm_factor = torch.std(y_true)
    elif normalization == "range":
        norm_factor = torch.max(y_true) - torch.min(y_true)
    elif normalization == "mean":
        norm_factor = torch.mean(torch.abs(y_true))
    elif normalization == "iqr":
        q75 = torch.quantile(y_true, 0.75)
        q25 = torch.quantile(y_true, 0.25)
        norm_factor = q75 - q25
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    # Avoid division by zero
    if norm_factor < 1e-8:
        return float("inf")

    result = rmse / norm_factor
    return result.item() if isinstance(y_true, np.ndarray) else result


def huber_loss(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    delta: float = 1.0,
    reduction: str = "mean",
) -> Union[torch.Tensor, float]:
    """
    Huber loss - a robust loss function that's less sensitive to outliers.

    Combines the benefits of MSE and MAE by being MSE-like for small residuals
    and MAE-like for large residuals.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        delta: Threshold where loss changes from MSE to MAE
        reduction: How to reduce the loss ("mean", "sum", "none")

    Returns:
        Huber loss value(s)
    """
    y_pred = convert_to_tensor(y_pred)
    y_true = convert_to_tensor(y_true)
    validate_inputs(y_pred, y_true)

    # Calculate absolute error
    abs_error = torch.abs(y_true - y_pred)

    # Apply Huber loss formula
    quadratic = torch.min(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta * linear

    # Apply reduction
    if reduction == "mean":
        return torch.mean(loss)
    elif reduction == "sum":
        return torch.sum(loss)
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def trimmed_mean_squared_error(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    proportion: float = 0.1,
) -> float:
    """
    Trimmed Mean Squared Error - robust to outliers.

    Calculates MSE after removing the top and bottom proportion of errors.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        proportion: Proportion of data to trim from each end (0-0.5)

    Returns:
        Trimmed MSE value
    """
    if not 0 <= proportion < 0.5:
        raise ValueError("Proportion must be between 0 and 0.5")

    y_pred = convert_to_tensor(y_pred)
    y_true = convert_to_tensor(y_true)
    validate_inputs(y_pred, y_true)

    # Calculate squared errors
    squared_errors = (y_true - y_pred) ** 2

    # Flatten if multi-dimensional
    squared_errors_flat = squared_errors.reshape(-1)

    # Sort errors
    sorted_errors, _ = torch.sort(squared_errors_flat)

    # Calculate trim indices
    n = len(sorted_errors)
    lower_idx = int(n * proportion)
    upper_idx = int(n * (1 - proportion))

    # Calculate trimmed mean
    trimmed_errors = sorted_errors[lower_idx:upper_idx]
    return torch.mean(trimmed_errors).item()


def median_absolute_deviation(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    scale: float = 1.4826,  # Scaling factor for normal distribution
) -> float:
    """
    Median Absolute Deviation - highly robust to outliers.

    MAD = scale * median(|errors - median(errors)|)

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        scale: Scaling factor (1.4826 for normal distribution)

    Returns:
        MAD value
    """
    y_pred = convert_to_tensor(y_pred)
    y_true = convert_to_tensor(y_true)
    validate_inputs(y_pred, y_true)

    # Calculate errors
    errors = y_true - y_pred

    # Flatten if multi-dimensional
    errors_flat = errors.reshape(-1)

    # Calculate median error
    median_error = torch.median(errors_flat)

    # Calculate deviations from median error
    deviations = torch.abs(errors_flat - median_error)

    # Calculate MAD
    mad = torch.median(deviations)

    # Apply scaling factor
    return (scale * mad).item()


def outlier_fraction(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.15,
    scale: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mode: str = "photometric",
) -> float:
    """
    Calculate the fraction of outliers in predictions.

    For photometric redshift mode, an outlier is defined as:
    |z_pred - z_true|/(1 + z_true) > threshold

    For standard mode, an outlier is defined as:
    |y_pred - y_true|/scale > threshold

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        threshold: Threshold to define an outlier
        scale: Scaling factor (optional, defaults to std of y_true)
        mode: "photometric" for photometric redshift style or "standard"

    Returns:
        Fraction of outliers
    """
    y_pred = convert_to_tensor(y_pred)
    y_true = convert_to_tensor(y_true)
    validate_inputs(y_pred, y_true)

    # Calculate error
    abs_error = torch.abs(y_true - y_pred)

    if mode.lower() == "photometric":
        # Photometric redshift outlier definition
        # |z_pred - z_true|/(1 + z_true) > threshold
        scaled_error = abs_error / (1.0 + y_true)
    else:
        # Standard outlier definition
        if scale is None:
            scale = torch.std(y_true)
        scale = convert_to_tensor(scale)
        scaled_error = abs_error / scale

    # Calculate outlier mask and fraction
    outliers = scaled_error > threshold
    outlier_fraction_value = torch.mean(outliers.float()).item()

    return outlier_fraction_value


def normalized_median_absolute_deviation(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    normalization: str = "median",
) -> float:
    """
    Calculate the Normalized Median Absolute Deviation.

    NMAD = 1.4826 * median(|Δz - median(Δz)|)

    Where Δz can be normalized in different ways:
    - "median": Δz = (y_pred - y_true)
    - "photometric": Δz = (y_pred - y_true) / (1 + y_true)

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        normalization: How to normalize differences ("median", "photometric")

    Returns:
        NMAD value
    """
    y_pred = convert_to_tensor(y_pred)
    y_true = convert_to_tensor(y_true)
    validate_inputs(y_pred, y_true)

    # Calculate differences
    diff = y_pred - y_true

    # Apply normalization
    if normalization == "photometric":
        # Photometric redshift normalization
        diff = diff / (1.0 + y_true)

    # Calculate median difference
    median_diff = torch.median(diff)

    # Calculate absolute deviations from median
    abs_dev = torch.abs(diff - median_diff)

    # Calculate NMAD
    nmad = 1.4826 * torch.median(abs_dev)

    return nmad.item()


def regression_metrics_report(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    sample_weight: Optional[Union[torch.Tensor, np.ndarray]] = None,
    include_robust: bool = True,
    include_outliers: bool = True,
) -> Dict[str, float]:
    """
    Generate a comprehensive regression metrics report.

    This is useful for model evaluation and comparison.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        sample_weight: Optional weights for samples
        include_robust: Whether to include robust metrics
        include_outliers: Whether to include outlier metrics

    Returns:
        Dictionary with multiple regression metrics
    """
    y_pred = convert_to_tensor(y_pred)
    y_true = convert_to_tensor(y_true)
    validate_inputs(y_pred, y_true)

    metrics = {}

    # Calculate basic metrics
    metrics["mse"] = MeanSquaredError()(y_pred, y_true)
    metrics["rmse"] = torch.sqrt(torch.tensor(metrics["mse"])).item()
    metrics["mae"] = MeanAbsoluteError()(y_pred, y_true)
    metrics["r2"] = R2Score()(y_pred, y_true)
    metrics["explained_var"] = ExplainedVariance()(y_pred, y_true)
    metrics["median_ae"] = median_absolute_error(y_pred, y_true)

    # Only include MAPE and MSLE if data is suitable
    if torch.all(y_true > 0) and torch.all(y_pred > 0):
        metrics["mape"] = MeanAbsolutePercentageError()(y_pred, y_true)
        metrics["msle"] = MeanSquaredLogError()(y_pred, y_true)

    # Add normalized RMSE variants
    for norm in ["std", "range", "mean", "iqr"]:
        metrics[f"nrmse_{norm}"] = normalized_rmse(y_pred, y_true, normalization=norm)

    # Calculate percentile errors
    abs_errors = torch.abs(y_pred - y_true)
    for p in [50, 90, 95, 99]:
        metrics[f"error_p{p}"] = torch.quantile(abs_errors, p / 100).item()

    # Add error distribution statistics
    errors = y_pred - y_true
    metrics["error_mean"] = torch.mean(errors).item()
    metrics["error_std"] = torch.std(errors).item()
    metrics["error_skew"] = torch.mean(
        ((errors - torch.mean(errors)) / torch.std(errors)) ** 3
    ).item()

    # Add robust metrics
    if include_robust:
        metrics["huber_loss"] = huber_loss(y_pred, y_true).item()
        metrics["trimmed_mse"] = trimmed_mean_squared_error(y_pred, y_true)
        metrics["mad"] = median_absolute_deviation(y_pred, y_true)
        metrics["nmad"] = normalized_median_absolute_deviation(y_pred, y_true)
        metrics["nmad_photometric"] = normalized_median_absolute_deviation(
            y_pred, y_true, "photometric"
        )

    # Add outlier metrics
    if include_outliers:
        metrics["outlier_fraction"] = outlier_fraction(y_pred, y_true)
        metrics["outlier_fraction_5pct"] = outlier_fraction(y_pred, y_true, threshold=0.05)
        metrics["outlier_fraction_10pct"] = outlier_fraction(y_pred, y_true, threshold=0.10)

    return metrics


# Define point metrics as TorchMetrics instances
mean_squared_error = MeanSquaredError()
mean_absolute_error = MeanAbsoluteError()
r2_score = R2Score()
explained_variance_score = ExplainedVariance()
mean_absolute_percentage_error = MeanAbsolutePercentageError()
mean_squared_log_error = MeanSquaredLogError()
