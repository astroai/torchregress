"""
Point prediction metrics for regression evaluation.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torchmetrics import (
    MeanSquaredError,
    Metric,
    R2Score,
)

from torchregress.metrics.utils import (
    convert_to_tensor,
    create_metric_result,
    validate_inputs,
    validate_sample_weight,
)


class MedianAbsoluteError(Metric):
    """
    Median absolute error regression loss.

    Robust to outliers.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, multioutput: str = "uniform_average", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.multioutput = multioutput
        self.add_state("errors", default=[], dist_reduce_fx="cat")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)
        validate_inputs(y_pred, y_true)

        abs_errors = torch.abs(y_pred - y_true)
        self.errors.append(abs_errors)

    def compute(self) -> torch.Tensor:
        """Compute median absolute error."""
        errors = torch.cat(self.errors)
        if self.multioutput == "raw_values" and errors.ndim > 1 and errors.shape[1] > 1:
            return torch.median(errors, dim=0)[0]
        else:
            if errors.ndim > 1 and errors.shape[1] > 1:
                errors = torch.mean(errors, dim=1)
            else:
                errors = errors.view(-1)
            return torch.median(errors)


class NormalizedRMSE(Metric):
    """
    Normalized Root Mean Square Error.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, normalization: str = "std", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.normalization = normalization
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")
        self.add_state("y_true", default=[], dist_reduce_fx="cat")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        self.y_pred.append(convert_to_tensor(y_pred))
        self.y_true.append(convert_to_tensor(y_true))

    def compute(self) -> torch.Tensor:
        """Compute normalized RMSE."""
        y_pred = torch.cat(self.y_pred)
        y_true = torch.cat(self.y_true)
        validate_inputs(y_pred, y_true)

        rmse = torch.sqrt(MeanSquaredError()(y_pred, y_true))

        if self.normalization == "std":
            norm_factor = torch.std(y_true)
        elif self.normalization == "range":
            norm_factor = torch.max(y_true) - torch.min(y_true)
        elif self.normalization == "mean":
            norm_factor = torch.mean(torch.abs(y_true))
        elif self.normalization == "iqr":
            q75 = torch.quantile(y_true, 0.75)
            q25 = torch.quantile(y_true, 0.25)
            norm_factor = q75 - q25
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")

        if norm_factor < 1e-8:
            return torch.tensor(float("inf"))

        return rmse / norm_factor


class HuberMetric(Metric):
    """
    Huber loss metric - a robust loss function that's less sensitive to outliers.
    """

    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, delta: float = 1.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.delta = delta
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)
        validate_inputs(y_pred, y_true)

        abs_error = torch.abs(y_true - y_pred)
        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear

        self.loss += torch.sum(loss)
        self.total += y_true.numel()

    def compute(self) -> torch.Tensor:
        """Compute Huber loss."""
        return self.loss / self.total


class TrimmedMeanSquaredError(Metric):
    """
    Trimmed Mean Squared Error - robust to outliers.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, proportion: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not 0 <= proportion < 0.5:
            raise ValueError("Proportion must be between 0 and 0.5")
        self.proportion = proportion
        self.add_state("errors", default=[], dist_reduce_fx="cat")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)
        validate_inputs(y_pred, y_true)

        squared_errors = (y_true - y_pred) ** 2
        self.errors.append(squared_errors.view(-1))

    def compute(self) -> torch.Tensor:
        """Compute trimmed mean squared error."""
        errors = torch.cat(self.errors)
        sorted_errors, _ = torch.sort(errors)
        n = len(sorted_errors)
        lower_idx = int(n * self.proportion)
        upper_idx = int(n * (1 - self.proportion))
        return torch.mean(sorted_errors[lower_idx:upper_idx])


class MedianAbsoluteDeviation(Metric):
    """
    Median Absolute Deviation - highly robust to outliers.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, scale: float = 1.4826, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.scale = scale
        self.add_state("errors", default=[], dist_reduce_fx="cat")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)
        validate_inputs(y_pred, y_true)

        errors = y_true - y_pred
        self.errors.append(errors.view(-1))

    def compute(self) -> torch.Tensor:
        """Compute median absolute deviation."""
        errors = torch.cat(self.errors)
        median_error = torch.median(errors)
        deviations = torch.abs(errors - median_error)
        mad = torch.median(deviations)
        return self.scale * mad


class OutlierFraction(Metric):
    """
    Calculate the fraction of outliers in predictions.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, threshold: float = 0.15, mode: str = "photometric", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.mode = mode
        self.add_state("outliers", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)
        validate_inputs(y_pred, y_true)

        abs_error = torch.abs(y_true - y_pred)

        if self.mode.lower() == "photometric":
            scaled_error = abs_error / (1.0 + y_true)
        else:
            scale = torch.std(y_true)
            scaled_error = abs_error / scale

        outliers = scaled_error > self.threshold
        self.outliers += torch.sum(outliers)
        self.total += y_true.numel()

    def compute(self) -> torch.Tensor:
        """Compute outlier fraction."""
        return self.outliers / self.total


class NormalizedMedianAbsoluteDeviation(Metric):
    """
    Calculate the Normalized Median Absolute Deviation.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, normalization: str = "median", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.normalization = normalization
        self.add_state("diffs", default=[], dist_reduce_fx="cat")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)
        validate_inputs(y_pred, y_true)

        diff = y_pred - y_true
        if self.normalization == "photometric":
            diff = diff / (1.0 + y_true)
        self.diffs.append(diff.view(-1))

    def compute(self) -> torch.Tensor:
        """Compute normalized median absolute deviation."""
        diffs = torch.cat(self.diffs)
        median_diff = torch.median(diffs)
        abs_dev = torch.abs(diffs - median_diff)
        nmad = 1.4826 * torch.median(abs_dev)
        return nmad


def _per_sample_mean(values: torch.Tensor) -> torch.Tensor:
    if values.dim() > 1:
        return values.reshape(values.shape[0], -1).mean(dim=1)
    return values.reshape(-1)


def _apply_sample_weight(
    values: torch.Tensor, sample_weight: Optional[Union[torch.Tensor, np.ndarray]]
) -> torch.Tensor:
    if sample_weight is None:
        return values

    weights = convert_to_tensor(sample_weight).to(device=values.device, dtype=values.dtype)
    if weights.dim() > 1 and weights.shape[1] > 1:
        weights = weights.mean(dim=1)
    weights = validate_sample_weight(weights, values.shape[0])
    return values * weights


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return values
    if reduction == "sum":
        return torch.sum(values)
    if reduction == "mean":
        return torch.mean(values)
    raise ValueError(f"Unknown reduction: {reduction}")


def mean_squared_error(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    sample_weight: Optional[Union[torch.Tensor, np.ndarray]] = None,
    reduction: str = "mean",
    as_numpy: bool = False,
) -> Union[torch.Tensor, float, np.ndarray]:
    """Compute Mean Squared Error (MSE)."""
    y_pred_t = convert_to_tensor(y_pred)
    y_true_t = convert_to_tensor(y_true)
    validate_inputs(y_pred_t, y_true_t)

    squared_error = (y_pred_t - y_true_t) ** 2
    per_sample = _per_sample_mean(squared_error)
    per_sample = _apply_sample_weight(per_sample, sample_weight)
    result = _reduce(per_sample, reduction)

    if as_numpy or isinstance(y_pred, np.ndarray) or isinstance(y_true, np.ndarray):
        return create_metric_result(result, as_numpy=True)
    return create_metric_result(result, as_numpy=False)


def mse(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    sample_weight: Optional[Union[torch.Tensor, np.ndarray]] = None,
    reduction: str = "mean",
    as_numpy: bool = False,
) -> Union[torch.Tensor, float, np.ndarray]:
    """Alias for mean_squared_error."""
    return mean_squared_error(
        y_pred, y_true, sample_weight=sample_weight, reduction=reduction, as_numpy=as_numpy
    )


def rmse(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    sample_weight: Optional[Union[torch.Tensor, np.ndarray]] = None,
    reduction: str = "mean",
    as_numpy: bool = False,
) -> Union[torch.Tensor, float, np.ndarray]:
    """Compute Root Mean Squared Error (RMSE)."""
    y_pred_t = convert_to_tensor(y_pred)
    y_true_t = convert_to_tensor(y_true)
    validate_inputs(y_pred_t, y_true_t)

    squared_error = (y_pred_t - y_true_t) ** 2
    per_sample = _per_sample_mean(squared_error)
    per_sample = _apply_sample_weight(per_sample, sample_weight)

    if reduction == "none":
        result = torch.sqrt(per_sample)
    elif reduction == "sum":
        result = torch.sqrt(torch.sum(per_sample))
    elif reduction == "mean":
        result = torch.sqrt(torch.mean(per_sample))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    if as_numpy or isinstance(y_pred, np.ndarray) or isinstance(y_true, np.ndarray):
        return create_metric_result(result, as_numpy=True)
    return create_metric_result(result, as_numpy=False)


def mean_absolute_error(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    sample_weight: Optional[Union[torch.Tensor, np.ndarray]] = None,
    reduction: str = "mean",
    as_numpy: bool = False,
) -> Union[torch.Tensor, float, np.ndarray]:
    """Compute Mean Absolute Error (MAE)."""
    y_pred_t = convert_to_tensor(y_pred)
    y_true_t = convert_to_tensor(y_true)
    validate_inputs(y_pred_t, y_true_t)

    abs_error = torch.abs(y_pred_t - y_true_t)
    per_sample = _per_sample_mean(abs_error)
    per_sample = _apply_sample_weight(per_sample, sample_weight)
    result = _reduce(per_sample, reduction)

    if as_numpy or isinstance(y_pred, np.ndarray) or isinstance(y_true, np.ndarray):
        return create_metric_result(result, as_numpy=True)
    return create_metric_result(result, as_numpy=False)


def mae(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    sample_weight: Optional[Union[torch.Tensor, np.ndarray]] = None,
    reduction: str = "mean",
    as_numpy: bool = False,
) -> Union[torch.Tensor, float, np.ndarray]:
    """Alias for mean_absolute_error."""
    return mean_absolute_error(
        y_pred, y_true, sample_weight=sample_weight, reduction=reduction, as_numpy=as_numpy
    )


def huber_loss(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    delta: float = 1.0,
    sample_weight: Optional[Union[torch.Tensor, np.ndarray]] = None,
    reduction: str = "mean",
    as_numpy: bool = False,
) -> Union[torch.Tensor, float, np.ndarray]:
    """Compute Huber loss."""
    y_pred_t = convert_to_tensor(y_pred)
    y_true_t = convert_to_tensor(y_true)
    validate_inputs(y_pred_t, y_true_t)

    abs_error = torch.abs(y_true_t - y_pred_t)
    delta_t = torch.tensor(delta, device=abs_error.device, dtype=abs_error.dtype)
    quadratic = torch.minimum(abs_error, delta_t)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic**2 + delta_t * linear

    per_sample = _per_sample_mean(loss)
    per_sample = _apply_sample_weight(per_sample, sample_weight)
    result = _reduce(per_sample, reduction)

    if as_numpy or isinstance(y_pred, np.ndarray) or isinstance(y_true, np.ndarray):
        return create_metric_result(result, as_numpy=True)
    return create_metric_result(result, as_numpy=False)


def median_absolute_error(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    multioutput: str = "uniform_average",
    as_numpy: bool = False,
) -> Union[torch.Tensor, float, np.ndarray]:
    """Compute Median Absolute Error."""
    y_pred_t = convert_to_tensor(y_pred)
    y_true_t = convert_to_tensor(y_true)
    validate_inputs(y_pred_t, y_true_t)

    abs_errors = torch.abs(y_pred_t - y_true_t)
    if multioutput == "raw_values" and abs_errors.ndim > 1 and abs_errors.shape[1] > 1:
        result = torch.median(abs_errors, dim=0)[0]
    else:
        if abs_errors.ndim > 1 and abs_errors.shape[1] > 1:
            abs_errors = torch.mean(abs_errors, dim=1)
        else:
            abs_errors = abs_errors.reshape(-1)
        result = torch.median(abs_errors)

    if as_numpy or isinstance(y_pred, np.ndarray) or isinstance(y_true, np.ndarray):
        return create_metric_result(result, as_numpy=True)
    return create_metric_result(result, as_numpy=False)


def trimmed_mean_squared_error(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    proportion: float = 0.1,
    as_numpy: bool = False,
) -> Union[torch.Tensor, float, np.ndarray]:
    """Compute Trimmed Mean Squared Error."""
    if not 0 <= proportion < 0.5:
        raise ValueError("Proportion must be between 0 and 0.5")

    y_pred_t = convert_to_tensor(y_pred)
    y_true_t = convert_to_tensor(y_true)
    validate_inputs(y_pred_t, y_true_t)

    squared_errors = (y_true_t - y_pred_t) ** 2
    flattened = squared_errors.reshape(-1)
    sorted_errors, _ = torch.sort(flattened)
    n = sorted_errors.numel()
    lower_idx = int(n * proportion)
    upper_idx = int(n * (1 - proportion))
    result = torch.mean(sorted_errors[lower_idx:upper_idx])

    if as_numpy or isinstance(y_pred, np.ndarray) or isinstance(y_true, np.ndarray):
        return create_metric_result(result, as_numpy=True)
    return create_metric_result(result, as_numpy=False)


def median_absolute_deviation(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    scale: float = 1.4826,
    as_numpy: bool = False,
) -> Union[torch.Tensor, float, np.ndarray]:
    """Compute Median Absolute Deviation (MAD)."""
    y_pred_t = convert_to_tensor(y_pred)
    y_true_t = convert_to_tensor(y_true)
    validate_inputs(y_pred_t, y_true_t)

    errors = (y_true_t - y_pred_t).reshape(-1)
    median_error = torch.median(errors)
    deviations = torch.abs(errors - median_error)
    result = scale * torch.median(deviations)

    if as_numpy or isinstance(y_pred, np.ndarray) or isinstance(y_true, np.ndarray):
        return create_metric_result(result, as_numpy=True)
    return create_metric_result(result, as_numpy=False)


def normalized_rmse(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    normalization: str = "std",
    as_numpy: bool = False,
) -> Union[torch.Tensor, float, np.ndarray]:
    """Compute Normalized RMSE."""
    y_pred_t = convert_to_tensor(y_pred)
    y_true_t = convert_to_tensor(y_true)
    validate_inputs(y_pred_t, y_true_t)

    rmse = torch.sqrt(torch.mean((y_pred_t - y_true_t) ** 2))
    y_true_flat = y_true_t.reshape(-1)

    if normalization == "std":
        norm_factor = torch.std(y_true_flat)
    elif normalization == "range":
        norm_factor = torch.max(y_true_flat) - torch.min(y_true_flat)
    elif normalization == "mean":
        norm_factor = torch.mean(torch.abs(y_true_flat))
    elif normalization == "iqr":
        q75 = torch.quantile(y_true_flat, 0.75)
        q25 = torch.quantile(y_true_flat, 0.25)
        norm_factor = q75 - q25
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    if norm_factor < 1e-8:
        result = torch.tensor(float("inf"), device=y_true_t.device, dtype=y_true_t.dtype)
    else:
        result = rmse / norm_factor

    if as_numpy or isinstance(y_pred, np.ndarray) or isinstance(y_true, np.ndarray):
        return create_metric_result(result, as_numpy=True)
    return create_metric_result(result, as_numpy=False)


def regression_metrics_report(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    sample_weight: Optional[Union[torch.Tensor, np.ndarray]] = None,
    as_numpy: bool = False,
) -> Dict[str, Union[torch.Tensor, float, np.ndarray]]:
    """Generate a comprehensive regression metrics report."""
    y_pred_t = convert_to_tensor(y_pred)
    y_true_t = convert_to_tensor(y_true)
    validate_inputs(y_pred_t, y_true_t)

    squared_error = (y_pred_t - y_true_t) ** 2
    abs_error = torch.abs(y_pred_t - y_true_t)

    per_sample_mse = _apply_sample_weight(_per_sample_mean(squared_error), sample_weight)
    per_sample_mae = _apply_sample_weight(_per_sample_mean(abs_error), sample_weight)

    mse = torch.mean(per_sample_mse)
    rmse = torch.sqrt(mse)
    mae = torch.mean(per_sample_mae)

    r2 = R2Score()(y_pred_t, y_true_t)
    huber = huber_loss(y_pred_t, y_true_t, reduction="mean")
    mad = median_absolute_deviation(y_pred_t, y_true_t)

    nmad_metric = NormalizedMedianAbsoluteDeviation()
    nmad_metric.update(y_pred_t, y_true_t)
    nmad = nmad_metric.compute()

    outlier_metric = OutlierFraction()
    outlier_metric.update(y_pred_t, y_true_t)
    outlier_fraction = outlier_metric.compute()

    result = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "huber_loss": huber,
        "mad": mad,
        "nmad": nmad,
        "outlier_fraction": outlier_fraction,
    }

    if as_numpy or isinstance(y_pred, np.ndarray) or isinstance(y_true, np.ndarray):
        return create_metric_result(result, as_numpy=True)
    return create_metric_result(result, as_numpy=False)
