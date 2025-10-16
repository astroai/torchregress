"""
Point prediction metrics for regression evaluation.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torchmetrics import Metric
from torchmetrics import (
    ExplainedVariance,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogError,
    R2Score,
)

from torchregress.metrics.utils import convert_to_tensor, validate_inputs


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


class HuberLoss(Metric):
    """
    Huber loss - a robust loss function that's less sensitive to outliers.
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
