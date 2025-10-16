"""
Multivariate regression metrics for vector-valued outputs.
"""

from typing import Any

import torch
from torchmetrics import Metric

from .utils import convert_to_tensor, validate_inputs


class MultivariateRMSE(Metric):
    """
    Root mean squared error over vector outputs.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)
        validate_inputs(y_pred, y_true)

        errs = torch.norm(y_pred - y_true, dim=1)
        self.sum_squared_error += torch.sum(errs**2)
        self.total += y_true.shape[0]

    def compute(self) -> torch.Tensor:
        """Compute multivariate RMSE."""
        return torch.sqrt(self.sum_squared_error / self.total)


class MultivariateMAE(Metric):
    """
    Mean absolute error over vector outputs.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)
        validate_inputs(y_pred, y_true)

        errs = torch.sum(torch.abs(y_pred - y_true), dim=1)
        self.sum_abs_error += torch.sum(errs)
        self.total += y_true.shape[0]

    def compute(self) -> torch.Tensor:
        """Compute multivariate MAE."""
        return self.sum_abs_error / self.total