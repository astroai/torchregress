"""
Multivariate regression metrics for vector-valued outputs.
"""

from typing import Any

import torch
from torchmetrics import Metric

from .utils import convert_to_tensor, metric_state_tensor, validate_inputs


class MultivariateRMSE(Metric):
    r"""
    Root mean squared Euclidean error over vector outputs.

    .. math::

        \mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \lVert \hat{y}_i - y_i \rVert_2^2}

    The squared error is summed over the D output dimensions, not averaged:
    the result is :math:`\sqrt{D}` times the per-element RMSE when all
    dimensions have the same error scale.
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
        metric_state_tensor(self.sum_squared_error).add_(torch.sum(errs**2))
        metric_state_tensor(self.total).add_(torch.as_tensor(y_true.shape[0], device=y_true.device))

    def compute(self) -> torch.Tensor:
        """Compute multivariate RMSE."""
        return torch.sqrt(
            metric_state_tensor(self.sum_squared_error) / metric_state_tensor(self.total)
        )


class MultivariateMAE(Metric):
    r"""
    Mean L1 error over vector outputs.

    .. math::

        \mathrm{MAE} = \frac{1}{N} \sum_{i=1}^{N} \lVert \hat{y}_i - y_i \rVert_1

    The absolute error is summed over the D output dimensions, not averaged:
    the result is D times the per-element MAE.
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
        metric_state_tensor(self.sum_abs_error).add_(torch.sum(errs))
        metric_state_tensor(self.total).add_(torch.as_tensor(y_true.shape[0], device=y_true.device))

    def compute(self) -> torch.Tensor:
        """Compute multivariate MAE."""
        return metric_state_tensor(self.sum_abs_error) / metric_state_tensor(self.total)
