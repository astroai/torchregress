"""
Multivariate regression metrics for vector-valued outputs.
"""

from typing import Union

import torch

from .utils import apply_reduction, convert_to_tensor, validate_inputs


def multivariate_rmse(
    y_pred: Union[torch.Tensor, any],
    y_true: Union[torch.Tensor, any],
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Root mean squared error over vector outputs.

    Computes per-sample Euclidean error: sqrt(sum_j (y_pred_ij - y_true_ij)^2)
    then applies reduction across batch.
    """
    y_pred = convert_to_tensor(y_pred)
    y_true = convert_to_tensor(y_true)
    validate_inputs(y_pred, y_true)
    # Euclidean norm per sample
    errs = torch.norm(y_pred - y_true, dim=1)
    return apply_reduction(errs, reduction)


def multivariate_mae(
    y_pred: Union[torch.Tensor, any],
    y_true: Union[torch.Tensor, any],
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Mean absolute error over vector outputs.

    Computes per-sample L1 norm across dimensions, then applies reduction.
    """
    y_pred = convert_to_tensor(y_pred)
    y_true = convert_to_tensor(y_true)
    validate_inputs(y_pred, y_true)
    errs = torch.sum(torch.abs(y_pred - y_true), dim=1)
    return apply_reduction(errs, reduction)
