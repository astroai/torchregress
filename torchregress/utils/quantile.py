"""
Quantile loss utilities for regression tasks.
"""

import torch
from torch import Tensor
from typing import Union


def quantile_loss(
    y_pred: Tensor,
    y_true: Tensor,
    quantile: Union[float, Tensor],
) -> Tensor:
    """
    Compute quantile loss for given predictions and targets.

    Args:
        y_pred: Predicted values [batch_size, ...]
        y_true: Ground truth values [batch_size, ...]
        quantile: Scalar or tensor of quantile level(s)

    Returns:
        Tensor of same shape as inputs representing loss values
    """
    # Ensure tensor quantile
    if not isinstance(quantile, Tensor):
        q = torch.tensor(quantile, device=y_pred.device, dtype=y_pred.dtype)
    else:
        q = quantile.to(y_pred.device).type_as(y_pred)
    # Calculate residuals
    res = y_true - y_pred
    return torch.max(q * res, (q - 1) * res)


def multi_quantile_loss(
    y_pred: Tensor,
    y_true: Tensor,
    quantiles: Tensor,
) -> Tensor:
    """
    Compute multi-quantile loss by averaging individual quantile losses.

    Args:
        y_pred: Predictions [batch_size, n_quantiles, ...]
        y_true: Targets [batch_size, ...]
        quantiles: Tensor of shape [n_quantiles]

    Returns:
        Tensor of shape [batch_size, ...] representing averaged loss
    """
    # Expand target for quantile dimension
    if y_pred.dim() >= 2:
        # y_pred shape: [batch, n_quantiles, *]
        y_shape = y_pred.shape
        if y_true.dim() < y_pred.dim():
            y_true = y_true.unsqueeze(1)
    # Broadcast quantiles to match prediction shape
    q = quantiles.view(1, -1, *[1] * (y_pred.dim() - 2)).to(y_pred.device).type_as(y_pred)
    # Residuals
    res = y_true - y_pred
    loss = torch.max(q * res, (q - 1) * res)
    # Average across quantile dimension
    return loss.mean(dim=1)
