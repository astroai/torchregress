"""
DeepAR-style autoregressive loss placeholder.
"""

import math
from typing import Optional

import torch
from torch import Tensor

from .base import RegressionLoss


class DeepARLoss(RegressionLoss):
    """
    DeepAR-style loss: Gaussian NLL per timestep.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-6) -> None:
        """
        DeepAR-style loss: Gaussian NLL per timestep.
        reduction: 'none'|'mean'|'sum'
        eps: minimum variance for stability.
        """
        super().__init__(reduction=reduction)
        self.eps = eps
        self.log2pi = math.log(2 * math.pi)

    def forward(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute Gaussian NLL: 0.5*(log var + (y-mean)^2/var + log2π) per element.
        y_pred: Tensor[..., 2*n_features] or (mean, log_var) tuple.
        y_true: Tensor[..., n_features].
        """
        # Extract mean and log-variance
        if isinstance(y_pred, tuple) and len(y_pred) == 2:
            mean, log_var = y_pred
        else:
            last = y_pred.shape[-1]
            # Determine n_features
            n_feat = y_true.shape[-1] if y_true.dim() > 1 else 1
            if last == 2 * n_feat:
                mean = y_pred[..., :n_feat]
                log_var = y_pred[..., n_feat:]
            else:
                raise ValueError(
                    f"DeepARLoss expects y_pred shape [..., 2*n_features], got {y_pred.shape}"
                )
        var = torch.exp(log_var).clamp(min=self.eps)
        err2 = (y_true - mean) ** 2
        nll = 0.5 * (torch.log(var) + err2 / var + self.log2pi)
        return self._reduce_with_mask(nll, mask, weights)
