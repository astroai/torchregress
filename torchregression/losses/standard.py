"""
Standard regression loss functions.

This module provides implementations of common loss functions used in regression,
including direct implementations (MSE, BCE) and wrapped versions of PyTorch losses
with added support for masking and weights.
"""

import torch
import torch.nn as nn

from .base import MaskedLoss, RegressionLoss, TorchLossWrapper


class MSELoss(RegressionLoss):
    """
    Mean Squared Error loss with masking support.

    This extends PyTorch's MSELoss with support for masks and per-sample weights.

    Args:
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)

    def forward(self, y_pred, target, mask=None, weights=None):
        """
        Calculate MSE loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            MSE loss
        """
        self._validate_inputs(y_pred, target, mask=mask)

        # Apply mask if provided
        y_pred = self._apply_mask(y_pred, mask)
        target = self._apply_mask(target, mask)

        # Calculate squared error
        squared_error = (target - y_pred) ** 2

        # Apply weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            squared_error = squared_error * weights

        return self._reduce(squared_error, mask)


class BCELoss(MaskedLoss):
    """
    Binary Cross Entropy loss with masking support.

    This extends PyTorch's BCELoss with support for masks and per-sample weights.

    Args:
        pos_weight: Weight for the positive class
        weight: Global weighting
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        eps: Small constant for numerical stability
    """

    def __init__(self, pos_weight=None, weight=None, reduction="mean", eps=1e-8):
        super().__init__(reduction=reduction)
        self.pos_weight = pos_weight
        self.weight = weight
        self.eps = eps

    def forward(self, y_pred, target, mask=None, weights=None):
        """
        Calculate binary cross entropy loss.

        Args:
            y_pred: Predicted probabilities [batch_size, ...] (0 to 1)
            target: Ground truth values [batch_size, ...] (0 or 1)
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional sample weights [batch_size, ...] or [batch_size]

        Returns:
            BCE loss value
        """
        # Apply mask if provided
        target = self._apply_mask(target, mask)
        y_pred = self._apply_mask(y_pred, mask)

        # Clamp predictions for numerical stability
        y_pred = torch.clamp(y_pred, min=self.eps, max=1 - self.eps)

        # Calculate BCELoss
        if self.pos_weight is not None:
            pos_weight = self.pos_weight
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight, device=y_pred.device, dtype=y_pred.dtype)
            loss = -(pos_weight * target * torch.log(y_pred) + (1 - target) * torch.log(1 - y_pred))
        else:
            loss = -(target * torch.log(y_pred) + (1 - target) * torch.log(1 - y_pred))

        # Apply global weight if provided
        if self.weight is not None:
            loss = loss * self.weight

        # Apply sample weights if provided
        if weights is not None:
            weights = self._apply_mask(weights, mask)
            loss = loss * weights

        return self._reduce(loss, mask)


# Create wrapped versions of common PyTorch losses with masking support
MaskedMSELoss = TorchLossWrapper(nn.MSELoss)
MaskedL1Loss = TorchLossWrapper(nn.L1Loss)
MaskedCrossEntropyLoss = TorchLossWrapper(nn.CrossEntropyLoss)
MaskedBCELoss = TorchLossWrapper(nn.BCELoss)
MaskedBCEWithLogitsLoss = TorchLossWrapper(nn.BCEWithLogitsLoss)
MaskedKLDivLoss = TorchLossWrapper(nn.KLDivLoss)
MaskedNLLLoss = TorchLossWrapper(nn.NLLLoss)
MaskedSmoothL1Loss = TorchLossWrapper(nn.SmoothL1Loss)
MaskedHuberLoss = TorchLossWrapper(nn.HuberLoss)
MaskedPoissonNLLLoss = TorchLossWrapper(nn.PoissonNLLLoss)
