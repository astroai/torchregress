"""
Quantile regression loss functions.

Quantile regression provides a more complete view of the conditional distribution,
useful for estimating prediction intervals and handling heteroscedastic data.
These losses support:
- Single quantile estimation
- Multiple quantile prediction
- Properly ordered quantiles with crossover penalties
"""

from typing import Any, List, Optional, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.quantile import multi_quantile_loss, quantile_loss
from ..utils.validation import validate_quantile, validate_range
from .base import RegressionLoss
from .loss_registry import register_regression_loss


@register_regression_loss("quantile")
class QuantileLoss(RegressionLoss):
    """
    Quantile regression loss function.

    The quantile loss is asymmetric, penalizing overestimation and
    underestimation differently based on the quantile level:

    L(y, f(x)) = q * (y - f(x))    if y > f(x)
                 (1-q) * (f(x) - y) if y ≤ f(x)

    or equivalently:

    L(y, f(x)) = max(q * (y - f(x)), (q-1) * (y - f(x)))

    where q is the quantile level (0 < q < 1).

    Args:
        quantile: Quantile level (0 < q < 1). Default: 0.5 (median)
        reduction: Reduction method ('none', 'mean', 'sum'). Default: 'mean'

    Example:
        >>> # Median (q=0.5)
        >>> loss_fn = QuantileLoss(quantile=0.5)
        >>> y_pred = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([0.0, 2.0, 4.0])
        >>> loss_fn(y_pred, target)
        tensor(0.6667)

        >>> # 90th percentile (q=0.9)
        >>> loss_fn = QuantileLoss(quantile=0.9)
        >>> y_pred = torch.tensor([2.0, 3.0, 4.0])
        >>> target = torch.tensor([1.0, 3.0, 5.0])
        >>> loss_fn(y_pred, target)
        tensor(0.2333)  # Underestimation is penalized 9x more than overestimation
    """

    def __init__(self, quantile: float = 0.5, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.quantile = validate_range(quantile, 0.0, 1.0, "quantile")

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate quantile loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Quantile loss
        """
        self._validate_inputs(y_pred, target, mask)

        # Elementwise quantile loss via shared utility
        loss = quantile_loss(y_pred, target, self.quantile)
        return self._reduce_with_mask(loss, mask, weights)


@register_regression_loss("multi_quantile")
class MultiQuantileLoss(RegressionLoss):
    """
    Loss for multiple quantile levels simultaneously.

    This loss is useful for models that predict multiple quantiles at once,
    such as when generating prediction intervals.

    The combined loss is:
    L(y, f₁(x), f₂(x), ..., fₖ(x)) = (1/k) * ∑ᵢ L_qᵢ(y, fᵢ(x))

    where L_qᵢ is the quantile loss for the i-th quantile level qᵢ.

    Args:
        quantiles: List of quantile levels in ascending order
        joint_prediction: Whether predictions are passed as a joint tensor
        reduction: Reduction method ('none', 'mean', 'sum'). Default: 'mean'

    Example:
        >>> # Predict 10th, 50th and 90th percentiles together
        >>> loss_fn = MultiQuantileLoss(quantiles=[0.1, 0.5, 0.9])
        >>> # Predictions shape: [batch_size, num_quantiles, features]
        >>> y_pred = torch.tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]])
        >>> target = torch.tensor([[2.0, 3.0]])
        >>> loss_fn(y_pred, target)
        tensor(0.4667)
    """

    def __init__(
        self,
        quantiles: Union[List[float], torch.Tensor],
        joint_prediction: bool = True,
        quantile_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)

        # Convert list to tensor if needed
        if isinstance(quantiles, list):
            quantiles = torch.tensor(quantiles, dtype=torch.float32)

        # Validate quantile levels
        self.register_buffer("quantiles", validate_quantile(quantiles))
        self.num_quantiles = self.quantiles.size(0)
        self.joint_prediction = joint_prediction

        # Handle quantile weights
        if quantile_weights is not None:
            if isinstance(quantile_weights, list):
                quantile_weights = torch.tensor(quantile_weights, dtype=torch.float32)
            if len(quantile_weights) != self.num_quantiles:
                raise ValueError(
                    f"quantile_weights length {len(quantile_weights)} must match "
                    f"number of quantiles {self.num_quantiles}"
                )
            self.register_buffer("quantile_weights", quantile_weights)
        else:
            self.register_buffer(
                "quantile_weights", torch.ones(self.num_quantiles) / self.num_quantiles
            )

    def forward(
        self,
        y_pred: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate combined quantile loss for multiple levels.

        Args:
            y_pred: When joint_prediction=True: [batch_size, num_quantiles, n_features]
                   or [batch_size, n_features * num_quantiles]
                   Otherwise: List of quantile predictions, each [batch_size, n_features]
            target: Target values [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features]

        Returns:
            Combined quantile loss value
        """
        batch_size = target.shape[0]
        n_features = target.shape[1] if target.dim() > 1 else 1

        # Reshape target for broadcasting if it's a 1D tensor
        if target.dim() == 1:
            target = target.unsqueeze(1)

        # Handle mask and weights
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(1)
        if weights is not None and weights.dim() == 1:
            weights = weights.unsqueeze(1)

        # Process predictions based on format
        if self.joint_prediction:
            # Handle joint predictions
            if isinstance(y_pred, torch.Tensor):
                y_pred_tensor = y_pred  # Type narrowing
                # Process based on prediction shape
                if y_pred_tensor.dim() == 3 and y_pred_tensor.shape[1] == self.num_quantiles:
                    # [batch_size, num_quantiles, n_features] format
                    quantile_preds = y_pred_tensor
                elif (
                    y_pred_tensor.dim() == 2
                    and y_pred_tensor.shape[1] == n_features * self.num_quantiles
                ):
                    # [batch_size, n_features * num_quantiles] format
                    # Reshape to [batch_size, num_quantiles, n_features]
                    quantile_preds = y_pred_tensor.reshape(
                        batch_size, self.num_quantiles, n_features
                    )
                else:
                    raise ValueError(
                        f"Expected y_pred shape to be either "
                        f"[batch_size, {self.num_quantiles}, {n_features}] or "
                        f"[batch_size, {n_features * self.num_quantiles}], "
                        f"got {list(y_pred_tensor.shape)}"
                    )
            else:
                raise TypeError("With joint_prediction=True, y_pred must be a tensor")
        else:
            # Handle separate predictions (list of tensors)
            if isinstance(y_pred, (list, tuple)) and len(y_pred) == self.num_quantiles:
                # Stack predictions [batch_size, num_quantiles, n_features]
                quantile_preds = torch.stack(cast(List[torch.Tensor], y_pred), dim=1)
            else:
                raise TypeError(
                    f"With joint_prediction=False, y_pred must be a list or tuple "
                    f"of {self.num_quantiles} tensors"
                )

        # Elementwise multi-quantile loss via shared utility
        loss = multi_quantile_loss(
            quantile_preds, target, self.quantiles, quantile_weights=self.quantile_weights
        )
        return self._reduce_with_mask(loss, mask, weights)


@register_regression_loss("quantile_crossover")
class QuantileCrossoverLoss(RegressionLoss):
    """
    Loss that encourages proper ordering of quantile curves.

    In quantile regression, we expect lower quantiles to be below higher ones.
    This loss adds a penalty when this constraint is violated.

    The loss is defined as:
    L(y, {fᵢ(x)}) = base_loss * L_quantile(y, {fᵢ(x)}) +
                    crossover_penalty * ∑ᵢmax(fᵢ(x) - fᵢ₊₁(x), 0)

    where L_quantile is the standard quantile loss for multiple levels,
    and the second term penalizes cases where fᵢ(x) > fᵢ₊₁(x).

    Args:
        quantiles: List of quantile levels in ascending order
        base_loss: Weight for standard quantile loss term
        crossover_penalty: Weight for crossover penalty term
        reduction: Reduction method ('none', 'mean', 'sum'). Default: 'mean'

    Example:
        >>> # Create loss that predicts 10th, 50th, 90th percentiles
        >>> loss_fn = QuantileCrossoverLoss(quantiles=[0.1, 0.5, 0.9], crossover_penalty=5.0)
        >>> # Properly ordered predictions
        >>> good_pred = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
        >>> # Predictions with crossover (q₁ > q₂)
        >>> bad_pred = torch.tensor([[[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]])
        >>> target = torch.tensor([[2.0, 2.0]])
        >>> loss_fn(good_pred, target)  # Normal loss
        tensor(0.4667)
        >>> loss_fn(bad_pred, target)  # Higher loss due to crossover penalty
        tensor(3.1333)
    """

    def __init__(
        self,
        quantiles: Union[List[float], torch.Tensor],
        base_loss: float = 1.0,
        crossover_penalty: float = 10.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        # Ensure quantiles are sorted in ascending order
        if isinstance(quantiles, list):
            quantiles = sorted(quantiles)
            quantiles_tensor = torch.tensor(quantiles, dtype=torch.float32)
        else:
            sorted_indices = torch.argsort(quantiles)
            quantiles_tensor = quantiles[sorted_indices]

        self.register_buffer("quantiles", quantiles_tensor)
        self.num_quantiles = len(quantiles)
        self.base_loss = base_loss
        self.crossover_penalty = crossover_penalty

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate quantile loss with crossover penalty.

        Args:
            y_pred: Predicted quantiles [batch_size, num_quantiles, n_features]
            target: Target values [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features] or [batch_size]

        Returns:
            Loss combining standard quantile loss and crossover penalty
        """
        # Shape validation for y_pred
        if y_pred.shape[1] != self.num_quantiles:
            raise ValueError(
                f"Expected y_pred shape [batch_size, {self.num_quantiles}, n_features], "
                f"got shape {y_pred.shape}"
            )

        # Calculate standard quantile losses using vectorized utility
        # multi_quantile_loss returns [batch_size, n_features] (mean over quantiles)
        base_loss = multi_quantile_loss(
            y_pred, target, self.quantiles, quantile_weights=None
        )

        # Calculate crossover penalties vectorized
        # violations: [batch_size, num_quantiles - 1, n_features]
        violations = F.relu(y_pred[:, :-1] - y_pred[:, 1:])

        # Apply mask if provided
        if mask is not None:
            # Mask typically [batch_size, n_features], needs unsqueeze for quantiles
            if mask.dim() == violations.dim() - 1:
                mask_expanded = mask.unsqueeze(1)
            else:
                mask_expanded = mask
            violations = violations * mask_expanded

        # Sum violations across quantiles [batch_size, n_features]
        crossover_penalties = torch.sum(violations, dim=1)

        # Handle weights for the final combination if provided
        if weights is not None:
            if weights.dim() < base_loss.dim():
                for _ in range(base_loss.dim() - weights.dim()):
                    weights = weights.unsqueeze(-1)
            # We apply weights during reduction, but here we combine losses first
            # Since _reduce handles weighting, we just return the combined loss map

        # Final loss is weighted combination of base loss and crossover penalty
        final_loss = self.base_loss * base_loss + self.crossover_penalty * crossover_penalties

        # Apply final reduction
        return self._reduce(final_loss, mask, weights)
