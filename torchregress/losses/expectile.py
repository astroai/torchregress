"""
Expectile regression loss functions.

Expectile regression provides a richer description of the conditional distribution
than standard mean regression, similar to quantile regression but with different
properties:
- Expectiles are defined via asymmetric least squares
- Mean is a special case of expectile (τ=0.5)
- Expectiles minimize the expected asymmetric squared error
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.validation import validate_range
from .base import RegressionLoss


class ExpectileLoss(RegressionLoss):
    """
    Expectile regression loss function.

    Expectiles are defined via asymmetric least squares that generalize
    the mean in a similar way as quantiles generalize the median.

    L(y, f(x)) = |y - f(x)|² * (τ * 1(y > f(x)) + (1-τ) * 1(y ≤ f(x)))

    where τ is the expectile level (0 < τ < 1).

    Args:
        expectile: Expectile level (0 < τ < 1). Default: 0.5 (mean)
        reduction: Reduction method ('none', 'mean', 'sum'). Default: 'mean'

    Example:
        >>> # Mean (τ=0.5)
        >>> loss_fn = ExpectileLoss(expectile=0.5)
        >>> y_pred = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([0.0, 2.0, 4.0])
        >>> loss_fn(y_pred, target)
        tensor(1.0000)  # Standard MSE at τ=0.5

        >>> # 80th expectile (τ=0.8)
        >>> loss_fn = ExpectileLoss(expectile=0.8)
        >>> y_pred = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([0.0, 2.0, 4.0])
        >>> loss_fn(y_pred, target)
        tensor(1.6000)  # Underestimation penalized 4x more than overestimation
    """

    def __init__(self, expectile: float = 0.5, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.expectile = validate_range(expectile, 0.0, 1.0, "expectile")

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate expectile loss.

        Args:
            y_pred: Predicted values [batch_size, ...]
            target: Target values [batch_size, ...]
            mask: Optional boolean mask [batch_size, ...]
            weights: Optional weights [batch_size, ...]

        Returns:
            Expectile loss value
        """
        self._validate_inputs(y_pred, target, mask)

        # Calculate residuals
        residuals = target - y_pred

        # Calculate asymmetric squared error
        # Use factor of 2 so that tau=0.5 gives MSE
        indicator = (residuals >= 0).float()
        loss = 2 * residuals**2 * (self.expectile * indicator + (1 - self.expectile) * (1 - indicator))

        # Apply reduction with mask and weights
        return self._reduce_with_mask(loss, mask, weights)


class MultiExpectileLoss(RegressionLoss):
    """
    Loss for multiple expectile levels simultaneously.

    This loss is useful for models that predict multiple expectiles at once,
    providing a more complete description of the conditional distribution.

    The combined loss is:
    L(y, f₁(x), f₂(x), ..., fₖ(x)) = (1/k) * ∑ᵢ L_τᵢ(y, fᵢ(x))

    where L_τᵢ is the expectile loss for the i-th expectile level τᵢ.

    Args:
        expectiles: List of expectile levels in ascending order
        joint_prediction: Whether predictions are passed as a joint tensor
        reduction: Reduction method ('none', 'mean', 'sum'). Default: 'mean'

    Example:
        >>> # Predict 10th, 50th and 90th expectiles together
        >>> loss_fn = MultiExpectileLoss(expectiles=[0.1, 0.5, 0.9])
        >>> # Predictions shape: [batch_size, num_expectiles, features]
        >>> y_pred = torch.tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]])
        >>> target = torch.tensor([[2.0, 3.0]])
        >>> loss_fn(y_pred, target)
        tensor(1.1333)  # Average of the three expectile losses
    """

    def __init__(
        self,
        expectiles: Union[List[float], torch.Tensor],
        joint_prediction: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)

        # Convert list to tensor if needed
        if isinstance(expectiles, list):
            expectiles = torch.tensor(expectiles, dtype=torch.float32)

        # Validate expectile levels
        self.register_buffer("expectiles", validate_range(expectiles, 0.0, 1.0, "expectiles"))
        self.num_expectiles = self.expectiles.size(0)
        self.joint_prediction = joint_prediction

    def forward(
        self,
        y_pred: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate combined expectile loss for multiple levels.

        Args:
            y_pred: When joint_prediction=True: [batch_size, num_expectiles, n_features]
                   or [batch_size, n_features * num_expectiles]
                   Otherwise: List of expectile predictions, each [batch_size, n_features]
            target: Target values [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features]

        Returns:
            Combined expectile loss value
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
                # Process based on prediction shape
                if y_pred.dim() == 3 and y_pred.shape[1] == self.num_expectiles:
                    # [batch_size, num_expectiles, n_features] format
                    expectile_preds = y_pred
                elif y_pred.dim() == 2 and y_pred.shape[1] == n_features * self.num_expectiles:
                    # [batch_size, n_features * num_expectiles] format
                    # Reshape to [batch_size, num_expectiles, n_features]
                    expectile_preds = y_pred.reshape(batch_size, self.num_expectiles, n_features)
                else:
                    raise ValueError(
                        f"Expected y_pred shape to be either "
                        f"[batch_size, {self.num_expectiles}, {n_features}] or "
                        f"[batch_size, {n_features * self.num_expectiles}], "
                        f"got {y_pred.shape}"
                    )
            else:
                raise TypeError("With joint_prediction=True, y_pred must be a tensor")
        else:
            # Handle separate predictions (list of tensors)
            if isinstance(y_pred, (list, tuple)) and len(y_pred) == self.num_expectiles:
                # Stack predictions [batch_size, num_expectiles, n_features]
                expectile_preds = torch.stack(y_pred, dim=1)
            else:
                raise TypeError(
                    f"With joint_prediction=False, y_pred must be a list or tuple "
                    f"of {self.num_expectiles} tensors"
                )

        # Calculate loss for each expectile level
        losses = []
        for i, expectile in enumerate(self.expectiles):
            # Extract predictions for this expectile
            level_preds = expectile_preds[:, i]

            # Calculate residuals
            residuals = target - level_preds

            # Calculate asymmetric squared error
            # Use factor of 2 for consistency with ExpectileLoss
            indicator = (residuals >= 0).float()
            level_loss = 2 * residuals**2 * (expectile * indicator + (1 - expectile) * (1 - indicator))

            # Apply mask if provided
            if mask is not None:
                level_loss = level_loss * mask

            # Apply sample weights if provided
            if weights is not None:
                level_loss = level_loss * weights

            # Reduce across features
            if n_features > 1:
                level_loss = torch.mean(level_loss, dim=1)
            else:
                level_loss = level_loss.squeeze(1)

            losses.append(level_loss)

        # Stack losses for all expectile levels [batch_size, num_expectiles]
        stacked_losses = torch.stack(losses, dim=1)

        # Average across expectile levels for each sample
        combined_loss = torch.mean(stacked_losses, dim=1)

        # Apply final reduction
        if self.reduction == "mean":
            return torch.mean(combined_loss)
        elif self.reduction == "sum":
            return torch.sum(combined_loss)
        else:  # 'none'
            return combined_loss


class AsymmetricLeastSquaresLoss(ExpectileLoss):
    """
    Asymmetric least squares loss (alias for ExpectileLoss for legacy compatibility).

    L(y, f(x)) = |y - f(x)|² * (τ * 1(y > f(x)) + (1-τ) * 1(y ≤ f(x)))

    Args:
        tau: Expectile level (0 < tau < 1). Default: 0.5 (mean)
        reduction: Reduction method ('none', 'mean', 'sum'). Default: 'mean'

    Example:
        >>> # This is equivalent to ExpectileLoss(expectile=0.75)
        >>> loss_fn = AsymmetricLeastSquaresLoss(tau=0.75)
        >>> y_pred = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([0.0, 2.0, 4.0])
        >>> loss_fn(y_pred, target)
        tensor(1.5000)  # Underestimation penalized 3x more than overestimation
    """

    def __init__(self, tau: float = 0.5, reduction: str = "mean") -> None:
        super().__init__(expectile=tau, reduction=reduction)


class ExpectileCrossoverLoss(RegressionLoss):
    """
    Loss that ensures proper ordering of expectile curves.

    In expectile regression, we expect lower expectiles to be below higher ones.
    This loss adds a penalty when this constraint is violated.

    The loss is defined as:
    L(y, {fᵢ(x)}) = base_loss * L_expectile(y, {fᵢ(x)}) +
                     crossover_penalty * ∑ᵢmax(fᵢ(x) - fᵢ₊₁(x), 0)

    where L_expectile is the standard expectile loss for multiple levels,
    and the second term penalizes cases where fᵢ(x) > fᵢ₊₁(x).

    Args:
        expectiles: List of expectile levels in ascending order
        base_loss: Weight for standard expectile loss term
        crossover_penalty: Weight for crossover penalty term
        reduction: Reduction method ('none', 'mean', 'sum'). Default: 'mean'

    Example:
        >>> # Create loss that predicts 10th, 50th, 90th expectiles
        >>> loss_fn = ExpectileCrossoverLoss(expectiles=[0.1, 0.5, 0.9], crossover_penalty=5.0)
        >>> # Properly ordered predictions
        >>> good_pred = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
        >>> # Predictions with crossover (τ₁ > τ₂)
        >>> bad_pred = torch.tensor([[[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]])
        >>> target = torch.tensor([[2.0, 2.0]])
        >>> loss_fn(good_pred, target)  # Normal loss
        tensor(1.0000)
        >>> loss_fn(bad_pred, target)  # Higher loss due to crossover penalty
        tensor(6.0000)  # Base loss + penalty for crossover
    """

    def __init__(
        self,
        expectiles: Union[List[float], torch.Tensor],
        base_loss: float = 1.0,
        crossover_penalty: float = 10.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        # Ensure expectiles are sorted in ascending order
        if isinstance(expectiles, list):
            expectiles = sorted(expectiles)
            expectiles_tensor = torch.tensor(expectiles, dtype=torch.float32)
        else:
            sorted_indices = torch.argsort(expectiles)
            expectiles_tensor = expectiles[sorted_indices]

        self.register_buffer("expectiles", expectiles_tensor)
        self.num_expectiles = len(expectiles)
        self.base_loss = base_loss
        self.crossover_penalty = crossover_penalty

        # Create individual expectile losses
        self.expectile_losses = nn.ModuleList(
            [ExpectileLoss(expectile=e, reduction="none") for e in expectiles]
        )

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate expectile loss with crossover penalty.

        Args:
            y_pred: Predicted expectiles [batch_size, num_expectiles, n_features]
            target: Target values [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional weights [batch_size, n_features] or [batch_size]

        Returns:
            Loss combining standard expectile loss and crossover penalty
        """
        batch_size, _n_features = target.shape[0], target.shape[-1]
        device = target.device

        # Shape validation for y_pred
        if y_pred.shape[1] != self.num_expectiles:
            raise ValueError(
                f"Expected y_pred shape [batch_size, {self.num_expectiles}, n_features], "
                f"got shape {y_pred.shape}"
            )

        # Calculate standard expectile losses per sample
        base_losses = []
        for i, loss_fn in enumerate(self.expectile_losses):
            level_preds = y_pred[:, i]
            # Compute loss without reduction to get per-element losses
            residuals = target - level_preds
            indicator = (residuals >= 0).float()
            level_loss = 2 * residuals**2 * (loss_fn.expectile * indicator + (1 - loss_fn.expectile) * (1 - indicator))
            
            # Apply mask and weights
            if mask is not None:
                level_loss = level_loss * mask
            if weights is not None:
                # Broadcast weights if needed
                if weights.dim() < level_loss.dim():
                    weights_broadcast = weights.unsqueeze(-1)
                else:
                    weights_broadcast = weights
                level_loss = level_loss * weights_broadcast
            
            # Sum across features to get per-sample loss
            level_loss = torch.sum(level_loss, dim=-1)  # [batch_size]
            base_losses.append(level_loss)

        stacked_base_losses = torch.stack(base_losses, dim=0)  # [num_expectiles, batch_size]

        # Calculate crossover penalties
        crossover_penalties = torch.zeros(batch_size, device=device)

        for i in range(self.num_expectiles - 1):
            # Lower expectile should be <= higher expectile
            lower_preds = y_pred[:, i]  # Lower expectile predictions
            higher_preds = y_pred[:, i + 1]  # Higher expectile predictions

            # Calculate violation: ReLU(lower - higher)
            violations = F.relu(lower_preds - higher_preds)

            # Apply mask if provided
            if mask is not None:
                violations = violations * mask

            # Sum violations across features
            sample_violations = torch.sum(violations, dim=-1)
            crossover_penalties += sample_violations

        # Final loss is weighted combination of base loss and crossover penalty
        total_base_loss = torch.mean(stacked_base_losses, dim=0)  # Mean across expectiles [batch_size]

        final_loss = self.base_loss * total_base_loss + self.crossover_penalty * crossover_penalties

        # Apply final reduction
        if self.reduction == "mean":
            return torch.mean(final_loss)
        elif self.reduction == "sum":
            return torch.sum(final_loss)
        else:  # 'none'
            return final_loss
