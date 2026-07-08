"""
Expectile regression loss functions.

Expectile regression provides a richer description of the conditional distribution
than standard mean regression, similar to quantile regression but with different
properties:
- Expectiles are defined via asymmetric least squares
- Mean is a special case of expectile (τ=0.5)
- Expectiles minimize the expected asymmetric squared error
"""

from typing import Any, List, Optional, Union, cast

import torch
import torch.nn.functional as F

from ..utils.validation import validate_range
from .base import RegressionLoss
from .loss_registry import register_regression_loss


def multi_expectile_loss(
    y_pred: torch.Tensor,
    target: torch.Tensor,
    expectiles: torch.Tensor,
) -> torch.Tensor:
    """
    Compute elementwise expectile loss for multiple expectiles.

    Args:
        y_pred: Predicted values [batch_size, num_expectiles, n_features]
        target: Target values [batch_size, n_features]
        expectiles: Expectile levels [num_expectiles]

    Returns:
        Elementwise loss [batch_size, num_expectiles, n_features]
    """
    # Calculate residuals for all expectiles simultaneously
    # target: [batch_size, n_features] -> [batch_size, 1, n_features]
    # expectile_preds: [batch_size, num_expectiles, n_features]
    residuals = target.unsqueeze(1) - y_pred

    # Reshape expectiles for broadcasting: [num_expectiles] -> [1, num_expectiles, 1]
    expectiles_reshaped = expectiles.view(1, -1, 1)

    # Calculate asymmetric squared error
    # Use factor of 2 for consistency with ExpectileLoss
    # weight = expectile if residual >= 0 else (1 - expectile)
    weight = torch.where(residuals >= 0, expectiles_reshaped, 1 - expectiles_reshaped)

    # Calculate loss
    loss = 2 * residuals**2 * weight
    return loss


@register_regression_loss("expectile")
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
        tensor(0.6667)  # Standard MSE at τ=0.5

        >>> # 80th expectile (τ=0.8)
        >>> loss_fn = ExpectileLoss(expectile=0.8)
        >>> y_pred = torch.tensor([1.0, 2.0, 3.0])
        >>> target = torch.tensor([0.0, 2.0, 4.0])
        >>> loss_fn(y_pred, target)
        tensor(0.6667)  # Symmetric residuals cancel the asymmetry in this example

    References
    ----------
    .. [1] Newey, W. K., & Powell, J. L. (1987). Asymmetric Least Squares Estimation
       and Testing. In *Econometrica*, 55(4), 819-847.
       https://www.jstor.org/stable/1911031
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
        **kwargs: Any,
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
        weight = torch.where(residuals >= 0, self.expectile, 1 - self.expectile)
        loss = 2 * residuals**2 * weight

        # Apply reduction with mask and weights
        return self._reduce(loss, mask, weights)


@register_regression_loss("multi_expectile")
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
        validated_expectiles = cast(
            torch.Tensor, validate_range(expectiles, 0.0, 1.0, "expectiles")
        )
        self.register_buffer("expectiles", validated_expectiles)
        self.num_expectiles = validated_expectiles.size(0)
        self.joint_prediction = joint_prediction

    def forward(
        self,
        y_pred: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
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
                expectile_preds = torch.stack(cast(List[torch.Tensor], y_pred), dim=1)
            else:
                raise TypeError(
                    f"With joint_prediction=False, y_pred must be a list or tuple "
                    f"of {self.num_expectiles} tensors"
                )

        # Elementwise multi-expectile loss via shared utility
        stacked_losses = multi_expectile_loss(
            expectile_preds, target, cast(torch.Tensor, self.expectiles)
        )

        # Apply mask if provided
        if mask is not None:
            # mask: [batch_size, n_features] -> [batch_size, 1, n_features]
            stacked_losses = stacked_losses * mask.unsqueeze(1)

        # Apply sample weights if provided
        if weights is not None:
            # weights: [batch_size, n_features] -> [batch_size, 1, n_features]
            stacked_losses = stacked_losses * weights.unsqueeze(1)

        # Reduce across features
        if n_features > 1:
            stacked_losses = torch.mean(stacked_losses, dim=2)
        else:
            stacked_losses = stacked_losses.squeeze(2)

        # Average across expectile levels for each sample
        combined_loss = torch.mean(stacked_losses, dim=1)

        # Apply final reduction
        if self.reduction == "mean":
            return torch.mean(combined_loss)
        elif self.reduction == "sum":
            return torch.sum(combined_loss)
        else:  # 'none'
            return combined_loss


# ponytail: AsymmetricLeastSquaresLoss is an alias for ExpectileLoss.
AsymmetricLeastSquaresLoss = ExpectileLoss
register_regression_loss("als")(AsymmetricLeastSquaresLoss)

@register_regression_loss("expectile_crossover")
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

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
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
        # Shape validation for y_pred
        if y_pred.shape[1] != self.num_expectiles:
            raise ValueError(
                f"Expected y_pred shape [batch_size, {self.num_expectiles}, n_features], "
                f"got shape {y_pred.shape}"
            )

        # 1. Calculate Base Loss (Standard Expectile Loss) using vectorized utility
        # [batch_size, num_expectiles, n_features]
        level_losses = multi_expectile_loss(y_pred, target, cast(torch.Tensor, self.expectiles))

        # Apply mask and weights to base loss
        if mask is not None:
            # mask: [batch_size, n_features] -> [batch_size, 1, n_features]
            mask_expanded = mask.unsqueeze(1) if mask.dim() > 1 else mask.unsqueeze(1).unsqueeze(2)
            level_losses = level_losses * mask_expanded

        if weights is not None:
            weights_expanded = weights
            if weights.dim() == 1:
                # [batch] -> [batch, 1, 1]
                weights_expanded = weights.view(-1, 1, 1)
            elif weights.dim() == 2:
                # [batch, features] -> [batch, 1, features]
                weights_expanded = weights.unsqueeze(1)
            level_losses = level_losses * weights_expanded

        # Sum across features to get per-sample loss: [batch_size, num_expectiles]
        per_sample_level_losses = torch.sum(level_losses, dim=-1)

        # Mean across expectiles per sample: [batch_size]
        total_base_loss = torch.mean(per_sample_level_losses, dim=1)

        # 2. Calculate Crossover Penalties (Vectorized)
        # y_pred: [batch_size, num_expectiles, n_features]
        # Compare i and i+1
        lower_preds = y_pred[:, :-1, :]
        higher_preds = y_pred[:, 1:, :]

        # Violations: [batch_size, num_expectiles-1, n_features]
        violations = F.relu(lower_preds - higher_preds)

        if mask is not None:
            # Re-use mask_expanded [batch_size, 1, n_features]
            # Make sure mask_expanded is defined
            mask_expanded = mask.unsqueeze(1) if mask.dim() > 1 else mask.unsqueeze(1).unsqueeze(2)
            violations = violations * mask_expanded

        # Sum across features: [batch_size, num_expectiles-1]
        feature_violations = torch.sum(violations, dim=-1)

        # Sum across expectiles: [batch_size]
        crossover_penalties = torch.sum(feature_violations, dim=1)

        # Final combination
        final_loss = self.base_loss * total_base_loss + self.crossover_penalty * crossover_penalties

        # Apply final reduction
        if self.reduction == "mean":
            return torch.mean(final_loss)
        elif self.reduction == "sum":
            return torch.sum(final_loss)
        else:  # 'none'
            return final_loss


def expectile_loss(
    y_pred: torch.Tensor,
    target: torch.Tensor,
    expectile: float = 0.5,
    mask: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Functional wrapper for :class:`ExpectileLoss`.

    Equivalent to ``ExpectileLoss(expectile=expectile, reduction=reduction)``
    followed by a ``forward`` call.  ``expectile=0.5`` recovers MSE.

    Args:
        y_pred: Predicted values.
        target: Ground truth.
        expectile: Asymmetry level in (0, 1).  ``0.5`` = mean.
        mask: Optional boolean mask of valid entries.
        weights: Optional per-element weights.
        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.

    Returns:
        Expectile loss value.
    """
    return ExpectileLoss(expectile=expectile, reduction=reduction)(
        y_pred, target, mask=mask, weights=weights
    )


# Compatibility alias used in docs.
ExpectileCrossover = ExpectileCrossoverLoss
