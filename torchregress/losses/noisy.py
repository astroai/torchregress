"""
Loss functions for regression with noisy labels.

This module provides loss functions designed to handle noisy or corrupted labels
in regression tasks. These methods are adapted from classification literature but
specifically designed for continuous target variables.

References:
    - Li et al. "Learning to Learn from Noisy Labeled Data" (CVPR 2019)
    - Han et al. "Co-teaching: Robust Training of DNNs with Extremely Noisy Labels" (NeurIPS 2018)
    - Chen et al. "Understanding and Utilizing Deep Neural Networks Trained with Noisy Labels"
      (ICML 2019)
"""

from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .base import RegressionLoss
from .loss_registry import register_regression_loss


@register_regression_loss("noise_adaptive")
class NoiseAdaptiveLoss(RegressionLoss):
    """
    Learns sample-specific noise levels via meta-learning.

    This loss function learns a confidence weight for each training sample,
    automatically downweighting noisy labels during training. The weights are
    learned parameters that are updated based on the loss behavior.

    The method maintains a learnable weight for each sample in the dataset,
    which is optimized to identify and downweight noisy labels.

    Args:
        n_samples: Total number of samples in the dataset (for initializing weights)
        base_loss: Base loss function ('mse', 'mae', 'huber'). Default: 'mse'
        initial_weight: Initial value for sample weights (0-1). Default: 1.0
        reduction: Loss reduction method. Default: 'mean'

    Example:
        >>> import torch
        >>> from torchregress.losses import NoiseAdaptiveLoss
        >>>
        >>> # Create loss for dataset with 1000 samples
        >>> loss_fn = NoiseAdaptiveLoss(n_samples=1000)
        >>>
        >>> # Training loop
        >>> for epoch in range(n_epochs):
        >>>     for batch_idx, (x, y, indices) in enumerate(train_loader):
        >>>         # Forward pass
        >>>         y_pred = model(x)
        >>>         # Loss computation with sample indices
        >>>         loss = loss_fn(y_pred, y, sample_indices=indices)
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>
        >>>         # Also update the sample weights
        >>>         weight_optimizer.step()
        >>>
        >>> # Get learned weights to identify noisy samples
        >>> sample_weights = loss_fn.get_sample_weights()
        >>> noisy_indices = torch.where(sample_weights < 0.5)[0]

    Notes:
        - Requires sample indices to be passed during training
        - Sample weights should be optimized alongside model parameters
        - Works best with a separate optimizer for sample weights (lower learning rate)
    """

    def __init__(
        self,
        n_samples: int,
        base_loss: str = "mse",
        initial_weight: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.n_samples = n_samples
        self.base_loss = base_loss.lower()

        if self.base_loss not in ["mse", "mae", "huber"]:
            raise ValueError(f"base_loss must be 'mse', 'mae', or 'huber', got {base_loss}")

        # Learnable per-sample weights (logits, will be passed through sigmoid)
        # Initialize to small positive values so sigmoid gives ~initial_weight
        init_logit = torch.log(torch.tensor(initial_weight / (1 - initial_weight + 1e-7)))
        self.sample_weight_logits = nn.Parameter(torch.full((n_samples,), init_logit))

    def get_sample_weights(self) -> Tensor:
        """
        Get current sample weights (confidence scores).

        Returns:
            Tensor of shape [n_samples] with values in [0, 1]
            Lower values indicate samples identified as likely noisy
        """
        return torch.sigmoid(self.sample_weight_logits)

    def _compute_base_loss(self, y_pred: Tensor, target: Tensor) -> Tensor:
        """Compute base loss without reduction."""
        if self.base_loss == "mse":
            return (y_pred - target) ** 2
        elif self.base_loss == "mae":
            return torch.abs(y_pred - target)
        elif self.base_loss == "huber":
            # Huber loss with delta=1.0
            diff = torch.abs(y_pred - target)
            return torch.where(diff < 1.0, 0.5 * diff**2, diff - 0.5)
        else:
            raise ValueError(f"Unknown base_loss: {self.base_loss}")

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        sample_indices: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Compute noise-adaptive loss.

        Args:
            y_pred: Model predictions [..., features]
            target: Ground truth values [..., features]
            sample_indices: Indices of samples in the dataset [batch_size]
                REQUIRED for this loss function
            mask: Optional mask for missing values
            weights: Optional additional sample weights (multiplied with learned weights)
            **kwargs: Additional arguments

        Returns:
            Weighted loss value

        Raises:
            ValueError: If sample_indices is not provided
        """
        if sample_indices is None:
            raise ValueError(
                "NoiseAdaptiveLoss requires sample_indices to be passed. "
                "Your DataLoader must return (x, y, indices) tuples."
            )

        self._validate_inputs(y_pred, target, mask)

        # Compute base loss (per-sample, per-feature)
        base_loss = self._compute_base_loss(y_pred, target)

        # Get learned weights for these samples
        sample_weights = torch.sigmoid(self.sample_weight_logits[sample_indices])

        # Expand weights to match loss shape if needed
        if base_loss.dim() > 1:
            # [batch_size] -> [batch_size, 1, 1, ...]
            for _ in range(base_loss.dim() - 1):
                sample_weights = sample_weights.unsqueeze(-1)
            sample_weights = sample_weights.expand_as(base_loss)

        # Apply learned weights
        weighted_loss = base_loss * sample_weights

        # Combine with optional external weights
        if weights is not None:
            if weights.dim() > 1:
                weighted_loss = weighted_loss * weights
            else:
                # Expand weights if needed
                w = weights
                for _ in range(weighted_loss.dim() - 1):
                    w = w.unsqueeze(-1)
                weighted_loss = weighted_loss * w.expand_as(weighted_loss)

        return self._reduce_with_mask(weighted_loss, mask, None)


@register_regression_loss("co_teaching")
class CoTeachingLoss(RegressionLoss):
    """
    Co-teaching loss for training with noisy labels in regression.

    Co-teaching trains two networks simultaneously, where each network selects
    small-loss samples (assumed clean) for the other network to learn from.
    The forget rate determines what fraction of samples to exclude (assumed noisy).

    This is adapted from the classification paper "Co-teaching: Robust Training
    of DNNs with Extremely Noisy Labels" for regression tasks.

    Args:
        forget_rate: Fraction of samples to exclude (assumed noisy). Default: 0.2
        base_loss: Base loss function ('mse', 'mae', 'huber'). Default: 'mse'
        num_gradual: Number of epochs to gradually increase forget rate. Default: 10
        reduction: Loss reduction method. Default: 'mean'

    Example:
        >>> from torchregress.losses import CoTeachingLoss
        >>>
        >>> # Create two models
        >>> model1 = MyModel()
        >>> model2 = MyModel()
        >>> loss_fn = CoTeachingLoss(forget_rate=0.2)
        >>>
        >>> # Training loop
        >>> for epoch in range(n_epochs):
        >>>     for x, y in train_loader:
        >>>         # Forward pass for both networks
        >>>         y_pred1 = model1(x)
        >>>         y_pred2 = model2(x)
        >>>
        >>>         # Co-teaching loss (each network teaches the other)
        >>>         loss1, loss2 = loss_fn(y_pred1, y_pred2, y, epoch=epoch)
        >>>
        >>>         # Separate backward passes
        >>>         optimizer1.zero_grad()
        >>>         loss1.backward()
        >>>         optimizer1.step()
        >>>
        >>>         optimizer2.zero_grad()
        >>>         loss2.backward()
        >>>         optimizer2.step()

    Notes:
        - Requires two networks to be trained simultaneously
        - Returns two losses: one for each network
        - Gradually increases forget rate during num_gradual epochs
        - Small-loss samples are assumed to be clean (robust to label noise)
    """

    def __init__(
        self,
        forget_rate: float = 0.2,
        base_loss: str = "mse",
        num_gradual: int = 10,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.forget_rate = forget_rate
        self.base_loss = base_loss.lower()
        self.num_gradual = num_gradual

        if not 0.0 <= forget_rate <= 1.0:
            raise ValueError(f"forget_rate must be in [0, 1], got {forget_rate}")

        if self.base_loss not in ["mse", "mae", "huber"]:
            raise ValueError(f"base_loss must be 'mse', 'mae', or 'huber', got {base_loss}")

    def _compute_base_loss(self, y_pred: Tensor, target: Tensor) -> Tensor:
        """Compute base loss (per-sample)."""
        if self.base_loss == "mse":
            loss = (y_pred - target) ** 2
        elif self.base_loss == "mae":
            loss = torch.abs(y_pred - target)
        elif self.base_loss == "huber":
            diff = torch.abs(y_pred - target)
            loss = torch.where(diff < 1.0, 0.5 * diff**2, diff - 0.5)
        else:
            raise ValueError(f"Unknown base_loss: {self.base_loss}")

        # Reduce to per-sample loss
        if loss.dim() > 1:
            loss = loss.mean(dim=tuple(range(1, loss.dim())))

        return loss

    def _get_forget_rate(self, epoch: int) -> float:
        """Calculate current forget rate with gradual increase."""
        if epoch < self.num_gradual:
            # Gradually increase forget rate
            return self.forget_rate * (epoch + 1) / self.num_gradual
        return self.forget_rate

    def forward(
        self,
        y_pred1: Tensor,
        y_pred2: Tensor,
        target: Tensor,
        epoch: int = 0,
        mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute co-teaching loss for two networks.

        Args:
            y_pred1: Predictions from network 1 [batch_size, ...]
            y_pred2: Predictions from network 2 [batch_size, ...]
            target: Ground truth values [batch_size, ...]
            epoch: Current training epoch (for gradual forget rate). Default: 0
            mask: Optional mask for missing values
            **kwargs: Additional arguments

        Returns:
            Tuple of (loss1, loss2):
                - loss1: Loss for network 1 (based on samples selected by network 2)
                - loss2: Loss for network 2 (based on samples selected by network 1)

        Notes:
            - Network 1 learns from small-loss samples of network 2
            - Network 2 learns from small-loss samples of network 1
        """
        self._validate_inputs(y_pred1, target, mask)
        self._validate_inputs(y_pred2, target, mask)

        # Compute per-sample losses for both networks
        loss1_vec = self._compute_base_loss(y_pred1, target)
        loss2_vec = self._compute_base_loss(y_pred2, target)

        # Calculate how many samples to remember (not forget)
        forget_rate = self._get_forget_rate(epoch)
        n_remember = max(1, int((1 - forget_rate) * len(loss1_vec)))

        # Network 1 learns from small-loss samples selected by network 2
        _, idx2 = torch.topk(loss2_vec, n_remember, largest=False)
        loss_1_update = loss1_vec[idx2]

        # Network 2 learns from small-loss samples selected by network 1
        _, idx1 = torch.topk(loss1_vec, n_remember, largest=False)
        loss_2_update = loss2_vec[idx1]

        # Apply reduction
        if self.reduction == "mean":
            return loss_1_update.mean(), loss_2_update.mean()
        elif self.reduction == "sum":
            return loss_1_update.sum(), loss_2_update.sum()
        else:  # 'none'
            return loss_1_update, loss_2_update


@register_regression_loss("rent")
class RENTLoss(RegressionLoss):
    """
    Robust Ensemble Training (RENT) loss for noisy labels.

    Uses ensemble disagreement to identify noisy labels. Samples with high
    disagreement among ensemble members are downweighted, as they are likely
    to have noisy labels or be difficult/ambiguous examples.

    Args:
        ensemble_size: Expected number of models in ensemble. Default: 5
        noise_threshold: Threshold for downweighting based on disagreement. Default: 2.0
        base_loss: Base loss function ('mse', 'mae', 'huber'). Default: 'mse'
        reduction: Loss reduction method. Default: 'mean'

    Example:
        >>> from torchregress.losses import RENTLoss
        >>> from torchregress.ensemble import DeepEnsemble
        >>>
        >>> # Create ensemble
        >>> ensemble = DeepEnsemble(base_model=MyModel(), ensemble_size=5)
        >>> loss_fn = RENTLoss(ensemble_size=5)
        >>>
        >>> # Training loop
        >>> for x, y in train_loader:
        >>>     # Get predictions from all ensemble members
        >>>     ensemble_preds = ensemble.forward_all(x)  # [ensemble_size, batch, features]
        >>>
        >>>     # Compute RENT loss
        >>>     loss = loss_fn(ensemble_preds, y)
        >>>     loss.backward()
        >>>     optimizer.step()

    Notes:
        - Requires predictions from all ensemble members
        - Automatically identifies and downweights noisy samples
        - Works well when ensemble members disagree on noisy samples
        - No need for sample indices or separate networks
    """

    def __init__(
        self,
        ensemble_size: int = 5,
        noise_threshold: float = 2.0,
        base_loss: str = "mse",
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.ensemble_size = ensemble_size
        self.noise_threshold = noise_threshold
        self.base_loss = base_loss.lower()

        if self.base_loss not in ["mse", "mae", "huber"]:
            raise ValueError(f"base_loss must be 'mse', 'mae', or 'huber', got {base_loss}")

    def _compute_base_loss(self, y_pred: Tensor, target: Tensor) -> Tensor:
        """Compute base loss without reduction."""
        if self.base_loss == "mse":
            return (y_pred - target) ** 2
        elif self.base_loss == "mae":
            return torch.abs(y_pred - target)
        elif self.base_loss == "huber":
            diff = torch.abs(y_pred - target)
            return torch.where(diff < 1.0, 0.5 * diff**2, diff - 0.5)
        else:
            raise ValueError(f"Unknown base_loss: {self.base_loss}")

    def forward(
        self,
        ensemble_preds: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Compute RENT loss using ensemble disagreement.

        Args:
            ensemble_preds: Predictions from all ensemble members
                Shape: [ensemble_size, batch_size, ...] or [batch_size, ensemble_size, ...]
            target: Ground truth values [batch_size, ...]
            mask: Optional mask for missing values
            weights: Optional additional sample weights
            **kwargs: Additional arguments

        Returns:
            Weighted loss based on ensemble disagreement

        Raises:
            ValueError: If ensemble_preds doesn't have correct shape
        """
        # Ensure ensemble_preds has shape [ensemble_size, batch_size, ...]
        if (
            ensemble_preds.shape[0] != self.ensemble_size
            and ensemble_preds.shape[1] == self.ensemble_size
        ):
            # Transpose if needed: [batch, ensemble, ...] -> [ensemble, batch, ...]
            ensemble_preds = ensemble_preds.transpose(0, 1)

        if ensemble_preds.shape[0] != self.ensemble_size:
            raise ValueError(
                f"Expected ensemble_preds.shape[0] == {self.ensemble_size}, "
                f"got {ensemble_preds.shape[0]}"
            )

        # Calculate ensemble mean and variance (disagreement)
        mean_pred = ensemble_preds.mean(dim=0)  # [batch_size, ...]
        ensemble_var = ensemble_preds.var(dim=0, unbiased=True)  # [batch_size, ...]

        # Compute residuals
        residuals = torch.abs(target - mean_pred)

        # Normalized disagreement: high when ensemble disagrees relative to error
        # Add epsilon to avoid division by zero
        normalized_disagreement = ensemble_var / (residuals + 1e-8)

        # Compute weights: downweight samples with high normalized disagreement
        # Using exponential decay
        sample_weights = torch.exp(-self.noise_threshold * normalized_disagreement)

        # Compute base loss
        base_loss = self._compute_base_loss(mean_pred, target)

        # Apply weights
        weighted_loss = base_loss * sample_weights

        # Combine with optional external weights
        if weights is not None:
            weighted_loss = weighted_loss * weights

        return self._reduce_with_mask(weighted_loss, mask, None)
