"""
Loss functions for imbalanced regression.

This module provides loss functions designed for regression tasks where the
target distribution is imbalanced (e.g., rare extreme values, dense regions
vs sparse regions). Special care is taken to preserve calibration properties.

References:
    - Yang et al. "Delving into Deep Imbalanced Regression" (ICML 2021)
    - Steininger et al. "Density-based weighting for imbalanced regression" (ML 2021)

Warning:
    Some imbalanced regression methods can break calibration. Use post-hoc
    calibration validation when using these losses. See documentation for details.
"""

from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from .base import RegressionLoss
from .loss_registry import register_regression_loss


@register_regression_loss("density_weighted")
class DensityWeightedLoss(RegressionLoss):
    """
    Density-weighted loss for imbalanced regression (calibration-safe).

    Weights samples inversely proportional to their local density in target space.
    This upweights rare/extreme values while downweighting common values.

    This method is SAFE for calibration because it only reweights the training
    samples without changing the conditional distribution p(y|x).

    Args:
        kernel_width: Bandwidth for kernel density estimation. Default: 0.5
            Smaller values = more local density estimation
        base_loss: Base loss function ('mse', 'mae', 'huber'). Default: 'mse'
        reweight_factor: Strength of reweighting (0-1). Default: 1.0
            0 = no reweighting, 1 = full inverse density weighting
        reduction: Loss reduction method. Default: 'mean'

    Example:
        >>> from torchregress.losses import DensityWeightedLoss
        >>> import torch
        >>>
        >>> # Create loss
        >>> loss_fn = DensityWeightedLoss(kernel_width=0.5)
        >>>
        >>> # Fit density on training targets
        >>> train_targets = torch.cat([y for _, y in train_loader])
        >>> loss_fn.fit_density(train_targets)
        >>>
        >>> # Training loop with sample indices
        >>> for x, y, indices in train_loader:
        >>>     y_pred = model(x)
        >>>     loss = loss_fn(y_pred, y, sample_indices=indices)
        >>>     loss.backward()
        >>>     optimizer.step()

    Notes:
        - Requires fitting density on training data before use
        - Requires sample indices during training to retrieve precomputed weights
        - Preserves calibration (safe method)
        - Alternative: Can pass target values directly without indices
    """

    def __init__(
        self,
        kernel_width: float = 0.5,
        base_loss: str = "mse",
        reweight_factor: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.kernel_width = kernel_width
        self.base_loss = base_loss.lower()
        self.reweight_factor = reweight_factor

        if not 0.0 <= reweight_factor <= 1.0:
            raise ValueError(f"reweight_factor must be in [0, 1], got {reweight_factor}")

        if self.base_loss not in ["mse", "mae", "huber"]:
            raise ValueError(f"base_loss must be 'mse', 'mae', or 'huber', got {base_loss}")

        # Will store precomputed density weights
        self.density_weights: Optional[Tensor] = None
        self._train_targets: Optional[Tensor] = None

    def fit_density(self, train_targets: Tensor) -> None:
        """
        Estimate target density and compute inverse density weights.

        Args:
            train_targets: All training targets [n_samples, features]
                Shape can be [n_samples] for single-output regression

        Example:
            >>> # Collect all training targets
            >>> all_targets = []
            >>> for _, y in train_loader:
            >>>     all_targets.append(y)
            >>> train_targets = torch.cat(all_targets)
            >>>
            >>> # Fit density
            >>> loss_fn.fit_density(train_targets)
        """
        from sklearn.neighbors import KernelDensity

        # Store targets for potential reuse
        self._train_targets = train_targets.detach().cpu()

        # Reshape if needed
        if train_targets.dim() == 1:
            targets_np = train_targets.cpu().numpy().reshape(-1, 1)
        else:
            targets_np = train_targets.cpu().numpy()

        # Fit kernel density estimator
        kde = KernelDensity(bandwidth=self.kernel_width, kernel="gaussian")
        kde.fit(targets_np)

        # Compute log density for each sample
        log_density = kde.score_samples(targets_np)

        # Convert to inverse density weights
        # Inverse density: exp(-log_density)
        inv_density = np.exp(-log_density)

        # Normalize weights to have mean 1.0 (preserves expected loss scale)
        inv_density = inv_density / inv_density.mean()

        # Apply reweight factor: w = 1 + factor * (inv_density - 1)
        # factor=0 → uniform weights, factor=1 → full inverse density
        weights = 1.0 + self.reweight_factor * (inv_density - 1.0)

        self.density_weights = torch.tensor(weights, dtype=torch.float32)

    def _compute_density_weight(self, target: Tensor) -> Tensor:
        """
        Compute density weight for given target values on-the-fly.

        This is used when sample_indices are not provided.
        """
        if self._train_targets is None:
            raise ValueError("Must call fit_density() before computing weights")

        from sklearn.neighbors import KernelDensity

        # Reshape target if needed
        if target.dim() == 1:
            target_np = target.detach().cpu().numpy().reshape(-1, 1)
        else:
            target_np = target.detach().cpu().numpy()

        # Reshape train targets
        if self._train_targets.dim() == 1:
            train_np = self._train_targets.numpy().reshape(-1, 1)
        else:
            train_np = self._train_targets.numpy()

        # Fit KDE and compute density
        kde = KernelDensity(bandwidth=self.kernel_width, kernel="gaussian")
        kde.fit(train_np)
        log_density = kde.score_samples(target_np)

        # Inverse density weights
        inv_density = np.exp(-log_density)
        inv_density = inv_density / np.mean(inv_density)
        weights = 1.0 + self.reweight_factor * (inv_density - 1.0)

        return torch.tensor(weights, dtype=target.dtype, device=target.device)

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
        y_pred: Tensor,
        target: Tensor,
        sample_indices: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Compute density-weighted loss.

        Args:
            y_pred: Model predictions [batch_size, ...]
            target: Ground truth values [batch_size, ...]
            sample_indices: Optional indices for precomputed weights [batch_size]
            mask: Optional mask for missing values
            weights: Optional additional sample weights
            **kwargs: Additional arguments

        Returns:
            Density-weighted loss

        Notes:
            - If sample_indices provided: uses precomputed weights (faster)
            - If not provided: computes weights on-the-fly from target values (slower)
        """
        if self.density_weights is None and self._train_targets is None:
            raise ValueError(
                "Must call fit_density() before using DensityWeightedLoss. "
                "Example: loss_fn.fit_density(train_targets)"
            )

        self._validate_inputs(y_pred, target, mask)

        # Get density weights
        if sample_indices is not None:
            # Use precomputed weights
            if self.density_weights is None:
                raise ValueError("fit_density() must be called before using sample_indices")
            density_w = self.density_weights[sample_indices].to(y_pred.device)
        else:
            # Compute weights on-the-fly from target values
            density_w = self._compute_density_weight(target)

        # Compute base loss
        base_loss = self._compute_base_loss(y_pred, target)

        # Expand weights if needed to match loss shape
        if base_loss.dim() > 1 and density_w.dim() == 1:
            for _ in range(base_loss.dim() - 1):
                density_w = density_w.unsqueeze(-1)
            density_w = density_w.expand_as(base_loss)

        # Apply density weights
        weighted_loss = base_loss * density_w

        # Combine with optional external weights
        if weights is not None:
            weighted_loss = weighted_loss * weights

        return self._reduce_with_mask(weighted_loss, mask, None)


@register_regression_loss("lds")
class LDSLoss(RegressionLoss):
    """
    Label Distribution Smoothing (LDS) loss for imbalanced regression.

    Smooths the label distribution using kernel smoothing and reweights samples
    based on the effective label frequency. This addresses imbalance but CAN
    affect calibration.

    WARNING: This method can break calibration because it changes the training
    targets through smoothing. Always validate calibration after training and
    consider post-hoc calibration methods.

    Args:
        kernel: Kernel type for smoothing ('gaussian', 'triang', 'laplace'). Default: 'gaussian'
        kernel_width: Bandwidth for kernel smoothing. Default: 2.0
        reweight_factor: Strength of reweighting (0-1). Default: 1.0
        base_loss: Base loss function ('mse', 'mae', 'huber'). Default: 'mse'
        reduction: Loss reduction method. Default: 'mean'

    Example:
        >>> from torchregress.losses import LDSLoss
        >>>
        >>> # Create loss with calibration awareness
        >>> loss_fn = LDSLoss(kernel='gaussian', kernel_width=2.0)
        >>>
        >>> # Fit on training data
        >>> train_targets = torch.cat([y for _, y in train_loader])
        >>> loss_fn.fit(train_targets)
        >>>
        >>> # Training
        >>> for x, y in train_loader:
        >>>     y_pred = model(x)
        >>>     loss = loss_fn(y_pred, y)
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>
        >>> # IMPORTANT: Validate calibration after training
        >>> from torchregress.metrics import calibration_error
        >>> cal_err = calibration_error(model, cal_loader)
        >>> print(f"Calibration error: {cal_err:.4f}")
        >>>
        >>> # If calibration is poor, apply post-hoc calibration
        >>> # (e.g., temperature scaling, isotonic regression)

    Notes:
        - Requires fitting on training data before use
        - CAN BREAK CALIBRATION - always validate!
        - Best used with post-hoc calibration methods
        - More aggressive than DensityWeightedLoss
    """

    def __init__(
        self,
        kernel: str = "gaussian",
        kernel_width: float = 2.0,
        reweight_factor: float = 1.0,
        base_loss: str = "mse",
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.kernel = kernel.lower()
        self.kernel_width = kernel_width
        self.reweight_factor = reweight_factor
        self.base_loss = base_loss.lower()

        if self.kernel not in ["gaussian", "triang", "laplace"]:
            raise ValueError(f"kernel must be 'gaussian', 'triang', or 'laplace', got {kernel}")

        if self.base_loss not in ["mse", "mae", "huber"]:
            raise ValueError(f"base_loss must be 'mse', 'mae', or 'huber', got {base_loss}")

        if not 0.0 <= reweight_factor <= 1.0:
            raise ValueError(f"reweight_factor must be in [0, 1], got {reweight_factor}")

        self.lds_weights: Optional[Tensor] = None
        self._train_targets: Optional[Tensor] = None

    def _get_kernel_window(self, kernel_width: float) -> np.ndarray:
        """Generate kernel window for smoothing."""
        # Create symmetric kernel window
        half_width = int(np.ceil(kernel_width * 3))  # 3 sigma for gaussian
        x = np.arange(-half_width, half_width + 1, dtype=np.float32)

        if self.kernel == "gaussian":
            window = np.exp(-0.5 * (x / kernel_width) ** 2)
        elif self.kernel == "triang":
            window = np.maximum(0, 1 - np.abs(x) / kernel_width)
        elif self.kernel == "laplace":
            window = np.exp(-np.abs(x) / kernel_width)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        # Normalize
        window = window / window.sum()
        return window

    def fit(self, train_targets: Tensor, n_bins: int = 100) -> None:
        """
        Compute LDS weights from training targets.

        Args:
            train_targets: All training targets [n_samples] or [n_samples, 1]
            n_bins: Number of bins for discretization. Default: 100

        Example:
            >>> all_targets = torch.cat([y for _, y in train_loader])
            >>> loss_fn.fit(all_targets)
        """
        self._train_targets = train_targets.detach().cpu()

        # Flatten if needed
        targets_np = train_targets.cpu().numpy().flatten()

        # Bin the continuous targets
        min_val, max_val = targets_np.min(), targets_np.max()
        bins = np.linspace(min_val, max_val, n_bins + 1)
        bin_indices = np.digitize(targets_np, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Count samples in each bin
        bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(np.float32)

        # Apply kernel smoothing to bin counts
        kernel_window = self._get_kernel_window(self.kernel_width)
        smoothed_counts = np.convolve(bin_counts, kernel_window, mode="same")

        # Avoid division by zero
        smoothed_counts = np.maximum(smoothed_counts, 1e-3)

        # Compute inverse frequency weights
        inv_freq = 1.0 / smoothed_counts

        # Normalize to mean 1.0
        inv_freq = inv_freq / inv_freq.mean()

        # Apply reweight factor
        weights = 1.0 + self.reweight_factor * (inv_freq - 1.0)

        # Map weights back to samples
        sample_weights = weights[bin_indices]

        self.lds_weights = torch.tensor(sample_weights, dtype=torch.float32)
        self._bins = bins
        self._weights_per_bin = torch.tensor(weights, dtype=torch.float32)

    def _compute_weight_for_target(self, target: Tensor) -> Tensor:
        """Compute LDS weight for given target values."""
        if self._weights_per_bin is None or not hasattr(self, "_bins"):
            raise ValueError("Must call fit() before computing weights")

        # Digitize targets
        targets_np = target.detach().cpu().numpy().flatten()
        bin_indices = np.digitize(targets_np, self._bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(self._weights_per_bin) - 1)

        # Get weights
        weights = self._weights_per_bin[bin_indices]
        return weights.to(target.device)

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
        y_pred: Tensor,
        target: Tensor,
        sample_indices: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Compute LDS-weighted loss.

        Args:
            y_pred: Model predictions [batch_size, ...]
            target: Ground truth values [batch_size, ...]
            sample_indices: Optional indices for precomputed weights [batch_size]
            mask: Optional mask for missing values
            weights: Optional additional sample weights
            **kwargs: Additional arguments

        Returns:
            LDS-weighted loss

        Warning:
            This loss can break calibration. Validate calibration after training!
        """
        if self.lds_weights is None:
            raise ValueError(
                "Must call fit() before using LDSLoss. " "Example: loss_fn.fit(train_targets)"
            )

        self._validate_inputs(y_pred, target, mask)

        # Get LDS weights
        if sample_indices is not None:
            # Use precomputed weights
            lds_w = self.lds_weights[sample_indices].to(y_pred.device)
        else:
            # Compute from target values
            lds_w = self._compute_weight_for_target(target)

        # Compute base loss
        base_loss = self._compute_base_loss(y_pred, target)

        # Expand weights if needed
        if base_loss.dim() > 1 and lds_w.dim() == 1:
            for _ in range(base_loss.dim() - 1):
                lds_w = lds_w.unsqueeze(-1)
            lds_w = lds_w.expand_as(base_loss)

        # Apply weights
        weighted_loss = base_loss * lds_w

        # Combine with optional external weights
        if weights is not None:
            weighted_loss = weighted_loss * weights

        return self._reduce_with_mask(weighted_loss, mask, None)


@register_regression_loss("focal_r")
class FocalRLoss(RegressionLoss):
    """
    Focal-R: Focal loss adapted for regression tasks.

    Applies adaptive loss scaling based on prediction error magnitude,
    emphasizing harder samples (larger errors) during training. This helps
    the model focus on difficult examples in imbalanced regression.

    The loss is: L = sigmoid(beta * |error|)^gamma * base_loss

    Args:
        beta: Error scaling factor. Default: 0.2
            Controls how errors are mapped to difficulty scores
        gamma: Focus parameter. Default: 1.0
            Higher values increase focus on hard samples
        base_loss: Base loss function ('mse', 'mae', 'huber'). Default: 'mse'
        reduction: Loss reduction method. Default: 'mean'

    Example:
        >>> from torchregress.losses import FocalRLoss
        >>> loss_fn = FocalRLoss(beta=0.2, gamma=1.0)
        >>> loss = loss_fn(y_pred, y_true)

    Reference:
        Yang et al. "Delving into Deep Imbalanced Regression" (ICML 2021)
    """

    def __init__(
        self,
        beta: float = 0.2,
        gamma: float = 1.0,
        base_loss: str = "mse",
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.beta = beta
        self.gamma = gamma
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
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute Focal-R loss.

        Args:
            y_pred: Model predictions [batch_size, ...]
            target: Ground truth values [batch_size, ...]
            mask: Optional mask for missing values
            weights: Optional sample weights

        Returns:
            Focal-R weighted loss
        """
        self._validate_inputs(y_pred, target, mask)

        # Compute absolute error for difficulty weighting
        abs_error = torch.abs(y_pred - target)

        # Focal weight: sigmoid(beta * |error|)^gamma
        # This upweights samples with larger errors
        focal_weight = torch.sigmoid(self.beta * abs_error).pow(self.gamma)

        # Compute base loss
        base_loss = self._compute_base_loss(y_pred, target)

        # Apply focal weighting
        weighted_loss = focal_weight * base_loss

        # Apply optional external weights
        if weights is not None:
            weighted_loss = weighted_loss * weights

        return self._reduce_with_mask(weighted_loss, mask, None)


@register_regression_loss("balanced_mse")
class BalancedMSELoss(RegressionLoss):
    """
    Balanced MSE (BMC) loss for imbalanced visual regression.

    Converts regression into a contrastive-like loss where each prediction-target
    pair is treated as matching samples while others serve as negatives. This
    naturally balances learning across different value ranges.

    The loss treats the negative squared distance as logits:
        logits[i,j] = -(pred[i] - target[j])^2 / (2 * noise_var)
        loss = CrossEntropy(logits, identity) * 2 * noise_var

    Args:
        init_noise_sigma: Initial noise standard deviation. Default: 1.0
            Can be learned during training if learnable=True
        learnable: Whether noise_sigma is a learnable parameter. Default: True
        reduction: Loss reduction method. Default: 'mean'

    Example:
        >>> from torchregress.losses import BalancedMSELoss
        >>> loss_fn = BalancedMSELoss(init_noise_sigma=1.0, learnable=True)
        >>> # Add loss_fn.parameters() to optimizer for learnable sigma
        >>> optimizer = Adam(list(model.parameters()) + list(loss_fn.parameters()))
        >>> loss = loss_fn(y_pred, y_true)

    Reference:
        Ren et al. "Balanced MSE for Imbalanced Visual Regression" (CVPR 2022)

    Notes:
        - Works best with batch sizes >= 32 for sufficient negative pairs
        - The learnable noise_sigma adapts to the data distribution
        - For multi-dimensional outputs, use BalancedMSELossMD
    """

    def __init__(
        self,
        init_noise_sigma: float = 1.0,
        learnable: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)

        if learnable:
            self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
        else:
            self.register_buffer("noise_sigma", torch.tensor(init_noise_sigma))

        self.learnable = learnable

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute Balanced MSE loss.

        Args:
            y_pred: Model predictions [batch_size, 1] or [batch_size]
            target: Ground truth values [batch_size, 1] or [batch_size]
            mask: Optional mask (not fully supported for contrastive loss)
            weights: Optional sample weights

        Returns:
            Balanced MSE loss
        """
        # Flatten to [batch_size] for contrastive computation
        pred_flat = y_pred.view(-1)
        target_flat = target.view(-1)
        batch_size = pred_flat.shape[0]

        # Compute noise variance
        noise_var = self.noise_sigma ** 2

        # Compute pairwise logits: -(pred[i] - target[j])^2 / (2 * noise_var)
        # Shape: [batch_size, batch_size]
        diff = pred_flat.unsqueeze(1) - target_flat.unsqueeze(0)  # [B, B]
        logits = -diff.pow(2) / (2 * noise_var)

        # Target: diagonal (each prediction matches its own target)
        labels = torch.arange(batch_size, device=y_pred.device)

        # Cross-entropy loss
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

        # Scale by noise variance to maintain gradient scale
        loss = loss * (2 * noise_var).detach()

        # Apply optional weights
        if weights is not None:
            weights_flat = weights.view(-1)
            loss = loss * weights_flat

        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            loss = loss * mask_flat
            if self.reduction == "mean":
                return loss.sum() / mask_flat.sum().clamp(min=1)
            elif self.reduction == "sum":
                return loss.sum()
            else:
                return loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


@register_regression_loss("balanced_mse_md")
class BalancedMSELossMD(RegressionLoss):
    """
    Multi-dimensional Balanced MSE loss for vector regression.

    Extends Balanced MSE to multi-dimensional outputs using multivariate
    normal distributions for the contrastive formulation.

    Args:
        n_features: Number of output features
        init_noise_sigma: Initial noise standard deviation. Default: 1.0
        learnable: Whether noise_sigma is learnable. Default: True
        reduction: Loss reduction method. Default: 'mean'

    Example:
        >>> from torchregress.losses import BalancedMSELossMD
        >>> loss_fn = BalancedMSELossMD(n_features=3, init_noise_sigma=1.0)
        >>> loss = loss_fn(y_pred, y_true)  # y_pred: [B, 3]

    Reference:
        Ren et al. "Balanced MSE for Imbalanced Visual Regression" (CVPR 2022)
    """

    def __init__(
        self,
        n_features: int = 1,
        init_noise_sigma: float = 1.0,
        learnable: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.n_features = n_features

        if learnable:
            self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
        else:
            self.register_buffer("noise_sigma", torch.tensor(init_noise_sigma))

        self.learnable = learnable

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute multi-dimensional Balanced MSE loss.

        Args:
            y_pred: Model predictions [batch_size, n_features]
            target: Ground truth values [batch_size, n_features]
            mask: Optional mask
            weights: Optional sample weights

        Returns:
            Balanced MSE loss
        """
        from torch.distributions import MultivariateNormal

        batch_size = y_pred.shape[0]
        noise_var = self.noise_sigma ** 2

        # Create isotropic covariance matrix
        cov = noise_var * torch.eye(self.n_features, device=y_pred.device)

        # Compute log probabilities for all pairs
        # logits[i,j] = log MVN(pred[i]; target[j], noise_var * I)
        mvn = MultivariateNormal(y_pred.unsqueeze(1), cov)  # [B, 1, D]
        logits = mvn.log_prob(target.unsqueeze(0))  # [B, B]

        # Target: diagonal
        labels = torch.arange(batch_size, device=y_pred.device)

        # Cross-entropy loss
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

        # Scale by noise variance
        loss = loss * (2 * noise_var).detach()

        # Apply optional weights
        if weights is not None:
            loss = loss * weights.view(-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


@register_regression_loss("sqinv")
class SQINVLoss(RegressionLoss):
    """
    Square-root Inverse frequency weighting for imbalanced regression.

    Uses sqrt(1/frequency) weighting instead of 1/frequency to prevent
    extreme weight disparities that can destabilize training.

    Args:
        kernel_width: Bandwidth for density estimation. Default: 0.5
        base_loss: Base loss function ('mse', 'mae', 'huber'). Default: 'mse'
        max_weight_ratio: Maximum ratio between largest and smallest weights. Default: 100.0
            Clips weights to prevent extreme values
        reduction: Loss reduction method. Default: 'mean'

    Example:
        >>> from torchregress.losses import SQINVLoss
        >>> loss_fn = SQINVLoss(kernel_width=0.5)
        >>> loss_fn.fit(train_targets)
        >>> loss = loss_fn(y_pred, y_true, sample_indices=indices)

    Reference:
        Yang et al. "Delving into Deep Imbalanced Regression" (ICML 2021)

    Notes:
        - Safer than full inverse weighting (INV)
        - Provides moderate rebalancing without extreme weights
        - Recommended as default over INV for most tasks
    """

    def __init__(
        self,
        kernel_width: float = 0.5,
        base_loss: str = "mse",
        max_weight_ratio: float = 100.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.kernel_width = kernel_width
        self.base_loss = base_loss.lower()
        self.max_weight_ratio = max_weight_ratio

        if self.base_loss not in ["mse", "mae", "huber"]:
            raise ValueError(f"base_loss must be 'mse', 'mae', or 'huber', got {base_loss}")

        self.sqinv_weights: Optional[Tensor] = None
        self._train_targets: Optional[Tensor] = None

    def fit(self, train_targets: Tensor, n_bins: int = 100) -> None:
        """
        Compute SQINV weights from training targets.

        Args:
            train_targets: All training targets [n_samples] or [n_samples, 1]
            n_bins: Number of bins for frequency estimation. Default: 100
        """
        self._train_targets = train_targets.detach().cpu()
        targets_np = train_targets.cpu().numpy().flatten()

        # Bin the continuous targets
        min_val, max_val = targets_np.min(), targets_np.max()
        bins = np.linspace(min_val, max_val, n_bins + 1)
        bin_indices = np.digitize(targets_np, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Count samples in each bin
        bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(np.float32)
        bin_counts = np.maximum(bin_counts, 1e-3)  # Avoid division by zero

        # SQINV: sqrt(1/frequency)
        sqinv = np.sqrt(1.0 / bin_counts)

        # Normalize to mean 1.0
        sqinv = sqinv / sqinv.mean()

        # Clip extreme weights
        min_weight = sqinv.min()
        max_allowed = min_weight * self.max_weight_ratio
        sqinv = np.clip(sqinv, None, max_allowed)

        # Re-normalize after clipping
        sqinv = sqinv / sqinv.mean()

        # Map weights to samples
        sample_weights = sqinv[bin_indices]
        self.sqinv_weights = torch.tensor(sample_weights, dtype=torch.float32)
        self._bins = bins
        self._weights_per_bin = torch.tensor(sqinv, dtype=torch.float32)

    def _compute_weight_for_target(self, target: Tensor) -> Tensor:
        """Compute SQINV weight for given target values."""
        if self._weights_per_bin is None:
            raise ValueError("Must call fit() before computing weights")

        targets_np = target.detach().cpu().numpy().flatten()
        bin_indices = np.digitize(targets_np, self._bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(self._weights_per_bin) - 1)

        weights = self._weights_per_bin[bin_indices]
        return weights.to(target.device)

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
        y_pred: Tensor,
        target: Tensor,
        sample_indices: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute SQINV-weighted loss.

        Args:
            y_pred: Model predictions [batch_size, ...]
            target: Ground truth values [batch_size, ...]
            sample_indices: Optional indices for precomputed weights
            mask: Optional mask for missing values
            weights: Optional additional sample weights

        Returns:
            SQINV-weighted loss
        """
        if self.sqinv_weights is None:
            raise ValueError("Must call fit() before using SQINVLoss")

        self._validate_inputs(y_pred, target, mask)

        # Get SQINV weights
        if sample_indices is not None:
            sqinv_w = self.sqinv_weights[sample_indices].to(y_pred.device)
        else:
            sqinv_w = self._compute_weight_for_target(target)

        # Compute base loss
        base_loss = self._compute_base_loss(y_pred, target)

        # Expand weights if needed
        if base_loss.dim() > 1 and sqinv_w.dim() == 1:
            for _ in range(base_loss.dim() - 1):
                sqinv_w = sqinv_w.unsqueeze(-1)
            sqinv_w = sqinv_w.expand_as(base_loss)

        # Apply weights
        weighted_loss = base_loss * sqinv_w

        if weights is not None:
            weighted_loss = weighted_loss * weights

        return self._reduce_with_mask(weighted_loss, mask, None)


@register_regression_loss("frequency_weighted")
class FrequencyWeightedLoss(RegressionLoss):
    """
    Simple frequency-weighted loss for imbalanced regression.

    Weights samples inversely proportional to their bin frequency.
    This is a straightforward approach: samples from rare value ranges
    get higher weights than samples from common value ranges.

    Args:
        n_bins: Number of bins for frequency estimation. Default: 100
        base_loss: Base loss function ('mse', 'mae', 'huber'). Default: 'mse'
        weighting: Weighting scheme ('inv' for 1/freq, 'sqinv' for sqrt(1/freq)). Default: 'inv'
        max_weight: Maximum weight to prevent extreme values. Default: 10.0
        reduction: Loss reduction method. Default: 'mean'

    Example:
        >>> from torchregress.losses import FrequencyWeightedLoss
        >>> loss_fn = FrequencyWeightedLoss(n_bins=50, weighting='inv')
        >>> loss_fn.fit(train_targets)
        >>> loss = loss_fn(y_pred, y_true)

    Notes:
        - Simple and interpretable
        - fit() must be called before use
        - For calibration-safe alternative, use DensityWeightedLoss
    """

    def __init__(
        self,
        n_bins: int = 100,
        base_loss: str = "mse",
        weighting: str = "inv",
        max_weight: float = 10.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.n_bins = n_bins
        self.base_loss = base_loss.lower()
        self.weighting = weighting.lower()
        self.max_weight = max_weight

        if self.base_loss not in ["mse", "mae", "huber"]:
            raise ValueError(f"base_loss must be 'mse', 'mae', or 'huber', got {base_loss}")

        if self.weighting not in ["inv", "sqinv"]:
            raise ValueError(f"weighting must be 'inv' or 'sqinv', got {weighting}")

        self._bin_weights: Optional[Tensor] = None
        self._bin_edges: Optional[np.ndarray] = None

    def fit(self, train_targets: Tensor) -> None:
        """
        Compute frequency weights from training targets.

        Args:
            train_targets: All training targets [n_samples] or [n_samples, 1]
        """
        targets_np = train_targets.cpu().numpy().flatten()

        # Create bins
        min_val, max_val = targets_np.min(), targets_np.max()
        self._bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)

        # Count samples per bin
        bin_indices = np.digitize(targets_np, self._bin_edges[1:-1])
        bin_counts = np.bincount(bin_indices, minlength=self.n_bins).astype(np.float32)

        # Avoid division by zero
        bin_counts = np.maximum(bin_counts, 1.0)

        # Compute weights
        if self.weighting == "inv":
            weights = 1.0 / bin_counts
        else:  # sqinv
            weights = np.sqrt(1.0 / bin_counts)

        # Normalize to mean 1.0
        weights = weights / weights.mean()

        # Clip extreme weights
        weights = np.clip(weights, 1.0 / self.max_weight, self.max_weight)

        # Re-normalize after clipping
        weights = weights / weights.mean()

        self._bin_weights = torch.tensor(weights, dtype=torch.float32)

    def _get_weight(self, target: Tensor) -> Tensor:
        """Get weight for each target value."""
        if self._bin_weights is None:
            raise ValueError("Must call fit() before using FrequencyWeightedLoss")

        targets_np = target.detach().cpu().numpy().flatten()
        bin_indices = np.digitize(targets_np, self._bin_edges[1:-1])
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        weights = self._bin_weights[bin_indices]
        return weights.to(target.device)

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
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute frequency-weighted loss.

        Args:
            y_pred: Model predictions [batch_size, ...]
            target: Ground truth values [batch_size, ...]
            mask: Optional mask for missing values
            weights: Optional additional sample weights

        Returns:
            Frequency-weighted loss
        """
        self._validate_inputs(y_pred, target, mask)

        # Get frequency weights
        freq_w = self._get_weight(target)

        # Compute base loss
        base_loss = self._compute_base_loss(y_pred, target)

        # Expand weights if needed
        if base_loss.dim() > 1 and freq_w.dim() == 1:
            for _ in range(base_loss.dim() - 1):
                freq_w = freq_w.unsqueeze(-1)
            freq_w = freq_w.expand_as(base_loss)

        # Apply weights
        weighted_loss = base_loss * freq_w

        if weights is not None:
            weighted_loss = weighted_loss * weights

        return self._reduce_with_mask(weighted_loss, mask, None)


@register_regression_loss("dist")
class DistLoss(RegressionLoss):
    """
    Dist Loss: Distribution Distance Constraint for imbalanced regression.

    Minimizes the distribution distance between predictions and labels,
    effectively integrating distribution information into model training.
    This helps the model focus on few-shot (rare) regions.

    The loss combines sample-wise error with a distribution alignment term:
        L_total = L_sample(pred, target) + alpha * L_dist(sorted_pred, pseudo_labels)

    Args:
        n_bins: Number of bins for label distribution estimation. Default: 100
        alpha: Weight for distribution loss term. Default: 1.0
        base_loss: Base loss for sample-wise error ('mse', 'mae'). Default: 'mse'
        dist_loss: Loss for distribution alignment ('mse', 'mae'). Default: 'mae'
        reduction: Loss reduction method. Default: 'mean'

    Example:
        >>> from torchregress.losses import DistLoss
        >>> loss_fn = DistLoss(n_bins=50, alpha=1.0)
        >>> loss_fn.fit(train_targets)  # Estimate label distribution
        >>> loss = loss_fn(y_pred, y_true)

    Reference:
        Nie et al. "Dist Loss: Enhancing Regression in Few-shot Region
        through Distribution Distance Constraint" (ICLR 2025)

    Notes:
        - Requires fitting on training data to estimate label distribution
        - Works best with larger batch sizes (256+)
        - Differentiable sorting enables gradient flow through distribution loss
    """

    def __init__(
        self,
        n_bins: int = 100,
        alpha: float = 1.0,
        base_loss: str = "mse",
        dist_loss: str = "mae",
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.n_bins = n_bins
        self.alpha = alpha
        self.base_loss = base_loss.lower()
        self.dist_loss = dist_loss.lower()

        if self.base_loss not in ["mse", "mae"]:
            raise ValueError(f"base_loss must be 'mse' or 'mae', got {base_loss}")

        if self.dist_loss not in ["mse", "mae"]:
            raise ValueError(f"dist_loss must be 'mse' or 'mae', got {dist_loss}")

        self._label_probs: Optional[Tensor] = None
        self._bin_centers: Optional[Tensor] = None
        self._fitted = False

    def fit(self, train_targets: Tensor) -> None:
        """
        Estimate label distribution from training targets.

        Args:
            train_targets: All training targets [n_samples] or [n_samples, 1]
        """
        targets_np = train_targets.cpu().numpy().flatten()

        # Create bins
        min_val, max_val = targets_np.min(), targets_np.max()
        bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)

        # Count samples per bin
        bin_indices = np.digitize(targets_np, bin_edges[1:-1])
        bin_counts = np.bincount(bin_indices, minlength=self.n_bins).astype(np.float32)

        # Convert to probabilities
        probs = bin_counts / bin_counts.sum()

        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        self._label_probs = torch.tensor(probs, dtype=torch.float32)
        self._bin_centers = torch.tensor(bin_centers, dtype=torch.float32)
        self._fitted = True

    def _generate_pseudo_labels(self, batch_size: int, device: torch.device) -> Tensor:
        """
        Generate pseudo-labels that match the estimated label distribution.

        Args:
            batch_size: Number of pseudo-labels to generate
            device: Device to create tensor on

        Returns:
            Sorted pseudo-labels [batch_size]
        """
        # Compute expected counts per bin
        expected_counts = self._label_probs * batch_size

        # Round to integers while maintaining sum = batch_size
        int_counts = expected_counts.floor().long()
        remainder = batch_size - int_counts.sum().item()

        # Add remainder to bins with highest fractional parts
        if remainder > 0:
            fractional = expected_counts - int_counts.float()
            _, top_indices = fractional.topk(int(remainder))
            int_counts[top_indices] += 1

        # Generate pseudo-labels by repeating bin centers
        pseudo_labels = []
        for i, count in enumerate(int_counts):
            if count > 0:
                pseudo_labels.extend([self._bin_centers[i].item()] * count.item())

        # Sort and convert to tensor
        pseudo_labels = sorted(pseudo_labels)
        return torch.tensor(pseudo_labels, dtype=torch.float32, device=device)

    def _compute_base_loss(self, y_pred: Tensor, target: Tensor) -> Tensor:
        """Compute sample-wise base loss."""
        if self.base_loss == "mse":
            return (y_pred - target) ** 2
        else:  # mae
            return torch.abs(y_pred - target)

    def _compute_dist_loss(self, sorted_pred: Tensor, pseudo_labels: Tensor) -> Tensor:
        """Compute distribution alignment loss."""
        if self.dist_loss == "mse":
            return ((sorted_pred - pseudo_labels) ** 2).mean()
        else:  # mae
            return torch.abs(sorted_pred - pseudo_labels).mean()

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Compute Dist Loss.

        Args:
            y_pred: Model predictions [batch_size, 1] or [batch_size]
            target: Ground truth values [batch_size, 1] or [batch_size]
            mask: Optional mask for missing values
            weights: Optional sample weights

        Returns:
            Combined sample loss + distribution loss
        """
        if not self._fitted:
            raise ValueError("Must call fit() before using DistLoss")

        self._validate_inputs(y_pred, target, mask)

        # Flatten predictions
        pred_flat = y_pred.view(-1)
        target_flat = target.view(-1)
        batch_size = pred_flat.shape[0]

        # 1. Sample-wise loss
        sample_loss = self._compute_base_loss(pred_flat, target_flat)

        # Apply weights if provided
        if weights is not None:
            weights_flat = weights.view(-1)
            sample_loss = sample_loss * weights_flat

        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            sample_loss = sample_loss * mask_flat
            sample_loss_mean = sample_loss.sum() / mask_flat.sum().clamp(min=1)
        else:
            sample_loss_mean = sample_loss.mean()

        # 2. Distribution loss
        # Sort predictions (differentiable via autograd)
        sorted_pred, _ = torch.sort(pred_flat)

        # Generate pseudo-labels matching target distribution
        pseudo_labels = self._generate_pseudo_labels(batch_size, pred_flat.device)

        # Ensure same length (handle edge cases)
        min_len = min(len(sorted_pred), len(pseudo_labels))
        dist_loss = self._compute_dist_loss(
            sorted_pred[:min_len], pseudo_labels[:min_len]
        )

        # Combined loss
        total_loss = sample_loss_mean + self.alpha * dist_loss

        return total_loss
