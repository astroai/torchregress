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

from typing import Any, Callable, Optional

import numpy as np
import torch
from torch import Tensor

from .base import RegressionLoss


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
        try:
            from sklearn.neighbors import KernelDensity
        except ImportError:
            raise ImportError(
                "DensityWeightedLoss requires scikit-learn. "
                "Install with: pip install scikit-learn"
            )

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

        try:
            from sklearn.neighbors import KernelDensity
        except ImportError:
            raise ImportError("DensityWeightedLoss requires scikit-learn")

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
                "Must call fit() before using LDSLoss. "
                "Example: loss_fn.fit(train_targets)"
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
