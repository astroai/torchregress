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

import math
from typing import Any, Optional, cast

import numpy as np
import torch
from torch import Tensor

from ..utils.propensity import ipw_weights
from .base import RegressionLoss
from .loss_registry import register_regression_loss

# ── Torch-native Gaussian KDE (replaces sklearn.neighbors.KernelDensity) ───


def _gaussian_kde_score_samples(
    train: Tensor,
    query: Tensor,
    bandwidth: float,
) -> Tensor:
    """Log-density of query points under a Gaussian KDE fitted on train.

    Equivalent to ``sklearn.neighbors.KernelDensity(kernel='gaussian').score_samples()``
    but runs on-device (GPU-compatible).
    """
    if train.dim() == 1:
        train = train.unsqueeze(-1)
    if query.dim() == 1:
        query = query.unsqueeze(-1)

    q_sq = (query**2).sum(dim=1, keepdim=True)
    t_sq = (train**2).sum(dim=1)
    sq_dists = q_sq + t_sq - 2 * (query @ train.T)

    inv_h2 = 1.0 / (bandwidth * bandwidth)
    log_kernel_vals = -0.5 * sq_dists * inv_h2

    log_sum = torch.logsumexp(log_kernel_vals, dim=1)

    d = train.shape[1]
    log_norm = math.log(train.shape[0]) + d * math.log(bandwidth * math.sqrt(2.0 * math.pi))

    return log_sum - log_norm


def _torch_kde_weights(
    train_targets: Tensor,
    bandwidth: float,
    reweight_factor: float,
) -> Tensor:
    """Compute inverse-density weights using torch-native Gaussian KDE."""
    log_density = _gaussian_kde_score_samples(train_targets, train_targets, bandwidth)

    inv_density = torch.exp(-log_density)
    inv_density = inv_density / inv_density.mean()

    weights = 1.0 + reweight_factor * (inv_density - 1.0)
    return weights


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

    References
    ----------
    .. [1] Steininger, M., Kobs, K., Padberg, P., & Hotho, A. (2021).
       Density-based weighting for imbalanced regression.
       In *Machine Learning*, 110(8), 2187-2209.
       https://link.springer.com/article/10.1007/s10994-021-06024-2
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
        # Store targets for potential reuse
        self._train_targets = train_targets.detach()

        weights = _torch_kde_weights(
            self._train_targets,
            bandwidth=self.kernel_width,
            reweight_factor=self.reweight_factor,
        )
        self.density_weights = weights.cpu()

    def _compute_density_weight(self, target: Tensor) -> Tensor:
        """
        Compute density weight for given target values on-the-fly.

        This is used when sample_indices are not provided.
        """
        if self._train_targets is None:
            raise ValueError("Must call fit_density() before computing weights")

        train = self._train_targets.to(device=target.device, dtype=target.dtype)

        log_density = _gaussian_kde_score_samples(train, target, self.kernel_width)

        inv_density = torch.exp(-log_density)
        inv_density = inv_density / inv_density.mean()
        weights = 1.0 + self.reweight_factor * (inv_density - 1.0)

        return weights

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


@register_regression_loss("propensity_weighted")
class PropensityWeightedLoss(RegressionLoss):
    """Inverse-propensity weighted regression loss for selection bias correction."""

    def __init__(
        self,
        base_loss: str = "mse",
        clip_min: float = 0.01,
        clip_max: float = 0.99,
        normalize_weights: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.base_loss = base_loss.lower()
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.normalize_weights = normalize_weights

        if self.base_loss not in ["mse", "mae", "huber"]:
            raise ValueError(f"base_loss must be 'mse', 'mae', or 'huber', got {base_loss}")

        if not (0.0 < clip_min < clip_max < 1.0):
            raise ValueError("clip_min/clip_max must satisfy 0 < clip_min < clip_max < 1")

    def _compute_base_loss(self, y_pred: Tensor, target: Tensor) -> Tensor:
        if self.base_loss == "mse":
            return (y_pred - target) ** 2
        if self.base_loss == "mae":
            return torch.abs(y_pred - target)
        if self.base_loss == "huber":
            diff = torch.abs(y_pred - target)
            return torch.where(diff < 1.0, 0.5 * diff**2, diff - 0.5)
        raise ValueError(f"Unknown base_loss: {self.base_loss}")

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        propensity: Optional[Tensor] = None,
        observed: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        self._validate_inputs(y_pred, target, mask)

        p = propensity if propensity is not None else kwargs.get("propensity_scores")
        if p is None:
            raise ValueError("propensity (or propensity_scores) must be provided")
        if not isinstance(p, torch.Tensor):
            raise TypeError("propensity must be a torch.Tensor")
        if p.shape != target.shape:
            if p.shape == target.shape[:1]:
                # Broadcast sample-level propensity across target dimensions.
                p = p.view(p.shape[0], *([1] * (target.dim() - 1))).expand_as(target)
            else:
                raise ValueError("propensity shape must match target or batch dimension")

        obs = observed
        if obs is not None and obs.shape != target.shape:
            if obs.shape == target.shape[:1]:
                obs = obs.view(obs.shape[0], *([1] * (target.dim() - 1))).expand_as(target)
            else:
                raise ValueError("observed shape must match target or batch dimension")

        ipw = ipw_weights(
            p,
            observed=obs,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            normalize=self.normalize_weights,
        ).to(device=target.device, dtype=target.dtype)

        loss = self._compute_base_loss(y_pred, target) * ipw
        if weights is not None:
            loss = loss * weights
        return self._reduce_with_mask(loss, mask, None)


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

    References
    ----------
    .. [1] Yang, Y., Zha, K., Chen, Y. C., Wang, H., & Katabi, D. (2021).
       Delving into Deep Imbalanced Regression. In *ICML 2021*.
       https://arxiv.org/abs/2102.09554
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
        self._bins: Optional[Tensor] = None
        self._weights_per_bin: Optional[Tensor] = None

    def _get_kernel_window(self, kernel_width: float) -> np.ndarray:
        """Generate kernel window for smoothing."""
        # Create symmetric kernel window
        half_width = int(np.ceil(kernel_width * 3))  # 3 sigma for gaussian
        x: np.ndarray = np.arange(-half_width, half_width + 1, dtype=np.float32)

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
        return cast(np.ndarray, window)

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
        self._bins = torch.tensor(bins, dtype=torch.float32)
        self._weights_per_bin = torch.tensor(weights, dtype=torch.float32)

    def _compute_weight_for_target(self, target: Tensor) -> Tensor:
        """Compute LDS weight for given target values."""
        if self._weights_per_bin is None or self._bins is None:
            raise ValueError("Must call fit() before computing weights")

        # Move bins and weights to target device if needed
        if self._bins.device != target.device:
            self._bins = self._bins.to(target.device)
        if self._weights_per_bin.device != target.device:
            self._weights_per_bin = self._weights_per_bin.to(target.device)

        # Digitize targets using torch.bucketize to stay on device
        bin_indices = torch.bucketize(target.flatten(), self._bins) - 1
        bin_indices = torch.clamp(bin_indices, 0, len(self._weights_per_bin) - 1)

        # Get weights
        weights = self._weights_per_bin[bin_indices]
        return weights

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
                "Must call fit() before using LDSLoss. Example: loss_fn.fit(train_targets)"
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
        **kwargs: Any,
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
