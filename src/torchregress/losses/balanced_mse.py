"""
Balanced MSE losses for imbalanced regression targets.

Splits the target range into bins and reweights squared error inversely to the
empirical bin mass (with optional Laplace-style smoothing). This follows the
continuous analogue of class-balanced losses used for long-tailed regression.

References:
    - Ren et al., "Balanced MSE for Long-Tailed Visual Recognition" (CVPR 2022)
      (classification); bin reweighting extends naturally to scalar regression.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, cast

import torch
from torch import Tensor

from .base import RegressionLoss
from .loss_registry import register_regression_loss


def _scalar_for_binning(target: Tensor) -> Tensor:
    """Map a target tensor to a 1-D scalar per sample for histogram binning."""
    if target.dim() == 0:
        return target.reshape(1)
    if target.dim() == 1:
        return target
    # Multi-output: use coordinate mean (user can pre-transform targets if needed).
    return target.mean(dim=-1)


def _bin_indices(y: Tensor, bin_edges: Tensor) -> Tensor:
    """Bin indices in ``[0, n_bins-1]`` for ``bin_edges`` of length ``n_bins+1``."""
    if bin_edges.dim() != 1 or bin_edges.numel() < 2:
        raise ValueError("bin_edges must be a 1-D tensor with at least two values.")
    if torch.any(bin_edges[1:] <= bin_edges[:-1]):
        raise ValueError("bin_edges must be strictly increasing.")
    n_bins = bin_edges.numel() - 1
    idx = torch.searchsorted(bin_edges, y, right=True) - 1
    return cast(Tensor, idx.clamp(0, n_bins - 1))


def _normalize_weights(w: Tensor) -> Tensor:
    """Scale weights so their mean is 1 (stable loss magnitude)."""
    mean_w = w.mean().clamp(min=torch.finfo(w.dtype).eps)
    return w / mean_w


@register_regression_loss("balanced_mse")
class BalancedMSELoss(RegressionLoss):
    """
    Inverse bin-frequency weighted MSE.

    After :meth:`fit`, each training target falls into a bin; per-bin weights are
    proportional to ``1 / count`` (optionally with additive smoothing). Weights are
    normalized to mean 1.

    Parameters
    ----------
    bin_edges:
        Strictly increasing boundaries ``[e0, e1, ..., eK]`` defining ``K`` bins.
    count_smoothing:
        Added to each bin count before inversion (Laplace-style). Default ``0.0``.
    reduction:
        ``'mean'`` | ``'sum'`` | ``'none'`` (per-element weighted squared error).

    Notes
    -----
    Call :meth:`fit` once on training targets before the optimization loop.
    """

    def __init__(
        self,
        bin_edges: Tensor,
        *,
        count_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.register_buffer("bin_edges", bin_edges.clone().detach().float())
        self.count_smoothing = float(count_smoothing)
        self.register_buffer("_bin_weights", torch.tensor([], dtype=torch.float32))

    @property
    def bin_weights(self) -> Tensor:
        if self._bin_weights.numel() == 0:
            raise RuntimeError("Call fit(train_targets) before using BalancedMSELoss.")
        return self._bin_weights

    def fit(self, train_targets: Tensor) -> BalancedMSELoss:
        """Compute bin counts on training targets and set inverse-frequency weights."""
        edges = self.bin_edges
        y = _scalar_for_binning(train_targets.detach().float()).reshape(-1)
        ind = _bin_indices(y, edges)
        n_bins = edges.numel() - 1
        counts = torch.bincount(ind, minlength=n_bins).to(dtype=torch.float32)
        smooth = counts + self.count_smoothing
        inv = 1.0 / smooth.clamp(min=torch.finfo(smooth.dtype).eps)
        w = _normalize_weights(inv).to(device=edges.device, dtype=edges.dtype)
        self._bin_weights = w
        return self

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        self._validate_inputs(y_pred, target, mask)
        bw = self.bin_weights
        edges = self.bin_edges
        y_scalar = _scalar_for_binning(target.detach().float())
        ind = _bin_indices(y_scalar, edges)
        per_bin = bw[ind]
        while per_bin.dim() < y_pred.dim():
            per_bin = per_bin.unsqueeze(-1)
        sq = (y_pred - target) ** 2
        weighted = sq * per_bin
        if weights is not None:
            if weights.dim() < weighted.dim():
                weights = weights.reshape(weights.shape + (1,) * (weighted.dim() - weights.dim()))
            weighted = weighted * weights
        return self._reduce_with_mask(weighted, mask, None)


@register_regression_loss("bmc")
class BMCLoss(RegressionLoss):
    """
    Balanced MSE with automatic bin edges and Laplace-smoothed counts.

    Fits ``num_bins`` bins on the training target range (equal width or quantile
    splits). Per-sample weights are ``1 / (count_b + noise_sigma)``, normalized to
    mean 1. Larger ``noise_sigma`` down-weights rare bins less aggressively.

    Parameters
    ----------
    num_bins:
        Number of contiguous bins (must be >= 1).
    noise_sigma:
        Pseudocount added to each bin before inversion. Matches the common
        ``BMCLoss(noise_sigma=...)`` sketch for smoothed inverse frequency.
    binning:
        ``'equal'`` — equal-width bins on ``[min(y), max(y)]``;
        ``'quantile'`` — quantile bins (approximately equal mass if ``y`` is continuous).
    reduction:
        ``'mean'`` | ``'sum'`` | ``'none'``.

    Notes
    -----
    Call :meth:`fit` once before training. For quantile binning, ``min``/``max``
    are taken from the same ``train_targets`` used to build edges.
    """

    def __init__(
        self,
        num_bins: int,
        *,
        noise_sigma: float = 1.0,
        binning: Literal["equal", "quantile"] = "equal",
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        if num_bins < 1:
            raise ValueError("num_bins must be >= 1.")
        self.num_bins = num_bins
        self.noise_sigma = float(noise_sigma)
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative.")
        self.binning = binning
        self.register_buffer("bin_edges", torch.tensor([], dtype=torch.float32))
        self.register_buffer("_bin_weights", torch.tensor([], dtype=torch.float32))

    def fit(self, train_targets: Tensor) -> BMCLoss:
        """Build bin edges from ``train_targets`` and inverse smoothed counts."""
        y = _scalar_for_binning(train_targets.detach().float()).reshape(-1)
        if y.numel() == 0:
            raise ValueError("train_targets must be non-empty.")
        lo, hi = y.min(), y.max()
        if lo == hi:
            hi = lo + 1.0
        k = self.num_bins
        if self.binning == "equal":
            edges = torch.linspace(lo, hi, k + 1, device=y.device, dtype=y.dtype)
        elif self.binning == "quantile":
            qs = torch.linspace(0.0, 1.0, k + 1, device=y.device, dtype=y.dtype)
            edges = torch.quantile(y, qs)
            edges = torch.unique(edges, sorted=True)
            if edges.numel() < 2:
                edges = torch.tensor([lo, hi], device=y.device, dtype=y.dtype)
            # If too few unique quantiles, fall back to equal width on [lo, hi].
            if edges.numel() != k + 1:
                edges = torch.linspace(lo, hi, k + 1, device=y.device, dtype=y.dtype)
        else:
            raise ValueError("binning must be 'equal' or 'quantile'.")
        self.bin_edges = edges.clone()
        ind = _bin_indices(y, self.bin_edges)
        n_bins = self.bin_edges.numel() - 1
        counts = torch.bincount(ind, minlength=n_bins).to(dtype=torch.float32)
        smooth = counts + self.noise_sigma
        inv = 1.0 / smooth.clamp(min=torch.finfo(smooth.dtype).eps)
        self._bin_weights = _normalize_weights(inv).to(
            device=self.bin_edges.device, dtype=self.bin_edges.dtype
        )
        return self

    @property
    def bin_weights(self) -> Tensor:
        if self._bin_weights.numel() == 0:
            raise RuntimeError("Call fit(train_targets) before using BMCLoss.")
        return self._bin_weights

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        self._validate_inputs(y_pred, target, mask)
        bw = self.bin_weights
        edges = self.bin_edges
        y_scalar = _scalar_for_binning(target.detach().float())
        ind = _bin_indices(y_scalar, edges)
        per_bin = bw[ind]
        while per_bin.dim() < y_pred.dim():
            per_bin = per_bin.unsqueeze(-1)
        sq = (y_pred - target) ** 2
        weighted = sq * per_bin
        if weights is not None:
            if weights.dim() < weighted.dim():
                weights = weights.reshape(weights.shape + (1,) * (weighted.dim() - weights.dim()))
            weighted = weighted * weights
        return self._reduce_with_mask(weighted, mask, None)
