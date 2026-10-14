"""
Binned inverse-frequency weighted MSE losses for imbalanced regression targets.

Splits the target range into bins and reweights squared error inversely to the
empirical bin mass (with optional additive smoothing). This follows the
continuous analogue of class-balanced losses used for long-tailed regression.
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


def _inverse_frequency_bin_weights(counts: Tensor, smoothing: float) -> Tensor:
    """Per-bin inverse (smoothed) frequency weights, mean 1 over training samples.

    ``w_b = 1 / (count_b + smoothing)``; bins whose smoothed count is zero
    (empty, unsmoothed) get weight 0 instead of ``1/eps``, so a single empty
    bin cannot collapse every populated bin's weight.  The weights are scaled
    so that their average over the TRAINING samples (``sum_b count_b w_b / N``)
    is 1, which keeps the training loss on the scale of the plain MSE and
    makes the weights of populated bins independent of empty bins.
    """
    smooth = counts + smoothing
    inv = torch.where(smooth > 0, 1.0 / smooth.clamp(min=torch.finfo(smooth.dtype).tiny), 0.0)
    n_train = counts.sum()
    per_sample_mean = (counts * inv).sum() / n_train.clamp(min=1.0)
    return inv / per_sample_mean.clamp(min=torch.finfo(inv.dtype).tiny)


@register_regression_loss("balanced_mse")
class BalancedMSELoss(RegressionLoss):
    """
    Inverse bin-frequency weighted MSE.

    After :meth:`fit`, each training target falls into a bin; per-bin weights are
    proportional to ``1 / count`` (optionally with additive smoothing). Weights are
    normalized so that their mean over the training samples is 1; bins with a
    zero (smoothed) count get weight 0.

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

    References
    ----------
    .. [1] Ren, J., Xiao, C., Chang, X., Huang, S., Li, G., & Wang, S. (2022).
       Balanced MSE for Long-Tailed Visual Recognition. In *CVPR 2022*.
       https://arxiv.org/abs/2203.16427
    """

    bin_edges: Tensor
    _bin_weights: Tensor

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
        w = _inverse_frequency_bin_weights(counts, self.count_smoothing)
        self._bin_weights = w.to(device=edges.device, dtype=edges.dtype)
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
        # User sample weights follow the BaseLoss._reduce contract (weighted
        # mean sum(w * l) / sum(w)); the bin weights stay part of the loss.
        return self._reduce(weighted, mask, weights)


@register_regression_loss("bin_weighted_mse")
class BinReweightedMSELoss(RegressionLoss):
    """
    Binned inverse-frequency weighted MSE with automatic bin edges.

    This is plain binned inverse-frequency weighted MSE: it fits ``num_bins``
    bins on the training target range (equal width or quantile splits) and
    assigns per-sample weights ``1 / (count_b + noise_sigma)``, normalized to
    mean 1 over the training samples (empty bins with ``noise_sigma=0`` get
    weight 0). Larger ``noise_sigma`` down-weights rare bins less aggressively.

    Parameters
    ----------
    num_bins:
        Number of contiguous bins (must be >= 1).
    noise_sigma:
        Pseudocount added to each bin before inversion (additive smoothing).
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

    def fit(self, train_targets: Tensor) -> "BinReweightedMSELoss":
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
        self._bin_weights = _inverse_frequency_bin_weights(counts, self.noise_sigma).to(
            device=self.bin_edges.device, dtype=self.bin_edges.dtype
        )
        return self

    @property
    def bin_weights(self) -> Tensor:
        if self._bin_weights.numel() == 0:
            raise RuntimeError("Call fit(train_targets) before using BinReweightedMSELoss.")
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
        # User sample weights follow the BaseLoss._reduce contract (weighted
        # mean sum(w * l) / sum(w)); the bin weights stay part of the loss.
        return self._reduce(weighted, mask, weights)
