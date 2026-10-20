"""Photo-z recipe objectives: CE + bin-CDF + Rank-N-Contrast.

This implements the first-stage fitting criterion of the two-stage photo-z
recipe used in the companion astronomy paper:

.. math::

    L = w_ce L_ce + w_cdf \\bar h \\sum_j (C_j - 1(c_j \\ge z))^2
        + w_rnc L_{RnC-N}.

The second stage is a classifier on the first-stage representation; see
``torchregress.models.PZRecipe``.  The objective is intentionally separate from
the density losses: it is a *binned* conditional CDF objective, not a continuous
flow likelihood, and is therefore a useful low-data/tabular baseline.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from .base import BaseLoss
from .contrastive import RankNContrastLoss
from .loss_registry import register_regression_loss


def cumulative_cdf_loss(
    logits: Tensor,
    target: Tensor,
    bin_centers: Tensor,
) -> Tensor:
    """Per-sample bin-CDF squared error (paper ``L_CDF``).

    Args:
        logits: ``[B, K]`` bin logits.
        target: ``[B]`` or ``[B, 1]`` reference redshifts.
        bin_centers: ``[K]`` increasing bin centers.

    Returns:
        ``[B]`` loss values.
    """
    centers = torch.as_tensor(bin_centers, dtype=logits.dtype, device=logits.device)
    y = target.reshape(-1).to(dtype=logits.dtype, device=logits.device)
    probability = torch.softmax(logits, dim=-1)
    cdf = probability.cumsum(dim=-1)
    step = (centers.unsqueeze(0) >= y.unsqueeze(1)).to(dtype=logits.dtype)
    mean_width = (centers[-1] - centers[0]) / max(centers.numel() - 1, 1)
    return mean_width * ((cdf - step) ** 2).sum(dim=-1)


@register_regression_loss("recipe_cdf")
class CumulativeCDFLoss(BaseLoss):
    """Binned cumulative-distribution loss for photo-z / ordinal regression."""

    bin_centers: Tensor

    def __init__(self, bin_centers: Tensor | list[float], reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        centers = torch.as_tensor(bin_centers, dtype=torch.float32)
        if centers.dim() != 1 or centers.numel() < 2:
            raise ValueError("bin_centers must be a 1-D sequence with at least two values")
        if not bool(torch.all(centers[1:] > centers[:-1])):
            raise ValueError("bin_centers must be strictly increasing")
        self.register_buffer("bin_centers", centers)

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs
        if y_pred.dim() != 2 or y_pred.shape[-1] != self.bin_centers.numel():
            raise ValueError(f"y_pred must be [B, {self.bin_centers.numel()}]")
        loss = cumulative_cdf_loss(y_pred, target, self.bin_centers)
        return self._reduce(loss, mask, weights)


@register_regression_loss("recipe_tri")
class TriLoss(BaseLoss):
    """CE + binned-CRPS + normalized Rank-N-Contrast objective.

    This is the methods-paper "tri-loss": cross-entropy over quantile bins,
    a CRPS computed exactly from the piecewise-linear bin CDF, and RnC-N on the
    learned representation.  Unlike :class:`PZRecipeLoss` the distribution term
    is CRPS rather than the squared CDF error.
    """

    bin_centers: Tensor

    def __init__(
        self,
        bin_centers: Tensor | list[float],
        *,
        ce_weight: float = 1.0,
        crps_weight: float = 0.5,
        rnc_weight: float = 0.1,
        rnc_temperature: float = 2.0,
        rnc_normalize: bool = True,
        rnc_dropouts: tuple[float, ...] | None = (0.05, 0.20),
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        centers = torch.as_tensor(bin_centers, dtype=torch.float32)
        if centers.dim() != 1 or centers.numel() < 2:
            raise ValueError("bin_centers must be a 1-D sequence with at least two values")
        if not bool(torch.all(centers[1:] > centers[:-1])):
            raise ValueError("bin_centers must be strictly increasing")
        self.register_buffer("bin_centers", centers)
        self.ce_weight = float(ce_weight)
        self.crps_weight = float(crps_weight)
        self.rnc_weight = float(rnc_weight)
        self.rnc = RankNContrastLoss(temperature=rnc_temperature)
        self.rnc_normalize = bool(rnc_normalize)
        self.rnc_dropouts = rnc_dropouts

    def _edges(self) -> Tensor:
        centers = self.bin_centers
        width = torch.diff(centers)
        edges = torch.empty(centers.numel() + 1, dtype=centers.dtype, device=centers.device)
        edges[1:-1] = centers[:-1] + width / 2.0
        edges[0] = centers[0] - width[0] / 2.0
        edges[-1] = centers[-1] + width[-1] / 2.0
        return edges

    def _crps(self, logits: Tensor, target: Tensor) -> Tensor:
        """Differentiable exact CRPS for a piecewise-linear bin CDF."""
        y = target.reshape(-1).to(dtype=logits.dtype, device=logits.device)
        edges = self._edges()
        cdf = torch.softmax(logits, dim=-1).cumsum(dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
        fa, fb = cdf[:, :-1], cdf[:, 1:]
        widths = (edges[1:] - edges[:-1]).unsqueeze(0)
        f2 = widths * (fa**2 + fa * fb + fb**2) / 3.0
        ga, gb = 1.0 - fa, 1.0 - fb
        g2 = widths * (ga**2 + ga * gb + gb**2) / 3.0
        below = edges[1:].unsqueeze(0) <= y.unsqueeze(1)
        above = edges[:-1].unsqueeze(0) >= y.unsqueeze(1)
        middle = ~(below | above)
        total = torch.where(below, f2, torch.zeros_like(f2)).sum(dim=-1)
        total = total + torch.where(above, g2, torch.zeros_like(g2)).sum(dim=-1)
        # Middle bin: split at the target; interpolation is linear in the CDF.
        a = edges[:-1].unsqueeze(0)
        b = edges[1:].unsqueeze(0)
        frac = ((y.unsqueeze(1) - a) / (b - a).clamp_min(1e-12)).clamp(0.0, 1.0)
        fy = fa + frac * (fb - fa)
        left_w = (y.unsqueeze(1) - a).clamp_min(0.0)
        right_w = (b - y.unsqueeze(1)).clamp_min(0.0)
        left = left_w * (fa**2 + fa * fy + fy**2) / 3.0
        right = right_w * ((1 - fy) ** 2 + (1 - fy) * (1 - fb) + (1 - fb) ** 2) / 3.0
        total = total + torch.where(middle, left + right, torch.zeros_like(left)).sum(dim=-1)
        return total

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        features: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs

        if y_pred.dim() != 2 or y_pred.shape[-1] != self.bin_centers.numel():
            raise ValueError(f"y_pred must be [B, {self.bin_centers.numel()}]")
        n = y_pred.shape[0]
        y = target.reshape(n)
        log_probs = F.log_softmax(y_pred, dim=-1)
        idx = torch.searchsorted(self.bin_centers, y.detach()).clamp(1, len(self.bin_centers)) - 1
        ce = -log_probs.gather(1, idx.reshape(-1, 1)).reshape(-1)
        crps = self._crps(y_pred, y)
        loss = self.ce_weight * ce + self.crps_weight * crps
        if self.rnc_weight > 0.0:
            if features is None:
                raise ValueError("features are required when rnc_weight > 0")
            feats = F.normalize(features, dim=-1) if self.rnc_normalize else features
            if self.rnc_dropouts:
                views = [feats]
                labels = [y]
                for p in self.rnc_dropouts:
                    keep = (torch.rand_like(feats) > float(p)).to(feats.dtype)
                    views.append(feats * keep)
                    labels.append(y)
                loss = loss + self.rnc_weight * self.rnc(
                    torch.cat(views, dim=0), torch.cat(labels, dim=0)
                )
            else:
                loss = loss + self.rnc_weight * self.rnc(feats, y)
        return self._reduce(loss, mask, weights)


class PZRecipeLoss(BaseLoss):
    """CE + binned-CDF + normalized Rank-N-Contrast objective.

    Args:
        bin_centers: increasing bin centers (quantiles of the training redshift).
        ce_weight, cdf_weight, rnc_weight: objective coefficients
            (paper defaults 0.5, 0.5, 0.1).
        rnc_temperature: Rank-N-Contrast temperature (paper 2).
        rnc_normalize: L2-normalize features before RnC (paper RnC-N).
        rnc_dropouts: feature-dropout fractions for the two augmented views
            (paper 0.05 and 0.20); ``None`` disables augmentation.
        reduction: loss reduction.
    """

    bin_centers: Tensor

    def __init__(
        self,
        bin_centers: Tensor | list[float],
        *,
        ce_weight: float = 0.5,
        cdf_weight: float = 0.5,
        rnc_weight: float = 0.1,
        rnc_temperature: float = 2.0,
        rnc_normalize: bool = True,
        rnc_dropouts: tuple[float, ...] | None = (0.05, 0.20),
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        centers = torch.as_tensor(bin_centers, dtype=torch.float32)
        if centers.dim() != 1 or centers.numel() < 2:
            raise ValueError("bin_centers must be a 1-D sequence with at least two values")
        if not bool(torch.all(centers[1:] > centers[:-1])):
            raise ValueError("bin_centers must be strictly increasing")
        self.register_buffer("bin_centers", centers)
        for name, value in {
            "ce_weight": ce_weight,
            "cdf_weight": cdf_weight,
            "rnc_weight": rnc_weight,
        }.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.ce_weight = float(ce_weight)
        self.cdf_weight = float(cdf_weight)
        self.rnc_weight = float(rnc_weight)
        self.rnc = RankNContrastLoss(temperature=rnc_temperature)
        self.rnc_normalize = bool(rnc_normalize)
        self.rnc_dropouts = rnc_dropouts

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        features: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs
        if y_pred.dim() != 2 or y_pred.shape[-1] != self.bin_centers.numel():
            raise ValueError(f"y_pred must be [B, {self.bin_centers.numel()}]")
        n = y_pred.shape[0]
        y = target.reshape(n)
        log_probs = F.log_softmax(y_pred, dim=-1)
        idx = torch.searchsorted(self.bin_centers, y.detach()).clamp(1, len(self.bin_centers)) - 1
        ce = -log_probs.gather(1, idx.reshape(-1, 1)).reshape(-1)
        cdf = cumulative_cdf_loss(y_pred, y, self.bin_centers)
        loss = self.ce_weight * ce + self.cdf_weight * cdf
        if self.rnc_weight > 0.0:
            if features is None:
                raise ValueError("features are required when rnc_weight > 0")
            feats = features
            labels = y
            if self.rnc_normalize:
                feats = F.normalize(feats, dim=-1)
            rnc = self.rnc(feats, labels)
            if self.rnc_dropouts:
                views = [feats]
                labels_all = [labels]
                for p in self.rnc_dropouts:
                    keep = (torch.rand_like(feats) > float(p)).to(feats.dtype)
                    views.append(feats * keep)
                    labels_all.append(labels)
                rnc = self.rnc(torch.cat(views, dim=0), torch.cat(labels_all, dim=0))
                # The repeated views contribute independent anchors; the
                # reduction over all of them is the average objective.
            loss = loss + self.rnc_weight * rnc
        return self._reduce(loss, mask, weights)
