"""
Contrastive representation losses for regression targets.

Rank-N-Contrast (Zha et al., NeurIPS 2023) orders the feature space by label
distance: for an anchor ``i`` and any other sample ``j``, every sample ``k``
whose label is *at least as far* from ``y_i`` as ``y_j`` is treated as a
negative for the pair ``(i, j)``.  The resulting representation is continuous
and distance-preserving along the regression target, which separates
degenerate branches that a pointwise regression head alone confuses.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from ..utils.validation import validate_positive
from .base import BaseLoss
from .loss_registry import register_regression_loss

__all__ = ["RankNContrastLoss", "rank_n_contrast_loss"]


def _pairwise_similarity(features: Tensor, kind: str) -> Tensor:
    if kind == "l2":
        # Negative Euclidean distance, as in the original paper.
        return -torch.cdist(features, features, p=2)
    if kind == "cosine":
        f = torch.nn.functional.normalize(features, dim=-1)
        return f @ f.T
    raise ValueError(f"similarity must be 'l2' or 'cosine', got {kind!r}")


def rank_n_contrast_loss(
    features: Tensor,
    labels: Tensor,
    temperature: float = 2.0,
    similarity: str = "l2",
) -> Tensor:
    """Per-anchor Rank-N-Contrast loss.

    Args:
        features: ``[N, D]`` feature vectors.
        labels: ``[N]`` or ``[N, L]`` continuous labels.  Label distance is the
            L1 norm of the difference.
        temperature: Softmax temperature ``tau`` (paper default 2.0 for L2).
        similarity: ``"l2"`` (negative Euclidean distance) or ``"cosine"``.

    Returns:
        ``[N]`` tensor: for anchor ``i`` the mean over ``j != i`` of
        ``-log( exp(s_ij/tau) / sum_{k in S_ij} exp(s_ik/tau) )`` with
        ``S_ij = {k != i : d(y_i, y_k) >= d(y_i, y_j)}``.  Ties in label
        distance are included in the denominator.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")
    n = features.shape[0]
    if n < 2:
        return features.new_zeros(n)
    y = labels.reshape(n, -1).to(features.dtype)
    label_dist = torch.cdist(y, y, p=1)  # [N, N]
    logits = _pairwise_similarity(features, similarity) / temperature  # [N, N]

    # Push the anchor itself to the end of a descending sort, then drop it.
    eye = torch.eye(n, dtype=torch.bool, device=features.device)
    d = label_dist.masked_fill(eye, float("-inf"))
    d_sorted, order = torch.sort(d, dim=1, descending=True)
    d_sorted = d_sorted[:, :-1]
    order = order[:, :-1]
    l_sorted = torch.gather(logits, 1, order)  # [N, N-1]

    # Denominator for position p: logsumexp over all positions q with d_q >= d_p,
    # i.e. cumulative over the descending order up to the last tie of d_p.
    cum = torch.logcumsumexp(l_sorted, dim=1)
    neg = -d_sorted  # ascending
    last_tie = torch.searchsorted(neg, neg, right=True) - 1  # [N, N-1]
    denom = torch.gather(cum, 1, last_tie)
    per_pair = denom - l_sorted  # -log softmax restricted to S_ij
    return per_pair.mean(dim=1)


@register_regression_loss("rank_n_contrast")
class RankNContrastLoss(BaseLoss):
    """Rank-N-Contrast loss for regression representation learning.

    Orders learned features by continuous label distance (Zha et al. 2023):
    for each anchor, samples with closer labels must have higher similarity
    than every sample with a label at least as far away.

    Args:
        temperature: Softmax temperature.  Default 2.0 (paper default for L2).
        similarity: ``"l2"`` (negative Euclidean distance, default) or ``"cosine"``.
        reduction: ``'none' | 'mean' | 'sum'`` over anchors.

    Shapes:
        ``y_pred``: ``[N, D]`` features.  ``target``: ``[N]`` or ``[N, L]`` labels.
        ``mask``/``weights``: ``[N]`` per-anchor; masked samples are removed from
        both the anchor set and every denominator set.

    Reference:
        Zha, K., Cao, P., Son, J., Yang, Y., Katabi, D.  "Rank-N-Contrast:
        Learning Continuous Representations for Regression." NeurIPS 2023.
    """

    def __init__(
        self, temperature: float = 2.0, similarity: str = "l2", reduction: str = "mean"
    ) -> None:
        super().__init__(reduction=reduction)
        self.temperature = float(validate_positive(float(temperature), "temperature"))
        if similarity not in {"l2", "cosine"}:
            raise ValueError(f"similarity must be 'l2' or 'cosine', got {similarity!r}")
        self.similarity = similarity

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        n = y_pred.shape[0]
        if mask is not None:
            keep = mask.reshape(n).to(torch.bool)
            idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
            per_anchor = y_pred.new_zeros(n)
            if idx.numel() >= 2:
                per_anchor[idx] = rank_n_contrast_loss(
                    y_pred[idx], target.reshape(n, -1)[idx], self.temperature, self.similarity
                )
            return self._reduce(
                per_anchor, keep, weights.reshape(n) if weights is not None else None
            )
        per_anchor = rank_n_contrast_loss(y_pred, target, self.temperature, self.similarity)
        return self._reduce(per_anchor, None, weights.reshape(n) if weights is not None else None)
