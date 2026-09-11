"""
1-Wasserstein (earth mover's) loss between discrete distributions on a shared 1D grid.

For two probability vectors ``p`` and ``q`` whose masses sit at the centres
``c_1 < ... < c_K`` of ``K`` bins, the 1-Wasserstein distance has the closed
form

    W1(p, q) = sum_{k=1}^{K-1} |F_k - G_k| (c_{k+1} - c_k),

where ``F``/``G`` are the cumulative sums.  It is differentiable almost
everywhere in ``p`` and, unlike KL or cross-entropy, it penalises a mode placed
in the wrong location proportionally to *how far* it is from the target mass
(a "displacement" penalty), which is what makes it useful next to a local
divergence when training binned or basis-expanded density heads.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import Tensor

from .base import BaseLoss
from .loss_registry import register_regression_loss

__all__ = ["DiscreteWasserstein1Loss", "discrete_wasserstein1"]


def discrete_wasserstein1(p: Tensor, q: Tensor, centers: Tensor) -> Tensor:
    """``W1`` between mass vectors ``p`` and ``q`` (``[..., K]``) located at ``centers`` (``[K]``).

    Both inputs are expected to be normalised along the last dimension; no
    normalisation is applied here.  Returns shape ``[...]``.
    """
    if p.shape != q.shape:
        raise ValueError(
            f"p and q must have the same shape, got {tuple(p.shape)} vs {tuple(q.shape)}"
        )
    if centers.dim() != 1 or centers.numel() != p.shape[-1]:
        raise ValueError("centers must be 1D with length equal to the last dimension of p")
    gaps = (centers[1:] - centers[:-1]).to(p)
    cdf_diff = torch.cumsum(p - q, dim=-1)[..., :-1]
    return (cdf_diff.abs() * gaps).sum(dim=-1)


@register_regression_loss("discrete_wasserstein1")
class DiscreteWasserstein1Loss(BaseLoss):
    """1-Wasserstein loss between predicted and target distributions on a fixed 1D grid.

    Args:
        bin_edges: ``K + 1`` sorted bin edges; masses are placed at bin centres.
            If ``None``, unit-spaced bins are assumed and the loss is measured
            in "bin units".
        from_logits: If ``True`` ``y_pred`` are logits and a softmax is applied
            along the last dimension; otherwise ``y_pred`` are masses and are
            renormalised to sum to one.
        reduction: ``'none' | 'mean' | 'sum'`` over samples.

    Shapes:
        ``y_pred``: ``[N, K]``.  ``target``: either ``[N, K]`` masses (renormalised)
        or ``[N]`` / ``[N, 1]`` scalar values that are converted to a point mass
        in the bin containing them (quantisation error at most half a bin width).
        ``mask``/``weights``: ``[N]``.

    Note:
        Point-mass targets are quantised to bin centres.  When the predicted
        distribution has a continuous parameterisation (e.g. a spline basis)
        prefer an exact ``E|Z - z|`` computation instead of this discrete form.
    """

    def __init__(
        self,
        bin_edges: Optional[Tensor | Sequence[float]] = None,
        from_logits: bool = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.from_logits = bool(from_logits)
        if bin_edges is None:
            self.bin_edges: Optional[Tensor] = None
        else:
            edges = torch.as_tensor(bin_edges, dtype=torch.get_default_dtype()).reshape(-1)
            if edges.numel() < 2 or not torch.all(edges[1:] > edges[:-1]):
                raise ValueError("bin_edges must be strictly increasing with at least two entries")
            self.bin_edges = edges

    def _edges_for(self, k: int, ref: Tensor) -> Tensor:
        if self.bin_edges is None:
            return torch.arange(k + 1, dtype=ref.dtype, device=ref.device) - 0.5
        if self.bin_edges.numel() != k + 1:
            raise ValueError(
                f"bin_edges has {self.bin_edges.numel() - 1} bins but y_pred has {k} columns"
            )
        return self.bin_edges.to(device=ref.device, dtype=ref.dtype)

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        if y_pred.dim() != 2:
            raise ValueError(f"y_pred must be [N, K], got {tuple(y_pred.shape)}")
        n, k = y_pred.shape
        edges = self._edges_for(k, y_pred)
        centers = 0.5 * (edges[:-1] + edges[1:])

        if self.from_logits:
            p = torch.softmax(y_pred, dim=-1)
        else:
            p = y_pred / y_pred.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(y_pred.dtype).tiny)

        if target.shape == y_pred.shape:
            q = target.to(p) / target.to(p).sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(p.dtype).tiny
            )
        elif target.numel() == n:
            idx = (torch.bucketize(target.reshape(n).to(edges), edges) - 1).clamp(0, k - 1)
            q = torch.nn.functional.one_hot(idx, num_classes=k).to(p)
        else:
            raise ValueError(
                "target must be [N, K] masses or [N] scalar values; "
                f"got {tuple(target.shape)} for y_pred {tuple(y_pred.shape)}"
            )

        per_sample = discrete_wasserstein1(p, q, centers)
        return self._reduce(
            per_sample,
            mask.reshape(n) if mask is not None else None,
            weights.reshape(n) if weights is not None else None,
        )
