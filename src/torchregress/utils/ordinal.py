"""Utilities for ordinal / ordered-target regression."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _validate_num_classes(num_classes: int) -> None:
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")


def labels_to_levels(target: Tensor, num_classes: int) -> Tensor:
    """Convert class-index labels to cumulative binary levels.

    For class index ``y in {0, ..., K-1}``, the returned level vector has length ``K-1``
    with entries ``1[y > k]`` for ``k=0..K-2``.
    """
    _validate_num_classes(num_classes)
    target_i = target.long()
    if torch.any(target_i < 0) or torch.any(target_i >= num_classes):
        raise ValueError("target class indices must be in [0, num_classes - 1]")

    thresholds = torch.arange(num_classes - 1, device=target_i.device)
    return (target_i.unsqueeze(-1) > thresholds).to(dtype=torch.float32)


def normalize_class_probs(
    target_probs: Tensor,
    *,
    class_dim: int = -1,
    eps: float = 1e-8,
) -> Tensor:
    """Normalize non-negative class probabilities along ``class_dim``."""
    if target_probs.shape[class_dim] < 2:
        raise ValueError("target_probs class dimension must have at least 2 entries")
    if torch.any(target_probs < 0):
        raise ValueError("target_probs must be non-negative")
    denom = target_probs.sum(dim=class_dim, keepdim=True).clamp_min(eps)
    return target_probs / denom


def class_probs_to_levels(
    target_probs: Tensor,
    *,
    class_dim: int = -1,
    eps: float = 1e-8,
) -> Tensor:
    """Convert per-class PMF targets to cumulative ordinal levels ``P(y > k)``."""
    pmf = normalize_class_probs(target_probs, class_dim=class_dim, eps=eps)
    moved = torch.movedim(pmf, class_dim, -1)
    flipped = torch.flip(moved, dims=[-1])
    tail = torch.cumsum(flipped, dim=-1)
    levels = torch.flip(tail[..., :-1], dims=[-1])
    return torch.movedim(levels, -1, class_dim)


def cumulative_probs_to_pmf(cumulative_probs: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert cumulative ordinal probabilities to per-class PMF probabilities."""
    if cumulative_probs.shape[-1] < 1:
        raise ValueError("cumulative_probs last dimension must be >= 1")

    cdf_tail = cumulative_probs.clamp(0.0, 1.0)
    p_first = 1.0 - cdf_tail[..., :1]
    p_last = cdf_tail[..., -1:]
    if cdf_tail.shape[-1] == 1:
        pmf = torch.cat([p_first, p_last], dim=-1)
    else:
        p_mid = cdf_tail[..., :-1] - cdf_tail[..., 1:]
        pmf = torch.cat([p_first, p_mid, p_last], dim=-1)

    pmf = pmf.clamp_min(0.0)
    pmf_sum = pmf.sum(dim=-1, keepdim=True).clamp_min(eps)
    return pmf / pmf_sum


def cumulative_logits_to_pmf(logits: Tensor) -> Tensor:
    """Convert cumulative logits (e.g. CORAL/cumulative-link heads) to class PMF."""
    return cumulative_probs_to_pmf(torch.sigmoid(logits))


def ordinal_predict(
    y_pred: Tensor,
    *,
    encoding: Literal[
        "cumulative_logits", "cumulative_probs", "class_logits", "class_probs"
    ] = "cumulative_logits",
    threshold: float = 0.5,
    strategy: Literal["argmax", "threshold"] = "argmax",
    num_classes: Optional[int] = None,
    return_pmf: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Decode ordinal outputs into class-index predictions.

    Args:
        y_pred: Model outputs.
        encoding: Output encoding used by the model.
        threshold: Threshold used by ``strategy='threshold'`` for cumulative encodings.
        strategy: ``argmax`` over PMF or thresholded cumulative decoding.
        num_classes: Optional expected number of classes for validation.
        return_pmf: If True, return ``(labels, pmf)``.
    """
    if num_classes is not None:
        _validate_num_classes(num_classes)
        # Expected shape depends on encoding
        if encoding in {"cumulative_logits", "cumulative_probs"}:
            if y_pred.shape[-1] != num_classes - 1:
                raise ValueError(
                    f"num_classes={num_classes} does not match input shape {y_pred.shape} "
                    f"for cumulative encoding (expected {num_classes - 1} columns)"
                )
        else:
            if y_pred.shape[-1] != num_classes:
                raise ValueError(
                    f"num_classes={num_classes} does not match input shape {y_pred.shape} "
                    f"for class-based encoding (expected {num_classes} columns)"
                )

    if strategy not in {"argmax", "threshold"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    if encoding == "cumulative_logits":
        cum_probs = torch.sigmoid(y_pred)
        pmf = cumulative_probs_to_pmf(cum_probs)
        if strategy == "threshold":
            labels = torch.sum(cum_probs >= threshold, dim=-1).long()
        else:
            labels = torch.argmax(pmf, dim=-1)
    elif encoding == "cumulative_probs":
        cum_probs = y_pred
        pmf = cumulative_probs_to_pmf(cum_probs)
        if strategy == "threshold":
            labels = torch.sum(cum_probs >= threshold, dim=-1).long()
        else:
            labels = torch.argmax(pmf, dim=-1)
    elif encoding == "class_logits":
        pmf = torch.softmax(y_pred, dim=-1)
        if strategy == "threshold":
            raise ValueError("strategy='threshold' is only valid for cumulative encodings")
        labels = torch.argmax(pmf, dim=-1)
    elif encoding == "class_probs":
        pmf_raw = y_pred.clamp_min(0.0)
        pmf = pmf_raw / pmf_raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if strategy == "threshold":
            raise ValueError("strategy='threshold' is only valid for cumulative encodings")
        labels = torch.argmax(pmf, dim=-1)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    if return_pmf:
        return labels, pmf
    return labels


__all__ = [
    "labels_to_levels",
    "normalize_class_probs",
    "class_probs_to_levels",
    "cumulative_probs_to_pmf",
    "cumulative_logits_to_pmf",
    "ordinal_predict",
    "CORALHead",
]


# ---------------------------------------------------------------------------
# CORAL shared-weight ordinal head (Cao, Mirjalili & Raschka 2020)
# ---------------------------------------------------------------------------


class CORALHead(nn.Module):
    """Shared-weight ordinal output head with monotonic bias constraints.

    Architecture from Cao et al. (2020): *Rank Consistent Ordinal Regression
    for Neural Networks with Application to Age Estimation*.
    (`arXiv:1901.07884 <https://arxiv.org/abs/1901.07884>`_).

    All ``K-1`` cumulative logits share a **single weight vector** ``w``.
    Only the bias terms differ per level: ``z_k = w·x + b_k``.
    Bias monotonicity ``b_1 ≥ b_2 ≥ ... ≥ b_{K-1}`` is enforced via
    cumulative sums of non-negative increments.

    Parameters
    ----------
    in_features : int
        Input feature dimension (output of the backbone network).
    num_classes : int
        Total number of ordinal classes ``K``.
    """

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.num_classes = int(num_classes)
        self.in_features = int(in_features)
        # Shared weight: (1, in_features) — a single direction in feature space
        self.weight = nn.Parameter(torch.empty(1, int(in_features)))
        # Raw non-negative bias increments δ₁, …, δ_{K-1}
        # b_k = −(softplus(δ₁) + … + softplus(δ_k))  ⇒  b₁ ≥ b₂ ≥ …
        self.bias_increments = nn.Parameter(torch.empty(int(num_classes) - 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias_increments)

    def forward(self, x: Tensor) -> Tensor:
        """Return monotonic cumulative logits.

        Args:
            x: ``(batch, in_features)`` feature vectors from the backbone.

        Returns:
            ``(batch, K-1)`` cumulative logits satisfying
            ``z_0 ≥ z_1 ≥ ... ≥ z_{K-2}`` for every row.
        """
        shared_logit = F.linear(x, self.weight, bias=None)  # (batch, 1)
        # Enforce b_1 ≥ b_2 ≥ … via negative cumulative sum of non-negative
        # increments.
        deltas = F.softplus(self.bias_increments)  # (K-1,)  all ≥ 0
        biases = -torch.cumsum(deltas, dim=0)  # (K-1,)  decreasing
        return shared_logit + biases.unsqueeze(0)  # (batch, K-1)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, num_classes={self.num_classes}"
