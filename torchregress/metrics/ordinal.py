"""Ordinal regression metrics for ordered class targets."""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import Tensor

from torchregress.utils.ordinal import ordinal_predict


def _to_labels(
    y_pred: Tensor,
    *,
    encoding: Literal[
        "labels", "cumulative_logits", "cumulative_probs", "class_logits", "class_probs"
    ] = "labels",
) -> Tensor:
    if encoding == "labels":
        return y_pred.long()
    decoded = ordinal_predict(y_pred, encoding=encoding)
    if isinstance(decoded, tuple):
        decoded = decoded[0]
    return cast(Tensor, decoded).long()


def ordinal_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    *,
    encoding: Literal[
        "labels", "cumulative_logits", "cumulative_probs", "class_logits", "class_probs"
    ] = "labels",
) -> Tensor:
    """Compute exact-match accuracy for ordinal labels."""
    pred_labels = _to_labels(y_pred, encoding=encoding)
    true_labels = y_true.long()
    if pred_labels.shape != true_labels.shape:
        raise ValueError("predicted labels and y_true must have identical shape")
    return (pred_labels == true_labels).float().mean()


def mean_absolute_class_error(
    y_pred: Tensor,
    y_true: Tensor,
    *,
    encoding: Literal[
        "labels", "cumulative_logits", "cumulative_probs", "class_logits", "class_probs"
    ] = "labels",
) -> Tensor:
    """Compute mean absolute class-index error (ordinal MAE in class space)."""
    pred_labels = _to_labels(y_pred, encoding=encoding)
    true_labels = y_true.long()
    if pred_labels.shape != true_labels.shape:
        raise ValueError("predicted labels and y_true must have identical shape")
    return torch.mean(torch.abs(pred_labels.float() - true_labels.float()))


def quadratic_weighted_kappa(
    y_pred: Tensor,
    y_true: Tensor,
    *,
    num_classes: int | None = None,
    encoding: Literal[
        "labels", "cumulative_logits", "cumulative_probs", "class_logits", "class_probs"
    ] = "labels",
) -> Tensor:
    """Compute quadratic weighted kappa (QWK) for ordinal predictions."""
    pred_labels = _to_labels(y_pred, encoding=encoding).reshape(-1).long()
    true_labels = y_true.reshape(-1).long()

    if pred_labels.shape != true_labels.shape:
        raise ValueError("predicted labels and y_true must have identical shape")

    if num_classes is None:
        max_label = int(torch.max(torch.cat([pred_labels, true_labels])).item())
        num_classes = max_label + 1
    if num_classes <= 1:
        return torch.tensor(1.0, dtype=torch.float32, device=pred_labels.device)

    if torch.any(pred_labels < 0) or torch.any(pred_labels >= num_classes):
        raise ValueError("predicted labels must be in [0, num_classes - 1]")
    if torch.any(true_labels < 0) or torch.any(true_labels >= num_classes):
        raise ValueError("y_true labels must be in [0, num_classes - 1]")

    cm = (
        torch.bincount(
            true_labels * num_classes + pred_labels,
            minlength=num_classes * num_classes,
        )
        .reshape(num_classes, num_classes)
        .float()
    )

    total = cm.sum().clamp_min(1.0)
    observed = cm / total

    true_hist = torch.bincount(true_labels, minlength=num_classes).float()
    pred_hist = torch.bincount(pred_labels, minlength=num_classes).float()
    expected = torch.outer(true_hist, pred_hist)
    expected = expected / expected.sum().clamp_min(1.0)

    idx = torch.arange(num_classes, device=pred_labels.device, dtype=torch.float32)
    weight = (idx[:, None] - idx[None, :]) ** 2 / float((num_classes - 1) ** 2)

    numerator = torch.sum(weight * observed)
    denominator = torch.sum(weight * expected)
    if denominator <= 1e-12:
        return torch.tensor(1.0, dtype=torch.float32, device=pred_labels.device)
    one = torch.tensor(1.0, dtype=numerator.dtype, device=numerator.device)
    return one - numerator / denominator


__all__ = [
    "ordinal_accuracy",
    "mean_absolute_class_error",
    "quadratic_weighted_kappa",
]
