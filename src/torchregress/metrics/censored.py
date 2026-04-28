"""Metrics for censored and interval-censored regression tasks."""

from __future__ import annotations

import torch
from torch import Tensor


def censoring_rate(censoring: Tensor) -> Tensor:
    """Fraction of censored samples (non-zero censoring codes)."""
    return (censoring != 0).float().mean()


def observed_mae(y_pred: Tensor, target: Tensor, censoring: Tensor | None = None) -> Tensor:
    """MAE over observed samples only (`censoring == 0`)."""
    if y_pred.shape != target.shape:
        raise ValueError("y_pred and target must have identical shape")

    if censoring is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    else:
        if censoring.shape != target.shape:
            raise ValueError("censoring must have same shape as target")
        mask = censoring == 0

    if torch.sum(mask) == 0:
        return torch.tensor(float("nan"), dtype=y_pred.dtype, device=y_pred.device)
    return torch.mean(torch.abs(y_pred[mask] - target[mask]))


def concordance_index(y_pred: Tensor, target: Tensor, censoring: Tensor | None = None) -> Tensor:
    """Harrell-style concordance index for right/left censored regression.

    Assumes larger `y_pred` indicates larger target value (e.g., survival time).
    """
    y_hat = y_pred.reshape(-1)
    y = target.reshape(-1)
    if y_hat.shape != y.shape:
        raise ValueError("y_pred and target must have identical number of samples")

    if censoring is None:
        c = torch.zeros_like(y, dtype=torch.int64)
    else:
        c = censoring.reshape(-1).long()
        if c.shape != y.shape:
            raise ValueError("censoring must have same number of samples as target")

    observed_mask = c == 0
    y_obs = y[observed_mask]
    y_hat_obs = y_hat[observed_mask]

    y_less = y_obs.unsqueeze(1) < y.unsqueeze(0)
    y_hat_less = y_hat_obs.unsqueeze(1) < y_hat.unsqueeze(0)
    y_hat_eq = y_hat_obs.unsqueeze(1) == y_hat.unsqueeze(0)

    comparable = y_less.sum(dtype=torch.float32)
    concordant = (y_less & y_hat_less).sum(dtype=torch.float32) + 0.5 * (y_less & y_hat_eq).sum(
        dtype=torch.float32
    )

    if comparable <= 0:
        return torch.tensor(float("nan"), dtype=torch.float32, device=y.device)
    return concordant / comparable


def interval_overlap_rate(
    pred_lower: Tensor,
    pred_upper: Tensor,
    lower_bound: Tensor,
    upper_bound: Tensor,
) -> Tensor:
    """Fraction of samples where predicted interval overlaps censor interval bounds."""
    if not (pred_lower.shape == pred_upper.shape == lower_bound.shape == upper_bound.shape):
        raise ValueError("all interval tensors must share identical shapes")

    overlap = (pred_upper >= lower_bound) & (pred_lower <= upper_bound)
    return overlap.float().mean()


__all__ = [
    "censoring_rate",
    "observed_mae",
    "concordance_index",
    "interval_overlap_rate",
]
