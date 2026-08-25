"""Metrics for censored and interval-censored regression tasks.

References
----------
.. [1] Tobin, J. (1958). Estimation of Relationships for Limited Dependent Variables.
   In *Econometrica*, 26(1), 24-34. https://doi.org/10.2307/1907382
"""

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
    """Harrell-style concordance index under right-censoring semantics (TR-MET-18).

    Censoring codes follow the library convention:
      - ``0``: observed (event time known exactly)
      - ``1``: right-censored (true value >= recorded target)
      - ``-1``: left-censored (true value <= recorded target)

    A pair (i, j) is comparable when subject i is observed and the ordering of
    the true values is determined by the data:
      - j observed with ``y_i < y_j``, or
      - j right-censored with ``y_i <= y_j`` (then true_j >= y_j >= y_i).
    Left-censored comparators are excluded because their ordering relative to
    an observed subject is undetermined.

    Assumes larger `y_pred` indicates larger target value (e.g., survival time).
    Ties in prediction contribute 0.5.
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

    c_j = c.unsqueeze(0)  # [1, n]
    y_i = y_obs.unsqueeze(1)  # [n_obs, 1]
    y_j = y.unsqueeze(0)  # [1, n]

    # Comparable pairs per Harrell right-censoring definition.
    j_observed_ordered = (c_j == 0) & (y_i < y_j)
    j_right_censored_ordered = (c_j == 1) & (y_j >= y_i)
    comparable = j_observed_ordered | j_right_censored_ordered

    y_hat_less = y_hat_obs.unsqueeze(1) < y_hat.unsqueeze(0)
    y_hat_eq = y_hat_obs.unsqueeze(1) == y_hat.unsqueeze(0)

    n_comparable = comparable.sum(dtype=torch.float32)
    concordant = (comparable & y_hat_less).sum(dtype=torch.float32) + 0.5 * (
        comparable & y_hat_eq
    ).sum(dtype=torch.float32)

    if n_comparable <= 0:
        return torch.tensor(float("nan"), dtype=torch.float32, device=y.device)
    return concordant / n_comparable


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
