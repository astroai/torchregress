"""Parse heteroscedastic Gaussian head outputs into mean and log-variance."""

from __future__ import annotations

from typing import Literal, Tuple, Union

import torch
from torch import Tensor


def variance_from_logvar(
    log_var: Tensor,
    *,
    min_logvar: float = -8.0,
    max_logvar: float = 6.0,
    eps: float = 1.0e-8,
) -> Tensor:
    """Convert log-variance to variance with training-time stabilization."""
    return torch.exp(log_var.clamp(min=min_logvar, max=max_logvar)).clamp_min(eps)


def split_mean_log_variance(
    y_pred: Union[Tensor, Tuple[Tensor, Tensor], dict[str, Tensor]],
    *,
    split_dim: int = -1,
    mean_only_log_var: Literal["zeros", "error"] = "error",
) -> Tuple[Tensor, Tensor]:
    """Split model output into ``(mean, log_variance)`` tensors.

    Supports tuple ``(mean, log_var)``, dict keys ``means``/``log_vars``, and
    concatenated ``[..., 2 * n_targets]`` layouts along ``split_dim``.
    """
    if isinstance(y_pred, tuple):
        if len(y_pred) != 2:
            raise ValueError("Tuple predictions must be (mean, log_variance)")
        mean, log_var = y_pred
        return mean, log_var

    if isinstance(y_pred, dict):
        if "means" in y_pred and "log_vars" in y_pred:
            return y_pred["means"], y_pred["log_vars"]
        raise ValueError("Dict predictions must contain 'means' and 'log_vars' keys")

    if not isinstance(y_pred, torch.Tensor):
        raise TypeError(f"Unsupported prediction type: {type(y_pred)!r}")

    dim_size = y_pred.shape[split_dim]
    if dim_size % 2 != 0:
        if mean_only_log_var == "zeros":
            return y_pred, torch.zeros_like(y_pred)
        raise ValueError(
            f"Concatenated predictions must have even size along split_dim={split_dim}, "
            f"got {dim_size}."
        )
    mean, log_var = torch.chunk(y_pred, 2, dim=split_dim)
    return mean, log_var


def low_rank_output_dim(n_features: int, rank: int) -> int:
    """Compute output dimension for low-rank Gaussian heads."""
    if n_features <= 0 or rank <= 0:
        raise ValueError("n_features and rank must be positive integers")
    return n_features + n_features * rank + n_features


def split_low_rank_gaussian_output(
    y_pred: Tensor,
    n_features: int,
    rank: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Split concatenated output into mean, low-rank factor, and diagonal."""
    expected = low_rank_output_dim(n_features, rank)
    if y_pred.shape[-1] != expected:
        raise ValueError(
            f"Expected last dimension {expected} for low-rank output, got {y_pred.shape[-1]}"
        )

    mean = y_pred[..., :n_features]
    factor_flat = y_pred[..., n_features : n_features + n_features * rank]
    cov_factor = factor_flat.reshape(*y_pred.shape[:-1], n_features, rank)
    cov_diag = y_pred[..., -n_features:]
    return mean, cov_factor, cov_diag


def parse_heteroscedastic_output(
    output: Union[Tensor, Tuple[Tensor, Tensor], dict[str, Tensor]],
) -> Tuple[Tensor, Tensor]:
    """Parse ensemble-style heteroscedastic outputs.

    Backward-compatible wrapper around :func:`split_mean_log_variance` that also
    accepts legacy 2-D tensors ``[batch, 2 * n_outputs]``.
    """
    if isinstance(output, torch.Tensor) and output.ndim >= 2 and output.shape[1] % 2 == 0:
        dim = output.shape[1] // 2
        return output[:, :dim], output[:, dim:]
    try:
        return split_mean_log_variance(output)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Model output format not recognized for heteroscedastic uncertainty. "
            "Expected tuple (mean, log_var), dict with 'means' and 'log_vars', "
            "or tensor with even number of features [mean, log_var]."
        ) from exc


__all__ = [
    "low_rank_output_dim",
    "parse_heteroscedastic_output",
    "split_low_rank_gaussian_output",
    "split_mean_log_variance",
    "variance_from_logvar",
]
