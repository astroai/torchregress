"""Diagnostics for causal overlap and effective sample support."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def _effective_sample_size(weights: Tensor) -> float:
    w = weights.reshape(-1).float()
    denom = torch.sum(w**2).clamp_min(1e-8)
    ess = torch.sum(w) ** 2 / denom
    return float(ess.item())


def causal_overlap_report(
    propensity: Tensor,
    treatment: Tensor,
    *,
    trim_threshold: float = 0.05,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Summarize overlap quality for binary-treatment causal estimation."""
    p = propensity.reshape(-1).float().clamp(min=eps, max=1.0 - eps)
    t = treatment.reshape(-1).float()
    if p.shape != t.shape:
        raise ValueError("propensity and treatment must share shape")
    if not 0.0 <= trim_threshold < 0.5:
        raise ValueError("trim_threshold must be in [0, 0.5)")

    treated = t > 0.5
    control = ~treated
    overlap_mask = (p >= trim_threshold) & (p <= 1.0 - trim_threshold)

    w_t = torch.where(treated, 1.0 / p, torch.zeros_like(p))
    w_c = torch.where(control, 1.0 / (1.0 - p), torch.zeros_like(p))

    treated_count = int(treated.sum().item())
    control_count = int(control.sum().item())
    ess_t = _effective_sample_size(w_t[treated]) if treated_count > 0 else 0.0
    ess_c = _effective_sample_size(w_c[control]) if control_count > 0 else 0.0

    return {
        "n_samples": float(p.numel()),
        "n_treated": float(treated_count),
        "n_control": float(control_count),
        "propensity_min": float(p.min().item()),
        "propensity_max": float(p.max().item()),
        "propensity_mean": float(p.mean().item()),
        "overlap_rate": float(overlap_mask.float().mean().item()),
        "trim_threshold": float(trim_threshold),
        "n_trimmed": float((~overlap_mask).sum().item()),
        "treated_ess": ess_t,
        "control_ess": ess_c,
        "min_group_ess": float(min(ess_t, ess_c)),
    }


__all__ = ["causal_overlap_report"]
