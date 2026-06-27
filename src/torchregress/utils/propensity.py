"""Propensity estimation and inverse-probability weighting utilities."""

from __future__ import annotations

from torch import Tensor


def ipw_weights(
    propensity: Tensor,
    observed: Tensor | None = None,
    *,
    clip_min: float = 0.01,
    clip_max: float = 0.99,
    normalize: bool = True,
) -> Tensor:
    """Compute inverse-probability weights.

    If `observed` is provided, computes symmetric weights:
    - observed=1 -> 1 / p
    - observed=0 -> 1 / (1 - p)

    If `observed` is omitted, returns 1 / p.
    """
    if not (0.0 < clip_min < clip_max < 1.0):
        raise ValueError("clip_min/clip_max must satisfy 0 < clip_min < clip_max < 1")

    p = propensity.float().clamp(min=clip_min, max=clip_max)

    w: Tensor
    if observed is None:
        w = 1.0 / p
    else:
        obs = observed.float()
        if obs.shape != p.shape:
            raise ValueError("observed and propensity must have matching shapes")
        w = obs / p + (1.0 - obs) / (1.0 - p)

    if normalize:
        w = w / w.mean().clamp_min(1e-8)
    return w


__all__ = ["ipw_weights"]
