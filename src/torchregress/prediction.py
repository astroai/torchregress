"""Prediction containers and density-conversion helpers for reusable tooling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor


def _maybe_collapse_support(support: Tensor) -> Tensor:
    if (
        support.ndim == 2
        and support.shape[0] > 0
        and torch.allclose(support, support[:1].expand_as(support), atol=1.0e-8)
    ):
        return support[0]
    return support


def quantiles_to_density_grid(
    quantiles: Tensor,
    quantile_levels: Sequence[float],
    *,
    n_support: int = 200,
    range_margin: float = 0.05,
) -> tuple[Tensor, Tensor]:
    q = torch.as_tensor(quantiles)
    if torch.is_grad_enabled():
        q = q.detach()
    # ponytail: force float for levels; int quantiles would truncate 0.1->0
    if not q.is_floating_point():
        q = q.float()
    if n_support < 2:
        raise ValueError("n_support must be >= 2")
    if not torch.isfinite(q).all():
        raise ValueError("quantiles must be finite")
    levels = torch.as_tensor(list(quantile_levels), dtype=q.dtype, device=q.device)
    if q.ndim != 2:
        raise ValueError("quantiles must have shape [batch, n_quantiles]")
    if q.shape[1] != levels.size(0):
        raise ValueError("quantile_levels must match quantiles.shape[1]")
    if levels.size(0) < 2:
        raise ValueError("at least two quantile levels are required")
    if not (levels[1:] > levels[:-1]).all():
        raise ValueError("quantile levels must be strictly increasing")

    q = q.cummax(dim=1).values
    q_lo = q[:, 0:1]
    q_hi = q[:, -1:]
    width = (q_hi - q_lo).clamp(min=1.0e-6)
    lo = q_lo - range_margin * width
    hi = q_hi + range_margin * width

    steps = torch.linspace(0, 1, n_support, dtype=q.dtype, device=q.device)
    support = lo + (hi - lo) * steps[None, :]

    slopes = levels.diff() / q.diff(dim=1).clamp(min=1.0e-8)

    dens = torch.zeros_like(support)

    for seg_idx in range(levels.size(0) - 1):
        left = q[:, seg_idx : seg_idx + 1]
        right = q[:, seg_idx + 1 : seg_idx + 2]
        mask = (support >= left) & (support <= right)
        dens = torch.where(mask, slopes[:, seg_idx : seg_idx + 1], dens)

    dens = dens.clamp(min=0.0)

    integral = torch.trapezoid(dens, support, dim=1).clamp(min=1.0e-8)
    density = dens / integral[:, None]
    return support, density


def bars_to_density_grid(
    bar_logits: Tensor,
    bin_edges: Tensor,
    *,
    n_support: int = 200,
    range_margin: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Convert piecewise-constant bar distributions to a regular support grid."""
    logits = torch.as_tensor(bar_logits)
    if torch.is_grad_enabled():
        logits = logits.detach()
    edges = torch.as_tensor(bin_edges)
    if torch.is_grad_enabled():
        edges = edges.detach()
    if logits.ndim != 2:
        raise ValueError("bar_logits must have shape [batch, n_bins]")
    if edges.ndim == 1:
        edges = edges[None, :].expand(logits.shape[0], -1)
    if edges.shape != (logits.shape[0], logits.shape[1] + 1):
        raise ValueError("bin_edges must have shape [batch, n_bins + 1] or [n_bins + 1]")

    logits = logits - logits.amax(dim=1, keepdim=True)
    probs = logits.exp()
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1.0e-8)
    lo = edges[:, 0:1]
    hi = edges[:, -1:]
    width = (hi - lo).clamp(min=1.0e-6)
    lo = lo - range_margin * width
    hi = hi + range_margin * width

    steps = torch.linspace(0, 1, n_support, dtype=logits.dtype, device=logits.device)
    support = lo + (hi - lo) * steps[None, :]
    widths = edges.diff(dim=1).clamp(min=1.0e-8)
    bar_density = probs / widths
    bin_idx = torch.searchsorted(edges[:, 1:-1], support).clamp(0, logits.shape[1] - 1)
    dens = torch.gather(bar_density, 1, bin_idx).clamp(min=0.0)
    integral = torch.trapezoid(dens, support, dim=1).clamp(min=1.0e-8)
    density = dens / integral[:, None]
    return support, density


def samples_to_density_grid(
    samples: Tensor,
    *,
    n_support: int = 200,
    range_margin: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Convert scalar predictive samples to a regular density grid."""
    draws = torch.as_tensor(samples)
    if torch.is_grad_enabled():
        draws = draws.detach()
    if draws.ndim == 3 and draws.shape[-1] == 1:
        draws = draws[..., 0]
    if draws.ndim != 2:
        raise ValueError("samples must have shape [batch, n_samples] or [batch, n_samples, 1]")
    if draws.shape[1] < 2:
        raise ValueError("at least two samples per example are required")

    sample_lo = draws.amin(dim=1)
    sample_hi = draws.amax(dim=1)
    width = (sample_hi - sample_lo).clamp(min=1.0e-6)
    lo = (sample_lo - range_margin * width)[:, None]
    hi = (sample_hi + range_margin * width)[:, None]

    steps = torch.linspace(0, 1, n_support, dtype=draws.dtype, device=draws.device)
    support = lo + (hi - lo) * steps[None, :]
    edges = (
        lo
        + (hi - lo)
        * torch.linspace(0, 1, n_support + 1, dtype=draws.dtype, device=draws.device)[None, :]
    )
    widths = edges.diff(dim=1).clamp(min=1.0e-8)

    # Bin membership follows torch.histogram's convention: half-open
    # [edge_i, edge_i+1) bins with a closed final bin.
    bin_idx = torch.searchsorted(edges[:, 1:-1].contiguous(), draws, right=True).clamp(
        0, n_support - 1
    )
    hist = torch.zeros(
        draws.shape[0], n_support, dtype=draws.dtype, device=draws.device
    ).scatter_add_(1, bin_idx, torch.ones_like(draws))
    dens = hist / float(draws.shape[1]) / widths

    row = dens.repeat_interleave(2, dim=1)
    edge_support = edges.repeat_interleave(2, dim=1)[:, 1:-1]
    idxs = torch.searchsorted(edge_support, support).clamp(0, row.shape[1] - 2)
    density = torch.gather(row, 1, idxs)
    integral = torch.trapezoid(density, support, dim=1).clamp(min=1.0e-8)
    density = density / integral[:, None]
    return support, density


@dataclass(frozen=True)
class PredictiveBatch:
    """Normalized predictive container across regression/test-time tooling."""

    point: np.ndarray | torch.Tensor | None = None
    mean: np.ndarray | torch.Tensor | None = None
    std: np.ndarray | torch.Tensor | None = None
    quantiles: np.ndarray | torch.Tensor | None = None
    quantile_levels: list[float] | None = None
    bar_logits: np.ndarray | torch.Tensor | None = None
    bin_edges: np.ndarray | torch.Tensor | None = None
    samples: np.ndarray | torch.Tensor | None = None
    support: np.ndarray | torch.Tensor | None = None
    density: np.ndarray | torch.Tensor | None = None
    extra: dict[str, Any] | None = None

    def with_density(
        self,
        *,
        n_support: int = 200,
        range_margin: float = 0.05,
    ) -> "PredictiveBatch":
        if self.support is not None and self.density is not None:
            return self
        if self.bar_logits is not None and self.bin_edges is not None:
            support, density = bars_to_density_grid(
                torch.as_tensor(self.bar_logits),
                torch.as_tensor(self.bin_edges),
                n_support=n_support,
                range_margin=range_margin,
            )
            return replace(self, support=_maybe_collapse_support(support), density=density)
        if self.quantiles is not None and self.quantile_levels is not None:
            support, density = quantiles_to_density_grid(
                torch.as_tensor(self.quantiles),
                self.quantile_levels,
                n_support=n_support,
                range_margin=range_margin,
            )
            return replace(self, support=_maybe_collapse_support(support), density=density)
        if self.samples is not None:
            support, density = samples_to_density_grid(
                torch.as_tensor(self.samples),
                n_support=n_support,
                range_margin=range_margin,
            )
            return replace(self, support=_maybe_collapse_support(support), density=density)
        return self


__all__ = [
    "PredictiveBatch",
    "bars_to_density_grid",
    "quantiles_to_density_grid",
    "samples_to_density_grid",
]
