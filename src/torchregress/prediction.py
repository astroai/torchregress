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
    """Convert quantile predictions to a per-row density on a regular support grid.

    The quantile function is linearly interpolated between the knots
    ``(tau_k, q_k)``, which gives a piecewise-constant density
    ``(tau_{k+1} - tau_k) / (q_{k+1} - q_k)`` on ``[q_k, q_{k+1}]``.

    .. note::
       **Tails.**  The mass below the first level (``tau_0``) and above the last
       (``1 - tau_K``) is spread uniformly over the ``range_margin`` margins, so the
       grid CDF equals ``tau_k`` at every knot and a central interval read off the
       grid keeps its nominal level.  The tail *shape* beyond the outer knots is
       unknown from quantiles alone; use wide outer levels (e.g. 0.01 / 0.99) and
       a larger ``range_margin`` when the tails matter.  Before 0.3.0 the tail mass
       was dropped and the grid renormalised (PRD-001).

    Parameters
    ----------
    quantiles : Tensor
        Quantile predictions, shape ``[batch, n_quantiles]``.  Crossing quantiles
        are repaired by sorting each row (monotone rearrangement, as in
        ``constraints.NonCrossingSort``).
    quantile_levels : sequence of float
        Strictly increasing levels ``tau_k``, one per column.
    n_support : int, default 200
        Number of grid points per row (>= 2).
    range_margin : float, default 0.05
        Fraction of ``q_K - q_0`` added on each side of the quantile range.

    Returns
    -------
    support : Tensor
        Grid of shape ``[batch, n_support]``.
    density : Tensor
        Density on the grid, same shape, trapezoid-normalised to integrate to one.

    Raises
    ------
    ValueError
        If the quantiles are non-finite or not 2-D, the levels do not match the
        columns or are not strictly increasing, or ``n_support < 2``.
    """
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

    # Monotone rearrangement (sort), not cummax: cummax creates ties whose mass is dropped.
    q = torch.sort(q, dim=1).values
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

    # Tail mass tau_0 / 1 - tau_K is spread uniformly over the grid margins, so the
    # grid CDF equals tau_k at every knot (the interior keeps its true slopes).
    tail_lo = levels[0] / (q_lo - lo).clamp(min=1.0e-8)
    tail_hi = (1.0 - levels[-1]) / (hi - q_hi).clamp(min=1.0e-8)
    dens = torch.where(support < q_lo, tail_lo, dens)
    dens = torch.where(support > q_hi, tail_hi, dens)
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
    """Convert piecewise-constant bar distributions to a regular support grid.

    The grid spans ``[edges[0], edges[-1]]`` widened by ``range_margin`` times the
    edge range on each side; the density is zero outside the bin edges (bars have no
    mass there) and is renormalised to integrate to one on the grid.
    """
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
    # searchsorted maps points left/right of the edges to the first/last bar; the
    # piecewise-constant law has no mass outside [edges[0], edges[-1]].
    inside = (support >= edges[:, :1]) & (support <= edges[:, -1:])
    dens = torch.where(inside, dens, torch.zeros_like(dens))
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
