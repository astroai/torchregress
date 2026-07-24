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


def _to_numpy(x: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def quantiles_to_density_grid(
    quantiles: Tensor,
    quantile_levels: Sequence[float],
    *,
    n_support: int = 200,
    range_margin: float = 0.05,
) -> tuple[Tensor, Tensor]:
    """Convert monotone quantile predictions to a regular density grid."""
    q = torch.as_tensor(quantiles)
    if torch.is_grad_enabled():
        q = q.detach()
    if q.device.type != "cpu":
        q = q.cpu()
    levels = torch.as_tensor(list(quantile_levels), dtype=q.dtype)
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

    steps = torch.linspace(0, 1, n_support, dtype=q.dtype)
    support = lo + (hi - lo) * steps[None, :]

    slopes = levels.diff() / q.diff(dim=1).clamp(min=1.0e-8)

    dens = slopes[:, 0:1].expand_as(support).clone()

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
    if logits.device.type != "cpu":
        logits = logits.cpu()
        edges = edges.cpu()
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

    steps = torch.linspace(0, 1, n_support, dtype=logits.dtype)
    support = lo + (hi - lo) * steps[None, :]
    density = torch.empty_like(support)
    for idx in range(logits.shape[0]):
        widths = edges[idx].diff().clamp(min=1.0e-8)
        bar_density = probs[idx] / widths
        bin_idx = torch.bucketize(support[idx], edges[idx][1:-1], right=False).clamp(
            0, logits.shape[1] - 1
        )
        dens = bar_density[bin_idx].clamp(min=0.0)
        integral = torch.trapezoid(dens, support[idx])
        integral_val = float(integral.item())
        density[idx] = dens / max(integral_val, 1.0e-8)
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
    if draws.device.type != "cpu":
        draws = draws.cpu()
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

    steps = torch.linspace(0, 1, n_support, dtype=draws.dtype)
    support = lo + (hi - lo) * steps[None, :]
    density = torch.empty_like(support)
    for idx in range(draws.shape[0]):
        edges = torch.linspace(lo[idx].item(), hi[idx].item(), n_support + 1, dtype=draws.dtype)
        hist = torch.histogram(draws[idx], bins=edges, density=False)[0]
        widths = edges.diff().clamp(min=1.0e-8)
        dens = hist.float() / max(float(draws.shape[1]), 1.0) / widths
        row = dens.repeat_interleave(2)
        edge_support = edges.repeat_interleave(2)[1:-1]
        density[idx] = torch.zeros_like(support[idx])
        # interp via searchsorted
        idxs = torch.searchsorted(edge_support, support[idx]).clamp(0, row.size(0) - 2)
        density[idx] = row[idxs]
        integral = torch.trapezoid(density[idx], support[idx])
        integral_val = float(integral.item())
        density[idx] = density[idx] / max(integral_val, 1.0e-8)
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
