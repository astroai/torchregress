"""Prediction containers and density-conversion helpers for reusable tooling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
import torch


def _maybe_collapse_support(support: np.ndarray) -> np.ndarray:
    if (
        support.ndim == 2
        and support.shape[0] > 0
        and np.allclose(support, support[:1], atol=1.0e-8)
    ):  # noqa: E501
        return support[0]
    return support


def _to_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
    return array.detach().cpu().numpy() if torch.is_tensor(array) else np.asarray(array)


def quantiles_to_density_grid(
    quantiles: np.ndarray | torch.Tensor,
    quantile_levels: Sequence[float],
    *,
    n_support: int = 200,
    range_margin: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert monotone quantile predictions to a regular density grid."""
    q = _to_numpy(quantiles)
    levels = np.asarray(list(quantile_levels), dtype=float)
    if q.ndim != 2:
        raise ValueError("quantiles must have shape [batch, n_quantiles]")
    if q.shape[1] != levels.size:
        raise ValueError("quantile_levels must match quantiles.shape[1]")
    if levels.size < 2:
        raise ValueError("at least two quantile levels are required")
    if not np.all(levels[1:] > levels[:-1]):
        raise ValueError("quantile levels must be strictly increasing")

    q = np.maximum.accumulate(q, axis=1)
    q_lo = q[:, 0]
    q_hi = q[:, -1]
    width = np.maximum(q_hi - q_lo, 1.0e-6)
    lo = q_lo - range_margin * width
    hi = q_hi + range_margin * width

    steps = np.linspace(0, 1, n_support, dtype=float)
    support = lo[:, None] + (hi - lo)[:, None] * steps[None, :]

    slopes = np.diff(levels) / np.clip(np.diff(q, axis=1), 1.0e-8, None)

    dens = np.broadcast_to(slopes[:, 0:1], support.shape).copy()

    for seg_idx in range(levels.size - 1):
        left = q[:, seg_idx : seg_idx + 1]
        right = q[:, seg_idx + 1 : seg_idx + 2]
        mask = (support >= left) & (support <= right)
        np.copyto(dens, slopes[:, seg_idx : seg_idx + 1], where=mask)

    dens = np.clip(dens, 0.0, None)

    integral = np.trapezoid(dens, support, axis=1)
    integral = np.maximum(integral, 1.0e-8)

    density = dens / integral[:, None]
    return support, density


def bars_to_density_grid(
    bar_logits: np.ndarray | torch.Tensor,
    bin_edges: np.ndarray | torch.Tensor,
    *,
    n_support: int = 200,
    range_margin: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert piecewise-constant bar distributions to a regular support grid."""
    logits = _to_numpy(bar_logits)
    edges = _to_numpy(bin_edges)
    if logits.ndim != 2:
        raise ValueError("bar_logits must have shape [batch, n_bins]")
    if edges.ndim == 1:
        edges = np.broadcast_to(edges[None, :], (logits.shape[0], edges.shape[0]))
    if edges.shape != (logits.shape[0], logits.shape[1] + 1):
        raise ValueError("bin_edges must have shape [batch, n_bins + 1] or [n_bins + 1]")

    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1.0e-8, None)
    lo = edges[:, 0]
    hi = edges[:, -1]
    width = np.maximum(hi - lo, 1.0e-6)
    lo = lo - range_margin * width
    hi = hi + range_margin * width
    support = np.stack(
        [np.linspace(a, b, n_support, dtype=float) for a, b in zip(lo, hi, strict=False)],
        axis=0,
    )
    density = np.empty_like(support)
    for idx in range(logits.shape[0]):
        widths = np.clip(np.diff(edges[idx]), 1.0e-8, None)
        bar_density = probs[idx] / widths
        bin_idx = np.clip(
            np.digitize(support[idx], edges[idx][1:-1], right=False), 0, logits.shape[1] - 1
        )
        dens = np.clip(bar_density[bin_idx], 0.0, None)
        integral = np.trapezoid(dens, support[idx])
        density[idx] = dens / max(integral, 1.0e-8)
    return support, density


def samples_to_density_grid(
    samples: np.ndarray | torch.Tensor,
    *,
    n_support: int = 200,
    range_margin: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert scalar predictive samples to a regular density grid."""
    draws = _to_numpy(samples)
    if draws.ndim == 3 and draws.shape[-1] == 1:
        draws = draws[..., 0]
    if draws.ndim != 2:
        raise ValueError("samples must have shape [batch, n_samples] or [batch, n_samples, 1]")
    if draws.shape[1] < 2:
        raise ValueError("at least two samples per example are required")

    sample_lo = draws.min(axis=1)
    sample_hi = draws.max(axis=1)
    width = np.maximum(sample_hi - sample_lo, 1.0e-6)
    lo = sample_lo - range_margin * width
    hi = sample_hi + range_margin * width

    support = np.stack(
        [np.linspace(a, b, n_support, dtype=float) for a, b in zip(lo, hi, strict=False)],
        axis=0,
    )
    density = np.empty_like(support)
    for idx in range(draws.shape[0]):
        edges = np.linspace(lo[idx], hi[idx], n_support + 1, dtype=float)
        hist, _ = np.histogram(draws[idx], bins=edges, density=False)
        widths = np.clip(np.diff(edges), 1.0e-8, None)
        dens = hist.astype(float) / max(float(draws.shape[1]), 1.0) / widths
        row = np.repeat(dens, 2)
        edge_support = np.repeat(edges, 2)[1:-1]
        density[idx] = np.interp(support[idx], edge_support, row, left=0.0, right=0.0)
        integral = np.trapezoid(density[idx], support[idx])
        density[idx] = density[idx] / max(integral, 1.0e-8)
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
                self.bar_logits,
                self.bin_edges,
                n_support=n_support,
                range_margin=range_margin,
            )
            return replace(self, support=_maybe_collapse_support(support), density=density)
        if self.quantiles is not None and self.quantile_levels is not None:
            support, density = quantiles_to_density_grid(
                self.quantiles,
                self.quantile_levels,
                n_support=n_support,
                range_margin=range_margin,
            )
            return replace(self, support=_maybe_collapse_support(support), density=density)
        if self.samples is not None:
            support, density = samples_to_density_grid(
                self.samples,
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
