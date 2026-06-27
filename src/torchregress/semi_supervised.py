"""Modular semi-supervised regression and teacher-student consistency workflows."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import cycle
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .prediction import PredictiveBatch
from .utils import update_ema_teacher_


@dataclass(frozen=True)
class SAGERegAgreement:
    """Distributional agreement statistics for one unlabeled batch."""

    loss: Tensor
    disagreement: Tensor
    weights: Tensor
    consensus: PredictiveBatch


@dataclass(frozen=True)
class SAGERegOutput:
    """Composite supervised + self-agreement objective."""

    total_loss: Tensor
    supervised_loss: Tensor
    agreement: SAGERegAgreement


def disagreement_to_weight(
    disagreement: Tensor,
    tau: float,
    *,
    power: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Convert disagreement scores into trust weights."""
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    if power <= 0.0:
        raise ValueError("power must be positive")
    weight = torch.exp(-disagreement / tau).clamp_max(1.0)
    if power != 1.0:
        weight = weight.pow(power)
    return weight


def uncertainty_to_weight(
    predictive_batch: PredictiveBatch,
    tau: float,
    *,
    power: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Convert teacher predictive uncertainty (std) into trust weights."""
    if predictive_batch.std is None:
        raise ValueError("predictive_batch must have standard deviation 'std' defined")
    std = torch.as_tensor(predictive_batch.std)
    if std.ndim > 1:
        std = std.mean(dim=list(range(1, std.ndim)))
    return disagreement_to_weight(std, tau, power=power, eps=eps)


def conformal_width_to_weight(
    lower: Tensor,
    upper: Tensor,
    tau: float | None = None,
    *,
    threshold: float | None = None,
) -> Tensor:
    """Compute pseudo-label weights or masks based on conformal prediction interval width.

    Parameters
    ----------
    lower : Tensor
        Lower prediction intervals, shape (N,).
    upper : Tensor
        Upper prediction intervals, shape (N,).
    tau : float, optional
        If set, returns soft weights exp(-width / tau).
    threshold : float, optional
        If set, returns binary mask (width <= threshold).
    """
    width = (upper - lower).clamp_min(0.0)
    if tau is not None:
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        return torch.exp(-width / tau)
    if threshold is not None:
        return (width <= threshold).to(dtype=torch.float32)
    raise ValueError("Either 'tau' or 'threshold' must be specified")


def _candidate_tensors(batch: PredictiveBatch) -> tuple[np.ndarray | Tensor | None, ...]:
    return (
        batch.mean,
        batch.std,
        batch.quantiles,
        batch.bar_logits,
        batch.density,
        batch.point,
        batch.samples,
        batch.support,
        batch.bin_edges,
    )


def _reference_tensor(views: Sequence[PredictiveBatch]) -> Tensor:
    for view in views:
        for value in _candidate_tensors(view):
            if torch.is_tensor(value):
                return value
    return torch.tensor(0.0, dtype=torch.float32)


def _to_tensor(
    value: np.ndarray | Tensor,
    *,
    reference: Tensor,
    name: str,
) -> Tensor:
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return torch.as_tensor(array, device=reference.device, dtype=reference.dtype)


def _scalar_head_tensor(
    value: np.ndarray | Tensor,
    *,
    reference: Tensor,
    name: str,
) -> Tensor:
    tensor = _to_tensor(value, reference=reference, name=name)
    if tensor.ndim == 1:
        return tensor
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        return tensor[:, 0]
    raise ValueError(
        f"{name} must represent scalar regression outputs with shape [batch] or [batch, 1]"
    )


def _matrix_tensor(
    value: np.ndarray | Tensor,
    *,
    reference: Tensor,
    name: str,
) -> Tensor:
    tensor = _to_tensor(value, reference=reference, name=name)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, width]")
    return tensor


def _grid_tensor(
    value: np.ndarray | Tensor,
    *,
    batch_size: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    tensor = _to_tensor(value, reference=reference, name=name)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0).expand(batch_size, -1)
    if tensor.ndim == 2 and tensor.shape[0] == batch_size:
        return tensor
    raise ValueError(f"{name} must have shape [width] or [batch, width]")


def _value_batch_size(value: np.ndarray | Tensor | None) -> int | None:
    if value is None:
        return None
    shape = tuple(value.shape) if hasattr(value, "shape") else ()
    if len(shape) == 0:
        return None
    if len(shape) == 1:
        return None
    return int(shape[0])


def _infer_batch_size(views: Sequence[PredictiveBatch]) -> int:
    batch_size: int | None = None
    for view in views:
        for value in (
            view.mean,
            view.std,
            view.quantiles,
            view.bar_logits,
            view.density,
            view.point,
            view.samples,
        ):
            candidate = _value_batch_size(value)
            if candidate is None and value is not None and getattr(value, "ndim", 0) == 1:
                candidate = int(value.shape[0])
            if candidate is None:
                continue
            if batch_size is None:
                batch_size = candidate
            elif batch_size != candidate:
                raise ValueError("all predictive views must share the same batch size")
    if batch_size is None:
        raise ValueError("could not infer batch size from predictive views")
    return batch_size


def _normalize_density(density: Tensor, support: Tensor, eps: float) -> Tensor:
    non_negative = density.clamp_min(0.0)
    integral = torch.trapezoid(non_negative, support, dim=-1).clamp_min(eps)
    return non_negative / integral.unsqueeze(-1)


def _moments_from_density(
    support: Tensor,
    density: Tensor,
    *,
    min_scale: float,
) -> tuple[Tensor, Tensor]:
    mean = torch.trapezoid(density * support, support, dim=-1)
    centered = support - mean.unsqueeze(-1)
    var = torch.trapezoid(density * centered.square(), support, dim=-1).clamp_min(min_scale**2)
    return mean.unsqueeze(-1), torch.sqrt(var).unsqueeze(-1)


def _support_grid(lo: Tensor, hi: Tensor, *, n_support: int) -> Tensor:
    steps = torch.linspace(0.0, 1.0, n_support, device=lo.device, dtype=lo.dtype)
    return lo.unsqueeze(-1) + (hi - lo).unsqueeze(-1) * steps.unsqueeze(0)


def _predictive_bounds(
    view: PredictiveBatch,
    *,
    batch_size: int,
    reference: Tensor,
    gaussian_std_span: float,
    range_margin: float,
    min_scale: float,
) -> tuple[Tensor, Tensor]:
    if view.support is not None:
        support = _grid_tensor(
            view.support,
            batch_size=batch_size,
            reference=reference,
            name="support",
        )
        return support[:, 0], support[:, -1]

    if view.mean is not None and view.std is not None:
        mean = _scalar_head_tensor(view.mean, reference=reference, name="mean")
        std = _scalar_head_tensor(view.std, reference=reference, name="std").clamp_min(min_scale)
        span = gaussian_std_span * std
        return mean - span, mean + span

    if view.quantiles is not None:
        quantiles = torch.cummax(
            _matrix_tensor(view.quantiles, reference=reference, name="quantiles"),
            dim=1,
        ).values
        lo = quantiles[:, 0]
        hi = quantiles[:, -1]
        width = (hi - lo).clamp_min(min_scale)
        return lo - range_margin * width, hi + range_margin * width

    if view.bar_logits is not None and view.bin_edges is not None:
        edges = _grid_tensor(
            view.bin_edges,
            batch_size=batch_size,
            reference=reference,
            name="bin_edges",
        )
        lo = edges[:, 0]
        hi = edges[:, -1]
        width = (hi - lo).clamp_min(min_scale)
        return lo - range_margin * width, hi + range_margin * width

    raise ValueError(
        "unsupported PredictiveBatch for SAGE-Reg; expected density/support, mean/std, "
        "quantiles/quantile_levels, or bar_logits/bin_edges"
    )


def _density_from_support(
    view: PredictiveBatch,
    support: Tensor,
    *,
    batch_size: int,
    reference: Tensor,
    eps: float,
) -> Tensor:
    if view.support is None or view.density is None:
        raise ValueError("support/density view requires both support and density")
    src_support = _grid_tensor(
        view.support,
        batch_size=batch_size,
        reference=reference,
        name="support",
    )
    src_density = _grid_tensor(
        view.density,
        batch_size=batch_size,
        reference=reference,
        name="density",
    )
    if src_support.shape != src_density.shape:
        raise ValueError("support and density must share the same shape")

    idx = torch.searchsorted(src_support, support, right=False).clamp(1, src_support.shape[1] - 1)
    left = idx - 1
    x0 = torch.gather(src_support, 1, left)
    x1 = torch.gather(src_support, 1, idx)
    y0 = torch.gather(src_density, 1, left)
    y1 = torch.gather(src_density, 1, idx)
    weight = (support - x0) / (x1 - x0).clamp_min(eps)
    interp = y0 + weight * (y1 - y0)
    valid = (support >= src_support[:, :1]) & (support <= src_support[:, -1:])
    resampled = torch.where(valid, interp, torch.zeros_like(interp))
    return _normalize_density(resampled, support, eps)


def _density_from_gaussian(
    view: PredictiveBatch,
    support: Tensor,
    *,
    reference: Tensor,
    min_scale: float,
    eps: float,
) -> Tensor:
    if view.mean is None or view.std is None:
        raise ValueError("gaussian view requires mean and std")
    mean = _scalar_head_tensor(view.mean, reference=reference, name="mean").unsqueeze(-1)
    std = (
        _scalar_head_tensor(view.std, reference=reference, name="std")
        .clamp_min(min_scale)
        .unsqueeze(-1)
    )
    z = (support - mean) / std
    density = torch.exp(-0.5 * z.square()) / (std * math.sqrt(2.0 * math.pi))
    return _normalize_density(density, support, eps)


def _density_from_quantiles(
    view: PredictiveBatch,
    support: Tensor,
    *,
    reference: Tensor,
    min_scale: float,
    eps: float,
) -> Tensor:
    if view.quantiles is None or view.quantile_levels is None:
        raise ValueError("quantile view requires quantiles and quantile_levels")
    quantiles = torch.cummax(
        _matrix_tensor(view.quantiles, reference=reference, name="quantiles"),
        dim=1,
    ).values
    levels = torch.as_tensor(view.quantile_levels, device=reference.device, dtype=reference.dtype)
    if levels.ndim != 1 or levels.numel() < 2:
        raise ValueError("quantile_levels must be a 1D sequence with at least two entries")
    if quantiles.shape[1] != levels.numel():
        raise ValueError("quantile_levels must match quantiles.shape[1]")
    if bool(torch.any(levels[1:] <= levels[:-1]).item()):
        raise ValueError("quantile_levels must be strictly increasing")

    delta_levels = torch.diff(levels).unsqueeze(0)
    delta_quantiles = torch.diff(quantiles, dim=1).clamp_min(min_scale)
    slopes = delta_levels / delta_quantiles

    left_tail = slopes[:, :1]
    right_tail = slopes[:, -1:]
    bin_idx = (
        torch.searchsorted(quantiles, support, right=False).clamp(1, quantiles.shape[1] - 1).sub(1)
    )
    density = torch.gather(slopes, 1, bin_idx)
    density = torch.where(support <= quantiles[:, :1], left_tail, density)
    density = torch.where(support >= quantiles[:, -1:], right_tail, density)
    return _normalize_density(density, support, eps)


def _density_from_bars(
    view: PredictiveBatch,
    support: Tensor,
    *,
    batch_size: int,
    reference: Tensor,
    min_scale: float,
    eps: float,
) -> Tensor:
    if view.bar_logits is None or view.bin_edges is None:
        raise ValueError("bar view requires bar_logits and bin_edges")
    logits = _matrix_tensor(view.bar_logits, reference=reference, name="bar_logits")
    edges = _grid_tensor(
        view.bin_edges,
        batch_size=batch_size,
        reference=reference,
        name="bin_edges",
    )
    if edges.shape[1] != logits.shape[1] + 1:
        raise ValueError("bin_edges must have shape [n_bins + 1] or [batch, n_bins + 1]")

    probs = torch.softmax(logits, dim=1)
    widths = (edges[:, 1:] - edges[:, :-1]).clamp_min(min_scale)
    density_per_bin = probs / widths

    points = support.unsqueeze(1)
    left = edges[:, :-1].unsqueeze(-1)
    right = edges[:, 1:].unsqueeze(-1)
    density = (density_per_bin.unsqueeze(-1) * ((points >= left) & (points < right))).sum(dim=1)

    on_right_edge = support == edges[:, -1].unsqueeze(-1)
    if bool(on_right_edge.any().item()):
        density = density + on_right_edge.to(density.dtype) * density_per_bin[:, -1:].expand_as(
            density
        )
    return _normalize_density(density, support, eps)


def _density_on_common_support(
    view: PredictiveBatch,
    support: Tensor,
    *,
    batch_size: int,
    reference: Tensor,
    min_scale: float,
    eps: float,
) -> Tensor:
    if view.support is not None and view.density is not None:
        return _density_from_support(
            view,
            support,
            batch_size=batch_size,
            reference=reference,
            eps=eps,
        )
    if view.mean is not None and view.std is not None:
        return _density_from_gaussian(
            view,
            support,
            reference=reference,
            min_scale=min_scale,
            eps=eps,
        )
    if view.quantiles is not None and view.quantile_levels is not None:
        return _density_from_quantiles(
            view,
            support,
            reference=reference,
            min_scale=min_scale,
            eps=eps,
        )
    if view.bar_logits is not None and view.bin_edges is not None:
        return _density_from_bars(
            view,
            support,
            batch_size=batch_size,
            reference=reference,
            min_scale=min_scale,
            eps=eps,
        )
    raise ValueError(
        "unsupported PredictiveBatch for SAGE-Reg; expected density/support, mean/std, "
        "quantiles/quantile_levels, or bar_logits/bin_edges"
    )


def _common_density_stack(
    predictive_views: Sequence[PredictiveBatch],
    *,
    n_support: int,
    range_margin: float,
    gaussian_std_span: float,
    min_scale: float,
    eps: float,
) -> tuple[Tensor, Tensor]:
    if len(predictive_views) == 0:
        raise ValueError("predictive_views must not be empty")

    reference = _reference_tensor(predictive_views)
    batch_size = _infer_batch_size(predictive_views)

    lo_bounds: list[Tensor] = []
    hi_bounds: list[Tensor] = []
    for view in predictive_views:
        lo, hi = _predictive_bounds(
            view,
            batch_size=batch_size,
            reference=reference,
            gaussian_std_span=gaussian_std_span,
            range_margin=range_margin,
            min_scale=min_scale,
        )
        lo_bounds.append(lo)
        hi_bounds.append(hi)

    lo = torch.stack(lo_bounds, dim=0).amin(dim=0)
    hi = torch.stack(hi_bounds, dim=0).amax(dim=0)
    support = _support_grid(lo, hi.clamp_min(lo + min_scale), n_support=n_support)
    densities = torch.stack(
        [
            _density_on_common_support(
                view,
                support,
                batch_size=batch_size,
                reference=reference,
                min_scale=min_scale,
                eps=eps,
            )
            for view in predictive_views
        ],
        dim=0,
    )
    return support, densities


def build_consensus_predictive_batch(
    predictive_views: Sequence[PredictiveBatch],
    *,
    n_support: int = 128,
    range_margin: float = 0.05,
    gaussian_std_span: float = 4.0,
    min_scale: float = 1e-4,
    eps: float = 1e-8,
) -> PredictiveBatch:
    """Build the consensus predictive law for a set of stochastic views."""
    support, densities = _common_density_stack(
        predictive_views,
        n_support=n_support,
        range_margin=range_margin,
        gaussian_std_span=gaussian_std_span,
        min_scale=min_scale,
        eps=eps,
    )
    consensus_density = _normalize_density(densities.mean(dim=0), support, eps)
    consensus_mean, consensus_std = _moments_from_density(
        support,
        consensus_density,
        min_scale=min_scale,
    )
    return PredictiveBatch(
        mean=consensus_mean,
        std=consensus_std,
        support=support,
        density=consensus_density,
        extra={"view_count": len(predictive_views)},
    )


def predictive_agreement_score(
    predictive_views: Sequence[PredictiveBatch],
    *,
    n_support: int = 128,
    range_margin: float = 0.05,
    gaussian_std_span: float = 4.0,
    min_scale: float = 1e-4,
    eps: float = 1e-8,
    reduction: str = "none",
) -> Tensor:
    """Average pairwise symmetric KL disagreement across predictive views."""
    if len(predictive_views) < 2:
        raise ValueError("predictive_agreement_score requires at least two predictive views")

    support, densities = _common_density_stack(
        predictive_views,
        n_support=n_support,
        range_margin=range_margin,
        gaussian_std_span=gaussian_std_span,
        min_scale=min_scale,
        eps=eps,
    )
    pairwise_terms: list[Tensor] = []
    for left in range(len(predictive_views) - 1):
        p = densities[left]
        log_p = torch.log(p.clamp_min(eps))
        for right in range(left + 1, len(predictive_views)):
            q = densities[right]
            log_q = torch.log(q.clamp_min(eps))
            kl_pq = torch.trapezoid(p * (log_p - log_q), support, dim=-1)
            kl_qp = torch.trapezoid(q * (log_q - log_p), support, dim=-1)
            pairwise_terms.append(0.5 * (kl_pq + kl_qp))

    disagreement = torch.stack(pairwise_terms, dim=0).mean(dim=0)
    if reduction == "mean":
        return disagreement.mean()
    if reduction == "sum":
        return disagreement.sum()
    if reduction != "none":
        raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")
    return disagreement


def _reduce_pseudo_loss_rows(
    row_loss: Tensor,
    sample_weights: Tensor | None,
    reduction: str,
    eps: float,
) -> Tensor:
    if sample_weights is not None:
        weights = sample_weights.to(device=row_loss.device, dtype=row_loss.dtype).reshape(-1)
        if weights.shape[0] != row_loss.shape[0]:
            raise ValueError("sample_weights must match the batch dimension")
        weighted = row_loss * weights
        if reduction == "mean":
            return weighted.sum() / weights.sum().clamp_min(eps)
        if reduction == "sum":
            return weighted.sum()
        if reduction == "none":
            return weighted
        raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")

    if reduction == "mean":
        return row_loss.mean()
    if reduction == "sum":
        return row_loss.sum()
    if reduction == "none":
        return row_loss
    raise ValueError("reduction must be one of {'none', 'mean', 'sum'}")


def _gaussian_pseudo_loss(
    student_prediction: PredictiveBatch,
    consensus_prediction: PredictiveBatch,
    reference: Tensor,
    min_scale: float,
) -> Tensor:
    assert student_prediction.mean is not None
    assert student_prediction.std is not None
    assert consensus_prediction.mean is not None
    assert consensus_prediction.std is not None

    student_mean = _scalar_head_tensor(
        student_prediction.mean,
        reference=reference,
        name="mean",
    )
    student_std = _scalar_head_tensor(
        student_prediction.std, reference=reference, name="std"
    ).clamp_min(min_scale)
    target_mean = _scalar_head_tensor(
        consensus_prediction.mean, reference=reference, name="consensus_mean"
    )
    target_std = _scalar_head_tensor(
        consensus_prediction.std, reference=reference, name="consensus_std"
    ).clamp_min(min_scale)
    student_var = student_std.square()
    target_var = target_std.square()
    return 0.5 * (
        math.log(2.0 * math.pi)
        + torch.log(student_var)
        + (target_var + (target_mean - student_mean).square()) / student_var
    )


def _binned_pseudo_loss(
    student_prediction: PredictiveBatch,
    consensus_prediction: PredictiveBatch,
    reference: Tensor,
    n_support: int,
    range_margin: float,
    gaussian_std_span: float,
    min_scale: float,
    eps: float,
) -> Tensor:
    batch_size = _infer_batch_size([student_prediction, consensus_prediction])
    assert student_prediction.bin_edges is not None
    assert student_prediction.bar_logits is not None

    edges = _grid_tensor(
        student_prediction.bin_edges,
        batch_size=batch_size,
        reference=reference,
        name="bin_edges",
    )
    support, densities = _common_density_stack(
        [student_prediction, consensus_prediction],
        n_support=n_support,
        range_margin=range_margin,
        gaussian_std_span=gaussian_std_span,
        min_scale=min_scale,
        eps=eps,
    )
    target_density = densities[1]
    target_probs = []
    for idx in range(edges.shape[1] - 1):
        left = edges[:, idx : idx + 1]
        right = edges[:, idx + 1 : idx + 2]
        mask = (support >= left) & (
            support <= right if idx == edges.shape[1] - 2 else support < right
        )
        bin_density = torch.where(mask, target_density, torch.zeros_like(target_density))
        target_probs.append(torch.trapezoid(bin_density, support, dim=-1))
    target_pmf = torch.stack(target_probs, dim=-1)
    target_pmf = target_pmf / target_pmf.sum(dim=-1, keepdim=True).clamp_min(eps)
    student_logits = _matrix_tensor(
        student_prediction.bar_logits,
        reference=reference,
        name="bar_logits",
    )
    return -(target_pmf * torch.log_softmax(student_logits, dim=-1)).sum(dim=-1)


def _general_pseudo_loss(
    student_prediction: PredictiveBatch,
    consensus_prediction: PredictiveBatch,
    n_support: int,
    range_margin: float,
    gaussian_std_span: float,
    min_scale: float,
    eps: float,
) -> Tensor:
    support, densities = _common_density_stack(
        [student_prediction, consensus_prediction],
        n_support=n_support,
        range_margin=range_margin,
        gaussian_std_span=gaussian_std_span,
        min_scale=min_scale,
        eps=eps,
    )
    student_density = densities[0]
    target_density = densities[1]
    return -torch.trapezoid(
        target_density * torch.log(student_density.clamp_min(eps)),
        support,
        dim=-1,
    )


def distributional_pseudo_loss(
    student_prediction: PredictiveBatch,
    consensus_prediction: PredictiveBatch,
    *,
    sample_weights: Tensor | None = None,
    reduction: str = "mean",
    n_support: int = 128,
    range_margin: float = 0.05,
    gaussian_std_span: float = 4.0,
    min_scale: float = 1e-4,
    eps: float = 1e-8,
) -> Tensor:
    """Backbone-aware pseudo-supervision loss on a shared predictive representation."""
    reference = _reference_tensor([student_prediction, consensus_prediction])

    if (
        student_prediction.mean is not None
        and student_prediction.std is not None
        and consensus_prediction.mean is not None
        and consensus_prediction.std is not None
    ):
        row_loss = _gaussian_pseudo_loss(
            student_prediction, consensus_prediction, reference, min_scale
        )
    elif student_prediction.bar_logits is not None and student_prediction.bin_edges is not None:
        row_loss = _binned_pseudo_loss(
            student_prediction,
            consensus_prediction,
            reference,
            n_support,
            range_margin,
            gaussian_std_span,
            min_scale,
            eps,
        )
    else:
        row_loss = _general_pseudo_loss(
            student_prediction,
            consensus_prediction,
            n_support,
            range_margin,
            gaussian_std_span,
            min_scale,
            eps,
        )

    return _reduce_pseudo_loss_rows(row_loss, sample_weights, reduction, eps)


class SAGERegLoss(nn.Module):
    """Composite supervised + weighted distributional pseudo-supervision loss."""

    def __init__(
        self,
        *,
        tau: float = 0.2,
        agreement_weight: float = 1.0,
        n_support: int = 128,
        range_margin: float = 0.05,
        gaussian_std_span: float = 4.0,
        min_scale: float = 1e-4,
        eps: float = 1e-8,
        detach_weights: bool = True,
        weight_power: float = 1.0,
    ) -> None:
        super().__init__()
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if agreement_weight < 0.0:
            raise ValueError("agreement_weight must be non-negative")
        if n_support < 8:
            raise ValueError("n_support must be at least 8")
        if weight_power <= 0.0:
            raise ValueError("weight_power must be positive")
        self.tau = tau
        self.agreement_weight = agreement_weight
        self.n_support = n_support
        self.range_margin = range_margin
        self.gaussian_std_span = gaussian_std_span
        self.min_scale = min_scale
        self.eps = eps
        self.detach_weights = detach_weights
        self.weight_power = weight_power

    def agreement(self, unlabeled_views: Sequence[PredictiveBatch]) -> SAGERegAgreement:
        consensus = build_consensus_predictive_batch(
            unlabeled_views,
            n_support=self.n_support,
            range_margin=self.range_margin,
            gaussian_std_span=self.gaussian_std_span,
            min_scale=self.min_scale,
            eps=self.eps,
        )
        disagreement = predictive_agreement_score(
            unlabeled_views,
            n_support=self.n_support,
            range_margin=self.range_margin,
            gaussian_std_span=self.gaussian_std_span,
            min_scale=self.min_scale,
            eps=self.eps,
            reduction="none",
        )
        raw_weights = disagreement_to_weight(
            disagreement,
            self.tau,
            power=self.weight_power,
            eps=self.eps,
        )
        weights = raw_weights.detach() if self.detach_weights else raw_weights
        anchor_prediction = unlabeled_views[0]
        loss = distributional_pseudo_loss(
            anchor_prediction,
            consensus,
            sample_weights=weights,
            reduction="mean",
            n_support=self.n_support,
            range_margin=self.range_margin,
            gaussian_std_span=self.gaussian_std_span,
            min_scale=self.min_scale,
            eps=self.eps,
        )
        return SAGERegAgreement(
            loss=loss,
            disagreement=disagreement,
            weights=weights,
            consensus=consensus,
        )

    def forward(
        self,
        supervised_loss: Tensor,
        unlabeled_views: Sequence[PredictiveBatch],
    ) -> SAGERegOutput:
        agreement = self.agreement(unlabeled_views)
        supervised = torch.as_tensor(
            supervised_loss,
            device=agreement.loss.device,
            dtype=agreement.loss.dtype,
        )
        if supervised.ndim > 0:
            supervised = supervised.mean()
        return SAGERegOutput(
            total_loss=supervised + self.agreement_weight * agreement.loss,
            supervised_loss=supervised,
            agreement=agreement,
        )


class TeacherStudentTrainer:
    """Modular Teacher-Student consistency trainer for semi-supervised regression.

    Orchestrates labeled and unlabeled training steps. Custom sample weighting
    policies can be injected to handle continuous heteroscedastic pseudo-labeling,
    conformal width gating, and target label shift prior correction.
    """

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        supervised_loss_fn: Callable[[nn.Module, Tensor, Tensor], Tensor],
        predictive_batch_fn: Callable[[nn.Module, Tensor], PredictiveBatch],
        augment_fn: Callable[[Tensor], Tensor] | None = None,
        unsupervised_loss_fn: Callable[[PredictiveBatch, PredictiveBatch, Tensor], Tensor]
        | None = None,
        sample_weight_fn: Callable[[Sequence[PredictiveBatch], PredictiveBatch], Tensor]
        | None = None,
        n_views: int = 4,
        agreement_weight: float = 1.0,
        ema_decay: float | None = 0.99,
        n_support: int = 128,
        range_margin: float = 0.05,
        gaussian_std_span: float = 4.0,
        min_scale: float = 1e-4,
        eps: float = 1e-8,
        detach_weights: bool = True,
    ) -> None:
        if n_views < 2:
            raise ValueError("n_views must be at least 2")

        self.optimizer = optimizer
        self.supervised_loss_fn = supervised_loss_fn
        self.predictive_batch_fn = predictive_batch_fn
        self.augment_fn = augment_fn if augment_fn is not None else (lambda x: x)
        self.n_views = n_views
        self.agreement_weight = agreement_weight
        self.ema_decay = ema_decay
        self.n_support = n_support
        self.range_margin = range_margin
        self.gaussian_std_span = gaussian_std_span
        self.min_scale = min_scale
        self.eps = eps
        self.detach_weights = detach_weights

        # Set default loss function if not provided
        if unsupervised_loss_fn is not None:
            self.unsupervised_loss_fn = unsupervised_loss_fn
        else:
            self.unsupervised_loss_fn = self._default_unsupervised_loss

        # Set default weight function if not provided (uses SAGEReg KL disagreement weights)
        if sample_weight_fn is not None:
            self.sample_weight_fn = sample_weight_fn
        else:
            self.sample_weight_fn = self._default_sample_weight

    def _default_unsupervised_loss(
        self,
        student_pred: PredictiveBatch,
        consensus_pred: PredictiveBatch,
        weight: Tensor,
    ) -> Tensor:
        weights = weight.detach() if self.detach_weights else weight
        return distributional_pseudo_loss(
            student_pred,
            consensus_pred,
            sample_weights=weights,
            reduction="mean",
            n_support=self.n_support,
            range_margin=self.range_margin,
            gaussian_std_span=self.gaussian_std_span,
            min_scale=self.min_scale,
            eps=self.eps,
        )

    def _default_sample_weight(
        self,
        predictive_views: Sequence[PredictiveBatch],
        consensus_pred: PredictiveBatch,
    ) -> Tensor:
        disagreement = predictive_agreement_score(
            predictive_views,
            n_support=self.n_support,
            range_margin=self.range_margin,
            gaussian_std_span=self.gaussian_std_span,
            min_scale=self.min_scale,
            eps=self.eps,
            reduction="none",
        )
        return disagreement_to_weight(disagreement, tau=0.2, eps=self.eps)

    def compute_consensus(self, predictive_views: Sequence[PredictiveBatch]) -> PredictiveBatch:
        return build_consensus_predictive_batch(
            predictive_views,
            n_support=self.n_support,
            range_margin=self.range_margin,
            gaussian_std_span=self.gaussian_std_span,
            min_scale=self.min_scale,
            eps=self.eps,
        )

    def _teacher_views(self, teacher: nn.Module, x_unlabeled: Tensor) -> list[PredictiveBatch]:
        with torch.no_grad():
            views = [self.predictive_batch_fn(teacher, x_unlabeled)]
            for _ in range(self.n_views - 1):
                views.append(self.predictive_batch_fn(teacher, self.augment_fn(x_unlabeled)))
        return views

    def fit(
        self,
        model: nn.Module,
        labeled_loader: Sequence[Any] | torch.utils.data.DataLoader[Any],
        unlabeled_loader: Sequence[Any] | torch.utils.data.DataLoader[Any],
        val_loader: Sequence[Any] | torch.utils.data.DataLoader[Any] | None = None,
        *,
        epochs: int = 1,
        lr_schedule: str = "constant",
        lr_min: float = 0.0,
    ) -> dict[str, list[float]]:
        teacher = copy.deepcopy(model).eval() if self.ema_decay is not None else model
        history: dict[str, list[float]] = {
            "total_loss": [],
            "supervised_loss": [],
            "unsupervised_loss": [],
            "mean_weight": [],
            "mean_disagreement": [],
        }
        base_lr = float(self.optimizer.param_groups[0]["lr"])
        for epoch_idx in range(epochs):
            if lr_schedule == "cosine":
                if epochs <= 1:
                    mult = 1.0
                else:
                    mult = 0.5 * (1.0 + math.cos(math.pi * float(epoch_idx) / float(epochs - 1)))
                lr = max(base_lr * mult, float(lr_min))
                for g in self.optimizer.param_groups:
                    g["lr"] = lr
            elif lr_schedule == "constant":
                for g in self.optimizer.param_groups:
                    g["lr"] = base_lr
            else:
                raise ValueError(
                    f"unknown lr_schedule: {lr_schedule!r} (expected 'constant' or 'cosine')"
                )
            unlabeled_iter = cycle(unlabeled_loader)
            for labeled_batch, unlabeled_batch in zip(
                labeled_loader,
                unlabeled_iter,
                strict=False,
            ):
                model.train()
                x_labeled, y_labeled = _split_labeled_batch(labeled_batch)
                x_unlabeled = _split_unlabeled_batch(unlabeled_batch)

                self.optimizer.zero_grad()
                supervised_loss = self.supervised_loss_fn(model, x_labeled, y_labeled)
                student_pred = self.predictive_batch_fn(model, x_unlabeled)
                predictive_views = self._teacher_views(teacher, x_unlabeled)
                consensus = self.compute_consensus(predictive_views)

                # Compute pluggable sample weights
                weight = self.sample_weight_fn(predictive_views, consensus)
                if self.detach_weights:
                    weight = weight.detach()

                unsupervised_loss = self.unsupervised_loss_fn(student_pred, consensus, weight)
                total_loss = supervised_loss + self.agreement_weight * unsupervised_loss
                total_loss.backward()
                self.optimizer.step()
                if teacher is not model and self.ema_decay is not None:
                    update_ema_teacher_(teacher, model, momentum=self.ema_decay)

                disagreement = predictive_agreement_score(
                    predictive_views,
                    n_support=self.n_support,
                    range_margin=self.range_margin,
                    gaussian_std_span=self.gaussian_std_span,
                    min_scale=self.min_scale,
                    eps=self.eps,
                    reduction="none",
                )
                history["total_loss"].append(float(total_loss.detach().item()))
                history["supervised_loss"].append(float(supervised_loss.detach().item()))
                history["unsupervised_loss"].append(float(unsupervised_loss.detach().item()))
                history["mean_weight"].append(float(weight.detach().mean().item()))
                history["mean_disagreement"].append(float(disagreement.detach().mean().item()))
        return history


def _split_labeled_batch(batch: Any) -> tuple[Tensor, Tensor]:
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
    else:
        raise ValueError("labeled batches must yield (x, y, ...) tuples")
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        raise ValueError("labeled batch entries must be tensors")
    return x, y


def _split_unlabeled_batch(batch: Any) -> Tensor:
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    if not torch.is_tensor(x):
        raise ValueError(
            "unlabeled batches must yield tensors or tuples whose first entry is a tensor"
        )
    return x


__all__ = [
    "SAGERegAgreement",
    "SAGERegLoss",
    "SAGERegOutput",
    "TeacherStudentTrainer",
    "build_consensus_predictive_batch",
    "disagreement_to_weight",
    "uncertainty_to_weight",
    "conformal_width_to_weight",
    "distributional_pseudo_loss",
    "predictive_agreement_score",
]
