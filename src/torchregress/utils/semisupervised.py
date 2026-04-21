"""Utilities for semi-supervised regression workflows."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


def generate_pseudo_labels(
    prediction: Tensor,
    *,
    log_variance: Optional[Tensor] = None,
    confidence: Optional[Tensor] = None,
    confidence_threshold: float = 0.0,
    max_std: Optional[float] = None,
    min_confidence: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build pseudo-labels, confidence weights, and accept-mask tensors.

    The returned confidence is in ``[0, 1]`` and can be passed directly to
    pseudo-label losses or weighting logic.
    """

    pseudo_target = prediction.detach()
    if confidence is not None:
        pseudo_conf = confidence.detach().to(prediction.device, prediction.dtype)
    elif log_variance is not None:
        std = torch.exp(0.5 * log_variance.detach()).to(prediction.device, prediction.dtype)
        if max_std is None:
            max_std = float(std.quantile(0.75).item()) if std.numel() > 0 else 1.0
        denom = max(float(max_std), 1e-8)
        pseudo_conf = (1.0 - std / denom).clamp(min=0.0, max=1.0)
    else:
        pseudo_conf = torch.ones_like(prediction)

    pseudo_conf = pseudo_conf.clamp(min=min_confidence, max=1.0)
    accepted = pseudo_conf >= confidence_threshold
    return pseudo_target, pseudo_conf, accepted


def update_ema_teacher_(
    teacher: nn.Module,
    student: nn.Module,
    *,
    momentum: float = 0.99,
) -> None:
    """Update teacher parameters in-place using student parameters."""

    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must lie in [0, 1)")

    teacher_params = dict(teacher.named_parameters())
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher_params.items():
        if name not in student_params:
            continue
        student_param = student_params[name]
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher_buffers.items():
        if name not in student_buffers:
            continue
        teacher_buffer.copy_(student_buffers[name])


__all__ = ["generate_pseudo_labels", "update_ema_teacher_"]
