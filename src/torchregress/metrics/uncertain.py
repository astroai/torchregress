"""Metrics for uncertain/weak ground-truth workflows."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor


def noisy_target_gaussian_nll(
    pred_mean: Tensor,
    pred_variance: Tensor,
    target: Tensor,
    target_variance: Tensor,
    *,
    min_variance: float = 1e-8,
) -> Tensor:
    """Mean Gaussian NLL when both prediction and target are uncertain."""
    if pred_mean.shape != pred_variance.shape or pred_mean.shape != target.shape:
        raise ValueError("pred_mean, pred_variance, and target must share shape")
    if target_variance.shape != pred_variance.shape:
        raise ValueError("target_variance must match pred_variance shape")
    total_var = (pred_variance + target_variance).clamp_min(min_variance)
    nll = 0.5 * (
        torch.log(total_var) + (target - pred_mean) ** 2 / total_var + math.log(2.0 * math.pi)
    )
    return nll.mean()


def consistency_error(
    student_pred: Tensor,
    teacher_pred: Tensor,
    *,
    p: int = 2,
) -> Tensor:
    """Mean student/teacher consistency error."""
    if student_pred.shape != teacher_pred.shape:
        raise ValueError("student_pred and teacher_pred must share shape")
    if p == 1:
        return torch.abs(student_pred - teacher_pred).mean()
    if p == 2:
        return ((student_pred - teacher_pred) ** 2).mean()
    raise ValueError("p must be 1 or 2")


def pseudo_label_acceptance_rate(
    pseudo_confidence: Tensor,
    *,
    threshold: float = 0.5,
) -> Tensor:
    """Fraction of pseudo-labels above confidence threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return (pseudo_confidence >= threshold).float().mean()


def uncertain_gt_metrics_report(
    *,
    pred_mean: Tensor,
    pred_variance: Tensor,
    target: Tensor,
    target_variance: Tensor,
    teacher_pred: Tensor | None = None,
    pseudo_confidence: Tensor | None = None,
) -> Dict[str, Tensor]:
    """Convenience report for uncertain-ground-truth evaluation."""
    report: Dict[str, Tensor] = {
        "NoisyTargetNLL": noisy_target_gaussian_nll(
            pred_mean,
            pred_variance,
            target,
            target_variance,
        )
    }
    if teacher_pred is not None:
        report["ConsistencyMSE"] = consistency_error(pred_mean, teacher_pred, p=2)
    if pseudo_confidence is not None:
        report["PseudoAcceptanceRate"] = pseudo_label_acceptance_rate(pseudo_confidence)
    return report


__all__ = [
    "noisy_target_gaussian_nll",
    "consistency_error",
    "pseudo_label_acceptance_rate",
    "uncertain_gt_metrics_report",
]
