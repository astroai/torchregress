from __future__ import annotations

import torch

from torchregress.losses.uncertain_gt import (
    ConsistencyRegLoss,
    NoisyTargetGaussianNLL,
    PseudoLabelConsistencyLoss,
    PseudoLabelNLL,
)
from torchregress.utils.propensity import ipw_weights


def _gaussian_head_inputs(n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.randn(n, 1)
    log_var = torch.zeros_like(mean) - 1.5
    y_pred = torch.cat([mean, log_var], dim=-1)
    target = mean + 0.3 * torch.randn_like(mean)
    return y_pred, target


def test_noisy_target_gaussian_nll_forward() -> None:
    torch.manual_seed(0)
    y_pred, target = _gaussian_head_inputs()
    target_var = torch.full_like(target, 0.05)
    loss_fn = NoisyTargetGaussianNLL()
    loss = loss_fn(y_pred, target, target_variance=target_var)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_noisy_target_gaussian_nll_propensity_scores_match_weights() -> None:
    torch.manual_seed(0)
    y_pred, target = _gaussian_head_inputs()
    score = torch.rand(target.shape[0], 1).clamp(0.1, 0.9)
    target_var = torch.full_like(target, 0.02)
    loss_fn = NoisyTargetGaussianNLL(reduction="mean", propensity_clip=10.0)
    by_scores = loss_fn(y_pred, target, target_variance=target_var, propensity_scores=score)
    by_weights = loss_fn(
        y_pred,
        target,
        target_variance=target_var,
        propensity_weights=ipw_weights(score, clip_min=0.1, clip_max=0.9),
    )
    assert torch.allclose(by_scores, by_weights, atol=1e-6)


def test_consistency_reg_loss_prefers_closer_teacher() -> None:
    torch.manual_seed(0)
    y_pred = torch.randn(32, 1)
    target = y_pred + 0.2 * torch.randn(32, 1)
    teacher_close = y_pred + 0.05 * torch.randn(32, 1)
    teacher_far = y_pred + 2.0 * torch.randn(32, 1)
    loss_fn = ConsistencyRegLoss(consistency_weight=1.0)
    loss_close = loss_fn(y_pred, target, teacher_close)
    loss_far = loss_fn(y_pred, target, teacher_far)
    assert float(loss_close.item()) < float(loss_far.item())


def test_pseudo_label_nll_supports_mixed_observed_and_pseudo() -> None:
    torch.manual_seed(0)
    y_pred, target = _gaussian_head_inputs(n=48)
    pseudo = target + 0.1 * torch.randn_like(target)
    label_mask = torch.zeros_like(target, dtype=torch.bool)
    label_mask[:24] = True
    pseudo_confidence = torch.full_like(target, 0.8)

    loss_fn = PseudoLabelNLL(pseudo_weight=0.7)
    loss = loss_fn(
        y_pred,
        target,
        pseudo_target=pseudo,
        pseudo_confidence=pseudo_confidence,
        label_mask=label_mask,
    )
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_pseudo_label_nll_requires_some_supervision_source() -> None:
    y_pred, _ = _gaussian_head_inputs(n=16)
    loss_fn = PseudoLabelNLL()
    try:
        loss_fn(y_pred, target=None, pseudo_target=None)
    except ValueError as exc:
        assert "At least one of target or pseudo_target must be provided" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing target and pseudo_target")


def test_pseudo_label_nll_backward_stays_finite_when_some_blend_weights_are_zero() -> None:
    y_pred, target = _gaussian_head_inputs(n=16)
    y_pred = y_pred.clone().requires_grad_(True)
    label_mask = torch.zeros_like(target, dtype=torch.bool)
    label_mask[:8] = True
    pseudo_confidence = torch.zeros_like(target)
    pseudo_confidence[8:10] = 0.8
    loss = PseudoLabelNLL()(
        y_pred,
        target,
        pseudo_target=target,
        pseudo_confidence=pseudo_confidence,
        label_mask=label_mask,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()


def test_semi_supervised_regression_loss_uses_masks_and_pseudo_targets() -> None:
    torch.manual_seed(0)
    y_pred = torch.randn(32, 1)
    target = y_pred + 0.2 * torch.randn(32, 1)
    pseudo = y_pred + 0.1 * torch.randn(32, 1)
    teacher = y_pred + 0.05 * torch.randn(32, 1)
    label_mask = torch.zeros_like(target, dtype=torch.bool)
    label_mask[:12] = True
    pseudo_confidence = torch.ones_like(target)
    pseudo_confidence[20:] = 0.0

    loss_fn = PseudoLabelConsistencyLoss(
        pseudo_weight=0.8,
        consistency_weight=0.4,
        confidence_threshold=0.1,
    )
    loss = loss_fn(
        y_pred,
        target,
        pseudo_target=pseudo,
        pseudo_confidence=pseudo_confidence,
        teacher_pred=teacher,
        label_mask=label_mask,
    )
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0
