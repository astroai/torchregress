from __future__ import annotations

import torch

from torchregress.metrics.uncertain import (
    consistency_error,
    noisy_target_gaussian_nll,
    pseudo_label_acceptance_rate,
    uncertain_gt_metrics_report,
)


def test_noisy_target_gaussian_nll_positive() -> None:
    torch.manual_seed(0)
    mean = torch.randn(128, 1)
    pred_var = torch.full_like(mean, 0.04)
    target = mean + 0.2 * torch.randn_like(mean)
    target_var = torch.full_like(mean, 0.01)
    value = noisy_target_gaussian_nll(mean, pred_var, target, target_var)
    assert torch.isfinite(value)
    assert float(value.item()) > -10.0


def test_consistency_error_l1_l2() -> None:
    a = torch.tensor([[0.0], [1.0], [2.0]])
    b = torch.tensor([[0.0], [2.0], [4.0]])
    e1 = consistency_error(a, b, p=1)
    e2 = consistency_error(a, b, p=2)
    assert float(e1.item()) > 0.0
    assert float(e2.item()) > float(e1.item())


def test_pseudo_label_acceptance_rate() -> None:
    confidence = torch.tensor([0.2, 0.6, 0.9, 0.4])
    rate = pseudo_label_acceptance_rate(confidence, threshold=0.5)
    assert torch.allclose(rate, torch.tensor(0.5))


def test_uncertain_gt_metrics_report_keys() -> None:
    torch.manual_seed(0)
    mean = torch.randn(32, 1)
    pred_var = torch.full_like(mean, 0.1)
    target = mean + 0.3 * torch.randn_like(mean)
    target_var = torch.full_like(mean, 0.02)
    teacher = mean + 0.1 * torch.randn_like(mean)
    confidence = torch.rand_like(mean)
    report = uncertain_gt_metrics_report(
        pred_mean=mean,
        pred_variance=pred_var,
        target=target,
        target_variance=target_var,
        teacher_pred=teacher,
        pseudo_confidence=confidence,
    )
    assert "NoisyTargetNLL" in report
    assert "ConsistencyMSE" in report
    assert "PseudoAcceptanceRate" in report
