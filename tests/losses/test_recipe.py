"""Tests for the CE + CDF + RnC-N photo-z recipe objective."""

from __future__ import annotations

import torch

from torchregress.losses import CumulativeCDFLoss, PZRecipeLoss, cumulative_cdf_loss


def test_cumulative_cdf_loss_zero_for_perfect_step() -> None:
    centers = torch.tensor([0.25, 0.75])
    # A target at 0.5 needs CDF [1, 1] at the two centers (perfect step).
    logits = torch.tensor([[0.0, 100.0]])
    loss = cumulative_cdf_loss(logits, torch.tensor([0.5]), centers)
    assert loss.shape == (1,)
    assert float(loss) < 0.1


def test_recipe_loss_components() -> None:
    torch.manual_seed(0)
    centers = torch.linspace(0.0, 2.0, 16)
    loss_fn = PZRecipeLoss(centers, ce_weight=0.5, cdf_weight=0.5, rnc_weight=0.1)
    logits = torch.randn(12, 16, requires_grad=True)
    features = torch.randn(12, 8, requires_grad=True)
    target = torch.rand(12) * 2.0
    loss = loss_fn(logits, target, features=features)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_recipe_loss_without_rnc() -> None:
    centers = torch.linspace(0.0, 2.0, 8)
    loss_fn = PZRecipeLoss(centers, rnc_weight=0.0)
    logits = torch.randn(5, 8)
    target = torch.rand(5) * 2.0
    assert torch.isfinite(loss_fn(logits, target))


def test_cumulative_cdf_class() -> None:
    loss_fn = CumulativeCDFLoss(torch.linspace(0.0, 2.0, 10))
    loss = loss_fn(torch.randn(6, 10), torch.rand(6) * 2.0)
    assert torch.isfinite(loss)


def test_tri_loss_crps_matches_metric_and_gradients() -> None:

    from torchregress.losses import TriLoss
    from torchregress.metrics import binned_cdf, crps_from_binned_cdf

    centers = torch.linspace(0.0, 2.0, 20)
    loss_fn = TriLoss(centers, ce_weight=0.0, crps_weight=1.0, rnc_weight=0.0)
    logits = torch.randn(7, 20, requires_grad=True)
    target = torch.rand(7) * 2.0
    loss = loss_fn(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    reference = crps_from_binned_cdf(
        loss_fn._edges().detach().numpy(),
        binned_cdf(torch.softmax(logits.detach(), -1).numpy()),
        target.numpy(),
    )
    assert abs(float(loss) - reference) < 1e-4
