"""Tests for the generic binned cumulative-distribution loss."""

from __future__ import annotations

import torch

from torchregress.losses import CumulativeCDFLoss, cumulative_cdf_loss


def test_cumulative_cdf_loss_zero_for_perfect_step() -> None:
    centers = torch.tensor([0.25, 0.75])
    logits = torch.tensor([[0.0, 100.0]])
    loss = cumulative_cdf_loss(logits, torch.tensor([0.5]), centers)
    assert loss.shape == (1,)
    assert float(loss) < 0.1


def test_cumulative_cdf_class() -> None:
    loss_fn = CumulativeCDFLoss(torch.linspace(0.0, 2.0, 10))
    loss = loss_fn(torch.randn(6, 10), torch.rand(6) * 2.0)
    assert torch.isfinite(loss)


def test_cdf_rejects_target_broadcast_and_nonfinite_grid():
    import pytest

    with pytest.raises(ValueError):
        cumulative_cdf_loss(torch.zeros((3, 2)), torch.zeros(1), torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError):
        CumulativeCDFLoss([0.0, float("inf")])
