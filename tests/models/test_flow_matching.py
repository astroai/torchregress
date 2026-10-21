"""Tests for the lean conditional flow-matching model."""

from __future__ import annotations

import torch

from torchregress.models import FlowMatchingMLP, cfm_loss, sample_cfm


def test_cfm_loss_and_sampling_shapes() -> None:
    torch.manual_seed(0)
    net = FlowMatchingMLP(n_features=2, context_dim=8, hidden=(32, 32))
    ctx = torch.randn(5, 8)
    y = torch.randn(5, 2)
    loss = cfm_loss(net, ctx, y)
    assert torch.isfinite(loss) and loss.ndim == 0
    loss.backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)
    samples = sample_cfm(net, ctx, n_samples=16, steps=8)
    assert samples.shape == (5, 16, 2)
    assert torch.isfinite(samples).all()


def test_cfm_scalar_target() -> None:
    net = FlowMatchingMLP(n_features=1, context_dim=4, hidden=(16,))
    ctx = torch.randn(3, 4)
    assert torch.isfinite(cfm_loss(net, ctx, torch.randn(3))).item()


def test_cfm_odd_time_embedding_width() -> None:
    net = FlowMatchingMLP(1, 2, hidden=(8,), time_embedding=3)
    x = torch.randn(4, 1)
    t = torch.rand(4)
    context = torch.randn(4, 2)
    assert net(x, t, context).shape == x.shape
