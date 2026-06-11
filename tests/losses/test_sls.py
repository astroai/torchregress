from __future__ import annotations

import torch

from torchregress.losses.sls import (
    MahalanobisFrontier,
    SLSLoss,
    UnionFrontier,
    VolumePreservingCouplingLayer,
    VolumePreservingFlow,
    sigmoidal_schedule,
)


def test_volume_preserving_coupling_layer_invertibility() -> None:
    d = 4
    mask = torch.tensor([True, True, False, False])
    layer = VolumePreservingCouplingLayer(d, mask, context_dim=2, hidden_dim=16)

    y = torch.randn(5, d)
    context = torch.randn(5, 2)

    # Forward
    z = layer(y, context=context)
    assert z.shape == y.shape

    # Inverse
    y_rec = layer.inverse(z, context=context)
    assert y_rec.shape == y.shape
    assert torch.allclose(y, y_rec, atol=1e-5)


def test_volume_preserving_flow_invertibility() -> None:
    d = 3
    flow = VolumePreservingFlow(d, context_dim=2, n_transforms=4, hidden_dim=16)

    y = torch.randn(10, d)
    context = torch.randn(10, 2)

    # Forward
    z = flow(y, context=context)
    assert z.shape == y.shape

    # Inverse
    y_rec = flow.inverse(z, context=context)
    assert torch.allclose(y, y_rec, atol=1e-5)


def test_mahalanobis_frontier_full_and_low_rank() -> None:
    # Full mode
    frontier_full = MahalanobisFrontier(d=2, context_dim=3, mode="full", hidden_dim=16)
    y = torch.randn(5, 2)
    context = torch.randn(5, 3)

    G, logdet = frontier_full(y, context)
    assert G.shape == (5,)
    assert logdet.shape == (5,)
    assert torch.all(G >= 0.0)

    # Low rank mode
    frontier_lr = MahalanobisFrontier(d=4, context_dim=3, mode="low_rank", rank=2, hidden_dim=16)
    y_4d = torch.randn(5, 4)
    G_lr, logdet_lr = frontier_lr(y_4d, context)
    assert G_lr.shape == (5,)
    assert logdet_lr.shape == (5,)
    assert torch.all(G_lr >= 0.0)


def test_union_frontier() -> None:
    frontier = UnionFrontier(d=2, K=3, context_dim=4, mode="full", hidden_dim=16)
    y = torch.randn(5, 2)
    context = torch.randn(5, 4)

    G, log_vol_term = frontier(y, context)
    assert G.shape == (5,)
    assert log_vol_term.shape == (5,)
    assert torch.all(G >= 0.0)


def test_sigmoidal_schedule() -> None:
    # Warmup phase
    val = sigmoidal_schedule(step=10, warmup_steps=50, init_val=0.5, min_val=0.05)
    assert val == 0.5

    # Post warmup decay
    val_post = sigmoidal_schedule(step=100, warmup_steps=50, init_val=0.5, min_val=0.05, t0=100.0)
    assert val_post < 0.5
    assert val_post >= 0.05


def test_sls_loss_warmup_and_forward() -> None:
    # 2D target, 3D context
    loss_fn = SLSLoss(d=2, context_dim=3, K=1, warmup_steps=5, reduction="mean")

    # Check that parameters are initialized
    assert len(list(loss_fn.parameters())) > 0

    x = torch.randn(10, 3, requires_grad=True)
    y = torch.randn(10, 2)

    # Step 1: Warmup phase forward and backward
    # Counter starts at 0. Calling forward increments to 1.
    loss_warmup = loss_fn(x, y)
    assert torch.isfinite(loss_warmup)
    loss_warmup.backward()
    assert x.grad is not None

    # Step 2: Push beyond warmup to trigger shrinking window
    loss_fn.step_counter = 10  # Manually advance counter
    x.grad = None
    loss_window = loss_fn(x, y)
    assert torch.isfinite(loss_window)
    loss_window.backward()
    assert x.grad is not None


def test_sls_loss_union_warmup_and_forward() -> None:
    # 2D target, 3D context, Union of K=2 flows
    loss_fn = SLSLoss(d=2, context_dim=3, K=2, warmup_steps=5, reduction="sum")

    x = torch.randn(10, 3, requires_grad=True)
    y = torch.randn(10, 2)

    loss_warmup = loss_fn(x, y)
    assert torch.isfinite(loss_warmup)
    loss_warmup.backward()
    assert x.grad is not None
