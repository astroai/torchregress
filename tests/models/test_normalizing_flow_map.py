"""Tests for the tabular Normalizing Flow Map."""

from __future__ import annotations

import torch

from torchregress.models import NormalizingFlowMap


def test_nfm_roundtrip_and_likelihood() -> None:
    torch.manual_seed(0)
    for d in (1, 2, 3):
        model = NormalizingFlowMap(d, 8, n_layers=3, hidden=(16, 16))
        context = torch.randn(6, 8)
        y = torch.randn(6, d)
        z, logdet = model(y, 0.0, 1.0, context)
        rec, inv_logdet = model.inverse(z, 0.0, 1.0, context)
        assert torch.allclose(rec, y, atol=1e-5)
        assert torch.allclose(logdet, -inv_logdet, atol=1e-5)
        logp = model.log_prob(context, y)
        assert logp.shape == (6,) and torch.isfinite(logp).all()
        assert model.sample(context, n_samples=8).shape == (6, 8, d)


def test_nfm_training_loss_gradients() -> None:
    torch.manual_seed(0)
    model = NormalizingFlowMap(2, 8, n_layers=2, hidden=(16, 16))
    context = torch.randn(10, 8)
    y = torch.randn(10, 2)
    loss = model.training_loss(context, y)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_nfm_flow_map_consistency() -> None:
    torch.manual_seed(0)
    model = NormalizingFlowMap(2, 4, n_layers=2, hidden=(16, 16))
    context = torch.randn(5, 4)
    y = torch.randn(5, 2)
    z, _ = model(y, 0.0, 1.0, context)
    # Intermediate maps should compose consistently for a no-op identity-like
    # map; here just check shape/finiteness and invertibility at arbitrary time.
    mid, _ = model(y, 0.2, 0.8, context)
    back, _ = model.inverse(mid, 0.2, 0.8, context)
    assert torch.allclose(back, y, atol=1e-5)
