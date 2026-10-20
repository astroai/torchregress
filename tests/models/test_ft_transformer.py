"""Tests for the lean FT-Transformer and PZRecipe model."""

from __future__ import annotations

import torch

from torchregress.models import FTTransformer, PZRecipe


def test_ft_transformer_shapes() -> None:
    model = FTTransformer(8, d_token=16, n_layers=1, n_heads=4, out_dim=3)
    x = torch.randn(5, 8)
    out = model(x)
    assert out.shape == (5, 3)


def test_pz_recipe_two_stages() -> None:
    model = PZRecipe(
        8,
        n_bins=20,
        d_token=16,
        n_layers=1,
        n_heads=4,
        stage2_bins=30,
        stage2_hidden=32,
    )
    x = torch.randn(7, 8)
    logits1 = model(x, stage=1)
    logits2 = model(x, stage=2)
    features = model.encode(x)
    assert logits1.shape == (7, 20)
    assert logits2.shape == (7, 30)
    assert features.shape == (7, 16)
    proba = model.predict_proba(x, stage=1)
    assert torch.allclose(proba.sum(dim=-1), torch.ones(7), atol=1e-5)


def test_fit_pz_recipe_smoke() -> None:
    from torchregress.models import fit_pz_recipe

    model = PZRecipe(
        6,
        n_bins=10,
        d_token=16,
        n_layers=1,
        n_heads=4,
        stage2_bins=12,
        stage2_hidden=16,
    )
    torch.manual_seed(0)
    x = torch.randn(64, 6)
    y = 0.5 * x[:, 0] + 0.2 * torch.randn(64)
    fit = fit_pz_recipe(model, x, y, stage1_epochs=3, stage2_epochs=2, batch_size=32)
    assert len(fit.history["stage1"]) == 3
    assert len(fit.history["stage2"]) == 2
    proba = fit.predict_proba(x)
    assert proba.shape == (64, 10)
    assert torch.allclose(proba.sum(-1), torch.ones(64), atol=1e-5)
    pred = fit.predict(x)
    assert pred.shape == (64,)
