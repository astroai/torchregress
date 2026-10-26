"""Tests for the lean FT-Transformer."""

from __future__ import annotations

import torch

from torchregress.models import FTTransformer


def test_ft_transformer_shapes() -> None:
    model = FTTransformer(8, d_token=16, n_layers=1, n_heads=4, out_dim=3)
    out = model(torch.randn(5, 8))
    assert out.shape == (5, 3)
