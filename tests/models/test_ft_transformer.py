"""Tests for the lean FT-Transformer."""

from __future__ import annotations

import torch

from torchregress.models import FTTransformer


def test_ft_transformer_shapes() -> None:
    model = FTTransformer(8, d_token=16, n_layers=1, n_heads=4, out_dim=3)
    out = model(torch.randn(5, 8))
    assert out.shape == (5, 3)


def test_construction_emits_no_nested_tensor_warning() -> None:
    import warnings

    from torchregress.models.normalizing_flow_map import AutoregressiveTimeLayer

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FTTransformer(8, d_token=16, n_layers=2, n_heads=4, out_dim=1)
        AutoregressiveTimeLayer(dim=2, context_dim=3, hidden=16, n_heads=4)
