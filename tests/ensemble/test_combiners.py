"""
Unit tests for torchregress.ensemble.combiners — ensemble combination methods.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchregress.ensemble.combiners import (
    StackingEnsemble,
    _batched_ensemble_forward,
)


def _make_linear(in_dim: int = 4, out_dim: int = 1) -> nn.Module:
    """Create a simple Linear model for ensemble testing."""
    return nn.Linear(in_dim, out_dim)


def _make_models(n: int = 3, in_dim: int = 4, out_dim: int = 1) -> list[nn.Module]:
    """Create n small linear models with different random weights."""
    models = [_make_linear(in_dim, out_dim) for _ in range(n)]
    # Give each model slightly different weights
    for i, m in enumerate(models):
        with torch.no_grad():
            m.weight.add_(i * 0.1)
            m.bias.add_(i * 0.1)
    return models


# ═══════════════════════════════════════════════════════════════════════════════
# _batched_ensemble_forward
# ═══════════════════════════════════════════════════════════════════════════════


class TestBatchedEnsembleForward:
    def test_stack_method(self) -> None:
        """Stack method returns [batch, n_models, out_dim]."""
        models = nn.ModuleList(_make_models(3))
        x = torch.randn(8, 4)
        out = _batched_ensemble_forward(models, x, method="stack")
        assert out.shape == (8, 3, 1)

    def test_cat_method(self) -> None:
        """Cat method returns [batch, n_models * out_dim]."""
        models = nn.ModuleList(_make_models(3))
        x = torch.randn(8, 4)
        out = _batched_ensemble_forward(models, x, method="cat")
        assert out.shape == (8, 3)

    def test_empty_models(self) -> None:
        """Empty model list returns empty tensor."""
        models = nn.ModuleList([])
        x = torch.randn(4, 4)
        out = _batched_ensemble_forward(models, x, method="stack")
        assert out.numel() == 0

    def test_single_model(self) -> None:
        """Single model works fine."""
        models = nn.ModuleList(_make_models(1))
        x = torch.randn(8, 4)
        out = _batched_ensemble_forward(models, x, method="stack")
        assert out.shape == (8, 1, 1)

    def test_fallback_when_gradients_enabled(self) -> None:
        """When gradients are enabled, falls back to sequential loop."""
        models = nn.ModuleList(_make_models(3))
        x = torch.randn(8, 4)
        with torch.enable_grad():
            out = _batched_ensemble_forward(models, x, method="stack")
        assert out.shape == (8, 3, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# StackingEnsemble
# ═══════════════════════════════════════════════════════════════════════════════


class TestStackingEnsemble:
    def test_init(self) -> None:
        """StackingEnsemble stores base models and meta-learner."""
        models = _make_models(3)
        meta = nn.Linear(3, 1)  # meta-learner takes concatenated outputs
        ensemble = StackingEnsemble(models, meta)
        assert len(ensemble.models) == 3

    def test_forward_shape(self) -> None:
        """Forward passes concatenated outputs through meta-learner."""
        models = _make_models(3)
        meta = nn.Linear(3, 1)  # 3 models * 1 output = 3
        ensemble = StackingEnsemble(models, meta)
        x = torch.randn(8, 4)
        out = ensemble(x)
        assert out.shape == (8, 1)

    def test_custom_meta_learner(self) -> None:
        """Meta-learner with hidden layers works."""
        models = _make_models(3)
        meta = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
        ensemble = StackingEnsemble(models, meta)
        x = torch.randn(8, 4)
        out = ensemble(x)
        assert out.shape == (8, 1)
        assert torch.all(torch.isfinite(out))


# ═══════════════════════════════════════════════════════════════════════════════
