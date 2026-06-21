"""
Unit tests for torchregress.ensemble.combiners — ensemble combination methods.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchregress.ensemble.combiners import (
    BayesianModelAveraging,
    DynamicEnsembleWeighting,
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
# BayesianModelAveraging
# ═══════════════════════════════════════════════════════════════════════════════


class TestBayesianModelAveraging:
    def test_init(self) -> None:
        """BMA initializes with uniform model weights."""
        models = _make_models(3)
        bma = BayesianModelAveraging(models)
        assert bma.n_models == 3
        assert bma.model_weights.shape == (3,)

    def test_forward_shape(self) -> None:
        """Forward returns weighted average of model predictions."""
        models = _make_models(3)
        bma = BayesianModelAveraging(models)
        x = torch.randn(8, 4)
        out = bma(x)
        assert out.shape == (8, 1)

    def test_get_model_weights_sums_to_one(self) -> None:
        """Model weights are softmax probabilities summing to 1."""
        models = _make_models(3)
        bma = BayesianModelAveraging(models)
        weights = bma.get_model_weights()
        assert weights.shape == (3,)
        assert float(weights.sum().item()) == pytest.approx(1.0)

    def test_predict_with_uncertainty_shapes(self) -> None:
        """predict_with_uncertainty returns mean and variance."""
        models = _make_models(3)
        bma = BayesianModelAveraging(models)
        x = torch.randn(8, 4)
        mean, var = bma.predict_with_uncertainty(x)
        assert mean.shape == (8, 1)
        assert var.shape == (8, 1)
        assert torch.all(var >= 0)

    def test_forward_matches_weighted_average(self) -> None:
        """BMA forward should equal manual weighted average of model outputs."""
        models = _make_models(3)
        bma = BayesianModelAveraging(models)
        x = torch.randn(4, 4)
        out = bma(x)

        # Manual weighted average
        weights = bma.get_model_weights()
        individual = torch.stack([m(x) for m in models], dim=1)  # [4, 3, 1]
        expected = torch.sum(individual * weights.view(1, -1, 1), dim=1)
        assert torch.allclose(out, expected, atol=1e-6)


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
# DynamicEnsembleWeighting
# ═══════════════════════════════════════════════════════════════════════════════


class TestDynamicEnsembleWeighting:
    def test_init(self) -> None:
        """DEW initializes with uniform weights."""
        models = _make_models(3)
        dew = DynamicEnsembleWeighting(models, window_size=50)
        assert dew.n_models == 3
        assert dew.window_size == 50
        assert len(dew.prediction_history) == 0
        assert len(dew.target_history) == 0

    def test_forward_shape(self) -> None:
        """Forward returns weighted predictions."""
        models = _make_models(3)
        dew = DynamicEnsembleWeighting(models)
        x = torch.randn(8, 4)
        out = dew(x)
        assert out.shape == (8, 1)

    def test_get_model_weights(self) -> None:
        """Initial weights are uniform probabilities."""
        models = _make_models(3)
        dew = DynamicEnsembleWeighting(models)
        weights = dew.get_model_weights()
        assert weights.shape == (3,)
        assert float(weights.sum().item()) == pytest.approx(1.0)

    def test_update_weights_with_single_batch(self) -> None:
        """update_weights stores history but needs 2+ batches to update."""
        models = _make_models(3)
        dew = DynamicEnsembleWeighting(models, window_size=100)
        initial_weights = dew.get_model_weights().clone()

        x = torch.randn(8, 4)
        preds = _batched_ensemble_forward(dew.models, x, method="stack")
        dew.update_weights(preds, torch.randn(8, 1))

        # With only 1 batch in history, weights should not change
        assert torch.equal(dew.get_model_weights(), initial_weights)

    def test_update_weights_multiple_batches(self) -> None:
        """After 2+ batches, weights update based on performance."""
        models = _make_models(3)
        dew = DynamicEnsembleWeighting(models, window_size=100, learning_rate=0.5)
        initial_weights = dew.get_model_weights().clone()

        x1 = torch.randn(8, 4)
        target1 = torch.randn(8, 1)
        # update_weights expects per-model predictions [batch, n_models, out]
        preds1 = _batched_ensemble_forward(dew.models, x1, method="stack")
        dew.update_weights(preds1, target1)

        x2 = torch.randn(8, 4)
        target2 = torch.randn(8, 1)
        preds2 = _batched_ensemble_forward(dew.models, x2, method="stack")
        dew.update_weights(preds2, target2)

        # Weights should have changed
        assert not torch.equal(dew.get_model_weights(), initial_weights)

    def test_history_bounded_by_window(self) -> None:
        """Prediction history is bounded by window_size."""
        models = _make_models(3)
        dew = DynamicEnsembleWeighting(models, window_size=5)
        x = torch.randn(2, 4)

        for _ in range(10):
            preds = _batched_ensemble_forward(dew.models, x, method="stack")
            dew.update_weights(preds, torch.randn(2, 1))

        assert len(dew.prediction_history) == 5
        assert len(dew.target_history) == 5

    def test_weights_remain_probabilities(self) -> None:
        """After many updates, weights still sum to 1."""
        models = _make_models(3)
        dew = DynamicEnsembleWeighting(models, window_size=20, learning_rate=0.1)
        x = torch.randn(4, 4)

        for _ in range(5):
            preds = _batched_ensemble_forward(dew.models, x, method="stack")
            dew.update_weights(preds, torch.randn(4, 1))

        weights = dew.get_model_weights()
        assert float(weights.sum().item()) == pytest.approx(1.0)
        assert torch.all(weights >= 0)

    def test_default_parameters(self) -> None:
        """Defaults: window_size=100, learning_rate=0.1."""
        models = _make_models(3)
        dew = DynamicEnsembleWeighting(models)
        assert dew.window_size == 100
        assert dew.learning_rate == 0.1
