"""
Unit tests for torchregress.test_time.dynamic — ParameterEMA.

Existing smoke tests in test_prediction_and_test_time.py cover basic happy paths.
This file fills gaps: edge cases, error handling, and per-method behavior.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchregress.test_time.dynamic import ParameterEMA


def _make_linear() -> nn.Module:
    return nn.Linear(3, 2)


def _make_model_with_frozen() -> nn.Module:
    """Model where one parameter has requires_grad=False."""
    model = nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# ParameterEMA — constructor
# ═══════════════════════════════════════════════════════════════════════════════


class TestParameterEMAInit:
    def test_default_decay(self) -> None:
        """Default decay."""
        ema = ParameterEMA()
        assert ema.decay == 0.99
        assert ema.shadow == {}

    def test_custom_decay(self) -> None:
        """Custom decay."""
        ema = ParameterEMA(decay=0.5)
        assert ema.decay == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# ParameterEMA — initialize
# ═══════════════════════════════════════════════════════════════════════════════


class TestParameterEMAInitialize:
    def test_initializes_all_trainable_params(self) -> None:
        """Initializes all trainable params."""
        model = _make_linear()
        ema = ParameterEMA()
        ema.initialize(model)
        assert len(ema.shadow) == 2  # weight + bias
        assert "weight" in ema.shadow
        assert "bias" in ema.shadow

    def test_skips_frozen_params(self) -> None:
        """Skips frozen params."""
        model = _make_model_with_frozen()
        ema = ParameterEMA()
        ema.initialize(model)
        assert len(ema.shadow) == 1  # only weight
        assert "weight" in ema.shadow
        assert "bias" not in ema.shadow

    def test_shadow_is_detached_clone(self) -> None:
        """Shadow is detached clone."""
        model = _make_linear()
        ema = ParameterEMA()
        ema.initialize(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert not ema.shadow[name].requires_grad
                assert ema.shadow[name] is not param
                assert torch.equal(ema.shadow[name], param.data)

    def test_multiple_initializations_overwrite(self) -> None:
        """Multiple initializations overwrite."""
        model1 = _make_linear()
        model2 = _make_linear()
        # Give model2 different weights
        with torch.no_grad():
            model2.weight.add_(1.0)
            model2.bias.add_(1.0)

        ema = ParameterEMA()
        ema.initialize(model1)
        first_weight = ema.shadow["weight"].clone()

        ema.initialize(model2)
        second_weight = ema.shadow["weight"].clone()
        assert not torch.equal(first_weight, second_weight)


# ═══════════════════════════════════════════════════════════════════════════════
# ParameterEMA — update
# ═══════════════════════════════════════════════════════════════════════════════


class TestParameterEMAUpdate:
    def test_auto_initializes_if_shadow_empty(self) -> None:
        """Auto initializes if shadow empty."""
        model = _make_linear()
        ema = ParameterEMA()
        ema.update(model)
        assert len(ema.shadow) == 2

    def test_updates_toward_current_params(self) -> None:
        """Updates toward current params."""
        model = _make_linear()
        ema = ParameterEMA(decay=0.5)
        ema.initialize(model)

        # Shift parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        old_shadow = {k: v.clone() for k, v in ema.shadow.items()}
        ema.update(model)

        # After update with decay=0.5: shadow = 0.5*old + 0.5*new
        for name in ema.shadow:
            new_param = dict(model.named_parameters())[name].data
            expected = 0.5 * old_shadow[name] + 0.5 * new_param
            assert torch.allclose(ema.shadow[name], expected)

    def test_decay_one_keeps_shadow_unchanged(self) -> None:
        """With decay=1.0, shadow never moves toward new params, stays at initial."""
        model = _make_linear()
        ema = ParameterEMA(decay=1.0)
        ema.initialize(model)
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        ema.update(model)
        for name in ema.shadow:
            assert torch.equal(ema.shadow[name], initial_shadow[name])

    def test_decay_near_zero_tracks_current_params(self) -> None:
        """With decay≈0, shadow moves almost entirely to the current params."""
        model = _make_linear()
        ema = ParameterEMA(decay=0.01)
        ema.initialize(model)

        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        ema.update(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                # shadow ≈ param (99% toward param, 1% old)
                assert torch.allclose(ema.shadow[name], param.data, atol=0.01)

    def test_multiple_updates_decay(self) -> None:
        """After many updates, shadow converges toward recent values."""
        model = _make_linear()
        ema = ParameterEMA(decay=0.9)
        ema.initialize(model)

        for _ in range(20):
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(0.1)
            ema.update(model)

        for name, param in model.named_parameters():
            if param.requires_grad:
                # After many updates with consistent drift, shadow should lag
                assert not torch.equal(ema.shadow[name], param.data)

    def test_update_with_frozen_params_skips_them(self) -> None:
        """Update with frozen params skips them."""
        model = _make_model_with_frozen()
        ema = ParameterEMA()
        ema.initialize(model)

        with torch.no_grad():
            model.weight.add_(1.0)
            model.bias.add_(1.0)  # frozen param changed

        ema.update(model)
        # Shadow for bias should not exist (was never initialized)
        assert "bias" not in ema.shadow

    def test_repeated_updates_reduce_variance(self) -> None:
        """EMA with decay < 1 should dampen parameter oscillations."""
        model = _make_linear()
        ema = ParameterEMA(decay=0.8)
        ema.initialize(model)

        # Oscillate parameters ±0.5
        shadows_over_time = []
        for i in range(10):
            sign = 1.0 if i % 2 == 0 else -1.0
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(sign * 0.5)
            ema.update(model)
            shadows_over_time.append(ema.shadow["weight"].clone())

        # Shadows should vary less than the raw params (dampened)
        shadow_std = torch.stack(shadows_over_time).std(dim=0).mean()
        assert shadow_std < 0.3  # much less than 0.5 oscillation


# ═══════════════════════════════════════════════════════════════════════════════
# ParameterEMA — copy_to
# ═══════════════════════════════════════════════════════════════════════════════


class TestParameterEMACopyTo:
    def test_copies_shadow_to_model(self) -> None:
        """Copies shadow to model."""
        model = _make_linear()
        ema = ParameterEMA()
        ema.initialize(model)

        # Shift model params, update EMA, then copy back
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        ema.update(model)

        # Reset model to different values
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()

        ema.copy_to(model)
        # Model params should now equal EMA shadow (not zero)
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert torch.equal(param.data, ema.shadow[name])

    def test_copy_to_empty_shadow_raises(self) -> None:
        """Copy to empty shadow raises."""
        model = _make_linear()
        ema = ParameterEMA()
        with pytest.raises(RuntimeError, match="empty"):
            ema.copy_to(model)

    def test_copy_to_preserves_requires_grad_flags(self) -> None:
        """Copy to preserves requires grad flags."""
        model = _make_model_with_frozen()
        ema = ParameterEMA()
        ema.initialize(model)

        with torch.no_grad():
            model.weight.add_(1.0)
        ema.update(model)

        # Reset
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()

        ema.copy_to(model)
        # Bias should still be zero (not copied) and frozen
        assert model.weight.requires_grad
        assert not model.bias.requires_grad
        assert torch.all(model.bias == 0.0)

    def test_copy_to_param_not_in_shadow_skipped(self) -> None:
        """If model gains a new requires_grad param not in shadow, it's skipped."""
        model = _make_linear()
        ema = ParameterEMA()
        ema.initialize(model)

        # Add a new trainable param to the model
        model.register_parameter("extra", nn.Parameter(torch.ones(1)))
        original_extra = model.extra.data.clone()

        ema.copy_to(model)
        # extra was not in shadow, should remain unchanged
        assert torch.equal(model.extra.data, original_extra)
