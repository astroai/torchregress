"""
Unit tests for torchregress.ensemble.mc_dropout — MC-Dropout uncertainty estimation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchregress.ensemble.mc_dropout import (
    MCDropoutWrapper,
    enable_dropout,
)


def _make_dropout_model() -> nn.Module:
    """Simple model with dropout for MC-Dropout testing."""
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(8, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# enable_dropout / disable_dropout
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnableDisableDropout:
    def test_enable_dropout_sets_train_mode(self) -> None:
        """enable_dropout sets all Dropout layers to train mode."""
        model = _make_dropout_model()
        model.eval()  # start in eval mode
        enable_dropout(model)
        # The Dropout layer should now be in train mode
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                assert module.training

    def test_enable_dropout_ignores_non_dropout(self) -> None:
        """Non-Dropout layers are unaffected by enable/disable."""
        model = _make_dropout_model()
        model.eval()
        enable_dropout(model)
        for module in model.modules():
            if not isinstance(module, nn.Dropout):
                assert not module.training


# ═══════════════════════════════════════════════════════════════════════════════
# MCDropoutWrapper
# ═══════════════════════════════════════════════════════════════════════════════


class TestMCDropoutWrapperInit:
    def test_default_construction(self) -> None:
        """Default n_samples=30, dropout_rate=None."""
        model = _make_dropout_model()
        wrapper = MCDropoutWrapper(model)
        assert wrapper.n_samples == 30

    def test_custom_n_samples(self) -> None:
        """Custom n_samples stored."""
        model = _make_dropout_model()
        wrapper = MCDropoutWrapper(model, n_samples=10)
        assert wrapper.n_samples == 10

    def test_dropout_rate_overrides(self) -> None:
        """When dropout_rate is set, existing Dropout rates are updated."""
        model = _make_dropout_model()
        wrapper = MCDropoutWrapper(model, dropout_rate=0.5)
        for module in wrapper.model.modules():
            if isinstance(module, nn.Dropout):
                assert module.p == 0.5


class TestMCDropoutWrapperForward:
    def test_forward_shape(self) -> None:
        """Standard forward returns [batch, out]."""
        model = _make_dropout_model()
        wrapper = MCDropoutWrapper(model)
        x = torch.randn(8, 4)
        out = wrapper(x)
        assert out.shape == (8, 1)

    def test_forward_disables_dropout_without_permanent_mutation(self) -> None:
        """Standard forward runs with dropout disabled and restores the prior mode."""
        model = _make_dropout_model()
        wrapper = MCDropoutWrapper(model)
        wrapper.model.train()  # force train mode
        x = torch.randn(4, 4)
        wrapper(x)
        # TR-ENS-03: the pass itself ran in eval; original training flags are restored.
        for module in wrapper.model.modules():
            if isinstance(module, nn.Dropout):
                assert module.training

    def test_forward_deterministic(self) -> None:
        """Two forward calls with same input should be identical (dropout disabled)."""
        model = _make_dropout_model()
        wrapper = MCDropoutWrapper(model)
        x = torch.randn(4, 4)
        out1 = wrapper(x)
        out2 = wrapper(x)
        assert torch.allclose(out1, out2)


class TestMCDropoutWrapperMCForward:
    def test_mc_forward_shape(self) -> None:
        """MC forward returns [n_samples, batch, out]."""
        model = _make_dropout_model()
        wrapper = MCDropoutWrapper(model, n_samples=5)
        x = torch.randn(4, 4)
        out = wrapper.mc_forward(x)
        assert out.shape == (5, 4, 1)

    def test_mc_forward_custom_n_samples(self) -> None:
        """Custom n_samples overrides default."""
        model = _make_dropout_model()
        wrapper = MCDropoutWrapper(model, n_samples=5)
        x = torch.randn(4, 4)
        out = wrapper.mc_forward(x, n_samples=3)
        assert out.shape == (3, 4, 1)

    def test_mc_forward_produces_variance(self) -> None:
        """MC samples should vary due to stochastic dropout."""
        model = _make_dropout_model()
        wrapper = MCDropoutWrapper(model, n_samples=20)
        x = torch.randn(2, 4)
        samples = wrapper.mc_forward(x)
        stds = samples.std(dim=0)
        assert torch.any(stds > 0)


class TestMCDropoutWrapperPredict:
    def test_predict_with_uncertainty_shapes(self) -> None:
        """Returns (mean, std) with correct shapes."""
        model = _make_dropout_model()
        wrapper = MCDropoutWrapper(model, n_samples=10)
        x = torch.randn(4, 4)
        mean, std = wrapper.predict_with_uncertainty(x)
        assert mean.shape == (4, 1)
        assert std.shape == (4, 1)

    def test_predict_interval_shapes(self) -> None:
        """Returns (lower, upper) with correct shapes."""
        model = _make_dropout_model()
        wrapper = MCDropoutWrapper(model, n_samples=10)
        x = torch.randn(4, 4)
        lower, upper = wrapper.predict_interval(x)
        assert lower.shape == (4, 1)
        assert upper.shape == (4, 1)
        assert torch.all(lower <= upper)

    def test_predict_interval_confidence(self) -> None:
        """Higher confidence gives wider intervals."""
        torch.manual_seed(42)
        model = _make_dropout_model()
        wrapper = MCDropoutWrapper(model, n_samples=50)
        x = torch.randn(4, 4)
        lower_90, upper_90 = wrapper.predict_interval(x, confidence=0.90)
        lower_99, upper_99 = wrapper.predict_interval(x, confidence=0.99)
        width_90 = (upper_90 - lower_90).mean()
        width_99 = (upper_99 - lower_99).mean()
        assert width_99 > width_90


# ═══════════════════════════════════════════════════════════════════════════════
# MCDropoutModel
# ═══════════════════════════════════════════════════════════════════════════════


class TestMCDropoutModelInit:
    def test_default_construction(self) -> None:
        """Default dropout_rate=0.2, n_samples=30."""
        model = MCDropoutWrapper(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 1)),
            n_samples=30,
        )
        for module in model.model.modules():
            if isinstance(module, nn.Dropout):
                assert module.p == 0.2
        assert model.n_samples == 30

    def test_custom_params(self) -> None:
        """Custom dropout_rate and n_samples stored."""
        model = MCDropoutWrapper(
            nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(8, 2),
            ),
            dropout_rate=0.4,
            n_samples=50,
        )
        for module in model.model.modules():
            if isinstance(module, nn.Dropout):
                assert module.p == 0.4
        assert model.n_samples == 50

    def test_has_dropout_layers(self) -> None:
        """Model includes Dropout layers between hidden layers."""
        model = MCDropoutWrapper(
            nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(8, 1),
            ),
            n_samples=30,
        )
        dropout_count = sum(1 for m in model.model.modules() if isinstance(m, nn.Dropout))
        assert dropout_count == 2  # one per hidden layer


class TestMCDropoutModelForward:
    def test_forward_shape(self) -> None:
        """Forward returns [batch, out]."""
        model = MCDropoutWrapper(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 2)),
            n_samples=30,
        )
        x = torch.randn(8, 4)
        out = model(x)
        assert out.shape == (8, 2)


class TestMCDropoutModelMCForward:
    def test_mc_forward_shape(self) -> None:
        """MC forward returns [n_samples, batch, out]."""
        model = MCDropoutWrapper(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 1)),
            n_samples=10,
        )
        x = torch.randn(4, 4)
        out = model.mc_forward(x)
        assert out.shape == (10, 4, 1)

    def test_mc_forward_custom_n_samples(self) -> None:
        """Custom n_samples overrides default."""
        model = MCDropoutWrapper(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 1)),
            n_samples=10,
        )
        x = torch.randn(4, 4)
        out = model.mc_forward(x, n_samples=3)
        assert out.shape == (3, 4, 1)

    def test_mc_forward_produces_variance(self) -> None:
        """MC samples should vary."""
        model = MCDropoutWrapper(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 1)),
            dropout_rate=0.5,
            n_samples=20,
        )
        x = torch.randn(2, 4)
        samples = model.mc_forward(x)
        stds = samples.std(dim=0)
        assert torch.any(stds > 0)


class TestMCDropoutModelPredict:
    def test_predict_with_uncertainty_shapes(self) -> None:
        """Returns (mean, std) with correct shapes."""
        model = MCDropoutWrapper(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 1)),
            n_samples=10,
        )
        x = torch.randn(4, 4)
        mean, std = model.predict_with_uncertainty(x)
        assert mean.shape == (4, 1)
        assert std.shape == (4, 1)

    def test_predict_interval_shapes(self) -> None:
        """Returns (lower, upper) and lower <= upper."""
        model = MCDropoutWrapper(
            nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 1)),
            n_samples=10,
        )
        x = torch.randn(4, 4)
        lower, upper = model.predict_interval(x)
        assert lower.shape == (4, 1)
        assert upper.shape == (4, 1)
        assert torch.all(lower <= upper)
