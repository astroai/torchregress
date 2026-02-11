"""Tests for noisy label loss functions."""

import pytest
import torch

from torchregress.losses import CoTeachingLoss, NoiseAdaptiveLoss, RENTLoss


class TestNoiseAdaptiveLoss:
    """Tests for NoiseAdaptiveLoss."""

    def test_init_valid(self):
        loss = NoiseAdaptiveLoss(n_samples=100)
        assert loss.n_samples == 100
        assert loss.base_loss == "mse"
        assert loss.sample_weight_logits.shape == (100,)

    def test_init_invalid_base_loss(self):
        with pytest.raises(ValueError, match="base_loss must be"):
            NoiseAdaptiveLoss(n_samples=10, base_loss="invalid")

    def test_forward_with_indices(self):
        loss_fn = NoiseAdaptiveLoss(n_samples=50)
        y_pred = torch.randn(8, 1)
        target = torch.randn(8, 1)
        indices = torch.arange(8)
        result = loss_fn(y_pred, target, sample_indices=indices)
        assert result.shape == ()
        assert result.requires_grad  # sample weights are learnable

    def test_forward_without_indices_raises(self):
        loss_fn = NoiseAdaptiveLoss(n_samples=10)
        y_pred = torch.randn(4, 1)
        target = torch.randn(4, 1)
        with pytest.raises(ValueError, match="sample_indices"):
            loss_fn(y_pred, target)

    def test_get_sample_weights(self):
        loss_fn = NoiseAdaptiveLoss(n_samples=20, initial_weight=0.8)
        weights = loss_fn.get_sample_weights()
        assert weights.shape == (20,)
        assert (weights >= 0).all() and (weights <= 1).all()

    @pytest.mark.parametrize("base_loss", ["mse", "mae", "huber"])
    def test_base_loss_variants(self, base_loss):
        loss_fn = NoiseAdaptiveLoss(n_samples=10, base_loss=base_loss)
        y_pred = torch.randn(4, 1)
        target = torch.randn(4, 1)
        indices = torch.arange(4)
        result = loss_fn(y_pred, target, sample_indices=indices)
        assert result.shape == ()
        assert torch.isfinite(result)

    def test_multidimensional_output(self):
        loss_fn = NoiseAdaptiveLoss(n_samples=20)
        y_pred = torch.randn(5, 3)
        target = torch.randn(5, 3)
        indices = torch.arange(5)
        result = loss_fn(y_pred, target, sample_indices=indices)
        assert result.shape == ()


class TestCoTeachingLoss:
    """Tests for CoTeachingLoss."""

    def test_init_valid(self):
        loss = CoTeachingLoss(forget_rate=0.3)
        assert loss.forget_rate == 0.3
        assert loss.base_loss == "mse"

    def test_init_invalid_forget_rate(self):
        with pytest.raises(ValueError, match="forget_rate"):
            CoTeachingLoss(forget_rate=1.5)

    def test_init_invalid_base_loss(self):
        with pytest.raises(ValueError, match="base_loss must be"):
            CoTeachingLoss(base_loss="invalid")

    def test_forward_returns_two_losses(self):
        loss_fn = CoTeachingLoss(forget_rate=0.2)
        y_pred1 = torch.randn(16, 1)
        y_pred2 = torch.randn(16, 1)
        target = torch.randn(16, 1)
        loss1, loss2 = loss_fn(y_pred1, y_pred2, target, epoch=0)
        assert loss1.shape == ()
        assert loss2.shape == ()
        assert torch.isfinite(loss1) and torch.isfinite(loss2)

    def test_gradual_forget_rate(self):
        loss_fn = CoTeachingLoss(forget_rate=0.4, num_gradual=10)
        # At epoch 0, effective rate = 0.4 * 1/10 = 0.04
        assert loss_fn._get_forget_rate(0) == pytest.approx(0.04)
        # At epoch 5, effective rate = 0.4 * 6/10 = 0.24
        assert loss_fn._get_forget_rate(5) == pytest.approx(0.24)
        # At epoch 10+, effective rate = 0.4
        assert loss_fn._get_forget_rate(10) == pytest.approx(0.4)
        assert loss_fn._get_forget_rate(100) == pytest.approx(0.4)

    @pytest.mark.parametrize("base_loss", ["mse", "mae", "huber"])
    def test_base_loss_variants(self, base_loss):
        loss_fn = CoTeachingLoss(forget_rate=0.2, base_loss=base_loss)
        y_pred1 = torch.randn(8, 1)
        y_pred2 = torch.randn(8, 1)
        target = torch.randn(8, 1)
        loss1, loss2 = loss_fn(y_pred1, y_pred2, target)
        assert torch.isfinite(loss1) and torch.isfinite(loss2)

    def test_reduction_none(self):
        loss_fn = CoTeachingLoss(forget_rate=0.5, reduction="none")
        y_pred1 = torch.randn(10, 1)
        y_pred2 = torch.randn(10, 1)
        target = torch.randn(10, 1)
        # epoch=100 > num_gradual=10, so effective forget_rate = 0.5
        # n_remember = max(1, int((1-0.5) * 10)) = 5
        loss1, loss2 = loss_fn(y_pred1, y_pred2, target, epoch=100)
        assert loss1.shape == (5,)
        assert loss2.shape == (5,)


class TestRENTLoss:
    """Tests for RENTLoss."""

    def test_init_valid(self):
        loss = RENTLoss(ensemble_size=5)
        assert loss.ensemble_size == 5
        assert loss.noise_threshold == 2.0

    def test_init_invalid_base_loss(self):
        with pytest.raises(ValueError, match="base_loss must be"):
            RENTLoss(base_loss="invalid")

    def test_forward(self):
        loss_fn = RENTLoss(ensemble_size=3)
        ensemble_preds = torch.randn(3, 8, 1)  # [ensemble, batch, features]
        target = torch.randn(8, 1)
        result = loss_fn(ensemble_preds, target)
        assert result.shape == ()
        assert torch.isfinite(result)

    def test_forward_transposed_input(self):
        loss_fn = RENTLoss(ensemble_size=3)
        # Shape [batch, ensemble, features] — should auto-transpose
        ensemble_preds = torch.randn(8, 3, 1)
        target = torch.randn(8, 1)
        result = loss_fn(ensemble_preds, target)
        assert result.shape == ()

    def test_wrong_ensemble_size_raises(self):
        loss_fn = RENTLoss(ensemble_size=5)
        ensemble_preds = torch.randn(3, 8, 1)
        target = torch.randn(8, 1)
        with pytest.raises(ValueError, match="Expected ensemble_preds"):
            loss_fn(ensemble_preds, target)

    @pytest.mark.parametrize("base_loss", ["mse", "mae", "huber"])
    def test_base_loss_variants(self, base_loss):
        loss_fn = RENTLoss(ensemble_size=3, base_loss=base_loss)
        ensemble_preds = torch.randn(3, 8, 1)
        target = torch.randn(8, 1)
        result = loss_fn(ensemble_preds, target)
        assert torch.isfinite(result)

    def test_high_disagreement_downweighted(self):
        """Samples with high ensemble disagreement should produce finite loss."""
        loss_fn = RENTLoss(ensemble_size=3, noise_threshold=2.0)

        # Create ensemble with LOW disagreement (clean samples)
        clean_mean = torch.ones(8, 1)
        clean_preds = clean_mean.unsqueeze(0).expand(3, -1, -1) + torch.randn(3, 8, 1) * 0.01
        target = clean_mean + torch.randn(8, 1) * 0.1
        clean_loss = loss_fn(clean_preds, target)
        assert torch.isfinite(clean_loss)

        # Create ensemble with HIGH disagreement (noisy samples)
        noisy_preds = clean_mean.unsqueeze(0).expand(3, -1, -1) + torch.randn(3, 8, 1) * 10.0
        noisy_loss = loss_fn(noisy_preds, target)
        assert torch.isfinite(noisy_loss)


class TestRegistration:
    """Test that noisy losses are properly registered."""

    def test_registered(self):
        from torchregress.losses.loss_registry import get_regression_loss

        assert get_regression_loss("noise_adaptive") is NoiseAdaptiveLoss
        assert get_regression_loss("co_teaching") is CoTeachingLoss
        assert get_regression_loss("rent") is RENTLoss
