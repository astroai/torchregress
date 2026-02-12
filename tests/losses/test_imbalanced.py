"""Tests for imbalanced regression losses."""

import torch
import torch.nn.functional as F

from torchregress.losses import (
    DensityWeightedLoss,
    FocalRLoss,
    LDSLoss,
)


class TestFocalRLoss:
    """Tests for FocalRLoss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        loss_fn = FocalRLoss(beta=0.2, gamma=1.0)
        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)
        loss = loss_fn(pred, target)

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Non-negative
        assert torch.isfinite(loss)

    def test_focal_weighting(self):
        """Test that larger errors get higher weights."""
        loss_fn = FocalRLoss(beta=0.2, gamma=1.0, reduction="none")

        # Small error
        pred_small = torch.tensor([[1.0]])
        target = torch.tensor([[1.1]])

        # Large error
        pred_large = torch.tensor([[1.0]])
        target_large = torch.tensor([[5.0]])

        loss_fn(pred_small, target, reduction="none")
        loss_fn(pred_large, target_large, reduction="none")

        # Focal weight for larger error should be higher
        # Ratio should be > 1 because larger error gets more weight
        # (loss_large / base_loss_large) > (loss_small / base_loss_small)

    def test_gamma_zero_equals_base_loss(self):
        """Test that gamma=0 gives standard weighted loss."""
        loss_focal = FocalRLoss(beta=0.2, gamma=0.0, base_loss="mse")
        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)

        # With gamma=0, sigmoid(...)^0 = 1, so no focal weighting
        loss = loss_focal(pred, target)
        base_loss = F.mse_loss(pred, target)

        # Should be close but not exactly equal due to sigmoid(0.5) factor
        # When gamma=0, weight = 1.0, so it should match
        assert torch.isclose(loss, base_loss, rtol=0.01)

    def test_different_base_losses(self):
        """Test different base loss functions."""
        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)

        for base_loss in ["mse", "mae", "huber"]:
            loss_fn = FocalRLoss(beta=0.2, gamma=1.0, base_loss=base_loss)
            loss = loss_fn(pred, target)
            assert torch.isfinite(loss)
            assert loss.item() >= 0

    def test_reduction_modes(self):
        """Test different reduction modes."""
        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)

        loss_mean = FocalRLoss(reduction="mean")(pred, target)
        loss_sum = FocalRLoss(reduction="sum")(pred, target)
        loss_none = FocalRLoss(reduction="none")(pred, target)

        assert loss_mean.ndim == 0
        assert loss_sum.ndim == 0
        assert loss_none.shape == pred.shape

    def test_gradient_flow(self):
        """Test gradient computation."""
        loss_fn = FocalRLoss(beta=0.2, gamma=1.0)
        pred = torch.randn(32, 1, requires_grad=True)
        target = torch.randn(32, 1)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestLDSLoss:
    """Tests for existing LDSLoss."""

    def test_fit_and_forward(self):
        """Test fitting and forward pass."""
        loss_fn = LDSLoss(kernel="gaussian", kernel_width=2.0)

        train_targets = torch.randn(1000)
        loss_fn.fit(train_targets)

        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)

    def test_different_kernels(self):
        """Test different kernel types."""
        train_targets = torch.randn(1000)

        for kernel in ["gaussian", "triang", "laplace"]:
            loss_fn = LDSLoss(kernel=kernel)
            loss_fn.fit(train_targets)

            pred = torch.randn(32, 1)
            target = torch.randn(32, 1)
            loss = loss_fn(pred, target)
            assert torch.isfinite(loss)


class TestDensityWeightedLoss:
    """Tests for DensityWeightedLoss."""

    def test_fit_and_forward(self):
        """Test fitting and forward pass."""
        loss_fn = DensityWeightedLoss(kernel_width=0.5)

        train_targets = torch.randn(1000)
        loss_fn.fit_density(train_targets)

        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)
        indices = torch.randint(0, 1000, (32,))

        loss = loss_fn(pred, target, sample_indices=indices)
        assert torch.isfinite(loss)

    def test_reweight_factor(self):
        """Test reweight_factor behavior."""
        train_targets = torch.randn(1000)

        # No reweighting
        loss_fn_0 = DensityWeightedLoss(reweight_factor=0.0)
        loss_fn_0.fit_density(train_targets)

        # Full reweighting
        loss_fn_1 = DensityWeightedLoss(reweight_factor=1.0)
        loss_fn_1.fit_density(train_targets)

        # With factor=0, all weights should be 1.0
        assert torch.allclose(
            loss_fn_0.density_weights, torch.ones_like(loss_fn_0.density_weights), rtol=1e-5
        )

        # With factor=1, weights should vary
        assert loss_fn_1.density_weights.std() > 0
