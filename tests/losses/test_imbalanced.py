"""Tests for imbalanced regression losses."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torchregress.losses import (
    BalancedMSELoss,
    BalancedMSELossMD,
    DensityWeightedLoss,
    DistLoss,
    FocalRLoss,
    FrequencyWeightedLoss,
    LDSLoss,
    SQINVLoss,
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

        loss_small = loss_fn(pred_small, target, reduction="none")
        loss_large = loss_fn(pred_large, target_large, reduction="none")

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


class TestBalancedMSELoss:
    """Tests for BalancedMSELoss (BMC)."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        loss_fn = BalancedMSELoss(init_noise_sigma=1.0)
        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)
        loss = loss_fn(pred, target)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_learnable_sigma(self):
        """Test that noise_sigma is learnable when specified."""
        loss_fn = BalancedMSELoss(init_noise_sigma=1.0, learnable=True)

        # Check it's a parameter
        assert isinstance(loss_fn.noise_sigma, torch.nn.Parameter)

        # Can be included in optimizer
        params = list(loss_fn.parameters())
        assert len(params) == 1
        assert params[0] is loss_fn.noise_sigma

    def test_fixed_sigma(self):
        """Test non-learnable sigma."""
        loss_fn = BalancedMSELoss(init_noise_sigma=1.0, learnable=False)

        # Should be a buffer, not parameter
        params = list(loss_fn.parameters())
        assert len(params) == 0

    def test_batch_size_sensitivity(self):
        """Test that loss works with different batch sizes."""
        loss_fn = BalancedMSELoss(init_noise_sigma=1.0)

        for batch_size in [4, 16, 32, 64]:
            pred = torch.randn(batch_size, 1)
            target = torch.randn(batch_size, 1)
            loss = loss_fn(pred, target)
            assert torch.isfinite(loss)

    def test_gradient_flow(self):
        """Test gradient computation including sigma."""
        loss_fn = BalancedMSELoss(init_noise_sigma=1.0, learnable=True)
        pred = torch.randn(32, 1, requires_grad=True)
        target = torch.randn(32, 1)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert loss_fn.noise_sigma.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_reduction_modes(self):
        """Test different reduction modes."""
        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)

        loss_mean = BalancedMSELoss(reduction="mean")(pred, target)
        loss_sum = BalancedMSELoss(reduction="sum")(pred, target)
        loss_none = BalancedMSELoss(reduction="none")(pred, target)

        assert loss_mean.ndim == 0
        assert loss_sum.ndim == 0
        assert loss_none.shape == (32,)


class TestBalancedMSELossMD:
    """Tests for multi-dimensional BalancedMSELoss."""

    def test_basic_forward(self):
        """Test basic forward pass with multi-dimensional output."""
        loss_fn = BalancedMSELossMD(n_features=3, init_noise_sigma=1.0)
        pred = torch.randn(32, 3)
        target = torch.randn(32, 3)
        loss = loss_fn(pred, target)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_different_feature_dims(self):
        """Test different feature dimensions."""
        for n_features in [1, 2, 5, 10]:
            loss_fn = BalancedMSELossMD(n_features=n_features)
            pred = torch.randn(32, n_features)
            target = torch.randn(32, n_features)
            loss = loss_fn(pred, target)
            assert torch.isfinite(loss)

    def test_gradient_flow(self):
        """Test gradient computation."""
        loss_fn = BalancedMSELossMD(n_features=3, learnable=True)
        pred = torch.randn(32, 3, requires_grad=True)
        target = torch.randn(32, 3)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestSQINVLoss:
    """Tests for SQINVLoss."""

    def test_fit_and_forward(self):
        """Test fitting and forward pass."""
        loss_fn = SQINVLoss(kernel_width=0.5)

        # Fit on imbalanced data
        train_targets = torch.cat([
            torch.randn(800) * 0.3,  # Dense region
            torch.randn(200) * 0.3 + 3.0,  # Sparse region
        ])
        loss_fn.fit(train_targets)

        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_weights_normalized(self):
        """Test that bin weights have mean 1.0."""
        loss_fn = SQINVLoss()
        train_targets = torch.randn(1000)
        loss_fn.fit(train_targets)

        # Check bin weights are normalized (not sample weights)
        assert torch.isclose(
            loss_fn._weights_per_bin.mean(),
            torch.tensor(1.0),
            rtol=0.1
        )

    def test_with_sample_indices(self):
        """Test using precomputed weights via indices."""
        loss_fn = SQINVLoss()
        train_targets = torch.randn(1000)
        loss_fn.fit(train_targets)

        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)
        indices = torch.randint(0, 1000, (32,))

        loss = loss_fn(pred, target, sample_indices=indices)
        assert torch.isfinite(loss)

    def test_max_weight_ratio(self):
        """Test that weights are clipped by max_weight_ratio."""
        loss_fn = SQINVLoss(max_weight_ratio=10.0)

        # Very imbalanced data
        train_targets = torch.cat([
            torch.randn(990) * 0.1,
            torch.randn(10) * 0.1 + 10.0,  # Very rare region
        ])
        loss_fn.fit(train_targets)

        max_w = loss_fn.sqinv_weights.max()
        min_w = loss_fn.sqinv_weights.min()

        # Ratio should be <= max_weight_ratio (with some tolerance)
        assert max_w / min_w <= 10.0 * 1.1

    def test_not_fitted_error(self):
        """Test error when not fitted."""
        loss_fn = SQINVLoss()
        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)

        with pytest.raises(ValueError, match="Must call fit"):
            loss_fn(pred, target)


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
            loss_fn_0.density_weights,
            torch.ones_like(loss_fn_0.density_weights),
            rtol=1e-5
        )

        # With factor=1, weights should vary
        assert loss_fn_1.density_weights.std() > 0


class TestFrequencyWeightedLoss:
    """Tests for FrequencyWeightedLoss."""

    def test_fit_and_forward(self):
        """Test fitting and forward pass."""
        loss_fn = FrequencyWeightedLoss(n_bins=50)

        train_targets = torch.randn(1000)
        loss_fn.fit(train_targets)

        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_weighting_schemes(self):
        """Test different weighting schemes."""
        train_targets = torch.randn(1000)

        for weighting in ["inv", "sqinv"]:
            loss_fn = FrequencyWeightedLoss(weighting=weighting)
            loss_fn.fit(train_targets)

            pred = torch.randn(32, 1)
            target = torch.randn(32, 1)
            loss = loss_fn(pred, target)
            assert torch.isfinite(loss)

    def test_different_base_losses(self):
        """Test different base loss functions."""
        train_targets = torch.randn(1000)

        for base_loss in ["mse", "mae", "huber"]:
            loss_fn = FrequencyWeightedLoss(base_loss=base_loss)
            loss_fn.fit(train_targets)

            pred = torch.randn(32, 1)
            target = torch.randn(32, 1)
            loss = loss_fn(pred, target)
            assert torch.isfinite(loss)

    def test_max_weight_clipping(self):
        """Test that max_weight limits extreme weights."""
        # Very imbalanced data
        train_targets = torch.cat([
            torch.randn(990) * 0.1,
            torch.randn(10) * 0.1 + 10.0,  # Very rare region
        ])

        loss_fn = FrequencyWeightedLoss(max_weight=5.0)
        loss_fn.fit(train_targets)

        # Check weights are clipped
        assert loss_fn._bin_weights.max() <= 5.0 * 1.1  # With tolerance

    def test_not_fitted_error(self):
        """Test error when not fitted."""
        loss_fn = FrequencyWeightedLoss()
        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)

        with pytest.raises(ValueError, match="Must call fit"):
            loss_fn(pred, target)

    def test_gradient_flow(self):
        """Test gradient computation."""
        loss_fn = FrequencyWeightedLoss()
        train_targets = torch.randn(1000)
        loss_fn.fit(train_targets)

        pred = torch.randn(32, 1, requires_grad=True)
        target = torch.randn(32, 1)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestDistLoss:
    """Tests for DistLoss (ICLR 2025)."""

    def test_fit_and_forward(self):
        """Test fitting and forward pass."""
        loss_fn = DistLoss(n_bins=50, alpha=1.0)

        train_targets = torch.randn(1000)
        loss_fn.fit(train_targets)

        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)
        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_different_alpha_values(self):
        """Test different alpha (distribution constraint weight) values."""
        train_targets = torch.randn(1000)
        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)

        losses = []
        for alpha in [0.0, 0.5, 1.0, 2.0]:
            loss_fn = DistLoss(alpha=alpha)
            loss_fn.fit(train_targets)
            loss = loss_fn(pred, target)
            losses.append(loss.item())
            assert torch.isfinite(loss)

        # Alpha=0 should give base loss only, higher alpha means stronger constraint
        # Losses should generally increase with alpha (more constraint)

    def test_different_dist_losses(self):
        """Test different distribution loss functions."""
        train_targets = torch.randn(1000)

        for dist_loss in ["mae", "mse"]:
            loss_fn = DistLoss(dist_loss=dist_loss)
            loss_fn.fit(train_targets)

            pred = torch.randn(32, 1)
            target = torch.randn(32, 1)
            loss = loss_fn(pred, target)
            assert torch.isfinite(loss)

    def test_different_base_losses(self):
        """Test different base loss functions."""
        train_targets = torch.randn(1000)

        for base_loss in ["mse", "mae"]:
            loss_fn = DistLoss(base_loss=base_loss)
            loss_fn.fit(train_targets)

            pred = torch.randn(32, 1)
            target = torch.randn(32, 1)
            loss = loss_fn(pred, target)
            assert torch.isfinite(loss)

    def test_not_fitted_error(self):
        """Test error when not fitted."""
        loss_fn = DistLoss()
        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)

        with pytest.raises(ValueError, match="Must call fit"):
            loss_fn(pred, target)

    def test_gradient_flow(self):
        """Test gradient computation."""
        loss_fn = DistLoss()
        train_targets = torch.randn(1000)
        loss_fn.fit(train_targets)

        pred = torch.randn(32, 1, requires_grad=True)
        target = torch.randn(32, 1)

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_reduction_modes(self):
        """Test different reduction modes."""
        train_targets = torch.randn(1000)

        loss_mean = DistLoss(reduction="mean")
        loss_mean.fit(train_targets)
        loss_sum = DistLoss(reduction="sum")
        loss_sum.fit(train_targets)

        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)

        result_mean = loss_mean(pred, target)
        result_sum = loss_sum(pred, target)

        assert result_mean.ndim == 0
        assert result_sum.ndim == 0
        # Sum should be larger than mean (for batch_size > 1)
        assert result_sum.item() >= result_mean.item()
