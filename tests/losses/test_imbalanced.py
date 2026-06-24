"""Tests for imbalanced regression losses."""

import torch
import torch.nn.functional as F

from torchregress.losses import (
    DensityWeightedLoss,
    FeatureDistributionSmoother,
    FocalRLoss,
    LDSLoss,
    PropensityWeightedLoss,
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


class TestPropensityWeightedLoss:
    """Tests for PropensityWeightedLoss."""

    def test_forward_with_propensity(self):
        loss_fn = PropensityWeightedLoss(base_loss="mse")
        pred = torch.randn(32, 1)
        target = torch.randn(32, 1)
        propensity = torch.rand(32, 1) * 0.8 + 0.1
        observed = torch.ones(32, 1)

        loss = loss_fn(pred, target, propensity=propensity, observed=observed)
        assert torch.isfinite(loss)

    def test_missing_propensity_raises(self):
        loss_fn = PropensityWeightedLoss()
        pred = torch.randn(16, 1)
        target = torch.randn(16, 1)
        try:
            loss_fn(pred, target)
        except ValueError as exc:
            assert "propensity" in str(exc).lower()
        else:
            raise AssertionError("Expected ValueError when propensity is missing")


class TestFeatureDistributionSmoother:
    """Tests for FeatureDistributionSmoother."""

    def test_init_and_fit(self):
        """Test FDS initialization and fitting of target bins."""
        fds = FeatureDistributionSmoother(feature_dim=10, n_bins=10)
        assert fds.feature_dim == 10
        assert fds.n_bins == 10
        assert fds.kernel == "gaussian"

        # Fit target bins
        targets = torch.linspace(0.0, 10.0, 100)
        fds.fit(targets, n_bins=10)
        assert fds._bins.shape == (11,)
        # Check that reset works
        assert torch.allclose(fds.running_mean, torch.zeros(10, 10))
        assert torch.allclose(fds.running_var, torch.ones(10, 10))

    def test_different_kernels(self):
        """Test initialization with different kernels."""
        for kernel in ["gaussian", "triang", "laplace"]:
            fds = FeatureDistributionSmoother(feature_dim=4, n_bins=20, kernel=kernel)
            assert fds.kernel_window.shape == (5,)

    def test_running_stats_updates(self):
        """Test that running stats are updated correctly."""
        fds = FeatureDistributionSmoother(feature_dim=2, n_bins=5, momentum=0.9)
        targets = torch.tensor([0.0, 2.5, 5.0, 7.5, 10.0])
        fds.fit(targets, n_bins=5)

        # Batch update
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        # Let features belong to different bins
        fds.update_running_stats(features, targets, epoch=0)

        # Check sample tracking
        assert torch.allclose(fds.num_samples_tracked, torch.ones(5))
        # With factor=0.0 on epoch=0 (start_update_epoch), running stats should equal batch features
        assert torch.allclose(fds.running_mean, features)

    def test_smoothing_convolution(self):
        """Test statistics smoothing via 1D convolution."""
        fds = FeatureDistributionSmoother(feature_dim=2, n_bins=5)
        targets = torch.linspace(0.0, 10.0, 5)
        fds.fit(targets, n_bins=5)

        # Pre-set some running statistics
        fds.running_mean.copy_(torch.arange(10, dtype=torch.float32).view(5, 2))
        fds.running_var.copy_(torch.arange(10, dtype=torch.float32).view(5, 2) + 1.0)

        fds.update_last_epoch_stats(epoch=1)

        # Running last epoch stats should match running stats
        assert torch.allclose(fds.running_mean_last_epoch, fds.running_mean)
        assert torch.allclose(fds.running_var_last_epoch, fds.running_var)

        # Smoothed stats should be different due to convolution
        assert not torch.allclose(fds.smoothed_mean_last_epoch, fds.running_mean)
        assert fds.smoothed_mean_last_epoch.shape == (5, 2)

    def test_forward_calibration(self):
        """Test feature calibration / forward pass."""
        fds = FeatureDistributionSmoother(feature_dim=2, n_bins=5, start_smooth_epoch=1)
        targets = torch.linspace(0.0, 10.0, 5)
        fds.fit(targets, n_bins=5)

        # Set statistics
        fds.running_mean.copy_(torch.ones(5, 2))
        fds.running_var.copy_(torch.ones(5, 2) * 2.0)
        fds.update_last_epoch_stats(epoch=1)

        features = torch.zeros(5, 2)
        # Epoch < start_smooth_epoch: should return features untouched
        out_raw = fds(features, targets, epoch=0)
        assert torch.allclose(out_raw, features)

        # Epoch >= start_smooth_epoch: calibration applies
        # features = 0.0, running_mean_last_epoch = 1.0, running_var_last_epoch = 2.0
        # smoothed_mean_last_epoch = 1.0, smoothed_var_last_epoch = 2.0 (since running stats are constant)
        # factor = v2 / v1 = 1.0
        # calibrated = (0.0 - 1.0) * sqrt(1.0) + 1.0 = 0.0
        out_cal = fds(features, targets, epoch=1)
        assert torch.allclose(out_cal, features)

    def test_gradient_flow(self):
        """Test that gradients propagate through calibrated features."""
        fds = FeatureDistributionSmoother(feature_dim=4, n_bins=10, start_smooth_epoch=0)
        targets = torch.linspace(0.0, 10.0, 10)
        fds.fit(targets, n_bins=10)
        fds.update_last_epoch_stats(epoch=0)

        features = torch.randn(10, 4, requires_grad=True)
        out = fds(features, targets, epoch=0)

        loss = out.sum()
        loss.backward()

        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_pad_mode_fallback(self):
        """Test replicate padding fallback when n_bins is small."""
        # half_ks = 2, n_bins = 2 (n_bins <= half_ks)
        fds = FeatureDistributionSmoother(feature_dim=2, n_bins=2, kernel_size=5)
        targets = torch.linspace(0.0, 10.0, 2)
        fds.fit(targets, n_bins=2)
        # Should not crash during stat update / convolution
        fds.update_last_epoch_stats(epoch=1)
        assert fds.smoothed_mean_last_epoch.shape == (2, 2)
