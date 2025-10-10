"""
Tests for advanced conformal prediction methods.
"""

import pytest
import torch
from torchregress.losses.conformal import (
    AdaptiveConformalLoss,
    ConformalizedQuantileLoss,
    MultiDimensionalConformalLoss,
)


class TestAdaptiveConformalLoss:
    """Test AdaptiveConformalLoss class."""

    def test_initialization(self):
        """Test initialization of AdaptiveConformalLoss."""
        loss_fn = AdaptiveConformalLoss(alpha=0.1, gamma=0.01)
        assert loss_fn.alpha == 0.1
        assert loss_fn.gamma == 0.01
        assert loss_fn.tau is None

    def test_forward_pass(self):
        """Test forward pass of AdaptiveConformalLoss."""
        loss_fn = AdaptiveConformalLoss(alpha=0.1)

        # Create dummy predictions and targets
        batch_size, n_features = 10, 3
        y_pred = torch.randn(batch_size, 2 * n_features)
        y_true = torch.randn(batch_size, n_features)

        # Forward pass
        loss = loss_fn(y_pred, y_true)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss

    def test_calibration(self):
        """Test calibration method."""
        loss_fn = AdaptiveConformalLoss(alpha=0.1)

        # Create dummy predictions and targets
        batch_size, n_features = 10, 3
        lower_pred = torch.randn(batch_size, n_features)
        upper_pred = torch.randn(batch_size, n_features)
        y_true = torch.randn(batch_size, n_features)

        # Calibrate
        tau = loss_fn.calibrate(lower_pred, upper_pred, y_true)
        assert isinstance(tau, torch.Tensor)
        assert tau.dim() == 0  # Scalar tau

    def test_predict_interval(self):
        """Test predict_interval method."""
        loss_fn = AdaptiveConformalLoss(alpha=0.1)

        # Create dummy predictions and targets
        batch_size, n_features = 10, 3
        lower_pred = torch.randn(batch_size, n_features)
        upper_pred = torch.randn(batch_size, n_features)
        y_true = torch.randn(batch_size, n_features)

        # Calibrate first
        loss_fn.calibrate(lower_pred, upper_pred, y_true)

        # Predict intervals
        lower_interval, upper_interval = loss_fn.predict_interval(lower_pred, upper_pred)
        assert isinstance(lower_interval, torch.Tensor)
        assert isinstance(upper_interval, torch.Tensor)
        assert lower_interval.shape == lower_pred.shape
        assert upper_interval.shape == upper_pred.shape


class TestConformalizedQuantileLoss:
    """Test ConformalizedQuantileLoss class."""

    def test_initialization(self):
        """Test initialization of ConformalizedQuantileLoss."""
        loss_fn = ConformalizedQuantileLoss(quantiles=(0.05, 0.95), alpha=0.1)
        assert loss_fn.quantiles == [0.05, 0.95]
        assert loss_fn.alpha == 0.1
        assert loss_fn.tau is None

    def test_forward_pass(self):
        """Test forward pass of ConformalizedQuantileLoss."""
        loss_fn = ConformalizedQuantileLoss(quantiles=(0.05, 0.95))

        # Create dummy predictions and targets
        batch_size, n_features = 10, 3
        y_pred = torch.randn(batch_size, 2 * n_features)  # 2 quantiles
        y_true = torch.randn(batch_size, n_features)

        # Forward pass
        loss = loss_fn(y_pred, y_true)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss

    def test_calibration(self):
        """Test calibration method."""
        loss_fn = ConformalizedQuantileLoss(quantiles=(0.05, 0.95))

        # Create dummy predictions and targets
        batch_size, n_features = 10, 3
        quantile_preds = torch.randn(batch_size, 2 * n_features)
        y_true = torch.randn(batch_size, n_features)

        # Calibrate
        tau = loss_fn.calibrate(quantile_preds, y_true)
        assert isinstance(tau, torch.Tensor)
        assert tau.dim() == 0  # Scalar tau

    def test_predict_interval(self):
        """Test predict_interval method."""
        loss_fn = ConformalizedQuantileLoss(quantiles=(0.05, 0.95))

        # Create dummy predictions and targets
        batch_size, n_features = 10, 3
        quantile_preds = torch.randn(batch_size, 2 * n_features)
        y_true = torch.randn(batch_size, n_features)

        # Calibrate first
        loss_fn.calibrate(quantile_preds, y_true)

        # Predict intervals
        lower_interval, upper_interval = loss_fn.predict_interval(quantile_preds)
        assert isinstance(lower_interval, torch.Tensor)
        assert isinstance(upper_interval, torch.Tensor)
        assert lower_interval.shape == (batch_size, n_features)
        assert upper_interval.shape == (batch_size, n_features)


class TestMultiDimensionalConformalLoss:
    """Test MultiDimensionalConformalLoss class."""

    def test_initialization(self):
        """Test initialization of MultiDimensionalConformalLoss."""
        loss_fn = MultiDimensionalConformalLoss(alpha=0.1)
        assert loss_fn.alpha == 0.1
        assert loss_fn.taus is None

    def test_forward_pass(self):
        """Test forward pass of MultiDimensionalConformalLoss."""
        loss_fn = MultiDimensionalConformalLoss(alpha=0.1)

        # Create dummy predictions and targets
        batch_size, n_features = 10, 3
        y_pred = torch.randn(batch_size, n_features)
        y_true = torch.randn(batch_size, n_features)

        # Forward pass
        loss = loss_fn(y_pred, y_true)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss

    def test_calibration(self):
        """Test calibration method."""
        loss_fn = MultiDimensionalConformalLoss(alpha=0.1)

        # Create dummy predictions and targets
        batch_size, n_features = 10, 3
        lower_preds = torch.randn(batch_size, n_features)
        upper_preds = torch.randn(batch_size, n_features)
        y_true = torch.randn(batch_size, n_features)

        # Calibrate
        taus = loss_fn.calibrate(lower_preds, upper_preds, y_true)
        assert isinstance(taus, torch.Tensor)
        assert taus.shape == (n_features,)

    def test_predict_intervals(self):
        """Test predict_intervals method."""
        loss_fn = MultiDimensionalConformalLoss(alpha=0.1)

        # Create dummy predictions and targets
        batch_size, n_features = 10, 3
        lower_preds = torch.randn(batch_size, n_features)
        upper_preds = torch.randn(batch_size, n_features)
        y_true = torch.randn(batch_size, n_features)

        # Calibrate first
        loss_fn.calibrate(lower_preds, upper_preds, y_true)

        # Predict intervals
        lower_intervals, upper_intervals = loss_fn.predict_intervals(lower_preds, upper_preds)
        assert isinstance(lower_intervals, torch.Tensor)
        assert isinstance(upper_intervals, torch.Tensor)
        assert lower_intervals.shape == lower_preds.shape
        assert upper_intervals.shape == upper_preds.shape


if __name__ == "__main__":
    pytest.main([__file__])
