import pytest
import torch
from torch.autograd import gradcheck
from torchregress.losses.quantile import QuantileLoss, MultiQuantileLoss, QuantileCrossoverLoss


class TestQuantileLoss:
    """Test suite for QuantileLoss."""

    @pytest.fixture
    def quantile_loss(self):
        return QuantileLoss(quantile=0.5)

    @pytest.fixture
    def sample_data(self, device):
        y_pred = torch.randn(10, 1, device=device)
        y_true = torch.randn(10, 1, device=device)
        return y_pred, y_true

    def test_basic_functionality(self, quantile_loss, sample_data):
        """Test basic loss calculation."""
        y_pred, y_true = sample_data
        loss = quantile_loss(y_pred, y_true)

        # Loss should be a scalar tensor
        assert torch.is_tensor(loss)
        assert loss.numel() == 1
        assert not torch.isnan(loss)

    def test_different_quantiles(self, sample_data):
        """Test with different quantile values."""
        y_pred, y_true = sample_data

        # Test with different quantile values
        q_values = [0.1, 0.5, 0.9]
        for q in q_values:
            loss_fn = QuantileLoss(quantile=q)
            loss = loss_fn(y_pred, y_true)
            assert not torch.isnan(loss)

    def test_with_mask(self, quantile_loss, sample_data, sample_mask):
        """Test masked loss calculation."""
        y_pred, y_true = sample_data

        # Calculate masked loss
        loss = quantile_loss(y_pred, y_true, mask=sample_mask)

        # Loss should be a scalar tensor
        assert torch.is_tensor(loss)
        assert not torch.isnan(loss)

    def test_with_weights(self, quantile_loss, sample_data, sample_weights):
        """Test weighted loss calculation."""
        y_pred, y_true = sample_data

        # Calculate weighted loss
        loss = quantile_loss(y_pred, y_true, weights=sample_weights)

        # Loss should be a scalar tensor
        assert torch.is_tensor(loss)
        assert not torch.isnan(loss)

    def test_quantile_crossover_constraint(self):
        """Test that QuantileCrossoverLoss enforces the quantile ordering constraint."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        y_true = torch.randn(10, 1, device=device)

        # Create model outputs that violate the quantile ordering constraint
        lower_quantile = torch.randn(10, 1, device=device)
        upper_quantile = lower_quantile - 1.0  # Upper should be higher, but we're setting it lower

        # Combined predictions
        y_pred = torch.cat([lower_quantile, upper_quantile], dim=1)

        # Create loss with crossover constraint
        loss_fn = QuantileCrossoverLoss(quantiles=[0.1, 0.9])

        # Calculate loss
        loss = loss_fn(y_pred, y_true)

        # Loss should include a penalty
        assert loss > 0

    @pytest.mark.parametrize("quantile", [0.1, 0.5, 0.9])
    def test_gradient_flow(self, quantile, device):
        """Test that gradients flow correctly."""
        loss_fn = QuantileLoss(quantile=quantile)

        # Create tensor requiring gradients
        y_pred = torch.randn(5, 1, requires_grad=True, dtype=torch.float64, device=device)
        y_true = torch.randn(5, 1, dtype=torch.float64, device=device)

        # Check gradients
        assert gradcheck(loss_fn, (y_pred, y_true), eps=1e-6, atol=1e-5)


class TestMultiQuantileLoss:
    """Test suite for MultiQuantileLoss."""

    @pytest.fixture
    def multi_quantile_loss(self):
        return MultiQuantileLoss(quantiles=[0.1, 0.5, 0.9])

    def test_basic_functionality(self, multi_quantile_loss, device):
        """Test basic multi-quantile loss calculation."""
        # For MultiQuantileLoss, y_pred should have shape [batch_size, n_quantiles]
        batch_size = 10
        n_quantiles = 3

        y_pred = torch.randn(batch_size, n_quantiles, device=device)
        y_true = torch.randn(batch_size, 1, device=device)

        loss = multi_quantile_loss(y_pred, y_true)

        # Loss should be a scalar tensor
        assert torch.is_tensor(loss)
        assert loss.numel() == 1
        assert not torch.isnan(loss)

    def test_with_weights(self, multi_quantile_loss, device):
        """Test with quantile weights."""
        batch_size = 10
        n_quantiles = 3

        y_pred = torch.randn(batch_size, n_quantiles, device=device)
        y_true = torch.randn(batch_size, 1, device=device)

        # Set different weights for different quantiles
        quantile_weights = torch.tensor([0.2, 0.5, 0.3], device=device)
        loss_fn = MultiQuantileLoss(quantiles=[0.1, 0.5, 0.9], quantile_weights=quantile_weights)

        loss = loss_fn(y_pred, y_true)
        assert not torch.isnan(loss)


class TestQuantileLossNumericalStability:
    """Tests for numerical stability of quantile losses."""

    def test_extreme_values(self, device):
        """Test with extreme prediction and target values."""
        loss_fn = QuantileLoss(quantile=0.5)

        # Very large values
        y_pred_large = torch.tensor([1e10], device=device)
        y_true_large = torch.tensor([1e10 + 1], device=device)

        loss_large = loss_fn(y_pred_large, y_true_large)
        assert torch.isfinite(loss_large)

        # Very small differences
        y_pred_small = torch.tensor([1e-10], device=device)
        y_true_small = torch.tensor([2e-10], device=device)

        loss_small = loss_fn(y_pred_small, y_true_small)
        assert torch.isfinite(loss_small)

    def test_nan_inf_handling(self, device):
        """Test handling of NaN and Inf values with masks."""
        loss_fn = QuantileLoss(quantile=0.5)

        # Create data with NaNs and Infs
        y_pred = torch.tensor([1.0, float("nan"), 3.0, float("inf")], device=device)
        y_true = torch.tensor([1.5, 2.0, float("nan"), 4.0], device=device)
        mask = torch.tensor([True, False, False, True], device=device)

        # This should only use the masked (valid) elements
        loss = loss_fn(y_pred, y_true, mask=mask)
        assert torch.isfinite(loss)
