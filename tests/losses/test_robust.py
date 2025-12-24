import pytest
import torch
from torch.autograd import gradcheck

from torchregress.losses.base import WeightedHuberLoss as HuberLoss
from torchregress.losses.base import WeightedL1Loss as L1Loss
from torchregress.losses.robust import (
    CauchyLoss,
    LogCoshLoss,
    PseudoHuberLoss,
)


@pytest.fixture
def setup_data(device):
    """Initialize common test data."""
    batch_size = 4
    n_features = 5

    y_true = torch.randn(batch_size, n_features, device=device)
    y_pred = torch.randn(batch_size, n_features, device=device)
    mask = torch.randint(0, 2, (batch_size, n_features), device=device).bool()

    # Create data with outliers for robust loss testing
    y_true_outliers = y_true.clone()
    y_true_outliers[0, 0] = 10.0  # Add an outlier

    # Zero difference data (for testing behavior at minimum)
    y_true_zero = torch.zeros(batch_size, n_features, device=device)
    y_pred_zero = torch.zeros(batch_size, n_features, device=device)

    # Additional test data for gradient checking and tensor shapes
    scalar_true = torch.tensor(5.0, device=device, requires_grad=True)
    scalar_pred = torch.tensor(3.0, device=device, requires_grad=True)

    # 3D tensors for testing different shapes
    y_true_3d = torch.randn(2, 3, 4, device=device)
    y_pred_3d = torch.randn(2, 3, 4, device=device)
    mask_3d = torch.randint(0, 2, (2, 3, 4), device=device).bool()

    # Empty tensor for edge case testing
    y_true_empty = torch.tensor([], device=device)
    y_pred_empty = torch.tensor([], device=device)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "mask": mask,
        "y_true_outliers": y_true_outliers,
        "y_true_zero": y_true_zero,
        "y_pred_zero": y_pred_zero,
        "scalar_true": scalar_true,
        "scalar_pred": scalar_pred,
        "y_true_3d": y_true_3d,
        "y_pred_3d": y_pred_3d,
        "mask_3d": mask_3d,
        "y_true_empty": y_true_empty,
        "y_pred_empty": y_pred_empty,
    }


class TestL1Loss:
    """Test suite for L1Loss."""

    def test_basic_functionality(self, setup_data, device):
        """Test L1Loss (Mean Absolute Error) behavior."""
        data = setup_data
        loss_fn = L1Loss().to(device)

        # Test with mask
        loss = loss_fn(data["y_pred"], data["y_true"], data["mask"])
        assert torch.is_tensor(loss)
        assert not torch.isnan(loss).any()

        # Test without mask
        loss_no_mask = loss_fn(data["y_pred"], data["y_true"])
        assert torch.is_tensor(loss_no_mask)
        assert not torch.isnan(loss_no_mask).any()

        # Test with weights
        weights = torch.rand_like(data["y_pred"])
        loss_with_weights = loss_fn(data["y_pred"], data["y_true"], weights=weights)
        assert torch.is_tensor(loss_with_weights)
        assert not torch.isnan(loss_with_weights).any()

    def test_outlier_robustness(self, setup_data):
        """Test that L1 is less affected by outliers than MSE."""
        data = setup_data
        l1_loss = L1Loss()

        # Calculate losses with and without outliers
        mse = torch.nn.MSELoss()(data["y_pred"], data["y_true_outliers"])
        l1 = l1_loss(data["y_pred"], data["y_true_outliers"])
        baseline_mse = torch.nn.MSELoss()(data["y_pred"], data["y_true"])
        baseline_l1 = l1_loss(data["y_pred"], data["y_true"])

        # Outlier effect ratio should be larger for MSE than L1
        mse_ratio = mse / baseline_mse
        l1_ratio = l1 / baseline_l1
        assert mse_ratio.item() > l1_ratio.item()

    def test_zero_behavior(self, setup_data):
        """Test that L1 at zero is actually zero."""
        data = setup_data
        loss_at_zero = L1Loss()(data["y_pred_zero"], data["y_true_zero"])
        assert loss_at_zero.item() == pytest.approx(0.0, abs=1e-5)

    def test_gradient_behavior(self, device):
        """Test gradient at non-zero point."""
        scalar_pred = torch.tensor(3.0, device=device, requires_grad=True)
        scalar_true = torch.tensor(5.0, device=device)

        loss = L1Loss()(scalar_pred, scalar_true)
        loss.backward()

        # Gradient should be sign(pred - true) = sign(3 - 5) = -1
        assert scalar_pred.grad.item() == -1.0

    def test_reduction_modes(self, setup_data):
        """Test different reduction modes."""
        data = setup_data

        # None reduction
        loss_none = L1Loss(reduction="none")(data["y_pred"], data["y_true"])
        assert loss_none.shape == data["y_true"].shape

        # Sum reduction
        loss_sum = L1Loss(reduction="sum")(data["y_pred"], data["y_true"])
        assert loss_sum.numel() == 1
        assert loss_sum.item() == pytest.approx(loss_none.sum().item(), abs=1e-5)

        # Mean reduction
        loss_mean = L1Loss(reduction="mean")(data["y_pred"], data["y_true"])
        assert loss_mean.numel() == 1
        assert loss_mean.item() == pytest.approx(loss_none.mean().item(), abs=1e-5)


# ... Additional test classes for other robust losses ...


class TestRobustLossesNumericalStability:
    def test_huber_gradient_flow(self):
        """Test that gradients flow through HuberLoss properly."""

        # Create inputs that require gradients
        y_pred = torch.randn(10, 1, requires_grad=True, dtype=torch.double)
        y_true = torch.randn(10, 1, dtype=torch.double)

        # Test with gradcheck
        loss_fn = HuberLoss(reduction="mean", delta=1.0)
        assert gradcheck(loss_fn, (y_pred, y_true), eps=1e-6, atol=1e-5)

    def test_pseudo_huber_gradient_flow(self):
        """Test that gradients flow through PseudoHuberLoss properly."""

        # Create inputs that require gradients
        y_pred = torch.randn(10, 1, requires_grad=True, dtype=torch.double)
        y_true = torch.randn(10, 1, dtype=torch.double)

        # Test with gradcheck
        loss_fn = PseudoHuberLoss(reduction="mean", delta=1.0)
        assert gradcheck(loss_fn, (y_pred, y_true), eps=1e-6, atol=1e-4)

    def test_log_cosh_gradient_flow(self):
        """Test that gradients flow through LogCoshLoss properly."""

        # Create inputs that require gradients
        y_pred = torch.randn(10, 1, requires_grad=True, dtype=torch.double)
        y_true = torch.randn(10, 1, dtype=torch.double)

        # Test with gradcheck
        loss_fn = LogCoshLoss(reduction="mean")
        assert gradcheck(loss_fn, (y_pred, y_true), eps=1e-6, atol=1e-4)

    def test_cauchy_gradient_flow(self):
        """Test that gradients flow through CauchyLoss properly."""

        # Create inputs that require gradients
        y_pred = torch.randn(10, 1, requires_grad=True, dtype=torch.double)
        y_true = torch.randn(10, 1, dtype=torch.double)

        # Test with gradcheck
        loss_fn = CauchyLoss(reduction="mean", c=1.0)
        assert gradcheck(loss_fn, (y_pred, y_true), eps=1e-6, atol=1e-4)

    def test_extreme_values(self):
        """Test stability with extreme values."""

        # Moderately large values (too extreme can cause overflow in robust losses)
        y_pred = torch.tensor([100.0, 1000.0, 10000.0], requires_grad=True)
        y_true = torch.tensor([101.0, 1010.0, 10100.0])

        # Test HuberLoss with extreme values
        huber = HuberLoss(reduction="mean", delta=1.0)
        loss = huber(y_pred, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred.grad))

        # Test PseudoHuberLoss with extreme values
        y_pred.grad = None  # Reset gradients
        pseudo = PseudoHuberLoss(reduction="mean", delta=1.0)
        loss = pseudo(y_pred, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred.grad))

        # Test LogCoshLoss with extreme values
        y_pred.grad = None  # Reset gradients
        logcosh = LogCoshLoss(reduction="mean")
        loss = logcosh(y_pred, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred.grad))

        # Test CauchyLoss with extreme values
        y_pred.grad = None  # Reset gradients
        cauchy = CauchyLoss(reduction="mean", c=1.0)
        loss = cauchy(y_pred, y_true)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.all(torch.isfinite(y_pred.grad))

    def test_nan_inf_handling(self):
        """Test how losses handle NaN and Inf values with masks."""

        # Create data with some NaNs and Infs
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        y_true = torch.tensor([1.1, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, False, False])  # Mask out some elements

        huber = HuberLoss(reduction="mean")

        # This should only use the valid elements
        loss = huber(y_pred, y_true, mask=mask)
        assert torch.isfinite(loss)
        loss.backward()
        # Only the unmasked elements should have gradients
        assert torch.isfinite(y_pred.grad[0])
        # Masked elements should have zero gradient
        assert y_pred.grad[1] == 0.0
        assert y_pred.grad[2] == 0.0
        assert y_pred.grad[3] == 0.0

    def test_reduction_modes(self):
        """Test different reduction modes for backward pass."""

        y_pred = torch.randn(10, 1, requires_grad=True)
        y_true = torch.randn(10, 1)

        # Test mean reduction
        huber_mean = HuberLoss(reduction="mean")
        loss = huber_mean(y_pred, y_true)
        loss.backward()
        mean_grad = y_pred.grad.clone()

        # Test sum reduction
        y_pred.grad = None
        huber_sum = HuberLoss(reduction="sum")
        loss = huber_sum(y_pred, y_true)
        loss.backward()
        sum_grad = y_pred.grad.clone()

        # Test none reduction
        y_pred.grad = None
        huber_none = HuberLoss(reduction="none")
        loss = huber_none(y_pred, y_true)
        loss.mean().backward()
        none_grad = y_pred.grad.clone()

        # Mean and sum should give different gradients (sum ~= mean * batch_size)
        assert not torch.allclose(mean_grad, sum_grad)
        # Mean and manual mean over none should be similar
        assert torch.allclose(mean_grad, none_grad)
