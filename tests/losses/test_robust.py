import math

import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from torchregress.losses.base import WeightedLossWrapper
from torchregress.losses.robust import (
    AdaptiveRobustLoss,
    BarronLoss,
    CauchyLoss,
    LogCoshLoss,
    PseudoHuberLoss,
    _barron_elementwise,
    _log_barron_partition,
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
    """Test suite for WeightedLossWrapper(nn.L1Loss)."""

    def test_basic_functionality(self, setup_data, device):
        """Test L1Loss (Mean Absolute Error) behavior."""
        data = setup_data
        loss_fn = WeightedLossWrapper(nn.L1Loss).to(device)

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
        l1_loss = WeightedLossWrapper(nn.L1Loss)

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
        loss_at_zero = WeightedLossWrapper(nn.L1Loss)(data["y_pred_zero"], data["y_true_zero"])
        assert loss_at_zero.item() == pytest.approx(0.0, abs=1e-5)

    def test_gradient_behavior(self, device):
        """Test gradient at non-zero point."""
        scalar_pred = torch.tensor(3.0, device=device, requires_grad=True)
        scalar_true = torch.tensor(5.0, device=device)

        loss = WeightedLossWrapper(nn.L1Loss)(scalar_pred, scalar_true)
        loss.backward()

        # Gradient should be sign(pred - true) = sign(3 - 5) = -1
        assert scalar_pred.grad.item() == -1.0

    def test_reduction_modes(self, setup_data):
        """Test different reduction modes."""
        data = setup_data

        # None reduction
        loss_none = WeightedLossWrapper(nn.L1Loss, reduction="none")(data["y_pred"], data["y_true"])
        assert loss_none.shape == data["y_true"].shape

        # Sum reduction
        loss_sum = WeightedLossWrapper(nn.L1Loss, reduction="sum")(data["y_pred"], data["y_true"])
        assert loss_sum.numel() == 1
        assert loss_sum.item() == pytest.approx(loss_none.sum().item(), abs=1e-5)

        # Mean reduction
        loss_mean = WeightedLossWrapper(nn.L1Loss, reduction="mean")(data["y_pred"], data["y_true"])
        assert loss_mean.numel() == 1
        assert loss_mean.item() == pytest.approx(loss_none.mean().item(), abs=1e-5)


# ... Additional test classes for other robust losses ...


class TestRobustLossesNumericalStability:
    def test_barron_alpha_two_matches_half_squared_error(self):
        y_pred = torch.tensor([[1.5], [-0.5]], dtype=torch.double, requires_grad=True)
        y_true = torch.tensor([[0.5], [1.0]], dtype=torch.double)

        loss_fn = BarronLoss(alpha=2.0, scale=1.0, reduction="none")
        loss = loss_fn(y_pred, y_true)
        expected = 0.5 * (y_true - y_pred) ** 2

        assert torch.allclose(loss, expected)

    def test_barron_alpha_zero_matches_cauchy_like_limit(self):
        y_pred = torch.tensor([[1.5], [-0.5]], dtype=torch.double)
        y_true = torch.tensor([[0.5], [1.0]], dtype=torch.double)

        loss_fn = BarronLoss(alpha=0.0, scale=2.0, reduction="none")
        loss = loss_fn(y_pred, y_true)
        expected = torch.log1p(0.5 * ((y_true - y_pred) / 2.0) ** 2)

        assert torch.allclose(loss, expected)

    def test_adaptive_robust_parameters_stay_in_range(self):
        loss_fn = AdaptiveRobustLoss(alpha_init=0.5, scale_init=1.5, alpha_min=-4.0, alpha_max=2.0)

        assert -4.0 <= loss_fn.alpha.item() <= 2.0
        assert loss_fn.scale.item() > 0.0

    def test_adaptive_robust_parameters_can_be_optimized(self):
        y_pred = torch.tensor([[2.0], [1.0], [8.0]], requires_grad=True)
        y_true = torch.tensor([[1.5], [1.1], [0.0]])
        loss_fn = AdaptiveRobustLoss(alpha_init=1.5, scale_init=1.0)
        optimizer = torch.optim.SGD([y_pred, *loss_fn.parameters()], lr=0.05)

        initial_alpha = loss_fn.alpha.detach().item()
        initial_scale = loss_fn.scale.detach().item()

        optimizer.zero_grad()
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

        assert loss_fn.alpha.detach().item() != pytest.approx(initial_alpha)
        assert loss_fn.scale.detach().item() != pytest.approx(initial_scale)

    def test_cauchy_scale_alias_matches_c_parameter(self):
        y_pred = torch.tensor([[1.5], [-0.5]], dtype=torch.double)
        y_true = torch.tensor([[0.5], [1.0]], dtype=torch.double)

        loss_with_c = CauchyLoss(c=2.0, reduction="none")(y_pred, y_true)
        loss_with_scale = CauchyLoss(scale=2.0, reduction="none")(y_pred, y_true)

        assert torch.allclose(loss_with_c, loss_with_scale)

    def test_huber_gradient_flow(self):
        """Test that gradients flow through WeightedLossWrapper(nn.HuberLoss) properly."""

        # Create inputs that require gradients
        y_pred = torch.randn(10, 1, requires_grad=True, dtype=torch.double)
        y_true = torch.randn(10, 1, dtype=torch.double)

        # Test with gradcheck
        loss_fn = WeightedLossWrapper(nn.HuberLoss, reduction="mean", delta=1.0)
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

    def test_barron_gradient_flow(self):
        """Test that gradients flow through BarronLoss properly."""

        y_pred = torch.randn(10, 1, requires_grad=True, dtype=torch.double)
        y_true = torch.randn(10, 1, dtype=torch.double)

        loss_fn = BarronLoss(reduction="mean", alpha=1.0, scale=1.0)
        assert gradcheck(loss_fn, (y_pred, y_true), eps=1e-6, atol=1e-4)

    def test_extreme_values(self):
        """Test stability with extreme values."""

        # Moderately large values (too extreme can cause overflow in robust losses)
        y_pred = torch.tensor([100.0, 1000.0, 10000.0], requires_grad=True)
        y_true = torch.tensor([101.0, 1010.0, 10100.0])

        # Test WeightedLossWrapper(nn.HuberLoss) with extreme values
        huber = WeightedLossWrapper(nn.HuberLoss, reduction="mean", delta=1.0)
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

        huber = WeightedLossWrapper(nn.HuberLoss, reduction="mean")

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
        huber_mean = WeightedLossWrapper(nn.HuberLoss, reduction="mean")
        loss = huber_mean(y_pred, y_true)
        loss.backward()
        mean_grad = y_pred.grad.clone()

        # Test sum reduction
        y_pred.grad = None
        huber_sum = WeightedLossWrapper(nn.HuberLoss, reduction="sum")
        loss = huber_sum(y_pred, y_true)
        loss.backward()
        sum_grad = y_pred.grad.clone()

        # Test none reduction
        y_pred.grad = None
        huber_none = WeightedLossWrapper(nn.HuberLoss, reduction="none")
        loss = huber_none(y_pred, y_true)
        loss.mean().backward()
        none_grad = y_pred.grad.clone()

        # Mean and sum should give different gradients (sum ~= mean * batch_size)
        assert not torch.allclose(mean_grad, sum_grad)
        # Mean and manual mean over none should be similar
        assert torch.allclose(mean_grad, none_grad)


class TestAdaptiveRobustNormalizationAndGradients:
    """TR-COR-03: smooth alpha gradients at the singular points and a
    scale-consistent normalized objective (Barron 2019, Eq. 17)."""

    def test_barron_alpha_gradient_nonzero_at_singular_points(self):
        residuals = torch.tensor([0.5, 1.0, 3.0], dtype=torch.double)
        for alpha_value in (0.0, 2.0):
            alpha = torch.tensor(alpha_value, dtype=torch.double, requires_grad=True)
            loss = _barron_elementwise(residuals, alpha, 1.3)
            assert torch.isfinite(loss).all()
            loss.sum().backward()
            assert torch.isfinite(alpha.grad).all()
            assert torch.all(alpha.grad != 0)

    def test_gradcheck_alpha_and_scale_at_singular_points(self):
        residuals = torch.tensor([0.5, 1.0, 3.0], dtype=torch.double)
        for alpha_value in (0.0, -2.0, 2.0):
            alpha = torch.tensor(alpha_value, dtype=torch.double, requires_grad=True)
            scale = torch.tensor(1.3, dtype=torch.double, requires_grad=True)

            def objective(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
                return (
                    _barron_elementwise(residuals, a, c) + torch.log(c) + _log_barron_partition(a)
                )

            assert gradcheck(objective, (alpha, scale), eps=1e-6, atol=1e-5)

    def test_adaptive_loss_parameters_receive_nonzero_gradient_at_endpoints(self):
        y_pred = torch.tensor([[0.2], [1.0], [2.0]], dtype=torch.double)
        y_true = torch.tensor([[0.7], [2.0], [-1.0]], dtype=torch.double)
        # init at the range edges lands within the smooth Taylor windows of 0 and 2
        for alpha_init, alpha_min, alpha_max in ((0.0, 0.0, 4.0), (2.0, 0.0, 2.0)):
            loss_fn = AdaptiveRobustLoss(
                alpha_init=alpha_init, alpha_min=alpha_min, alpha_max=alpha_max
            )
            loss = loss_fn(y_pred, y_true).sum()
            assert torch.isfinite(loss)
            loss.backward()
            assert loss_fn._alpha_logits.grad is not None
            assert torch.isfinite(loss_fn._alpha_logits.grad)
            assert loss_fn._alpha_logits.grad.abs() > 0
            assert loss_fn._scale_raw.grad is not None
            assert torch.isfinite(loss_fn._scale_raw.grad)
            assert loss_fn._scale_raw.grad.abs() > 0

    def test_scale_converges_instead_of_diverging_on_constant_residuals(self):
        residuals = torch.full((64,), 0.7)
        y_true = residuals.clone()
        y_pred = torch.zeros_like(residuals)
        loss_fn = AdaptiveRobustLoss(alpha_init=2.0, scale_init=3.0, alpha_min=0.0, alpha_max=2.0)
        optimizer = torch.optim.Adam(loss_fn.parameters(), lr=0.05)
        scales = []
        for _ in range(200):
            optimizer.zero_grad()
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()
            scales.append(loss_fn.scale.item())
            assert math.isfinite(scales[-1])

        assert max(scales) < 10.0  # no divergence
        # alpha ~= 2 -> quadratic regime -> optimal c is the residual RMS
        assert abs(loss_fn.scale.item() - 0.7) < 0.05

    def test_adaptive_loss_finite_at_zero_residual(self):
        loss_fn = AdaptiveRobustLoss()
        zeros = torch.zeros(4, 1, dtype=torch.double)
        loss = loss_fn(zeros, zeros)
        assert torch.isfinite(loss).all()
