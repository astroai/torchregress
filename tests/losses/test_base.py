import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchregression.losses.base import (
    BaseLoss,
    MaskedLoss,
    RegressionLoss,
    DistributionLoss,
    TorchLossWrapper,
)


# Helper implementation classes for testing abstract base classes
class SimpleLoss(BaseLoss):
    def forward(self, y_pred, target, **kwargs):
        loss = torch.abs(y_pred - target)
        return self._reduce(loss, **kwargs)


class SimpleMaskedLoss(MaskedLoss):
    def forward(self, y_pred, target, mask=None, weights=None, **kwargs):
        self._validate_inputs(y_pred, target, mask)
        loss = torch.abs(y_pred - target)
        return self._reduce_with_mask(loss, mask, weights)


class SimpleRegressionLoss(RegressionLoss):
    def forward(self, y_pred, target, mask=None, weights=None, **kwargs):
        self._validate_inputs(y_pred, target, mask)
        loss = torch.abs(y_pred - target)
        return self._reduce_with_mask(loss, mask, weights)


class SimpleDistributionLoss(DistributionLoss):
    def _extract_distribution_parameters(self, y_pred):
        # Assume y_pred is [mean, log_var]
        if y_pred.shape[-1] != 2:
            raise ValueError("Expected y_pred to have shape [..., 2]")
        return {"mean": y_pred[..., 0], "log_var": y_pred[..., 1]}

    def _calculate_nll(self, y_pred, target, mask=None):
        params = self._extract_distribution_parameters(y_pred)
        mean = params["mean"]
        log_var = params["log_var"]
        nll = 0.5 * (log_var + (target - mean) ** 2 / torch.exp(log_var))
        return nll

    def forward(self, y_pred, target, mask=None, weights=None, **kwargs):
        self._validate_inputs(y_pred[..., 0], target, mask)
        nll = self._calculate_nll(y_pred, target, mask)
        return self._reduce_with_mask(nll, mask, weights)


# Tests for BaseLoss
class TestBaseLoss:
    def test_init(self):
        # Test valid reductions
        loss_mean = SimpleLoss(reduction="mean")
        loss_sum = SimpleLoss(reduction="sum")
        loss_none = SimpleLoss(reduction="none")

        assert loss_mean.reduction == "mean"
        assert loss_sum.reduction == "sum"
        assert loss_none.reduction == "none"

        # Test invalid reduction
        with pytest.raises(ValueError):
            SimpleLoss(reduction="invalid")

    def test_reduce(self):
        # Create sample data
        loss_values = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # Test different reduction methods
        loss_mean = SimpleLoss(reduction="mean")
        loss_sum = SimpleLoss(reduction="sum")
        loss_none = SimpleLoss(reduction="none")

        # Call _reduce directly for testing
        reduced_mean = loss_mean._reduce(loss_values)
        reduced_sum = loss_sum._reduce(loss_values)
        reduced_none = loss_none._reduce(loss_values)

        assert torch.isclose(reduced_mean, torch.tensor(2.5))
        assert torch.isclose(reduced_sum, torch.tensor(10.0))
        assert torch.allclose(reduced_none, loss_values)

    def test_reduce_with_weights(self):
        # Create sample data
        loss_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        weights = torch.tensor([0.5, 1.0, 1.5, 0.0])

        # Test different reduction methods with weights
        loss_mean = SimpleLoss(reduction="mean")
        loss_sum = SimpleLoss(reduction="sum")
        loss_none = SimpleLoss(reduction="none")

        reduced_mean = loss_mean._reduce(loss_values, weights=weights)
        reduced_sum = loss_sum._reduce(loss_values, weights=weights)
        reduced_none = loss_none._reduce(loss_values, weights=weights)

        # Expected: (0.5*1.0 + 1.0*2.0 + 1.5*3.0 + 0.0*4.0) / weights.sum()
        expected_mean = (0.5 + 2.0 + 4.5) / 3.0
        expected_sum = 0.5 + 2.0 + 4.5
        expected_none = loss_values * weights

        assert torch.isclose(reduced_mean, torch.tensor(expected_mean))
        assert torch.isclose(reduced_sum, torch.tensor(expected_sum))
        assert torch.allclose(reduced_none, expected_none)

    def test_forward_not_implemented(self):
        # BaseLoss.forward raises NotImplementedError
        loss = BaseLoss()
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 1.5, 2.5])

        with pytest.raises(NotImplementedError):
            loss(y_pred, target)


# Tests for MaskedLoss
class TestMaskedLoss:
    def test_apply_mask(self):
        loss = SimpleMaskedLoss()
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])

        masked = loss._apply_mask(tensor, mask)
        # Masked values should be the tensor values where mask is True
        assert masked.shape == torch.tensor([2]).shape
        assert torch.allclose(masked, torch.tensor([1.0, 3.0]))

    def test_validate_inputs(self):
        loss = SimpleMaskedLoss()
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 1.5, 2.5])

        # Should not raise error for valid inputs
        loss._validate_inputs(y_pred, target)

        # Should not raise error for valid inputs with mask
        mask = torch.tensor([True, False, True])
        loss._validate_inputs(y_pred, target, mask)

        # Should raise error for mismatched shapes
        target_wrong_shape = torch.tensor([1.5, 1.5])
        with pytest.raises(ValueError):
            loss._validate_inputs(y_pred, target_wrong_shape)

        # Should raise error for mismatched mask shape
        mask_wrong_shape = torch.tensor([True, False])
        with pytest.raises(ValueError):
            loss._validate_inputs(y_pred, target, mask_wrong_shape)

    def test_reduce_with_mask(self):
        loss = SimpleMaskedLoss()
        loss_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, True, False, True])

        # With mean reduction
        loss.reduction = "mean"
        reduced_mean = loss._reduce_with_mask(loss_values, mask)
        # Expected: (1.0 + 2.0 + 4.0) / 3
        assert torch.isclose(reduced_mean, torch.tensor(7.0 / 3.0))

        # With sum reduction
        loss.reduction = "sum"
        reduced_sum = loss._reduce_with_mask(loss_values, mask)
        # Expected: 1.0 + 2.0 + 4.0
        assert torch.isclose(reduced_sum, torch.tensor(7.0))

        # With no reduction
        loss.reduction = "none"
        reduced_none = loss._reduce_with_mask(loss_values, mask)
        # Expected: [1.0, 2.0, 4.0]
        assert reduced_none.shape == torch.tensor([3]).shape
        assert torch.allclose(reduced_none, torch.tensor([1.0, 2.0, 4.0]))

    def test_forward_with_mask(self):
        # This tests the entire end-to-end flow
        loss = SimpleMaskedLoss(reduction="mean")
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([0.0, 0.0, 3.0, 0.0])
        mask = torch.tensor([False, True, True, False])

        # Only consider positions 1 and 2: |2-0| + |3-3| = 2
        result = loss(y_pred, target, mask=mask)
        assert torch.isclose(result, torch.tensor(1.0))

    def test_forward_with_weights_and_mask(self):
        loss = SimpleMaskedLoss(reduction="mean")
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([0.0, 0.0, 3.0, 0.0])
        mask = torch.tensor([False, True, True, False])
        weights = torch.tensor([0.5, 1.0, 1.5, 2.0])

        # Only consider positions 1 and 2 with weights: (1.0*|2-0| + 1.5*|3-3|) / (1.0+1.5)
        # = (1.0*2 + 1.5*0) / 2.5 = 2.0/2.5
        result = loss(y_pred, target, mask=mask, weights=weights)
        assert torch.isclose(result, torch.tensor(0.8))


# Tests for RegressionLoss
class TestRegressionLoss:
    def test_forward_not_implemented(self):
        # RegressionLoss.forward is abstract
        loss = RegressionLoss()
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 1.5, 2.5])

        with pytest.raises(NotImplementedError):
            loss(y_pred, target)

    def test_implementation(self):
        # Test a concrete implementation
        loss = SimpleRegressionLoss(reduction="mean")
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([0.0, 2.0, 4.0])

        # |1-0| + |2-2| + |3-4| / 3 = 1.0/3 + 0 + 1.0/3 = 2.0/3
        result = loss(y_pred, target)
        assert torch.isclose(result, torch.tensor(2.0 / 3.0))


# Tests for DistributionLoss
class TestDistributionLoss:
    def test_abstract_methods(self):
        loss = DistributionLoss()

        with pytest.raises(NotImplementedError):
            loss._extract_distribution_parameters(torch.tensor([1.0]))

        with pytest.raises(NotImplementedError):
            loss._calculate_nll(torch.tensor([1.0]), torch.tensor([1.0]))

        with pytest.raises(NotImplementedError):
            loss(torch.tensor([1.0]), torch.tensor([1.0]))

    def test_implementation(self):
        loss = SimpleDistributionLoss(reduction="mean")
        # Create a batch of [mean, log_var] predictions
        y_pred = torch.tensor(
            [
                [0.0, 0.0],  # mean=0, log_var=0 (var=1)
                [1.0, 0.0],  # mean=1, log_var=0 (var=1)
                [2.0, -1.0],  # mean=2, log_var=-1 (var=0.368)
            ]
        )
        target = torch.tensor([0.0, 0.0, 3.0])

        # Calculate expected NLL manually:
        # For item 0: 0.5 * (0 + (0-0)^2/e^0) = 0
        # For item 1: 0.5 * (0 + (0-1)^2/e^0) = 0.5
        # For item 2: 0.5 * (-1 + (3-2)^2/e^-1) = 0.5 * (-1 + 1*e^1) = 0.5 * (-1 + 2.718) = 0.859
        # Mean: (0 + 0.5 + 0.859) / 3 = 0.453

        # Calculate with our loss
        result = loss(y_pred, target)
        expected = torch.tensor(0.453)

        assert torch.isclose(result, expected, atol=1e-3)


# Tests for TorchLossWrapper
class TestTorchLossWrapper:
    def test_wrap_loss_class(self):
        # Test wrapping a loss class
        loss = TorchLossWrapper(nn.MSELoss, reduction="mean")
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([0.0, 2.0, 4.0])

        # MSE: ((1-0)^2 + (2-2)^2 + (3-4)^2) / 3 = (1 + 0 + 1) / 3 = 2/3
        result = loss(y_pred, target)
        assert torch.isclose(result, torch.tensor(2.0 / 3.0))

    def test_wrap_loss_instance(self):
        # Test wrapping a loss instance
        torch_loss = nn.L1Loss(reduction="none")
        loss = TorchLossWrapper(torch_loss, reduction="sum")
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([0.0, 2.0, 4.0])

        # L1: |1-0| + |2-2| + |3-4| = 1 + 0 + 1 = 2
        result = loss(y_pred, target)
        assert torch.isclose(result, torch.tensor(2.0))

    def test_wrap_with_mask(self):
        # Test using mask with wrapped loss
        loss = TorchLossWrapper(nn.MSELoss, reduction="mean")
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([0.0, 2.0, 0.0, 5.0])
        mask = torch.tensor([True, True, False, True])

        # MSE with mask: ((1-0)^2 + (2-2)^2 + (4-5)^2) / 3 = (1 + 0 + 1) / 3 = 2/3
        result = loss(y_pred, target, mask=mask)
        assert torch.isclose(result, torch.tensor(2.0 / 3.0))

    def test_wrap_with_weights(self):
        # Test using weights with wrapped loss
        loss = TorchLossWrapper(nn.MSELoss, reduction="mean")
        y_pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([0.0, 2.0, 4.0])
        weights = torch.tensor([0.5, 0.0, 1.5])

        # MSE with weights: (0.5*(1-0)^2 + 0.0*(2-2)^2 + 1.5*(3-4)^2) / 2.0
        # = (0.5*1 + 0 + 1.5*1) / 2.0 = 2.0/2.0 = 1.0
        result = loss(y_pred, target, weights=weights)
        assert torch.isclose(result, torch.tensor(1.0))


# Tests for edge cases and numerical stability
class TestEdgeCases:
    def test_zero_tensor(self):
        # Test with zero tensors
        loss = SimpleLoss(reduction="mean")
        y_pred = torch.zeros(3)
        target = torch.zeros(3)

        result = loss(y_pred, target)
        assert torch.isclose(result, torch.tensor(0.0))

    def test_nan_handling(self):
        # Test with NaN values (should propagate NaN)
        loss = SimpleMaskedLoss(reduction="mean")
        y_pred = torch.tensor([1.0, float("nan"), 3.0])
        target = torch.tensor([0.0, 2.0, 4.0])

        result = loss(y_pred, target)
        assert torch.isnan(result)

        # Test with mask to exclude NaN
        mask = torch.tensor([True, False, True])
        result = loss(y_pred, target, mask=mask)
        assert not torch.isnan(result)

    def test_inf_handling(self):
        # Test with Inf values
        loss = SimpleMaskedLoss(reduction="mean")
        y_pred = torch.tensor([1.0, float("inf"), 3.0])
        target = torch.tensor([0.0, 2.0, 4.0])

        result = loss(y_pred, target)
        assert torch.isinf(result)

        # Test with mask to exclude Inf
        mask = torch.tensor([True, False, True])
        result = loss(y_pred, target, mask=mask)
        assert not torch.isinf(result)


# Tests for gradient flow
class TestGradientFlow:
    def test_gradient_flow_basic(self):
        # Test gradient flows properly
        loss_fn = SimpleLoss(reduction="mean")
        y_pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 2.0, 4.0])

        loss_value = loss_fn(y_pred, target)
        loss_value.backward()

        # Check that gradients were computed
        assert y_pred.grad is not None
        # For L1 loss, gradients should be sign of (y_pred - target)
        expected_grad = (
            torch.tensor([1.0, 0.0, -1.0]) / 3.0
        )  # divided by 3 because of mean reduction
        assert torch.allclose(y_pred.grad, expected_grad)

    def test_gradient_flow_masked(self):
        # Test gradient flows properly with mask
        loss_fn = SimpleMaskedLoss(reduction="mean")
        y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        target = torch.tensor([0.0, 2.0, 0.0, 5.0])
        mask = torch.tensor([True, True, False, True])

        loss_value = loss_fn(y_pred, target, mask=mask)
        loss_value.backward()

        # Masked gradients - only positions 0, 1, 3 contribute
        expected_grad = torch.zeros_like(y_pred)
        expected_grad[0] = 1.0 / 3.0  # sign(1-0)/3
        expected_grad[1] = 0.0  # sign(2-2)/3
        expected_grad[3] = -1.0 / 3.0  # sign(4-5)/3

        assert torch.allclose(y_pred.grad, expected_grad)

    def test_gradient_flow_distribution(self):
        # Test gradient flows properly in distribution loss
        loss_fn = SimpleDistributionLoss(reduction="mean")
        y_pred = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0]], requires_grad=True  # mean=0, log_var=0  # mean=1, log_var=0
        )
        target = torch.tensor([0.0, 0.0])

        loss_value = loss_fn(y_pred, target)
        loss_value.backward()

        # Check that gradients were computed
        assert y_pred.grad is not None
        # Gradient shape should match y_pred
        assert y_pred.grad.shape == y_pred.shape
