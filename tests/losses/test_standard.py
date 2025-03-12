import pytest
import torch
import numpy as np

from torchregression.losses.standard import (
    MSELoss,
    BCELoss,
    MaskedMSELoss,
    MaskedL1Loss,
    MaskedBCELoss,
    MaskedBCEWithLogitsLoss,
    MaskedCrossEntropyLoss,
    MaskedKLDivLoss,
    MaskedNLLLoss,
    MaskedSmoothL1Loss,
    MaskedHuberLoss,
    MaskedPoissonNLLLoss,
)


class TestMSELoss:
    def test_basic_functionality(self):
        """Test basic MSE loss calculation."""
        loss_fn = MSELoss()
        y_pred = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
        target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        loss = loss_fn(y_pred, target)
        expected = torch.mean((y_pred - target) ** 2)

        assert torch.isclose(loss, expected)

    def test_with_mask(self):
        """Test MSE loss with masking."""
        loss_fn = MSELoss()
        y_pred = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32)
        target = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        mask = torch.tensor([True, True, False, True], dtype=torch.bool)

        loss = loss_fn(y_pred, target, mask=mask)

        # Only consider elements where mask is True
        expected = torch.mean((torch.tensor([0.5, 1.5, 3.5]) - torch.tensor([1.0, 2.0, 4.0])) ** 2)
        assert torch.isclose(loss, expected)

    def test_with_weights(self):
        """Test MSE loss with sample weights."""
        loss_fn = MSELoss()
        y_pred = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
        target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        weights = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)

        loss = loss_fn(y_pred, target, weights=weights)

        # Calculate weighted MSE manually
        expected = torch.sum(weights * (y_pred - target) ** 2) / torch.sum(weights)
        assert torch.isclose(loss, expected)

    def test_with_mask_and_weights(self):
        """Test MSE loss with both masking and weights."""
        loss_fn = MSELoss()
        y_pred = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32)
        target = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        mask = torch.tensor([True, True, False, True], dtype=torch.bool)
        weights = torch.tensor([0.5, 1.0, 2.0, 1.5], dtype=torch.float32)

        loss = loss_fn(y_pred, target, mask=mask, weights=weights)

        # Filter based on mask first
        y_pred_masked = y_pred[mask]
        target_masked = target[mask]
        weights_masked = weights[mask]

        expected = torch.sum(weights_masked * (y_pred_masked - target_masked) ** 2) / torch.sum(
            weights_masked
        )
        assert torch.isclose(loss, expected)

    def test_different_reductions(self):
        """Test different reduction methods."""
        # Test 'sum' reduction
        loss_fn_sum = MSELoss(reduction="sum")
        y_pred = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
        target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        loss_sum = loss_fn_sum(y_pred, target)
        expected_sum = torch.sum((y_pred - target) ** 2)
        assert torch.isclose(loss_sum, expected_sum)

        # Test 'none' reduction
        loss_fn_none = MSELoss(reduction="none")
        loss_none = loss_fn_none(y_pred, target)
        expected_none = (y_pred - target) ** 2
        assert torch.allclose(loss_none, expected_none)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through the loss."""
        loss_fn = MSELoss()
        y_pred = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32, requires_grad=True)
        target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        loss = loss_fn(y_pred, target)
        loss.backward()

        # For MSE, the gradient is proportional to (y_pred - target)
        expected_grad = 2 * (y_pred - target) / y_pred.numel()
        assert torch.allclose(y_pred.grad, expected_grad)

    def test_numeric_edge_cases(self):
        """Test behavior with edge case inputs."""
        loss_fn = MSELoss()

        # Test with zeros
        y_pred = torch.zeros(3, dtype=torch.float32)
        target = torch.zeros(3, dtype=torch.float32)
        loss = loss_fn(y_pred, target)
        assert loss.item() == 0.0

        # Test with large values
        y_pred = torch.tensor([1e5, 1e5, 1e5], dtype=torch.float32)
        target = torch.tensor([1e5 + 1, 1e5 - 1, 1e5], dtype=torch.float32)
        loss = loss_fn(y_pred, target)
        assert torch.isfinite(loss)  # Loss should be finite

        # Test with very small differences
        y_pred = torch.tensor([1e-8, 1e-8, 1e-8], dtype=torch.float32)
        target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        loss = loss_fn(y_pred, target)
        assert torch.isfinite(loss)  # Loss should be finite


class TestBCELoss:
    def test_basic_functionality(self):
        """Test basic BCE loss calculation."""
        loss_fn = BCELoss()
        y_pred = torch.tensor([0.2, 0.7, 0.9], dtype=torch.float32)
        target = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)

        loss = loss_fn(y_pred, target)

        # Calculate expected BCE loss manually
        expected = -torch.mean(target * torch.log(y_pred) + (1 - target) * torch.log(1 - y_pred))
        assert torch.isclose(loss, expected)

    def test_with_mask(self):
        """Test BCE loss with masking."""
        loss_fn = BCELoss()
        y_pred = torch.tensor([0.2, 0.7, 0.9, 0.3], dtype=torch.float32)
        target = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        mask = torch.tensor([True, True, False, True], dtype=torch.bool)

        loss = loss_fn(y_pred, target, mask=mask)

        # Only consider elements where mask is True
        y_pred_masked = torch.tensor([0.2, 0.7, 0.3])
        target_masked = torch.tensor([0.0, 1.0, 0.0])
        expected = -torch.mean(
            target_masked * torch.log(y_pred_masked)
            + (1 - target_masked) * torch.log(1 - y_pred_masked)
        )
        assert torch.isclose(loss, expected)

    def test_with_pos_weight(self):
        """Test BCE loss with positive class weighting."""
        loss_fn = BCELoss(pos_weight=2.0)
        y_pred = torch.tensor([0.2, 0.7, 0.9], dtype=torch.float32)
        target = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)

        loss = loss_fn(y_pred, target)

        # Calculate expected weighted BCE loss manually
        pos_weight = torch.tensor(2.0)
        expected = -torch.mean(
            pos_weight * target * torch.log(y_pred) + (1 - target) * torch.log(1 - y_pred)
        )
        assert torch.isclose(loss, expected)

    def test_numeric_stability(self):
        """Test numeric stability with values very close to 0 or 1."""
        loss_fn = BCELoss()

        # Test with predictions very close to 0 where target is 0
        y_pred = torch.tensor([1e-10, 0.7, 0.9], dtype=torch.float32)
        target = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)
        loss = loss_fn(y_pred, target)
        assert torch.isfinite(loss)  # Loss should be finite

        # Test with predictions very close to 1 where target is 1
        y_pred = torch.tensor([0.2, 0.7, 1.0 - 1e-10], dtype=torch.float32)
        target = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)
        loss = loss_fn(y_pred, target)
        assert torch.isfinite(loss)  # Loss should be finite


class TestWrappedLosses:
    def test_masked_mse_loss(self):
        """Test MaskedMSELoss wrapper."""
        loss_fn = MaskedMSELoss()
        y_pred = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32)
        target = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        mask = torch.tensor([True, True, False, True], dtype=torch.bool)

        loss = loss_fn(y_pred, target, mask=mask)

        # Calculate expected loss manually
        y_pred_masked = y_pred[mask]
        target_masked = target[mask]
        expected = torch.nn.functional.mse_loss(y_pred_masked, target_masked)
        assert torch.isclose(loss, expected)

    def test_masked_bce_loss(self):
        """Test MaskedBCELoss wrapper."""
        loss_fn = MaskedBCELoss()
        y_pred = torch.tensor([0.2, 0.7, 0.9, 0.3], dtype=torch.float32)
        target = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        mask = torch.tensor([True, True, False, True], dtype=torch.bool)

        loss = loss_fn(y_pred, target, mask=mask)

        # Calculate expected loss manually
        y_pred_masked = y_pred[mask]
        target_masked = target[mask]
        expected = torch.nn.functional.binary_cross_entropy(y_pred_masked, target_masked)
        assert torch.isclose(loss, expected)

    def test_masked_l1_loss(self):
        """Test MaskedL1Loss wrapper."""
        loss_fn = MaskedL1Loss()
        y_pred = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32)
        target = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        mask = torch.tensor([True, True, False, True], dtype=torch.bool)

        loss = loss_fn(y_pred, target, mask=mask)

        # Calculate expected loss manually
        y_pred_masked = y_pred[mask]
        target_masked = target[mask]
        expected = torch.nn.functional.l1_loss(y_pred_masked, target_masked)
        assert torch.isclose(loss, expected)

    def test_masked_huber_loss(self):
        """Test MaskedHuberLoss wrapper."""
        loss_fn = MaskedHuberLoss()
        y_pred = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32)
        target = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        mask = torch.tensor([True, True, False, True], dtype=torch.bool)

        loss = loss_fn(y_pred, target, mask=mask)

        # Calculate expected loss manually
        y_pred_masked = y_pred[mask]
        target_masked = target[mask]
        expected = torch.nn.functional.huber_loss(y_pred_masked, target_masked)
        assert torch.isclose(loss, expected)

    def test_with_weights(self):
        """Test wrapped losses with weights."""
        loss_fn = MaskedMSELoss()
        y_pred = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32)
        target = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        weights = torch.tensor([0.5, 1.0, 2.0, 1.5], dtype=torch.float32)

        loss = loss_fn(y_pred, target, weights=weights)

        # Manually calculate weighted loss
        weighted_squared_error = weights * (y_pred - target) ** 2
        expected = torch.sum(weighted_squared_error) / torch.sum(weights)
        assert torch.isclose(loss, expected)


def test_gradient_preservation_all_losses():
    """Test that all losses preserve gradients properly."""
    losses = [
        MSELoss(),
        BCELoss(),
        MaskedMSELoss(),
        MaskedL1Loss(),
        MaskedSmoothL1Loss(),
        MaskedHuberLoss(),
    ]

    for loss_fn in losses:
        # Skip non-regression losses that require special input formats
        if loss_fn.__class__.__name__ in [
            "MaskedCrossEntropyLoss",
            "MaskedNLLLoss",
            "MaskedKLDivLoss",
        ]:
            continue

        # Prepare inputs appropriate for the loss
        if isinstance(loss_fn, BCELoss) or isinstance(loss_fn, MaskedBCELoss):
            y_pred = torch.tensor([0.2, 0.7, 0.9], dtype=torch.float32, requires_grad=True)
            target = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)
        else:
            y_pred = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32, requires_grad=True)
            target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        # Check gradient flow
        loss = loss_fn(y_pred, target)
        loss.backward()

        # Just verify gradients exist and are finite
        assert y_pred.grad is not None
        assert torch.all(torch.isfinite(y_pred.grad))
