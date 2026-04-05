import pytest
import torch
import torch.nn as nn

from torchregress.losses import (
    CharbonnierLoss,
    GaussianNLLLoss,
    LogCoshLoss,
    PseudoHuberLoss,
    QuantileLoss,
    WeightedLossWrapper,
)


# Define standard regression losses using the wrapper for testing
def WeightedMSELoss():
    return WeightedLossWrapper(nn.MSELoss())


def WeightedL1Loss():
    return WeightedLossWrapper(nn.L1Loss())


def WeightedHuberLoss():
    return WeightedLossWrapper(nn.HuberLoss())


# List of losses to test for general regression properties
REGRESSION_LOSSES = [
    WeightedMSELoss,
    WeightedL1Loss,
    WeightedHuberLoss,
    QuantileLoss,
    PseudoHuberLoss,
    LogCoshLoss,
    CharbonnierLoss,
]


@pytest.mark.parametrize("loss_factory", REGRESSION_LOSSES)
def test_finite_loss_values(loss_factory):
    """Test that loss returns finite values for reasonable inputs."""
    loss_fn = loss_factory()
    input = torch.randn(10, 1)
    target = torch.randn(10, 1)
    loss = loss_fn(input, target)
    assert torch.isfinite(loss).all(), f"{loss_fn.__class__.__name__} returned non-finite loss"


@pytest.mark.parametrize("loss_factory", REGRESSION_LOSSES)
def test_masking_invariance(loss_factory):
    """Test that masked values do not affect the loss."""
    loss_fn = loss_factory()
    # Ensure reduction is mean (default) or set it explicitly if needed
    if hasattr(loss_fn, "reduction"):
        loss_fn.reduction = "mean"

    input = torch.randn(10, 1)
    target = torch.randn(10, 1)
    # weights = torch.ones(10, 1)  # Unused

    # Mask half the batch
    mask = torch.ones(10, 1, dtype=torch.bool)
    mask[5:] = False

    # Weighted/Masked loss
    loss_masked = loss_fn(input, target, mask=mask)

    # Compute manually on unmasked portion
    loss_fn_none = loss_factory()
    # Force reduction to none
    if hasattr(loss_fn_none, "reduction"):
        loss_fn_none.reduction = "none"
    elif hasattr(loss_fn_none, "torch_loss") and hasattr(loss_fn_none.torch_loss, "reduction"):
        loss_fn_none.torch_loss.reduction = "none"

    loss_vals = loss_fn_none(input, target)
    expected_loss = loss_vals[:5].mean()

    # Depending on implementation, loss_masked might be scalar or tensor
    assert torch.allclose(loss_masked, expected_loss), (
        f"{loss_fn.__class__.__name__} masking failed"
    )


@pytest.mark.parametrize("loss_factory", REGRESSION_LOSSES)
def test_monotonicity(loss_factory):
    """Test that loss increases as prediction moves away from target."""
    loss_fn = loss_factory()
    # Force reduction to none
    if hasattr(loss_fn, "reduction"):
        loss_fn.reduction = "none"

    target = torch.zeros(10, 1)

    # Create predictions moving away from 0
    input_near = torch.ones(10, 1) * 0.1
    input_far = torch.ones(10, 1) * 1.0

    loss_near = loss_fn(input_near, target)
    loss_far = loss_fn(input_far, target)

    # Check if any element in far is greater than near
    # Most losses should be strictly increasing with error magnitude
    assert (loss_far > loss_near).all(), (
        f"{loss_fn.__class__.__name__} is not monotonic increasing with error"
    )


def test_gaussian_nll_finite():
    """Specific test for GaussianNLLLoss financing."""
    loss_fn = GaussianNLLLoss()
    # Input is (mean, var) or (mean, log_var)
    mean = torch.randn(10, 1)
    # Ensure variance is positive and distinct if using directly
    log_var = torch.randn(10, 1)
    target = torch.randn(10, 1)

    input = (mean, log_var)
    loss = loss_fn(input, target)
    assert torch.isfinite(loss).all()


def test_extreme_values_handling():
    """Test handling of extreme values."""
    loss_fn = WeightedMSELoss()
    input = torch.tensor([[1e6]])
    target = torch.tensor([[1e6 + 1.0]])
    loss = loss_fn(input, target)
    assert torch.isfinite(loss)

    # Very small numbers
    input = torch.tensor([[1e-6]])
    target = torch.tensor([[0.0]])
    loss = loss_fn(input, target)
    assert torch.isfinite(loss)


def test_nan_propagation():
    """Test that NaNs propogate (unless masked)."""
    loss_fn = WeightedMSELoss()
    input = torch.tensor([[float("nan")]])
    target = torch.tensor([[0.0]])
    loss = loss_fn(input, target)
    # MSE loss with NaN input should be NaN
    assert torch.isnan(loss).any()
