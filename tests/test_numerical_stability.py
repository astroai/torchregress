import pytest
import torch

from torchregress.losses import (
    CauchyLoss,
    ExpectileLoss,
    GaussianNLLLoss,
    HuberLoss,
    LogCoshLoss,
    MAELoss,
    MSELoss,
    PoissonLoss,
    PseudoHuberLoss,
    QuantileLoss,
    TweedieLoss,
    # Import other losses as needed
)


@pytest.mark.parametrize(
    "loss_class",
    [
        MSELoss,
        MAELoss,
        GaussianNLLLoss,
        HuberLoss,
        PseudoHuberLoss,
        LogCoshLoss,
        CauchyLoss,
        QuantileLoss,
        ExpectileLoss,
        PoissonLoss,
        TweedieLoss,
        # Add other losses
    ],
)
def test_gradient_flow(loss_class):
    """Test that gradients flow properly through the loss function."""
    loss_fn = loss_class()

    # Create tensors requiring gradients
    y_pred = torch.randn(10, requires_grad=True)
    y_true = torch.randn(10)

    # Forward pass
    loss = loss_fn(y_pred, y_true)

    # Backward pass
    loss.backward()

    # Check that gradients were computed
    assert y_pred.grad is not None
    assert not torch.isnan(y_pred.grad).any()
    assert not torch.isinf(y_pred.grad).any()


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize(
    "loss_class",
    [
        MSELoss,
        MAELoss,
        HuberLoss,
        PseudoHuberLoss,
        # Add other losses that support different reduction modes
    ],
)
def test_reduction_modes(loss_class, reduction):
    """Test different reduction modes."""
    loss_fn = loss_class(reduction=reduction)

    y_pred = torch.randn(10)
    y_true = torch.randn(10)

    loss = loss_fn(y_pred, y_true)

    if reduction == "none":
        assert loss.shape == y_pred.shape
    else:
        assert loss.numel() == 1


def test_extreme_values_stability():
    """Test numerical stability with extreme values."""
    losses = [
        MSELoss(),
        MAELoss(),
        HuberLoss(),
        # Add other loss instances
    ]

    # Extremely large values
    y_pred_large = torch.tensor([1e30])
    y_true_large = torch.tensor([1e30])

    # Extremely small values
    y_pred_small = torch.tensor([1e-30])
    y_true_small = torch.tensor([1e-30])

    for loss_fn in losses:
        # Large values
        loss_large = loss_fn(y_pred_large, y_true_large)
        assert torch.isfinite(loss_large)

        # Small values
        loss_small = loss_fn(y_pred_small, y_true_small)
        assert torch.isfinite(loss_small)
