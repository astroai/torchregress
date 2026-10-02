"""Tests for FaithfulGaussianLoss."""

import math

import torch

from torchregress.losses import FaithfulGaussianLoss
from torchregress.losses.loss_registry import create_loss_from_config


def test_faithful_nll_does_not_backprop_to_mean() -> None:
    mean = torch.randn(8, 1, requires_grad=True)
    logvar = torch.zeros(8, 1, requires_grad=True)
    target = torch.randn(8, 1)
    loss_fn = FaithfulGaussianLoss(mean_weight=0.0, variance_weight=1.0)
    loss = loss_fn((mean, logvar), target)
    loss.backward()
    # Mean is detached in the NLL residual; with mean_weight=0 it is unused.
    assert mean.grad is None


def test_faithful_mean_only_backprop_from_mse() -> None:
    mean = torch.tensor([[0.0]], requires_grad=True)
    logvar = torch.tensor([[0.0]], requires_grad=True)
    target = torch.tensor([[1.0]])
    loss_fn = FaithfulGaussianLoss(mean_weight=1.0, variance_weight=0.0)
    loss = loss_fn((mean, logvar), target)
    loss.backward()
    assert mean.grad is not None and abs(float(mean.grad[0, 0]) + 2.0) < 1e-5
    assert logvar.grad is None


def test_faithful_matches_nll_when_mean_weight_zero_and_mean_zero() -> None:
    """If mean is fixed at target, detached residual is 0; NLL reduces to log-var term."""
    mean = torch.zeros(4, 1)
    logvar = torch.zeros(4, 1, requires_grad=True)
    target = torch.zeros(4, 1)
    f = FaithfulGaussianLoss(mean_weight=0.0, variance_weight=1.0)((mean, logvar), target)
    # 0.5 * (log 2pi + log(1) + 0) = 0.5 * log(2pi)
    expected = 0.5 * (math.log(2 * math.pi))
    assert abs(float(f.item()) - expected) < 1e-5


def test_create_loss_from_config() -> None:
    fn = create_loss_from_config({"type": "faithful_gaussian", "mean_weight": 0.5})
    assert isinstance(fn, FaithfulGaussianLoss)
    assert fn.mean_weight == 0.5


def test_per_output_mean_weight_scales_each_output() -> None:
    mean = torch.zeros(3, 2, requires_grad=True)
    logvar = torch.zeros(3, 2)
    target = torch.ones(3, 2)
    weights = [1.0, 0.25]
    loss_fn = FaithfulGaussianLoss(mean_weight=weights, variance_weight=0.0, reduction="sum")
    loss_fn((mean, logvar), target).backward()
    # d/dmean of w * (mean - 1)^2 = 2 w (mean - 1) = -2 w
    expected = torch.tensor([[-2.0, -0.5]] * 3)
    torch.testing.assert_close(mean.grad, expected)


def test_per_output_mean_weight_zero_entry_gives_no_mean_gradient() -> None:
    mean = torch.zeros(2, 2, requires_grad=True)
    logvar = torch.zeros(2, 2)
    target = torch.ones(2, 2)
    loss_fn = FaithfulGaussianLoss(mean_weight=[1.0, 0.0], variance_weight=0.0)
    loss_fn((mean, logvar), target).backward()
    assert torch.all(mean.grad[:, 1] == 0)
    assert torch.all(mean.grad[:, 0] != 0)


def test_scalar_and_uniform_vector_mean_weight_agree() -> None:
    torch.manual_seed(0)
    mean = torch.randn(5, 3, requires_grad=True)
    logvar = torch.randn(5, 3)
    target = torch.randn(5, 3)
    scalar = FaithfulGaussianLoss(mean_weight=0.7)((mean, logvar), target)
    vector = FaithfulGaussianLoss(mean_weight=[0.7, 0.7, 0.7])((mean, logvar), target)
    torch.testing.assert_close(scalar, vector)


def test_mean_weight_validation() -> None:
    import pytest

    with pytest.raises(ValueError):
        FaithfulGaussianLoss(mean_weight=[1.0, -0.1])
    with pytest.raises(ValueError):
        FaithfulGaussianLoss(mean_weight=float("nan"))
    with pytest.raises(ValueError):
        FaithfulGaussianLoss(mean_weight=torch.ones(2, 2))
    with pytest.raises(ValueError):
        FaithfulGaussianLoss(variance_weight=-1.0)
    with pytest.raises(ValueError, match="entries"):
        FaithfulGaussianLoss(mean_weight=[1.0, 1.0])(
            (torch.zeros(4, 3), torch.zeros(4, 3)), torch.zeros(4, 3)
        )


def test_per_output_mean_weight_follows_module_device_dtype_moves() -> None:
    loss_fn = FaithfulGaussianLoss(mean_weight=[1.0, 2.0]).double()
    assert loss_fn.mean_weight.dtype == torch.float64
    mean = torch.zeros(2, 2, dtype=torch.float32)
    # Forward casts the weight to the prediction dtype, so mixed dtypes still work.
    out = loss_fn((mean, mean.clone()), torch.ones(2, 2))
    assert out.dtype == torch.float32


def test_huber_matches_mse_in_quadratic_zone_and_is_linear_beyond() -> None:
    logvar = torch.zeros(2, 1)
    near_target = torch.tensor([[0.5], [-0.5]])
    mean = torch.zeros(2, 1, requires_grad=True)
    mse = FaithfulGaussianLoss(variance_weight=0.0)((mean, logvar), near_target)
    hub = FaithfulGaussianLoss(variance_weight=0.0, mean_loss="huber", huber_delta=1.0)(
        (mean, logvar), near_target
    )
    torch.testing.assert_close(mse, hub)

    # Beyond delta the gradient on the mean is bounded by 2 * delta.
    far_target = torch.tensor([[100.0], [-100.0]])
    mean = torch.zeros(2, 1, requires_grad=True)
    FaithfulGaussianLoss(variance_weight=0.0, mean_loss="huber", huber_delta=1.0, reduction="sum")(
        (mean, logvar), far_target
    ).backward()
    torch.testing.assert_close(mean.grad.abs(), torch.full((2, 1), 2.0))


def test_huber_option_validation_and_config() -> None:
    import pytest

    with pytest.raises(ValueError):
        FaithfulGaussianLoss(mean_loss="l1")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FaithfulGaussianLoss(mean_loss="huber", huber_delta=0.0)
    fn = create_loss_from_config(
        {"type": "faithful_gaussian", "mean_weight": [1.0, 0.5], "mean_loss": "huber"}
    )
    assert fn.mean_loss == "huber"
    assert fn.mean_weight.tolist() == [1.0, 0.5]
