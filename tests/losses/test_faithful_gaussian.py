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
