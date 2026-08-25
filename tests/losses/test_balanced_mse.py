"""Tests for BalancedMSELoss and BinReweightedMSELoss (renamed from BMCLoss per A11)."""

import pytest
import torch

from torchregress.losses import BalancedMSELoss, BinReweightedMSELoss
from torchregress.losses.loss_registry import create_loss_from_config


def test_balanced_mse_requires_fit() -> None:
    edges = torch.tensor([0.0, 0.5, 1.0])
    loss_fn = BalancedMSELoss(bin_edges=edges)
    y_pred = torch.tensor([0.2, 0.8])
    target = torch.tensor([0.1, 0.9])
    with pytest.raises(RuntimeError, match="fit"):
        loss_fn(y_pred, target)


def test_balanced_mse_upweights_rare_bin() -> None:
    """Most mass in [0, 0.5]; rare in [0.5, 1] — rare-bin sample should dominate mean loss."""
    edges = torch.tensor([0.0, 0.5, 1.0])
    train_y = torch.cat([torch.zeros(99), torch.ones(1)])
    loss_fn = BalancedMSELoss(bin_edges=edges).fit(train_y)
    w = loss_fn.bin_weights
    assert w[1] > w[0]


def test_balanced_mse_forward_and_reduction_none() -> None:
    edges = torch.tensor([0.0, 1.0, 2.0])
    train_y = torch.tensor([0.25, 0.25, 1.75, 1.75])
    loss_fn = BalancedMSELoss(bin_edges=edges).fit(train_y)
    y_pred = torch.tensor([0.0, 0.0, 2.0, 2.0])
    target = torch.tensor([0.5, 0.5, 1.5, 1.5])
    m = loss_fn(y_pred, target)
    assert m.ndim == 0 and m > 0
    loss_none = BalancedMSELoss(bin_edges=edges, reduction="none").fit(train_y)
    per = loss_none(y_pred, target)
    assert per.shape == y_pred.shape


def test_bin_weighted_mse_equal_binning_runs() -> None:
    train_y = torch.linspace(0.0, 1.0, 50)
    loss_fn = BinReweightedMSELoss(num_bins=5, noise_sigma=1.0, binning="equal").fit(train_y)
    y_pred = torch.zeros(10, 1)
    target = torch.ones(10, 1) * 0.5
    m = loss_fn(y_pred, target)
    assert m.ndim == 0


def test_bin_weighted_mse_larger_noise_sigma_reduces_spread_of_weights() -> None:
    train_y = torch.cat([torch.zeros(90), torch.ones(10)])
    w_tight = (
        BinReweightedMSELoss(num_bins=2, noise_sigma=0.01, binning="equal").fit(train_y).bin_weights
    )
    w_smooth = (
        BinReweightedMSELoss(num_bins=2, noise_sigma=5.0, binning="equal").fit(train_y).bin_weights
    )
    assert float(w_tight.std()) > float(w_smooth.std())


def test_create_loss_from_config_balanced_mse() -> None:
    edges = [0.0, 1.0, 2.0]
    loss_fn = create_loss_from_config({"type": "balanced_mse", "bin_edges": torch.tensor(edges)})
    assert isinstance(loss_fn, BalancedMSELoss)


def test_create_loss_from_config_bmc() -> None:
    loss_fn = create_loss_from_config({"type": "bin_weighted_mse", "num_bins": 3})
    assert isinstance(loss_fn, BinReweightedMSELoss)
