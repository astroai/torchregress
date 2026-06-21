"""Unit tests for torchregress.utils.quantile — the pinball loss primitives."""

from __future__ import annotations

import pytest
import torch

from torchregress.utils.quantile import multi_quantile_loss, quantile_loss

# ── quantile_loss ────────────────────────────────────────────────────────────


def test_quantile_loss_scalar_quantile() -> None:
    """Scalar quantile produces per-element pinball loss."""
    y_pred = torch.tensor([0.0, 2.0, 4.0])
    y_true = torch.tensor([1.0, 1.0, 1.0])
    # q=0.5: max(0.5*res, -0.5*res) = 0.5*|res|
    loss = quantile_loss(y_pred, y_true, 0.5)
    expected = 0.5 * (y_true - y_pred).abs()
    torch.testing.assert_close(loss, expected)


def test_quantile_loss_upper_quantile_penalizes_underprediction() -> None:
    """q=0.9 penalizes under-prediction more than over-prediction."""
    y_pred = torch.tensor([0.0, 10.0])
    y_true = torch.tensor([5.0, 5.0])
    # res = [5, -5]
    # q=0.9: max(0.9*res, -0.1*res)
    #  → max(4.5, -0.5) = 4.5 for first, max(-4.5, 0.5) = 0.5 for second
    loss = quantile_loss(y_pred, y_true, 0.9)
    expected = torch.tensor([4.5, 0.5])
    torch.testing.assert_close(loss, expected)


def test_quantile_loss_lower_quantile_penalizes_overprediction() -> None:
    """q=0.1 penalizes over-prediction more than under-prediction."""
    y_pred = torch.tensor([0.0, 10.0])
    y_true = torch.tensor([5.0, 5.0])
    # q=0.1: max(0.1*res, -0.9*res)
    #  → max(0.5, -4.5) = 0.5 for first, max(-0.5, 4.5) = 4.5 for second
    loss = quantile_loss(y_pred, y_true, 0.1)
    expected = torch.tensor([0.5, 4.5])
    torch.testing.assert_close(loss, expected)


def test_quantile_loss_tensor_quantile() -> None:
    """Per-element quantiles via a tensor."""
    y_pred = torch.tensor([0.0, 0.0])
    y_true = torch.tensor([1.0, -1.0])
    q = torch.tensor([0.9, 0.1])
    # First: res=1, q=0.9 → max(0.9, -0.1)=0.9
    # Second: res=-1, q=0.1 → max(-0.1, -0.9) → wait, max(-0.1, -0.9) = -0.1
    #  (q-1)*res for q=0.1 → -0.9 * -1 = 0.9
    # Actually: q*res = 0.1*(-1) = -0.1; (q-1)*res = -0.9*(-1) = 0.9 → max = 0.9
    loss = quantile_loss(y_pred, y_true, q)
    expected = torch.tensor([0.9, 0.9])
    torch.testing.assert_close(loss, expected)


def test_quantile_loss_zero_quantile() -> None:
    """q=0 always penalizes over-prediction."""
    y_pred = torch.tensor([0.0, 10.0])
    y_true = torch.tensor([5.0, 5.0])
    loss = quantile_loss(y_pred, y_true, 0.0)
    # q=0: max(0, -1*res) = max(0, -res) = ReLU(-res)
    expected = torch.tensor([0.0, 5.0])  # res=5→0, res=-5→5
    torch.testing.assert_close(loss, expected)


def test_quantile_loss_one_quantile() -> None:
    """q=1 always penalizes under-prediction."""
    y_pred = torch.tensor([0.0, 10.0])
    y_true = torch.tensor([5.0, 5.0])
    loss = quantile_loss(y_pred, y_true, 1.0)
    # q=1: max(res, 0) = ReLU(res)
    expected = torch.tensor([5.0, 0.0])
    torch.testing.assert_close(loss, expected)


def test_quantile_loss_device_transfer() -> None:
    """Float quantile is moved to y_pred device."""
    y_pred = torch.tensor([1.0, 2.0])
    y_true = torch.tensor([0.0, 3.0])
    loss = quantile_loss(y_pred, y_true, 0.5)
    assert loss.device == y_pred.device


def test_quantile_loss_dtype_preserved() -> None:
    """Output dtype matches y_pred dtype."""
    y_pred = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y_true = torch.tensor([0.0, 3.0], dtype=torch.float64)
    loss = quantile_loss(y_pred, y_true, 0.5)
    assert loss.dtype == torch.float64


def test_quantile_loss_tensor_quantile_device() -> None:
    """Tensor quantile is moved to y_pred device."""
    y_pred = torch.tensor([1.0])
    y_true = torch.tensor([0.0])
    q = torch.tensor(0.5)
    loss = quantile_loss(y_pred, y_true, q)
    assert loss.device == y_pred.device


def test_quantile_loss_batch_shape() -> None:
    """Batch dimensions are preserved."""
    y_pred = torch.randn(4, 3)
    y_true = torch.randn(4, 3)
    loss = quantile_loss(y_pred, y_true, 0.5)
    assert loss.shape == (4, 3)


def test_quantile_loss_exact_match_gives_zero() -> None:
    """When y_pred == y_true, loss is exactly zero for any quantile."""
    for q in [0.0, 0.1, 0.5, 0.9, 1.0]:
        loss = quantile_loss(torch.ones(5), torch.ones(5), q)
        assert (loss == 0.0).all()


# ── multi_quantile_loss ─────────────────────────────────────────────────────


def test_multi_quantile_loss_averages_across_quantiles() -> None:
    """Loss is the mean of per-quantile pinball losses."""
    y_pred = torch.tensor([[0.0, 2.0, 4.0], [1.0, 1.0, 1.0]])
    y_true = torch.tensor([1.0, 0.0])
    qs = torch.tensor([0.1, 0.5, 0.9])
    loss = multi_quantile_loss(y_pred, y_true, qs)
    assert loss.shape == (2,)
    # Verify against manual per-quantile computation
    manual = torch.stack(
        [quantile_loss(y_pred[:, i], y_true, qs[i].item()) for i in range(3)], dim=1
    ).mean(dim=1)
    torch.testing.assert_close(loss, manual)


def test_multi_quantile_loss_with_weights() -> None:
    """Weighted average across quantile dimension."""
    y_pred = torch.randn(8, 3)
    y_true = torch.randn(8)
    qs = torch.tensor([0.1, 0.5, 0.9])
    weights = torch.tensor([0.0, 1.0, 0.0])
    loss = multi_quantile_loss(y_pred, y_true, qs, quantile_weights=weights)
    assert loss.shape == (8,)
    # With only median weighted, result should match single-quantile median
    loss_median = quantile_loss(y_pred[:, 1], y_true, 0.5)
    torch.testing.assert_close(loss, loss_median)


def test_multi_quantile_loss_weights_length_mismatch_raises() -> None:
    """Mismatched quantile_weights raises ValueError."""
    y_pred = torch.randn(4, 3)
    y_true = torch.randn(4)
    qs = torch.tensor([0.1, 0.5, 0.9])
    bad_weights = torch.tensor([1.0, 2.0])  # 2 != 3
    with pytest.raises(ValueError, match="must match quantiles length"):
        multi_quantile_loss(y_pred, y_true, qs, quantile_weights=bad_weights)


def test_multi_quantile_loss_expands_target() -> None:
    """1-D target is auto-unsqueezed to match [batch, n_quantiles] pred."""
    y_pred = torch.randn(4, 3)
    y_true = torch.randn(4)  # [batch], not [batch, 1]
    qs = torch.tensor([0.1, 0.5, 0.9])
    loss = multi_quantile_loss(y_pred, y_true, qs)
    assert loss.shape == (4,)


def test_multi_quantile_loss_multivariate() -> None:
    """Multivariate targets: shape [batch, n_quantiles, features]."""
    y_pred = torch.randn(4, 3, 2)  # [batch, n_quantiles, features]
    y_true = torch.randn(4, 2)
    qs = torch.tensor([0.1, 0.5, 0.9])
    loss = multi_quantile_loss(y_pred, y_true, qs)
    assert loss.shape == (4, 2)
    assert (loss >= 0).all()


def test_multi_quantile_loss_weights_multivariate() -> None:
    """Weighted averaging works with multivariate targets."""
    y_pred = torch.randn(4, 3, 2)
    y_true = torch.randn(4, 2)
    qs = torch.tensor([0.1, 0.5, 0.9])
    weights = torch.tensor([0.0, 1.0, 0.0])
    loss = multi_quantile_loss(y_pred, y_true, qs, quantile_weights=weights)
    assert loss.shape == (4, 2)
    # Should match the median quantile only
    loss_median = quantile_loss(y_pred[:, 1, :], y_true, 0.5)
    torch.testing.assert_close(loss, loss_median)


def test_multi_quantile_loss_device_and_dtype() -> None:
    """Quantile tensors are moved to y_pred device and dtype."""
    y_pred = torch.randn(4, 3, dtype=torch.float64)
    y_true = torch.randn(4, dtype=torch.float64)
    qs = torch.tensor([0.1, 0.5, 0.9])  # default float32
    loss = multi_quantile_loss(y_pred, y_true, qs)
    assert loss.dtype == torch.float64
    assert loss.device == y_pred.device


def test_multi_quantile_loss_exact_match_gives_zero() -> None:
    """All zeros when predictions match targets exactly."""
    y_pred = torch.zeros(5, 3)
    y_true = torch.zeros(5)
    qs = torch.tensor([0.1, 0.5, 0.9])
    loss = multi_quantile_loss(y_pred, y_true, qs)
    assert (loss == 0.0).all()


def test_multi_quantile_loss_gradient_flow() -> None:
    """Gradients flow through the unweighted averaging path."""
    y_pred = torch.randn(4, 3, requires_grad=True)
    y_true = torch.randn(4)
    qs = torch.tensor([0.1, 0.5, 0.9])
    loss = multi_quantile_loss(y_pred, y_true, qs)
    loss.sum().backward()
    assert y_pred.grad is not None
    assert torch.isfinite(y_pred.grad).all()
