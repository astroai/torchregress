from __future__ import annotations

import torch

from torchregress.losses.transforms import (
    BoxCoxTransformLoss,
    LogTransformLoss,
    SqrtTransformLoss,
    TransformedTargetLoss,
    YeoJohnsonTransformLoss,
)


def test_log_transform_loss_matches_manual_log_mse() -> None:
    y_pred = torch.tensor([[0.5], [1.0], [2.0], [4.0]])
    target = torch.tensor([[0.4], [1.2], [1.8], [5.0]])
    loss_fn = LogTransformLoss()
    loss = loss_fn(y_pred, target)
    expected = ((torch.log(y_pred + 1e-6) - torch.log(target + 1e-6)) ** 2).mean()
    assert torch.allclose(loss, expected)


def test_transformed_target_loss_supports_weights_and_mask() -> None:
    y_pred = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    target = torch.tensor([[1.2], [2.5], [2.8], [3.7]])
    mask = torch.tensor([[True], [False], [True], [True]])
    weights = torch.tensor([[1.0], [1.0], [2.0], [3.0]])
    loss_fn = TransformedTargetLoss("sqrt", reduction="mean")
    loss = loss_fn(y_pred, target, mask=mask, weights=weights)
    raw = (torch.sqrt(y_pred + 1e-6) - torch.sqrt(target + 1e-6)).square()
    expected = (raw[mask] * weights[mask]).sum() / weights[mask].sum()
    assert torch.allclose(loss, expected)


def test_boxcox_and_sqrt_transform_losses_are_finite() -> None:
    y_pred = torch.linspace(0.2, 2.0, 32).unsqueeze(-1)
    target = y_pred + 0.1
    assert torch.isfinite(BoxCoxTransformLoss(lam=0.25)(y_pred, target))
    assert torch.isfinite(SqrtTransformLoss()(y_pred, target))


def test_yeojohnson_transform_loss_supports_signed_targets() -> None:
    y_pred = torch.linspace(-2.0, 2.0, 32).unsqueeze(-1)
    target = y_pred + 0.25
    loss = YeoJohnsonTransformLoss(lam=0.5)(y_pred, target)
    assert torch.isfinite(loss)


def test_transformed_target_loss_inverse_uses_transform_inverse() -> None:
    transformed = torch.tensor([[0.0], [0.5], [1.0]])
    restored = LogTransformLoss().inverse(transformed)
    expected = torch.exp(transformed) - 1e-6
    assert torch.allclose(restored, expected.clamp_min(0.0))
