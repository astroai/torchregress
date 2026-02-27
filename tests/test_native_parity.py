from typing import cast

import torch
import torch.nn as nn
from torchmetrics import MeanAbsoluteError as TMMeanAbsoluteError
from torchmetrics import MeanSquaredError as TMMeanSquaredError
from torchmetrics import R2Score as TMR2Score

from torchregress.losses import (
    WeightedCrossEntropyLoss,
    WeightedGaussianNLLLoss,
    WeightedHuberLoss,
    WeightedL1Loss,
    WeightedLossWrapper,
    WeightedMSELoss,
    WeightedNLLLoss,
)
from torchregress.metrics import (
    gaussian_nll,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    rmse,
)


def test_weighted_point_loss_wrappers_match_native_without_mask_or_weights() -> None:
    torch.manual_seed(0)
    y_pred = torch.randn(16, 3)
    y_true = torch.randn(16, 3)

    assert torch.allclose(WeightedMSELoss()(y_pred, y_true), nn.MSELoss()(y_pred, y_true))
    assert torch.allclose(WeightedL1Loss()(y_pred, y_true), nn.L1Loss()(y_pred, y_true))

    delta = 1.5
    assert torch.allclose(
        WeightedHuberLoss(delta=delta)(y_pred, y_true),
        nn.HuberLoss(delta=delta)(y_pred, y_true),
    )


def test_weighted_gaussian_nll_wrapper_matches_native_for_var_and_logvar_inputs() -> None:
    torch.manual_seed(0)
    mean = torch.randn(12, 2)
    target = torch.randn(12, 2)
    var = torch.rand(12, 2) + 0.1

    native = nn.GaussianNLLLoss()

    tr_var = WeightedGaussianNLLLoss(log_variance=False)
    loss_var = tr_var((mean, var), target)
    assert torch.allclose(loss_var, native(mean, target, var), atol=1e-6, rtol=1e-6)

    tr_logvar = WeightedGaussianNLLLoss(log_variance=True)
    loss_logvar = tr_logvar((mean, var.log()), target)
    assert torch.allclose(loss_logvar, native(mean, target, var), atol=1e-6, rtol=1e-6)


def test_functional_gaussian_nll_matches_torch_full_gaussian_nll_loss() -> None:
    torch.manual_seed(0)
    mean = torch.randn(20, 1)
    target = torch.randn(20, 1)
    var = torch.rand(20, 1) + 0.05

    tr_loss = gaussian_nll(mean, target, var, reduction="mean")
    native = nn.GaussianNLLLoss(full=True, reduction="mean")
    native_loss = native(mean, target, var).item()

    assert abs(tr_loss - native_loss) < 1e-6


def test_weighted_cross_entropy_wrapper_matches_native_without_mask_or_weights() -> None:
    torch.manual_seed(0)
    logits = torch.randn(10, 4)
    target = torch.randint(0, 4, (10,))

    tr_loss = WeightedCrossEntropyLoss()(logits, target)
    native_loss = nn.CrossEntropyLoss()(logits, target)
    assert torch.allclose(tr_loss, native_loss, atol=1e-6, rtol=1e-6)


def test_weighted_nll_wrapper_matches_native_without_mask_or_weights() -> None:
    torch.manual_seed(0)
    logits = torch.randn(12, 3)
    log_probs = torch.log_softmax(logits, dim=-1)
    target = torch.randint(0, 3, (12,))

    tr_loss = WeightedNLLLoss()(log_probs, target)
    native_loss = nn.NLLLoss()(log_probs, target)
    assert torch.allclose(tr_loss, native_loss, atol=1e-6, rtol=1e-6)


def test_generic_weighted_loss_wrapper_matches_native_smooth_l1_without_mask_or_weights() -> None:
    torch.manual_seed(0)
    y_pred = torch.randn(14, 2)
    y_true = torch.randn(14, 2)
    beta = 0.7

    tr_loss = WeightedLossWrapper(nn.SmoothL1Loss(beta=beta))(y_pred, y_true)
    native_loss = nn.SmoothL1Loss(beta=beta)(y_pred, y_true)
    assert torch.allclose(tr_loss, native_loss, atol=1e-6, rtol=1e-6)


def test_point_metric_functions_match_torchmetrics_baselines() -> None:
    torch.manual_seed(0)
    y_pred = torch.randn(32, 3)
    y_true = torch.randn(32, 3)

    tr_mse = float(cast(float, mean_squared_error(y_pred, y_true)))
    tm_mse = float(TMMeanSquaredError()(y_pred, y_true).item())
    assert abs(tr_mse - tm_mse) < 1e-6

    tr_mae = float(cast(float, mean_absolute_error(y_pred, y_true)))
    tm_mae = float(TMMeanAbsoluteError()(y_pred, y_true).item())
    assert abs(tr_mae - tm_mae) < 1e-6

    tr_r2 = float(cast(float, r2_score(y_pred, y_true)))
    tm_r2 = float(TMR2Score()(y_pred, y_true).item())
    assert abs(tr_r2 - tm_r2) < 1e-6

    tr_rmse = float(cast(float, rmse(y_pred, y_true)))
    tm_rmse = float(TMMeanSquaredError(squared=False)(y_pred, y_true).item())
    assert abs(tr_rmse - tm_rmse) < 1e-6
