from __future__ import annotations

import torch

from torchregress.metrics import tail_mae, tail_rmse


def test_tail_mae_upper_tail() -> None:
    y_true = torch.tensor([0.0, 1.0, 2.0, 10.0, 12.0])
    y_pred = torch.tensor([0.0, 1.0, 2.5, 8.0, 14.0])

    result = tail_mae(y_pred, y_true, quantile=0.8, tail="upper")
    # Upper 20% roughly selects 10 and 12 -> abs errors 2 and 2
    assert isinstance(result, float)
    assert abs(result - 2.0) < 1e-6


def test_tail_rmse_lower_tail() -> None:
    y_true = torch.tensor([-10.0, -8.0, 0.0, 1.0, 2.0])
    y_pred = torch.tensor([-12.0, -7.0, 0.0, 1.0, 2.0])

    result = tail_rmse(y_pred, y_true, quantile=0.8, tail="lower")
    assert isinstance(result, float)
    assert abs(result - 2.0) < 1e-6
