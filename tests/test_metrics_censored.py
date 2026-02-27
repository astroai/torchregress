from __future__ import annotations

import torch

from torchregress.metrics import (
    censoring_rate,
    concordance_index,
    interval_overlap_rate,
    observed_mae,
)


def test_censoring_rate() -> None:
    censoring = torch.tensor([0, 1, 0, -1, 0])
    rate = censoring_rate(censoring)
    assert torch.isclose(rate, torch.tensor(0.4))


def test_observed_mae_uses_observed_only() -> None:
    y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([1.5, 1.0, 2.5, 5.0])
    censoring = torch.tensor([0, 1, 0, -1])

    mae = observed_mae(y_pred, y_true, censoring)
    # observed entries are indices 0 and 2 -> errors 0.5 and 0.5
    assert torch.isclose(mae, torch.tensor(0.5))


def test_concordance_index_basic_range() -> None:
    y_pred = torch.tensor([0.5, 1.0, 1.5, 2.0])
    y_true = torch.tensor([0.6, 1.1, 1.4, 2.2])
    censoring = torch.tensor([0, 0, 1, 0])

    c_idx = concordance_index(y_pred, y_true, censoring)
    assert torch.isfinite(c_idx)
    assert 0.0 <= float(c_idx) <= 1.0


def test_interval_overlap_rate() -> None:
    pred_l = torch.tensor([0.0, 1.0, 2.0])
    pred_u = torch.tensor([0.8, 1.4, 2.1])
    low = torch.tensor([0.1, 1.2, 1.0])
    up = torch.tensor([0.6, 1.3, 1.5])

    overlap = interval_overlap_rate(pred_l, pred_u, low, up)
    assert torch.isclose(overlap, torch.tensor(2.0 / 3.0))
