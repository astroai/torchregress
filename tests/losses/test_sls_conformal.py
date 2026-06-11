from __future__ import annotations

import pytest
import torch

from torchregress.losses import SLSLoss
from torchregress.losses.conformal import SLSConformal


def test_sls_conformal_predictor() -> None:
    d = 1
    context_dim = 2
    sls_loss = SLSLoss(d=d, context_dim=context_dim, K=1, warmup_steps=5)

    # Dummy dataset
    n_cal = 20
    n_test = 10
    x_cal = torch.randn(n_cal, context_dim)
    y_cal = torch.randn(n_cal, d)

    x_test = torch.randn(n_test, context_dim)

    # Initialize conformal predictor
    conformal = SLSConformal(sls_loss, alpha=0.1, grid_size=100)

    # Check uncalibrated predictions raise error
    with pytest.raises(RuntimeError):
        conformal.predict_interval_from_grid(x_test, y_min=-3.0, y_max=3.0)

    # Calibrate
    conformal.calibrate(x_cal, y_cal)
    assert conformal._is_calibrated
    assert conformal.q_hat is not None

    # Predict intervals
    lower, upper = conformal.predict_interval_from_grid(x_test, y_min=-3.0, y_max=3.0)

    assert lower.shape == (n_test, 1)
    assert upper.shape == (n_test, 1)
    assert torch.all(lower <= upper)
