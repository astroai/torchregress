from __future__ import annotations

import torch

from torchregress.calibration import (
    IsotonicMeanCalibrator,
    PITCalibrator,
    VarianceTemperatureScaler,
)
from torchregress.constraints import (
    BoundedHead,
    NonCrossingSort,
    NonNegativeHead,
    SimplexHead,
    SpectralNormWrapper,
)


def test_constrained_heads_basic_behavior() -> None:
    x = torch.randn(16, 4)

    nonneg = NonNegativeHead(torch.nn.Linear(4, 2))
    out_nonneg = nonneg(x)
    assert torch.all(out_nonneg >= 0)

    bounded = BoundedHead(torch.nn.Linear(4, 1), low=-1.0, high=2.0)
    out_bounded = bounded(x)
    assert torch.all(out_bounded >= -1.0)
    assert torch.all(out_bounded <= 2.0)

    simplex = SimplexHead(torch.nn.Linear(4, 3))
    out_simplex = simplex(x)
    assert torch.allclose(out_simplex.sum(dim=-1), torch.ones(x.shape[0]), atol=1e-5)


def test_non_crossing_sort() -> None:
    q = torch.tensor([[0.5, 0.2, 0.7], [1.2, 0.8, 0.9]])
    sorted_q = NonCrossingSort(dim=-1)(q)
    diffs = sorted_q[:, 1:] - sorted_q[:, :-1]
    assert torch.all(diffs >= 0)


def test_spectral_norm_wrapper_forward() -> None:
    layer = SpectralNormWrapper(torch.nn.Linear(4, 1))
    out = layer(torch.randn(8, 4))
    assert out.shape == (8, 1)


def test_variance_temperature_scaler_fit_transform() -> None:
    torch.manual_seed(0)
    mean = torch.randn(128)
    target = mean + 0.2 * torch.randn(128)
    var = torch.full_like(mean, 0.01)

    scaler = VarianceTemperatureScaler()
    scaler.fit(mean, var, target, max_iter=50)
    var_scaled = scaler.transform(var)
    assert torch.all(var_scaled > 0)


def test_isotonic_and_pit_calibrators() -> None:
    torch.manual_seed(0)
    pred = torch.linspace(-2, 2, 128)
    target = pred + 0.3 * torch.randn(128)

    iso = IsotonicMeanCalibrator().fit(pred, target)
    pred_iso = iso.transform(pred)
    assert pred_iso.shape == pred.shape

    pit_raw = PITCalibrator.pit_from_gaussian(pred, torch.ones_like(pred) * 0.5, target)
    pit_cal = PITCalibrator().fit(pit_raw)
    pit_adj = pit_cal.transform(pit_raw)
    assert torch.all(pit_adj > 0.0)
    assert torch.all(pit_adj < 1.0)
