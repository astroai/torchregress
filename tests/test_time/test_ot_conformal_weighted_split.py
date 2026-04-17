"""Weighted split conformal adapter diagnostics and PredictiveBatch helpers."""

from __future__ import annotations

import pytest
import torch

import torchregress as tr


def test_weighted_split_uniform_weights_near_nominal_coverage_diag() -> None:
    torch.manual_seed(0)
    n = 200
    scores = torch.abs(torch.randn(n)) + 0.1
    w = torch.ones(n)
    alpha = 0.1
    ad = tr.test_time.WeightedSplitConformalAdapter(alpha=alpha).calibrate(scores, w)
    diag = ad.coverage_diagnostics(scores, w)
    assert diag["n_calibration"] == n
    assert abs(diag["weighted_empirical_coverage"] - diag["nominal_coverage"]) < 0.08


def test_coverage_diagnostics_requires_calibrate() -> None:
    ad = tr.test_time.WeightedSplitConformalAdapter(alpha=0.1)
    with pytest.raises(RuntimeError, match="calibrate"):
        ad.coverage_diagnostics(torch.randn(5), torch.ones(5))


def test_weighted_split_predictive_batch_extra_keys() -> None:
    n = 40
    scores = torch.linspace(0.1, 2.0, n)
    w = torch.ones(n)
    ad = tr.test_time.WeightedSplitConformalAdapter(alpha=0.1).calibrate(scores, w)
    cand = torch.tensor([[0.5, 1.5], [0.2, 0.3]])
    pb = tr.test_time.weighted_split_classification_predictive_batch(
        ad,
        cand,
        gap_diagnostics={"l2_cdf_gap": 0.01},
        calibration_ess_inv_square=40.0,
    )
    assert pb.extra is not None
    assert "label_inclusion_mask" in pb.extra
    assert pb.extra["label_inclusion_mask"].shape == cand.shape
    assert "shift_gap_diagnostics" in pb.extra
    assert pb.extra["calibration_ess_inv_square"] == 40.0
    assert pb.mean is not None and pb.mean.shape == (2, 1)
