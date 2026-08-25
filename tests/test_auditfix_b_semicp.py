"""Workstream B audit-fix regression test for the semicp weighted path (B4 / TR-MET-12).

``SemiConformalCalibrator.compute_thresholds`` must apply the finite-sample
correction: the smallest score whose normalized cumulative weight reaches
``ceil((n+1)*(1-alpha))/n`` (exact order statistic in the unweighted case).
"""

import math

import numpy as np
import torch

from torchregress.calibration.semicp import SemiConformalCalibrator


def test_unweighted_path_matches_order_statistic() -> None:
    """Zero target weight + uniform calibration weights -> exact order statistic."""
    scores = np.array([2.0, 0.5, 3.5, 1.0, 4.5])
    calibrator = SemiConformalCalibrator().fit(scores)
    # w_tgt = 0 and uniform weights: threshold = sorted[k-1], k = ceil((n+1)(1-a)).
    n = len(scores)
    for alpha in (0.1, 0.3):
        k = min(math.ceil((n + 1) * (1.0 - alpha)), n)
        expected = np.sort(scores)[k - 1]
        assert np.isclose(calibrator.compute_thresholds(0.0, alpha=alpha), expected)


def test_weighted_path_uses_corrected_mass() -> None:
    """Weighted path targets ceil((n+1)*(1-alpha))/n of the cumulative curve."""
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
    weights = torch.tensor([3.0, 1.0, 1.0, 1.0])
    calibrator = SemiConformalCalibrator().fit(scores, weights)
    alpha = 0.5
    n = 4
    q_adj = min(math.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)  # ceil(2.5)/4 = 0.75

    # Target weight w = 0 keeps denom = sum_w_cal = 6; cumulative curve:
    # [3/6, 4/6, 5/6, 1.0]. First index reaching 0.75 is index 2 (score 3.0).
    # The uncorrected level 1 - alpha = 0.5 would pick index 0 (score 1.0).
    q = calibrator.compute_thresholds(torch.tensor([0.0]), alpha=alpha)
    assert q_adj == 0.75
    assert np.isclose(float(q[0]), 3.0)


def test_thresholds_monotone_in_alpha() -> None:
    """Tighter alpha must never produce a smaller threshold (n=25)."""
    generator = torch.Generator().manual_seed(7)
    scores = torch.rand(25, generator=generator)
    calibrator = SemiConformalCalibrator().fit(scores)
    q_loose = calibrator.compute_thresholds(0.0, alpha=0.2)
    q_tight = calibrator.compute_thresholds(0.0, alpha=0.05)
    assert float(q_tight) >= float(q_loose)


def test_coverage_simulation_respects_nominal_level() -> None:
    """Monte-Carlo check: unweighted thresholds cover at least 1 - alpha."""
    generator = torch.Generator().manual_seed(42)
    n_cal, alpha, trials = 20, 0.1, 400
    hits = 0
    for _ in range(trials):
        cal_scores = torch.rand(n_cal, generator=generator)
        calibrator = SemiConformalCalibrator().fit(cal_scores)
        threshold = float(calibrator.compute_thresholds(0.0, alpha=alpha))
        test_score = float(torch.rand(1, generator=generator))
        hits += test_score <= threshold
    coverage = hits / trials
    assert coverage >= 1.0 - alpha - 0.03, f"coverage {coverage:.3f} below nominal"
