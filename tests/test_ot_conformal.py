import math

import pytest
import torch

import torchregress as tr
from torchregress.losses.conformal import _weighted_quantile
from torchregress.test_time.ot_conformal import (
    OptimalTransportCoverageGap,
    ScoreCDFReweighter,
    WeightedSplitConformalAdapter,
    _weighted_ecdf_on_grid,
)


def test_weighted_ecdf_matches_hand() -> None:
    scores = torch.tensor([0.0, 1.0, 2.0])
    w = torch.tensor([0.5, 0.25, 0.25])
    grid = torch.tensor([-0.5, 0.5, 1.5])
    out = _weighted_ecdf_on_grid(scores, w, grid)
    # at -0.5: 0; at 0.5: 0.5 (first point); at 1.5: 0.75
    expected = torch.tensor([0.0, 0.5, 0.75])
    torch.testing.assert_close(out, expected)


def test_weighted_split_matches_weighted_quantile() -> None:
    """Adapter threshold equals _weighted_quantile at the raw target level."""
    torch.manual_seed(0)
    scores = torch.rand(20)
    w = torch.rand(20)
    alpha = 0.1
    expected = _weighted_quantile(scores, 1.0 - alpha, w)
    ad = WeightedSplitConformalAdapter(alpha=alpha)
    ad.calibrate(scores, w)
    assert ad.threshold_ is not None
    torch.testing.assert_close(ad.threshold_, expected)


def test_uniform_weights_match_unweighted_quantile() -> None:
    scores = torch.tensor([0.1, 0.4, 0.2, 0.9])
    w = torch.ones_like(scores)
    alpha = 0.1
    # Augmented (n+1) distribution reproduces the exact order statistic.
    k = min(math.ceil((scores.numel() + 1) * (1.0 - alpha)), scores.numel())
    expected = torch.sort(scores).values[k - 1]
    ad = WeightedSplitConformalAdapter(alpha=alpha)
    ad.calibrate(scores, w)
    assert float(ad.threshold_) == float(expected)


def test_ot_reweighter_weights_simplex() -> None:
    torch.manual_seed(1)
    cal = torch.randn(30)
    tgt = torch.randn(25) * 0.5 + 0.2
    rw = ScoreCDFReweighter(n_steps=80, learning_rate=0.1, entropy_penalty=1e-2).fit(cal, tgt)
    assert rw.weights_ is not None
    assert rw.weights_.shape == (30,)
    torch.testing.assert_close(rw.weights_.sum(), torch.tensor(1.0))
    assert (rw.weights_ >= 0).all()


def test_ot_reweighter_near_uniform_on_same_distribution() -> None:
    torch.manual_seed(2)
    cal = torch.randn(40)
    tgt = torch.randn(35)
    rw = ScoreCDFReweighter(n_steps=120, learning_rate=0.08, entropy_penalty=0.5).fit(cal, tgt)
    u = torch.full_like(rw.weights_, 1.0 / rw.weights_.numel())
    cos = (rw.weights_ * u).sum() / (rw.weights_.norm() * u.norm())
    assert float(cos.item()) > 0.85


def test_coverage_gap_smaller_when_same_pool() -> None:
    torch.manual_seed(3)
    s = torch.randn(100)
    gap_same = OptimalTransportCoverageGap().estimate(calibration_scores=s, target_score_summary=s)[
        "l2_cdf_gap"
    ]
    t = torch.randn(100) * 3.0 + 1.0
    gap_shift = OptimalTransportCoverageGap().estimate(
        calibration_scores=s, target_score_summary=t
    )["l2_cdf_gap"]
    assert gap_same < 1e-5
    assert gap_shift > 0.05


def test_predict_from_test_scores_shape() -> None:
    scores = torch.linspace(0, 1, 10)
    ad = WeightedSplitConformalAdapter(alpha=0.2)
    ad.calibrate(scores, torch.ones_like(scores))
    cand = torch.randn(5, 4)
    m = ad.predict_from_test_scores(cand)
    assert m.shape == cand.shape
    assert m.dtype == torch.bool


def test_bad_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        WeightedSplitConformalAdapter(alpha=1.0)


def test_public_module_exports() -> None:
    assert hasattr(tr.test_time, "ScoreCDFReweighter")
    assert hasattr(tr.test_time, "WeightedSplitConformalAdapter")
    assert hasattr(tr.test_time, "OptimalTransportCoverageGap")
