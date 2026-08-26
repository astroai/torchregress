"""Workstream B audit-fix regression tests (B1-B3).

Covers:
- ``finite_sample_quantile`` identity vs brute-force order statistic (TR-MET-07).
- Mondrian predict_interval rejects unseen group ids instead of silently
  producing zero-width intervals (TR-LOSS-22).
- CQR ``debias`` shrinkage removal: debias=True is a no-op (B3 / TR-MET-07).
- CV+ finite-sample order statistics replace linear-interpolated quantiles.
"""

import math

import pytest
import torch

from torchregress.losses.conformal import (
    CQR,
    CVPlus,
    SplitConformal,
    finite_sample_quantile,
)

# ---------------------------------------------------------------------------
# B1 / TR-MET-07 — finite_sample_quantile semantics
# ---------------------------------------------------------------------------


def test_finite_sample_quantile_plan_spot_check() -> None:
    """Plan spot check: n=9, alpha=0.2 -> k=ceil(10*0.8)=8 -> sorted[7]=8.0."""
    scores = torch.arange(1.0, 10.0)
    assert float(finite_sample_quantile(scores, 0.2)) == 8.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_finite_sample_quantile_matches_order_statistic(seed: int) -> None:
    """Result equals the k-th smallest score with k = ceil((n+1)*(1-alpha))."""
    generator = torch.Generator().manual_seed(seed)
    for n in (1, 2, 5, 17, 100):
        scores = torch.rand(n, generator=generator) * 10
        for alpha in (0.05, 0.1, 0.25, 0.5):
            k = min(math.ceil((n + 1) * (1.0 - alpha)), n)
            expected = torch.sort(scores).values[k - 1]
            assert torch.isclose(finite_sample_quantile(scores, alpha), expected)


def test_finite_sample_quantile_never_exceeds_max() -> None:
    """Small n with tiny alpha must clamp k to n, not index out of range."""
    scores = torch.tensor([3.0, 1.0, 2.0])
    assert float(finite_sample_quantile(scores, 1e-9)) == 3.0


def test_finite_sample_quantile_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="alpha"):
        finite_sample_quantile(torch.randn(10), 0.0)
    with pytest.raises(ValueError, match="empty"):
        finite_sample_quantile(torch.empty(0), 0.1)


# ---------------------------------------------------------------------------
# B2 / TR-LOSS-22 — Mondrian unseen group validation
# ---------------------------------------------------------------------------


def _calibrate_mondrian(alpha_groups: torch.Tensor) -> SplitConformal:
    predictor = SplitConformal(alpha=0.1)
    y_cal = torch.randn(60, 1)
    y_pred = torch.randn(60, 1)
    predictor.calibrate(y_pred, y_cal, groups=alpha_groups)
    return predictor


def test_mondrian_unseen_int_group_raises() -> None:
    """Unseen int id below max previously returned q_hat=0 (zero-width interval)."""
    predictor = _calibrate_mondrian(torch.tensor([0, 2] * 30))
    # Group id 1 lies strictly between calibrated keys {0, 2}: pre-fix LUT
    # zero-fill produced q_hat=0 and zero-width intervals.
    with pytest.raises(ValueError, match="unseen group id\\(s\\) \\[1\\].*PrevalenceAdjustedCP"):
        predictor.predict_interval(torch.randn(4, 1), groups=torch.tensor([0, 1, 2, 1]))


def test_mondrian_unseen_int_group_above_max_raises() -> None:
    predictor = _calibrate_mondrian(torch.tensor([0, 1] * 30))
    with pytest.raises(ValueError, match="unseen group id\\(s\\) \\[5\\]"):
        predictor.predict_interval(torch.randn(3, 1), groups=torch.tensor([0, 5, 1]))


def test_mondrian_unseen_float_key_raises() -> None:
    """Float-key searchsorted path previously read uninitialized memory."""
    predictor = _calibrate_mondrian(torch.tensor([0.5, 1.5] * 30).float())
    with pytest.raises(ValueError, match="unseen group id\\(s\\)"):
        predictor.predict_interval(torch.randn(3, 1), groups=torch.tensor([0.5, 9.5, 0.5]))


def test_mondrian_known_groups_still_predict() -> None:
    predictor = _calibrate_mondrian(torch.tensor([0, 1, 2] * 20))
    lo, hi = predictor.predict_interval(torch.randn(6, 1), groups=torch.tensor([0, 1, 2, 0, 1, 2]))
    assert torch.isfinite(lo).all() and torch.isfinite(hi).all()
    assert (hi > lo).all()


# ---------------------------------------------------------------------------
# B3 — CQR debias removal
# ---------------------------------------------------------------------------


def test_cqr_debias_is_noop() -> None:
    torch.manual_seed(11)
    n_cal = 80
    y_cal = torch.randn(n_cal, 1)
    y_pred = torch.cat([y_cal - 0.3 * torch.rand(n_cal, 1), y_cal + 0.3 * torch.rand(n_cal, 1)], -1)

    cqr_on = CQR(alpha=0.1, debias=True)
    cqr_off = CQR(alpha=0.1, debias=False)
    cqr_on.calibrate(y_pred, y_cal)
    cqr_off.calibrate(y_pred, y_cal)
    assert torch.isclose(cqr_on.q_hat, cqr_off.q_hat)


def test_cqr_q_hat_matches_finite_sample_rule() -> None:
    """Threshold is exactly the ceil((n+1)*(1-alpha)) order statistic of CQR scores."""
    torch.manual_seed(12)
    n_cal, alpha = 40, 0.2
    y_cal = torch.randn(n_cal, 1)
    lo = y_cal - torch.rand(n_cal, 1)
    hi = y_cal + torch.rand(n_cal, 1)
    cqr = CQR(alpha=alpha)
    cqr.calibrate(torch.cat([lo, hi], -1), y_cal)

    scores = torch.maximum(lo - y_cal, y_cal - hi).squeeze(-1)
    k = min(math.ceil((n_cal + 1) * (1.0 - alpha)), n_cal)
    expected = torch.sort(scores).values[k - 1]
    assert torch.isclose(cqr.q_hat.squeeze(), expected)


# ---------------------------------------------------------------------------
# B3 — CV+ finite-sample ranks
# ---------------------------------------------------------------------------


def test_cvplus_uses_order_statistics_not_interpolation() -> None:
    """Upper bound must be an actual candidate value at rank ceil((n+1)*(1-a))."""
    torch.manual_seed(13)
    n_cal, n_test, alpha = 20, 4, 0.25
    cv = CVPlus(alpha=alpha)
    residuals = torch.abs(torch.randn(n_cal)) + 0.1
    fold_indices = torch.randint(0, 3, (n_cal,))
    cv.calibrate_ensemble(torch.randn(n_cal, 1) + 5.0, torch.randn(n_cal, 1), fold_indices)
    cv.residuals = residuals

    member_preds = torch.randn(3, n_test, 1) + 5.0
    lo, hi = cv.predict_interval(member_preds)

    res_unsq = residuals.view(-1, 1, 1)
    pred_per_cal = member_preds[fold_indices]
    upper_candidates = (pred_per_cal + res_unsq).squeeze(-1)
    lower_candidates = (pred_per_cal - res_unsq).squeeze(-1)
    k_up = min(math.ceil((n_cal + 1) * (1.0 - alpha)), n_cal)
    k_lo = min(math.ceil((n_cal + 1) * alpha), n_cal)
    expected_hi = torch.sort(upper_candidates, dim=0).values[k_up - 1]
    expected_lo = torch.sort(lower_candidates, dim=0).values[k_lo - 1]
    assert torch.allclose(hi.squeeze(-1), expected_hi)
    assert torch.allclose(lo.squeeze(-1), expected_lo)
