"""Unit tests for torchregress.causal.diagnostics."""

from __future__ import annotations

import pytest
import torch

from torchregress.causal.diagnostics import _effective_sample_size, causal_overlap_report

# ── _effective_sample_size ──────────────────────────────────────────────────


def test_ess_uniform_weights() -> None:
    """ESS of uniform weights equals the number of samples."""
    w = torch.ones(50)
    ess = _effective_sample_size(w)
    assert ess == pytest.approx(50.0)


def test_ess_single_nonzero_weight() -> None:
    """ESS with exactly one non-zero weight is 1."""
    w = torch.tensor([0.0, 10.0, 0.0])
    ess = _effective_sample_size(w)
    assert ess == pytest.approx(1.0)


def test_ess_decreases_with_imbalance() -> None:
    """ESS is smaller for unequal weights than for uniform."""
    w_uniform = torch.ones(100)
    w_skewed = torch.cat([torch.ones(50), torch.full((50,), 0.01)])
    ess_uniform = _effective_sample_size(w_uniform)
    ess_skewed = _effective_sample_size(w_skewed)
    assert ess_skewed < ess_uniform


def test_ess_all_zeros() -> None:
    """ESS of all-zero weights is 0 (denom clamped to 1e-8)."""
    w = torch.zeros(10)
    ess = _effective_sample_size(w)
    # sum(w)^2 = 0, so ESS = 0
    assert ess == pytest.approx(0.0)


def test_ess_1d_and_2d_same() -> None:
    """ESS is invariant to shape (uses reshape)."""
    w1 = torch.tensor([1.0, 2.0, 3.0])
    w2 = torch.tensor([[1.0, 2.0, 3.0]])
    assert _effective_sample_size(w1) == pytest.approx(_effective_sample_size(w2))


# ── causal_overlap_report ───────────────────────────────────────────────────


def test_overlap_report_all_fields_present() -> None:
    """All 12 expected keys are in the report."""
    p = torch.sigmoid(torch.randn(200)).clamp(0.05, 0.95)
    t = torch.bernoulli(p)
    report = causal_overlap_report(p, t)
    expected_keys = {
        "n_samples",
        "n_treated",
        "n_control",
        "propensity_min",
        "propensity_max",
        "propensity_mean",
        "overlap_rate",
        "trim_threshold",
        "n_trimmed",
        "treated_ess",
        "control_ess",
        "min_group_ess",
    }
    assert set(report.keys()) == expected_keys


def test_overlap_report_counts() -> None:
    """n_treated + n_control == n_samples."""
    p = torch.tensor([0.3, 0.7, 0.3, 0.7, 0.5])
    t = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    report = causal_overlap_report(p, t)
    assert report["n_treated"] == 3.0
    assert report["n_control"] == 2.0
    assert report["n_samples"] == 5.0


def test_overlap_report_propensity_bounds() -> None:
    """Report captures min/max/mean of clamped propensity."""
    p = torch.tensor([0.0, 0.5, 1.0])
    t = torch.tensor([0.0, 1.0, 0.0])
    report = causal_overlap_report(p, t, eps=1e-6)
    # 0.0 clamped to eps, 1.0 clamped to 1-eps
    assert report["propensity_min"] == pytest.approx(1e-6)
    assert report["propensity_max"] == pytest.approx(1.0 - 1e-6)


def test_overlap_report_overlap_rate() -> None:
    """overlap_rate is the fraction inside [trim_threshold, 1-trim_threshold]."""
    p = torch.tensor([0.02, 0.06, 0.5, 0.94, 0.98])
    t = torch.ones(5)  # all treated for simplicity
    report = causal_overlap_report(p, t, trim_threshold=0.05)
    # Inside [0.05, 0.95]: indices 1,2,3 → 3/5 = 0.6
    assert report["overlap_rate"] == pytest.approx(0.6)
    assert report["n_trimmed"] == pytest.approx(2.0)


def test_overlap_report_trim_threshold_zero() -> None:
    """trim_threshold=0 means no trimming."""
    p = torch.tensor([0.01, 0.5, 0.99])
    t = torch.ones(3)
    report = causal_overlap_report(p, t, trim_threshold=0.0)
    assert report["overlap_rate"] == pytest.approx(1.0)
    assert report["n_trimmed"] == pytest.approx(0.0)


def test_overlap_report_trim_threshold_49() -> None:
    """trim_threshold=0.49 (just under 0.5) is valid."""
    p = torch.tensor([0.5, 0.5, 0.5])
    t = torch.ones(3)
    report = causal_overlap_report(p, t, trim_threshold=0.49)
    assert report["overlap_rate"] == pytest.approx(1.0)


def test_overlap_report_invalid_trim_threshold_raises() -> None:
    """trim_threshold must be in [0, 0.5)."""
    p = torch.rand(10)
    t = torch.randint(0, 2, (10,))
    with pytest.raises(ValueError, match="trim_threshold must be in"):
        causal_overlap_report(p, t, trim_threshold=0.5)
    with pytest.raises(ValueError, match="trim_threshold must be in"):
        causal_overlap_report(p, t, trim_threshold=-0.1)


def test_overlap_report_shape_mismatch_raises() -> None:
    """Propensity and treatment must share shape."""
    p = torch.rand(10)
    t = torch.rand(12)
    with pytest.raises(ValueError, match="must share shape"):
        causal_overlap_report(p, t)


def test_overlap_report_all_treated() -> None:
    """ESS is computed correctly when all samples are treated."""
    p = torch.tensor([0.3, 0.5, 0.7])
    t = torch.ones(3)
    report = causal_overlap_report(p, t)
    assert report["n_treated"] == 3.0
    assert report["n_control"] == 0.0
    assert report["control_ess"] == 0.0
    assert report["treated_ess"] > 0.0
    assert report["min_group_ess"] == 0.0


def test_overlap_report_all_control() -> None:
    """ESS is computed correctly when all samples are control."""
    p = torch.tensor([0.3, 0.5, 0.7])
    t = torch.zeros(3)
    report = causal_overlap_report(p, t)
    assert report["n_control"] == 3.0
    assert report["n_treated"] == 0.0
    assert report["treated_ess"] == 0.0
    assert report["control_ess"] > 0.0
    assert report["min_group_ess"] == 0.0


def test_overlap_report_balanced_propensity() -> None:
    """Perfect overlap when all propensities are 0.5."""
    p = torch.full((100,), 0.5)
    t = torch.cat([torch.ones(50), torch.zeros(50)])
    report = causal_overlap_report(p, t)
    assert report["overlap_rate"] == pytest.approx(1.0)
    assert report["treated_ess"] == pytest.approx(50.0)
    assert report["control_ess"] == pytest.approx(50.0)


def test_overlap_report_ess_scales_with_sample_size() -> None:
    """Doubling sample size approximately doubles ESS for same distribution."""
    p = torch.full((100,), 0.5)
    t = torch.ones(100)
    report_100 = causal_overlap_report(p, t)
    p2 = torch.full((200,), 0.5)
    t2 = torch.ones(200)
    report_200 = causal_overlap_report(p2, t2)
    assert report_200["treated_ess"] == pytest.approx(2.0 * report_100["treated_ess"])


def test_overlap_report_default_eps() -> None:
    """Default eps=1e-6 clamps propensity away from 0 and 1."""
    p = torch.tensor([0.0, 1.0])
    t = torch.tensor([0.0, 1.0])
    report = causal_overlap_report(p, t)
    assert report["propensity_min"] > 0.0
    assert report["propensity_max"] < 1.0
