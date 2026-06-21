"""
Unit tests for torchregress.causal.diagnostics.
"""

from __future__ import annotations

import pytest
import torch

from torchregress.causal.diagnostics import (
    _effective_sample_size,
    causal_overlap_report,
)


class TestEffectiveSampleSize:
    def test_uniform_weights(self) -> None:
        """Uniform weights give ESS = n."""
        w = torch.ones(100)
        ess = _effective_sample_size(w)
        assert ess == pytest.approx(100.0)

    def test_concentrated_weights(self) -> None:
        """Concentrated weights give ESS < n."""
        w = torch.tensor([10.0] + [0.0] * 99)
        ess = _effective_sample_size(w)
        assert ess == pytest.approx(1.0)

    def test_random_weights_below_n(self) -> None:
        """Random positive weights give ESS < sum of weights."""
        w = torch.rand(50) * 5
        ess = _effective_sample_size(w)
        assert ess > 0
        assert ess < float(w.sum().item())

    def test_zero_weights(self) -> None:
        """All-zero weights give ESS = 0."""
        w = torch.zeros(20)
        ess = _effective_sample_size(w)
        assert ess == pytest.approx(0.0)


class TestCausalOverlapReport:
    def test_basic_report(self) -> None:
        """Produce overlap report for balanced data."""
        n = 100
        propensity = torch.full((n,), 0.5)
        treatment = torch.cat([torch.ones(n // 2), torch.zeros(n // 2)])
        report = causal_overlap_report(propensity, treatment)
        assert report["n_samples"] == n
        assert report["n_treated"] == n // 2
        assert report["n_control"] == n // 2
        assert 0 <= report["overlap_rate"] <= 1

    def test_clamp_propensity(self) -> None:
        """Propensities near 0 or 1 are clamped to [eps, 1-eps]."""
        propensity = torch.tensor([0.0, 0.5, 1.0])
        treatment = torch.tensor([1.0, 0.0, 1.0])
        report = causal_overlap_report(propensity, treatment)
        assert report["propensity_min"] > 0  # clamped up from 0
        assert report["propensity_max"] < 1  # clamped down from 1

    def test_trim_threshold(self) -> None:
        """trim_threshold controls which samples are trimmed."""
        propensity = torch.linspace(0, 1, 100)
        treatment = torch.randint(0, 2, (100,)).float()
        r1 = causal_overlap_report(propensity, treatment, trim_threshold=0.05)
        r2 = causal_overlap_report(propensity, treatment, trim_threshold=0.20)
        # Higher threshold trims more
        assert r2["n_trimmed"] >= r1["n_trimmed"]

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched shapes raise ValueError."""
        propensity = torch.rand(10)
        treatment = torch.rand(20)
        with pytest.raises(ValueError, match="share shape"):
            causal_overlap_report(propensity, treatment)

    def test_invalid_trim_threshold_raises(self) -> None:
        """trim_threshold >= 0.5 raises ValueError."""
        with pytest.raises(ValueError, match="trim_threshold"):
            causal_overlap_report(torch.rand(10), torch.rand(10), trim_threshold=0.5)

    def test_effective_sample_sizes_finite(self) -> None:
        """ESS values are >= 0."""
        propensity = torch.rand(50)
        treatment = torch.randint(0, 2, (50,)).float()
        report = causal_overlap_report(propensity, treatment)
        assert report["treated_ess"] >= 0
        assert report["control_ess"] >= 0
        assert report["min_group_ess"] >= 0

    def test_empty_treatment_arm(self) -> None:
        """All-control data produces valid report."""
        propensity = torch.full((20,), 0.3)
        treatment = torch.zeros(20)
        report = causal_overlap_report(propensity, treatment)
        assert report["n_treated"] == 0.0
        assert report["treated_ess"] == 0.0
