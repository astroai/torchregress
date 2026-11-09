"""Tests for the sample-based (multivariate) highest-density calibration metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchregress.metrics import sample_hpd_coverage, sample_hpd_level


def _gaussian_log_prob(y: torch.Tensor, scale: float) -> torch.Tensor:
    dist = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros_like(y), scale), 1
    )
    return dist.log_prob(y)


def test_level_counts_higher_density_draws_with_half_ties() -> None:
    target = torch.tensor([0.0, 1.0])
    samples = torch.tensor([[1.0, -1.0, 0.0, 2.0], [0.0, 0.5, -3.0, -2.0]])
    # row 0: two draws above, one tie -> (2 + 0.5) / 4; row 1: none above.
    torch.testing.assert_close(sample_hpd_level(target, samples), torch.tensor([0.625, 0.0]))


def test_numpy_inputs_are_accepted() -> None:
    levels = sample_hpd_level(np.array([0.0]), np.array([[1.0, -1.0]]))
    torch.testing.assert_close(levels, torch.tensor([0.5], dtype=levels.dtype))


@pytest.mark.parametrize("dim", [1, 3, 6])
def test_calibrated_joint_predictive_has_nominal_coverage(dim: int) -> None:
    torch.manual_seed(dim)
    n, s = 4000, 256
    y = torch.randn(n, dim)
    draws = torch.randn(n, s, dim)
    coverage = sample_hpd_coverage(
        _gaussian_log_prob(y, 1.0), _gaussian_log_prob(draws, 1.0), credibility=(0.5, 0.9)
    )
    assert float((coverage - torch.tensor([0.5, 0.9])).abs().max()) < 0.025


def test_miscalibration_direction() -> None:
    torch.manual_seed(11)
    n, s, dim = 3000, 256, 4
    y = torch.randn(n, dim)
    for scale, expect_under in ((0.6, True), (1.6, False)):
        draws = scale * torch.randn(n, s, dim)
        coverage = sample_hpd_coverage(
            _gaussian_log_prob(y, scale), _gaussian_log_prob(draws, scale), credibility=0.9
        )
        if expect_under:
            assert float(coverage) < 0.75  # over-confident: regions too small
        else:
            assert float(coverage) > 0.98  # under-confident: regions too large


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match=r"\[N\]"):
        sample_hpd_level(torch.zeros(2, 1), torch.zeros(2, 3))
    with pytest.raises(ValueError, match=r"\[N, S\]"):
        sample_hpd_level(torch.zeros(2), torch.zeros(3, 3))
    with pytest.raises(ValueError, match="at least one"):
        sample_hpd_level(torch.zeros(2), torch.zeros(2, 0))
    with pytest.raises(ValueError, match="finite"):
        sample_hpd_level(torch.tensor([float("nan")]), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="credibility"):
        sample_hpd_coverage(torch.zeros(2), torch.zeros(2, 3), credibility=1.5)
