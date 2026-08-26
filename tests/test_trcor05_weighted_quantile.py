"""TR-COR-05 validation: _weighted_quantile includes the (n+1) test-point weight."""

from __future__ import annotations

import math

import torch

from torchregress.losses.conformal import (
    _weighted_quantile,
    finite_sample_quantile,
)


def test_augmented_uniform_weights_recover_finite_sample_quantile() -> None:
    """agy_issues Task 3.1 assertion: uniform weights == unweighted order statistic."""
    torch.manual_seed(0)
    scores = torch.randn(200)
    q = 0.9
    got = _weighted_quantile(scores, q, weights=torch.ones(200))
    expected = finite_sample_quantile(scores, 1.0 - q)
    assert float(got) == float(expected)


def test_augmented_threshold_is_order_statistic_index() -> None:
    """Uniform weights select exactly the ceil((n+1)(1-alpha))-th order statistic."""
    torch.manual_seed(1)
    n = 150
    scores = torch.randn(n)
    alpha = 0.2
    k = min(math.ceil((n + 1) * (1.0 - alpha)), n)
    got = float(_weighted_quantile(scores, 1.0 - alpha, weights=torch.ones(n)))
    assert got == float(torch.sort(scores).values[k - 1])


def test_nonuniform_weights_are_consistent() -> None:
    """Concentrated mass on a point pulls the threshold to that order statistic."""
    torch.manual_seed(2)
    n = 100
    scores = torch.linspace(0, 1, n)
    w = torch.zeros(n)
    w[-1] = 5.0  # all mass on the largest score
    # Augmented mass: w_max/(w_sum+1) = 5/6 = 0.833 < 0.9 -> next index carries
    # cum = 1.0 >= 0.9, but clamp keeps within range; with all remaining mass at
    # zero weights the CDF jumps from 0.833 to 1.0 only AT the last point.
    got = float(_weighted_quantile(scores, 0.9, weights=w))
    assert got == float(scores[-1])
