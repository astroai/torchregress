"""Tests for DiscreteWasserstein1Loss against scipy's 1D optimal transport."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import wasserstein_distance

from torchregress.losses import DiscreteWasserstein1Loss, discrete_wasserstein1


def _rand_masses(n: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(n, k, generator=g, dtype=torch.float64) * 2, dim=-1)


def test_matches_scipy_on_irregular_grid() -> None:
    edges = torch.tensor([0.0, 0.2, 0.5, 1.1, 1.3, 2.0, 3.5], dtype=torch.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = _rand_masses(5, 6, 0)
    q = _rand_masses(5, 6, 1)
    ours = discrete_wasserstein1(p, q, centers)
    for i in range(5):
        ref = wasserstein_distance(centers.numpy(), centers.numpy(), p[i].numpy(), q[i].numpy())
        assert abs(ours[i].item() - ref) < 1e-12


def test_point_mass_targets_and_translation() -> None:
    edges = torch.linspace(0.0, 4.0, 9, dtype=torch.float64)  # width 0.5
    loss = DiscreteWasserstein1Loss(edges, reduction="none")
    p = torch.zeros(3, 8, dtype=torch.float64)
    p[:, 1] = 1.0  # mass at centre 0.75
    z = torch.tensor([0.75, 2.75, 3.9], dtype=torch.float64)  # bins 1, 5, 7
    out = loss(p, z)
    assert torch.allclose(out, torch.tensor([0.0, 2.0, 3.0], dtype=torch.float64))


def test_metric_properties() -> None:
    edges = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p, q, r = (_rand_masses(4, 10, s) for s in (2, 3, 4))
    assert torch.all(discrete_wasserstein1(p, p, centers) == 0)
    assert torch.allclose(
        discrete_wasserstein1(p, q, centers), discrete_wasserstein1(q, p, centers)
    )
    assert torch.all(
        discrete_wasserstein1(p, r, centers)
        <= discrete_wasserstein1(p, q, centers) + discrete_wasserstein1(q, r, centers) + 1e-12
    )


def test_logits_normalisation_and_reduction() -> None:
    edges = torch.linspace(0.0, 2.0, 5)
    logits = torch.randn(6, 4)
    target = _rand_masses(6, 4, 7).to(torch.float32)
    a = DiscreteWasserstein1Loss(edges, from_logits=True, reduction="none")(logits, target)
    b = DiscreteWasserstein1Loss(edges, reduction="none")(
        torch.softmax(logits, -1) * 5.0, target * 3.0
    )
    assert torch.allclose(a, b, atol=1e-6)
    mean = DiscreteWasserstein1Loss(edges, from_logits=True)(logits, target)
    assert mean.ndim == 0 and torch.allclose(mean, a.mean())
    mask = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.bool)
    masked = DiscreteWasserstein1Loss(edges, from_logits=True)(logits, target, mask=mask)
    assert torch.allclose(masked, a[mask].mean())
    w = torch.rand(6)
    weighted = DiscreteWasserstein1Loss(edges, from_logits=True, reduction="sum")(
        logits, target, weights=w
    )
    assert torch.allclose(weighted, (a * w).sum())


def test_default_unit_bins_and_validation() -> None:
    p = torch.tensor([[1.0, 0.0, 0.0]])
    q = torch.tensor([[0.0, 0.0, 1.0]])
    assert torch.allclose(DiscreteWasserstein1Loss()(p, q), torch.tensor(2.0))
    with pytest.raises(ValueError):
        DiscreteWasserstein1Loss([0.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        DiscreteWasserstein1Loss([0.0, 1.0])(p, q)  # 1 bin vs 3 columns
    with pytest.raises(ValueError):
        DiscreteWasserstein1Loss()(p, torch.zeros(2, 2))


def test_gradient_moves_mass_toward_target() -> None:
    edges = torch.linspace(0.0, 1.0, 6)
    logits = torch.zeros(1, 5, requires_grad=True)
    target = torch.tensor([0.9])  # last bin
    loss = DiscreteWasserstein1Loss(edges, from_logits=True)(logits, target)
    loss.backward()
    grad = logits.grad[0]
    # Increasing logits of far-left bins increases the loss; of the last bin decreases it.
    assert grad[0] > 0 and grad[-1] < 0
    assert np.all(np.diff(grad.numpy()) <= 1e-7)
