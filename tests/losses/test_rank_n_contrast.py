"""Tests for RankNContrastLoss against a brute-force reference implementation."""

from __future__ import annotations

import math

import pytest
import torch

from torchregress.losses import RankNContrastLoss, rank_n_contrast_loss


def _reference(features: torch.Tensor, labels: torch.Tensor, tau: float) -> torch.Tensor:
    """Literal transcription of Zha et al. (2023), eq. (2), with L2 similarity."""
    n = features.shape[0]
    out = torch.zeros(n, dtype=features.dtype)
    for i in range(n):
        acc = 0.0
        for j in range(n):
            if j == i:
                continue
            d_ij = (labels[i] - labels[j]).abs().sum()
            num = -torch.dist(features[i], features[j]) / tau
            den_terms = []
            for k in range(n):
                if k == i:
                    continue
                if (labels[i] - labels[k]).abs().sum() >= d_ij:
                    den_terms.append(-torch.dist(features[i], features[k]) / tau)
            den = torch.logsumexp(torch.stack(den_terms), 0)
            acc = acc - (num - den)
        out[i] = acc / (n - 1)
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_matches_bruteforce_reference(seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(9, 4, generator=g, dtype=torch.float64)
    y = torch.rand(9, generator=g, dtype=torch.float64) * 3
    ours = rank_n_contrast_loss(f, y, temperature=2.0)
    ref = _reference(f, y, 2.0)
    assert torch.allclose(ours, ref, atol=1e-10)


def test_ties_in_label_distance_are_included_in_denominator() -> None:
    g = torch.Generator().manual_seed(3)
    f = torch.randn(8, 3, generator=g, dtype=torch.float64)
    y = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0], dtype=torch.float64)
    ours = rank_n_contrast_loss(f, y, temperature=1.0)
    ref = _reference(f, y, 1.0)
    assert torch.allclose(ours, ref, atol=1e-10)


def test_perfectly_ordered_features_score_lower_than_shuffled() -> None:
    y = torch.linspace(0.0, 1.0, 32, dtype=torch.float64)
    ordered = (10.0 * y)[:, None]  # feature distance proportional to label distance
    shuffled = ordered[torch.randperm(32, generator=torch.Generator().manual_seed(0))]
    lo = rank_n_contrast_loss(ordered, y, temperature=1.0).mean()
    hi = rank_n_contrast_loss(shuffled, y, temperature=1.0).mean()
    assert lo < hi


def test_invariance_to_label_shift_and_feature_translation() -> None:
    g = torch.Generator().manual_seed(4)
    f = torch.randn(12, 5, generator=g, dtype=torch.float64)
    y = torch.randn(12, generator=g, dtype=torch.float64)
    a = rank_n_contrast_loss(f, y)
    b = rank_n_contrast_loss(f + 3.0, y + 100.0)
    assert torch.allclose(a, b, atol=1e-12)


def test_module_reduction_mask_and_weights() -> None:
    g = torch.Generator().manual_seed(5)
    f = torch.randn(10, 4, generator=g, dtype=torch.float64)
    y = torch.randn(10, generator=g, dtype=torch.float64)
    loss = RankNContrastLoss(temperature=1.5, reduction="none")
    per = loss(f, y)
    assert per.shape == (10,)
    assert torch.allclose(RankNContrastLoss(temperature=1.5)(f, y), per.mean())
    assert torch.allclose(RankNContrastLoss(temperature=1.5, reduction="sum")(f, y), per.sum())

    # Masked samples are removed from anchors *and* denominators: equals the
    # loss computed on the sub-batch.
    mask = torch.tensor([1, 1, 0, 1, 1, 0, 1, 1, 1, 0], dtype=torch.bool)
    masked = loss(f, y, mask=mask)
    sub = rank_n_contrast_loss(f[mask], y[mask], temperature=1.5)
    assert torch.all(masked[~mask] == 0)
    assert torch.allclose(masked[mask], sub, atol=1e-12)
    assert torch.allclose(RankNContrastLoss(temperature=1.5)(f, y, mask=mask), sub.mean())

    w = torch.rand(10, generator=g, dtype=torch.float64)
    weighted = RankNContrastLoss(temperature=1.5, reduction="sum")(f, y, weights=w)
    assert torch.allclose(weighted, (per * w).sum())


def test_cosine_similarity_and_multidim_labels() -> None:
    g = torch.Generator().manual_seed(6)
    f = torch.randn(7, 3, generator=g)
    y = torch.randn(7, 2, generator=g)
    out = RankNContrastLoss(similarity="cosine", reduction="none")(f, y)
    assert out.shape == (7,) and torch.isfinite(out).all()
    with pytest.raises(ValueError):
        RankNContrastLoss(similarity="dot")


def test_gradients_and_degenerate_batches() -> None:
    f = torch.randn(6, 3, requires_grad=True)
    y = torch.randn(6)
    RankNContrastLoss()(f, y).backward()
    assert f.grad is not None and torch.isfinite(f.grad).all()
    assert rank_n_contrast_loss(torch.randn(1, 3), torch.randn(1)).shape == (1,)
    # Single positive pair: -log(1) = 0 when the only other sample is the only negative.
    two = rank_n_contrast_loss(torch.randn(2, 3), torch.randn(2))
    assert torch.allclose(two, torch.zeros(2), atol=1e-7)
    assert math.isfinite(
        float(
            RankNContrastLoss()(torch.randn(2, 3), torch.randn(2), mask=torch.tensor([True, False]))
        )
    )
