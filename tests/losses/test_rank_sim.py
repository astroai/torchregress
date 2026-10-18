"""Tests for the RankSim ranking-similarity regularizer (Gong et al., ICML 2022)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torchregress.losses.loss_registry import create_loss_from_config, loss_registry
from torchregress.losses.rank_sim import (
    RankSimLoss,
    _rank_normalised,
    rank_sim_regularizer,
)

# ── literal reference implementation ─────────────────────────────────────────


def _literal_rank(values: torch.Tensor) -> torch.Tensor:
    """Eq. (1): ``rk(a)_i = 1 + |{j : a_j > a_i}|``, normalised by the length."""
    n = values.numel()
    ranks = torch.tensor(
        [1 + int((values > values[i]).sum()) for i in range(n)], dtype=values.dtype
    )
    return ranks / n


def _literal_similarity(features: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "cosine":
        normalised = features / features.norm(dim=-1, keepdim=True)
        return normalised @ normalised.T
    return -torch.cdist(features, features, p=2)


def _literal_regularizer(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    duplicate_policy: str = "keep",
    similarity: str = "cosine",
    reduction: str = "mean",
) -> torch.Tensor:
    """Literal transcription of Eq. (1) and Eq. (4), independent of the module."""
    if duplicate_policy == "keep":
        rep_features, rep_labels = features, labels
    else:
        unique = torch.unique(labels, sorted=True)
        reps = []
        for value in unique:
            group = features[labels == value]
            reps.append(group.mean(dim=0) if duplicate_policy == "mean" else group[0])
        rep_features = torch.stack(reps)
        rep_labels = unique

    n = rep_labels.numel()
    sim = _literal_similarity(rep_features, similarity)
    total = features.new_zeros(())
    for i in range(n):
        label_ranks = _literal_rank(-(rep_labels[i] - rep_labels).abs())
        feature_ranks = _literal_rank(sim[i])
        total = total + ((feature_ranks - label_ranks) ** 2).mean()
    return total if reduction == "sum" else total / n


# ── ranking operation ────────────────────────────────────────────────────────


def test_competition_ranking_matches_the_paper_definition_with_ties() -> None:
    scores = torch.tensor(
        [
            [0.9, 0.1, 0.5, 0.5, 0.3],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, -1.0, -1.0, -2.0],
        ],
        dtype=torch.float64,
    )
    ranks = _rank_normalised(scores)
    n = scores.shape[1]
    assert ranks.shape == scores.shape
    assert torch.all(ranks > 0) and torch.all(ranks <= 1)
    for row in range(scores.shape[0]):
        assert torch.equal(ranks[row], _literal_rank(scores[row]))
    # Equal scores share a rank and the largest score always gets rank 1/n.
    assert ranks[1, 0] == pytest.approx(1.0 / n)
    assert torch.allclose(ranks[0, 2], ranks[0, 3])


def test_ordered_half_circle_features_have_zero_regularizer() -> None:
    target = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)
    angles = target * (torch.pi / 4.0)
    features = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    # Feature ranks exactly reproduce the label ranks for every anchor.
    assert float(rank_sim_regularizer(features, target)) == pytest.approx(0.0, abs=1e-12)
    assert float(rank_sim_regularizer(features, target, reduction="sum")) == pytest.approx(
        0.0, abs=1e-12
    )


def test_hand_computable_rank_similarity_value() -> None:
    # Unit vectors at 0, 36.9 and 90 degrees with labels 0, 1, 3.
    features = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=torch.float64)
    target = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64)

    # Ordered features score exactly zero; reversing the feature order leaves the
    # two end anchors perfectly ordered and the middle anchor swapped, so the
    # per-anchor values are 0, 2/27, 0 and the mean is 2/81.
    reversed_features = features.flip(0)
    assert float(rank_sim_regularizer(reversed_features, target)) == pytest.approx(
        2.0 / 81.0, abs=1e-12
    )
    assert float(rank_sim_regularizer(reversed_features, target, reduction="sum")) == (
        pytest.approx(2.0 / 27.0, abs=1e-12)
    )


def test_random_batches_match_the_literal_reference() -> None:
    generator = torch.Generator().manual_seed(0)
    for seed, n in enumerate((4, 6, 9)):
        generator.manual_seed(seed)
        features = torch.randn(n, 5, generator=generator, dtype=torch.float64)
        labels = torch.rand(n, generator=generator, dtype=torch.float64) * 3.0
        for similarity in ("cosine", "l2"):
            for policy in ("mean", "first", "keep"):
                ours = rank_sim_regularizer(
                    features, labels, similarity=similarity, duplicate_policy=policy
                )
                reference = _literal_regularizer(
                    features, labels, similarity=similarity, duplicate_policy=policy
                )
                assert torch.allclose(ours, reference, atol=1e-12)


def test_functional_per_anchor_values_match_the_literal_summands() -> None:
    generator = torch.Generator().manual_seed(1)
    features = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    labels = torch.tensor([0.0, 0.5, 0.5, 1.0, 2.0, 2.5, 4.0], dtype=torch.float64)
    per_anchor = rank_sim_regularizer(features, labels, reduction="none", duplicate_policy="first")
    # Only distinct-label representatives are anchors: indices 0, 1, 3, 4, 5, 6.
    assert per_anchor.shape == (6,)
    rep_features = torch.stack([features[i] for i in (0, 1, 3, 4, 5, 6)])
    rep_labels = labels[torch.tensor([0, 1, 3, 4, 5, 6])]
    sim = _literal_similarity(rep_features, "cosine")
    expected = []
    for i in range(6):
        label_ranks = _literal_rank(-(rep_labels[i] - rep_labels).abs())
        feature_ranks = _literal_rank(sim[i])
        expected.append(((feature_ranks - label_ranks) ** 2).mean())
    assert torch.allclose(per_anchor, torch.stack(expected), atol=1e-12)


# ── module API, gamma and reductions ─────────────────────────────────────────


def test_gamma_scales_the_returned_regularizer_once() -> None:
    generator = torch.Generator().manual_seed(2)
    features = torch.randn(11, 4, generator=generator)
    target = torch.rand(11, generator=generator)

    base = RankSimLoss(gamma=1.0)(features, target)
    unscaled = rank_sim_regularizer(features, target)
    assert torch.allclose(base, unscaled, atol=1e-6)
    assert torch.allclose(RankSimLoss(gamma=2.5)(features, target), 2.5 * unscaled, atol=1e-6)
    assert float(RankSimLoss(gamma=0.0)(features, target)) == 0.0


def test_sum_reduction_is_the_paper_loss_and_mean_is_its_per_anchor_average() -> None:
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(13, 6, generator=generator)
    target = torch.rand(13, generator=generator) * 5.0  # distinct labels, M = 13

    mean_loss = RankSimLoss(gamma=1.0)(features, target)
    sum_loss = RankSimLoss(gamma=1.0, reduction="sum")(features, target)
    assert torch.allclose(sum_loss, 13.0 * mean_loss, atol=1e-6)

    per_anchor = RankSimLoss(gamma=1.0, reduction="none")(features, target)
    assert per_anchor.shape == (13,)
    assert torch.allclose(mean_loss, per_anchor.mean(), atol=1e-6)
    assert torch.allclose(sum_loss, per_anchor.sum(), atol=1e-6)


def test_mask_removes_samples_from_anchors_and_rankings() -> None:
    generator = torch.Generator().manual_seed(4)
    features = torch.randn(12, 4, generator=generator)
    target = torch.rand(12, generator=generator)
    mask = torch.tensor([1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1], dtype=torch.bool)

    per_anchor = RankSimLoss(gamma=1.0, reduction="none")(features, target, mask=mask)
    assert per_anchor.shape == (12,)
    assert torch.all(per_anchor[~mask] == 0.0)

    sub = rank_sim_regularizer(features[mask], target[mask])
    assert torch.allclose(RankSimLoss(gamma=1.0)(features, target, mask=mask), sub, atol=1e-6)
    # Column-vector masks are accepted, as everywhere else in the library.
    assert torch.allclose(
        RankSimLoss(gamma=1.0)(features, target, mask=mask[:, None]), sub, atol=1e-6
    )


def test_weights_follow_the_shared_weighted_mean_contract() -> None:
    generator = torch.Generator().manual_seed(5)
    features = torch.randn(10, 3, generator=generator)
    target = torch.rand(10, generator=generator)
    weights = torch.rand(10, generator=generator)

    per_anchor = RankSimLoss(gamma=1.0, reduction="none")(features, target)
    expected = (per_anchor * weights).sum() / weights.sum()
    assert torch.allclose(
        RankSimLoss(gamma=1.0)(features, target, weights=weights), expected, atol=1e-6
    )
    # 'none' applies the sample weights elementwise, per the BaseLoss contract.
    weighted_none = RankSimLoss(gamma=1.0, reduction="none")(features, target, weights=weights)
    assert torch.allclose(weighted_none, per_anchor * weights, atol=1e-6)
    # Column-vector weights are accepted.
    assert torch.allclose(
        RankSimLoss(gamma=1.0)(features, target, weights=weights[:, None]), expected, atol=1e-6
    )


def test_weights_are_averaged_within_duplicate_label_groups() -> None:
    generator = torch.Generator().manual_seed(16)
    features = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    target = torch.tensor([0.0, 0.0, 1.0, 2.0], dtype=torch.float64)
    weights = torch.tensor([0.2, 0.6, 0.5, 0.9], dtype=torch.float64)

    # 'first' keeps indices 0, 2 and 3; the representative at index 0 inherits
    # the mean weight of its label group: (0.2 + 0.6) / 2 = 0.4.
    representative_terms = rank_sim_regularizer(
        features[[0, 2, 3]], target[[0, 2, 3]], reduction="none"
    )
    representative_weights = torch.tensor([0.4, 0.5, 0.9], dtype=torch.float64)
    expected = (representative_terms * representative_weights).sum() / representative_weights.sum()
    loss = RankSimLoss(gamma=1.0, duplicate_policy="first")
    assert torch.allclose(loss(features, target, weights=weights), expected, atol=1e-12)

    per_anchor = RankSimLoss(gamma=1.0, reduction="none", duplicate_policy="first")(
        features, target, weights=weights
    )
    expected_full = torch.zeros(4, dtype=torch.float64)
    expected_full[[0, 2, 3]] = representative_terms * representative_weights
    assert torch.allclose(per_anchor, expected_full, atol=1e-12)


def test_mask_and_weights_combine() -> None:
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(9, 4, generator=generator)
    target = torch.rand(9, generator=generator)
    mask = torch.tensor([1, 0, 1, 1, 0, 1, 1, 1, 0], dtype=torch.bool)
    weights = torch.rand(9, generator=generator)

    per_anchor = RankSimLoss(gamma=1.0, reduction="none")(features, target, mask=mask)
    expected = (per_anchor[mask] * weights[mask]).sum() / weights[mask].sum()
    assert torch.allclose(
        RankSimLoss(gamma=1.0)(features, target, mask=mask, weights=weights),
        expected,
        atol=1e-6,
    )


def test_column_targets_and_float64_are_supported() -> None:
    generator = torch.Generator().manual_seed(6)
    features = torch.randn(9, 4, generator=generator, dtype=torch.float64)
    target = torch.rand(9, 1, generator=generator, dtype=torch.float64)
    assert float(RankSimLoss()(features, target)) == pytest.approx(
        float(RankSimLoss()(features, target.reshape(9))), abs=1e-12
    )


# ── duplicate labels ─────────────────────────────────────────────────────────


def test_duplicate_label_policies() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(6, 4, generator=generator, dtype=torch.float64)
    target = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 4.0], dtype=torch.float64)

    kept = rank_sim_regularizer(features, target, duplicate_policy="keep")
    first = rank_sim_regularizer(features, target, duplicate_policy="first")
    averaged = rank_sim_regularizer(features, target, duplicate_policy="mean")
    assert kept != first and kept != averaged
    # 'first' keeps the first occurrence of the duplicated label (index 0).
    assert first == pytest.approx(
        float(_literal_regularizer(features, target, duplicate_policy="first")), abs=1e-12
    )
    assert averaged == pytest.approx(
        float(_literal_regularizer(features, target, duplicate_policy="mean")), abs=1e-12
    )

    per_anchor = RankSimLoss(gamma=1.0, reduction="none", duplicate_policy="first")(
        features, target
    )
    assert torch.all(per_anchor[1:3] == 0.0)  # duplicate copies are not anchors
    assert float(per_anchor[0]) > 0.0
    mean_over_reps = RankSimLoss(gamma=1.0, duplicate_policy="first")(features, target)
    assert torch.allclose(mean_over_reps, first, atol=1e-12)
    # 'none' zeros the duplicate copies, so its plain mean is diluted by them;
    # the 'mean' reduction averages over the retained anchors only.
    assert not torch.allclose(per_anchor.mean(), mean_over_reps, atol=1e-12)
    assert torch.allclose(per_anchor[per_anchor != 0].mean(), mean_over_reps, atol=1e-12)


def test_single_distinct_label_has_no_ordering_signal() -> None:
    generator = torch.Generator().manual_seed(8)
    features = torch.randn(8, 3, generator=generator)
    target = torch.full((8,), 2.5)
    assert float(rank_sim_regularizer(features, target)) == 0.0
    loss = RankSimLoss()
    assert float(loss(features, target)) == 0.0
    assert torch.isfinite(loss(features, target))
    # Fewer than two valid samples is also handled without a ranking call.
    mask = torch.zeros(8, dtype=torch.bool)
    mask[0] = True
    assert float(loss(features, target, mask=mask)) == 0.0


# ── invariance and equivariance ──────────────────────────────────────────────


def test_permutation_and_target_affine_invariance() -> None:
    generator = torch.Generator().manual_seed(9)
    features = torch.randn(10, 5, generator=generator, dtype=torch.float64)
    target = torch.rand(10, generator=generator, dtype=torch.float64)
    # Include duplicate labels to exercise the permutation-invariant 'mean' policy.
    target[1] = target[0]

    base = rank_sim_regularizer(features, target)
    permutation = torch.randperm(10, generator=generator)
    permuted = rank_sim_regularizer(features[permutation], target[permutation])
    assert torch.allclose(base, permuted, atol=1e-12)
    # Ranks depend on the ordering of distances only: positive affine target
    # transforms leave the regularizer unchanged.
    assert torch.allclose(base, rank_sim_regularizer(features, 3.0 * target + 5.0), atol=1e-12)


def test_feature_scale_rotation_and_similarity_choice() -> None:
    generator = torch.Generator().manual_seed(10)
    features = torch.randn(9, 3, generator=generator, dtype=torch.float64)
    target = torch.rand(9, generator=generator, dtype=torch.float64)

    cosine = rank_sim_regularizer(features, target, similarity="cosine")
    assert torch.allclose(cosine, rank_sim_regularizer(2.0 * features, target), atol=1e-12)
    # A rotation (orthogonal matrix) leaves all cosine similarities unchanged.
    matrix = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64)).Q
    rotated = rank_sim_regularizer(features @ matrix.T, target)
    assert torch.allclose(cosine, rotated, atol=1e-10)

    # Euclidean similarity is translation invariant instead.
    euclidean = rank_sim_regularizer(features, target, similarity="l2")
    shifted = rank_sim_regularizer(features + 7.0, target, similarity="l2")
    assert torch.allclose(euclidean, shifted, atol=1e-10)


# ── gradients ────────────────────────────────────────────────────────────────


def test_gradients_are_finite_and_permutation_equivariant() -> None:
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(14, 5, generator=generator, dtype=torch.float64)
    target = torch.rand(14, generator=generator, dtype=torch.float64)
    permutation = torch.randperm(14, generator=generator)

    def value_and_grad(
        data: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = data.clone().requires_grad_(True)
        value = RankSimLoss(gamma=1.0)(data, labels)
        value.backward()
        assert data.grad is not None
        return value.detach(), data.grad

    value, gradient = value_and_grad(features, target)
    permuted_value, permuted_gradient = value_and_grad(features[permutation], target[permutation])
    assert torch.isfinite(value)
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0
    assert torch.allclose(value, permuted_value, atol=1e-12)
    # f(P z, P y) = f(z, y)  =>  grad f(P z, P y) = P grad f(z, y).
    assert torch.allclose(permuted_gradient, gradient[permutation], atol=1e-10)


def test_interpolation_strength_changes_only_the_gradient() -> None:
    """The forward value is the hard rank; lambda only shapes the surrogate gradient."""
    generator = torch.Generator().manual_seed(21)
    features = torch.randn(32, 6, generator=generator, dtype=torch.float64)
    target = torch.rand(32, generator=generator, dtype=torch.float64)

    def value_and_grad(lam: float) -> tuple[torch.Tensor, torch.Tensor]:
        data = features.clone().requires_grad_(True)
        value = RankSimLoss(interpolation_strength=lam)(data, target)
        value.backward()
        assert data.grad is not None
        return value.detach(), data.grad

    coarse_value, coarse_gradient = value_and_grad(0.5)
    fine_value, fine_gradient = value_and_grad(2.0)
    assert torch.allclose(coarse_value, fine_value, atol=1e-12)
    assert torch.isfinite(coarse_gradient).all() and torch.isfinite(fine_gradient).all()
    assert not torch.allclose(coarse_gradient, fine_gradient, atol=1e-6)


def test_orthogonal_gradient_equivariance() -> None:
    generator = torch.Generator().manual_seed(12)
    features = torch.randn(12, 4, generator=generator, dtype=torch.float64)
    target = torch.rand(12, generator=generator, dtype=torch.float64)
    # An exact signed permutation avoids the floating-point roundoff of a general
    # rotation: the rank surrogate is piecewise constant in the similarities, so
    # tiny roundoff could legitimately flip a rank boundary and the gradient.
    permutation = torch.eye(4, dtype=torch.float64)[:, torch.randperm(4, generator=generator)]
    signs = torch.where(torch.rand(4, generator=generator) < 0.5, -1.0, 1.0)
    matrix = permutation * signs

    def gradient_at(data: torch.Tensor) -> torch.Tensor:
        data = data.clone().requires_grad_(True)
        RankSimLoss(gamma=1.0)(data, target).backward()
        assert data.grad is not None
        return data.grad

    gradient = gradient_at(features)
    rotated_gradient = gradient_at(features @ matrix.T)
    assert torch.isfinite(rotated_gradient).all()
    # f(z R^T) = f(z)  =>  grad f(z R^T) = grad f(z) R^T.
    assert torch.allclose(rotated_gradient, gradient @ matrix.T, atol=1e-12)


def test_optimising_features_reduces_the_regularizer() -> None:
    generator = torch.Generator().manual_seed(13)
    target = torch.rand(40, generator=generator, dtype=torch.float64)
    features = nn.Parameter(torch.randn(40, 6, generator=generator, dtype=torch.float64))
    loss_fn = RankSimLoss(gamma=1.0)
    optimizer = torch.optim.Adam([features], lr=0.05)

    history = []
    for _ in range(60):
        optimizer.zero_grad()
        value = loss_fn(features, target)
        value.backward()
        optimizer.step()
        history.append(float(value.detach()))
    assert history[-1] < 0.6 * history[0]
    assert torch.isfinite(features).all()


# ── validation and registry ──────────────────────────────────────────────────


def test_invalid_inputs_raise() -> None:
    loss = RankSimLoss()
    features = torch.randn(6, 3)
    target = torch.rand(6)

    with pytest.raises(ValueError, match=r"\[N, D\]"):
        loss(torch.randn(6), target)
    with pytest.raises(ValueError, match=r"\[N, 1\]"):
        loss(features, torch.rand(6, 2))
    with pytest.raises(ValueError, match=r"\[N, 1\]"):
        loss(features, torch.rand(5))
    with pytest.raises(ValueError, match="mask"):
        loss(features, target, mask=torch.ones(5, dtype=torch.bool))
    with pytest.raises(ValueError, match="weights"):
        loss(features, target, weights=torch.ones(5))
    with pytest.raises(ValueError, match="weights"):
        loss(features, target, weights=-torch.ones(6))
    with pytest.raises(ValueError, match="finite"):
        loss(features, target + float("nan"))
    with pytest.raises(ValueError, match="finite"):
        loss(features + float("inf"), target)
    with pytest.raises(ValueError, match="floating"):
        loss(torch.ones(6, 3, dtype=torch.long), target)
    with pytest.raises(ValueError, match="feature dimension"):
        loss(torch.randn(6, 0), target)
    with pytest.raises(ValueError, match=r"\[N, D\]"):
        rank_sim_regularizer(torch.randn(6), target)
    with pytest.raises(ValueError, match="floating"):
        rank_sim_regularizer(torch.ones(6, 3, dtype=torch.long), target)
    with pytest.raises(ValueError, match="feature dimension"):
        rank_sim_regularizer(torch.randn(6, 0), target)
    with pytest.raises(ValueError, match="reduction"):
        rank_sim_regularizer(features, target, reduction="median")
    with pytest.raises(ValueError, match="gamma"):
        RankSimLoss(gamma=-1.0)
    with pytest.raises(ValueError, match="interpolation_strength"):
        RankSimLoss(interpolation_strength=0.0)
    with pytest.raises(ValueError, match="similarity"):
        RankSimLoss(similarity="dot")
    with pytest.raises(ValueError, match="duplicate_policy"):
        RankSimLoss(duplicate_policy="drop")
    with pytest.raises(ValueError):
        RankSimLoss(reduction="median")


def test_registered_for_config_factories() -> None:
    assert loss_registry["rank_sim"] is RankSimLoss
    loss = create_loss_from_config({"type": "rank_sim", "gamma": 0.25})
    assert isinstance(loss, RankSimLoss)
    assert loss.gamma == 0.25


def test_module_is_deterministic_and_masked_nans_are_ignored() -> None:
    generator = torch.Generator().manual_seed(14)
    features = torch.randn(10, 4, generator=generator)
    target = torch.rand(10, generator=generator)
    loss = RankSimLoss()
    assert torch.equal(loss(features, target), loss(features, target))

    # Masked (possibly NaN) features/targets are excluded before finite checks.
    features_bad = features.clone()
    target_bad = target.clone()
    features_bad[0] = float("nan")
    target_bad[0] = float("nan")
    mask = torch.ones(10, dtype=torch.bool)
    mask[0] = False
    assert torch.isfinite(loss(features_bad, target_bad, mask=mask))


# ── synthetic imbalanced-regression comparison ──────────────────────────────


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    left_ranks = left.argsort().argsort().to(torch.float64)
    right_ranks = right.argsort().argsort().to(torch.float64)
    left_ranks = (left_ranks - left_ranks.mean()) / (left_ranks.std() + 1e-12)
    right_ranks = (right_ranks - right_ranks.mean()) / (right_ranks.std() + 1e-12)
    return float((left_ranks * right_ranks).mean())


def _feature_ordering(features: torch.Tensor, target: torch.Tensor) -> float:
    """Spearman correlation between label distance and cosine feature distance."""
    normalised = torch.nn.functional.normalize(features, dim=-1)
    feature_distance = 1.0 - normalised @ normalised.T
    label_distance = (target[:, None] - target[None, :]).abs()
    upper = torch.triu_indices(target.numel(), target.numel(), offset=1)
    return _spearman(feature_distance[upper[0], upper[1]], label_distance[upper[0], upper[1]])


def _train_half_circle(gamma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train a bottleneck MLP on a long-tailed half-circle target, return probe outputs."""
    torch.manual_seed(0)
    # 8% of the training targets live in the tail (y > 0.65); the rest crowd y < 0.45.
    n_train, n_tail = 240, 19
    generator = torch.Generator().manual_seed(0)
    target = torch.cat(
        [
            0.45 * torch.rand(n_train - n_tail, generator=generator) ** 1.2,
            0.65 + 0.35 * torch.rand(n_tail, generator=generator),
        ]
    )
    angle = torch.pi * target
    inputs = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)

    # Balanced probe: 101 uniformly spaced targets, as in the DIR benchmarks.
    probe_target = torch.linspace(0.0, 1.0, 101)
    probe_angle = torch.pi * probe_target
    probe_inputs = torch.stack([torch.cos(probe_angle), torch.sin(probe_angle)], dim=1)

    encoder = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 4), nn.SiLU())
    head = nn.Linear(4, 1)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-2)
    task_loss = nn.MSELoss()
    regularizer = RankSimLoss(gamma=gamma)
    for _ in range(200):
        optimizer.zero_grad()
        features = encoder(inputs)
        prediction = head(features).squeeze(-1)
        loss = task_loss(prediction, target)
        if gamma > 0.0:
            loss = loss + regularizer(features, target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probe_features = encoder(probe_inputs)
        probe_prediction = head(probe_features).squeeze(-1)
    return probe_prediction, probe_features, probe_target


def test_regularizer_improves_tail_error_and_feature_ordering() -> None:
    """RankSim helps a tail metric and the learned feature ordering on a tiny DIR task.

    The data are a long-tailed half-circle: 92% of the training targets crowd
    ``y < 0.45`` and only 8% populate the tail.  Both models are trained from the
    same seed for 200 full-batch Adam steps; the only difference is the RankSim
    term (``gamma = 1.0``).  Evaluation uses a balanced 101-point probe.
    """
    baseline_prediction, baseline_features, probe_target = _train_half_circle(gamma=0.0)
    ranksim_prediction, ranksim_features, _ = _train_half_circle(gamma=1.0)

    tail = probe_target > 0.6
    baseline_tail_mae = float((baseline_prediction[tail] - probe_target[tail]).abs().mean())
    ranksim_tail_mae = float((ranksim_prediction[tail] - probe_target[tail]).abs().mean())
    baseline_ordering = _feature_ordering(baseline_features, probe_target)
    ranksim_ordering = _feature_ordering(ranksim_features, probe_target)

    # The regularizer directly shapes the feature ordering ...
    assert ranksim_ordering > baseline_ordering + 0.1
    assert ranksim_ordering > 0.98
    # ... and, on this task, that also improves the tail prediction error.
    assert ranksim_tail_mae < 0.8 * baseline_tail_mae
    assert ranksim_tail_mae < baseline_tail_mae


def test_gamma_zero_contributes_exactly_zero_and_zero_gradient() -> None:
    """gamma=0 makes the regularizer a no-op in the total objective."""
    generator = torch.Generator().manual_seed(15)
    features = torch.randn(10, 4, generator=generator, requires_grad=True)
    target = torch.rand(10, generator=generator)
    value = RankSimLoss(gamma=0.0)(features, target)
    assert float(value) == 0.0
    value.backward()
    assert features.grad is not None
    assert torch.all(features.grad == 0.0)
