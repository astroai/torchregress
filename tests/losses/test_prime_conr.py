"""Tests for the representation-level imbalanced-regression losses (PRIME and ConR)."""

from __future__ import annotations

import pytest
import torch

from torchregress.losses.loss_registry import create_loss_from_config, loss_registry
from torchregress.losses.prime import (
    ConR,
    PrimeLoss,
    conr_loss,
    uniform_proxy_targets,
)

# ── PRIME: proxy targets ─────────────────────────────────────────────────────


def test_uniform_proxy_targets_are_equal_width_and_exclude_endpoints() -> None:
    targets = uniform_proxy_targets(0.0, 10.0, 5)
    assert targets.shape == (5,)
    assert torch.allclose(targets, torch.linspace(0.0, 10.0, 7)[1:-1])
    assert torch.allclose(targets.diff(), torch.full((4,), 10.0 / 6.0))
    assert float(targets[0]) > 0.0 and float(targets[-1]) < 10.0

    with pytest.raises(ValueError):
        uniform_proxy_targets(1.0, 0.0, 3)
    with pytest.raises(ValueError):
        uniform_proxy_targets(0.0, 1.0, 0)


def test_proxy_parameter_and_target_buffers_are_exposed() -> None:
    loss = PrimeLoss(feature_dim=4, n_proxies=6, label_range=(0.0, 1.0), seed=0)
    assert isinstance(loss.proxies, torch.nn.Parameter)
    assert loss.proxies.shape == (6, 4)
    assert "proxies" in dict(loss.named_parameters())
    assert "proxy_targets" in dict(loss.named_buffers())
    assert "proxy_weights" in dict(loss.named_buffers())
    assert loss.proxy_targets.shape == (6,)
    assert torch.allclose(loss.proxy_weights, torch.ones(6))

    state = loss.state_dict()
    assert "proxies" in state and "proxy_targets" in state


def test_fit_sets_uniform_grid_over_training_range() -> None:
    generator = torch.Generator().manual_seed(0)
    # Strongly skewed target distribution: the proxy targets should still be a
    # uniform grid over the observed range (paper's "uniformly distributed in
    # the target space" design), not empirical quantiles.
    targets = torch.cat([torch.rand(300, generator=generator) * 0.2, torch.rand(10) * 0.1 + 0.9])
    loss = PrimeLoss(feature_dim=3, n_proxies=8, seed=0)
    assert not loss.is_fitted
    returned = loss.fit(targets)
    assert returned is loss
    assert loss.is_fitted
    assert loss.proxy_targets.shape == (loss.n_proxies,)
    expected = torch.linspace(float(targets.min()), float(targets.max()), loss.n_proxies + 2)[1:-1]
    assert torch.allclose(loss.proxy_targets, expected, atol=1e-6)
    assert bool((loss.proxy_targets.diff() > 0).all())


def test_label_range_pins_grid_and_allows_forward_without_fit() -> None:
    pinned = PrimeLoss(feature_dim=4, n_proxies=5, label_range=(0.0, 1.0), seed=0)
    assert pinned.is_fitted
    assert torch.allclose(pinned.proxy_targets, uniform_proxy_targets(0.0, 1.0, 5))

    generator = torch.Generator().manual_seed(1)
    features = torch.randn(11, 4, generator=generator)
    labels = torch.rand(11, generator=generator)
    assert torch.isfinite(pinned(features, labels))

    # fit() keeps the pinned grid but can recompute the class-balanced weights.
    pinned.fit(labels * 0.1)
    assert torch.allclose(pinned.proxy_targets, uniform_proxy_targets(0.0, 1.0, 5))

    unpinned = PrimeLoss(feature_dim=4, n_proxies=5, seed=0)
    with pytest.raises(RuntimeError, match="fit"):
        unpinned(features, labels)


# ── PRIME: loss behaviour ────────────────────────────────────────────────────


def test_forward_is_finite_differentiable_and_proxies_receive_gradients() -> None:
    loss = PrimeLoss(feature_dim=4, n_proxies=6, label_range=(0.0, 1.0), seed=0)
    generator = torch.Generator().manual_seed(2)
    features = torch.randn(13, 4, generator=generator, requires_grad=True)
    labels = torch.rand(13, generator=generator)
    value = loss(features, labels)
    assert value.ndim == 0 and torch.isfinite(value)
    value.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert loss.proxies.grad is not None and torch.isfinite(loss.proxies.grad).all()
    assert float(loss.proxies.grad.norm()) > 0.0


def test_proxy_geometry_is_minimised_by_target_ordered_proxies() -> None:
    n_proxies = 8
    # Match temperatures so that a 1-D proxy layout with the same spacing as the
    # proxy targets makes Q == P, i.e. the KL term is exactly zero.
    tau_f, tau_t = 1.0, 5.0
    common = dict(
        feature_dim=2,
        n_proxies=n_proxies,
        label_range=(0.0, 1.0),
        seed=0,
        temperature=tau_f,
        proxy_temperature=tau_t,
        alignment_weight=0.0,
        spread_weight=0.0,
    )
    loss = PrimeLoss(**common)
    proxy_targets = loss.proxy_targets
    assert torch.allclose(proxy_targets.diff(), proxy_targets.diff()[0])

    layout = torch.stack([(tau_t / tau_f) * proxy_targets, torch.zeros(n_proxies)], dim=1)
    loss.proxies.data.copy_(layout)
    ordered = loss(torch.zeros(3, 2), torch.zeros(3))

    loss.proxies.data.copy_(torch.ones(n_proxies, 2))  # all proxies collapse
    collapsed = loss(torch.zeros(3, 2), torch.zeros(3))

    assert float(ordered) == pytest.approx(0.0, abs=1e-5)
    assert float(collapsed) > 0.05
    assert float(ordered) < float(collapsed)


def test_spread_term_rewards_separated_proxies() -> None:
    n_proxies = 8
    angles = torch.linspace(0.0, 1.0, n_proxies + 1)[:-1] * (2.0 * torch.pi)
    layout = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    common = dict(
        feature_dim=2,
        n_proxies=n_proxies,
        label_range=(0.0, 1.0),
        seed=0,
        temperature=1.0,
        proxy_temperature=5.0,
        alignment_weight=0.0,
    )
    no_spread = PrimeLoss(spread_weight=0.0, **common)
    with_spread = PrimeLoss(spread_weight=1.0, **common)
    for loss in (no_spread, with_spread):
        loss.proxies.data.copy_(layout)
    dummy = torch.zeros(3, 2)
    labels = torch.zeros(3)
    assert float(with_spread(dummy, labels)) < float(no_spread(dummy, labels))


def test_target_ordered_features_score_lower_than_collapsed_features() -> None:
    n_proxies = 8
    tau = 5.0
    loss = PrimeLoss(
        feature_dim=4,
        n_proxies=n_proxies,
        label_range=(0.0, 1.0),
        seed=0,
        temperature=tau,
        proxy_temperature=tau,
    )
    proxy_targets = loss.proxy_targets
    layout = torch.stack([tau * proxy_targets] + [torch.zeros(n_proxies)] * 3, dim=1)
    loss.proxies.data.copy_(layout)

    generator = torch.Generator().manual_seed(3)
    labels = torch.linspace(0.0, 1.0, 64)
    nearest = (labels[:, None] - proxy_targets[None, :]).abs().argmin(dim=1)
    ordered = loss.proxies[nearest].detach() + 0.001 * torch.randn(64, 4, generator=generator)
    collapsed = torch.zeros_like(ordered)

    assert float(loss(ordered, labels)) < float(loss(collapsed, labels))


def test_alignment_loss_decreases_when_features_are_optimised_towards_proxies() -> None:
    loss = PrimeLoss(
        feature_dim=6,
        n_proxies=8,
        label_range=(0.0, 1.0),
        seed=0,
        temperature=5.0,
        proxy_temperature=5.0,
        proxy_weight=0.0,  # isolate the alignment term
    )
    generator = torch.Generator().manual_seed(4)
    features = torch.nn.Parameter(torch.randn(16, 6, generator=generator))
    labels = torch.linspace(0.0, 1.0, 16)
    optimizer = torch.optim.Adam([features], lr=0.05)
    history = []
    for _ in range(40):
        optimizer.zero_grad()
        value = loss(features, labels)
        value.backward()
        optimizer.step()
        history.append(float(value.detach()))
    assert history[-1] < history[0]
    assert history[-1] < 0.9 * history[0]


def test_reduction_mask_and_weights_follow_the_shared_contract() -> None:
    generator = torch.Generator().manual_seed(5)
    features = torch.randn(9, 3, generator=generator)
    labels = torch.rand(9, generator=generator)
    common = dict(feature_dim=3, n_proxies=5, label_range=(0.0, 1.0), seed=7)

    per_anchor = PrimeLoss(reduction="none", **common)(features, labels)
    assert per_anchor.shape == (9,)
    mean_loss = PrimeLoss(reduction="mean", **common)
    sum_loss = PrimeLoss(reduction="sum", **common)
    assert torch.allclose(mean_loss(features, labels), per_anchor.mean())
    assert torch.allclose(sum_loss(features, labels), per_anchor.sum())

    mask = torch.tensor([1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=torch.bool)
    valid = mask.bool()
    masked_per_anchor = PrimeLoss(reduction="none", **common)(features, labels, mask=mask)
    assert torch.all(masked_per_anchor[~valid] == 0.0)
    assert torch.allclose(masked_per_anchor[valid], per_anchor[valid])
    assert torch.allclose(
        mean_loss(features, labels, mask=mask), per_anchor[valid].mean(), atol=1e-6
    )

    weights = torch.rand(9, generator=generator)
    expected = (per_anchor * weights).sum() / weights.sum()
    assert torch.allclose(mean_loss(features, labels, weights=weights), expected, atol=1e-6)
    # Column-vector mask/weights are accepted like every other loss in the library.
    assert torch.allclose(
        mean_loss(features, labels, mask=mask[:, None]),
        per_anchor[valid].mean(),
        atol=1e-6,
    )


def test_class_balanced_weights_upweight_rare_proxy_targets() -> None:
    generator = torch.Generator().manual_seed(6)
    targets = torch.cat([torch.rand(200, generator=generator) * 0.2, torch.rand(8) * 0.2 + 0.8])
    plain = PrimeLoss(feature_dim=3, n_proxies=8, seed=0).fit(targets)
    balanced = PrimeLoss(feature_dim=3, n_proxies=8, seed=0, class_balanced=True).fit(targets)

    assert torch.allclose(plain.proxy_weights, torch.ones(8))
    assert not torch.allclose(balanced.proxy_weights, torch.ones(8))
    proxy_targets = balanced.proxy_targets
    rare_index = int((proxy_targets - 0.98).abs().argmin())
    common_index = int((proxy_targets - 0.1).abs().argmin())
    assert float(balanced.proxy_weights[rare_index]) > 1.0
    assert float(balanced.proxy_weights[rare_index]) > float(balanced.proxy_weights[common_index])

    # The class-balanced coefficients enter the alignment term, so they change
    # the loss value on the same batch.
    features = torch.randn(12, 3, generator=generator)
    labels = torch.rand(12, generator=generator)
    common = dict(feature_dim=3, n_proxies=8, seed=0)
    loss_plain = PrimeLoss(**common).fit(targets)
    loss_balanced = PrimeLoss(class_balanced=True, **common).fit(targets)
    assert not torch.allclose(loss_plain(features, labels), loss_balanced(features, labels))


def test_permutation_invariance_and_batch_independent_proxy_term() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(11, 4, generator=generator)
    labels = torch.rand(11, generator=generator)
    loss = PrimeLoss(feature_dim=4, n_proxies=6, label_range=(0.0, 1.0), seed=0)
    permutation = torch.randperm(11, generator=generator)
    assert torch.allclose(
        loss(features, labels), loss(features[permutation], labels[permutation]), atol=1e-6
    )

    geometry_only = PrimeLoss(
        feature_dim=4,
        n_proxies=6,
        label_range=(0.0, 1.0),
        seed=0,
        alignment_weight=0.0,
    )
    other = torch.randn(5, 4, generator=generator)
    other_labels = torch.rand(5, generator=generator)
    assert torch.allclose(geometry_only(features, labels), geometry_only(other, other_labels))


def test_invalid_inputs_raise() -> None:
    loss = PrimeLoss(feature_dim=4, n_proxies=5, label_range=(0.0, 1.0), seed=0)
    with pytest.raises(ValueError, match="feature dim"):
        loss(torch.randn(4, 3), torch.rand(4))
    with pytest.raises(ValueError, match="\\[N, D\\]"):
        loss(torch.randn(4), torch.rand(4))
    with pytest.raises(ValueError, match="scalar labels"):
        loss(torch.randn(4, 4), torch.rand(4, 2))
    with pytest.raises(ValueError, match="mask"):
        loss(torch.randn(4, 4), torch.rand(4), mask=torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="n_proxies"):
        PrimeLoss(feature_dim=4, n_proxies=0)
    with pytest.raises(ValueError, match="label_range"):
        PrimeLoss(feature_dim=4, n_proxies=4, label_range=(1.0, 1.0))
    with pytest.raises(RuntimeError, match="fit"):
        PrimeLoss(feature_dim=4, n_proxies=4)(torch.randn(3, 4), torch.rand(3))


def test_prime_is_registered_for_config_factories() -> None:
    assert loss_registry["prime"] is PrimeLoss
    loss = create_loss_from_config({"type": "prime", "feature_dim": 5, "n_proxies": 4})
    assert isinstance(loss, PrimeLoss)
    assert loss.feature_dim == 5
    assert "conr" not in loss_registry  # ConR requires the prediction argument


# ── ConR: formula equality against a naive loop ──────────────────────────────


def _naive_conr_per_anchor(
    features: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    *,
    label_threshold: float,
    prediction_threshold: float | None = None,
    temperature: float = 0.07,
    push_scale: float = 0.2,
    density_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Literal transcription of Keramati et al. (2024), Eq. (1)-(2), per anchor."""
    n = features.shape[0]
    labels_2d = labels.reshape(n, -1).to(features.dtype)
    if labels_2d.shape[1] == 1:
        label_dist = (labels_2d - labels_2d.T).abs()
    else:
        label_dist = torch.cdist(labels_2d, labels_2d, p=2)
    preds = predictions.reshape(n, -1).mean(dim=-1).to(features.dtype)
    pred_dist = (preds[:, None] - preds[None, :]).abs()
    if prediction_threshold is None:
        prediction_threshold = label_threshold
    eta = (
        torch.ones(n, dtype=features.dtype)
        if density_weights is None
        else density_weights.reshape(n).to(features.dtype)
    )
    z = features / features.norm(dim=-1, keepdim=True)

    per_anchor = torch.zeros(n, dtype=features.dtype)
    for i in range(n):
        pos = [j for j in range(n) if j != i and float(label_dist[i, j]) <= label_threshold]
        neg = [
            j
            for j in range(n)
            if j != i
            and float(label_dist[i, j]) > label_threshold
            and float(pred_dist[i, j]) <= prediction_threshold
        ]
        if not pos or not neg:
            continue
        logits_pos = torch.stack([(z[i] @ z[j]) / temperature for j in pos])
        logits_neg = torch.stack(
            [
                torch.log(label_dist[i, q] * eta[i] * push_scale) + (z[i] @ z[q]) / temperature
                for q in neg
            ]
        )
        log_denom = torch.logsumexp(torch.cat([logits_pos, logits_neg]), 0)
        per_anchor[i] = (-(logits_pos - log_denom)).sum() / len(pos)
    return per_anchor


def test_conr_matches_naive_loop() -> None:
    generator = torch.Generator().manual_seed(11)
    features = torch.randn(8, 5, generator=generator, dtype=torch.float64)
    labels = torch.rand(8, generator=generator, dtype=torch.float64) * 3.0
    predictions = torch.randn(8, generator=generator, dtype=torch.float64)
    density = torch.rand(8, generator=generator, dtype=torch.float64) + 0.5

    for temperature in (0.07, 0.5):
        for use_density in (False, True):
            kwargs = dict(
                label_threshold=0.5,
                prediction_threshold=1.0,
                temperature=temperature,
                push_scale=0.3,
                density_weights=density if use_density else None,
            )
            ours = conr_loss(features, labels, predictions, **kwargs)
            reference = _naive_conr_per_anchor(features, labels, predictions, **kwargs)
            assert ours.shape == (8,)
            assert torch.allclose(ours, reference, atol=1e-12)
            assert float(reference.abs().sum()) > 0.0


def test_conr_matches_naive_loop_with_multidim_labels_and_predictions() -> None:
    generator = torch.Generator().manual_seed(12)
    features = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    labels = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    predictions = torch.randn(7, 2, generator=generator, dtype=torch.float64)
    kwargs = dict(
        label_threshold=1.5,
        prediction_threshold=1.2,
        temperature=0.2,
        push_scale=0.4,
    )
    ours = conr_loss(features, labels, predictions, **kwargs)
    reference = _naive_conr_per_anchor(features, labels, predictions, **kwargs)
    assert torch.allclose(ours, reference, atol=1e-12)


# ── ConR: pair selection, density weights, reductions ────────────────────────


def test_conr_anchor_without_negative_or_positive_pair_is_zero() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.0, 1.1], [-1.0, 0.0], [-1.0, 0.1]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0.0, 0.05, 0.5, 0.55, 2.0, 2.05], dtype=torch.float64)
    # Predictions are grouped within label clusters -> pairs across clusters are
    # label-distant but prediction-similar, i.e. negatives for clusters 0 and 1.
    predictions = torch.tensor([0.1, 0.1, 0.5, 0.5, 2.0, 2.0], dtype=torch.float64)
    per_anchor = conr_loss(
        features, labels, predictions, label_threshold=0.2, prediction_threshold=0.5
    )
    assert torch.all(per_anchor[4:] == 0.0)  # cluster 2 has no negative partner
    assert torch.all(per_anchor[:4] > 0.0)

    # With predictions equal to the labels there is no prediction-similar but
    # label-distant pair anywhere -> the regularizer is exactly zero.
    perfect = conr_loss(features, labels, labels, label_threshold=0.2)
    assert torch.all(perfect == 0.0)


def test_prediction_threshold_controls_negative_selection() -> None:
    generator = torch.Generator().manual_seed(13)
    features = torch.randn(10, 3, generator=generator, dtype=torch.float64)
    labels = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
    # All predictions equal: every label-distant pair is a negative pair.
    collapsed_preds = torch.zeros(10, dtype=torch.float64)
    loosened = conr_loss(
        features, labels, collapsed_preds, label_threshold=0.15, prediction_threshold=0.0
    )
    assert float(loosened.mean()) > 0.0
    # A strict prediction threshold finds no negative pairs when predictions are
    # perfectly ordered with the labels.
    assert float(conr_loss(features, labels, labels, label_threshold=0.15).mean()) == 0.0
    # Predictions spread far apart: no prediction-similar pairs -> zero loss.
    spread_preds = torch.linspace(0.0, 10.0, 10, dtype=torch.float64)
    assert float(conr_loss(features, labels, spread_preds, label_threshold=0.15).mean()) == 0.0


def test_conr_density_weights_favour_rare_labels() -> None:
    generator = torch.Generator().manual_seed(14)
    train_targets = torch.cat(
        [torch.rand(1000, generator=generator) * 0.2, torch.rand(30) * 0.1 + 0.9]
    )
    conr = ConR().fit(train_targets, n_bins=10)
    assert conr.is_fitted
    edges = conr.bin_edges
    common_bin = int(torch.searchsorted(edges, torch.tensor(0.1), right=True) - 1)
    rare_bin = int(torch.searchsorted(edges, torch.tensor(0.95), right=True) - 1)
    assert float(conr.bin_weights[rare_bin]) > float(conr.bin_weights[common_bin])

    # Fitted density weights boost the pushing power of minority anchors.
    labels = torch.cat([torch.full((5,), 0.1), torch.full((2,), 0.95)]).to(torch.float64)
    features = torch.randn(7, 4, generator=generator, dtype=torch.float64)
    predictions = torch.zeros(7, dtype=torch.float64)  # every pair is prediction-similar
    config = dict(label_threshold=0.2)
    plain = ConR(reduction="none", **config)(features, labels, prediction=predictions)
    fitted_loss = ConR(reduction="none", **config).fit(train_targets, n_bins=10)
    fitted = fitted_loss(features, labels, prediction=predictions)
    assert torch.isfinite(fitted).all()
    assert torch.all(fitted[5:] > plain[5:])  # minority anchors are pushed harder

    # Passing per-sample weights overrides the fitted eta (all-ones restores the
    # unfitted pushing power).
    override = ConR(**config).fit(train_targets, n_bins=10)(
        features, labels, prediction=predictions, weights=torch.ones(7, dtype=torch.float64)
    )
    assert torch.allclose(override, plain.mean(), atol=1e-6)


def test_conr_mask_reduction_and_weights() -> None:
    generator = torch.Generator().manual_seed(15)
    features = torch.randn(10, 4, generator=generator, dtype=torch.float64)
    labels = torch.rand(10, generator=generator, dtype=torch.float64)
    predictions = torch.rand(10, generator=generator, dtype=torch.float64)
    common = dict(label_threshold=0.5, prediction_threshold=1.0)

    per_anchor = ConR(reduction="none", **common)(features, labels, prediction=predictions)
    assert per_anchor.shape == (10,)
    mean_loss = ConR(reduction="mean", **common)
    sum_loss = ConR(reduction="sum", **common)
    assert torch.allclose(mean_loss(features, labels, prediction=predictions), per_anchor.mean())
    assert torch.allclose(sum_loss(features, labels, prediction=predictions), per_anchor.sum())

    mask = torch.tensor([1, 1, 0, 1, 1, 1, 0, 1, 1, 1], dtype=torch.bool)
    keep = mask.bool()
    expected = conr_loss(features[keep], labels[keep], predictions[keep], **common).mean()
    assert torch.allclose(
        mean_loss(features, labels, prediction=predictions, mask=mask), expected, atol=1e-12
    )

    weights = torch.rand(10, generator=generator, dtype=torch.float64)
    weighted = mean_loss(features, labels, prediction=predictions, weights=weights)
    # weights double as pushing powers eta: compare against the direct reference.
    reference = _naive_conr_per_anchor(
        features,
        labels,
        predictions,
        label_threshold=0.5,
        prediction_threshold=1.0,
        density_weights=weights,
    )
    assert torch.allclose(weighted, (reference * weights).sum() / weights.sum(), atol=1e-12)


def test_conr_extreme_similarities_are_stable() -> None:
    generator = torch.Generator().manual_seed(16)
    features = torch.randn(12, 6, generator=generator, dtype=torch.float64) * 1e3
    labels = torch.rand(12, generator=generator, dtype=torch.float64) * 3.0
    predictions = torch.zeros(12, dtype=torch.float64)
    for temperature in (1e-3, 1e3):
        value = ConR(temperature=temperature, label_threshold=0.5)(
            features, labels, prediction=predictions
        )
        assert torch.isfinite(value)
    features = features.clone().requires_grad_(True)
    value = ConR(label_threshold=0.5)(features, labels, prediction=predictions)
    value.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_conr_permutation_invariance() -> None:
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(9, 3, generator=generator, dtype=torch.float64)
    labels = torch.rand(9, generator=generator, dtype=torch.float64)
    predictions = torch.rand(9, generator=generator, dtype=torch.float64)
    loss = ConR(label_threshold=0.4, prediction_threshold=0.8)
    permutation = torch.randperm(9, generator=generator)
    assert torch.allclose(
        loss(features, labels, prediction=predictions),
        loss(
            features[permutation],
            labels[permutation],
            prediction=predictions[permutation],
        ),
        atol=1e-12,
    )


def test_conr_collapsed_minority_features_score_higher_than_ordered_features() -> None:
    generator = torch.Generator().manual_seed(18)
    # Imbalanced labels: six majority samples near 0, two minority samples at 1.
    labels = torch.cat([torch.zeros(6), torch.ones(2)]).to(torch.float64)
    predictions = torch.zeros(8, dtype=torch.float64)  # collapsed predictions
    loss = ConR(label_threshold=0.2)

    # Ordered: unit vectors whose angle grows with the label (same label ->
    # same direction).  Collapsed: random directions at the same scale.
    angles = labels * (torch.pi / 2.0)
    ordered = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    collapsed = torch.randn(8, 4, generator=generator, dtype=torch.float64)
    assert float(loss(ordered, labels, prediction=predictions)) < float(
        loss(collapsed, labels, prediction=predictions)
    )


def test_conr_requires_prediction_and_validates_shapes() -> None:
    loss = ConR()
    features = torch.randn(6, 3)
    labels = torch.rand(6)
    with pytest.raises(ValueError, match="prediction"):
        loss(features, labels)
    with pytest.raises(ValueError):
        loss(torch.randn(6), labels, prediction=labels)
    with pytest.raises(ValueError):
        loss(features, labels[:-1], prediction=labels)
    with pytest.raises(ValueError):
        loss(features, labels, prediction=labels, mask=torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError):
        ConR(temperature=0.0)
    with pytest.raises(ValueError):
        ConR().fit(torch.rand(10), n_bins=1)
    with pytest.raises(RuntimeError):
        ConR().bin_weights


def test_conr_helpers_are_consistent() -> None:
    # The functional form and the module must agree.
    generator = torch.Generator().manual_seed(19)
    features = torch.randn(7, 4, generator=generator)
    labels = torch.rand(7, generator=generator)
    predictions = torch.zeros(7)
    module = ConR(label_threshold=0.3, prediction_threshold=0.5)
    functional = conr_loss(
        features, labels, predictions, label_threshold=0.3, prediction_threshold=0.5
    )
    assert torch.allclose(module(features, labels, prediction=predictions), functional.mean())
