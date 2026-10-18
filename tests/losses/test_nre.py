"""Tests for classifier-based Neural Ratio Estimation (NRE) losses."""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchregress.losses.nre as nre_module
from torchregress.losses.nre import (
    NeuralRatioLoss,
    build_ratio_classifier,
    neural_ratio_logits,
    neural_ratio_loss,
)

SIGMA = 0.5
GRID = torch.linspace(-3.0, 3.0, 121).reshape(-1, 1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_classifier(dtype: torch.dtype = torch.float64) -> nn.Module:
    """Ratio classifier ``s = 1.5 * c - 2.0 * x + 0.25`` with known weights."""
    classifier = nn.Linear(2, 1, bias=True).to(dtype)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[1.5, -2.0]], dtype=dtype))
        classifier.bias.copy_(torch.tensor([0.25], dtype=dtype))
    return classifier


def _identity_ratio_classifier() -> nn.Module:
    """Ratio classifier ``s = c - x`` (zero logit for perfectly matched pairs)."""
    classifier = nn.Linear(2, 1, bias=True).double()
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[1.0, -1.0]], dtype=torch.float64))
        classifier.bias.zero_()
    return classifier


def _linear_logits(context: torch.Tensor, observable: torch.Tensor) -> torch.Tensor:
    """Reference implementation of the known linear classifier logits."""
    return 1.5 * context - 2.0 * observable + 0.25


def _formula_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed float64 inputs shared by the hand-computed formula tests."""
    context = torch.tensor([[0.5], [-1.0], [2.0], [0.0]], dtype=torch.float64)
    observable = torch.tensor([[0.25], [0.75], [-0.5], [1.5]], dtype=torch.float64)
    negative_context = torch.tensor(
        [
            [[-0.5], [1.0], [2.5]],
            [[0.0], [-1.5], [0.5]],
            [[1.0], [3.0], [-2.0]],
            [[-3.0], [2.0], [0.5]],
        ],
        dtype=torch.float64,
    )
    return context, observable, negative_context


def _reference_loss(positive: torch.Tensor, negative: torch.Tensor, criterion: str) -> torch.Tensor:
    """Independent (softplus / logsumexp) reference implementation of the criteria."""
    if criterion == "bce":
        # -log sigmoid(s+) - mean_k log sigmoid(-s-_k)
        return F.softplus(-positive) + F.softplus(negative).mean(dim=1)
    # -log softmax([s+, s-_1, ..., s-_K])_0
    return torch.logsumexp(torch.cat([positive.unsqueeze(1), negative], dim=1), dim=1) - positive


def _spearman_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Spearman rank correlation per row of two ``[N, K]`` tensors."""
    rank_a = a.argsort(dim=1).argsort(dim=1).to(a.dtype)
    rank_b = b.argsort(dim=1).argsort(dim=1).to(b.dtype)
    centered_a = rank_a - rank_a.mean(dim=1, keepdim=True)
    centered_b = rank_b - rank_b.mean(dim=1, keepdim=True)
    return (centered_a * centered_b).sum(dim=1) / (centered_a.norm(dim=1) * centered_b.norm(dim=1))


def _small_loss_fn(
    *,
    criterion: str = "bce",
    reduction: str = "none",
    context_dim: int = 1,
    observable_dim: int = 1,
    n: int = 4,
    k: int = 2,
    seed: int = 0,
) -> tuple[NeuralRatioLoss, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    classifier = build_ratio_classifier(context_dim, observable_dim, hidden_dims=(8,))
    loss_fn = NeuralRatioLoss(classifier, criterion=criterion, reduction=reduction)
    context = torch.randn(n, context_dim)
    observable = torch.randn(n, observable_dim)
    negative_context = torch.randn(n, k, context_dim)
    return loss_fn, context, observable, negative_context


def _train_gaussian_ratio_estimator(
    criterion: str,
    *,
    seed: int = 0,
    sigma: float = SIGMA,
    n_train: int = 1024,
    n_negatives: int = 8,
    steps: int = 400,
    hidden_dims: tuple[int, ...] = (32, 32),
    lr: float = 3e-3,
) -> tuple[nn.Module, NeuralRatioLoss, float, float]:
    """Train an NRE classifier on ``c ~ N(0, 1)``, ``x | c ~ N(c, sigma^2)``.

    Negative contexts are drawn from the same prior, independently of the
    observation, which is the reference-prior sampling required for the
    classifier logit to estimate ``log p(x | c) - log p(x)``.
    """
    torch.manual_seed(seed)
    classifier = build_ratio_classifier(1, 1, hidden_dims=hidden_dims)
    loss_fn = NeuralRatioLoss(classifier, criterion=criterion)
    optimiser = torch.optim.Adam(classifier.parameters(), lr=lr)

    first, last = [], []
    for step in range(steps):
        context = torch.randn(n_train, 1)
        observable = context + sigma * torch.randn(n_train, 1)
        negative_context = torch.randn(n_train, n_negatives, 1)
        loss = loss_fn(context, observable, negative_context=negative_context)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step < 20:
            first.append(float(loss.detach()))
        if step >= steps - 20:
            last.append(float(loss.detach()))
    return classifier, loss_fn, sum(first) / len(first), sum(last) / len(last)


def _candidate_scores(
    classifier: nn.Module,
    observable: torch.Tensor,
    grid: torch.Tensor = GRID,
) -> torch.Tensor:
    """Score every grid candidate for every observation: ``[N, K]`` log-ratios."""
    n_observations = observable.shape[0]
    n_candidates = grid.shape[0]
    context_flat = grid.unsqueeze(0).expand(n_observations, n_candidates, 1).reshape(-1, 1)
    observable_flat = observable.unsqueeze(1).expand(n_observations, n_candidates, 1).reshape(-1, 1)
    with torch.no_grad():
        return neural_ratio_logits(classifier, context_flat, observable_flat).reshape(
            n_observations, n_candidates
        )


def _formula_loss_fn(criterion: str = "bce", reduction: str = "none") -> NeuralRatioLoss:
    return NeuralRatioLoss(_linear_classifier(), criterion=criterion, reduction=reduction)


# ---------------------------------------------------------------------------
# build_ratio_classifier
# ---------------------------------------------------------------------------


def test_build_ratio_classifier_shapes_and_metadata() -> None:
    classifier = build_ratio_classifier(2, 3, hidden_dims=(16, 8), activation="relu")
    assert isinstance(classifier, nn.Module)
    logits = classifier(torch.zeros(5, 5))
    assert logits.shape == (5, 1)
    assert classifier.context_dim == 2
    assert classifier.observable_dim == 3
    assert classifier.input_dim == 5
    assert sum(isinstance(module, nn.Linear) for module in classifier.modules()) == 3


def test_build_ratio_classifier_linear_and_activations() -> None:
    linear = build_ratio_classifier(1, 1, hidden_dims=())
    assert sum(isinstance(module, nn.Linear) for module in linear.modules()) == 1
    assert linear(torch.zeros(4, 2)).shape == (4, 1)

    for activation, activation_type in [
        ("silu", nn.SiLU),
        ("gelu", nn.GELU),
        ("tanh", nn.Tanh),
        ("elu", nn.ELU),
        ("leaky_relu", nn.LeakyReLU),
    ]:
        module = build_ratio_classifier(1, 1, hidden_dims=(4,), activation=activation)
        assert any(isinstance(layer, activation_type) for layer in module)


def test_build_ratio_classifier_validation_errors() -> None:
    with pytest.raises(ValueError, match="context_dim"):
        build_ratio_classifier(0, 1)
    with pytest.raises(ValueError, match="observable_dim"):
        build_ratio_classifier(1, -2)
    with pytest.raises(TypeError, match="hidden_dims"):
        build_ratio_classifier(1, 1, hidden_dims=16)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hidden_dims"):
        build_ratio_classifier(1, 1, hidden_dims=(16, 0))
    with pytest.raises(ValueError, match="activation"):
        build_ratio_classifier(1, 1, activation="swish")


# ---------------------------------------------------------------------------
# Exact hand-computed loss formulas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("criterion", ["bce", "multiclass"])
def test_loss_matches_hand_computed_formula(criterion: str) -> None:
    context, observable, negative_context = _formula_inputs()
    positive = _linear_logits(context, observable).squeeze(-1)
    negative = _linear_logits(negative_context, observable.unsqueeze(1)).squeeze(-1)
    expected = _reference_loss(positive, negative, criterion)

    classifier = _linear_classifier()
    loss_fn = NeuralRatioLoss(classifier, criterion=criterion, reduction="none")
    actual = loss_fn(context, observable, negative_context=negative_context)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)

    for reduction, expected_reduced in [
        ("mean", expected.mean()),
        ("sum", expected.sum()),
    ]:
        reduced_fn = NeuralRatioLoss(classifier, criterion=criterion, reduction=reduction)
        reduced = reduced_fn(context, observable, negative_context=negative_context)
        torch.testing.assert_close(reduced, expected_reduced, rtol=0, atol=1e-12)


def test_loss_prefers_matching_context() -> None:
    # With s = c - x, a matched pair has logit 0 and a mismatched pair is penalised.
    loss_fn = NeuralRatioLoss(_identity_ratio_classifier())
    context = torch.tensor([[0.0], [0.5], [1.0], [1.5]], dtype=torch.float64)
    observable = context.clone()
    # Good: matched positive hypotheses; negatives shifted by -2 (logit -2).
    good_loss = loss_fn(context, observable, negative_context=(context - 2.0).unsqueeze(1))
    # Bad: mismatched positive hypotheses (logit -2); matched negatives (logit 0).
    bad_loss = loss_fn(context - 2.0, observable, negative_context=context.unsqueeze(1))
    assert float(good_loss) < float(bad_loss)


def test_log_likelihood_ratio_sign_reflects_hypothesis_preference() -> None:
    loss_fn = NeuralRatioLoss(_identity_ratio_classifier())
    context = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    observable = context.clone()
    # Positive hypothesis matched (logit 0); negatives shifted by -2 (logit -2).
    ratios = loss_fn.log_likelihood_ratio(
        context, observable, negative_context=(context - 2.0).unsqueeze(1)
    )
    torch.testing.assert_close(
        ratios,
        torch.full_like(ratios, 2.0),
        rtol=0,
        atol=1e-12,
    )
    # A mismatched positive hypothesis loses against matched negatives.
    ratios_bad = loss_fn.log_likelihood_ratio(
        context - 2.0, observable, negative_context=context.unsqueeze(1)
    )
    torch.testing.assert_close(
        ratios_bad,
        torch.full_like(ratios_bad, -2.0),
        rtol=0,
        atol=1e-12,
    )


def test_invalid_criterion_and_classifier_type() -> None:
    with pytest.raises(TypeError, match="classifier"):
        NeuralRatioLoss("not-a-module")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="criterion"):
        NeuralRatioLoss(_linear_classifier(), criterion="hinge")
    # Criterion names are case-insensitive.
    loss_fn = NeuralRatioLoss(_linear_classifier(), criterion="BCE")
    assert loss_fn.criterion == "bce"


# ---------------------------------------------------------------------------
# Reductions, masks, weights
# ---------------------------------------------------------------------------


def test_reduction_mask_and_weight_semantics() -> None:
    context, observable, negative_context = _formula_inputs()
    loss_fn = _formula_loss_fn()
    per_sample = loss_fn(context, observable, negative_context=negative_context)
    mask = torch.tensor([True, False, True, False])
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)

    torch.testing.assert_close(
        _formula_loss_fn(reduction="mean")(context, observable, negative_context),
        per_sample.mean(),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        _formula_loss_fn(reduction="sum")(context, observable, negative_context),
        per_sample.sum(),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        _formula_loss_fn(reduction="mean")(
            context, observable, negative_context=negative_context, mask=mask
        ),
        per_sample[mask].mean(),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        _formula_loss_fn(reduction="sum")(
            context, observable, negative_context=negative_context, mask=mask
        ),
        per_sample[mask].sum(),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        _formula_loss_fn()(context, observable, negative_context=negative_context, mask=mask),
        torch.where(mask, per_sample, torch.zeros_like(per_sample)),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        _formula_loss_fn(reduction="mean")(
            context, observable, negative_context=negative_context, weights=weights
        ),
        (per_sample * weights).sum() / weights.sum(),
        rtol=0,
        atol=1e-12,
    )
    valid = mask.to(weights.dtype)
    torch.testing.assert_close(
        _formula_loss_fn(reduction="mean")(
            context,
            observable,
            negative_context=negative_context,
            mask=mask,
            weights=weights,
        ),
        (per_sample * weights * valid).sum() / (weights * valid).sum(),
        rtol=0,
        atol=1e-12,
    )


def test_mask_and_weight_validation() -> None:
    loss_fn, context, observable, negative_context = _small_loss_fn()
    with pytest.raises(ValueError, match="sample-level masks"):
        loss_fn(
            context,
            observable,
            negative_context=negative_context,
            mask=torch.ones(4, 2, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="sample-level weights"):
        loss_fn(
            context,
            observable,
            negative_context=negative_context,
            weights=torch.ones(4, 2),
        )
    with pytest.raises(ValueError, match="non-negative"):
        loss_fn(
            context,
            observable,
            negative_context=negative_context,
            weights=torch.tensor([1.0, -1.0, 1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="reduction"):
        NeuralRatioLoss(build_ratio_classifier(1, 1), reduction="median")


def test_masked_samples_do_not_poison_loss_or_gradients() -> None:
    loss_fn, context, observable, negative_context = _small_loss_fn(n=3, k=2)
    context = context.clone()
    observable = observable.clone()
    negative_context = negative_context.clone()
    context[1] = float("nan")
    observable[1] = float("nan")
    negative_context[1] = float("nan")
    mask = torch.tensor([True, False, True])

    per_sample = loss_fn(context, observable, negative_context=negative_context, mask=mask)
    assert torch.isfinite(per_sample).all()
    assert float(per_sample[1]) == 0.0

    reduced_loss_fn = NeuralRatioLoss(loss_fn.classifier, criterion="bce", reduction="mean")
    reduced = reduced_loss_fn(context, observable, negative_context=negative_context, mask=mask)
    reduced.backward()
    for parameter in loss_fn.classifier.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


# ---------------------------------------------------------------------------
# Negative-bank shapes, ambiguity rules, negative-target semantics
# ---------------------------------------------------------------------------


def test_negative_context_required() -> None:
    loss_fn, context, observable, _ = _small_loss_fn()
    with pytest.raises(ValueError, match="negative_context"):
        loss_fn(context, observable)


def test_negative_y_pred_alias() -> None:
    loss_fn, context, observable, negative_context = _small_loss_fn()
    explicit = loss_fn(context, observable, negative_context=negative_context)
    alias = loss_fn(context, observable, negative_y_pred=negative_context)
    torch.testing.assert_close(alias, explicit)

    with pytest.raises(ValueError, match="1-D or 2-D"):
        loss_fn(torch.tensor(1.0), torch.zeros(1), negative_context=negative_context)


def test_negative_context_shape_errors() -> None:
    loss_fn, context, observable, _ = _small_loss_fn(n=4, k=2, context_dim=1)
    cases = [
        (torch.empty(0, 1), {"shared_negative_context": True}, "at least one negative"),
        (torch.empty(1, 0, 1), {}, "at least one negative"),
        (torch.empty(4, 0, 1), {}, "at least one negative"),
        (torch.randn(4, 2, 2), {}, "last dimension"),
        (torch.randn(4, 2, 1, 1), {}, "must have shape"),
        (torch.randn(3, 2, 1), {}, "3D negative_context"),
        (torch.randn(2, 1), {"shared_negative_context": False}, "per-sample"),
        (torch.randn(4, 1), {}, "ambiguous"),
    ]
    for negative_context, kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            loss_fn(context, observable, negative_context=negative_context, **kwargs)

    with pytest.raises(TypeError, match="negative_context"):
        loss_fn(context, observable, negative_context=[[0.0]])
    with pytest.raises(TypeError, match="shared_negative_context"):
        loss_fn(
            context,
            observable,
            negative_context=torch.randn(1, 2, 1),
            shared_negative_context=1,
        )


@pytest.mark.parametrize(
    ("negative_context", "kwargs", "n_negatives", "batch_size"),
    [
        (torch.randn(2, 1), {}, 2, 4),  # shared 2-D bank, K != N
        (torch.randn(1, 2, 1), {}, 2, 4),  # explicit shared 3-D bank
        (torch.randn(4, 2, 1), {}, 2, 4),  # per-sample bank
        (torch.randn(4, 1), {"shared_negative_context": False}, 1, 4),
        (torch.randn(4, 1), {"shared_negative_context": True}, 4, 4),
    ],
)
def test_negative_context_shapes(
    negative_context: torch.Tensor,
    kwargs: dict[str, object],
    n_negatives: int,
    batch_size: int,
) -> None:
    loss_fn, context, observable, _ = _small_loss_fn(n=batch_size, k=2)
    ratios = loss_fn.log_likelihood_ratio(
        context,
        observable,
        negative_context=negative_context,
        **kwargs,  # type: ignore[arg-type]
    )
    assert ratios.shape == (batch_size, n_negatives)
    assert torch.isfinite(ratios).all()
    loss = loss_fn(context, observable, negative_context=negative_context, **kwargs)
    assert loss.shape == (batch_size,)


def test_shared_bank_expansion_is_consistent() -> None:
    loss_fn, context, observable, negative_context = _small_loss_fn(n=4, k=2)
    two_d = negative_context[:, 0, :]  # [4, 1], K=1
    shared_2d = loss_fn.log_likelihood_ratio(
        context,
        observable,
        negative_context=two_d,
        shared_negative_context=True,
    )
    shared_3d = loss_fn.log_likelihood_ratio(
        context, observable, negative_context=two_d.unsqueeze(0)
    )
    torch.testing.assert_close(shared_2d, shared_3d)

    per_sample_2d = loss_fn.log_likelihood_ratio(
        context,
        observable,
        negative_context=two_d,
        shared_negative_context=False,
    )
    explicit_3d = loss_fn.log_likelihood_ratio(
        context, observable, negative_context=two_d.unsqueeze(1)
    )
    torch.testing.assert_close(per_sample_2d, explicit_3d)


def test_paired_vs_mismatched_negative_target_semantics() -> None:
    context, observable, negative_context = _formula_inputs()
    negative_target = torch.tensor(
        [
            [[2.0], [-1.0], [0.5]],
            [[-1.0], [0.5], [3.0]],
            [[0.25], [3.0], [-2.0]],
            [[1.0], [-0.5], [0.0]],
        ],
        dtype=torch.float64,
    )  # [4, 3, 1]
    loss_fn = NeuralRatioLoss(_linear_classifier(), reduction="none")

    positive = _linear_logits(context, observable).squeeze(-1)
    paired_negative = _linear_logits(negative_context, observable.unsqueeze(1)).squeeze(-1)
    mismatched_negative = _linear_logits(negative_context, negative_target).squeeze(-1)
    expected_paired = _reference_loss(positive, paired_negative, "bce")
    expected_mismatched = _reference_loss(positive, mismatched_negative, "bce")

    actual_paired = loss_fn(context, observable, negative_context=negative_context)
    actual_mismatched = loss_fn(
        context,
        observable,
        negative_context=negative_context,
        negative_target=negative_target,
    )
    torch.testing.assert_close(actual_paired, expected_paired, rtol=0, atol=1e-12)
    torch.testing.assert_close(actual_mismatched, expected_mismatched, rtol=0, atol=1e-12)
    assert not torch.allclose(actual_paired, actual_mismatched)


def test_negative_target_shape_handling() -> None:
    loss_fn, context, observable, negative_context = _small_loss_fn(n=4, k=2)
    torch.manual_seed(7)
    shared_bank = torch.randn(1, 2, 1)
    per_sample = loss_fn(
        context,
        observable,
        negative_context=negative_context,
        negative_target=shared_bank.expand(4, -1, -1),
    )
    shared = loss_fn(
        context,
        observable,
        negative_context=negative_context,
        negative_target=shared_bank,
    )
    bank = loss_fn(
        context,
        observable,
        negative_context=negative_context,
        negative_target=shared_bank[0],
    )
    torch.testing.assert_close(per_sample, shared)
    torch.testing.assert_close(per_sample, bank)

    with pytest.raises(ValueError, match="same number of negatives"):
        loss_fn(
            context,
            observable,
            negative_context=negative_context,
            negative_target=torch.randn(4, 3, 1),
        )
    with pytest.raises(ValueError, match="last dimension"):
        loss_fn(
            context,
            observable,
            negative_context=negative_context,
            negative_target=torch.randn(4, 2, 2),
        )
    with pytest.raises(ValueError, match="ambiguous"):
        loss_fn(
            context,
            observable,
            negative_context=negative_context,
            negative_target=torch.randn(4, 1),
        )


def test_negative_target_ambiguity_rules() -> None:
    loss_fn, context, observable, negative_context = _small_loss_fn(n=4, k=4)
    torch.manual_seed(11)
    ambiguous_target = torch.randn(4, 1)  # [K, D] with K == batch_size
    with pytest.raises(ValueError, match="ambiguous"):
        loss_fn(
            context,
            observable,
            negative_context=negative_context,
            negative_target=ambiguous_target,
        )
    ratios = loss_fn.log_likelihood_ratio(
        context,
        observable,
        negative_context=negative_context,
        negative_target=ambiguous_target,
        shared_negative_target=True,
    )
    assert ratios.shape == (4, 4)

    # Explicit per-sample negatives (one negative observation per sample).
    per_sample_loss = loss_fn(
        context,
        observable,
        negative_context=torch.randn(4, 1),
        negative_target=ambiguous_target,
        shared_negative_context=False,
        shared_negative_target=False,
    )
    assert per_sample_loss.shape == (4,)


# ---------------------------------------------------------------------------
# log_likelihood_ratio, lazy dimensions, logits helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_negative_target", [False, True])
def test_log_likelihood_ratio_equals_logit_difference(with_negative_target: bool) -> None:
    context, observable, negative_context = _formula_inputs()
    negative_target = (
        torch.tensor(
            [
                [[2.0], [-1.0], [0.5]],
                [[-1.0], [0.5], [3.0]],
                [[0.25], [3.0], [-2.0]],
                [[1.0], [-0.5], [0.0]],
            ],
            dtype=torch.float64,
        )
        if with_negative_target
        else None
    )
    classifier = _linear_classifier()
    loss_fn = NeuralRatioLoss(classifier)

    positive = neural_ratio_logits(classifier, context, observable)
    n_samples = context.shape[0]
    n_negatives = negative_context.shape[1]
    if negative_target is None:
        negative_observable = observable.unsqueeze(1).expand(n_samples, n_negatives, 1)
    else:
        negative_observable = negative_target
    negative = neural_ratio_logits(
        classifier,
        negative_context.reshape(-1, 1),
        negative_observable.reshape(-1, 1),
    ).reshape(n_samples, n_negatives)
    expected = positive.unsqueeze(1) - negative

    actual = loss_fn.log_likelihood_ratio(
        context,
        observable,
        negative_context,
        negative_target,
    )
    torch.testing.assert_close(actual, expected)

    # 1-D convenience inputs encode a single context/observable dimension.
    squeezed = loss_fn.log_likelihood_ratio(
        context.squeeze(-1),
        observable.squeeze(-1),
        negative_context,
        None if negative_target is None else negative_target,
    )
    torch.testing.assert_close(squeezed, expected)


def test_dimensions_are_inferred_lazily_and_validated() -> None:
    loss_fn = NeuralRatioLoss(_linear_classifier())
    assert loss_fn.context_dim is None
    assert loss_fn.observable_dim is None

    loss_fn(
        torch.zeros(3, 1, dtype=torch.float64),
        torch.zeros(3, 1, dtype=torch.float64),
        negative_context=torch.zeros(3, 2, 1, dtype=torch.float64),
    )
    assert loss_fn.context_dim == 1
    assert loss_fn.observable_dim == 1

    with pytest.raises(ValueError, match="context dimension changed"):
        loss_fn(
            torch.zeros(3, 2, dtype=torch.float64),
            torch.zeros(3, 1, dtype=torch.float64),
            negative_context=torch.zeros(3, 2, 2, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="observable dimension changed"):
        loss_fn(
            torch.zeros(3, 1, dtype=torch.float64),
            torch.zeros(3, 2, dtype=torch.float64),
            negative_context=torch.zeros(3, 2, 1, dtype=torch.float64),
        )


def test_build_ratio_classifier_tags_dimensions_for_loss() -> None:
    loss_fn = NeuralRatioLoss(build_ratio_classifier(2, 3, hidden_dims=(4,)))
    assert loss_fn.context_dim == 2
    assert loss_fn.observable_dim == 3
    with pytest.raises(ValueError, match="input width"):
        neural_ratio_logits(loss_fn.classifier, torch.zeros(5, 1), torch.zeros(5, 3))


def test_neural_ratio_logits_validation_errors() -> None:
    classifier = build_ratio_classifier(1, 1, hidden_dims=(4,))
    with pytest.raises(TypeError, match="classifier"):
        neural_ratio_logits(
            "not-a-module",  # type: ignore[arg-type]
            torch.zeros(2, 1),
            torch.zeros(2, 1),
        )
    with pytest.raises(ValueError, match="same batch dimension"):
        neural_ratio_logits(classifier, torch.zeros(2, 1), torch.zeros(3, 1))
    with pytest.raises(ValueError, match="1-D or 2-D"):
        neural_ratio_logits(classifier, torch.zeros(2, 1, 1), torch.zeros(2, 1))
    with pytest.raises(TypeError, match="floating-point"):
        neural_ratio_logits(classifier, torch.zeros(2, 1, dtype=torch.long), torch.zeros(2, 1))
    with pytest.raises(ValueError, match="same dtype"):
        neural_ratio_logits(classifier, torch.zeros(2, 1), torch.zeros(2, 1, dtype=torch.float64))

    class _WideOutput(nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(inputs.shape[0], 2)

    with pytest.raises(ValueError, match="one logit per pair"):
        neural_ratio_logits(_WideOutput(), torch.zeros(2, 1), torch.zeros(2, 1))
    with pytest.raises(ValueError, match="same device"):
        neural_ratio_logits(classifier, torch.zeros(2, 1, device="meta"), torch.zeros(2, 1))


# ---------------------------------------------------------------------------
# Functional helper and gradient flow
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("criterion", ["bce", "multiclass"])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("with_negative_target", [False, True])
def test_functional_helper_matches_class(
    criterion: str, reduction: str, with_negative_target: bool
) -> None:
    loss_fn, context, observable, negative_context = _small_loss_fn(
        criterion=criterion, reduction=reduction, n=4, k=2
    )
    torch.manual_seed(3)
    negative_target = torch.randn(4, 2, 1) if with_negative_target else None

    expected = loss_fn(
        context,
        observable,
        negative_context=negative_context,
        negative_target=negative_target,
    )
    actual = neural_ratio_loss(
        loss_fn.classifier,
        context,
        observable,
        negative_context,
        negative_target,
        criterion=criterion,
        reduction=reduction,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("with_negative_target", [False, True])
def test_gradients_are_finite(with_negative_target: bool) -> None:
    loss_fn, context, observable, negative_context = _small_loss_fn(n=4, k=2)
    torch.manual_seed(5)
    negative_target = torch.randn(4, 2, 1) if with_negative_target else None
    loss = loss_fn(
        context,
        observable,
        negative_context=negative_context,
        negative_target=negative_target,
    ).sum()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in loss_fn.classifier.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)


# ---------------------------------------------------------------------------
# Public-loss contract
# ---------------------------------------------------------------------------


def test_forward_signature_contract() -> None:
    signature = inspect.signature(NeuralRatioLoss.forward)
    parameters = list(signature.parameters.values())
    names = [parameter.name for parameter in parameters]
    assert names[:3] == ["self", "y_pred", "target"]
    assert "mask" in names and "weights" in names
    assert names.index("mask") < names.index("weights")
    assert signature.parameters["mask"].default is None
    assert signature.parameters["weights"].default is None
    assert any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    assert sorted(nre_module.__all__) == [
        "NeuralRatioLoss",
        "build_ratio_classifier",
        "neural_ratio_logits",
        "neural_ratio_loss",
    ]


# ---------------------------------------------------------------------------
# Scientific consistency: analytic Gaussian model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("criterion", ["bce", "multiclass"])
def test_gaussian_nre_recovers_mle_ranking(criterion: str) -> None:
    """Trained NRE logits rank candidates by likelihood, recovering the MLE.

    Data: ``c ~ N(0, 1)``, ``x | c ~ N(c, sigma^2)`` with ``sigma = 0.5``.  The
    negative contexts are drawn from the same prior, so the Bayes-optimal logit
    is ``log p(x | c) - log p(x)``.  The analytic maximum-likelihood estimate for
    a flat prior is ``argmax_c p(x | c) = x`` (closest grid point); the
    generating ``c`` itself is only identifiable up to the observation noise.
    """
    classifier, loss_fn, first_loss, last_loss = _train_gaussian_ratio_estimator(criterion)

    # Training actually optimises the criterion.
    assert last_loss < first_loss - 0.2

    torch.manual_seed(10_000)
    n_test = 64
    c_true = torch.randn(n_test, 1)
    x_obs = c_true + SIGMA * torch.randn(n_test, 1)

    scores = _candidate_scores(classifier, x_obs)
    assert scores.shape == (n_test, GRID.shape[0])
    assert torch.isfinite(scores).all()

    # Analytic log p(x | c) up to a constant independent of the candidate.
    analytic_scores = -((x_obs - GRID.reshape(1, -1)) ** 2) / (2 * SIGMA**2)
    grid_flat = GRID.reshape(-1)
    mle = grid_flat[(x_obs - GRID.reshape(1, -1)).abs().argmin(dim=1)]
    estimate = grid_flat[scores.argmax(dim=1)]

    assert float((estimate - mle).abs().mean()) < 0.15
    assert float((estimate - c_true.squeeze(-1)).abs().mean()) < 0.6

    # Estimated log-ratios are monotone with the analytic log-likelihood over a
    # dense candidate grid (rank correlation well above the 0.9 evidence bound).
    rank_scores = _spearman_rows(scores, analytic_scores)
    assert float(rank_scores.mean()) > 0.95
    assert float(rank_scores.min()) > 0.9

    # Pairwise log-ratios against a reference hypothesis match the analytic
    # ordering: r_k = s(c_ref, x) - s(c_k, x) increases as c_k becomes less likely.
    reference_context = x_obs + 0.2
    ratios = loss_fn.log_likelihood_ratio(reference_context, x_obs, GRID)
    analytic_plus = -((x_obs - reference_context) ** 2) / (2 * SIGMA**2)
    analytic_ratios = analytic_plus - analytic_scores
    rank_ratios = _spearman_rows(ratios, analytic_ratios)
    assert float(rank_ratios.mean()) > 0.95
    assert float(rank_ratios.min()) > 0.9

    # No shrinkage toward the training prior: on a shifted evaluation prior the
    # estimate still tracks the likelihood maximum, not the training prior mean.
    torch.manual_seed(20_000)
    c_shifted = 1.0 + 0.5 * torch.randn(n_test, 1)
    x_shifted = c_shifted + SIGMA * torch.randn(n_test, 1)
    shifted_scores = _candidate_scores(classifier, x_shifted)
    shifted_estimate = grid_flat[shifted_scores.argmax(dim=1)]
    shifted_mle = grid_flat[(x_shifted - GRID.reshape(1, -1)).abs().argmin(dim=1)]
    assert float((shifted_estimate - shifted_mle).abs().mean()) < 0.15
    assert float((shifted_estimate - c_shifted.squeeze(-1)).abs().mean()) < 0.6
