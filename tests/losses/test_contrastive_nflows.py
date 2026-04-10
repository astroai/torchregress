from __future__ import annotations

import pytest
import torch
from torch import nn

from torchregress.losses.nflows import ContrastiveFlowLoss


class _DummyConditionalFlow(nn.Module):
    def __init__(self, n_features: int = 1, context_dim: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(context_dim, n_features, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
        self.context = context_dim
        self._n_features = n_features

    def base(self) -> torch.distributions.Distribution:
        return torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(self._n_features), torch.ones(self._n_features)),
            1,
        )

    def forward(self, context: torch.Tensor | None = None) -> torch.distributions.Distribution:
        if context is None:
            loc = torch.zeros(self._n_features)
        else:
            loc = self.linear(context)
        scale = torch.ones_like(loc) * 0.2
        return torch.distributions.Independent(torch.distributions.Normal(loc, scale), 1)


@pytest.fixture
def loss_fn() -> ContrastiveFlowLoss:
    return ContrastiveFlowLoss(flow=_DummyConditionalFlow(), temperature=0.5, margin=0.1)


def test_contrastive_flow_requires_negative_context(loss_fn: ContrastiveFlowLoss) -> None:
    with pytest.raises(ValueError, match="negative_context"):
        loss_fn(torch.tensor([[1.0]]), torch.tensor([[1.0]]))


def test_contrastive_flow_prefers_matching_context(loss_fn: ContrastiveFlowLoss) -> None:
    target = torch.tensor([[1.0], [2.0]])
    positive_context = target.clone()
    swapped_context = torch.flip(target, dims=[0]).unsqueeze(1)

    good_loss = loss_fn(
        positive_context,
        target,
        negative_context=swapped_context,
    )
    bad_loss = loss_fn(
        torch.flip(target, dims=[0]),
        target,
        negative_context=positive_context.unsqueeze(1),
    )

    assert float(good_loss.item()) < float(bad_loss.item())


def test_contrastive_flow_supports_shared_negative_bank(loss_fn: ContrastiveFlowLoss) -> None:
    target = torch.tensor([[0.5], [1.0], [1.5]])
    positive_context = target.clone()
    negative_bank = torch.tensor([[-1.0], [2.5]])

    loss = loss_fn(positive_context, target, negative_context=negative_bank)
    ratios = loss_fn.log_likelihood_ratio(positive_context, target, negative_bank)

    assert torch.isfinite(loss)
    assert ratios.shape == (3, 2)
    assert torch.all(ratios > -5.0)


def test_contrastive_flow_rejects_partial_feature_mask() -> None:
    target = torch.tensor([[1.0, 0.0]])
    context = torch.tensor([[1.0]])
    negative_context = torch.tensor([[[0.0]]])

    multi_feature_loss = ContrastiveFlowLoss(
        flow=_DummyConditionalFlow(n_features=2, context_dim=1),
        temperature=0.5,
        margin=0.1,
    )
    with pytest.raises(ValueError, match="sample-level masks"):
        multi_feature_loss(
            context,
            target,
            mask=torch.tensor([[True, False]]),
            negative_context=negative_context,
        )


def test_contrastive_flow_rejects_ambiguous_2d_negative_context(
    loss_fn: ContrastiveFlowLoss,
) -> None:
    target = torch.tensor([[0.0], [0.5]])
    positive_context = target.clone()
    ambiguous = torch.tensor([[-1.0], [2.0]])

    with pytest.raises(ValueError, match="ambiguous"):
        loss_fn.negative_log_likelihoods(positive_context, target, ambiguous)


def test_contrastive_flow_accepts_explicit_shared_bank_when_batch_sizes_match(
    loss_fn: ContrastiveFlowLoss,
) -> None:
    target = torch.tensor([[0.0], [0.5]])
    positive_context = target.clone()
    shared_bank = torch.tensor([[[-1.0], [2.0]]])

    ratios = loss_fn.log_likelihood_ratio(positive_context, target, shared_bank)

    assert ratios.shape == (2, 2)


def test_contrastive_flow_accepts_explicit_shared_flag_for_2d_bank(
    loss_fn: ContrastiveFlowLoss,
) -> None:
    target = torch.tensor([[0.0], [0.5]])
    positive_context = target.clone()
    ambiguous_2d_bank = torch.tensor([[-1.0], [2.0]])

    ratios = loss_fn.log_likelihood_ratio(
        positive_context,
        target,
        ambiguous_2d_bank,
        shared_negative_context=True,
    )

    assert ratios.shape == (2, 2)


def test_contrastive_flow_rejects_empty_negative_banks(
    loss_fn: ContrastiveFlowLoss,
) -> None:
    target = torch.tensor([[0.0], [0.5]])
    positive_context = target.clone()

    for neg, kwargs in [
        (torch.empty(0, 1), {"shared_negative_context": True}),
        (torch.empty(1, 0, 1), {}),
        (torch.empty(2, 0, 1), {}),
    ]:
        with pytest.raises(ValueError, match="at least one negative hypothesis"):
            loss_fn(positive_context, target, negative_context=neg, **kwargs)
