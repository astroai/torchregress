from __future__ import annotations

import pytest
import torch
from torch import nn

from torchregress.losses import (
    EnsembleEIVLoss,
    FunctionalEIVLoss,
    MDNLoss,
    OrthogonalDistanceRegressionLoss,
    StructuralEIVLoss,
    create_mdn_loss,
    nflows,
)


def test_mdn_factory_mask_behavior_and_error_paths() -> None:
    torch.manual_seed(0)
    loss_fn = create_mdn_loss(
        n_components=2,
        n_features=2,
        covariance_type="diagonal",
        reduction="none",
    )
    assert isinstance(loss_fn, MDNLoss)

    y_pred = torch.randn(3, loss_fn.expected_output_size)
    target = torch.randn(3, 2)
    mask = torch.tensor([[True, True], [True, False], [False, False]])

    loss = loss_fn(y_pred, target, mask=mask)
    assert loss.shape == (3,)
    assert torch.isfinite(loss[0])
    assert loss[1].item() == 0.0
    assert loss[2].item() == 0.0

    with pytest.raises(ValueError, match="Expected 2 features"):
        loss_fn(y_pred, torch.randn(3, 3))

    mdn_full = MDNLoss(n_components=2, n_features=2, covariance_type="full")
    with pytest.raises(NotImplementedError, match="sample currently only supports diagonal"):
        mdn_full.sample(torch.randn(2, mdn_full.expected_output_size), n_samples=4)


def test_eiv_factory_dispatch_and_functional_mc_behavior() -> None:
    def model(x: torch.Tensor) -> torch.Tensor:
        return x[:, :1] + 0.5 * x[:, 1:2]

    functional = FunctionalEIVLoss(model=model, sigma_x=0.1, sigma_y=0.1)
    assert isinstance(functional, FunctionalEIVLoss)

    structural = StructuralEIVLoss(
        model=model,
        sigma_x=torch.tensor([0.1, 0.2]),
        sigma_y=torch.tensor([0.3]),
        sigma_xy=torch.zeros(1, 2),
    )
    assert isinstance(structural, StructuralEIVLoss)

    odr = OrthogonalDistanceRegressionLoss(model=model, sigma_x=0.1, sigma_y=0.1, max_iterations=2)
    assert isinstance(odr, OrthogonalDistanceRegressionLoss)
    odr_alias = OrthogonalDistanceRegressionLoss(model=model, sigma_x=0.1, sigma_y=0.1)
    assert isinstance(odr_alias, OrthogonalDistanceRegressionLoss)

    ensemble = EnsembleEIVLoss(
        model=model,
        sigma_x=0.1,
        n_samples=3,
    )
    assert isinstance(ensemble, EnsembleEIVLoss)

    mc_loss = FunctionalEIVLoss(
        model=model,
        sigma_x=torch.tensor([0.1, 0.2]),
        sigma_y=0.1,
        monte_carlo=True,
        n_samples=4,
        reduction="none",
    )
    x_obs = torch.randn(4, 2)
    y_obs = model(x_obs).detach()
    mask = torch.tensor([[True], [False], [True], [True]])
    weights = torch.tensor([1.0, 2.0, 1.0, 0.5])

    out = mc_loss(x_obs, y_obs, mask=mask, weights=weights)
    assert out.ndim == 1
    # A9 unified ZERO-FILL mask policy: reduction='none' keeps the original
    # shape; masked-out entries are exactly zero (TR-LOSS-30 mask semantics).
    assert out.numel() == mask.numel()
    assert float(out[1].abs()) == 0.0
    assert torch.isfinite(out).all()

    # ODR edge path: zero optimization iterations still performs final evaluation without error.
    odr_zero_iter = OrthogonalDistanceRegressionLoss(
        model=model,
        sigma_x=0.1,
        sigma_y=0.1,
        max_iterations=0,
        reduction="mean",
    )
    odr_loss = odr_zero_iter(x_obs, y_obs)
    assert torch.isfinite(odr_loss)


class _DummyEventBase:
    def __init__(self, n_features: int) -> None:
        self.event_shape = (n_features,)


class _DummyDist:
    def __init__(self, n_features: int, batch_shape: tuple[int, ...]) -> None:
        self.n_features = n_features
        self.batch_shape = batch_shape

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        if target.shape[-1] != self.n_features:
            raise ValueError("target feature mismatch")
        return -(target**2).sum(dim=-1)

    def sample(self, sample_shape: tuple[int, ...] = ()) -> torch.Tensor:
        shape = sample_shape + self.batch_shape + (self.n_features,)
        return torch.zeros(shape)


class _DummyFlow(nn.Module):
    def __init__(self, n_features: int, *, context_attr: int | None = None) -> None:
        super().__init__()
        self._n_features = n_features
        self.calls: list[torch.Tensor | None] = []
        if context_attr is not None:
            self.context = context_attr

    def base(self) -> _DummyEventBase:
        return _DummyEventBase(self._n_features)

    def forward(self, context: torch.Tensor | None = None) -> _DummyDist:
        self.calls.append(context)
        batch_shape = (int(context.shape[0]),) if context is not None else ()
        return _DummyDist(self._n_features, batch_shape=batch_shape)


class _BrokenFlow(nn.Module):
    def base(self) -> _DummyEventBase:
        raise RuntimeError("base unavailable")

    def forward(self, context: torch.Tensor | None = None) -> _DummyDist:
        batch_shape = (int(context.shape[0]),) if context is not None else ()
        return _DummyDist(1, batch_shape=batch_shape)


def test_normalizing_flow_loss_behavior_with_dummy_flow_and_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conditional_flow = _DummyFlow(2)  # No context attr => inferred on first forward.
    loss_fn = nflows.NormalizingFlowLoss(flow=conditional_flow, reduction="mean")

    context = torch.randn(5, 3)
    target = torch.randn(5, 2)
    loss = loss_fn(context, target)
    assert torch.isfinite(loss)
    assert loss_fn.context_dim == 3
    assert conditional_flow.calls and conditional_flow.calls[-1] is not None

    samples = loss_fn.sample(context, n_samples=4)
    assert samples.shape == (5, 4, 2)
    one_sample = loss_fn.sample(context, n_samples=1)
    assert one_sample.shape == (5, 2)
    log_prob = loss_fn.log_prob(context, target)
    assert log_prob.shape == (5,)
    assert torch.allclose(log_prob, -(target**2).sum(dim=-1))

    feature_mask = torch.tensor(
        [[True, True], [True, False], [True, True], [False, False], [True, True]]
    )
    feature_weights = torch.ones(5, 2)
    with pytest.raises(ValueError, match="sample-level masks"):
        loss_fn(context, target, mask=feature_mask, weights=feature_weights)

    sample_mask = torch.tensor([True, True, False, True, True])
    masked_loss = loss_fn(context, target, mask=sample_mask, weights=torch.ones(5))
    assert torch.isfinite(masked_loss)

    with pytest.raises(ValueError, match="Expected 2 features"):
        loss_fn(context, torch.randn(5, 1))

    unconditional_flow = _DummyFlow(2, context_attr=0)
    unconditional_loss = nflows.NormalizingFlowLoss(flow=unconditional_flow, reduction="none")
    unconditional_samples = unconditional_loss.sample(torch.randn(6, 1), n_samples=3)
    assert unconditional_samples.shape == (6, 3, 2)
    assert unconditional_flow.calls and unconditional_flow.calls[-1] is None

    scalar_flow = _DummyFlow(1)
    scalar_loss = nflows.NormalizingFlowLoss(flow=scalar_flow, reduction="mean")
    scalar_context = torch.randn(4, 2)
    quantiles = scalar_loss.quantile(scalar_context, [0.1, 0.5, 0.9], n_samples=16)
    assert quantiles.shape == (4, 3)
    assert torch.allclose(quantiles, torch.zeros_like(quantiles))
    cdf = scalar_loss.cdf(
        scalar_context,
        torch.tensor([[-1.0], [0.0], [0.5], [1.0]], dtype=scalar_context.dtype),
        n_samples=16,
    )
    assert cdf.shape == (4,)
    assert torch.all((cdf >= 0.0) & (cdf <= 1.0))
    assert cdf[0].item() == 0.0
    assert cdf[1].item() == 1.0

    def _fake_create_flow_model(**_: object) -> nn.Module:
        return _DummyFlow(2, context_attr=4)

    monkeypatch.setattr(nflows, "create_flow_model", _fake_create_flow_model)
    created = nflows.create_flow_loss(n_features=2, context_dim=4, reduction="sum")
    assert isinstance(created, nflows.NormalizingFlowLoss)
    assert created.reduction == "sum"


def test_normalizing_flow_loss_init_wraps_base_extraction_failure() -> None:
    with pytest.raises(ValueError, match="Could not extract feature dimension"):
        nflows.NormalizingFlowLoss(flow=_BrokenFlow())
