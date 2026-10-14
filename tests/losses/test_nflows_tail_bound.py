"""Spline tail bound and recommended training recipe for NSF conditional flows."""

from __future__ import annotations

import warnings

import pytest
import torch

from torchregress.losses.nflows import (
    DEFAULT_TAIL_BOUND,
    NormalizingFlowLoss,
    create_flow_model,
    recommended_tail_bound,
)

zuko = pytest.importorskip("zuko")


# --- recommended_tail_bound (pure torch) ------------------------------------


def test_recommended_tail_bound_is_data_range_with_margin_clipped():
    assert recommended_tail_bound(torch.randn(500)) == DEFAULT_TAIL_BOUND  # never below zuko's 5
    assert recommended_tail_bound(torch.tensor([-12.0, 0.0, 3.0])) == pytest.approx(15.0)
    assert recommended_tail_bound(torch.tensor([[-12.0], [30.0]])) == 25.0  # upper clip
    assert recommended_tail_bound([2.0, -8.0], margin=1.0) == 8.0
    assert recommended_tail_bound(torch.tensor([float("nan"), 9.0, float("inf")])) == pytest.approx(
        11.25
    )
    assert recommended_tail_bound(torch.tensor([float("nan")])) == DEFAULT_TAIL_BOUND
    with pytest.raises(ValueError, match="margin"):
        recommended_tail_bound(torch.randn(4), margin=0.5)
    with pytest.raises(ValueError, match="minimum"):
        recommended_tail_bound(torch.randn(4), minimum=10.0, maximum=5.0)


# --- bound= on create_flow_model --------------------------------------------


def test_default_bound_builds_the_plain_zuko_nsf():
    assert type(create_flow_model(1, 4, "nsf", 2)).__name__ == "NSF"
    assert type(create_flow_model(1, 4, "nsf", 2, bound=5.0)).__name__ == "NSF"


def test_bound_is_validated_and_nsf_only():
    with pytest.raises(ValueError, match="nsf"):
        create_flow_model(1, 4, "maf", 2, bound=10.0)
    for bad in (0.0, -3.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="bound"):
            create_flow_model(1, 4, "nsf", 2, bound=bad)


@pytest.mark.parametrize("bound", [5.0, 12.0])
def test_density_integrates_to_one_with_any_bound(bound):
    torch.manual_seed(0)
    flow = create_flow_model(1, 3, "nsf", 2, bound=bound, bins=12)
    loss = NormalizingFlowLoss(flow=flow)
    grid = torch.linspace(-45.0, 45.0, 90001)
    context = torch.randn(1, 3).expand(grid.numel(), 3)
    with torch.no_grad():
        density = loss.log_prob(context, grid[:, None]).exp()
    mass = float(torch.trapezoid(density, grid))
    assert mass == pytest.approx(1.0, abs=2e-3)


def test_wider_bound_moves_the_spline_domain_and_flow_is_identity_outside():
    torch.manual_seed(1)
    flow = create_flow_model(1, 2, "nsf", 1, bound=14.0)
    assert getattr(flow, "tail_bound") == 14.0
    context = torch.zeros(1, 2)
    inside = torch.tensor([[11.0]])
    outside = torch.tensor([[40.0]])
    transform = flow(context).transform
    with torch.no_grad():
        # Beyond the bound the spline is the identity; within it the (randomly
        # initialised) spline is a non-trivial monotone map.
        assert torch.allclose(transform(outside), outside)
        assert not torch.allclose(transform(inside), inside, atol=1e-6)


def test_range_warning_reports_the_actual_bound():
    flow = create_flow_model(1, 2, "nsf", 1, bound=8.0)
    loss = NormalizingFlowLoss(flow=flow)
    context = torch.zeros(3, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss(context, torch.tensor([[7.9], [-7.0], [0.0]]))  # inside [-8, 8]: silent
    with pytest.warns(UserWarning, match=r"exceeds 8.*create_flow_model\(bound="):
        loss(context, torch.tensor([[9.0], [0.0], [0.0]]))
    plain = NormalizingFlowLoss(flow=create_flow_model(1, 2, "nsf", 1))
    with pytest.warns(UserWarning, match=r"exceeds 5"):
        plain(context, torch.tensor([[6.0], [0.0], [0.0]]))


# --- the recommended recipe on a heavy-tailed target ------------------------


def _heavy_tailed_regression(n: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=generator)
    student = torch.randn(n, generator=generator) / torch.sqrt(
        torch.empty(n).exponential_(generator=generator) * 2.0 / 2.5 + 1e-6
    )
    y = x[:, 0] + (0.5 + 0.3 * x[:, 1].abs()) * student
    return x, y


def _fit_flow(x, y, *, bound, bins, epochs=25, seed=0):
    torch.manual_seed(seed)
    flow = create_flow_model(1, 16, "nsf", 2, hidden_features=[32, 32], bins=bins, bound=bound)
    backbone = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 16), torch.nn.Tanh())
    loss = NormalizingFlowLoss(flow=flow)
    params = list(backbone.parameters()) + list(flow.parameters())
    optimiser = torch.optim.Adam(params, lr=3e-3)
    for _ in range(epochs):
        order = torch.randperm(x.shape[0])
        for start in range(0, x.shape[0], 128):
            idx = order[start : start + 128]
            optimiser.zero_grad()
            loss(backbone(x[idx]), y[idx, None]).backward()
            optimiser.step()
    return backbone, loss


def test_data_derived_tail_bound_improves_heavy_tailed_nll():
    x, y = _heavy_tailed_regression(1500, seed=3)
    x_test, y_test = _heavy_tailed_regression(1500, seed=4)
    mean, scale = y.mean(), y.std()
    y_std, y_test_std = (y - mean) / scale, (y_test - mean) / scale
    bound = recommended_tail_bound(y_std)
    assert bound > DEFAULT_TAIL_BOUND  # Student-t(2.5) reaches |y| >> 5 sigma at n = 1500

    nll = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for name, tail in (("default", None), ("derived", bound)):
            backbone, loss = _fit_flow(x, y_std, bound=tail, bins=8)
            with torch.no_grad():
                nll[name] = float(loss(backbone(x_test), y_test_std[:, None]))
    assert nll["derived"] < nll["default"]
