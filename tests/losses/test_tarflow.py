"""Transformer autoregressive flows (TarFlow / STARFlow-style) for conditional densities."""

from __future__ import annotations

import math
import warnings

import pytest
import torch

zuko = pytest.importorskip("zuko")

from torchregress.losses import (  # noqa: E402
    NormalizingFlowLoss,
    create_flow_model,
    recommended_tail_bound,
)
from torchregress.losses.tarflow import (  # noqa: E402
    TransformerAutoregressiveTransform,
    TransformerFlow,
)
from torchregress.metrics import sample_hpd_coverage  # noqa: E402

UNIVARIATES = ("spline", "affine")


def _randomize(flow: TransformerFlow, std: float = 0.3) -> TransformerFlow:
    """Move every block away from its identity initialization."""
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for block in flow.transform.transforms:
            block.head.weight.copy_(torch.randn(block.head.weight.shape, generator=generator) * std)
            block.head.bias.copy_(torch.randn(block.head.bias.shape, generator=generator) * std)
    return flow


@pytest.mark.parametrize("univariate", UNIVARIATES)
@pytest.mark.parametrize("features", [1, 2, 4])
def test_inverse_and_log_det_are_exact(univariate: str, features: int) -> None:
    torch.manual_seed(0)
    flow = _randomize(
        TransformerFlow(features, 3, 3, univariate=univariate, hidden=16, n_heads=2).double()
    )
    context = torch.randn(6, 3, dtype=torch.float64)
    target = torch.randn(6, features, dtype=torch.float64)
    transform = flow(context).transform
    latent = transform(target)
    torch.testing.assert_close(transform.inv(latent), target, rtol=0, atol=1e-8)
    for row in range(3):
        single = flow(context[row : row + 1]).transform
        jacobian = torch.autograd.functional.jacobian(
            lambda v, t=single: t(v.unsqueeze(0))[0], target[row]
        )
        expected = torch.linalg.slogdet(jacobian)[1]
        ladj = single.log_abs_det_jacobian(target[row : row + 1], single(target[row : row + 1]))
        torch.testing.assert_close(ladj.reshape(()), expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 1, 0, 2]])
def test_block_is_autoregressive_in_its_order(order: list[int]) -> None:
    torch.manual_seed(1)
    block = TransformerAutoregressiveTransform(4, 2, order=torch.tensor(order), hidden=16).double()
    torch.nn.init.normal_(block.head.weight, std=0.5)
    context = torch.randn(1, 2, dtype=torch.float64)
    target = torch.randn(4, dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(
        lambda v: block(context)(v.unsqueeze(0))[0], target
    )
    permuted = jacobian[order][:, order]
    # Lower triangular with a positive diagonal in the block's own order.
    assert torch.equal(torch.triu(permuted, diagonal=1), torch.zeros_like(permuted))
    assert bool((torch.diagonal(permuted) > 0).all())
    assert bool(torch.tril(permuted, diagonal=-1).abs().sum() > 0)


def test_blocks_alternate_orders_and_start_as_identity() -> None:
    flow = TransformerFlow(3, 2, 4, hidden=16)
    orders = [block.order.tolist() for block in flow.transform.transforms]
    assert orders == [[0, 1, 2], [2, 1, 0], [0, 1, 2], [2, 1, 0]]
    context, target = torch.randn(5, 2), torch.randn(5, 3)
    standard = torch.distributions.Normal(0.0, 1.0).log_prob(target).sum(-1)
    torch.testing.assert_close(flow(context).log_prob(target), standard)


def test_deep_shallow_depths() -> None:
    flow = TransformerFlow(3, 2, n_layers=(4, 1, 1), hidden=16)
    depths = [len(block.encoder.layers) for block in flow.transform.transforms]
    assert depths == [4, 1, 1]


def test_sampling_shapes_and_broadcasting() -> None:
    torch.manual_seed(2)
    flow = _randomize(TransformerFlow(3, 2, 2, hidden=16))
    context = torch.randn(5, 2)
    distribution = flow(context)
    samples = distribution.sample((7,))
    assert samples.shape == (7, 5, 3) and bool(torch.isfinite(samples).all())
    assert distribution.log_prob(samples).shape == (7, 5)
    unconditional = TransformerFlow(2, 0, 2, hidden=16)
    assert unconditional().sample((4,)).shape == (4, 2)


def test_create_flow_model_maps_the_shared_knobs() -> None:
    flow = create_flow_model(3, 4, flow_type="tarflow", n_transforms=2, hidden_features=[32, 48])
    assert isinstance(flow, TransformerFlow)
    blocks = flow.transform.transforms
    assert len(blocks) == 2
    assert blocks[0].head.in_features == 48  # width = largest hidden entry
    assert len(blocks[0].encoder.layers) == 2  # one layer per hidden entry
    assert getattr(flow, "context") == 4
    deep = create_flow_model(3, 4, flow_type="tarflow", n_layers=(3, 1), hidden_features=32)
    assert [len(b.encoder.layers) for b in deep.transform.transforms] == [3, 1]

    spline = create_flow_model(2, 1, flow_type="tarflow", bound=9.0)
    assert spline.tail_bound == 9.0
    assert spline.transform.transforms[0].univariate.keywords["bound"] == 9.0
    affine = create_flow_model(2, 1, flow_type="tarflow", univariate="affine", bound=7.0)
    assert not hasattr(affine, "tail_bound")
    assert affine.transform.transforms[0].conditioner_bound == 7.0


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="univariate"):
        TransformerFlow(2, 1, univariate="cubic")
    with pytest.raises(ValueError, match="divisible"):
        TransformerFlow(2, 1, hidden=30, n_heads=4)
    with pytest.raises(ValueError, match="permutation"):
        TransformerAutoregressiveTransform(3, order=torch.tensor([0, 0, 1]))
    with pytest.raises(ValueError, match="at least one"):
        TransformerFlow(2, 1, n_layers=(2, 0))
    with pytest.raises(ValueError, match="bins"):
        TransformerFlow(2, 1, bins=1)
    with pytest.raises(ValueError, match="slope"):
        TransformerFlow(2, 1, slope=1.5)


def test_loss_warns_when_targets_leave_the_spline_domain() -> None:
    loss = NormalizingFlowLoss(create_flow_model(2, 2, flow_type="tarflow"))
    with pytest.warns(UserWarning, match="tail bound"):
        loss(torch.randn(4, 2), torch.tensor([[0.0, 7.0]] * 4))
    affine = NormalizingFlowLoss(create_flow_model(2, 2, flow_type="tarflow", univariate="affine"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        affine(torch.randn(4, 2), torch.tensor([[0.0, 7.0]] * 4))


def test_state_dict_round_trip() -> None:
    torch.manual_seed(3)
    flow = _randomize(create_flow_model(3, 2, flow_type="tarflow", hidden_features=16))
    clone = create_flow_model(3, 2, flow_type="tarflow", hidden_features=16)
    clone.load_state_dict(flow.state_dict())
    context, target = torch.randn(4, 2), torch.randn(4, 3)
    torch.testing.assert_close(clone(context).log_prob(target), flow(context).log_prob(target))


def _banana(n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Conditional banana with a known density: y2 | y1, x is curved in y1."""
    x = torch.randn(n, 1, generator=generator)
    y1 = x[:, 0] + 0.6 * torch.randn(n, generator=generator)
    y2 = 0.8 * y1**2 - 1.0 + 0.3 * torch.randn(n, generator=generator)
    y = torch.stack([y1, y2], dim=1)
    log_p = torch.distributions.Normal(x[:, 0], 0.6).log_prob(y1) + torch.distributions.Normal(
        0.8 * y1**2 - 1.0, 0.3
    ).log_prob(y2)
    return x, y, log_p


def _fit_banana(seed: int, **flow_kwargs: object) -> tuple[float, torch.Tensor]:
    """Train on a standardized banana; return (KL gap in nats, HPD coverage at 0.5/0.9)."""
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    x, y_raw, _ = _banana(3000, generator)
    x_test, y_test_raw, log_p_raw = _banana(2000, generator)
    mean_y, std_y = y_raw.mean(0), y_raw.std(0)
    y, y_test = (y_raw - mean_y) / std_y, (y_test_raw - mean_y) / std_y
    log_p_test = log_p_raw + torch.log(std_y).sum()
    flow = create_flow_model(
        2,
        1,
        n_transforms=3,
        hidden_features=[32, 32],
        bound=recommended_tail_bound(y),
        **flow_kwargs,  # type: ignore[arg-type]
    )
    loss_fn = NormalizingFlowLoss(flow)
    optimizer = torch.optim.Adam(flow.parameters(), lr=3e-3)
    for _ in range(25):
        for batch in torch.randperm(3000, generator=generator).split(250):
            optimizer.zero_grad()
            loss_fn(x[batch], y[batch]).backward()
            optimizer.step()
    flow.eval()
    with torch.no_grad():
        distribution = flow(x_test)
        log_q = distribution.log_prob(y_test)
        samples = distribution.sample((200,))
        coverage = sample_hpd_coverage(log_q, distribution.log_prob(samples).T, (0.5, 0.9))
    return float((log_p_test - log_q).mean()), coverage


@pytest.mark.parametrize("univariate", UNIVARIATES)
def test_matches_nsf_on_a_curved_conditional_density(univariate: str) -> None:
    gap, coverage = _fit_banana(4, flow_type="tarflow", univariate=univariate)
    nsf_gap, _ = _fit_banana(4, flow_type="nsf")
    assert math.isfinite(gap) and 0.0 < nsf_gap < 0.3
    # Same budget and width as NSF: at least as close to the true density
    # (up to seed noise), with calibrated joint highest-density regions.
    assert gap < nsf_gap + 0.05
    torch.testing.assert_close(coverage, torch.tensor([0.5, 0.9]), rtol=0, atol=0.06)
