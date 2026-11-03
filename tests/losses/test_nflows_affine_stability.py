"""Sampling stability of the affine flows built by ``create_flow_model``.

``flow_type="maf"`` and ``"realnvp"``/``"cinn"``/``"nice"`` use zuko's affine
layers with two changes: a soft log-scale bound (``|log s| < 3``) and
conditioners that see soft-clipped inputs.  Without them a far-tail base draw can
send the inverse pass into a geometric cascade (finite samples of ``1e11`` on
``synthetic_multitarget_big``).
"""

from __future__ import annotations

import math
import os

import pytest
import torch

from torchregress.losses.nflows import (
    NormalizingFlowLoss,
    _soft_clip,
    _StableMAF,
    _StableRealNVP,
    create_flow_model,
)

zuko = pytest.importorskip("zuko")

AFFINE_TYPES = ["maf", "realnvp", "cinn", "nice"]
PAIRS = [(zuko.flows.MAF, _StableMAF), (zuko.flows.RealNVP, _StableRealNVP)]


def _devices() -> list[str]:
    devices = ["cpu"]
    target = os.environ.get("TORCHREGRESS_PARITY_DEVICE")
    if target and target != "cpu":
        devices.append(target)
    elif torch.cuda.is_available():
        devices.append("cuda")
    return devices


def test_soft_clip_is_odd_increasing_bounded_and_identity_near_zero():
    x = torch.linspace(-1e6, 1e6, 20001, dtype=torch.float64)
    y = _soft_clip(x, 10.0)
    assert torch.all(y.abs() < 10.0)
    assert torch.all(torch.diff(y) > 0)
    assert torch.allclose(_soft_clip(-x, 10.0), -y)
    small = torch.linspace(-0.1, 0.1, 11, dtype=torch.float64)
    assert torch.allclose(_soft_clip(small, 10.0), small, atol=1e-5)


@pytest.mark.parametrize("flow_type", AFFINE_TYPES)
def test_create_flow_model_uses_the_stable_affine_layers(flow_type):
    flow = create_flow_model(3, 4, flow_type, 2)
    expected = _StableMAF if flow_type == "maf" else _StableRealNVP
    assert type(flow) is expected
    for layer in flow.transform.transforms:
        assert layer.conditioner_bound == 10.0
    flow_wide = create_flow_model(3, 4, flow_type, 2, bound=25.0)
    assert all(layer.conditioner_bound == 25.0 for layer in flow_wide.transform.transforms)


@pytest.mark.parametrize(("zuko_cls", "stable_cls"), PAIRS)
def test_only_the_two_safeguards_differ_from_zuko(zuko_cls, stable_cls):
    """Same layout and state dict as zuko; zuko's numbers without the safeguards."""
    torch.manual_seed(0)
    ref = zuko_cls(features=4, context=3, transforms=3, hidden_features=[16, 16])
    loose = stable_cls(
        features=4,
        context=3,
        transforms=3,
        hidden_features=[16, 16],
        slope=1e-3,
        conditioner_bound=float("inf"),
    )
    assert isinstance(loose, zuko_cls)
    assert list(loose.state_dict()) == list(ref.state_dict())
    loose.load_state_dict(ref.state_dict())
    c = torch.randn(7, 3)
    y = 4.0 * torch.randn(7, 4)
    torch.testing.assert_close(loose(c).log_prob(y), ref(c).log_prob(y))


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("flow_type", AFFINE_TYPES)
def test_log_prob_is_exact_change_of_variables_and_inverse_round_trips(flow_type, device):
    torch.manual_seed(1)
    # MPS has no float64; assert_close then applies its float32 tolerances.
    dtype = torch.float32 if torch.device(device).type == "mps" else torch.float64
    flow = create_flow_model(3, 2, flow_type, 3, hidden_features=[16, 16])
    flow = flow.to(device=device, dtype=dtype)
    c = torch.randn(1, 2, dtype=dtype, device=device)
    dist = flow(c)
    # Far beyond the soft-clip bound so the clipped conditioner is exercised.
    x = torch.tensor([[-30.0, 0.7, 45.0]], dtype=dtype, device=device)

    def forward(v):
        return dist.transform(v.unsqueeze(0)).squeeze(0)

    jac = torch.autograd.functional.jacobian(forward, x.squeeze(0))
    z = dist.transform(x)
    expected = dist.base.log_prob(z) + torch.linalg.slogdet(jac).logabsdet
    torch.testing.assert_close(dist.log_prob(x), expected)
    torch.testing.assert_close(dist.transform.inv(z), x)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("flow_type", ["maf", "realnvp"])
def test_samples_have_log_prob_consistent_with_their_base_draws(flow_type, device):
    torch.manual_seed(2)
    flow = create_flow_model(4, 3, flow_type, 3, hidden_features=[16, 16]).to(device)
    loss = NormalizingFlowLoss(flow=flow)
    ctx = torch.randn(5, 3, device=device)
    samples = loss.sample(ctx, n_samples=64)
    assert samples.shape == (5, 64, 4)
    assert samples.device.type == torch.device(device).type
    flat = samples.reshape(-1, 4)
    ctx_r = ctx.repeat_interleave(64, dim=0)
    dist = flow(ctx_r)
    z = dist.transform(flat)
    ladj = dist.transform.log_abs_det_jacobian(flat, z)
    torch.testing.assert_close(
        loss.log_prob(ctx_r, flat), dist.base.log_prob(z) + ladj, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize(("zuko_cls", "stable_cls"), PAIRS)
def test_extrapolating_conditioner_does_not_explode_samples(zuko_cls, stable_cls):
    """Reproduction of the sampling cascade with an extrapolating conditioner.

    Scaling every hyper-network weight by 3 makes the conditioners respond
    strongly to large inputs, as a trained network does outside its data.  The
    plain zuko flow then produces astronomically large finite samples from
    ordinary base draws; the stable flow, with identical weights, does not.
    """
    torch.manual_seed(0)
    ref = zuko_cls(features=6, context=4, transforms=3, hidden_features=[64, 64])
    stable = stable_cls(features=6, context=4, transforms=3, hidden_features=[64, 64])
    stable.load_state_dict(ref.state_dict())
    with torch.no_grad():
        for flow in (ref, stable):
            for name, param in flow.named_parameters():
                if name.endswith("weight"):
                    param.mul_(3.0)
    c = torch.zeros(1, 4)
    u = torch.randn(200_000, 6, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref_max = float(ref(c).transform.inv(u).abs().amax())
        stable_x = stable(c).transform.inv(u)
    assert ref_max > 1e8  # the failure mode is reproduced
    assert torch.isfinite(stable_x).all()
    assert float(stable_x.abs().amax()) < 2e3


def test_log_scale_is_softly_bounded_by_three():
    flow = create_flow_model(2, 0, "maf", 1, hidden_features=[8])
    layer = flow.transform.transforms[0]
    t = layer.univariate(torch.zeros(5), torch.tensor([-1e6, -10.0, 0.0, 10.0, 1e6]))
    assert torch.all(t.log_scale.abs() < 3.0)
    assert math.isclose(float(t.log_scale[2]), 0.0)


def test_bound_validation_for_flow_types():
    with pytest.raises(ValueError, match="bound"):
        create_flow_model(1, 4, "bpf", 2, bound=10.0)
    for bad in (0.0, -3.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="bound"):
            create_flow_model(1, 4, "maf", 2, bound=bad)
    with pytest.raises(ValueError, match="slope"):
        create_flow_model(1, 4, "maf", 2, slope=1.5)


@pytest.mark.parametrize("flow_type", AFFINE_TYPES)
def test_scalar_target_uses_context_only_layers(flow_type):
    """With one feature there are no autoregressive/coupling inputs to clip."""
    torch.manual_seed(3)
    flow = create_flow_model(1, 3, flow_type, 2, hidden_features=[8, 8])
    loss = NormalizingFlowLoss(flow=flow)
    ctx = torch.randn(4, 3)
    samples = loss.sample(ctx, n_samples=16)
    assert samples.shape == (4, 16, 1)
    assert torch.isfinite(loss.log_prob(ctx, torch.randn(4, 1))).all()
