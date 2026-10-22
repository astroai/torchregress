"""BPF map boundaries, architecture contract and old-state compatibility."""

import pytest
import torch

from torchregress.losses.nflows import HAS_ZUKO, create_flow_model

pytestmark = pytest.mark.skipif(not HAS_ZUKO, reason="optional zuko dependency")


def _transform(logits):
    from torchregress.losses.nflows import _IdentityTailBernsteinTransform

    return _IdentityTailBernsteinTransform(logits)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_coefficients_are_identity_inside_and_outside_bounds(dtype):
    values = torch.tensor([-9.0, -5.0, -4.99999, -1.0, 0.0, 1.0, 4.99999, 5.0, 9.0], dtype=dtype)
    transform = _transform(torch.zeros(len(values), 9, dtype=dtype))
    output, logdet = transform.call_and_ladj(values)
    exterior = values.abs() >= 5
    assert torch.equal(output[exterior], values[exterior])
    assert torch.equal(logdet[exterior], torch.zeros_like(logdet[exterior]))
    torch.testing.assert_close(
        output, values, rtol=0, atol=3e-6 if dtype == torch.float32 else 1e-12
    )
    torch.testing.assert_close(transform.inv(values)[exterior], values[exterior], rtol=0, atol=0)


def test_true_endpoints_are_continuous_with_unit_derivative():
    torch.manual_seed(119)
    values = torch.tensor(
        [-5 - 1e-9, -5.0, -5 + 1e-9, 5 - 1e-9, 5.0, 5 + 1e-9], dtype=torch.float64
    )
    # One shared nonidentity polynomial, evaluated from both sides.
    logits = torch.randn(1, 9, dtype=values.dtype).expand(len(values), -1)
    transform = _transform(logits)
    output, logdet = transform.call_and_ladj(values)
    torch.testing.assert_close(output, values, rtol=0, atol=1e-12)
    torch.testing.assert_close(logdet, torch.zeros_like(logdet), rtol=0, atol=1e-7)


def test_nonidentity_polynomial_is_monotonic_with_valid_inverse_jacobian_and_gradients():
    torch.manual_seed(523)
    logits = torch.randn(1, 9, dtype=torch.float64, requires_grad=True)
    transform = _transform(logits)
    values = torch.linspace(-7, 7, 401, dtype=logits.dtype)
    output, logdet = transform.call_and_ladj(values)
    assert torch.all(output[1:] > output[:-1])
    assert torch.isfinite(logdet).all()
    torch.testing.assert_close(transform.inv(output), values, rtol=0, atol=1e-6)
    point = torch.tensor(1.23, dtype=logits.dtype, requires_grad=True)
    value = transform(point)
    derivative = torch.autograd.grad(value, point, create_graph=True)[0]
    torch.testing.assert_close(transform.log_abs_det_jacobian(point, value), derivative.log())
    gradient = torch.autograd.grad((output.square() + logdet).sum(), logits)[0]
    assert torch.isfinite(gradient).all()


def test_factory_honors_requested_bpf_transform_count_and_hidden_widths():
    flow = create_flow_model(
        3, context_dim=2, flow_type="bpf", n_transforms=4, hidden_features=[11, 13], degree=8
    )
    assert len(flow.transform.transforms) == 4
    first = flow.transform.transforms[0].hyper
    assert first[0].out_features == 11
    assert first[2].out_features == 13
    assert flow.context == 2
    with pytest.raises(ValueError, match="degree"):
        create_flow_model(3, flow_type="bpf", degree=0)


def test_old_actual_architecture_state_loads_strictly_and_interior_scores_match():
    from zuko.flows import BPF

    torch.manual_seed(52)
    old = BPF(features=3, context=2, transforms=3, hidden_features=[64, 64], degree=8).double()
    fixed = create_flow_model(
        3, context_dim=2, flow_type="bpf", n_transforms=3, hidden_features=[64, 64], degree=8
    ).double()
    assert list(old.state_dict()) == list(fixed.state_dict())
    fixed.load_state_dict(old.state_dict(), strict=True)
    context = torch.randn(8, 2, dtype=torch.float64)
    values = torch.randn(8, 3, dtype=torch.float64) * 0.2
    torch.testing.assert_close(
        fixed(context).log_prob(values), old(context).log_prob(values), rtol=1e-12, atol=1e-12
    )
    exterior = torch.tensor([[-7.0, 7.0, -9.0]], dtype=torch.float64).expand(8, -1)
    torch.testing.assert_close(fixed(context).transform(exterior), exterior, rtol=0, atol=0)
