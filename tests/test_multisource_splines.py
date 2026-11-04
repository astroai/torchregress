"""Gauge and Jacobian checks, including linear extrapolation."""

import numpy as np
import pytest
import torch
from scipy.interpolate import BSpline

from torchregress.calibration.multisource import (
    CommonScaleSplineHead,
    ConditionBiasHead,
    RelativeSurveySplineHead,
    bspline_basis_with_derivative,
)


def knots():
    return torch.tensor(
        [-2.0, -2.0, -2.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.double
    )


def test_basis_and_derivatives_against_scipy_and_linear_tails():
    k = knots()
    x = torch.tensor(
        [-2.5, -2.0, -1.7, -0.3, 0.5, 1.9, 2.0, 2.5], dtype=torch.double, requires_grad=True
    )
    basis, derivative, ood = bspline_basis_with_derivative(x, k)
    coeff = np.eye(len(k) - 4)
    spline = BSpline(k.numpy(), coeff, 3, extrapolate=False)
    clipped = x.detach().numpy().clip(-2, 2)
    expected = spline(clipped) + (x.detach().numpy() - clipped)[:, None] * spline.derivative()(
        clipped
    )
    np.testing.assert_allclose(basis.detach(), expected, atol=2e-14)
    np.testing.assert_allclose(derivative.detach(), spline.derivative()(clipped), atol=2e-14)
    assert ood.tolist() == [True, False, False, False, False, False, False, True]
    probe = torch.randn(basis.shape[1], dtype=torch.double)
    grad = torch.autograd.grad((basis * probe).sum(), x)[0]
    torch.testing.assert_close(grad, (derivative * probe).sum(-1), atol=1e-12, rtol=1e-12)


def test_relative_weighted_gauge_and_jacobian_autograd():
    torch.manual_seed(2)
    w = torch.tensor([[1.0, 1.0], [1.0, 0.0], [2.0, 1.0]], dtype=torch.double)
    head = RelativeSurveySplineHead(
        [knots(), knots()],
        w,
        input_scale=torch.tensor([2.0, 0.5]),
        output_scale=torch.tensor([3.0, 0.2]),
    ).double()
    with torch.no_grad():
        for p in head.parameters():
            p.normal_()
    theta = torch.tensor([[0.4, -0.3], [4.5, 0.3]], dtype=torch.double, requires_grad=True)
    values, jac = head(theta)
    torch.testing.assert_close(
        (values * head.weights).sum(1), torch.zeros(2, 2, dtype=torch.double), atol=1e-13, rtol=0
    )
    torch.testing.assert_close(
        (jac * head.weights[None, :, :, None]).sum(1),
        torch.zeros(2, 2, 2, dtype=torch.double),
        atol=1e-13,
        rtol=0,
    )
    assert not head.support_diagnostics(theta)[0].any() and head.support_diagnostics(theta)[1, 0]
    for source in range(3):
        for target in range(2):
            actual = torch.autograd.grad(values[:, source, target].sum(), theta, retain_graph=True)[
                0
            ]
            torch.testing.assert_close(actual, jac[:, source, target], atol=1e-12, rtol=1e-12)


def test_reference_condition_is_zero_but_other_conditions_not_zero_sum():
    torch.manual_seed(4)
    head = ConditionBiasHead([knots(), knots()], torch.tensor([[1.0], [2.0]])).double()
    with torch.no_grad():
        for p in head.parameters():
            p.normal_()
    theta = torch.tensor([[0.3, -0.5]], dtype=torch.double, requires_grad=True)
    ref = head.reference_conditions[None]
    zero, zero_jac = head(theta, ref)
    assert not zero.any() and not zero_jac.any()
    values, jac = head(theta, ref + 1)
    actual = torch.autograd.grad(values[0, 0, 1], theta)[0]
    torch.testing.assert_close(actual[0], jac[0, 0, 1])
    assert values.sum(1).abs().max() > 0


def test_common_head_and_invalid_gauge():
    head = CommonScaleSplineHead([knots()]).double()
    with torch.no_grad():
        head.coefficients[0].fill_(2)
    value, jac = head(torch.tensor([[0.3]], dtype=torch.double))
    torch.testing.assert_close(value, torch.tensor([[2.0]], dtype=torch.double))
    torch.testing.assert_close(jac, torch.zeros_like(jac), atol=1e-13, rtol=0)
    with pytest.raises(ValueError, match="comparable"):
        RelativeSurveySplineHead([knots()], torch.zeros(2, 1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_exact_endpoints_in_all_supported_float_types(dtype):
    k = knots().to(dtype)
    values, deriv, ood = bspline_basis_with_derivative(torch.tensor([-2.0, 2.0], dtype=dtype), k)
    assert values[0, 0] == 1 and values[1, -1] == 1
    assert torch.count_nonzero(values) == 2 and not ood.any()
    assert deriv[0, 0] == -3 and deriv[1, -1] == 3


def test_constant_and_error_boundary_are_explicit():
    x = torch.tensor([-3.0, 3.0], dtype=torch.double, requires_grad=True)
    basis, derivative, ood = bspline_basis_with_derivative(x, knots(), boundary="constant")
    assert ood.all() and not derivative.any()
    assert basis[0, 0] == 1 and basis[1, -1] == 1
    grad = torch.autograd.grad(basis.sum(), x)[0]
    assert not grad.any()
    with pytest.raises(ValueError, match="outside"):
        bspline_basis_with_derivative(x, knots(), boundary="error")


def test_narrow_half_precision_knots_remain_finite_with_autocast():
    k = torch.tensor([0.0, 0.0, 0.0, 0.0, 1e-5, 1e-5, 1e-5, 1e-5])
    head = CommonScaleSplineHead([k]).half()
    x = torch.tensor([[5e-6]], dtype=torch.float16, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        values, jac = head(x)
    assert values.dtype == torch.float32 and jac.dtype == torch.float32
    assert torch.isfinite(values).all() and torch.isfinite(jac).all()
    assert not values.any() and not jac.any()
    values.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_knot_multiplicity_and_degree_are_validated():
    with pytest.raises(ValueError, match="degree"):
        bspline_basis_with_derivative(torch.tensor([0.0]), knots(), degree=True)
    bad = torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="multiplicities"):
        bspline_basis_with_derivative(torch.tensor([0.0]), bad)
