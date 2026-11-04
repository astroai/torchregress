"""Differentiable survey-bias splines with explicit gauge conventions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn


def _basis_inside(x: Tensor, knots: Tensor, degree: int, right_end: Tensor) -> Tensor:
    basis = ((x[..., None] >= knots[:-1]) & (x[..., None] < knots[1:])).to(x.dtype)
    final_interval = (knots[:-1] < right_end) & (knots[1:] == right_end)
    basis = torch.where((x == right_end)[..., None], final_interval.to(x.dtype), basis)
    for p in range(1, degree + 1):
        n = knots.numel() - p - 1
        left_den = knots[p : p + n] - knots[:n]
        right_den = knots[p + 1 : p + n + 1] - knots[1 : n + 1]
        left_safe = torch.where(left_den != 0, left_den, torch.ones_like(left_den))
        right_safe = torch.where(right_den != 0, right_den, torch.ones_like(right_den))
        left = (x[..., None] - knots[:n]) / left_safe * (left_den != 0)
        right = (knots[p + 1 : p + n + 1] - x[..., None]) / right_safe * (right_den != 0)
        basis = left * basis[..., :n] + right * basis[..., 1 : n + 1]
    return basis


def bspline_basis_with_derivative(
    x: Tensor,
    knots: Tensor,
    degree: int = 3,
    *,
    boundary: Literal["linear", "constant", "error"] = "linear",
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute spline values and derivatives in at least float32, outside AMP."""
    if not x.is_floating_point():
        raise ValueError("Spline inputs must be floating point")
    with torch.autocast(device_type=x.device.type, enabled=False):
        work_x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return _bspline_basis_no_autocast(work_x, knots, degree, boundary=boundary)


def _bspline_basis_no_autocast(
    x: Tensor,
    knots: Tensor,
    degree: int,
    *,
    boundary: Literal["linear", "constant", "error"],
) -> tuple[Tensor, Tensor, Tensor]:
    """Return B-spline values, analytic derivatives, and an out-of-domain flag.

    ``knots`` is the full nondecreasing clamped knot vector, with exactly
    ``degree + 1`` repeated boundary knots. Outside the supported interval the
    basis can continue linearly, stay at the boundary value, or reject
    extrapolation. Out-of-domain values are flagged for every policy.
    """
    if boundary not in {"linear", "constant", "error"}:
        raise ValueError("Unknown spline boundary policy")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be a positive integer")
    knots = knots.to(device=x.device, dtype=x.dtype)
    if knots.ndim != 1 or knots.numel() < 2 * degree + 2:
        raise ValueError("Insufficient one-dimensional knot vector")
    if not torch.isfinite(knots).all() or not torch.isfinite(x).all():
        raise ValueError("Spline inputs and knots must be finite")
    if (knots[1:] < knots[:-1]).any():
        raise ValueError("Knots must be nondecreasing")
    lo, hi = knots[degree], knots[-degree - 1]
    if (
        not lo < hi
        or not (knots[: degree + 1] == lo).all()
        or not (knots[-degree - 1 :] == hi).all()
    ):
        raise ValueError("Require clamped knots with nonempty domain")
    _, multiplicities = torch.unique_consecutive(knots, return_counts=True)
    if (
        multiplicities[0] != degree + 1
        or multiplicities[-1] != degree + 1
        or (multiplicities[1:-1] > degree).any()
    ):
        raise ValueError("Knot multiplicities exceed the supported clamped spline")
    ood = (x < lo) | (x > hi)
    if boundary == "error" and ood.any():
        raise ValueError("Input outside spline knot domain")
    inside = torch.where(x < lo, lo, torch.where(x > hi, hi, x))
    basis = _basis_inside(inside, knots, degree, hi)
    lower = _basis_inside(inside, knots, degree - 1, hi)
    n = knots.numel() - degree - 1
    a = knots[degree : degree + n] - knots[:n]
    b = knots[degree + 1 : degree + n + 1] - knots[1 : n + 1]
    a_safe = torch.where(a != 0, a, torch.ones_like(a))
    b_safe = torch.where(b != 0, b, torch.ones_like(b))
    derivative = degree * (
        lower[..., :n] / a_safe * (a != 0) - lower[..., 1 : n + 1] / b_safe * (b != 0)
    )
    if boundary == "linear":
        distance = torch.where(x < lo, x - lo, torch.where(x > hi, x - hi, torch.zeros_like(x)))
        basis = basis + distance[..., None] * derivative
    elif boundary == "constant":
        derivative = derivative.masked_fill(ood[..., None], 0)
    return basis, derivative, ood


class _SplineBase(nn.Module):
    input_center: Tensor
    input_scale: Tensor
    output_scale: Tensor
    coefficients: nn.ParameterList

    def __init__(
        self,
        knots: Sequence[Tensor],
        degree: int = 3,
        input_center: Tensor | None = None,
        input_scale: Tensor | None = None,
        output_scale: Tensor | None = None,
        boundary: Literal["linear", "constant", "error"] = "constant",
    ) -> None:
        super().__init__()
        if not knots:
            raise ValueError("At least one input dimension is required")
        if boundary not in {"linear", "constant", "error"}:
            raise ValueError("Unknown spline boundary policy")
        self.degree = degree
        self.boundary = boundary
        self.dims = len(knots)
        for j, knot in enumerate(knots):
            k = torch.as_tensor(knot).clone().detach()
            if not k.is_floating_point():
                k = k.float()
            bspline_basis_with_derivative(k[degree : degree + 1], k, degree)
            self.register_buffer(f"knots_{j}", k)
        center = torch.zeros(self.dims) if input_center is None else torch.as_tensor(input_center)
        scale = torch.ones(self.dims) if input_scale is None else torch.as_tensor(input_scale)
        out = torch.ones(self.dims) if output_scale is None else torch.as_tensor(output_scale)
        if any(
            x.shape != (self.dims,) or not torch.isfinite(x).all() for x in (center, scale, out)
        ):
            raise ValueError("Center and scales must be finite [D] vectors")
        if (scale <= 0).any() or (out <= 0).any():
            raise ValueError("Input and output scales must be positive")
        self.register_buffer("input_center", center.clone().detach())
        self.register_buffer("input_scale", scale.clone().detach())
        self.register_buffer("output_scale", out.clone().detach())

    def _bases(self, theta: Tensor) -> tuple[list[Tensor], list[Tensor], Tensor]:
        if theta.ndim != 2 or theta.shape[-1] != self.dims or not torch.isfinite(theta).all():
            raise ValueError("theta must be finite [B,D]")
        standardized = (theta - self.input_center) / self.input_scale
        values, derivatives, ood = [], [], []
        for j in range(self.dims):
            basis, deriv, flag = bspline_basis_with_derivative(
                standardized[:, j], getattr(self, f"knots_{j}"), self.degree, boundary=self.boundary
            )
            values.append(basis)
            derivatives.append(deriv / self.input_scale[j])
            ood.append(flag)
        return values, derivatives, torch.stack(ood, dim=-1)

    def support_diagnostics(self, theta: Tensor) -> Tensor:
        """Per-input numerical extrapolation flags, not physical-support proof."""
        with torch.autocast(device_type=theta.device.type, enabled=False):
            work = theta.float() if theta.dtype in (torch.float16, torch.bfloat16) else theta
            return self._bases(work)[2]

    def regularization(self) -> Tensor:
        """Squared coefficients plus adjacent second differences (unscaled)."""
        total = next(self.parameters()).new_zeros(())
        for coeff in self.coefficients:
            total = total + coeff.square().sum()
            if coeff.shape[-1] >= 3:
                total = total + coeff.diff(n=2, dim=-1).square().sum()
        return total


class RelativeSurveySplineHead(_SplineBase):
    """Additive per-source corrections with exact per-target zero-sum gauge.

    Weights are fixed, nonnegative and normalized across comparable sources for
    each output dimension. Entries with zero weight are unavailable; they are
    zeroed rather than presented as calibrated estimates.
    """

    weights: Tensor

    def __init__(
        self,
        knots: Sequence[Tensor],
        weights: Tensor,
        degree: int = 3,
        input_center: Tensor | None = None,
        input_scale: Tensor | None = None,
        output_scale: Tensor | None = None,
        boundary: Literal["linear", "constant", "error"] = "constant",
    ) -> None:
        super().__init__(knots, degree, input_center, input_scale, output_scale, boundary)
        w = torch.as_tensor(weights).clone().detach()
        if not w.is_floating_point():
            w = w.float()
        if w.ndim != 2 or w.shape[1] != self.dims or not torch.isfinite(w).all() or (w < 0).any():
            raise ValueError("weights must be finite nonnegative [S,D]")
        if (w.sum(0) <= 0).any():
            raise ValueError("Each target needs at least one comparable source")
        self.register_buffer("weights", w / w.sum(0))
        self.coefficients = nn.ParameterList(
            [nn.Parameter(torch.zeros(w.shape[0], self.dims, len(k) - degree - 1)) for k in knots]
        )

    def forward(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=theta.device.type, enabled=False):
            work = theta.float() if theta.dtype in (torch.float16, torch.bfloat16) else theta
            return self._forward_no_autocast(work)

    def _forward_no_autocast(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        basis, deriv, _ = self._bases(theta)
        raw = torch.stack(
            [
                torch.einsum("bn,sdn->bsd", b, c.to(theta.dtype))
                for b, c in zip(basis, self.coefficients, strict=True)
            ]
        ).sum(0)
        jac = torch.stack(
            [
                torch.einsum("bn,sdn->bsd", d, c.to(theta.dtype))
                for d, c in zip(deriv, self.coefficients, strict=True)
            ],
            dim=-1,
        )
        weights = self.weights.to(theta.dtype)
        weights = weights / weights.sum(0)
        available = (weights > 0).to(raw.dtype)
        average = (raw * weights).sum(1, keepdim=True)
        average_jac = (jac * weights[None, :, :, None]).sum(1, keepdim=True)
        values = (raw - average) * available * self.output_scale
        jac = (
            (jac - average_jac)
            * available[None, :, :, None]
            * self.output_scale[None, None, :, None]
        )
        return values, jac


class CommonScaleSplineHead(_SplineBase):
    """Unconstrained common correction: only fit with independent anchors."""

    def __init__(
        self,
        knots: Sequence[Tensor],
        degree: int = 3,
        input_center: Tensor | None = None,
        input_scale: Tensor | None = None,
        output_scale: Tensor | None = None,
        boundary: Literal["linear", "constant", "error"] = "constant",
    ) -> None:
        super().__init__(knots, degree, input_center, input_scale, output_scale, boundary)
        self.coefficients = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.dims, len(k) - degree - 1)) for k in knots]
        )

    def forward(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=theta.device.type, enabled=False):
            work = theta.float() if theta.dtype in (torch.float16, torch.bfloat16) else theta
            return self._forward_no_autocast(work)

    def _forward_no_autocast(self, theta: Tensor) -> tuple[Tensor, Tensor]:
        basis, deriv, _ = self._bases(theta)
        values = torch.stack(
            [
                torch.einsum("bn,dn->bd", b, c.to(theta.dtype))
                for b, c in zip(basis, self.coefficients, strict=True)
            ]
        ).sum(0)
        jac = torch.stack(
            [
                torch.einsum("bn,dn->bd", d, c.to(theta.dtype))
                for d, c in zip(deriv, self.coefficients, strict=True)
            ],
            dim=-1,
        )
        return values * self.output_scale, jac * self.output_scale[None, :, None]


class ConditionBiasHead(_SplineBase):
    """Quality-conditioned source biases exactly zero at source reference q."""

    reference_conditions: Tensor

    def __init__(
        self,
        knots: Sequence[Tensor],
        reference_conditions: Tensor,
        degree: int = 3,
        input_center: Tensor | None = None,
        input_scale: Tensor | None = None,
        output_scale: Tensor | None = None,
        boundary: Literal["linear", "constant", "error"] = "constant",
    ) -> None:
        super().__init__(knots, degree, input_center, input_scale, output_scale, boundary)
        ref = torch.as_tensor(reference_conditions).clone().detach()
        if not ref.is_floating_point():
            ref = ref.float()
        if ref.ndim != 2 or not torch.isfinite(ref).all():
            raise ValueError("reference_conditions must be finite [S,Q]")
        self.register_buffer("reference_conditions", ref)
        self.coefficients = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(ref.shape[0], self.dims, ref.shape[1], len(k) - degree - 1)
                )
                for k in knots
            ]
        )

    def forward(self, theta: Tensor, conditions: Tensor) -> tuple[Tensor, Tensor]:
        with torch.autocast(device_type=theta.device.type, enabled=False):
            work = theta.float() if theta.dtype in (torch.float16, torch.bfloat16) else theta
            return self._forward_no_autocast(work, conditions.to(work.dtype))

    def _forward_no_autocast(self, theta: Tensor, conditions: Tensor) -> tuple[Tensor, Tensor]:
        if (
            conditions.shape != (theta.shape[0], *self.reference_conditions.shape)
            or not torch.isfinite(conditions).all()
        ):
            raise ValueError("conditions must be finite [B,S,Q]")
        basis, deriv, _ = self._bases(theta)
        dq = conditions - self.reference_conditions
        values = torch.stack(
            [
                torch.einsum("bn,bsq,sdqn->bsd", b, dq, c.to(theta.dtype))
                for b, c in zip(basis, self.coefficients, strict=True)
            ]
        ).sum(0)
        jac = torch.stack(
            [
                torch.einsum("bn,bsq,sdqn->bsd", d, dq, c.to(theta.dtype))
                for d, c in zip(deriv, self.coefficients, strict=True)
            ],
            dim=-1,
        )
        return values * self.output_scale, jac * self.output_scale[None, None, :, None]
