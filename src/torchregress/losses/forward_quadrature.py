"""Scalar Gaussian-latent forward likelihoods by adaptive Gauss–Hermite quadrature."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

_SQRT2 = math.sqrt(2.0)
# Derivative-free re-centring passes use a fixed budget (no data-dependent stopping, so
# no host synchronisation): at least 3 passes and about 48 map evaluations per row,
# so coarse rules get more passes. One pass is exact for linear maps; the remaining
# passes refine nonlinear maps by successive parabolic interpolation.
_MIN_ADAPTIVE_PASSES = 3
_ADAPTIVE_EVALUATIONS = 48


def _adaptive_passes(quadrature_points: int) -> int:
    return max(_MIN_ADAPTIVE_PASSES, -(-_ADAPTIVE_EVALUATIONS // quadrature_points))


class GaussianLatentForwardObservationNLLLoss(nn.Module):
    r"""Approximate ``-log E_z Normal(y; observation_map(z), error)``.

    The latent is scalar ``z ~ Normal(latent_mean, latent_std)``. Inputs have
    shape ``[B]`` and ``observation_map`` receives nodes of shape ``[N,Q]`` for
    the N observed, positive-weight rows. It must act elementwise and return the
    same shape, on the same device. Negative observed values are valid. Missing
    entries may contain NaNs and never enter the map or likelihood. Errors must
    be finite and strictly positive; latent standard deviations may also be zero
    for the deterministic limit. All errors are standard deviations, not
    variances.

    Writing ``z = mean + std * t`` with ``t ~ N(0, 1)``, the integral
    ``I = int phi(t) p(y | mean + std*t) dt`` is evaluated by adaptive
    Gauss–Hermite quadrature: nodes ``x_q`` and weights ``w_q`` are placed on a
    Gaussian proposal ``N(c, tau^2)`` in ``t``,
    ``I ~= sqrt(2) tau sum_q w_q exp(x_q^2) phi(t_q) p(y | mean + std*t_q)``
    with ``t_q = c + sqrt(2) tau x_q``. ``(c, tau)`` approximate the mode and
    curvature scale of the integrand (prior times likelihood). They are found by
    a fixed number of derivative-free bracketing and parabolic-interpolation
    passes that evaluate the map on ``[N,Q]`` node grids without autograd. The
    rule is exact for linear maps at any order with at least three nodes, and
    stays accurate when the likelihood is much narrower than the prior pushed
    through the map. With one or two nodes the prior-centred rule ``c=0, tau=1``
    is used. ``c`` and ``tau`` are a stop-gradient change of variables. The
    integral does not depend on them, so gradients with respect to the latent
    parameters, the observation and the map parameters are those of the
    integral, up to quadrature error.

    Results remain a finite-order numerical approximation. Convergence slows
    when the integrand is far from Gaussian around its peak, for example when
    the map's derivative vanishes there, or when the likelihood keeps a plateau
    over the prior (observations near an asymptote of the map). Non-monotone
    maps whose preimage of ``y`` has several branches give multimodal integrands
    that one proposal cannot cover. Compare quadrature orders before relying on
    results. Multiple dependent observations of a shared latent must be
    integrated jointly, rather than multiplying calls to this loss.

    Computation uses at least fp32 with autocast disabled; fp64 inputs retain
    fp64 arithmetic. Quadrature constants follow the input device and dtype at
    use. The map must itself preserve autograd. It is called once per adaptive
    pass under ``torch.no_grad()`` and once more for the differentiable result,
    so it must be deterministic and free of side effects (no dropout or
    batch-norm statistics updates). ``mean`` reduction averages over
    positive-weight observed rows, ``sum`` sums weighted NLLs, and ``none``
    returns [B], zero at missing or zero-weight entries.
    """

    nodes: Tensor
    log_weights: Tensor

    def __init__(
        self,
        quadrature_points: int = 32,
        reduction: Literal["none", "sum", "mean"] = "mean",
    ) -> None:
        super().__init__()
        if (
            isinstance(quadrature_points, bool)
            or not isinstance(quadrature_points, int)
            or not 1 <= quadrature_points <= 256
        ):
            raise ValueError("quadrature_points must be an integer in [1,256]")
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("Unknown reduction")
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
        # Float64 host constants (not buffers): Module.to() must not cast them or
        # move them to devices without float64 (MPS); they are converted at use.
        self.nodes = torch.from_numpy(nodes)
        self.log_weights = torch.from_numpy(np.log(weights) - 0.5 * math.log(math.pi))
        # log(w) + x^2 is O(log spacing) even where w underflows fp32.
        self._log_weights_x2 = torch.from_numpy(np.log(weights) + nodes**2)
        self._constant_cache: dict[tuple[torch.device, torch.dtype], tuple[Tensor, ...]] = {}
        self.quadrature_points = quadrature_points
        self.reduction = reduction

    def _constants(self, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, ...]:
        key = (device, dtype)
        cached = self._constant_cache.get(key)
        if cached is None:
            cached = tuple(
                t.to(device=device, dtype=dtype)
                for t in (self.nodes, self.log_weights, self._log_weights_x2)
            )
            self._constant_cache[key] = cached
        return cached

    @staticmethod
    def _check_map_output(mapped: object, nodes: Tensor) -> Tensor:
        if (
            not isinstance(mapped, Tensor)
            or mapped.shape != nodes.shape
            or not mapped.is_floating_point()
            or mapped.device != nodes.device
        ):
            raise ValueError(
                "observation_map must return finite floating [N,Q] on the input device"
            )
        return mapped

    def _adapt(
        self,
        mean: Tensor,
        std: Tensor,
        y: Tensor,
        error: Tensor,
        observation_map: Callable[[Tensor], Tensor],
        x: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return detached proposal centre ``c`` and ``log tau`` in standardized ``t``.

        Each pass evaluates ``h(t) = -t^2/2 - r(t)^2/2`` on the current grid. For a
        unimodal integrand the peak lies between the best node's neighbours (or
        beyond an extremal best node), and that bracket is intersected with the
        earlier ones. A parabola through the best node and its neighbours (the
        log-likelihood part from residual differences, the prior part exactly)
        gives a vertex and curvature. Once the fit resolves the peak (concave,
        interior best node, neighbours within a few fitted scales) the next grid
        uses them. Otherwise it is centred on the bracket and wide enough to cover
        it, which shrinks the bracket like bisection. The last pass uses any
        concave fit. Non-finite or non-concave fits keep the previous proposal,
        and ``tau`` never exceeds the prior scale 1.
        """
        q = x.shape[0]
        centre = torch.zeros_like(mean)
        log_tau = torch.zeros_like(mean)
        inverse_error = error.reciprocal()[:, None]
        offsets = torch.arange(-1, 2, device=mean.device)
        bracket_low = torch.full_like(mean, -math.inf)
        bracket_high = torch.full_like(mean, math.inf)
        passes = _adaptive_passes(q)
        for step in range(passes):
            t = torch.addcmul(centre[:, None], (_SQRT2 * log_tau.exp())[:, None], x)
            mapped = observation_map(torch.addcmul(mean[:, None], std[:, None], t))
            f = self._check_map_output(mapped, t).to(t.dtype)
            r = (y[:, None] - f).mul_(inverse_error)
            # -2 h(t); non-finite map values are never selected.
            score = torch.addcmul(t.square(), r, r).nan_to_num_(nan=math.inf)
            best_score, best = score.min(dim=-1)
            index = best.clamp(1, q - 2)[:, None] + offsets
            t0, t1, t2 = t.gather(1, index).unbind(1)
            f0, f1, f2 = f.gather(1, index).unbind(1)
            r0, r1, r2 = r.gather(1, index).unbind(1)
            # Divided differences of -r^2/2 via (r_i - r_j)(r_i + r_j): the residual
            # difference comes straight from map differences, avoiding cancellation.
            d01 = -0.5 * ((f0 - f1) / error) * (r1 + r0) / (t1 - t0)
            d12 = -0.5 * ((f1 - f2) / error) * (r2 + r1) / (t2 - t1)
            second = (d12 - d01) / (t2 - t0)
            slope = d01 + second * (t1 - t0) - t1  # h'(t1), prior part exact
            curvature = 1.0 - 2.0 * second  # -h'' of the fitted parabola
            concave = curvature > 0
            # The integrand is at most prior-wide, so the proposal never exceeds tau=1.
            curvature = curvature.clamp_min(1.0)
            vertex = t1 + slope / curvature
            # A unimodal integrand peaks between the best node's neighbours; past an
            # edge node, allow one grid half-width (at least one prior sd) outward.
            outward = (t[:, -1] - centre).clamp_min(1.0)
            lower = torch.where(
                best == 0,
                t[:, 0] - outward,
                t.gather(1, (best - 1).clamp_min(0)[:, None]).squeeze(1),
            )
            upper = torch.where(
                best == q - 1,
                t[:, -1] + outward,
                t.gather(1, (best + 1).clamp_max(q - 1)[:, None]).squeeze(1),
            )
            # Brackets from earlier passes stay valid, so intersect them (falling back
            # to the local bracket if they are inconsistent, e.g. for a multimodal h).
            low, high = torch.maximum(bracket_low, lower), torch.minimum(bracket_high, upper)
            consistent = low < high
            lower = torch.where(consistent, low, lower)
            upper = torch.where(consistent, high, upper)
            vertex = torch.minimum(torch.maximum(vertex, lower), upper)
            # Trust the fitted scale only for a concave fit around an interior best node
            # whose neighbours lie within a few fitted scales (else the parabola spans
            # too much of a nonlinear h). Otherwise widen the next grid to cover the
            # bracket, so a coarse fit cannot shrink the proposal past the mode. The
            # last pass uses any concave fit and otherwise keeps the previous proposal.
            fitted = curvature.rsqrt()
            resolved = concave & (best > 0) & (best < q - 1) & (t2 - t0 <= 4.0 * fitted)
            ok = torch.isfinite(best_score)
            if step == passes - 1:
                resolved = concave
                ok = ok & concave
            cover = 0.5 * (upper - lower) / (_SQRT2 * x[-1])
            tau = torch.where(resolved, fitted, torch.maximum(cover, fitted.where(concave, 0.0)))
            tau = tau.clamp_max(1.0)
            vertex = torch.where(resolved, vertex, 0.5 * (lower + upper))
            ok = ok & torch.isfinite(vertex) & torch.isfinite(tau)
            centre = torch.where(ok, vertex, centre)
            log_tau = torch.where(ok, tau.log(), log_tau)
            bracket_low = torch.where(ok, lower, bracket_low)
            bracket_high = torch.where(ok, upper, bracket_high)
        return centre, log_tau

    def forward(
        self,
        latent_mean: Tensor,
        latent_std: Tensor,
        observed: Tensor,
        observation_std: Tensor,
        observation_map: Callable[[Tensor], Tensor],
        mask: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> Tensor:
        """Evaluate the observed scalar rows without detaching latent or map parameters."""
        inputs = (latent_mean, latent_std, observed, observation_std)
        if latent_mean.ndim != 1 or any(t.shape != latent_mean.shape for t in inputs):
            raise ValueError("All scalar inputs must have matching [B] shapes")
        if any(not t.is_floating_point() or t.device != latent_mean.device for t in inputs):
            raise ValueError("Scalar inputs must be floating point on the same device")
        if mask is None:
            mask = torch.ones_like(latent_mean, dtype=torch.bool)
        if (
            mask.shape != latent_mean.shape
            or mask.dtype != torch.bool
            or mask.device != latent_mean.device
        ):
            raise ValueError("mask must be boolean [B] on the input device")
        if weights is None:
            weights = torch.ones_like(latent_mean)
        if (
            weights.shape != latent_mean.shape
            or weights.device != latent_mean.device
            or not weights.is_floating_point()
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError("weights must be finite nonnegative floating [B] on the input device")
        dtype = torch.float32
        for tensor in (*inputs, weights):
            dtype = torch.promote_types(dtype, tensor.dtype)
        with torch.autocast(device_type=latent_mean.device.type, enabled=False):
            active = mask & (weights > 0)
            # Keep empty results differentiable without reading missing NaNs.
            safe_mean = torch.where(active, latent_mean.to(dtype), 0.0)
            safe_std = torch.where(active, latent_std.to(dtype), 0.0)
            result = (safe_mean + safe_std) * 0.0
            if active.any():
                mean, std, y, error = (t[active].to(dtype) for t in inputs)
                if not all(torch.isfinite(t).all() for t in (mean, std, y, error)):
                    raise ValueError("Observed inputs must be finite")
                if (std < 0).any() or (error <= 0).any():
                    raise ValueError("Latent std must be nonnegative and observation std positive")
                x, log_weights, log_weights_x2 = self._constants(mean.device, dtype)
                if self.quadrature_points >= 3:
                    with torch.no_grad():
                        centre, log_tau = self._adapt(
                            mean.detach(),
                            std.detach(),
                            y.detach(),
                            error.detach(),
                            observation_map,
                            x,
                        )
                    t = torch.addcmul(centre[:, None], (_SQRT2 * log_tau.exp())[:, None], x)
                    # log w + x^2 - t^2/2 for t = c + sqrt(2) tau x, plus the Jacobian
                    # log(sqrt(2) tau) and the -log(sqrt(2 pi)) of phi(t).
                    log_rule = (
                        torch.add(log_weights_x2, t.square(), alpha=-0.5)
                        + (log_tau - 0.5 * math.log(math.pi))[:, None]
                    )
                else:
                    t = x.expand(mean.shape[0], -1) * _SQRT2
                    log_rule = log_weights.expand(mean.shape[0], -1)
                nodes = mean[:, None] + std[:, None] * t
                mapped = self._check_map_output(observation_map(nodes), nodes)
                if not torch.isfinite(mapped).all():
                    raise ValueError(
                        "observation_map must return finite floating [N,Q] on the input device"
                    )
                # Preserve higher-precision output from a user-provided map.
                dtype = torch.promote_types(dtype, mapped.dtype)
                mapped, y, error = mapped.to(dtype), y.to(dtype), error.to(dtype)
                residual = (y[:, None] - mapped) / error[:, None]
                log_likelihood = (
                    -0.5 * residual.square() - error[:, None].log() - 0.5 * math.log(2 * math.pi)
                )
                log_integral = torch.logsumexp(log_likelihood + log_rule.to(dtype), dim=-1)
                if not torch.isfinite(log_integral).all():
                    raise ValueError(
                        "Nonfinite quadrature integral; use higher precision or rescale inputs"
                    )
                result = result.to(dtype).masked_scatter(active, -log_integral)
            effective = torch.where(active, weights.to(result.dtype), 0.0)
            result = result * effective
            if self.reduction == "none":
                return result
            if self.reduction == "sum":
                return result.sum()
            denominator = effective.sum()
            return result.sum() / denominator if active.any() else result.sum()
