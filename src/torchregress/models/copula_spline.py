"""Correlated spline flows with analytic marginals for missing dimensions."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GaussianCopulaSplineHead(nn.Module):
    """Conditional vector flow: a full Gaussian followed by coordinate splines.

    Every coordinate has a context-dependent monotone rational-quadratic spline;
    the Gaussian base has a full Cholesky covariance. Consequently the output
    is a *joint* normalizing spline flow, but its copula is restricted to the
    Gaussian family. In particular it cannot represent arbitrary tail
    dependence or multimodal conditional copulas.

    Because each spline acts on one coordinate, arbitrary missing coordinates
    are marginalized exactly in the intrinsic distribution by selecting the
    corresponding Gaussian subvector. Gaussian observation noise is integrated
    separately using caller-supplied standard-normal nodes; finite-node estimates
    are approximate and must be checked for convergence.
    """

    row: Tensor
    col: Tensor

    def __init__(
        self,
        context_dim: int,
        target_dim: int,
        *,
        width: int = 128,
        bins: int = 8,
        bound: float = 5.0,
        min_scale: float = 1e-4,
    ) -> None:
        super().__init__()
        if min(context_dim, target_dim, width) < 1 or bins < 2:
            raise ValueError("Positive dimensions and at least two bins are required")
        if not math.isfinite(bound) or bound <= 0 or not 0 < min_scale < 1:
            raise ValueError("Positive spline bound and a scale floor below one required")
        # Import lazily so torchregress without the optional flows extra still imports.
        try:
            from zuko.transforms import MonotonicRQSTransform
        except ImportError as error:
            raise ImportError("GaussianCopulaSplineHead requires the flows extra (zuko)") from error
        self._transform_class = MonotonicRQSTransform
        self.target_dim, self.bins, self.bound, self.min_scale = target_dim, bins, bound, min_scale
        row, col = torch.tril_indices(target_dim, target_dim)
        self.register_buffer("row", row, persistent=False)
        self.register_buffer("col", col, persistent=False)
        self.net = nn.Sequential(
            nn.Linear(context_dim, width),
            nn.GELU(),
            nn.Linear(width, target_dim + len(row) + target_dim * (3 * bins - 1)),
        )
        nn.init.zeros_(self._out.weight)
        nn.init.zeros_(self._out.bias)
        with torch.no_grad():
            # softplus(raw)+floor = 1 at initialization, making the initial
            # transform and base covariance the identity.
            self._out.bias[target_dim : target_dim + len(row)][row == col] = math.log(
                math.expm1(1 - min_scale)
            )

    @property
    def _out(self) -> nn.Linear:
        layer = self.net[-1]
        assert isinstance(layer, nn.Linear)
        return layer

    def _distribution_parameters(self, context: Tensor, *, work_dtype=None):
        if context.ndim != 2 or not torch.isfinite(context).all():
            raise ValueError("Context must be finite [B,C]")
        parameter_dtype = self._out.weight.dtype
        if parameter_dtype not in (torch.float32, torch.float64):
            raise ValueError("Keep spline-flow parameters in float32 or float64")
        raw = self.net(context.to(parameter_dtype))
        if work_dtype is not None:
            raw = raw.to(work_dtype)
        d, n = self.target_dim, len(self.row)
        mean = raw[:, :d]
        packed = raw[:, d : d + n]
        packed = torch.where(self.row == self.col, F.softplus(packed) + self.min_scale, packed)
        lower = raw.new_zeros((len(raw), d, d))
        lower[:, self.row, self.col] = packed
        spline = raw[:, d + n :].reshape(len(raw), d, 3 * self.bins - 1)
        widths, heights, derivatives = spline.split((self.bins, self.bins, self.bins - 1), -1)
        transform = self._transform_class(widths, heights, derivatives, bound=self.bound)
        if not torch.isfinite(lower).all() or not torch.isfinite(mean).all():
            raise FloatingPointError("Nonfinite Gaussian base parameters")
        return mean, lower, transform

    def _intrinsic(
        self, mean: Tensor, lower: Tensor, transform, value: Tensor, mask: Tensor
    ) -> Tensor:
        # value is [B,D] or [K,B,D], mask is always [B,D].
        if value.ndim not in (2, 3) or value.shape[-2:] != mask.shape:
            raise ValueError("Values must be [B,D] or [K,B,D], matching the mask")
        if mask.dtype != torch.bool or mask.shape != mean.shape:
            raise ValueError("A boolean [B,D] mask is required")
        expanded = mask if value.ndim == 2 else mask.unsqueeze(0)
        if not torch.isfinite(value.masked_select(expanded.expand_as(value))).all():
            raise ValueError("Observed values must be finite")
        safe = value.masked_fill(~expanded, 0)
        latent = transform.inv(safe)
        log_jacobian = transform.log_abs_det_jacobian(latent, safe)
        if not torch.isfinite(latent).all() or not torch.isfinite(log_jacobian).all():
            raise FloatingPointError("Nonfinite spline inverse or Jacobian")
        covariance = lower @ lower.transpose(-1, -2)
        result = safe[..., 0] * 0
        for pattern in torch.unique(mask, dim=0):
            rows = (mask == pattern).all(-1).nonzero().flatten()
            columns = pattern.nonzero().flatten()
            if columns.numel() == 0:
                continue
            cov = covariance[rows][:, columns][:, :, columns]
            chol = torch.linalg.cholesky(cov)
            z = latent.index_select(-2, rows).index_select(-1, columns)
            residual = z - mean[rows][:, columns]
            whitened = torch.linalg.solve_triangular(
                chol, residual.unsqueeze(-1), upper=False
            ).squeeze(-1)
            normal = -0.5 * (columns.numel() * math.log(2 * math.pi) + whitened.square().sum(-1))
            normal = normal - torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(-1)
            jac = log_jacobian.index_select(-2, rows).index_select(-1, columns).sum(-1)
            result = result.index_copy(-1, rows, normal - jac)
        return result

    def intrinsic_log_prob(self, context: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        """Exact intrinsic subvector density for each row; empty rows return zero."""
        dtype = torch.promote_types(self._out.weight.dtype, value.dtype)
        with torch.autocast(device_type=context.device.type, enabled=False):
            mean, lower, transform = self._distribution_parameters(context, work_dtype=dtype)
            return self._intrinsic(mean, lower, transform, value.to(dtype), mask)

    def observed_log_prob(
        self, context: Tensor, value: Tensor, sigma: Tensor, mask: Tensor, nodes: Tensor
    ) -> tuple[Tensor, Tensor]:
        """QMC-convolve quoted independent Gaussian errors over observed axes.

        Returns log density and relative importance ESS per row. Nodes must be
        standard-normal draws [K,D]. Missing-coordinate integration is analytic;
        the remaining noise integral is not certified by a finite ESS alone.
        """
        if sigma.shape != mask.shape or nodes.ndim != 2 or nodes.shape[1] != self.target_dim:
            raise ValueError("Noise or node shape differs")
        if len(nodes) < 2 or not torch.isfinite(nodes).all():
            raise ValueError("At least two finite standard-normal nodes are required")
        if not torch.isfinite(sigma[mask]).all() or (sigma[mask] < 0).any():
            raise ValueError("Observed noise scales must be finite and nonnegative")
        dtype = torch.promote_types(
            self._out.weight.dtype, torch.promote_types(value.dtype, sigma.dtype)
        )
        dtype = torch.promote_types(dtype, nodes.dtype)
        with torch.autocast(device_type=context.device.type, enabled=False):
            mean, lower, transform = self._distribution_parameters(context, work_dtype=dtype)
            safe_value = value.to(dtype).masked_fill(~mask, 0)
            safe_sigma = sigma.to(dtype).masked_fill(~mask, 0)
            perturbed = safe_value[None] + nodes.to(dtype)[:, None] * safe_sigma[None]
            log_weights = self._intrinsic(mean, lower, transform, perturbed, mask)
            log_sum = torch.logsumexp(log_weights, dim=0)
            log_prob = log_sum - math.log(len(nodes))
            relative_ess = (2 * log_sum - torch.logsumexp(2 * log_weights, dim=0)).exp() / len(
                nodes
            )
            # Empty rows have no likelihood factor and exactly unit relative ESS.
            relative_ess = torch.where(mask.any(-1), relative_ess, torch.ones_like(relative_ess))
            return log_prob, relative_ess.detach()

    def sample(self, context: Tensor, count: int, *, generator=None) -> Tensor:
        """Draw correlated complete vectors with shape [sample,B,D]."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("A positive sample count is required")
        with torch.autocast(device_type=context.device.type, enabled=False):
            mean, lower, transform = self._distribution_parameters(context)
            eps = torch.randn(
                (count, *mean.shape), device=mean.device, dtype=mean.dtype, generator=generator
            )
            latent = mean[None] + torch.einsum("bij,kbj->kbi", lower, eps)
            return transform(latent)
