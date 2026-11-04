"""Correlated spline flows with analytic marginals for missing dimensions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..utils.pytorch_compat import _collect_transform_cycles

# A deferred validation: (0-d boolean flag, error class, message).
_Check = tuple[Tensor, type[Exception], str]


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

    def _flow_parameters(self, context: Tensor, work_dtype: torch.dtype | None):
        """Gaussian base and spline parameters plus a deferred finiteness flag."""
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
        finite = torch.isfinite(packed).all() & torch.isfinite(mean).all()
        return mean, lower, transform, finite

    def _distribution_parameters(self, context: Tensor, *, work_dtype=None):
        mean, lower, transform, finite = self._flow_parameters(context, work_dtype)
        if not finite:
            raise FloatingPointError("Nonfinite Gaussian base parameters")
        return mean, lower, transform

    def _intrinsic(
        self,
        mean: Tensor,
        lower: Tensor,
        transform,
        value: Tensor,
        mask: Tensor,
        checks: Sequence[_Check],
    ) -> Tensor:
        """Exact marginal log density of the observed coordinates of each row.

        ``value`` is [B,D] or [K,B,D]; ``mask`` is always [B,D]. ``checks`` are
        deferred (flag, error, message) validations of the caller; they are read
        together with this method's own checks in a single device-to-host copy
        and the first failing one is raised.

        Missing coordinates are marginalized without grouping rows by pattern.
        With ``Sigma = L L^T`` and observed set ``o`` of a row, the masked matrix
        ``S = diag(m) Sigma diag(m) + diag(1 - m)`` equals ``Sigma_oo`` on ``o``,
        the identity on the missing set and zero between them. Hence
        ``det S = det Sigma_oo`` and, for a residual that is zero off ``o``,
        ``r^T S^{-1} r = r_o^T Sigma_oo^{-1} r_o``: one batched Cholesky of ``S``
        gives every row's exact marginal Gaussian density. Rows without missing
        coordinates use ``L`` directly. Empty rows give exactly zero.
        """
        if value.ndim not in (2, 3) or value.shape[-2:] != mask.shape:
            raise ValueError("Values must be [B,D] or [K,B,D], matching the mask")
        if mask.dtype != torch.bool or mask.shape != mean.shape:
            raise ValueError("A boolean [B,D] mask is required")
        expanded = mask if value.ndim == 2 else mask.unsqueeze(0)
        finite_value = torch.isfinite(value)
        observed_finite = (finite_value | ~expanded).all()
        # Nonfinite observed entries are replaced as well, so the spline never
        # sees them before the deferred check below raises.
        safe = torch.where(expanded & finite_value, value, torch.zeros_like(value))
        latent = transform.inv(safe)
        log_jacobian = transform.log_abs_det_jacobian(latent, safe)
        spline_finite = torch.isfinite(latent).all() & torch.isfinite(log_jacobian).all()
        checks = [
            *checks,
            (observed_finite, ValueError, "Observed values must be finite"),
            (spline_finite, FloatingPointError, "Nonfinite spline inverse or Jacobian"),
        ]
        *passed, complete = torch.stack([flag for flag, _, _ in checks] + [mask.all()]).tolist()
        for good, (_, error, message) in zip(passed, checks, strict=True):
            if not good:
                raise error(message)
        if complete:
            chol = lower
            residual = latent - mean
            jacobian = log_jacobian.sum(-1)
            count = torch.full_like(jacobian, mask.shape[-1])
        else:
            weight = mask.to(lower.dtype)
            masked_lower = lower * weight.unsqueeze(-1)
            covariance = masked_lower @ masked_lower.transpose(-1, -2)
            covariance = covariance + torch.diag_embed(1 - weight).to(
                device=covariance.device, dtype=covariance.dtype
            )
            chol, info = torch.linalg.cholesky_ex(covariance)
            if info.any():
                raise torch.linalg.LinAlgError(
                    "Observed marginal covariance is not positive definite"
                )
            zero = torch.zeros((), dtype=latent.dtype, device=latent.device)
            residual = torch.where(expanded, latent - mean, zero)
            jacobian = torch.where(expanded, log_jacobian, zero).sum(-1)
            count = weight.sum(-1)
        # Solve all K node sets as right-hand sides of one batched system.
        rhs = residual.unsqueeze(-1) if residual.ndim == 2 else residual.permute(1, 2, 0)
        whitened = torch.linalg.solve_triangular(chol, rhs, upper=False)
        quadratic = whitened.square().sum(-2)
        quadratic = quadratic.squeeze(-1) if residual.ndim == 2 else quadratic.T
        log_det = torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(-1)
        return -0.5 * (count * math.log(2 * math.pi) + quadratic) - log_det - jacobian

    def intrinsic_log_prob(self, context: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        """Exact intrinsic subvector density for each row; empty rows return zero."""
        # The spline inverse leaves a transform <-> inverse reference cycle that
        # holds the spline parameters (and their autograd graph) until GC.
        with _collect_transform_cycles():
            return self._intrinsic_log_prob(context, value, mask)

    def _intrinsic_log_prob(self, context: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        dtype = torch.promote_types(self._out.weight.dtype, value.dtype)
        with torch.autocast(device_type=context.device.type, enabled=False):
            mean, lower, transform, finite = self._flow_parameters(context, dtype)
            base_check: _Check = (finite, FloatingPointError, "Nonfinite Gaussian base parameters")
            return self._intrinsic(mean, lower, transform, value.to(dtype), mask, [base_check])

    def observed_log_prob(
        self, context: Tensor, value: Tensor, sigma: Tensor, mask: Tensor, nodes: Tensor
    ) -> tuple[Tensor, Tensor]:
        """QMC-convolve quoted independent Gaussian errors over observed axes.

        Returns log density and relative importance ESS per row. Nodes must be
        standard-normal draws [K,D]. Missing-coordinate integration is analytic;
        the remaining noise integral is not certified by a finite ESS alone.
        """
        with _collect_transform_cycles():
            return self._observed_log_prob(context, value, sigma, mask, nodes)

    def _observed_log_prob(
        self, context: Tensor, value: Tensor, sigma: Tensor, mask: Tensor, nodes: Tensor
    ) -> tuple[Tensor, Tensor]:
        if sigma.shape != mask.shape or nodes.ndim != 2 or nodes.shape[1] != self.target_dim:
            raise ValueError("Noise or node shape differs")
        if len(nodes) < 2:
            raise ValueError("At least two finite standard-normal nodes are required")
        # Validations are deferred to one device-to-host read inside _intrinsic.
        checks: list[_Check] = [
            (
                torch.isfinite(nodes).all(),
                ValueError,
                "At least two finite standard-normal nodes are required",
            ),
            (
                ((torch.isfinite(sigma) & (sigma >= 0)) | ~mask).all(),
                ValueError,
                "Observed noise scales must be finite and nonnegative",
            ),
        ]
        dtype = torch.promote_types(
            self._out.weight.dtype, torch.promote_types(value.dtype, sigma.dtype)
        )
        dtype = torch.promote_types(dtype, nodes.dtype)
        with torch.autocast(device_type=context.device.type, enabled=False):
            mean, lower, transform, finite = self._flow_parameters(context, dtype)
            checks.append((finite, FloatingPointError, "Nonfinite Gaussian base parameters"))
            safe_value = value.to(dtype).masked_fill(~mask, 0)
            safe_sigma = sigma.to(dtype).masked_fill(~mask, 0)
            perturbed = safe_value[None] + nodes.to(dtype)[:, None] * safe_sigma[None]
            log_weights = self._intrinsic(mean, lower, transform, perturbed, mask, checks)
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
