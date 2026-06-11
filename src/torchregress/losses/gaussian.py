"""
Gaussian loss functions for regression tasks.
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import LowRankMultivariateNormal

from .base import DistributionLoss, WeightedMSELoss, reduce_per_sample
from .loss_registry import register_regression_loss

# Module-level constants hoisted out of the hot forward path. CRPS uses
# ``torch.special.ndtr`` (native, autodiff-stable) in place of the textbook
# ``0.5 * (1 + erf(z/√2))`` construction, which avoids a hand-rolled
# per-call ``math.sqrt``.
_LOG_2PI = math.log(2.0 * math.pi)
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def _is_legacy_mask_argument(covariance_matrices: Optional[torch.Tensor], mask: Any) -> bool:
    """Heuristic for the legacy ``forward(y_pred, target, mask, weights, cov)`` ordering.

    The diagonal / CRPS losses ignore ``covariance_matrices`` outright, so
    historically callers passed an extra positional tensor that was really
    a mask.  Rather than sniffing dim() comparisons (which silently broke
    multi-feature masks because ``mask.dim() == target.dim()``), check
    that the candidate is *not* a square / batched square covariance: a
    genuine covariance has shape ``[D, D]`` or ``[B, D, D]``, while a
    mask is broadcast-compatible with the target (``[B]`` or ``[B, D]``).
    """
    if covariance_matrices is None or not isinstance(covariance_matrices, torch.Tensor):
        return False
    if covariance_matrices.dtype == torch.bool:
        return True
    if covariance_matrices.dtype.is_floating_point is False:
        return False
    # A real covariance has trailing dims matching each other: [..., D, D].
    if (
        covariance_matrices.dim() >= 2
        and covariance_matrices.shape[-1] == covariance_matrices.shape[-2]
    ):
        return False
    return True


def low_rank_output_dim(n_features: int, rank: int) -> int:
    """
    Compute output dimension for low-rank Gaussian heads.
    """
    if n_features <= 0 or rank <= 0:
        raise ValueError("n_features and rank must be positive integers")
    return n_features + n_features * rank + n_features


def split_low_rank_gaussian_output(
    y_pred: torch.Tensor,
    n_features: int,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split concatenated output into mean, low-rank factor, and diagonal.

    Expected layout: [mean | cov_factor | cov_diag]
    - mean: n_features
    - cov_factor: n_features * rank
    - cov_diag: n_features
    """
    expected = low_rank_output_dim(n_features, rank)
    if y_pred.shape[-1] != expected:
        raise ValueError(
            f"Expected last dimension {expected} for low-rank output, got {y_pred.shape[-1]}"
        )

    mean = y_pred[..., :n_features]
    factor_flat = y_pred[..., n_features : n_features + n_features * rank]
    cov_factor = factor_flat.reshape(*y_pred.shape[:-1], n_features, rank)
    cov_diag = y_pred[..., -n_features:]
    return mean, cov_factor, cov_diag


def create_gaussian_nll(
    covariance_type: str = "diagonal",
    *,
    use_mse_for_unit_variance: bool = False,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union["GaussianNLLLoss", "MultivariateGaussianLoss", "LowRankGaussianLoss", WeightedMSELoss]:
    """
    Convenience factory for Gaussian-family regression losses.

    Args:
        covariance_type: ``"diagonal"``, ``"full"``, or ``"low_rank"``.
        use_mse_for_unit_variance: If True with diagonal covariance and no learned
            variance, return ``WeightedMSELoss`` for a point-regression objective.
        rank: Included for API symmetry with low-rank heads (not required by the loss).
        **kwargs: Forwarded to the selected loss constructor.
    """
    _ = rank  # Rank is determined by model outputs, not the loss constructor.
    covariance_type = covariance_type.lower()

    if covariance_type == "diagonal":
        if use_mse_for_unit_variance:
            return WeightedMSELoss(**kwargs)
        return GaussianNLLLoss(**kwargs)
    if covariance_type in {"full", "multivariate"}:
        return MultivariateGaussianLoss(**kwargs)
    if covariance_type in {"low_rank", "low-rank"}:
        return LowRankGaussianLoss(**kwargs)

    raise ValueError(
        "covariance_type must be one of {'diagonal', 'full', 'low_rank'}, "
        f"got {covariance_type!r}"
    )


@register_regression_loss("gaussian_nll")
class GaussianNLLLoss(DistributionLoss):
    """
    Gaussian Negative Log-Likelihood loss for diagonal covariance models.

    Supports tuple outputs (mean, log_variance), concatenated tensors
    [mean, log_variance], or mean-only predictions when fixed_variance is set.
    """

    def __init__(
        self,
        covariance_type: str = "diagonal",
        fixed_variance: Optional[Union[float, torch.Tensor]] = None,
        min_variance: float = 1e-6,
        eps: float = 1e-8,
        reduction: str = "mean",
        split_dim: int = -1,
    ) -> None:
        super().__init__(reduction=reduction)
        if covariance_type != "diagonal":
            raise ValueError(
                f"covariance_type must be 'diagonal'. "
                f"For full covariance use MultivariateGaussianLoss, got {covariance_type}"
            )
        self.covariance_type = covariance_type
        self.min_variance = min_variance
        self.eps = eps
        self.split_dim = split_dim

        if fixed_variance is not None:
            fixed_tensor = torch.as_tensor(fixed_variance, dtype=torch.float32)
            if torch.any(fixed_tensor <= 0):
                raise ValueError("fixed_variance must be positive")
            self.register_buffer("fixed_variance", fixed_tensor)
        else:
            self.fixed_variance = None

    def _extract_distribution_parameters(
        self, y_pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fixed_variance is not None:
            mean = y_pred[0] if isinstance(y_pred, (tuple, list)) else y_pred
            var = self.fixed_variance.to(device=mean.device, dtype=mean.dtype)
            var = var + torch.zeros_like(mean)
            return mean, var.clamp(min=self.min_variance)

        if isinstance(y_pred, (tuple, list)):
            if len(y_pred) != 2:
                raise ValueError(
                    "Tuple predictions must be (mean, log_variance), "
                    f"but got {len(y_pred)} elements"
                )
            mean, log_var = y_pred
        else:
            dim = self.split_dim
            if y_pred.shape[dim] % 2 != 0:
                raise ValueError(
                    f"Concatenated predictions must have even size along split_dim={dim} "
                    f"([mean, log_variance]), but got size {y_pred.shape[dim]}."
                )
            mean, log_var = torch.chunk(y_pred, 2, dim=dim)
        var = torch.exp(log_var).clamp(min=self.min_variance)
        return mean, var

    def forward(
        self,
        y_pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        target: torch.Tensor,
        covariance_matrices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Backward compatibility for legacy positional ordering:
        # forward(y_pred, target, mask, weights, covariance_matrices)
        # The diagonal Gaussian loss ignores ``covariance_matrices``; if the
        # caller passed a mask-shaped tensor there by mistake, reinterpret it
        # using the legacy-mask heuristic (not a square covariance).
        if _is_legacy_mask_argument(covariance_matrices, mask):
            legacy_mask = covariance_matrices
            legacy_weights = mask if isinstance(mask, torch.Tensor) else None
            mask = legacy_mask
            weights = legacy_weights
            covariance_matrices = None

        mean, var = self._extract_distribution_parameters(y_pred)
        self._validate_inputs(mean, target, mask)

        # NLL of N(mean, var) for independent dims:
        #   0.5 * (log(2π) + log var + (y - μ)² / var)
        # All constants hoisted to module level; ``eps`` keeps ``log var``
        # finite when var → 0 without breaking gradients.
        nll = 0.5 * (_LOG_2PI + torch.log(var + self.eps) + (target - mean) ** 2 / (var + self.eps))

        return self._reduce_with_mask(nll, mask, weights)


@register_regression_loss("gaussian_crps")
class GaussianCRPSLoss(GaussianNLLLoss):
    """
    Analytic Continuous Ranked Probability Score for diagonal Gaussian predictions.

    Supports the same prediction formats as ``GaussianNLLLoss``:
    - tuple ``(mean, log_variance)``
    - concatenated tensor ``[mean, log_variance]``
    - mean-only tensor when ``fixed_variance`` is set
    """

    def forward(
        self,
        y_pred: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        target: torch.Tensor,
        covariance_matrices: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if _is_legacy_mask_argument(covariance_matrices, mask):
            legacy_mask = covariance_matrices
            legacy_weights = mask if isinstance(mask, torch.Tensor) else None
            mask = legacy_mask
            weights = legacy_weights
            covariance_matrices = None

        mean, var = self._extract_distribution_parameters(y_pred)
        self._validate_inputs(mean, target, mask)

        std = torch.sqrt(var + self.eps)
        z = (target - mean) / (std + self.eps)
        # Analytic Gaussian CRPS (Hersbach 2000, Eq. 4 decomposed):
        #   CRPS = σ * [ z * (2 Φ(z) - 1) + 2 φ(z) - 1/√π ]
        # Use ``torch.special.ndtr`` (native, autodiff-stable) instead of the
        # hand-rolled ``0.5 * (1 + erf(z/√2))``: ndtr is numerically stable in
        # the deep tails where erf saturates and is one fewer op in autograd.
        cdf = torch.special.ndtr(z)
        pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
        crps = std * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - _INV_SQRT_PI)

        return self._reduce_with_mask(crps, mask, weights)


@register_regression_loss("multivariate_gaussian_nll")
class MultivariateGaussianLoss(DistributionLoss):
    """
    Negative Log-Likelihood loss for multivariate Gaussian with full covariance.
    """

    def __init__(
        self,
        n_features: Optional[int] = None,
        learnable_adjustment: bool = False,
        jitter: float = 1e-6,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.n_features = n_features
        self.learnable_adjustment = learnable_adjustment
        self.jitter = jitter
        self.eps = eps

        if learnable_adjustment:
            if n_features is None:
                raise ValueError("n_features is required when learnable_adjustment=True")
            self.log_variance_adjustment = nn.Parameter(torch.zeros(n_features))

    def _prepare_covariance(
        self, covariance_matrices: torch.Tensor, batch_size: int, n_features: int
    ) -> torch.Tensor:
        if covariance_matrices.dim() == 2:
            if covariance_matrices.shape != (n_features, n_features):
                raise ValueError(
                    f"covariance_matrices has shape {list(covariance_matrices.shape)}, "
                    f"expected [{n_features}, {n_features}]"
                )
            cov = covariance_matrices.unsqueeze(0).expand(batch_size, -1, -1)
        elif covariance_matrices.dim() == 3:
            if covariance_matrices.shape[1:] != (n_features, n_features):
                raise ValueError(
                    f"covariance_matrices has shape {list(covariance_matrices.shape)}, "
                    f"expected [batch, {n_features}, {n_features}]"
                )
            if covariance_matrices.shape[0] != batch_size:
                raise ValueError(
                    f"covariance_matrices batch {covariance_matrices.shape[0]} "
                    f"must match target batch {batch_size}"
                )
            cov = covariance_matrices
        else:
            raise ValueError("covariance_matrices must be 2D or 3D tensor")

        if torch.isnan(cov).any() or torch.isinf(cov).any():
            raise RuntimeError("covariance_matrices contains NaN or Inf values")

        if self.learnable_adjustment:
            adjustment = torch.exp(self.log_variance_adjustment).to(cov.device, cov.dtype)
            cov = cov + torch.diag_embed(adjustment)

        eye = torch.eye(n_features, device=cov.device, dtype=cov.dtype)
        return cov + eye * self.jitter

    def _calculate_nll(
        self, target: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
        diff = (target - mean).unsqueeze(-1)  # [B, D, 1]
        try:
            L = torch.linalg.cholesky(cov)
            sol = torch.linalg.solve_triangular(L, diff, upper=False).squeeze(-1)
            quad = torch.sum(sol**2, dim=-1)
            # log|Σ| = 2 * Σ log L_ii.  An in-place chain
            # (diagonal.add_(eps).log_().mul_(2.0).sum(-1)) was profiled and
            # found to be ~equal in cost (37.4 → 38.9 us at B=1024 D=5) while
            # breaking autograd because ``L`` is the Cholesky output (a
            # CopySlices op that cannot be mutated in place during backward).
            # Keep the explicit (allocation-friendly) form.
            log_det = 2 * torch.sum(
                torch.log(torch.diagonal(L, dim1=-2, dim2=-1) + self.eps), dim=-1
            )
        except RuntimeError:
            eigvals, eigvecs = torch.linalg.eigh(cov)
            eigvals = eigvals.clamp(min=self.eps)
            log_det = torch.sum(torch.log(eigvals), dim=-1)
            whitened = torch.matmul(eigvecs.transpose(-1, -2), diff).squeeze(-1)
            quad = torch.sum(whitened**2 / eigvals, dim=-1)

        return 0.5 * (log_det + quad + mean.shape[-1] * math.log(2 * math.pi))

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        covariance_matrices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if target.dim() == 1:
            target = target.unsqueeze(1)
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)

        if y_pred.shape != target.shape:
            raise ValueError(
                f"y_pred shape {list(y_pred.shape)} must match target shape {list(target.shape)}"
            )

        batch_size = target.shape[0]
        n_features = target.shape[-1]

        cov = self._prepare_covariance(covariance_matrices, batch_size, n_features)
        nll = self._calculate_nll(target, y_pred, cov)

        # Collapse per-feature weights to per-sample once; defer the rest to
        # the shared reducer so the four-way branch in the previous version
        # can never drift.
        if weights is not None and weights.dim() > 1:
            weights = weights.mean(dim=-1)
        return reduce_per_sample(nll, mask, weights, self.reduction)


@register_regression_loss("low_rank_gaussian_nll")
class LowRankGaussianLoss(DistributionLoss):
    """
    Negative Log-Likelihood loss for low-rank Gaussian with diagonal correction.

    Covariance is parameterized as: cov = cov_factor @ cov_factor.T + diag(cov_diag).
    """

    def __init__(
        self,
        min_variance: float = 1e-6,
        jitter: float = 1e-6,
        eps: float = 1e-8,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.min_variance = min_variance
        self.jitter = jitter
        self.eps = eps

    def _prepare_low_rank(
        self,
        cov_factor: torch.Tensor,
        cov_diag: torch.Tensor,
        batch_size: int,
        n_features: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cov_factor.dim() == 2:
            if cov_factor.shape[0] != n_features:
                raise ValueError(
                    f"cov_factor has shape {list(cov_factor.shape)}, expected [{n_features}, rank]"
                )
            cov_factor = cov_factor.unsqueeze(0).expand(batch_size, -1, -1)
        elif cov_factor.dim() == 3:
            if cov_factor.shape[0] != batch_size or cov_factor.shape[1] != n_features:
                raise ValueError(
                    f"cov_factor has shape {list(cov_factor.shape)}, "
                    f"expected [batch, {n_features}, rank]"
                )
        else:
            raise ValueError("cov_factor must be 2D or 3D tensor")

        if cov_diag.dim() == 1:
            if cov_diag.shape[0] != n_features:
                raise ValueError(
                    f"cov_diag has shape {list(cov_diag.shape)}, expected [{n_features}]"
                )
            cov_diag = cov_diag.unsqueeze(0).expand(batch_size, -1)
        elif cov_diag.dim() == 2:
            if cov_diag.shape[0] != batch_size or cov_diag.shape[1] != n_features:
                raise ValueError(
                    f"cov_diag has shape {list(cov_diag.shape)}, expected [batch, {n_features}]"
                )
        else:
            raise ValueError("cov_diag must be 1D or 2D tensor")

        if torch.isnan(cov_factor).any() or torch.isinf(cov_factor).any():
            raise RuntimeError("cov_factor contains NaN or Inf values")
        if torch.isnan(cov_diag).any() or torch.isinf(cov_diag).any():
            raise RuntimeError("cov_diag contains NaN or Inf values")

        cov_diag = cov_diag.clamp(min=self.min_variance) + self.jitter
        return cov_factor, cov_diag

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        cov_factor: torch.Tensor,
        cov_diag: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if target.dim() == 1:
            target = target.unsqueeze(1)
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)

        if y_pred.shape != target.shape:
            raise ValueError(
                f"y_pred shape {list(y_pred.shape)} must match target shape {list(target.shape)}"
            )

        batch_size = target.shape[0]
        n_features = target.shape[-1]

        cov_factor, cov_diag = self._prepare_low_rank(cov_factor, cov_diag, batch_size, n_features)

        dist = LowRankMultivariateNormal(loc=y_pred, cov_factor=cov_factor, cov_diag=cov_diag)
        nll = -dist.log_prob(target)

        if weights is not None and weights.dim() > 1:
            weights = weights.mean(dim=-1)
        return reduce_per_sample(nll, mask, weights, self.reduction)
