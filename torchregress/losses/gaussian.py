"""
Gaussian loss functions for regression tasks.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributions import LowRankMultivariateNormal

from .base import DistributionLoss, WeightedMSELoss
from .loss_registry import register_regression_loss


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
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        covariance_matrices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        mean, var = self._extract_distribution_parameters(y_pred)
        self._validate_inputs(mean, target, mask)

        nll = 0.5 * (
            math.log(2 * math.pi)
            + torch.log(var + self.eps)
            + (target - mean) ** 2 / (var + self.eps)
        )

        return self._reduce_with_mask(nll, mask, weights)


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
            diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
            log_det = 2 * torch.sum(torch.log(diag_L + self.eps), dim=-1)
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
        **kwargs,
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

        sample_mask = None
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask > 0
            sample_mask = mask.all(dim=-1) if mask.dim() > 1 else mask

        if weights is not None and weights.dim() > 1:
            weights = weights.mean(dim=-1)
        if sample_mask is None:
            return self._reduce_with_mask(nll, sample_mask, weights)

        if weights is not None:
            weights = weights.to(device=nll.device, dtype=nll.dtype)
            if weights.shape[0] != nll.shape[0]:
                raise ValueError("weights must match batch size for MultivariateGaussianLoss")
            masked_weights = weights[sample_mask]
            masked_nll = nll[sample_mask]
            if self.reduction == "none":
                result = torch.zeros_like(nll)
                result[sample_mask] = masked_nll * masked_weights
                return result
            if self.reduction == "sum":
                return torch.sum(masked_nll * masked_weights)
            if self.reduction == "mean":
                denom = torch.sum(masked_weights).clamp(min=1)
                return torch.sum(masked_nll * masked_weights) / denom

        masked_nll = nll[sample_mask]
        if self.reduction == "none":
            result = torch.zeros_like(nll)
            result[sample_mask] = masked_nll
            return result
        if self.reduction == "sum":
            return torch.sum(masked_nll)
        if self.reduction == "mean":
            valid = sample_mask.sum().clamp(min=1)
            return torch.sum(masked_nll) / valid

        return self._reduce_with_mask(nll, sample_mask, weights)


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
        **kwargs,
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

        sample_mask = None
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask > 0
            sample_mask = mask.all(dim=-1) if mask.dim() > 1 else mask

        if weights is not None and weights.dim() > 1:
            weights = weights.mean(dim=-1)

        if sample_mask is None:
            return self._reduce_with_mask(nll, sample_mask, weights)

        if weights is not None:
            weights = weights.to(device=nll.device, dtype=nll.dtype)
            if weights.shape[0] != nll.shape[0]:
                raise ValueError("weights must match batch size for LowRankGaussianLoss")
            masked_weights = weights[sample_mask]
            masked_nll = nll[sample_mask]
            if self.reduction == "none":
                result = torch.zeros_like(nll)
                result[sample_mask] = masked_nll * masked_weights
                return result
            if self.reduction == "sum":
                return torch.sum(masked_nll * masked_weights)
            if self.reduction == "mean":
                denom = torch.sum(masked_weights).clamp(min=1)
                return torch.sum(masked_nll * masked_weights) / denom

        masked_nll = nll[sample_mask]
        if self.reduction == "none":
            result = torch.zeros_like(nll)
            result[sample_mask] = masked_nll
            return result
        if self.reduction == "sum":
            return torch.sum(masked_nll)
        if self.reduction == "mean":
            valid = sample_mask.sum().clamp(min=1)
            return torch.sum(masked_nll) / valid

        return self._reduce_with_mask(nll, sample_mask, weights)


def create_gaussian_nll(
    n_features: int,
    covariance_type: str = "diagonal",
    model_predicts_variance: bool = True,
    fixed_variance: Optional[float] = None,
    use_mse_for_unit_variance: bool = False,
    jitter: float = 1e-6,
    reduction: str = "mean",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a Gaussian NLL loss for common configurations.

    Args:
        n_features: Output dimension.
        covariance_type: 'diagonal', 'full', or 'low_rank'.
        model_predicts_variance: Whether the model outputs variance parameters.
        fixed_variance: Fixed variance for homoscedastic models.
        use_mse_for_unit_variance: If True and fixed_variance is 1.0, return MSE instead
            of Gaussian NLL (changes the optimization objective).
        jitter: Diagonal jitter for stability.
        reduction: Reduction to apply.
    """
    if covariance_type not in {"diagonal", "full", "low_rank"}:
        raise ValueError(
            f"Invalid covariance_type: {covariance_type}. "
            f"Must be 'diagonal', 'full', or 'low_rank'."
        )

    extra_kwargs = dict(kwargs)
    min_variance = extra_kwargs.pop("min_variance", 1e-6)
    eps = extra_kwargs.pop("eps", 1e-8)

    if covariance_type == "full":
        return MultivariateGaussianLoss(
            n_features=n_features,
            jitter=jitter,
            eps=eps,
            reduction=reduction,
            **extra_kwargs,
        )
    if covariance_type == "low_rank":
        return LowRankGaussianLoss(
            min_variance=min_variance,
            jitter=jitter,
            eps=eps,
            reduction=reduction,
            **extra_kwargs,
        )

    if model_predicts_variance:
        if fixed_variance is not None:
            raise ValueError("fixed_variance should be None when model_predicts_variance=True")
        return GaussianNLLLoss(
            fixed_variance=None, min_variance=min_variance, eps=eps, reduction=reduction
        )

    if fixed_variance is None:
        raise ValueError("fixed_variance must be provided when model_predicts_variance=False")

    if use_mse_for_unit_variance and math.isclose(
        float(fixed_variance), 1.0, rel_tol=0.0, abs_tol=0.0
    ):
        return WeightedMSELoss(reduction=reduction)

    return GaussianNLLLoss(
        fixed_variance=fixed_variance, min_variance=min_variance, eps=eps, reduction=reduction
    )
