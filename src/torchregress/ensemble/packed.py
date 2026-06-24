"""
Packed (batch) ensemble regressor with structured uncertainty outputs.

Wraps :class:`HeteroscedasticBatchEnsembleModel` (and a homoscedastic variant)
with an optional ``alpha`` scaling factor on BatchEnsemble fast weights for
extra diversity, and :class:`PackedEnsembleOutput` for impact-style access
(``mean``, ``std_epistemic``, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from .layers import BatchEnsembleLinear
from .models import (
    HeteroscedasticBatchEnsembleModel,
    _variance_across_members,
    _variance_from_logvar,
)


def _scale_batch_ensemble_factors(module: nn.Module, alpha: float) -> None:
    """Scale BatchEnsemble rank-1 factors (``r``, ``s``) by ``alpha`` (diversity knob)."""
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}.")
    if abs(alpha - 1.0) < 1e-12:
        return
    for m in module.modules():
        if isinstance(m, BatchEnsembleLinear):
            m.r_vectors.data.mul_(alpha)
            m.s_vectors.data.mul_(alpha)


@dataclass
class BatchEnsembleOutput:
    """Structured prediction from :meth:`BatchEnsembleRegressor.predict_output`."""

    mean: torch.Tensor
    member_means: torch.Tensor
    epistemic_variance: torch.Tensor
    aleatoric_variance: Optional[torch.Tensor]
    predictive_variance: torch.Tensor
    std_epistemic: torch.Tensor


# Deprecated alias
PackedEnsembleOutput = BatchEnsembleOutput


class MeanOnlyBatchEnsembleModel(nn.Module):
    """Batch-ensemble head predicting only per-member means (epistemic uncertainty only)."""

    def __init__(
        self,
        backbone: nn.Module,
        input_size: int,
        output_size: int,
        ensemble_size: int = 4,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.ensemble_size = ensemble_size
        self.output_size = output_size
        dev = str(device) if isinstance(device, torch.device) else device
        self.output_layer = BatchEnsembleLinear(
            in_features=input_size,
            out_features=output_size,
            ensemble_size=ensemble_size,
            device=dev,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        means = self.output_layer(features)
        return {"means": means}

    def predict(self, x: torch.Tensor, correction: int = 0) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            means = self.forward(x)["means"]
            ensemble_mean = torch.mean(means, dim=1)
            epistemic_var = _variance_across_members(means, dim=1, correction=correction)
            return {
                "mean": ensemble_mean,
                "variance": epistemic_var.clamp_min(1.0e-8),
                "epistemic_variance": epistemic_var,
                "aleatoric_variance": torch.zeros_like(epistemic_var),
            }


class BatchEnsembleRegressor(nn.Module):
    """
    Parameter-efficient BatchEnsemble regressor for regression.

    This is a thin facade over :class:`HeteroscedasticBatchEnsembleModel` (or a
    mean-only batch ensemble) with ``alpha`` scaling of fast weights and
    :meth:`predict_output` returning :class:`BatchEnsembleOutput`.

    Args:
        backbone: Feature extractor. Must map ``x`` to a tensor that the final
            :class:`BatchEnsembleLinear` can consume (typically ``[batch, feats]``).
        feature_dim: Last dimension of ``backbone(x)`` (``F`` in ``[batch, F]``).
        output_dim: Target dimension ``D``.
        ensemble_size: Number of ensemble members ``M``.
        heteroscedastic: If True, each member predicts ``(mean, log_var)``; else mean only.
        alpha: Multiplier applied to all BatchEnsemble ``r_vectors`` / ``s_vectors``
            after construction (``1.0`` = default initialization).
        device: Device for BatchEnsemble layers.

    Example
    -------
    >>> from torchregress.ensemble import BatchEnsembleMLPBackbone, BatchEnsembleRegressor
    >>> bb = BatchEnsembleMLPBackbone(3, 16, ensemble_size=4, hidden_dims=[16])
    >>> model = BatchEnsembleRegressor(
    ...     bb, feature_dim=bb.feature_dim, output_dim=1, ensemble_size=4, alpha=1.0
    ... )
    >>> y = model.predict_output(torch.randn(8, 3))
    >>> y.mean.shape
    torch.Size([8, 1])

    References
    ----------
    .. [1] Wen, Y., Tran, D., & Ba, J. (2020). BatchEnsemble: An Alternative Approach
       to Efficient Ensemble and Lifelong Learning. In *ICLR 2020*.
       https://arxiv.org/abs/2002.06715
    """

    _model: Union[HeteroscedasticBatchEnsembleModel, MeanOnlyBatchEnsembleModel]

    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int,
        output_dim: int,
        ensemble_size: int = 4,
        heteroscedastic: bool = True,
        alpha: float = 1.0,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.ensemble_size = ensemble_size
        self.heteroscedastic = heteroscedastic
        dev = str(device) if isinstance(device, torch.device) else device
        if heteroscedastic:
            self._model = HeteroscedasticBatchEnsembleModel(
                backbone,
                feature_dim,
                output_dim,
                ensemble_size,
                dev,
            )
        else:
            self._model = MeanOnlyBatchEnsembleModel(
                backbone,
                feature_dim,
                output_dim,
                ensemble_size,
                dev,
            )
        _scale_batch_ensemble_factors(self._model, float(alpha))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._model(x)

    def predict(self, x: torch.Tensor, correction: int = 0) -> dict[str, torch.Tensor]:
        return self._model.predict(x, correction=correction)

    @torch.no_grad()
    def predict_output(self, x: torch.Tensor, correction: int = 0) -> BatchEnsembleOutput:
        """Return structured moments; use :meth:`forward` for training dicts."""
        self.eval()
        out = self.forward(x)
        if self.heteroscedastic:
            means = out["means"]
            log_vars = out["log_vars"]
            variances = _variance_from_logvar(log_vars)
            ensemble_mean = torch.mean(means, dim=1)
            aleatoric_var = torch.mean(variances, dim=1)
            epistemic_var = _variance_across_members(means, dim=1, correction=correction)
            total_var = (epistemic_var + aleatoric_var).clamp_min(1.0e-8)
            return BatchEnsembleOutput(
                mean=ensemble_mean,
                member_means=means,
                epistemic_variance=epistemic_var,
                aleatoric_variance=aleatoric_var,
                predictive_variance=total_var,
                std_epistemic=torch.sqrt(epistemic_var.clamp_min(1.0e-8)),
            )
        means = out["means"]
        ensemble_mean = torch.mean(means, dim=1)
        epistemic_var = _variance_across_members(means, dim=1, correction=correction)
        return BatchEnsembleOutput(
            mean=ensemble_mean,
            member_means=means,
            epistemic_variance=epistemic_var,
            aleatoric_variance=None,
            predictive_variance=epistemic_var.clamp_min(1.0e-8),
            std_epistemic=torch.sqrt(epistemic_var.clamp_min(1.0e-8)),
        )


class PackedEnsembleRegressor(BatchEnsembleRegressor):
    """
    Deprecated alias for BatchEnsembleRegressor.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int,
        output_dim: int,
        ensemble_size: int = 4,
        heteroscedastic: bool = True,
        alpha: float = 1.0,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        import warnings

        warnings.warn(
            "PackedEnsembleRegressor is deprecated and will be removed in a future release. "
            "Use BatchEnsembleRegressor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            backbone=backbone,
            feature_dim=feature_dim,
            output_dim=output_dim,
            ensemble_size=ensemble_size,
            heteroscedastic=heteroscedastic,
            alpha=alpha,
            device=device,
        )
