"""
Error-aware feature encoders and regressors for noisy tabular inputs.

These modules provide lightweight alternatives to explicit latent-input
optimization. They expose measurement quality to the network through
feature-wise gates and engineered uncertainty channels such as log-error
and signal-to-noise ratio.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn


def _expand_sigma(
    sigma_x: float | torch.Tensor,
    x: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    if isinstance(sigma_x, (int, float)):
        return torch.full_like(x, float(sigma_x)).clamp_min(eps)
    sigma = sigma_x.to(device=x.device, dtype=x.dtype)
    if sigma.ndim == 0:
        return torch.full_like(x, float(sigma.item())).clamp_min(eps)
    if sigma.shape == x.shape:
        return sigma.clamp_min(eps)
    if sigma.ndim == 1 and sigma.shape[0] == x.shape[-1]:
        return sigma.view(1, -1).expand_as(x).clamp_min(eps)
    if sigma.ndim == 2 and sigma.shape[-2:] == (x.shape[-1], x.shape[-1]):
        diag = torch.diagonal(sigma, dim1=-2, dim2=-1).clamp_min(eps**2).sqrt()
        if sigma.shape[0] == x.shape[0]:
            return diag.clamp_min(eps)
        return diag.view(1, -1).expand_as(x).clamp_min(eps)
    if (
        sigma.ndim == 3
        and sigma.shape[0] == x.shape[0]
        and sigma.shape[-2:]
        == (
            x.shape[-1],
            x.shape[-1],
        )
    ):
        return torch.diagonal(sigma, dim1=-2, dim2=-1).clamp_min(eps**2).sqrt()
    raise ValueError(
        "sigma_x must be scalar, match x, be a feature vector, or be a diagonal/full covariance"
    )


class ErrorAwareFeatureEncoder(nn.Module):
    """
    Encode features jointly with their uncertainty and derived quality signals.

    The encoder produces a compact representation that includes:
    - quality-gated raw features
    - raw features
    - log feature uncertainty
    - signed signal-to-noise ratio
    - feature gate values

    The quality gate is a learnable sigmoid threshold over ``log(sigma_x)``,
    which keeps the module lightweight while still letting the network adapt
    how aggressively low-SNR inputs are downweighted.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int,
        eps: float = 1.0e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)

        self.log_sigma_ref = nn.Parameter(torch.zeros(self.input_dim))
        self.log_sigma_temperature = nn.Parameter(torch.zeros(self.input_dim))

        feature_dim = self.input_dim * 5
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, x: torch.Tensor, sigma_x: float | torch.Tensor) -> torch.Tensor:
        sigma = _expand_sigma(sigma_x, x, eps=self.eps)
        gate, features = self._raw_features(x, sigma)
        return self.proj(features)

    def quality_gate(self, x: torch.Tensor, sigma_x: float | torch.Tensor) -> torch.Tensor:
        sigma = _expand_sigma(sigma_x, x, eps=self.eps)
        gate, _ = self._raw_features(x, sigma)
        return gate

    def _raw_features(
        self, x: torch.Tensor, sigma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_sigma = sigma.log()
        temperature = (
            torch.nn.functional.softplus(self.log_sigma_temperature).view(1, -1) + self.eps
        )
        gate = torch.sigmoid((self.log_sigma_ref.view(1, -1) - log_sigma) / temperature)
        signed_snr = x / sigma
        precision = sigma.reciprocal()
        features = torch.cat([x * gate, x, log_sigma, signed_snr, gate * precision], dim=-1)
        return gate, features


class NoiseAwareRegressor(nn.Module):
    """
    Wrap a backbone with an error-aware feature encoder.

    Args:
        input_dim: Number of raw input features.
        output_dim: Number of outputs from the prediction head.
        encoder_hidden_dim: Hidden dimension used inside the encoder.
        backbone_hidden_dims: Hidden dimensions for the prediction backbone.
        dropout: Dropout applied in the encoder/backbone.
        encoder: Optional custom encoder. If supplied, ``input_dim`` and
            ``encoder_hidden_dim`` are used only for validation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        encoder_hidden_dim: int = 128,
        backbone_hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.0,
        encoder: Optional[ErrorAwareFeatureEncoder] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder or ErrorAwareFeatureEncoder(
            input_dim,
            hidden_dim=encoder_hidden_dim,
            dropout=dropout,
        )
        dims = [encoder_hidden_dim, *list(backbone_hidden_dims), output_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma_x: float | torch.Tensor) -> torch.Tensor:
        return self.backbone(self.encoder(x, sigma_x))
