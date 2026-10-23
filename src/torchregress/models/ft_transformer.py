"""Lean FT-Transformer for tabular regression and representation learning."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class FeatureTokenizer(nn.Module):
    """Map each numeric feature to a token with its own scale and bias."""

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2 or x.shape[1] != self.weight.shape[0]:
            raise ValueError(f"x must be [B, {self.weight.shape[0]}], got {tuple(x.shape)}")
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    """Feature-tokenizer Transformer for tabular regression/classification."""

    def __init__(
        self,
        n_features: int,
        *,
        d_token: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError("d_token must be divisible by n_heads")
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=4 * d_token,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.out_dim = out_dim
        self.head = nn.Linear(d_token, out_dim) if out_dim is not None else None

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        z = self.encoder(torch.cat([cls, tokens], dim=1))
        features = self.norm(z[:, 0])
        return self.head(features) if self.head is not None else features
