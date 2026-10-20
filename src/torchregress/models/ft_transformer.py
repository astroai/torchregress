"""Lean FT-Transformer and a two-stage photo-z recipe model.

The FT-Transformer follows Gorishniy et al. (2021): each numeric feature is
tokenized into its own embedding, a CLS token is prepended, and a standard
Transformer encoder produces a 128-dimensional representation.  ``PZRecipe``
uses that representation for (i) a 100-bin quantile redshift classifier with the
CE+CDF+RnC-N objective from the companion paper and (ii) a second-stage
classifier on the frozen features.  The second stage defaults to a compact MLP;
``tabm`` can be plugged in by the caller if desired.
"""

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


class PZRecipe(nn.Module):
    """Two-stage quantile-bin photo-z model.

    Stage 1: FT-Transformer -> ``n_bins`` logits; train with
    :class:`torchregress.losses.PZRecipeLoss` on the CLS features and logits.
    Stage 2: freeze stage 1 and classify the same features into
    ``stage2_bins`` quantile bins with a compact MLP (or an external TabM-like
    classifier supplied as ``stage2_head``).

    The model is deliberately lean and dependency-light; it is the structural
    baseline from the paper, not the full historical training pipeline.
    """

    def __init__(
        self,
        n_features: int,
        *,
        n_bins: int = 100,
        d_token: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        stage2_bins: int = 200,
        stage2_hidden: int = 256,
        stage2_head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.n_bins = int(n_bins)
        self.stage2_bins = int(stage2_bins)
        self.encoder = FTTransformer(
            n_features,
            d_token=d_token,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.stage1_head = nn.Linear(d_token, self.n_bins)
        self.stage2_head = (
            stage2_head
            if stage2_head is not None
            else nn.Sequential(
                nn.Linear(d_token, stage2_hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(stage2_hidden, self.stage2_bins),
            )
        )

    def encode(self, x: Tensor) -> Tensor:
        with torch.no_grad() if not self.training else torch.enable_grad():
            return self.encoder(x)

    def forward(self, x: Tensor, *, stage: int = 1) -> Tensor:
        if stage not in (1, 2):
            raise ValueError("stage must be 1 or 2")
        features = self.encoder(x)
        head = self.stage1_head if stage == 1 else self.stage2_head
        return head(features)

    def predict_proba(self, x: Tensor, *, stage: int = 1) -> Tensor:
        return torch.softmax(self.forward(x, stage=stage), dim=-1)
