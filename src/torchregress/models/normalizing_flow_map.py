"""Tractable tabular Normalizing Flow Maps (NFM).

NFM (Ruscio et al., 2026, arXiv:2610.22711) learns a time-conditioned family of
invertible pushforward maps along a prescribed probability path.  This module is
a lean tabular instantiation for conditional regression:

* each layer is an affine coupling map whose scale/shift depend on the
  unchanged coordinates, the two times ``(s, t)`` and the context;
* composing the layers gives a diffeomorphism ``f_{s,t}(x_s | c)``;
* the endpoint map ``f_{0,1}`` gives an exact conditional likelihood with a
  change-of-variables Jacobian, while any admissible ``(s,t)`` can be used as a
  few-step flow map.

Training combines pointwise flow-map matching with the endpoint NLL.  It does
not reproduce the image-scale autoregressive architecture of the paper.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class _TimeFeatures(nn.Module):
    freqs: Tensor

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.width = int(width)
        self.register_buffer("freqs", torch.arange(1, max(self.width // 2, 1) + 1).float())

    def forward(self, t: Tensor) -> Tensor:
        arg = t.reshape(-1, 1) * self.freqs.to(t.device, t.dtype) * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)


class TimeConditionedAffineLayer(nn.Module):
    """Invertible affine coupling conditioned on ``(s, t, context)``."""

    mask: Tensor

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden: tuple[int, ...] = (128, 128),
        mask: Optional[Tensor] = None,
        time_features: int = 16,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.time = _TimeFeatures(time_features)
        if mask is None:
            mask = torch.zeros(self.dim, dtype=torch.bool)
            if self.dim > 1:
                mask[::2] = True
        if self.dim == 1:
            # A scalar cannot be split; use a global invertible affine map.
            mask = torch.ones(1, dtype=torch.bool)
        self.register_buffer("mask", mask.bool())
        transformed = int((~self.mask).sum())
        if self.dim > 1 and transformed == 0:
            raise ValueError("mask must leave at least one transformed coordinate")
        time_dim = 2 * self.time.freqs.numel()
        in_dim = self.dim + 2 * time_dim + self.context_dim
        if self.dim == 1:
            in_dim = 2 * time_dim + self.context_dim
        out_dim = 2 if self.dim == 1 else 2 * transformed
        widths = (in_dim, *hidden, out_dim)
        layers: list[nn.Module] = []
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i + 1]))
            if i < len(widths) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _params(self, x: Tensor, s: Tensor, t: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        b = x.shape[0]
        sf = self.time(s.reshape(-1))
        tf = self.time(t.reshape(-1))
        if sf.shape[0] != b:
            sf = sf[:1].expand(b, -1)
            tf = tf[:1].expand(b, -1)
        if self.dim == 1:
            inp = torch.cat([sf, tf, context], dim=-1)
        else:
            inp = torch.cat([x * self.mask.to(x.dtype), sf, tf, context], dim=-1)
        raw = self.net(inp)
        scale_raw, shift = raw.chunk(2, dim=-1)
        scale = 0.5 * torch.tanh(scale_raw)
        return scale, shift

    def forward(self, x: Tensor, s: Tensor, t: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        scale, shift = self._params(x, s, t, context)
        y = x.clone()
        if self.dim == 1:
            y = x * torch.exp(scale) + shift
            return y, scale.sum(dim=-1)
        comp = ~self.mask
        y[:, comp] = x[:, comp] * torch.exp(scale) + shift
        return y, scale.sum(dim=-1)

    def inverse(self, y: Tensor, s: Tensor, t: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        x = y.clone()
        if self.dim == 1:
            scale, shift = self._params(y, s, t, context)
            x = (y - shift) * torch.exp(-scale)
            return x, -scale.sum(dim=-1)
        comp = ~self.mask
        # The mask coordinates are unchanged, so the coupling params are exact.
        scale, shift = self._params(y, s, t, context)
        x[:, comp] = (y[:, comp] - shift) * torch.exp(-scale)
        return x, -scale.sum(dim=-1)


class NormalizingFlowMap(nn.Module):
    """Time-conditioned invertible flow map for conditional tabular targets.

    Args:
        n_features: Target dimensionality.
        context_dim: Context/conditioning width.
        n_layers: Number of affine coupling layers.
        hidden: Hidden widths of each coupling network.
        time_features: Fourier features for each time variable.
        map_weight: Weight of the pointwise flow-map regression term in
            :meth:`training_loss`.
        nll_weight: Weight of the endpoint NLL term.
    """

    def __init__(
        self,
        n_features: int,
        context_dim: int,
        *,
        n_layers: int = 4,
        hidden: tuple[int, ...] = (128, 128),
        time_features: int = 16,
        map_weight: float = 1.0,
        nll_weight: float = 0.1,
        arch: str = "coupling",
        tarflow_layers: int = 3,
        tarflow_heads: int = 4,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.context_dim = int(context_dim)
        if arch not in ("coupling", "tarflow"):
            raise ValueError("arch must be 'coupling' or 'tarflow'")
        self.map_weight = float(map_weight)
        self.nll_weight = float(nll_weight)
        self.arch = str(arch)
        layers = []
        for i in range(int(n_layers)):
            if self.arch == "tarflow" and self.n_features > 1:
                layers.append(
                    TokenizedTARFlowLayer(
                        self.n_features,
                        self.context_dim,
                        hidden=int(hidden[0]) if hidden else 128,
                        n_heads=max(1, int(tarflow_heads)),
                        n_encoder_layers=max(1, int(tarflow_layers)),
                        time_features=time_features,
                    )
                )
                continue
            if self.n_features == 1:
                mask = torch.ones(1, dtype=torch.bool)
            else:
                mask = torch.zeros(self.n_features, dtype=torch.bool)
                mask[i % 2 :: 2] = True
                if not bool((~mask).any()):
                    mask[0] = False
                if not bool(mask.any()):
                    mask[0] = True
            layers.append(
                TimeConditionedAffineLayer(
                    self.n_features,
                    self.context_dim,
                    hidden=hidden,
                    mask=mask,
                    time_features=time_features,
                )
            )
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def _time_like(value: float | Tensor, x: Tensor) -> Tensor:
        if isinstance(value, Tensor):
            value = value.reshape(-1).to(x.device, x.dtype)
            if value.numel() == 1:
                return value.expand(x.shape[0])
            if value.numel() != x.shape[0]:
                raise ValueError("time must be scalar or have one value per batch row")
            return value
        return torch.full((x.shape[0],), float(value), device=x.device, dtype=x.dtype)

    def forward(
        self, x: Tensor, s: float | Tensor, t: float | Tensor, context: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Map ``x_s`` to ``x_t`` with log|det Jacobian|."""
        s_t = self._time_like(s, x)
        t_t = self._time_like(t, x)
        logdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, ld = layer.forward(x, s_t, t_t, context)
            logdet = logdet + ld
        return x, logdet

    def inverse(
        self, z: Tensor, s: float | Tensor, t: float | Tensor, context: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Inverse map from ``z_t`` to ``z_s``."""
        s_t = self._time_like(s, z)
        t_t = self._time_like(t, z)
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z, s_t, t_t, context)
            logdet = logdet + ld
        return z, logdet

    def log_prob(self, context: Tensor, target: Tensor) -> Tensor:
        """Exact conditional log-density ``log p(y | context)``."""
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        z, logdet = self.inverse(target, 0.0, 1.0, context)
        base = -0.5 * (z**2).sum(dim=-1) - 0.5 * self.n_features * math.log(2.0 * math.pi)
        return base + logdet

    @torch.no_grad()
    def sample(self, context: Tensor, n_samples: int = 128) -> Tensor:
        """Draw ``[B, n_samples, D]`` conditional samples from the flow map."""
        b = context.shape[0]
        z = torch.randn(
            b * n_samples,
            self.n_features,
            dtype=context.dtype,
            device=context.device,
        )
        ctx = context.repeat_interleave(n_samples, dim=0)
        x, _ = self.forward(z, 0.0, 1.0, ctx)
        return x.reshape(b, n_samples, self.n_features)

    def flow_map_loss(
        self,
        context: Tensor,
        target: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        eps: float = 1e-3,
    ) -> Tensor:
        """Pointwise matching of the flow map along the linear interpolant."""
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        n = target.shape[0]
        x0 = torch.randn(
            target.shape, dtype=target.dtype, device=target.device, generator=generator
        )
        s = eps + (1.0 - 2.0 * eps) * torch.rand(
            n, dtype=target.dtype, device=target.device, generator=generator
        )
        t = eps + (1.0 - 2.0 * eps) * torch.rand(
            n, dtype=target.dtype, device=target.device, generator=generator
        )
        s, t = torch.minimum(s, t), torch.maximum(s, t)
        xs = (1.0 - s[:, None]) * x0 + s[:, None] * target
        xt = (1.0 - t[:, None]) * x0 + t[:, None] * target
        pred, _ = self.forward(xs, s, t, context)
        return torch.mean((pred - xt) ** 2)

    def training_loss(
        self,
        context: Tensor,
        target: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """NFM objective: flow-map matching + endpoint NLL."""
        return (
            self.map_weight * self.flow_map_loss(context, target, generator=generator)
            - self.nll_weight * self.log_prob(context, target).mean()
        )


class AutoregressiveTimeLayer(nn.Module):
    """Causal-transformer autoregressive flow-map layer.

    This is a TARFlow-style tabular layer: each target coordinate is transformed
    with scale/shift depending only on earlier coordinates, the two times and
    the context.  A causal Transformer produces the parameters; inversion is
    sequential in the target dimension.  It is more expressive than a single
    affine coupling for low-dimensional targets while remaining exactly
    invertible.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden: int = 128,
        n_heads: int = 4,
        time_features: int = 16,
    ) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("AutoregressiveTimeLayer requires dim >= 2")
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.hidden = int(hidden)
        self.time = _TimeFeatures(time_features)
        time_dim = 2 * self.time.freqs.numel()
        self.token = nn.Linear(1, hidden)
        self.context_proj = nn.Linear(context_dim + 2 * time_dim, hidden)
        self.start = nn.Parameter(torch.zeros(1, 1, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=4 * hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _tokens(self, x: Tensor, s: Tensor, t: Tensor, context: Tensor) -> Tensor:
        b = x.shape[0]
        time = torch.cat([self.time(s.reshape(-1)), self.time(t.reshape(-1))], dim=-1)
        if time.shape[0] != b:
            time = time[:1].expand(b, -1)
        cond = self.context_proj(torch.cat([context, time], dim=-1))
        start = self.start.expand(b, -1, -1) + cond.unsqueeze(1)
        tokens = self.token(x.unsqueeze(-1)) + cond.unsqueeze(1)
        return torch.cat([start, tokens], dim=1)

    def _mask(self, length: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def _params_from_hidden(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        raw = self.head(self.norm(hidden[:, :-1]))  # [B, D, 2]
        scale_raw, shift = raw[..., 0], raw[..., 1]
        return 0.5 * torch.tanh(scale_raw), shift

    def forward(self, x: Tensor, s: Tensor, t: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self._tokens(x, s, t, context)
        hidden = self.encoder(tokens, mask=self._mask(tokens.shape[1], x.device))
        scale, shift = self._params_from_hidden(hidden)
        y = x * torch.exp(scale) + shift
        return y, scale.sum(dim=-1)

    def inverse(self, y: Tensor, s: Tensor, t: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        b, d = y.shape
        # Sequential causal inversion.  Prefix tokens use already recovered x.
        recovered = []
        logdet = torch.zeros(b, device=y.device, dtype=y.dtype)
        for j in range(d):
            x_prefix = torch.stack(recovered, dim=-1) if recovered else y.new_empty(b, 0)
            tokens = self._tokens(x_prefix, s, t, context)
            hidden = self.encoder(tokens, mask=self._mask(tokens.shape[1], y.device))
            raw = self.head(self.norm(hidden[:, -1]))  # [B, 2]
            scale = 0.5 * torch.tanh(raw[..., 0])
            shift = raw[..., 1]
            xj = (y[:, j] - shift) * torch.exp(-scale)
            recovered.append(xj)
            logdet = logdet - scale
        return torch.stack(recovered, dim=-1), logdet


class TokenizedTARFlowLayer(nn.Module):
    """Tokenized transformer autoregressive flow-map layer.

    A more expressive TARFlow-style layer: explicit context and time tokens are
    followed by one token per target coordinate, and a causal multi-layer
    Transformer produces the scale/shift for each coordinate from that
    coordinate's prefix.  Unlike :class:`AutoregressiveTimeLayer`, the context
    and both times are first-class tokens and the encoder is multi-layer.

    Exact inversion is sequential in the target dimension; the forward map is a
    single causal Transformer pass.  This is a tabular refinement of TARFlow,
    not the image-scale autoregressive architecture.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        time_features: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("TokenizedTARFlowLayer requires dim >= 2")
        self.dim = int(dim)
        self.context_dim = int(context_dim)
        self.hidden = int(hidden)
        self.time = _TimeFeatures(time_features)
        time_dim = 2 * self.time.freqs.numel()
        self.context_proj = nn.Linear(context_dim, hidden)
        self.time_proj = nn.Linear(2 * time_dim, hidden)
        self.x_proj = nn.Linear(1, hidden)
        self.pos = nn.Parameter(torch.zeros(1, dim, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=4 * hidden,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(n_encoder_layers)))
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _tokens(self, x: Tensor, s: Tensor, t: Tensor, context: Tensor) -> Tensor:
        b = x.shape[0]
        time = torch.cat([self.time(s.reshape(-1)), self.time(t.reshape(-1))], dim=-1)
        if time.shape[0] != b:
            time = time[:1].expand(b, -1)
        start = torch.stack([self.context_proj(context), self.time_proj(time)], dim=1)  # [B, 2, H]
        if x.shape[1] == 0:
            return start
        target = self.x_proj(x.unsqueeze(-1)) + self.pos[:, : x.shape[1]]
        return torch.cat([start, target], dim=1)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, x: Tensor, s: Tensor, t: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self._tokens(x, s, t, context)
        hidden = self.encoder(tokens, mask=self._causal_mask(tokens.shape[1], x.device))
        raw = self.head(self.norm(hidden[:, 1 : 1 + self.dim]))  # [B, D, 2]
        scale = 0.5 * torch.tanh(raw[..., 0])
        shift = raw[..., 1]
        y = x * torch.exp(scale) + shift
        return y, scale.sum(dim=-1)

    def inverse(self, y: Tensor, s: Tensor, t: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        b, d = y.shape
        recovered: list[Tensor] = []
        logdet = torch.zeros(b, device=y.device, dtype=y.dtype)
        for j in range(d):
            prefix = torch.stack(recovered, dim=-1) if recovered else y.new_empty(b, 0)
            tokens = self._tokens(prefix, s, t, context)
            hidden = self.encoder(tokens, mask=self._causal_mask(tokens.shape[1], y.device))
            raw = self.head(self.norm(hidden[:, -1]))
            scale = 0.5 * torch.tanh(raw[..., 0])
            shift = raw[..., 1]
            xj = (y[:, j] - shift) * torch.exp(-scale)
            recovered.append(xj)
            logdet = logdet - scale
        return torch.stack(recovered, dim=-1), logdet
