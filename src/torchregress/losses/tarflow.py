"""Transformer autoregressive flows (TarFlow / STARFlow) for conditional densities.

A tabular, conditional adaptation of TarFlow (Zhai et al., ICML 2025,
arXiv:2412.06329) and its deep-shallow scaling, STARFlow (Gu et al., 2025,
arXiv:2506.06276).  Each block is an exact autoregressive bijection whose
univariate parameters for the target coordinate ``y_{pi(j)}`` come from a
causal Transformer that reads a context token followed by one token per
preceding coordinate ``y_{pi(<j)}``.  Consecutive blocks reverse the
coordinate order.

Two element-wise transforms are available:

* ``univariate="spline"`` (default): monotonic rational-quadratic splines
  (Durkan et al., 2019), as in NSF.  The conditioner is a causal Transformer
  where NSF uses a MADE network.
* ``univariate="affine"``: the TarFlow transform (affine, with the log-scale
  softly bounded for stable sampling).

The image-specific parts of TarFlow (patch tokens, noise augmentation with a
score-based denoising step, classifier-free guidance) and STARFlow's
pretrained autoencoder latent space are deliberately left out: tabular
targets are continuous, low-dimensional and modelled in data space.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial
from typing import Any, Optional

import torch
from torch import Tensor, nn
from zuko.distributions import DiagNormal  # type: ignore[import-untyped]
from zuko.flows import (  # type: ignore[import-untyped]
    Flow,
    LazyTransform,
    UnconditionalDistribution,
)
from zuko.transforms import (  # type: ignore[import-untyped]
    AutoregressiveTransform,
    DependentTransform,
    MonotonicAffineTransform,
    MonotonicRQSTransform,
)

__all__ = ["TransformerAutoregressiveTransform", "TransformerFlow"]

_UNIVARIATES = ("spline", "affine")


def _soft_clip(x: Tensor, bound: float) -> Tensor:
    # Same squash as ``nflows._soft_clip``; duplicated to avoid an import cycle.
    return x * torch.rsqrt(1.0 + (x / bound).square())


class TransformerAutoregressiveTransform(LazyTransform):  # type: ignore[misc]
    r"""One TarFlow block: a causal-Transformer autoregressive bijection.

    For a coordinate order :math:`\pi`, the block maps

    .. math:: z_{\pi(j)} = f\bigl(y_{\pi(j)};\ \theta_j(c, y_{\pi(<j)})\bigr),

    where :math:`f` is a monotonic spline or affine map and :math:`\theta_j`
    is read from position :math:`j` of a causal Transformer over the tokens
    ``[context, y_pi(0), ..., y_pi(D-2)]``.  The Jacobian is triangular, so
    the log-determinant is the sum of the univariate log-derivatives.
    Density evaluation is one Transformer pass; inversion (sampling) takes
    ``D`` passes.

    Arguments:
        features: Target dimension ``D``.
        context: Context dimension (0 for an unconditional flow).
        order: Coordinate order :math:`\pi` (default ``range(D)``).
        univariate: ``"spline"`` or ``"affine"``.
        bins: Spline bins (``"spline"`` only).
        bound: Spline domain half-width (``"spline"``), identity outside.
        slope: Minimum spline slope, or for ``"affine"`` the bound
            ``|log scale| < -log(slope)``.
        hidden: Transformer width.
        n_layers: Transformer layers in this block.
        n_heads: Attention heads (must divide ``hidden``).
        mlp_ratio: Feed-forward width as a multiple of ``hidden``.
        dropout: Dropout inside the Transformer (regularizes small data).
        conditioner_bound: The conditioner reads ``soft_clip(y, bound)`` so a
            far-tail coordinate cannot push it into extrapolation while sampling.
    """

    order: Tensor
    inverse_order: Tensor

    def __init__(
        self,
        features: int,
        context: int = 0,
        *,
        order: Optional[Tensor] = None,
        univariate: str = "spline",
        bins: int = 8,
        bound: float = 5.0,
        slope: float = 1e-3,
        hidden: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.0,
        conditioner_bound: float = 10.0,
    ) -> None:
        super().__init__()
        if features < 1:
            raise ValueError("features must be at least 1")
        if univariate not in _UNIVARIATES:
            raise ValueError(f"univariate must be one of {_UNIVARIATES}, got {univariate!r}")
        if hidden % n_heads != 0:
            raise ValueError("hidden must be divisible by n_heads")
        if univariate == "spline" and bins < 2:
            raise ValueError("bins must be at least 2")
        if not 0.0 < slope < 1.0:
            raise ValueError("slope must lie in (0, 1)")
        self.features = int(features)
        self.context = int(context)
        self.univariate_name = univariate
        self.conditioner_bound = float(conditioner_bound)
        if order is None:
            order = torch.arange(features)
        order = torch.as_tensor(order, dtype=torch.long)
        if sorted(order.tolist()) != list(range(features)):
            raise ValueError("order must be a permutation of range(features)")
        self.register_buffer("order", order)
        self.register_buffer("inverse_order", torch.argsort(order))

        if univariate == "spline":
            self.shapes: list[tuple[int, ...]] = [(bins,), (bins,), (bins - 1,)]
            self.univariate: Any = partial(MonotonicRQSTransform, bound=bound, slope=slope)
        else:
            self.shapes = [(), ()]
            self.univariate = partial(MonotonicAffineTransform, slope=slope)
        self.total = sum(math.prod(s) for s in self.shapes)

        self.value_embed = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.start = nn.Parameter(torch.zeros(1, 1, hidden))
        self.position = nn.Parameter(torch.randn(1, self.features, hidden) * 0.02)
        self.context_embed = nn.Linear(self.context, hidden) if self.context > 0 else None
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=mlp_ratio * hidden,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=max(1, int(n_layers)), enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, self.total)
        # Zero-initialized output head: every block starts as the identity.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        mask = torch.triu(torch.ones(self.features, self.features, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _parameters_for(self, c: Optional[Tensor], x: Tensor) -> Tensor:
        """Univariate parameters ``[..., D, total]`` in natural coordinate order."""
        if c is not None:
            batch = torch.broadcast_shapes(x.shape[:-1], c.shape[:-1])
            x = x.expand(*batch, x.shape[-1])
            c = c.expand(*batch, c.shape[-1])
        batch_shape = x.shape[:-1]
        flat = x.reshape(-1, self.features)
        n = flat.shape[0]
        permuted = _soft_clip(flat, self.conditioner_bound)[:, self.order]
        values = self.value_embed(permuted[:, :-1].unsqueeze(-1))  # [N, D-1, H]
        tokens = torch.cat([self.start.expand(n, -1, -1), values], dim=1) + self.position
        if self.context_embed is not None and c is not None:
            tokens = tokens + self.context_embed(c.reshape(n, -1)).unsqueeze(1)
        hidden = self.encoder(tokens, mask=self.causal_mask, is_causal=True)
        phi = self.head(self.norm(hidden))  # [N, D, total], permuted order
        phi = phi[:, self.inverse_order]
        return phi.reshape(*batch_shape, self.features, self.total)

    def meta(self, c: Optional[Tensor], x: Tensor) -> Any:
        phi = self._parameters_for(c, x)
        params = []
        start = 0
        for shape in self.shapes:
            size = math.prod(shape)
            chunk = phi[..., start : start + size]
            params.append(chunk.reshape(*phi.shape[:-1], *shape))
            start += size
        return DependentTransform(self.univariate(*params), 1)

    def forward(self, c: Optional[Tensor] = None) -> Any:
        return AutoregressiveTransform(partial(self.meta, c), self.features)


def _standard_normal(features: int) -> Any:
    return UnconditionalDistribution(
        DiagNormal, loc=torch.zeros(features), scale=torch.ones(features), buffer=True
    )


class TransformerFlow(Flow):  # type: ignore[misc]
    """Conditional transformer autoregressive flow (TarFlow / STARFlow-style).

    A stack of :class:`TransformerAutoregressiveTransform` blocks with
    alternating coordinate orders over a standard normal base.  ``n_layers``
    may be a sequence giving the depth of each block, e.g. ``(4, 1, 1)`` for
    STARFlow's deep-shallow design (one deep block, then shallow ones).

    Arguments:
        features: Target dimension.
        context: Context dimension.
        transforms: Number of blocks (ignored when ``n_layers`` is a sequence).
        n_layers: Transformer layers per block, an int or one per block.
        randperm: Random coordinate orders instead of alternating ones.
        kwargs: Forwarded to :class:`TransformerAutoregressiveTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        *,
        n_layers: int | Sequence[int] = 2,
        randperm: bool = False,
        **kwargs: Any,
    ) -> None:
        if isinstance(n_layers, int):
            depths = [int(n_layers)] * int(transforms)
        else:
            depths = [int(d) for d in n_layers]
        if not depths or min(depths) < 1:
            raise ValueError("need at least one block with at least one layer")
        ascending = torch.arange(features)
        blocks = [
            TransformerAutoregressiveTransform(
                features,
                context,
                order=(
                    torch.randperm(features)
                    if randperm
                    else (ascending if i % 2 == 0 else ascending.flip(0))
                ),
                n_layers=depth,
                **kwargs,
            )
            for i, depth in enumerate(depths)
        ]
        super().__init__(blocks, _standard_normal(features))
        self.univariate = blocks[0].univariate_name
        if self.univariate == "spline":
            self.tail_bound = float(kwargs.get("bound", 5.0))
