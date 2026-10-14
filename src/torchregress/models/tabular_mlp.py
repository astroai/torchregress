"""Tabular MLP with periodic numeric embeddings."""

from __future__ import annotations

import math
from typing import Callable, Literal, Sequence, Union

import torch
from torch import nn

_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "silu": nn.SiLU,
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "mish": nn.Mish,
}


class PeriodicEmbedding(nn.Module):
    """Per-feature periodic embedding (Gorishniy et al., 2022).

    Each scalar feature :math:`x_j` is mapped to

    .. math::

        v_j = 2\\pi c_j x_j, \\qquad
        e_j = \\mathrm{ReLU}\\big(W_j [\\sin v_j \\,\\|\\, \\cos v_j] + b_j\\big),

    with learnable frequencies :math:`c_j \\in \\mathbb{R}^k` initialised from
    :math:`\\mathcal{N}(0, \\sigma^2)` and a separate linear layer
    :math:`W_j \\in \\mathbb{R}^{d \\times 2k}` per feature. The raw (scaled) value
    :math:`x_j` is concatenated to :math:`e_j`, so the output has
    ``n_features * (embedding_dim + 1)`` columns.

    Parameters
    ----------
    n_features : int
        Number of input columns.
    n_frequencies : int
        Frequencies :math:`k` per feature.
    embedding_dim : int
        Width :math:`d` of each feature embedding.
    frequency_scale : float
        Standard deviation :math:`\\sigma` of the frequency initialisation. Inputs are
        expected to be robust-scaled to roughly unit range.
    """

    def __init__(
        self,
        n_features: int,
        n_frequencies: int = 16,
        embedding_dim: int = 8,
        frequency_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_frequencies = n_frequencies
        self.embedding_dim = embedding_dim
        self.frequencies = nn.Parameter(torch.randn(n_features, n_frequencies) * frequency_scale)
        bound = 1.0 / math.sqrt(2 * n_frequencies)
        self.weight = nn.Parameter(
            torch.empty(n_features, 2 * n_frequencies, embedding_dim).uniform_(-bound, bound)
        )
        self.bias = nn.Parameter(torch.empty(n_features, embedding_dim).uniform_(-bound, bound))

    @property
    def out_features(self) -> int:
        """Width of the flattened embedding."""
        return self.n_features * (self.embedding_dim + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed ``x`` of shape ``(batch, n_features)`` to ``(batch, out_features)``."""
        v = 2 * math.pi * x.unsqueeze(-1) * self.frequencies
        periodic = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
        emb = torch.relu(torch.einsum("bfk,fkd->bfd", periodic, self.weight) + self.bias)
        out = torch.cat([x.unsqueeze(-1), emb], dim=-1)
        return out.flatten(1)


class TabularMLP(nn.Module):
    """Strong-default MLP for tabular regression.

    Pipeline: optional :class:`PeriodicEmbedding` of every input column, then
    ``Linear -> activation -> Dropout`` blocks and a linear head. The head width
    ``out_features`` is chosen by the caller to match the loss, for example ``D`` for
    :class:`~torchregress.losses.WeightedMSELoss`, ``2 * D`` for
    :class:`~torchregress.losses.GaussianNLLLoss` (``[mean, log_variance]``) or
    ``len(quantiles)`` for :class:`~torchregress.losses.MultiQuantileLoss` with one
    target. The module returns raw head outputs of shape ``(batch, out_features)``.

    Inputs should be preprocessed with :class:`TabularPreprocessor`
    (:func:`fit_tabular` does this).

    Parameters
    ----------
    in_features : int
        Input columns (``TabularPreprocessor.n_features_out_``).
    out_features : int
        Output width.
    hidden : sequence of int, default (256, 256, 256)
        Hidden layer widths.
    embedding : {"periodic", "none"}, default "periodic"
        Numeric embedding.
    n_frequencies : int, default 16
        Frequencies per feature for the periodic embedding.
    embedding_dim : int, default 8
        Per-feature embedding width.
    frequency_scale : float, default 0.5
        Frequency initialisation scale :math:`\\sigma`.
    dropout : float, default 0.1
        Dropout probability after each hidden activation.
    activation : {"silu", "selu", "relu", "gelu", "mish"} or callable, default "silu"
        Activation name, or a zero-argument callable returning an ``nn.Module``.

    Examples
    --------
    >>> import torch
    >>> from torchregress.models import TabularMLP
    >>> TabularMLP(8, 2, hidden=(32, 32))(torch.randn(5, 8)).shape
    torch.Size([5, 2])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Sequence[int] = (256, 256, 256),
        embedding: Literal["periodic", "none"] = "periodic",
        n_frequencies: int = 16,
        embedding_dim: int = 8,
        frequency_scale: float = 0.5,
        dropout: float = 0.1,
        activation: Union[str, Callable[[], nn.Module]] = "silu",
    ) -> None:
        super().__init__()
        if embedding not in ("periodic", "none"):
            raise ValueError(f"embedding must be 'periodic' or 'none', got {embedding!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if isinstance(activation, str):
            if activation not in _ACTIVATIONS:
                raise ValueError(f"unknown activation {activation!r}; use {sorted(_ACTIVATIONS)}")
            act_factory: Callable[[], nn.Module] = _ACTIVATIONS[activation]
        else:
            act_factory = activation
        self.in_features = in_features
        self.out_features = out_features
        self.hidden = tuple(hidden)
        self.embedding_kind = embedding

        self.embedding: nn.Module
        if embedding == "periodic":
            self.embedding = PeriodicEmbedding(
                in_features, n_frequencies, embedding_dim, frequency_scale
            )
            width = self.embedding.out_features
        else:
            self.embedding = nn.Identity()
            width = in_features
        layers: list[nn.Module] = []
        for h in self.hidden:
            layers += [nn.Linear(width, h), act_factory(), nn.Dropout(dropout)]
            width = h
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(width, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map preprocessed features ``(batch, in_features)`` to ``(batch, out_features)``."""
        return self.head(self.blocks(self.embedding(x)))
