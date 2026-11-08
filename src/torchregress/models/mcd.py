"""Marginal Contrastive Discrimination (MCD) for conditional density estimation.

Meziani, Ndiaye & Riu (2022, arXiv:2206.01592) write the conditional density as
the marginal times a density ratio,

.. math::

    p(y \\mid x) = p(y)\\, r(x, y), \\qquad r(x, y) = \\frac{p(x, y)}{p(x)\\, p(y)},

so conditional density estimation splits into an unconditional density
estimate of the targets and a classification problem: telling joint pairs
``(x_i, y_i)`` from shuffled pairs ``(x_j, y_i)``.  Here the marginal is any
unconditional flow and the ratio classifier is trained with contrastive neural
ratio estimation (``NeuralRatioLoss(criterion="cnre")``), whose logit
estimates the absolute log ratio rather than the ratio up to a constant.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import Tensor, nn

from ..inference import effective_sample_size
from ..losses.nre import NeuralRatioLoss, neural_ratio_logits
from ..utils.pytorch_compat import _collect_transform_cycles

__all__ = ["MarginalContrastiveDensity"]

#: Largest number of (context, proposal) pairs scored by the classifier at once.
_PAIR_CHUNK = 1 << 20


class MarginalContrastiveDensity(nn.Module):
    """Conditional density ``p(y | x) = p(y) r(x, y)`` from a marginal flow and a ratio classifier.

    Args:
        marginal: Unconditional density of the targets: a module that, called
            with no arguments, returns a distribution with ``log_prob`` and
            ``sample``, e.g. ``create_flow_model(n_features=D, context_dim=0)``.
        classifier: Ratio classifier over ``concat([x, y])``, e.g.
            ``build_ratio_classifier(context_dim=x_dim, observable_dim=D)``.
        gamma: CNRE prior odds of the dependent classes (see
            :class:`~torchregress.losses.NeuralRatioLoss`).
        n_negatives: Shuffled pairs per joint pair in :meth:`loss`: each ``y_i``
            is paired with ``x`` from ``n_negatives`` other rows of the batch.

    :meth:`log_prob` divides by the Monte-Carlo normalizer
    ``Z(x) = E_{y ~ p(y)}[r(x, y)]`` (4096 marginal draws) by default.
    ``Z = 1`` for the exact ratio, so the correction only matters when the
    learned ratio is off, and it keeps NLL and highest-density comparisons with
    flows honest.

    :meth:`sample` draws by sampling-importance-resampling: a bank of marginal
    draws, resampled with weights ``r(x, y)``.  When ``p(y | x)`` is much
    narrower than ``p(y)``, few bank draws carry the weight; check
    :meth:`effective_sample_size` before trusting corner plots.
    """

    def __init__(
        self,
        marginal: nn.Module,
        classifier: nn.Module,
        *,
        gamma: float = 1.0,
        n_negatives: int = 8,
    ) -> None:
        super().__init__()
        if not isinstance(marginal, nn.Module) or not isinstance(classifier, nn.Module):
            raise TypeError("marginal and classifier must be torch.nn.Module instances")
        if int(n_negatives) < 1:
            raise ValueError("n_negatives must be at least 1")
        self.marginal = marginal
        self.classifier = classifier
        self.n_negatives = int(n_negatives)
        self.ratio_loss = NeuralRatioLoss(classifier, criterion="cnre", gamma=gamma)

    # ------------------------------------------------------------------ pieces
    def _marginal_distribution(self) -> Any:
        return cast(Any, self.marginal())

    def marginal_log_prob(self, target: Tensor) -> Tensor:
        """``log p(y)`` under the marginal, ``[N]``."""
        return cast(Tensor, self._marginal_distribution().log_prob(_as_2d(target, "target")))

    def log_ratio(self, x: Tensor, target: Tensor) -> Tensor:
        """Estimated ``log r(x, y)`` on matched rows, ``[N]``."""
        return neural_ratio_logits(self.classifier, _as_2d(x, "x"), _as_2d(target, "target"))

    def _marginal_bank(self, n: int, like: Tensor) -> Tensor:
        with torch.no_grad(), _collect_transform_cycles():
            bank = cast(Tensor, self._marginal_distribution().sample((n,)))
        return bank.to(device=like.device, dtype=like.dtype)

    def _bank_log_ratio(self, x: Tensor, bank: Tensor) -> Tensor:
        """``log r(x_i, y_k)`` for every row and bank draw, ``[N, K]``."""
        n, k = x.shape[0], bank.shape[0]
        rows = max(1, _PAIR_CHUNK // k)
        blocks = []
        for start in range(0, n, rows):
            xb = x[start : start + rows]
            pairs_x = xb.unsqueeze(1).expand(-1, k, -1).reshape(-1, x.shape[1])
            pairs_y = bank.unsqueeze(0).expand(xb.shape[0], -1, -1).reshape(-1, bank.shape[1])
            blocks.append(neural_ratio_logits(self.classifier, pairs_x, pairs_y).view(-1, k))
        return torch.cat(blocks, dim=0)

    # ---------------------------------------------------------------- training
    def loss(self, x: Tensor, target: Tensor) -> Tensor:
        """Marginal NLL plus the CNRE ratio loss on a batch (scalar).

        The two terms have disjoint parameters, so one optimizer over
        ``self.parameters()`` trains both.  Negatives pair each ``y_i`` with
        ``x`` from other rows of the batch (cyclic shifts), so the batch needs
        at least two rows.
        """
        x, target = _as_2d(x, "x"), _as_2d(target, "target")
        n = x.shape[0]
        if n < 2:
            raise ValueError("loss needs at least two rows to build shuffled pairs")
        k = min(self.n_negatives, n - 1)
        shifts = torch.arange(1, k + 1, device=x.device)
        index = (torch.arange(n, device=x.device).unsqueeze(1) + shifts) % n  # [N, K]
        negative_x = x[index]  # [N, K, C]
        marginal_nll = -self.marginal_log_prob(target).mean()
        ratio = self.ratio_loss(x, target, negative_context=negative_x)
        return marginal_nll + ratio

    # --------------------------------------------------------------- inference
    def log_normalizer(self, x: Tensor, n_samples: int = 4096) -> Tensor:
        """Monte-Carlo ``log Z(x) = log E_{y ~ p(y)}[r(x, y)]``, ``[N]`` (0 for the exact ratio)."""
        x = _as_2d(x, "x")
        bank = self._marginal_bank(int(n_samples), x)
        return torch.logsumexp(self._bank_log_ratio(x, bank), dim=1) - math.log(bank.shape[0])

    def log_prob(
        self, x: Tensor, target: Tensor, *, normalize: bool = True, n_samples: int = 4096
    ) -> Tensor:
        """``log p(y | x) = log p(y) + log r(x, y) [- log Z(x)]``, ``[N]``.

        The Monte-Carlo error of ``log Z`` is about ``sd(r(x, .)) / sqrt(n_samples)``,
        which grows for contexts whose conditional sits in the tail of ``p(y)``.
        To score several targets per context, compute :meth:`log_normalizer` once
        and call this with ``normalize=False``.
        """
        x, target = _as_2d(x, "x"), _as_2d(target, "target")
        out = self.marginal_log_prob(target) + self.log_ratio(x, target)
        if normalize:
            out = out - self.log_normalizer(x, n_samples=n_samples)
        return out

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """Negative log-density ``-log p(y | x)`` per row (a conformal score)."""
        return -self.log_prob(x, target)

    @torch.no_grad()
    def sample(self, x: Tensor, n_samples: int = 1, *, n_proposals: int = 4096) -> Tensor:
        """``[N, n_samples, D]`` draws by resampling ``n_proposals`` marginal draws."""
        x = _as_2d(x, "x")
        bank = self._marginal_bank(int(n_proposals), x)
        weights = torch.softmax(self._bank_log_ratio(x, bank).double(), dim=1)
        index = torch.multinomial(weights, int(n_samples), replacement=True)
        return bank[index]

    @torch.no_grad()
    def effective_sample_size(self, x: Tensor, *, n_proposals: int = 4096) -> Tensor:
        """Kish effective size of the resampling weights per row, ``[N]``.

        Between 1 and ``n_proposals``; small values mean few bank draws carry
        the weight and :meth:`sample` repeats them.
        """
        x = _as_2d(x, "x")
        bank = self._marginal_bank(int(n_proposals), x)
        return effective_sample_size(self._bank_log_ratio(x, bank), dim=1)


def _as_2d(tensor: Tensor, name: str) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    if tensor.dim() == 1:
        return tensor.unsqueeze(-1)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {tuple(tensor.shape)}")
    return tensor
