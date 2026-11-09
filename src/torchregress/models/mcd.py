"""Marginal Contrastive Discrimination (MCD) for conditional density estimation.

Meziani, Ndiaye & Riu (2022, arXiv:2206.01592) write the conditional density as
the marginal times a density ratio,

.. math::

    p(y \\mid x) = p(y)\\, r(x, y), \\qquad r(x, y) = \\frac{p(x, y)}{p(x)\\, p(y)},

so conditional density estimation splits into an unconditional density
estimate of the targets and a classification problem: telling joint pairs
``(x_i, y_i)`` from shuffled pairs ``(x_j, y_i)``.  Here the marginal is any
unconditional flow and the ratio classifier is trained with contrastive neural
ratio estimation (``criterion="cnre"``), whose logit estimates the absolute log
ratio rather than the ratio up to a constant.
"""

from __future__ import annotations

import math
from itertools import chain
from typing import Any, cast

import torch
from torch import Tensor, nn

from ..inference import effective_sample_size
from ..losses.nre import neural_ratio_logits, neural_ratio_loss
from ..utils.pytorch_compat import _collect_transform_cycles

__all__ = ["MarginalContrastiveDensity"]

#: Largest number of input elements (pairs x (context + target width)) scored by
#: the classifier in one chunk.
_PAIR_ELEMENTS = 1 << 24


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
            is paired with ``x`` from ``n_negatives`` other rows of the batch,
            chosen at random.

    Train with :meth:`loss`.  :meth:`log_prob` (and :meth:`forward`, its
    negative) divides by the Monte-Carlo normalizer
    ``Z(x) = E_{y ~ p(y)}[r(x, y)]``, computed without gradients from a bank
    of marginal draws (4096 by default).  ``Z = 1`` for the exact ratio, so the
    correction only matters when the learned ratio is off, and it keeps NLL and
    highest-density comparisons with flows honest.  It is not a training
    objective: the normalizer carries no gradient.

    :meth:`sample` draws by sampling-importance-resampling from a bank of
    marginal draws, resampled with weights ``r(x, y)``.  When ``p(y | x)`` is
    much narrower than ``p(y)``, few bank draws carry the weight; check
    :meth:`effective_sample_size` before trusting corner plots.

    In eval mode the banks are cached (one per size) and shared by
    :meth:`log_prob`, :meth:`sample` and :meth:`effective_sample_size`, so
    repeated calls use the same normalizer for the same context, and the
    effective sample size describes the bank that :meth:`sample` resampled.
    Switching to train mode, or calling :meth:`refresh_bank`, discards them.
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
        if not (math.isfinite(float(gamma)) and float(gamma) > 0.0):
            raise ValueError("gamma must be positive and finite")
        self.marginal = marginal
        self.classifier = classifier
        self.gamma = float(gamma)
        self.n_negatives = int(n_negatives)
        self._banks: dict[int, Tensor] = {}

    # ------------------------------------------------------------------ banks
    def train(self, mode: bool = True) -> MarginalContrastiveDensity:
        if mode:
            self._banks.clear()
        return cast(MarginalContrastiveDensity, super().train(mode))

    def refresh_bank(self) -> None:
        """Discard the cached marginal banks; the next call draws new ones."""
        self._banks.clear()

    def load_state_dict(self, state_dict: Any, strict: bool = True, assign: bool = False) -> Any:
        self._banks.clear()  # banks drawn from the old weights are stale
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _marginal_dtype(self) -> torch.dtype | None:
        """Floating dtype of the marginal's tensors, or ``None`` if it has none."""
        for tensor in chain(self.marginal.parameters(), self.marginal.buffers()):
            if tensor.is_floating_point():
                return tensor.dtype
        return None

    def _marginal_bank(self, n: int) -> Tensor:
        """``[n, D]`` marginal draws; cached per size in eval mode."""
        if not self.training and n in self._banks:
            return self._banks[n]
        with torch.no_grad(), _collect_transform_cycles():
            bank = cast(Tensor, cast(Any, self.marginal()).sample((n,)))
        if not self.training:
            self._banks[n] = bank
        return bank

    # ------------------------------------------------------------------ pieces
    def marginal_log_prob(self, target: Tensor) -> Tensor:
        """``log p(y)`` under the marginal, ``[N]``."""
        target = _as_2d(target, "target")
        dtype = self._marginal_dtype()
        if dtype is not None:
            target = target.to(dtype)
        return cast(Tensor, cast(Any, self.marginal()).log_prob(target))

    def log_ratio(self, x: Tensor, target: Tensor) -> Tensor:
        """Estimated ``log r(x, y)`` on matched rows, ``[N]``."""
        x = _as_2d(x, "x")
        return neural_ratio_logits(self.classifier, x, _as_2d(target, "target").to(x.dtype))

    def _bank_log_ratio(self, x: Tensor, bank: Tensor) -> Tensor:
        """``log r(x_i, y_k)`` for every row and bank draw, ``[N, K]``."""
        n, k = x.shape[0], bank.shape[0]
        bank = bank.to(device=x.device, dtype=x.dtype)
        rows = max(1, _PAIR_ELEMENTS // (k * (x.shape[1] + bank.shape[1])))
        blocks = []
        for start in range(0, n, rows):
            xb = x[start : start + rows]
            pairs = torch.cat(
                [
                    xb.unsqueeze(1).expand(-1, k, -1),
                    bank.unsqueeze(0).expand(xb.shape[0], -1, -1),
                ],
                dim=-1,
            )
            blocks.append(self.classifier(pairs.reshape(-1, pairs.shape[-1])).view(-1, k))
        return torch.cat(blocks, dim=0)

    # ---------------------------------------------------------------- training
    def loss(self, x: Tensor, target: Tensor) -> Tensor:
        """Marginal NLL plus the CNRE ratio loss on a batch (scalar).

        The two terms have disjoint parameters, so one optimizer over
        ``self.parameters()`` trains both.  Each ``y_i`` is paired with ``x``
        from ``n_negatives`` other rows drawn at random (never row ``i``
        itself), so the negatives do not depend on how the batch is ordered.
        The batch needs at least two rows.
        """
        x, target = _as_2d(x, "x"), _as_2d(target, "target")
        n = x.shape[0]
        if n < 2:
            raise ValueError("loss needs at least two rows to build shuffled pairs")
        offsets = torch.randint(1, n, (n, self.n_negatives), device=x.device)
        index = (torch.arange(n, device=x.device).unsqueeze(1) + offsets) % n  # [N, K]
        marginal_nll = -self.marginal_log_prob(target).mean()
        ratio = neural_ratio_loss(
            self.classifier,
            x,
            target.to(x.dtype),
            negative_context=x[index],
            criterion="cnre",
            gamma=self.gamma,
        )
        return marginal_nll + ratio

    # --------------------------------------------------------------- inference
    def log_normalizer(self, x: Tensor, n_samples: int = 4096) -> Tensor:
        """Monte-Carlo ``log Z(x) = log E_{y ~ p(y)}[r(x, y)]``, ``[N]`` (0 for the exact ratio).

        Computed without gradients, once per unique row of ``x``, from the
        (cached, in eval mode) bank of ``n_samples`` marginal draws.
        """
        x = _as_2d(x, "x")
        with torch.no_grad():
            unique, inverse = torch.unique(x, dim=0, return_inverse=True)
            bank = self._marginal_bank(int(n_samples))
            log_z = torch.logsumexp(self._bank_log_ratio(unique, bank), dim=1)
            return (log_z - math.log(bank.shape[0]))[inverse]

    def log_prob(
        self, x: Tensor, target: Tensor, *, normalize: bool = True, n_samples: int = 4096
    ) -> Tensor:
        """``log p(y | x) = log p(y) + log r(x, y) [- log Z(x)]``, ``[N]``.

        The Monte-Carlo error of ``log Z`` is about ``sd(r(x, .)) / sqrt(n_samples)``,
        which grows for contexts whose conditional sits in the tail of ``p(y)``.
        """
        x, target = _as_2d(x, "x"), _as_2d(target, "target")
        out = self.marginal_log_prob(target).to(x.dtype) + self.log_ratio(x, target)
        if normalize:
            out = out - self.log_normalizer(x, n_samples=n_samples)
        return out

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """Negative log-density ``-log p(y | x)`` per row (a conformal score, not a loss)."""
        return -self.log_prob(x, target)

    @torch.no_grad()
    def sample(self, x: Tensor, n_samples: int = 1, *, n_proposals: int = 4096) -> Tensor:
        """``[N, n_samples, D]`` draws (in ``x``'s dtype) by resampling marginal draws."""
        x = _as_2d(x, "x")
        bank = self._marginal_bank(int(n_proposals)).to(device=x.device, dtype=x.dtype)
        weights = torch.softmax(self._bank_log_ratio(x, bank).double(), dim=1)
        index = torch.multinomial(weights, int(n_samples), replacement=True)
        return bank[index]

    @torch.no_grad()
    def effective_sample_size(self, x: Tensor, *, n_proposals: int = 4096) -> Tensor:
        """Kish effective size of the resampling weights per row, ``[N]``.

        Between 1 and ``n_proposals``; small values mean few bank draws carry
        the weight and :meth:`sample` repeats them.  In eval mode this is the
        bank :meth:`sample` uses for the same ``n_proposals``.
        """
        x = _as_2d(x, "x")
        bank = self._marginal_bank(int(n_proposals))
        return effective_sample_size(self._bank_log_ratio(x, bank), dim=1)


def _as_2d(tensor: Tensor, name: str) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    if tensor.dim() == 1:
        return tensor.unsqueeze(-1)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {tuple(tensor.shape)}")
    return tensor
