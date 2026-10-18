r"""Continuous-target Group DRO (distributionally robust optimization).

Group DRO (Sagawa et al., 2020; Oren et al., 2019) replaces the average
training risk by a minimax risk over a small number of pre-defined groups::

    min_theta  max_{g in 1..G}  E_{(x, y) ~ P_g}[ell(f_theta(x), y)]

For a *continuous* regression target there is no natural group label, so this
module bins the target range into ``G`` contiguous regimes (e.g. metallicity or
redshift bins) and treats each bin as a group.  The maximum is parameterised by
adversarial group weights ``q`` on the simplex and solved by exponentiated
gradient ascent, a special case of mirror descent:

.. math::

    q_g^{(t)} \propto q_g^{(t-1)} \exp(\eta \, L_g^{(t)})

where ``L_g`` is the batch mean loss of group ``g``.  The returned training
loss is :math:`\sum_g q_g L_g` with ``q`` detached, so gradients flow through
the per-sample losses but never through the group weights.

This differs from :class:`torchregress.losses.robust.CVaRLoss`, which chases
the worst *individual samples* (noise / outliers): Group DRO equalises the
worst *group mean*, so a noisy sample inside a large group is averaged out
instead of dominating the objective.

References
----------
.. [1] Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020).
   Distributionally Robust Neural Networks for Group Shifts: On the Importance
   of Regularization for Worst-Case Generalization. In *ICML 2020*.
   https://arxiv.org/abs/1911.08731
.. [2] Oren, Y., Sagawa, S., Hashimoto, T. B., & Liang, P. (2019).
   Distributionally Robust Language Modeling. In *EMNLP 2019*.
   https://arxiv.org/abs/1909.02060
.. [3] Hashimoto, T. B., Srivastava, M., Namkoong, H., & Liang, P. (2018).
   Fairness Without Demographics in Repeated Loss Minimization. In *ICML 2018*.
   https://arxiv.org/abs/1806.08010
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

from ..utils.validation import check_tensor, validate_reduction

__all__ = ["ContinuousGroupDRO", "GroupDROTrainer"]

_BASE_LOSSES = frozenset({"mse", "mae"})
# Maximum exponent fed into ``log_q`` by one update.  Large finite losses then
# still produce finite exponentials instead of ``inf - inf = nan`` in the
# log-sum-exp normalisation.
_MAX_LOG_UPDATE = 1e30


class _BatchStats(NamedTuple):
    """Per-batch quantities shared by :meth:`forward` and ``group_losses``."""

    per_sample: Tensor
    group_index: Tensor
    valid: Tensor
    weights: Tensor
    group_losses: Tensor
    group_weight_sums: Tensor


def _as_edge_tensor(value: Sequence[float] | Tensor) -> Tensor:
    """Return ``value`` as a 1-D ``float64`` CPU tensor (no validation)."""
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return torch.as_tensor(list(value), dtype=torch.float64, device="cpu")


def _validate_edges(group_edges: Sequence[float] | Tensor) -> Tensor:
    """Validate monotone bin edges and return them as a ``float64`` tensor."""
    edges = _as_edge_tensor(group_edges)
    if edges.numel() < 2:
        raise ValueError(
            f"group_edges must contain at least 2 edges (one group), got {edges.numel()}"
        )
    if not bool(torch.isfinite(edges).all()):
        raise ValueError("group_edges must be finite")
    if not bool((edges[1:] > edges[:-1]).all()):
        raise ValueError("group_edges must be strictly increasing")
    return edges


def _infer_edges(n_groups: int, values: Tensor) -> Tensor:
    """Equal-width bin edges spanning the finite training-target range."""
    finite = values.detach().reshape(-1)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        raise ValueError(
            "cannot infer group_edges: no finite target values were supplied to fit_group_edges()"
        )
    low = float(finite.min().item())
    high = float(finite.max().item())
    if high <= low:
        # Degenerate target: centre ``n_groups`` equal-width bins around the value.
        pad = 1e-6 * max(1.0, abs(low))
        low, high = low - pad, high + pad
    return torch.linspace(low, high, n_groups + 1, dtype=torch.float64)


def _right_pad(tensor: Tensor, ndim: int) -> Tensor:
    """Append singleton dimensions to ``tensor`` until it has ``ndim`` dims."""
    if tensor.dim() > ndim:
        raise ValueError(f"expected at most {ndim}-D tensor, got shape {tuple(tensor.shape)}")
    return tensor.reshape(tensor.shape + (1,) * (ndim - tensor.dim()))


def _broadcast_to(tensor: Tensor, shape: torch.Size) -> Tensor:
    """Broadcast ``tensor`` to ``shape`` with a friendly error message."""
    if tensor.shape == shape:
        return tensor
    try:
        padded = _right_pad(tensor, len(shape))
        return torch.broadcast_to(padded, shape)
    except RuntimeError as exc:
        raise ValueError(
            f"tensor of shape {tuple(tensor.shape)} cannot be broadcast to {tuple(shape)}"
        ) from exc


def _align_shapes(y_pred: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Right-pad the lower-rank of ``y_pred`` / ``target`` so both share a rank."""
    if y_pred.shape == target.shape:
        return y_pred, target
    if target.dim() < y_pred.dim() and y_pred.shape[: target.dim()] == target.shape:
        return y_pred, _right_pad(target, y_pred.dim())
    if y_pred.dim() < target.dim() and target.shape[: y_pred.dim()] == y_pred.shape:
        return _right_pad(y_pred, target.dim()), target
    raise ValueError(
        f"y_pred shape {tuple(y_pred.shape)} and target shape {tuple(target.shape)} "
        "must have the same shape or differ only in trailing singleton dimensions"
    )


def _project_floor(weights: Tensor, floor: float) -> Tensor:
    """Euclidean projection of simplex weights onto ``{q_g >= floor, sum q = 1}``.

    The projection shifts by ``floor`` and projects the residual mass onto a
    scaled simplex.  It is only a no-op when the floor is already satisfied.
    """
    n_groups = weights.numel()
    remaining = 1.0 - n_groups * floor
    if remaining <= 0.0:
        return torch.full_like(weights, 1.0 / n_groups)
    residual = weights - floor
    # Project ``residual`` onto {p >= 0, sum p = remaining} (scaled simplex):
    # find the threshold ``tau`` with sum(max(residual - tau, 0)) = remaining.
    sorted_residual, _ = torch.sort(residual, descending=True)
    cumulative = torch.cumsum(sorted_residual, dim=0) - remaining
    ranks = torch.arange(1, n_groups + 1, dtype=weights.dtype, device=weights.device)
    positive = sorted_residual - cumulative / ranks > 0
    active = int(torch.nonzero(positive).max().item())
    tau = cumulative[active] / (active + 1)
    return (residual - tau).clamp_min(0.0) + floor


class ContinuousGroupDRO(nn.Module):
    r"""Group DRO over target-value bins for continuous regression targets.

    The continuous target is discretised into ``G`` contiguous regimes with
    ``G + 1`` monotonically increasing ``group_edges``.  Every batch computes
    the mean base loss per group; in training mode an exponentiated-gradient
    step ascends the adversarial group weights, and the batch loss is the
    ``q``-weighted mixture of group means with ``q`` detached:

    .. math::

        \mathcal{L}(\theta) = \sum_{g=1}^{G} q_g \, L_g(\theta),
        \qquad
        q_g \leftarrow \frac{q_g e^{\eta L_g}}{\sum_{g'} q_{g'} e^{\eta L_{g'}}}

    Groups with no samples in a batch keep their previous log-weight before the
    renormalisation.  The update is performed in the log domain, so very large
    finite losses cannot overflow the softmax.

    Bin assignment is half-open: a target in ``[e_g, e_{g+1})`` belongs to group
    ``g``; a target exactly on an interior edge belongs to the *higher* group;
    values below ``e_0`` / above ``e_G`` are clamped into the first / last
    group.  Non-finite targets cannot be assigned and are excluded from the
    objective (pass ``mask`` for missing targets as in the other losses).

    Args:
        group_edges: Either ``G + 1`` strictly increasing bin edges in target
            space (e.g. ``[-4.0, -3.0, -2.0, -1.0, 0.5]`` defines four
            regimes), or an integer ``G`` meaning ``G`` equal-width bins
            inferred from the training targets via :meth:`fit_group_edges`
            (or automatically on the first forward pass).
        eta: Exponentiated-gradient step size (learning rate of the adversary).
            Must be positive.
        base_loss: ``"mse"``, ``"mae"``, or an ``nn.Module`` computing the
            *element-wise* loss (i.e. configured with ``reduction="none"``).
        min_group_weight: Optional lower bound ``m`` on every group weight,
            projected after each update.  Must lie in ``[0, 1 / G]``.  A
            positive floor guarantees every group keeps some influence at the
            cost of worst-group optimality and extra variance; ``0`` is the
            exact Group DRO objective.
        update_every: Refresh the group weights on every ``update_every``-th
            training forward pass (see :attr:`n_updates` for the exact
            schedule; ``1`` updates on every batch).
        reduction: ``"mean"`` (the DRO objective itself), ``"sum"`` (``N``
            times the objective, matching sum reduction), or ``"none"``
            (per-sample surrogate losses whose mean is the objective).

    Attributes:
        log_q: Buffer of normalised log group weights, shape ``(G,)``.
        n_updates: Buffer counting training-mode forward passes since
            :meth:`reset`; a refresh happens when ``n_updates % update_every``
            is ``0``.
        edges: Buffer of bin edges (empty until inferred for an integer
            ``group_edges``).

    Example:
        >>> import torch
        >>> from torchregress.algorithms.group_dro import ContinuousGroupDRO
        >>>
        >>> dro = ContinuousGroupDRO([-1.0, 0.0, 1.0], eta=0.1)
        >>> y_pred = torch.zeros(6, 1)
        >>> y = torch.tensor([[-0.5], [-0.8], [0.2], [0.4], [0.9], [1.2]])
        >>> loss = dro(y_pred, y)  # training mode: q is updated
        >>> dro.group_weights()  # shifted towards the worst-performing bin
        tensor(...)
    """

    log_q: Tensor
    n_updates: Tensor
    edges: Tensor

    def __init__(
        self,
        group_edges: Sequence[float] | int,
        *,
        eta: float = 0.01,
        base_loss: str | nn.Module = "mse",
        min_group_weight: float = 0.0,
        update_every: int = 1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if isinstance(group_edges, bool):
            raise TypeError("group_edges must be a sequence of edges or an integer group count")
        if isinstance(group_edges, int):
            if group_edges < 1:
                raise ValueError(f"integer group_edges must be >= 1, got {group_edges}")
            n_groups = group_edges
            edges = torch.empty(0, dtype=torch.float64)
            self._inferred_n_groups: int | None = n_groups
        else:
            edges = _validate_edges(group_edges)
            n_groups = edges.numel() - 1
            self._inferred_n_groups = None

        if not (float(eta) > 0.0) or not math.isfinite(float(eta)):
            raise ValueError(f"eta must be a positive finite number, got {eta}")
        self.eta = float(eta)
        self.update_every = int(update_every)
        if self.update_every < 1:
            raise ValueError(f"update_every must be >= 1, got {update_every}")
        if not (0.0 <= float(min_group_weight) <= 1.0 / n_groups):
            raise ValueError(
                "min_group_weight must lie in [0, 1 / n_groups] = "
                f"[0, {1.0 / n_groups}], got {min_group_weight}"
            )
        self.min_group_weight = float(min_group_weight)
        self.reduction = validate_reduction(reduction)

        if isinstance(base_loss, str):
            self.base_loss: str | nn.Module = base_loss.lower()
            if self.base_loss not in _BASE_LOSSES:
                raise ValueError(
                    f"base_loss must be one of {sorted(_BASE_LOSSES)} or an nn.Module, "
                    f"got {base_loss!r}"
                )
        elif isinstance(base_loss, nn.Module):
            configured = getattr(base_loss, "reduction", "none")
            if configured != "none":
                raise ValueError(
                    "custom base_loss modules must use reduction='none' so the loss is "
                    f"element-wise, got reduction={configured!r}"
                )
            self.base_loss = base_loss
        else:
            raise TypeError(
                f"base_loss must be a string or nn.Module, got {type(base_loss).__name__}"
            )

        self.register_buffer(
            "log_q",
            torch.full((n_groups,), -math.log(n_groups), dtype=torch.float64),
        )
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("edges", edges)

    # ── introspection ──────────────────────────────────────────────────

    @property
    def n_groups(self) -> int:
        """Number of target regimes ``G``."""
        return int(self.log_q.numel())

    def group_weights(self) -> Tensor:
        """Current simplex weights ``q`` as a **detached** tensor of shape ``(G,)``."""
        return torch.exp(self.log_q.detach())

    def group_indices(self, target: Tensor) -> Tensor:
        """Assign per-sample target values to groups (no side effects).

        Args:
            target: Targets of shape ``(N,)`` or ``(N, D)``.  Multi-output
                targets are binned by their mean over the trailing dimensions.

        Returns:
            Integer tensor of shape ``(N,)`` with values in ``[0, G)``.
            Values outside the edges are clamped to the boundary groups and
            non-finite targets are reported as group ``0``.
        """
        centers, finite = self._target_centers(target)
        safe = torch.where(
            finite, centers, torch.zeros((), dtype=centers.dtype, device=centers.device)
        )
        return self._assign_groups(safe)

    def group_counts(self, target: Tensor) -> Tensor:
        """Number of samples per group for ``target`` (non-finite targets are skipped)."""
        centers, finite = self._target_centers(target)
        safe = torch.where(
            finite, centers, torch.zeros((), dtype=centers.dtype, device=centers.device)
        )
        index = self._assign_groups(safe)
        return torch.bincount(index[finite], minlength=self.n_groups).to(torch.long)

    def group_losses(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
    ) -> Tensor:
        """Per-group mean base losses for the current batch.

        This is a diagnostic: it never updates the group weights and works the
        same in train and eval mode.  Groups absent from the batch are reported
        as ``NaN``.

        Args:
            y_pred: Model predictions, shape ``(N,)`` or ``(N, D)``.
            target: Ground-truth targets broadcastable to ``y_pred``.
            mask: Optional boolean mask (``False`` = missing).
            weights: Optional non-negative sample weights.

        Returns:
            Tensor of shape ``(G,)`` with the masked, weight-normalised mean
            base loss of each group.
        """
        return self._batch_stats(y_pred, target, mask, weights).group_losses

    def fit_group_edges(self, targets: Tensor) -> ContinuousGroupDRO:
        """Infer equal-width bin edges from training targets.

        Only meaningful when ``group_edges`` was an integer; for explicit edges
        this is a no-op.  Called automatically on the first forward pass if the
        edges have not been set yet.  The inferred edges are **kept** by
        :meth:`reset`.

        Args:
            targets: Training targets of any shape (non-finite values ignored).

        Returns:
            self (for chaining).
        """
        if self._inferred_n_groups is None:
            return self
        edges = _infer_edges(self._inferred_n_groups, targets)
        self.edges = edges.to(device=targets.device)
        return self

    def reset(self) -> None:
        """Restore uniform group weights and zero the update counter."""
        with torch.no_grad():
            self.log_q.fill_(-math.log(self.n_groups))
            self.n_updates.zero_()

    # ── internals ─────────────────────────────────────────────────────

    def _target_centers(self, target: Tensor) -> tuple[Tensor, Tensor]:
        """Per-sample target centre used for binning, plus a finite mask."""
        if target.dim() == 0:
            raise ValueError("target must have at least one dimension (batch dimension)")
        if target.dim() == 1:
            centers = target
            return centers, torch.isfinite(centers)
        flat = target.reshape(target.shape[0], -1)
        entries = torch.isfinite(flat)
        finite = entries.all(dim=1)
        safe = torch.where(entries, flat, torch.zeros((), dtype=flat.dtype, device=flat.device))
        centers = safe.sum(dim=1) / entries.sum(dim=1).clamp_min(1).to(safe.dtype)
        return centers, finite

    def _ensure_edges(self, values: Tensor) -> Tensor:
        if self.edges.numel() == 0:
            if self._inferred_n_groups is None:
                raise RuntimeError("group edges have not been initialised; call fit_group_edges()")
            self.fit_group_edges(values)
        return self.edges

    def _assign_groups(self, values: Tensor) -> Tensor:
        """Map finite per-sample target centres to group indices."""
        edges = self._ensure_edges(values.detach())
        cast_edges = edges.to(device=values.device)
        index = (
            torch.bucketize(values.detach().to(dtype=cast_edges.dtype), cast_edges, right=True) - 1
        )
        return index.clamp(0, self.n_groups - 1)

    def _elementwise_base_loss(self, y_pred: Tensor, target: Tensor) -> Tensor:
        base_loss = self.base_loss
        if isinstance(base_loss, nn.Module):
            elementwise = base_loss(y_pred, target)
            if elementwise.shape != y_pred.shape:
                raise ValueError(
                    "custom base_loss must return an element-wise loss with shape "
                    f"{tuple(y_pred.shape)}, got {tuple(elementwise.shape)}"
                )
            return elementwise
        residuals = y_pred - target
        if base_loss == "mse":
            return residuals**2
        return residuals.abs()

    def _batch_stats(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Tensor | None,
        weights: Tensor | None,
    ) -> _BatchStats:
        """Per-sample losses, group assignment and per-group statistics."""
        if y_pred.dim() == 0:
            raise ValueError("y_pred and target must have at least one dimension")
        if y_pred.shape[0] != target.shape[0]:
            raise ValueError(
                f"y_pred and target must share the batch dimension, got {tuple(y_pred.shape)} "
                f"and {tuple(target.shape)}"
            )
        y_pred, target = _align_shapes(y_pred, target)
        if y_pred.shape[0] == 0:
            raise ValueError("GroupDRO requires a non-empty batch")
        elementwise = self._elementwise_base_loss(y_pred, target)
        trailing = tuple(range(1, elementwise.dim()))
        n_samples = elementwise.shape[0]
        zeros = torch.zeros((), dtype=elementwise.dtype, device=elementwise.device)

        mask_full = None if mask is None else _broadcast_to(mask, elementwise.shape).to(torch.bool)
        if mask_full is None:
            per_sample = elementwise.mean(trailing) if trailing else elementwise
            valid = torch.isfinite(per_sample)
        else:
            masked = torch.where(mask_full, elementwise, zeros)
            if trailing:
                valid_count = mask_full.sum(trailing)
                per_sample = masked.sum(trailing) / valid_count.clamp_min(1)
                valid = valid_count > 0
            else:
                per_sample = masked
                valid = mask_full
            valid = valid & torch.isfinite(per_sample)

        # ── user sample weights (per-sample, element-wise weights averaged) ──
        if weights is None:
            sample_weights = torch.ones(
                n_samples, dtype=elementwise.dtype, device=elementwise.device
            )
        else:
            weight_full = _broadcast_to(weights, elementwise.shape)
            weight_check = weight_full[mask_full] if mask_full is not None else weight_full
            weight_check = weight_check[torch.isfinite(weight_check)]
            if weight_check.numel() > 0 and bool((weight_check < 0).any()):
                raise ValueError("weights must be non-negative")
            if not bool(torch.isfinite(weight_full[valid]).all()):
                raise ValueError("weights must be finite where the loss is valid")
            if trailing:
                if mask_full is not None:
                    counts = mask_full.sum(trailing).clamp_min(1)
                    sample_weights = (
                        torch.where(mask_full, weight_full, zeros).sum(trailing) / counts
                    )
                else:
                    sample_weights = weight_full.mean(trailing)
            else:
                sample_weights = weight_full
            valid = valid & (sample_weights > 0)
        sample_weights = torch.where(valid, sample_weights, zeros)
        per_sample = torch.where(valid, per_sample, zeros)

        # ── group assignment from the (mask-aware) target centre ─────────
        target_full = torch.broadcast_to(target, elementwise.shape)
        if trailing:
            if mask_full is None:
                center = target_full.mean(trailing)
                center_valid = torch.isfinite(target_full).reshape(n_samples, -1).all(dim=1)
            else:
                entries = mask_full & torch.isfinite(target_full)
                valid_count = entries.sum(trailing)
                center = torch.where(entries, target_full, zeros).sum(
                    trailing
                ) / valid_count.clamp_min(1)
                center_valid = valid_count > 0
        else:
            center = target_full
            center_valid = torch.isfinite(target_full)
            if mask_full is not None:
                center_valid = center_valid & mask_full
        valid = valid & center_valid
        safe_center = torch.where(center_valid, center, zeros)
        safe_center = torch.where(valid, safe_center, zeros)
        if not bool(valid.any()):
            empty = torch.full(
                (self.n_groups,), float("nan"), dtype=elementwise.dtype, device=elementwise.device
            )
            return _BatchStats(
                per_sample=per_sample,
                group_index=torch.zeros(n_samples, dtype=torch.long, device=elementwise.device),
                valid=valid,
                weights=sample_weights,
                group_losses=empty,
                group_weight_sums=torch.zeros_like(empty),
            )

        if self.edges.numel() == 0:
            # Infer bin edges from the *valid* targets only (invalid samples hold
            # zero placeholders that would otherwise skew the equal-width range).
            self.fit_group_edges(safe_center[valid])
        group_index = self._assign_groups(safe_center.detach())
        index = group_index[valid]
        weight_valid = sample_weights[valid]
        loss_valid = per_sample[valid]
        group_weight_sums = torch.zeros(
            self.n_groups, dtype=elementwise.dtype, device=elementwise.device
        ).index_add(0, index, weight_valid)
        group_loss_sums = torch.zeros(
            self.n_groups, dtype=elementwise.dtype, device=elementwise.device
        ).index_add(0, index, weight_valid * loss_valid)
        counts = torch.bincount(index, minlength=self.n_groups)
        safe_sums = group_weight_sums.clamp_min(torch.finfo(elementwise.dtype).tiny)
        group_losses = group_loss_sums / safe_sums
        group_losses = torch.where(
            counts > 0, group_losses, torch.full_like(group_losses, float("nan"))
        )
        return _BatchStats(
            per_sample=per_sample,
            group_index=group_index,
            valid=valid,
            weights=sample_weights,
            group_losses=group_losses,
            group_weight_sums=group_weight_sums,
        )

    @torch.no_grad()
    def _ascend_group_weights(self, group_losses: Tensor) -> None:
        """Exponentiated-gradient ascent step on the simplex (log-domain)."""
        losses = group_losses.detach().to(device=self.log_q.device, dtype=self.log_q.dtype)
        finite = torch.isfinite(losses)
        delta = torch.where(
            finite,
            (self.eta * losses).clamp(-_MAX_LOG_UPDATE, _MAX_LOG_UPDATE),
            torch.zeros((), dtype=self.log_q.dtype, device=self.log_q.device),
        )
        updated = self.log_q + delta
        updated = updated - torch.logsumexp(updated, dim=0)
        if self.min_group_weight > 0.0:
            updated = torch.log(_project_floor(torch.exp(updated), self.min_group_weight))
        self.log_q.copy_(updated)

    def _surrogate_loss(self, stats: _BatchStats) -> Tensor:
        """Per-sample surrogate losses whose mean is the DRO objective."""
        weights = self.group_weights().to(
            dtype=stats.per_sample.dtype, device=stats.per_sample.device
        )
        n_valid = stats.valid.sum().to(dtype=stats.per_sample.dtype)
        group_mass = stats.group_weight_sums[stats.group_index].clamp_min(
            torch.finfo(stats.per_sample.dtype).tiny
        )
        coefficient = weights[stats.group_index] * n_valid / group_mass
        coefficient = torch.where(stats.valid, coefficient, torch.zeros_like(coefficient))
        return coefficient.detach() * stats.weights * stats.per_sample

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the group-DRO loss, updating the adversarial weights in training.

        In training mode the per-group means feed one exponentiated-gradient
        step (subject to ``update_every``); the returned loss then uses the
        refreshed, detached weights.  In eval mode the group weights are frozen,
        so repeated calls on the same input are idempotent.

        Args:
            y_pred: Model predictions, shape ``(N,)`` or ``(N, D)``.
            target: Ground-truth targets broadcastable to ``y_pred``.
            mask: Optional boolean mask (``False`` = missing; masked entries
                contribute zero to the per-group means and their denominator).
            weights: Optional non-negative sample weights, broadcastable to
                ``target``.  With more than one non-batch dimension they are
                averaged per sample (mask-aware), matching the shared loss
                contract.
            **kwargs: Ignored (accepted for API compatibility).

        Returns:
            ``reduction="mean"`` returns the scalar DRO objective
            :math:`\\sum_g q_g L_g`; ``"sum"`` returns ``N`` times that value;
            ``"none"`` returns the per-sample surrogate losses.
        """
        stats = self._batch_stats(y_pred, target, mask, weights)
        training = self.training
        if training:
            update_due = int(self.n_updates) % self.update_every == 0
            self.n_updates += 1
            if update_due and self.n_groups > 1:
                self._ascend_group_weights(stats.group_losses)

        if self.n_groups == 1:
            # G = 1: the adversarial problem is trivial; reduce the plain loss.
            if self.reduction == "none":
                return stats.weights * stats.per_sample
            weighted = (stats.weights * stats.per_sample).sum()
            if self.reduction == "sum":
                return weighted
            denominator = stats.weights[stats.valid].sum()
            return weighted / denominator.clamp_min(torch.finfo(stats.per_sample.dtype).tiny)

        surrogate = self._surrogate_loss(stats)
        if self.reduction == "none":
            return surrogate
        total = surrogate.sum()
        if self.reduction == "sum":
            return total
        n_valid = stats.valid.sum().clamp_min(1).to(dtype=surrogate.dtype)
        return total / n_valid


class GroupDROTrainer:
    """Lightweight end-to-end trainer for :class:`ContinuousGroupDRO`.

    Trains a model with Adam on the group-DRO objective, mirroring the style of
    :class:`torchregress.algorithms.warmup_mc.WarmupMCTrainer`: batches are
    gathered by device-side permutation (no ``DataLoader``), shuffling uses a
    local seeded generator so runs are reproducible, and no global RNG state is
    touched.

    Args:
        model_factory: Callable returning a freshly initialised model.
        group_edges: ``G + 1`` bin edges or an integer ``G`` (equal-width bins
            inferred from the training targets).
        base_loss: ``"mse"``, ``"mae"``, or an element-wise ``nn.Module``.
        eta: Exponentiated-gradient step size for the group weights.
        epochs: Number of full passes over the training data.
        lr: Adam learning rate.
        batch_size: Mini-batch size (clipped to the dataset size).
        weight_decay: Adam weight decay (L2 penalty).
        seed: Seed for the local shuffling generator.  ``None`` draws a random
            seed at :meth:`fit` time; the results are deterministic when set.

    Attributes:
        model: The fitted model (``None`` before :meth:`fit`).
        loss_fn: The fitted :class:`ContinuousGroupDRO` module (``None`` before
            :meth:`fit`), useful to inspect the adversarial weights per batch.
        group_weights_: Final group weights ``q``, shape ``(G,)``.
        group_losses_: Final per-group mean losses on the training set.
        history_: One dict per epoch with keys ``"loss"`` (mean batch objective)
            and ``"q"`` (group weights after that epoch, as a list of floats).

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from torchregress.algorithms.group_dro import GroupDROTrainer
        >>>
        >>> def make_model():
        ...     return nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
        >>>
        >>> trainer = GroupDROTrainer(
        ...     model_factory=make_model,
        ...     group_edges=[-3.0, 0.0, 3.0],
        ...     epochs=20,
        ...     seed=0,
        ... )
        >>> trainer.fit(X_train, y_train)  # doctest: +SKIP
        >>> trainer.group_weights_  # doctest: +SKIP
        tensor([...])
    """

    def __init__(
        self,
        *,
        model_factory: Callable[[], nn.Module],
        group_edges: Sequence[float] | int,
        base_loss: str | nn.Module = "mse",
        eta: float = 0.01,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.model_factory = model_factory
        self.group_edges = group_edges
        self.base_loss = base_loss
        self.eta = float(eta)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.seed = seed
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if not (self.lr > 0.0):
            raise ValueError(f"lr must be positive, got {lr}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not (self.weight_decay >= 0.0):
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")

        self.model: nn.Module | None = None
        self.loss_fn: ContinuousGroupDRO | None = None
        self.group_weights_: Tensor | None = None
        self.group_losses_: Tensor | None = None
        self.history_: list[dict[str, Any]] = []

    def fit(self, X: Tensor, y: Tensor) -> GroupDROTrainer:
        """Fit the model with the group-DRO objective.

        Args:
            X: Inputs of shape ``(N, D)``.
            y: Targets of shape ``(N,)`` or ``(N, 1)``.

        Returns:
            self (for chaining).
        """
        check_tensor(X, "X")
        check_tensor(y, "y")
        if X.dim() != 2:
            raise ValueError(f"X must be a 2-D tensor (N, D), got shape {tuple(X.shape)}")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if y.dim() == 1:
            y = y.unsqueeze(1)
        if y.dim() != 2:
            raise ValueError(f"y must have shape (N,) or (N, 1), got shape {tuple(y.shape)}")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}"
            )

        device = X.device
        generator = torch.Generator(device="cpu")
        if self.seed is not None:
            generator.manual_seed(int(self.seed))
        if self.seed is not None:
            # Local RNG scope: deterministic model init without touching global state.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(self.seed))
                model = self.model_factory()
        else:
            model = self.model_factory()
        model = model.to(device)
        loss_fn = ContinuousGroupDRO(self.group_edges, eta=self.eta, base_loss=self.base_loss).to(
            device
        )
        loss_fn.fit_group_edges(y)
        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        n_samples = X.shape[0]
        batch_size = max(1, min(self.batch_size, n_samples))
        self.history_ = []
        for _ in range(self.epochs):
            model.train()
            loss_fn.train()
            order = torch.randperm(n_samples, generator=generator).to(device)
            running = 0.0
            n_batches = 0
            for start in range(0, n_samples, batch_size):
                batch = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(X[batch]), y[batch])
                loss.backward()
                optimizer.step()
                running += float(loss.detach())
                n_batches += 1
            self.history_.append(
                {
                    "loss": running / n_batches,
                    "q": loss_fn.group_weights().detach().cpu().tolist(),
                }
            )

        model.eval()
        loss_fn.eval()
        self.group_weights_ = loss_fn.group_weights().detach()
        with torch.no_grad():
            group_loss_sums = torch.zeros(loss_fn.n_groups, dtype=y.dtype, device=device)
            group_counts = torch.zeros(loss_fn.n_groups, dtype=torch.long, device=device)
            for start in range(0, n_samples, batch_size):
                y_batch = y[start : start + batch_size]
                batch_losses = loss_fn.group_losses(model(X[start : start + batch_size]), y_batch)
                counts = loss_fn.group_counts(y_batch)
                present = counts > 0
                group_loss_sums[present] += batch_losses[present] * counts[present].to(y.dtype)
                group_counts += counts
            self.group_losses_ = group_loss_sums / group_counts.clamp_min(1).to(y.dtype)
        self.model = model
        self.loss_fn = loss_fn
        return self

    def predict(self, X: Tensor) -> Tensor:
        """Predict with the fitted model.

        Args:
            X: Inputs of shape ``(N, D)``.

        Returns:
            Model predictions, shape ``(N, 1)``.
        """
        check_tensor(X, "X")
        if self.model is None:
            raise RuntimeError("GroupDROTrainer must be fit before calling predict")
        if X.dim() != 2:
            raise ValueError(f"X must be a 2-D tensor (N, D), got shape {tuple(X.shape)}")
        device = next(self.model.parameters()).device
        with torch.no_grad():
            return self.model(X.to(device))
