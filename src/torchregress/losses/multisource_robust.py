"""Normalized small-block outlier mixtures for one shared Gaussian latent."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import torch
from torch import Tensor

from .multisource import (
    _NOT_FINITE,
    MarginalizedMultiSourceGaussianNLLLoss,
    PackedObservations,
    _LazyDiagnostics,
)

_MIXTURE = "independent measurement-block indicators; one shared Gaussian latent"


class _WeightedLogSumExp(torch.autograd.Function):
    """Stable log mixture with one-sided weight derivatives at zero weights."""

    @staticmethod
    def forward(ctx, log_density: Tensor, weights: Tensor) -> Tensor:
        """Mix over the last dimension."""
        result = torch.logsumexp(log_density + weights.log(), dim=-1)
        ctx.save_for_backward(log_density, weights, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        log_density, weights, result = ctx.saved_tensors
        result, grad_output = result.unsqueeze(-1), grad_output.unsqueeze(-1)
        density_gradient = (log_density + weights.log() - result).exp()
        weight_gradient = (log_density - result).exp()
        return grad_output * density_gradient, grad_output * weight_gradient


def _check(flags: list[tuple[Tensor, str]], extra: list[Tensor]) -> list[int]:
    """Raise the first failed check; return the ``extra`` values (one host sync)."""
    values = torch.cat(
        [torch.stack([flag for flag, _ in flags]).long(), *(x.long() for x in extra)]
    ).tolist()
    for (_, message), bad in zip(flags, values, strict=False):
        if bad:
            raise ValueError(message)
    return values[len(flags) :]


class RobustMultiSourceGaussianNLLLoss(MarginalizedMultiSourceGaussianNLLLoss):
    r"""Marginalize coherent inlier/outlier assignments of measurement blocks.

    For block indicators ``z_b ~ Bernoulli(p_b)``, the observation noise block
    is ``R_b`` when inlying and ``s_b**2 R_b`` when outlying. Every assignment
    has the *same single* latent covariance contribution ``H V H.T``. Its
    normalized Gaussian density is mixed with the normalized product Bernoulli
    probabilities. Full within-block covariance is preserved. Cross-block
    measurement covariance is rejected, since independent block inflation
    would otherwise require a different, explicitly specified noise model.

    ``observations.block_ids`` must contain nonnegative integer IDs for observed
    scalars. IDs are local to each star and need not be contiguous. Missing
    blocks integrate to one and are not enumerated. Optional call arguments
    ``outlier_probabilities`` and ``outlier_scales`` are scalars or ``[B,K]``;
    entries must be constant within each observed block. Their gradients are
    retained, including probability endpoints. Default parameters are fixed
    floats, not learned parameters. Scales are standard-deviation multipliers
    of at least one. This model inflates *measurement* noise, not latent spread.

    Enumeration is exponential: both ``max_blocks`` and ``max_assignments``
    bound work explicitly. Use a small, scientifically justified block model.
    Each assignment is scored for the whole batch at once (objects with fewer
    blocks skip assignments they do not have). The Gaussian linearization and
    covariance-jitter caveats of the base class apply. No pseudo-Huber
    transformation or unnormalized density is used.
    """

    def __init__(
        self,
        reduction: Literal["none", "sum", "mean"] = "mean",
        method: Literal["auto", "direct", "information"] = "auto",
        jitter: float = 0.0,
        max_jitter: float = 0.0,
        *,
        outlier_probability: float = 0.05,
        outlier_scale: float = 10.0,
        max_blocks: int = 8,
        max_assignments: int = 256,
    ) -> None:
        super().__init__(reduction, method, jitter, max_jitter)
        if not math.isfinite(outlier_probability) or not 0 <= outlier_probability <= 1:
            raise ValueError("Outlier probability must be finite in [0,1]")
        if not math.isfinite(outlier_scale) or outlier_scale < 1:
            raise ValueError("Outlier standard-deviation scale must be finite and >= 1")
        if (
            isinstance(max_blocks, bool)
            or not isinstance(max_blocks, int)
            or max_blocks < 1
            or isinstance(max_assignments, bool)
            or not isinstance(max_assignments, int)
            or max_assignments < 1
        ):
            raise ValueError("Positive integer enumeration bounds required")
        self.outlier_probability = float(outlier_probability)
        self.outlier_scale = float(outlier_scale)
        self.max_blocks = max_blocks
        self.max_assignments = max_assignments

    @staticmethod
    def _expand(value: Tensor | float, obs: PackedObservations, name: str) -> Tensor:
        result = torch.as_tensor(
            value,
            device=obs.values.device,
            dtype=(
                None
                if isinstance(value, Tensor)
                else torch.promote_types(obs.values.dtype, torch.float32)
            ),
        )
        if not result.is_floating_point():
            result = result.to(torch.promote_types(obs.values.dtype, torch.float32))
        if result.ndim == 0:
            return result.expand_as(obs.values)
        if result.shape != obs.values.shape:
            raise ValueError(f"{name} must be scalar or [B,K]")
        return result

    def per_star(
        self,
        mean: Tensor,
        scale_tril: Tensor,
        obs: PackedObservations,
        *,
        outlier_probabilities: Tensor | float | None = None,
        outlier_scales: Tensor | float | None = None,
    ) -> Tensor:
        """Return per-star normalized mixture NLLs, zero for empty stars."""
        # An outer AMP context must not demote the explicit fp32/fp64 solves.
        with torch.autocast(device_type=mean.device.type, enabled=False):
            return self._mixture_nll(mean, scale_tril, obs, outlier_probabilities, outlier_scales)

    def _set_diagnostics(self, base: Mapping[str, object], block_count: Tensor) -> None:
        counts = torch.where(block_count == 0, 1, 1 << block_count).detach()
        self.last_diagnostics = _LazyDiagnostics(
            lambda: {
                "inlier": base,
                "assignment_counts": counts.tolist(),
                "mixture": _MIXTURE,
            }
        )

    def _mixture_nll(
        self,
        mean: Tensor,
        scale_tril: Tensor,
        obs: PackedObservations,
        outlier_probabilities: Tensor | float | None,
        outlier_scales: Tensor | float | None,
    ) -> Tensor:
        # Validate all base shapes and selected values before inspecting blocks.
        inlier, base_batch, base_r = self._marginal_nll(mean, scale_tril, obs)
        base_diagnostics = self.last_diagnostics
        if obs.block_ids is None:
            raise ValueError("Robust likelihood requires explicit block_ids")
        probabilities = self._expand(
            self.outlier_probability if outlier_probabilities is None else outlier_probabilities,
            obs,
            "outlier_probabilities",
        )
        scales = self._expand(
            self.outlier_scale if outlier_scales is None else outlier_scales,
            obs,
            "outlier_scales",
        )
        mask = obs.mask
        batch_size, k = mask.shape
        bad_p = ~torch.isfinite(probabilities) | (probabilities < 0) | (probabilities > 1)
        flags = [
            ((mask & bad_p).any(), "Observed outlier probabilities must be finite in [0,1]"),
            (
                (mask & (~torch.isfinite(scales) | (scales < 1))).any(),
                "Observed outlier scales must be finite and >= 1",
            ),
        ]
        work_dtype = torch.promote_types(
            inlier.dtype, torch.promote_types(probabilities.dtype, scales.dtype)
        )
        probabilities, scales = probabilities.to(work_dtype), scales.to(work_dtype)
        if base_batch is None or base_r is None:  # no objects or no measurement slots
            _check(flags, [])
            self._set_diagnostics(base_diagnostics, mask.sum(-1) * 0)
            return inlier

        # Dense per-object block ranks: observed slots sorted by block ID come first.
        ids = obs.block_ids.long()
        flags.append((((ids < 0) & mask).any(), "Observed block IDs must be nonnegative"))
        by_id = torch.argsort(ids, dim=-1, stable=True)
        observed_first = torch.argsort(
            (~mask.gather(-1, by_id)).to(torch.int32), dim=-1, stable=True
        )
        order = by_id.gather(-1, observed_first)
        sorted_ids, sorted_mask = ids.gather(-1, order), mask.gather(-1, order)
        starts = torch.cat(
            (sorted_mask[:, :1], sorted_mask[:, 1:] & (sorted_ids[:, 1:] != sorted_ids[:, :-1])),
            dim=-1,
        )
        rank = torch.empty_like(order).scatter(-1, order, starts.cumsum(-1) - 1)
        rank = torch.where(mask, rank, 0)
        block_count = starts.sum(-1)
        # count > max_blocks or 2**count > max_assignments.
        limit = min(self.max_blocks, self.max_assignments.bit_length() - 1)
        flags.append(
            ((block_count > limit).any(), "Robust assignment enumeration exceeds configured bound")
        )
        cross = (rank.unsqueeze(-1) != rank.unsqueeze(-2)) & (base_r != 0)
        flags.append((cross.any(), "Cross-block measurement covariance is unsupported"))
        # Unobserved slots scatter to a scratch column ``k``.
        slot = torch.where(mask, rank, k)
        varying = torch.zeros((), dtype=torch.bool, device=mask.device)
        block_parameters = []
        for values in (probabilities, scales):
            with torch.no_grad():
                for reduce in ("amin", "amax"):
                    extreme = values.new_zeros(batch_size, k + 1).scatter_reduce(
                        1, slot, values, reduce, include_self=False
                    )
                    varying = varying | (mask & (extreme.gather(1, slot) != values)).any()
            total = values.new_zeros(batch_size, k + 1).scatter_add(
                1, slot, torch.where(mask, values, 0)
            )
            size = values.new_zeros(batch_size, k + 1).scatter_add(1, slot, mask.to(values.dtype))
            # The block mean preserves gradients to all repeated scalar entries.
            block_parameters.append(total[:, :k] / size[:, :k].clamp_min(1))
        flags.append((varying, "Outlier parameters must be constant within each block"))
        block_p, block_s = block_parameters
        thresholds = torch.arange(1, limit + 1, device=mask.device)
        at_least = (block_count.unsqueeze(-1) >= thresholds).sum(0)
        extra = _check(flags, [block_count.max().reshape(1), at_least])
        most_blocks, rows_with = extra[0], extra[1:]
        self._set_diagnostics(base_diagnostics, block_count)
        if most_blocks == 0:
            return inlier

        # Outlying assignments are scored in the mixture work dtype.
        pending: list[tuple[Tensor, str]] = []
        if work_dtype == base_batch.counts.dtype:
            batch, r = base_batch, base_r
        else:
            batch, r, not_finite = self._build_batch(scale_tril, obs, work_dtype)
            pending.append((not_finite, _NOT_FINITE))
        assignments = 1 << most_blocks
        codes = torch.arange(assignments, device=mask.device)
        shifts = torch.arange(most_blocks, device=mask.device)
        bits = ((codes.unsqueeze(-1) >> shifts) & 1).bool()
        p = block_p[:, None, :most_blocks]
        mixture = torch.where(bits, p, 1 - p).prod(-1)
        # Assignments of absent blocks get weight zero and no gradient path.
        valid = codes < (1 << block_count).unsqueeze(-1)
        slot_bits = torch.where(mask, 1 << rank, 0)
        slot_scale = block_s.gather(1, rank)
        # Objects with the most blocks first: an assignment ``code`` needs
        # ``code.bit_length()`` blocks, so the objects that have it form a prefix.
        row_order = torch.argsort(block_count, descending=True, stable=True)
        densities = [-inlier.to(work_dtype)]
        for code in range(1, assignments):
            count = rows_with[code.bit_length() - 1]
            if count == batch_size:
                sub, sub_r, sub_bits, sub_scale = batch, r, slot_bits, slot_scale
            else:
                rows = row_order[:count]
                sub = batch.subset(rows, self.method)
                sub_r, sub_bits, sub_scale = r[rows], slot_bits[rows], slot_scale[rows]
            multipliers = torch.where((sub_bits & code) != 0, sub_scale, 1)
            noise = sub_r.to(work_dtype) * multipliers[:, :, None] * multipliers[:, None, :]
            finite = (~torch.isfinite(noise).all(), _NOT_FINITE)
            nll = self._solve(sub, noise, [*pending, finite])[0]
            if count == batch_size:
                densities.append(-nll)
            else:
                column = nll.new_zeros(batch_size).index_copy(0, row_order[:count], -nll)
                densities.append(column)
        log_density = torch.where(valid, torch.stack(densities, dim=-1), 0)
        weights = torch.where(valid, mixture, 0).to(log_density.dtype)
        mixed = -_WeightedLogSumExp.apply(log_density, weights)
        return torch.where(block_count == 0, inlier.to(mixed.dtype), mixed)

    def forward(
        self,
        mean: Tensor,
        scale_tril: Tensor,
        observations: PackedObservations,
        weights: Tensor | None = None,
        *,
        outlier_probabilities: Tensor | float | None = None,
        outlier_scales: Tensor | float | None = None,
    ) -> Tensor:
        """Use the base loss's nonempty-star weighted reduction convention."""
        per_object = self.per_star(
            mean,
            scale_tril,
            observations,
            outlier_probabilities=outlier_probabilities,
            outlier_scales=outlier_scales,
        )
        active = observations.mask.any(-1)
        if weights is None:
            weights = torch.ones_like(per_object)
        elif (
            weights.shape != per_object.shape
            or weights.device != per_object.device
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError("weights must be finite nonnegative [B] on the same device")
        effective = weights.to(per_object.dtype) * active
        weighted = per_object * effective
        if self.reduction == "none":
            return weighted
        if self.reduction == "sum":
            return weighted.sum()
        if effective.sum().item() == 0:
            return mean.sum() * 0
        return weighted.sum() / effective.sum()
