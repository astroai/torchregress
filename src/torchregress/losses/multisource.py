"""Joint Gaussian likelihood for sparse observations of one shared latent vector.

The likelihood is exact for a first-order linear observation model; it is not
an exact marginalization of a nonlinear observation function. Multiple sources
and anchors must be stacked for each object before calling the loss.

All objects of a batch are evaluated together. Unobserved slots are replaced by
decoupled unit-variance coordinates with zero residual and zero Jacobian, so
they add exactly nothing to the quadratic form or the log-determinant; the
``log(2 pi)`` constant counts observed slots only. Input validation is done
once per batch with a single host synchronization.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

_LOG_2PI = math.log(2 * math.pi)
_NOT_FINITE = "Observed values, mappings, and covariance must be finite"
_NOT_SYMMETRIC = "Observed measurement covariance must be symmetric"
_NOT_PD = "Observed measurement covariance is not positive definite"
_METHOD_NAMES = ("empty", "direct", "information")


@dataclass(frozen=True)
class PackedObservations:
    """Padded observations and their linearization at a latent predictive mean.

    ``values`` and ``mapped_mean`` are ``[B,K]``, ``jacobian`` is ``[B,K,D]``,
    ``measurement_covariance`` is ``[B,K,K]``, and ``mask`` is boolean ``[B,K]``.
    Unobserved entries may contain NaNs; only the selected subproblem is read.
    The covariance may include cross-source or anchor correlations. Duplicate
    measurements with unknown dependencies must be resolved by the caller.
    """

    values: Tensor
    mapped_mean: Tensor
    jacobian: Tensor
    measurement_covariance: Tensor
    mask: Tensor
    block_ids: Tensor | None = None


class _LazyDiagnostics(Mapping[str, object]):
    """Read-only diagnostics, copied to host lists only when first read."""

    def __init__(self, build: Callable[[], dict[str, object]]) -> None:
        self._build: Callable[[], dict[str, object]] | None = build
        self._data: dict[str, object] = {}

    def _items(self) -> dict[str, object]:
        if self._build is not None:
            self._data = self._build()
            self._build = None
        return self._data

    def __getitem__(self, key: str) -> object:
        return self._items()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items())

    def __len__(self) -> int:
        return len(self._items())

    def __repr__(self) -> str:
        return repr(self._items())


@dataclass(frozen=True)
class _Groups:
    """Objects scored by the direct or the information factorization."""

    kind: Literal["direct", "information", "mixed"]
    direct: Tensor | None = None
    information: Tensor | None = None


def _split_groups(method: str, use_information: Tensor) -> _Groups:
    if method == "direct":
        return _Groups("direct")
    if method == "information":
        return _Groups("information")
    total = use_information.numel()
    count = int(use_information.sum().item())
    if count == 0:
        return _Groups("direct")
    if count == total:
        return _Groups("information")
    order = torch.argsort(use_information.to(torch.int32), stable=True)
    return _Groups("mixed", order[: total - count], order[total - count :])


@dataclass(frozen=True)
class _Batch:
    """Padded, work-dtype representation of the observed subproblems."""

    residual: Tensor  # [B,K], zero where unobserved
    hl: Tensor  # [B,K,D] = H @ L, zero rows where unobserved
    mask: Tensor  # [B,K] bool
    maskf: Tensor  # [B,K] work dtype
    counts: Tensor  # [B] work dtype
    use_information: Tensor  # [B] bool, the per-object choice of ``method="auto"``
    groups: _Groups
    tolerance: float

    def subset(self, rows: Tensor, method: str) -> _Batch:
        use_information = self.use_information[rows]
        return _Batch(
            self.residual[rows],
            self.hl[rows],
            self.mask[rows],
            self.maskf[rows],
            self.counts[rows],
            use_information,
            _split_groups(method, use_information),
            self.tolerance,
        )


def _direct_terms(residual: Tensor, hl: Tensor, r_jittered: Tensor) -> tuple[Tensor, ...]:
    covariance = r_jittered + hl @ hl.mT
    factor, info = torch.linalg.cholesky_ex(covariance)
    whitened = torch.linalg.solve_triangular(factor, residual.unsqueeze(-1), upper=False)
    quadratic = whitened.square().sum((-2, -1))
    logdet = 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return quadratic, logdet, info, covariance


def _information_terms(residual: Tensor, hl: Tensor, r_factor: Tensor) -> tuple[Tensor, ...]:
    z = torch.linalg.solve_triangular(r_factor, residual.unsqueeze(-1), upper=False)
    a = torch.linalg.solve_triangular(r_factor, hl, upper=False)
    eye = torch.eye(hl.shape[-1], dtype=hl.dtype, device=hl.device)
    capacitance = eye + a.mT @ a
    factor, info = torch.linalg.cholesky_ex(capacitance)
    solution = torch.cholesky_solve(a.mT @ z, factor)
    quadratic = (z - a @ solution).square().sum((-2, -1)) + solution.square().sum((-2, -1))
    logdet = 2 * (
        r_factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        + factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    )
    return quadratic, logdet, info, capacitance


class MarginalizedMultiSourceGaussianNLLLoss(nn.Module):
    """Normalized NLL of all observed measurements sharing one Gaussian latent.

    With latent covariance ``L @ L.T`` and the supplied local Jacobian ``H``,
    each observed stacked vector has covariance ``R + H @ L @ L.T @ H.T``.
    The information backend whitens by the Cholesky factor of ``R`` and uses a
    residual-plus-ridge quadratic rather than subtracting large quadratic forms.
    No explicit matrix inverse is formed. Measurement covariance must be SPD;
    optional jitter is bounded and recorded in ``last_diagnostics``.

    All objects are factorized together as one padded batch: there is no
    per-object Python loop, and inputs are validated once per batch.

    Args:
        reduction: ``mean`` averages over nonempty stars, ``sum`` sums them,
            and ``none`` returns one loss per star (zero for an empty star).
        method: ``direct``, ``information``, or ``auto`` (information if K > D).
        jitter: Initial nonnegative measurement-covariance diagonal addition.
        max_jitter: Maximum allowed addition; defaults to no adaptive retry.
    """

    def __init__(
        self,
        reduction: Literal["none", "sum", "mean"] = "mean",
        method: Literal["auto", "direct", "information"] = "auto",
        jitter: float = 0.0,
        max_jitter: float = 0.0,
    ) -> None:
        super().__init__()
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("Unknown reduction")
        if method not in {"auto", "direct", "information"}:
            raise ValueError("Unknown factorization method")
        if not 0 <= jitter <= max_jitter or not math.isfinite(max_jitter):
            raise ValueError("Require 0 <= jitter <= finite max_jitter")
        self.reduction = reduction
        self.method = method
        self.jitter = jitter
        self.max_jitter = max_jitter
        self.last_diagnostics: Mapping[str, object] = {}

    def per_star(self, mean: Tensor, scale_tril: Tensor, obs: PackedObservations) -> Tensor:
        """Return normalized per-object NLLs, including the Gaussian constant."""
        # An outer AMP context must not demote the explicit fp32/fp64 solves.
        with torch.autocast(device_type=mean.device.type, enabled=False):
            return self._marginal_nll(mean, scale_tril, obs)[0]

    @staticmethod
    def _validate_shapes(mean: Tensor, scale_tril: Tensor, obs: PackedObservations) -> torch.dtype:
        """Check shapes, dtypes and devices (no data access); return the work dtype."""
        if mean.ndim != 2:
            raise ValueError("mean must be [B,D]")
        batch, dims = mean.shape
        if dims < 1 or scale_tril.shape != (batch, dims, dims):
            raise ValueError("scale_tril must be [B,D,D]")
        if obs.values.ndim != 2 or obs.values.shape[0] != batch:
            raise ValueError("values must be [B,K]")
        k = obs.values.shape[1]
        if (
            obs.mapped_mean.shape != (batch, k)
            or obs.jacobian.shape != (batch, k, dims)
            or obs.measurement_covariance.shape != (batch, k, k)
            or obs.mask.shape != (batch, k)
            or obs.mask.dtype != torch.bool
        ):
            raise ValueError("Observation shapes or mask dtype differ")
        if obs.block_ids is not None and (
            obs.block_ids.shape != (batch, k)
            or obs.block_ids.device != mean.device
            or obs.block_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("block_ids must be integer [B,K] on the same device")
        tensors = (
            scale_tril,
            obs.values,
            obs.mapped_mean,
            obs.jacobian,
            obs.measurement_covariance,
        )
        if not mean.is_floating_point() or any(not x.is_floating_point() for x in tensors):
            raise ValueError("All values and covariances must be floating point")
        if any(x.device != mean.device for x in (*tensors, obs.mask)):
            raise ValueError("All tensors must be on the same device")
        work_dtype = torch.promote_types(mean.dtype, scale_tril.dtype)
        for x in tensors[1:]:
            work_dtype = torch.promote_types(work_dtype, x.dtype)
        if work_dtype in (torch.float16, torch.bfloat16):
            work_dtype = torch.float32
        return work_dtype

    def _build_batch(
        self,
        scale_tril: Tensor,
        obs: PackedObservations,
        work_dtype: torch.dtype,
    ) -> tuple[_Batch, Tensor, Tensor]:
        """Return the padded batch, padded raw covariance, and a non-finite flag."""
        mask = obs.mask
        dims = scale_tril.shape[-1]
        zero = torch.zeros((), dtype=work_dtype, device=mask.device)
        pair = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        # Promote before subtracting so half-precision residuals cannot overflow.
        residual = torch.where(
            mask, obs.values.to(work_dtype) - obs.mapped_mean.to(work_dtype), zero
        )
        h = torch.where(mask.unsqueeze(-1), obs.jacobian.to(work_dtype), zero)
        r = torch.where(pair, obs.measurement_covariance.to(work_dtype), zero)
        not_finite = ~(torch.isfinite(residual).all() & torch.isfinite(h).all())
        not_finite = not_finite | ~torch.isfinite(r).all()
        counts = mask.sum(-1)
        use_information = counts > dims
        batch = _Batch(
            residual=residual,
            hl=h @ scale_tril.to(work_dtype),
            mask=mask,
            maskf=mask.to(work_dtype),
            counts=counts.to(work_dtype),
            use_information=use_information,
            groups=_split_groups(self.method, use_information),
            tolerance=1e-6 if work_dtype == torch.float32 else 1e-12,
        )
        return batch, r, not_finite

    def _evaluate(self, batch: _Batch, r_jittered: Tensor, r_factor: Tensor) -> tuple[Tensor, ...]:
        """Per-object NLL plus inner Cholesky ``info`` codes and their matrices."""
        groups = batch.groups
        if groups.kind == "direct":
            quadratic, logdet, info, matrix = _direct_terms(batch.residual, batch.hl, r_jittered)
            checks: tuple[Tensor, ...] = (info, matrix)
        elif groups.kind == "information":
            quadratic, logdet, info, matrix = _information_terms(batch.residual, batch.hl, r_factor)
            checks = (info, matrix)
        else:
            assert groups.direct is not None and groups.information is not None
            d, i = groups.direct, groups.information
            qd, ld, info_d, matrix_d = _direct_terms(batch.residual[d], batch.hl[d], r_jittered[d])
            qi, li, info_i, matrix_i = _information_terms(
                batch.residual[i], batch.hl[i], r_factor[i]
            )
            zeros = batch.counts.new_zeros(batch.counts.shape)
            quadratic = zeros.index_copy(0, d, qd).index_copy(0, i, qi)
            logdet = zeros.index_copy(0, d, ld).index_copy(0, i, li)
            checks = (info_d, matrix_d, info_i, matrix_i)
        nll = 0.5 * (batch.counts * _LOG_2PI + quadratic + logdet)
        return (nll, *checks)

    def _retry_jitter(
        self, r_sym: Tensor, maskf: Tensor, failed: Tensor
    ) -> tuple[Tensor, Tensor, list[float]]:
        """Raise the diagonal addition per object, level by level, until all factor."""
        amount = self.jitter
        level_amounts = [amount]
        amounts = torch.full_like(maskf[:, 0], amount)
        levels = torch.zeros_like(failed, dtype=torch.long)
        with torch.no_grad():
            base = r_sym.detach()
            diagonal = torch.diag_embed(maskf)
            while bool(failed.any()):
                if self.max_jitter <= amount:
                    raise ValueError(_NOT_PD)
                amount = min(self.max_jitter, max(1e-12, amount * 10))
                level_amounts.append(amount)
                _, info = torch.linalg.cholesky_ex(base + amount * diagonal)
                ok = failed & (info == 0)
                amounts = torch.where(ok, amount, amounts)
                levels = torch.where(ok, len(level_amounts) - 1, levels)
                failed = failed & ~ok
        return amounts, levels, level_amounts

    def _solve(
        self,
        batch: _Batch,
        r: Tensor,
        pending: list[tuple[Tensor, str]],
    ) -> tuple[Tensor, Tensor | None, list[float]]:
        """Factor ``R`` (jittered within bounds) and score every object.

        ``pending`` holds ``(flag, message)`` checks raised, in order, before the
        symmetry and positive-definiteness checks. All checks share one host sync.
        """
        tolerance = batch.tolerance
        asymmetric = ((r - r.mT).abs() > tolerance + tolerance * r.mT.abs()).any()
        # Padded diagonal is one: decoupled coordinates with zero log-determinant.
        r_sym = (r + r.mT) * 0.5 + torch.diag_embed(1 - batch.maskf)
        if self.jitter:
            r_jittered = r_sym + torch.diag_embed(batch.maskf * self.jitter)
        else:
            r_jittered = r_sym
        r_factor, info = torch.linalg.cholesky_ex(r_jittered)
        failed = info != 0
        nll, *checks = self._evaluate(batch, r_jittered, r_factor)
        flags = [*pending, (asymmetric, _NOT_SYMMETRIC)]
        status = torch.stack(
            [flag for flag, _ in flags] + [failed.any()] + [(c != 0).any() for c in checks[::2]]
        ).tolist()
        for (_, message), bad in zip(flags, status, strict=False):
            if bad:
                raise ValueError(message)
        levels: Tensor | None = None
        level_amounts = [self.jitter]
        inner_status = status[len(flags) + 1 :]
        if status[len(flags)]:
            if self.max_jitter <= self.jitter:
                raise ValueError(_NOT_PD)
            amounts, levels, level_amounts = self._retry_jitter(r_sym, batch.maskf, failed)
            r_jittered = r_sym + torch.diag_embed(batch.maskf * amounts.unsqueeze(-1))
            r_factor, info = torch.linalg.cholesky_ex(r_jittered)
            nll, *checks = self._evaluate(batch, r_jittered, r_factor)
            status = torch.stack([(info != 0).any()] + [(c != 0).any() for c in checks[::2]])
            status_list = status.tolist()
            if status_list[0]:
                raise ValueError(_NOT_PD)
            inner_status = status_list[1:]
        for matrix, bad in zip(checks[1::2], inner_status, strict=True):
            if bad:  # Re-run the strict factorization for PyTorch's own error.
                torch.linalg.cholesky(matrix)
                raise ValueError(_NOT_PD)
        return nll, levels, level_amounts

    def _marginal_nll(
        self, mean: Tensor, scale_tril: Tensor, obs: PackedObservations
    ) -> tuple[Tensor, _Batch | None, Tensor | None]:
        """Return per-object NLLs with the padded batch and raw covariance."""
        work_dtype = self._validate_shapes(mean, scale_tril, obs)
        batch_size = mean.shape[0]
        counts = obs.mask.sum(-1).detach()
        diagonal = torch.diagonal(scale_tril, dim1=-2, dim2=-1)
        pending = [
            (
                ~(torch.isfinite(mean).all() & torch.isfinite(scale_tril).all()),
                "Latent parameters must be finite",
            ),
            ((diagonal <= 0).any(), "Latent scale must have a positive diagonal"),
            (
                (torch.triu(scale_tril, diagonal=1) != 0).any(),
                "Latent scale must be lower triangular",
            ),
        ]
        if batch_size == 0 or obs.values.shape[1] == 0:
            for (_, message), bad in zip(
                pending, torch.stack([flag for flag, _ in pending]).tolist(), strict=True
            ):
                if bad:
                    raise ValueError(message)
            self.last_diagnostics = _LazyDiagnostics(
                lambda: {
                    "jitter": [0.0] * batch_size,
                    "method": ["empty"] * batch_size,
                    "observed_counts": counts.tolist(),
                }
            )
            return mean.to(work_dtype).sum(-1) * 0, None, None
        batch, r, not_finite = self._build_batch(scale_tril, obs, work_dtype)
        pending.append((not_finite, _NOT_FINITE))
        nll, levels, level_amounts = self._solve(batch, r, pending)
        empty = counts == 0
        out = torch.where(empty, mean.to(work_dtype).sum(-1) * 0, nll)
        if self.method == "auto":
            code = torch.where(batch.use_information, 2, 1)
        else:
            code = torch.full_like(counts, _METHOD_NAMES.index(self.method))
        code = torch.where(empty, 0, code)

        def diagnostics() -> dict[str, object]:
            codes = code.tolist()
            level_list = [0] * len(codes) if levels is None else levels.tolist()
            return {
                "jitter": [
                    0.0 if c == 0 else level_amounts[lv]
                    for c, lv in zip(codes, level_list, strict=True)
                ],
                "method": [_METHOD_NAMES[c] for c in codes],
                "observed_counts": counts.tolist(),
            }

        self.last_diagnostics = _LazyDiagnostics(diagnostics)
        return out, batch, r

    def forward(
        self,
        mean: Tensor,
        scale_tril: Tensor,
        observations: PackedObservations,
        weights: Tensor | None = None,
    ) -> Tensor:
        per_object = self.per_star(mean, scale_tril, observations)
        active = observations.mask.any(-1)
        if weights is None:
            weights = torch.ones_like(per_object)
        else:
            if weights.shape != per_object.shape or weights.device != per_object.device:
                raise ValueError("weights must match the batch and device")
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("weights must be finite and nonnegative")
            weights = weights.to(per_object.dtype)
        effective = weights * active.to(weights.dtype)
        weighted = per_object * effective
        if self.reduction == "none":
            return weighted
        if self.reduction == "sum":
            return weighted.sum()
        denominator = effective.sum()
        if denominator.item() == 0:
            return mean.sum() * 0
        return weighted.sum() / denominator
