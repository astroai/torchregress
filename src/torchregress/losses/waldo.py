"""WALDO: Wald-statistic confidence regions from conditional mean/covariance.

WALDO (Masserano et al., AISTATS 2023, arXiv:2205.15680) replaces the classical
Wald statistic's MLE and variance with a conditional mean and covariance
estimator:

.. math::

    \\tau(y_0 \\mid x)
    = (\\mathbb E[y\\mid x]-y_0)^T\\,\\mathbb V[y\\mid x]^{-1}
      (\\mathbb E[y\\mid x]-y_0).

The confidence region is the Neyman inversion
``{y0 : tau(y0|x) <= C(y0)}``.  The original method estimates the critical
value function ``C`` by quantile regression on simulated parameters.  This
module implements the same score and a deterministic binned quantile estimate
of ``C(y0)`` (or a user-supplied callable) for regression targets; it is a
practical tabular adaptation, not a reproduction of the SBI simulator
pipeline.

The split-conformal threshold is the special case of a constant ``C``.  With
the binned critical-value function the region can be locally tighter, but the
coverage guarantee then depends on the quality of that regression estimate;
always validate empirical coverage by parameter region.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor

from .conformal import finite_sample_quantile

# Elements per vectorised chunk in the grid/region searches (bounded memory).
_ELEMENT_BUDGET = 1 << 22
# Sample vectors (rows x n_samples) per ``sample_fn`` call / moment chunk.
_SAMPLE_BUDGET = 1 << 20


def waldo_statistic(
    mean: Tensor,
    target: Tensor,
    covariance: Optional[Tensor] = None,
    *,
    covariance_type: str = "diagonal",
    eps: float = 1e-8,
) -> Tensor:
    """Wald statistic ``(mean-target)^T Cov^{-1} (mean-target)`` per row.

    Args:
        mean: Conditional mean ``[N, D]`` or ``[N]``.
        target: Candidate/target values, same shape as ``mean``.
        covariance: Diagonal variances ``[N, D]`` or full matrices ``[N, D, D]``.
            If ``None``, an identity covariance is used.
        covariance_type: ``"diagonal"`` or ``"full"``.
        eps: Variance floor for numerical stability.
    """
    if mean.dim() == 1:
        mean = mean.unsqueeze(-1)
    if target.dim() == 1:
        target = target.unsqueeze(-1)
    if mean.shape != target.shape:
        raise ValueError("mean and target must have the same shape")
    diff = target - mean
    if covariance is None:
        return (diff**2).sum(dim=-1)
    if covariance_type == "diagonal":
        var = covariance.reshape(mean.shape).clamp_min(eps)
        return (diff**2 / var).sum(dim=-1)
    if covariance_type == "full":
        cov = covariance.reshape(mean.shape[0], mean.shape[1], mean.shape[1])
        cov = cov + eps * torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
        solved = torch.linalg.solve(cov, diff.unsqueeze(-1)).squeeze(-1)
        return (diff * solved).sum(dim=-1).clamp_min(0.0)
    raise ValueError("covariance_type must be 'diagonal' or 'full'")


def _interp_critical(left: Tensor, right: Tensor, frac: Tensor) -> Tensor:
    """Linear interpolation between bin critical values.

    A bin with too few points for the requested level has a ``+inf`` finite-sample
    threshold; next to such a bin the critical value is ``+inf`` (the region is
    everything) instead of the ``nan`` that ``inf - inf`` would give.
    """
    inf = torch.isinf(left) | torch.isinf(right)
    safe_left = torch.where(inf, torch.zeros_like(left), left)
    safe_right = torch.where(inf, torch.zeros_like(right), right)
    return torch.where(inf, torch.maximum(left, right), safe_left + frac * (safe_right - safe_left))


class WaldoCriticalValueRegressor:
    """Dependency-free local quantile estimate of a WALDO critical value.

    The original WALDO method learns ``C(theta_0)`` by quantile regression on
    simulator draws.  When no simulator is available, this class fits a
    **local quantile curve** on held-out calibration data: it bins a scalar key
    (the mean of the supplied features, typically the candidate target) and
    estimates the ``1 - alpha`` quantile of the WALDO statistic in each bin,
    with interpolation between bin centers.  Sparse bins fall back to the global
    split-conformal quantile.

    This is a practical simulator-free approximation, not a conditional
    guarantee.  Always validate coverage by region.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        n_bins: int = 10,
        min_samples_per_bin: int = 8,
    ) -> None:
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if int(n_bins) < 2:
            raise ValueError("n_bins must be >= 2")
        if int(min_samples_per_bin) < 1:
            raise ValueError("min_samples_per_bin must be positive")
        self.alpha = float(alpha)
        self.n_bins = int(n_bins)
        self.min_samples_per_bin = int(min_samples_per_bin)
        self.centers_: Optional[Tensor] = None
        self.values_: Optional[Tensor] = None
        self.global_value_: Optional[Tensor] = None
        self.is_fitted = False

    @staticmethod
    def _key(features: Tensor) -> Tensor:
        if features.dim() == 1:
            return features.reshape(-1)
        if features.dim() != 2:
            raise ValueError("features must be [N] or [N, F]")
        return features.mean(dim=-1)

    def fit(self, features: Tensor, scores: Tensor) -> "WaldoCriticalValueRegressor":
        """Fit local quantiles from calibration features and Wald scores."""
        if features.shape[0] != scores.numel():
            raise ValueError("features and scores must agree on the number of rows")
        key = self._key(features).detach()
        scores = scores.detach().reshape(-1)
        if key.numel() < 2 or scores.numel() < 2:
            raise ValueError("at least two calibration rows are required")
        self.global_value_ = finite_sample_quantile(scores, self.alpha)
        n_bins = min(self.n_bins, max(2, key.numel() // self.min_samples_per_bin))
        q = torch.linspace(0.0, 1.0, n_bins + 1, device=key.device, dtype=key.dtype)
        edges = torch.quantile(key, q).unique()
        if edges.numel() < 3:
            self.centers_ = None
            self.values_ = None
            self.is_fitted = True
            return self
        centers, values = [], []
        for left, right in zip(edges[:-1], edges[1:]):
            mask = (key >= left) & (key <= right)
            if int(mask.sum()) < self.min_samples_per_bin:
                continue
            centers.append((left + right) / 2.0)
            values.append(finite_sample_quantile(scores[mask], self.alpha))
        if len(centers) < 2:
            self.centers_ = None
            self.values_ = None
        else:
            self.centers_ = torch.stack(centers)
            self.values_ = torch.stack(values).clamp_min(0.0)
        self.is_fitted = True
        return self

    def predict_critical(self, features: Tensor) -> Tensor:
        """Interpolated critical value for each row of features."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        key = self._key(features)
        if self.centers_ is None or self.values_ is None:
            if self.global_value_ is None:
                raise RuntimeError("regressor has no fitted values")
            return self.global_value_.to(key.device, key.dtype).expand_as(key)
        centers = self.centers_.to(key.device, key.dtype)
        values = self.values_.to(key.device, key.dtype)
        idx = torch.searchsorted(centers, key).clamp(1, centers.numel() - 1)
        left, right = idx - 1, idx
        frac = ((key - centers[left]) / (centers[right] - centers[left]).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )
        return _interp_critical(values[left], values[right], frac)


class WaldoConformal:
    """WALDO-style confidence regions for conditional regression targets.

    Args:
        alpha: Nominal miscoverage; regions target ``1 - alpha`` coverage.
        critical_bins: If >1, estimate a target-dependent critical value
            ``C(y)`` with that many quantile bins (Neyman inversion).  ``None``
            uses one global split-conformal threshold.
        critical_value_fn: Optional callable ``target -> critical value`` used
            instead of the binned estimate (for externally fitted quantile
            regression).  It must accept ``[N, D]`` and return ``[N]``.
        min_variance: Variance floor used by the score.

    Shapes:
        Fitted statistics are tensors; scalar targets are accepted as ``[N]``.
        Multi-target intervals are not manufactured: use
        :meth:`contains`/:meth:`predict_region` for joint Wald ellipsoids.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        critical_bins: Optional[int] = None,
        critical_value_fn: Optional[Callable[[Tensor], Tensor]] = None,
        critical_value_regressor: Optional[WaldoCriticalValueRegressor] = None,
        covariance_type: str = "diagonal",
        min_variance: float = 1e-8,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if critical_bins is not None and int(critical_bins) < 2:
            raise ValueError("critical_bins must be None or >= 2")
        if covariance_type not in ("diagonal", "full"):
            raise ValueError("covariance_type must be 'diagonal' or 'full'")
        self.alpha = float(alpha)
        self.critical_bins = None if critical_bins is None else int(critical_bins)
        if critical_value_fn is not None and critical_value_regressor is not None:
            raise ValueError("pass critical_value_fn or critical_value_regressor, not both")
        self.critical_value_fn = critical_value_fn
        self.critical_value_regressor = critical_value_regressor
        self._critical_fitted = False
        self.covariance_type = covariance_type
        self.min_variance = float(min_variance)
        self.q_hat: Optional[Tensor] = None
        self._bin_centers: Optional[Tensor] = None
        self._bin_values: Optional[Tensor] = None
        self._cal_scores: Optional[Tensor] = None
        self._cal_targets: Optional[Tensor] = None

    @property
    def threshold(self) -> Optional[Tensor]:
        """Global calibrated Wald threshold (``None`` before calibration)."""
        return self.q_hat

    def _critical_value(self, target: Tensor) -> Tensor:
        if self._bin_centers is None or self._bin_values is None:
            if self.q_hat is None:
                raise RuntimeError("Call calibrate() first.")
            return self.q_hat.to(target.device, target.dtype).expand(target.shape[0])
        centers = self._bin_centers.to(target.device, target.dtype)
        values = self._bin_values.to(target.device, target.dtype)
        key = target.reshape(target.shape[0], -1).mean(dim=-1)
        idx = torch.searchsorted(centers, key)
        idx = idx.clamp(1, centers.numel() - 1)
        left, right = idx - 1, idx
        frac = ((key - centers[left]) / (centers[right] - centers[left]).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )
        return _interp_critical(values[left], values[right], frac)

    def calibrate(
        self,
        mean: Tensor,
        target: Tensor,
        covariance: Optional[Tensor] = None,
    ) -> None:
        """Calibrate the Wald critical value(s) on held-out data.

        Args:
            mean: Conditional mean on the calibration split.
            target: Observed targets.
            covariance: Conditional variances/covariances (same shape as mean
                for diagonal, or ``[N, D, D]`` for full).
        """
        target_2d = target if target.dim() > 1 else target.unsqueeze(-1)
        scores = waldo_statistic(
            mean,
            target_2d,
            covariance,
            covariance_type=self.covariance_type,
            eps=self.min_variance,
        ).detach()
        self._fit_scores(scores, target_2d)

    def _fit_scores(self, scores: Tensor, target_2d: Tensor) -> None:
        """Fit the threshold and critical-value curve from calibration scores."""
        self._cal_scores = scores
        self._cal_targets = target_2d.detach()
        self.q_hat = finite_sample_quantile(scores, self.alpha)
        if self.critical_value_regressor is not None:
            self.critical_value_regressor.fit(target_2d.detach(), scores)
            self._critical_fitted = True
        if self.critical_value_fn is not None:
            self._bin_centers = None
            self._bin_values = None
            return
        if self.critical_bins is None:
            self._bin_centers = None
            self._bin_values = None
            return

        # Neyman-inversion estimate: a quantile of tau at each target bin.
        key = target_2d.detach().mean(dim=-1)
        n_bins = min(int(self.critical_bins), max(2, scores.numel() // 10))
        q = torch.linspace(0.0, 1.0, n_bins + 1, device=key.device, dtype=key.dtype)
        edges = torch.quantile(key, q).unique()
        if edges.numel() < 3:
            self._bin_centers = None
            self._bin_values = None
            return
        centers, values = [], []
        for left, right in zip(edges[:-1], edges[1:]):
            mask = (key >= left) & (key <= right)
            if not bool(mask.any()):
                continue
            centers.append((left + right) / 2.0)
            values.append(finite_sample_quantile(scores[mask], self.alpha))
        if len(centers) < 2:
            self._bin_centers = None
            self._bin_values = None
        else:
            self._bin_centers = torch.stack(centers)
            self._bin_values = torch.stack(values)

    def score(
        self,
        mean: Tensor,
        target: Tensor,
        covariance: Optional[Tensor] = None,
    ) -> Tensor:
        """Wald statistic for candidate targets."""
        target_2d = target if target.dim() > 1 else target.unsqueeze(-1)
        return waldo_statistic(
            mean,
            target_2d,
            covariance,
            covariance_type=self.covariance_type,
            eps=self.min_variance,
        )

    def critical_value(self, target: Tensor) -> Tensor:
        """Critical value used by the Neyman inversion for each target."""
        target_2d = target if target.dim() > 1 else target.unsqueeze(-1)
        if self.critical_value_fn is not None:
            return self.critical_value_fn(target_2d).reshape(-1)
        if self.critical_value_regressor is not None:
            return self.critical_value_regressor.predict_critical(target_2d).reshape(-1)
        return self._critical_value(target_2d)

    def contains(
        self,
        mean: Tensor,
        target: Tensor,
        covariance: Optional[Tensor] = None,
    ) -> Tensor:
        """Joint Wald confidence-set membership per row."""
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() first.")
        target_2d = target if target.dim() > 1 else target.unsqueeze(-1)
        return self.score(mean, target_2d, covariance) <= self.critical_value(target_2d)

    def predict_interval(
        self,
        mean: Tensor,
        std: Tensor,
        *,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        grid_size: int = 500,
    ) -> tuple[Tensor, Tensor]:
        """Scalar WALDO intervals by Neyman inversion on a grid.

        ``std`` is the conditional standard deviation (diagonal covariance).
        All rows share one grid over ``[y_min, y_max]`` and are evaluated in
        vectorised row chunks.
        For multi-target inputs this method raises; use :meth:`contains` or
        :meth:`predict_region` for joint regions.
        """
        if mean.dim() > 1 and mean.shape[-1] != 1:
            raise ValueError("predict_interval is scalar-only; use contains() for multi-target")
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() first.")
        m = mean.reshape(-1)
        s = std.reshape(-1).clamp_min(math.sqrt(self.min_variance))
        if y_min is None or y_max is None:
            qmax = float(torch.max(self.critical_value(m.reshape(-1, 1))).item())
            radius = math.sqrt(max(qmax, 1.0)) * 8.0
            y_min = float((m - radius * s).min())
            y_max = float((m + radius * s).max())
        g = int(grid_size)
        grid = torch.linspace(y_min, y_max, g, device=m.device, dtype=m.dtype)
        # The grid is shared by all rows, so the critical values are too.
        crit = self.critical_value(grid.reshape(-1, 1))
        rows = max(1, _ELEMENT_BUDGET // max(g, 1))
        lower_parts, upper_parts = [], []
        for start in range(0, m.numel(), rows):
            m_b = m[start : start + rows].unsqueeze(1)
            s_b = s[start : start + rows].unsqueeze(1)
            accepted = (grid.unsqueeze(0) - m_b) ** 2 / (s_b**2) <= crit.unsqueeze(0)
            has = accepted.any(dim=1)
            first = accepted.int().argmax(dim=1)
            last = g - 1 - accepted.flip(dims=[1]).int().argmax(dim=1)
            # Rows with an empty acceptance set collapse to the mean.
            lower_parts.append(torch.where(has, grid[first], m_b.squeeze(1)))
            upper_parts.append(torch.where(has, grid[last], m_b.squeeze(1)))
        lower = torch.cat(lower_parts) if lower_parts else torch.empty_like(m)
        upper = torch.cat(upper_parts) if upper_parts else torch.empty_like(m)
        return lower.reshape(-1, 1), upper.reshape(-1, 1)

    def predict_region(
        self,
        mean: Tensor,
        target_grid: Tensor,
        covariance: Optional[Tensor] = None,
    ) -> Tensor:
        """Boolean Wald confidence region ``[N, G]`` for a candidate grid.

        Each row of ``target_grid`` is one candidate target; the method computes
        the Wald statistic and compares it to the critical value at that
        candidate.  This is the direct Neyman inversion used for multi-target
        regions where scalar intervals are not meaningful.
        """
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() first.")
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)
        if target_grid.dim() not in (2, 3):
            raise ValueError("target_grid must be [G, D] or [N, G, D]")
        n, d = mean.shape[0], mean.shape[-1]
        g = target_grid.shape[-2]
        per_row = g * d * (d if covariance is not None and self.covariance_type == "full" else 1)
        rows = max(1, _ELEMENT_BUDGET // max(per_row, 1))
        if n <= rows:
            return self._predict_region(mean, target_grid, covariance)
        if target_grid.dim() == 3 and target_grid.shape[0] != n:
            raise ValueError("per-row target_grid must be [N, G, D]")
        if covariance is not None:
            covariance = (
                covariance.reshape(n, d)
                if self.covariance_type == "diagonal"
                else covariance.reshape(n, d, d)
            )
        return torch.cat(
            [
                self._predict_region(
                    mean[i : i + rows],
                    target_grid if target_grid.dim() == 2 else target_grid[i : i + rows],
                    None if covariance is None else covariance[i : i + rows],
                )
                for i in range(0, n, rows)
            ]
        )

    def _predict_region(
        self,
        mean: Tensor,
        target_grid: Tensor,
        covariance: Optional[Tensor],
    ) -> Tensor:
        """Unchunked :meth:`predict_region` for ``[N, D]`` means."""
        if target_grid.dim() == 2:
            if target_grid.shape[-1] != mean.shape[-1]:
                raise ValueError("target_grid last dimension must match mean")
            diff = target_grid.unsqueeze(0) - mean.unsqueeze(1)  # [N, G, D]
            if covariance is None:
                scores = (diff**2).sum(dim=-1)
            elif self.covariance_type == "diagonal":
                var = covariance.reshape(mean.shape).clamp_min(self.min_variance)
                scores = (diff**2 / var.unsqueeze(1)).sum(dim=-1)
            else:
                cov = covariance.reshape(mean.shape[0], -1, mean.shape[-1])
                eye = torch.eye(mean.shape[-1], device=mean.device, dtype=mean.dtype)
                cov = cov + self.min_variance * eye
                solved = torch.linalg.solve(cov.unsqueeze(1), diff.unsqueeze(-1)).squeeze(-1)
                scores = (diff * solved).sum(dim=-1).clamp_min(0.0)
            crit = self.critical_value(target_grid)  # [G]
            return scores <= crit.reshape(1, -1)
        if target_grid.dim() == 3:
            if target_grid.shape[0] != mean.shape[0] or target_grid.shape[-1] != mean.shape[-1]:
                raise ValueError("per-row target_grid must be [N, G, D]")
            n, g, d = target_grid.shape
            diff = (target_grid - mean.unsqueeze(1)).reshape(n * g, d)
            if covariance is None:
                scores = (diff**2).sum(dim=-1)
            elif self.covariance_type == "diagonal":
                var = covariance.reshape(n, d).unsqueeze(1).expand(n, g, d).reshape(n * g, d)
                scores = (diff**2 / var.clamp_min(self.min_variance)).sum(dim=-1)
            else:
                cov = (
                    covariance.reshape(n, d, d).unsqueeze(1).expand(n, g, d, d).reshape(n * g, d, d)
                )
                scores = torch.linalg.solve(
                    cov + self.min_variance * torch.eye(d, device=mean.device, dtype=mean.dtype),
                    diff.unsqueeze(-1),
                ).squeeze(-1)
                scores = (diff * scores).sum(dim=-1).clamp_min(0.0)
            crit = self.critical_value(target_grid.reshape(n * g, d))
            return (scores <= crit).reshape(n, g)
        raise ValueError("target_grid must be [G, D] or [N, G, D]")


def _sample_moments(samples: Tensor) -> tuple[Tensor, Tensor]:
    """Mean and covariance from ``[N, S, D]`` predictive samples."""
    if samples.dim() != 3:
        raise ValueError("samples must be [N, S, D]")
    mean = samples.mean(dim=1)
    centered = samples - mean.unsqueeze(1)
    cov = centered.transpose(1, 2) @ centered / max(samples.shape[1] - 1, 1)
    return mean, cov


def _moments_for(samples: Tensor, covariance_type: str) -> tuple[Tensor, Tensor]:
    """Sample mean and the covariance in the layout ``covariance_type`` expects."""
    mean, cov = _sample_moments(samples)
    if covariance_type == "diagonal":
        cov = cov.diagonal(dim1=-2, dim2=-1)  # per-target sample variances [N, D]
    return mean, cov


def _resolve_chunk_rows(chunk_size: Optional[int], n_samples: int) -> int:
    """Rows per chunk: explicit ``chunk_size`` or the sample-vector budget."""
    if chunk_size is not None:
        if int(chunk_size) < 1:
            raise ValueError(f"chunk_size must be None or a positive integer, got {chunk_size}")
        return int(chunk_size)
    return max(1, _SAMPLE_BUDGET // max(int(n_samples), 1))


def _row_chunks(n: int, rows: int) -> list[tuple[int, int]]:
    return [(i, min(i + rows, n)) for i in range(0, n, rows)] or [(0, 0)]


class WaldoPosteriorCalibrator:
    """WALDO calibration from predictive samples (posterior-estimator mode).

    The conditional mean/covariance are estimated from samples (e.g. a flow or
    ensemble); the Wald statistic and critical values are then the same as in
    :class:`WaldoConformal`.  For multi-target targets, :meth:`contains` and
    :meth:`predict_region` define joint ellipsoidal Wald regions; scalar
    intervals are not manufactured.

    The sample moments and Wald scores are computed in row chunks, so no
    temporary as large as ``samples`` is created; the caller still holds the
    full ``[N, S, D]`` tensor.  To stream the samples themselves, use
    :class:`WaldoDensityConformal` with a sampler callable.

    Args:
        alpha: Miscoverage rate.
        critical_bins: Optional target-binned critical values.
        covariance_type: ``"diagonal"`` or ``"full"``.
        chunk_size: Rows per moment/score chunk.  ``None`` (default) uses
            ``max(1, 2**20 // S)`` rows for ``S`` samples per row.  Results do
            not depend on it.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        critical_bins: Optional[int] = None,
        critical_value_regressor: Optional[WaldoCriticalValueRegressor] = None,
        covariance_type: str = "full",
        chunk_size: Optional[int] = None,
    ) -> None:
        self.calibrator = WaldoConformal(
            alpha=alpha,
            critical_bins=critical_bins,
            critical_value_regressor=critical_value_regressor,
            covariance_type=covariance_type,
        )
        self.alpha = float(alpha)
        self.covariance_type = covariance_type
        if chunk_size is not None:
            _resolve_chunk_rows(chunk_size, 1)  # validate early
        self.chunk_size = chunk_size

    @property
    def threshold(self) -> Optional[Tensor]:
        return self.calibrator.threshold

    def _chunks(self, samples: Tensor) -> list[tuple[int, int]]:
        if samples.dim() != 3:
            raise ValueError("samples must be [N, S, D]")
        rows = _resolve_chunk_rows(self.chunk_size, samples.shape[1])
        return _row_chunks(samples.shape[0], rows)

    def calibrate(self, samples: Tensor, target: Tensor) -> None:
        target_2d = target if target.dim() > 1 else target.unsqueeze(-1)
        scores = []
        for a, b in self._chunks(samples):
            mean, cov = _moments_for(samples[a:b], self.covariance_type)
            scores.append(self.calibrator.score(mean, target_2d[a:b], cov).detach())
        self.calibrator._fit_scores(torch.cat(scores), target_2d.detach())
        self._is_calibrated = True

    def contains(self, samples: Tensor, target: Tensor) -> Tensor:
        parts = []
        for a, b in self._chunks(samples):
            mean, cov = _moments_for(samples[a:b], self.covariance_type)
            parts.append(self.calibrator.contains(mean, target[a:b], cov))
        return torch.cat(parts)

    def predict_region(self, samples: Tensor, target_grid: Tensor) -> Tensor:
        parts = []
        for a, b in self._chunks(samples):
            mean, cov = _moments_for(samples[a:b], self.covariance_type)
            grid = target_grid if target_grid.dim() != 3 else target_grid[a:b]
            parts.append(self.calibrator.predict_region(mean, grid, cov))
        return torch.cat(parts)


class WaldoDensityConformal:
    """WALDO regions for any conditional density sampler, without a simulator.

    The conditional mean/covariance are estimated from samples of the density
    (flow, MDN, ensemble, or bootstrap), so WALDO's prediction/posterior score
    can be applied to a standard conditional-density model.  Calibration uses a
    held-out split of observed targets; this is a split-conformal guarantee
    (marginal), with optional target-binned critical values for a local
    approximation.  No simulator or parameter prior is required.

    ``sample_fn`` is called on consecutive row chunks of ``context`` (in row
    order), so only ``[chunk, n_samples, D]`` samples exist at a time.  When the
    split has at most ``chunk`` rows this is the single call of earlier
    releases and the result is bit-identical for a fixed seed.  With several
    chunks the per-row Monte Carlo moments are drawn from the same
    distribution, but a sampler that draws its noise as one ``[S, N, D]`` block
    (e.g. ``flow(c).sample((S,))``) consumes the generator in a different order,
    so individual draws, and hence thresholds, differ by Monte Carlo error
    (the score distribution is unchanged).  Deterministic samplers, or samplers
    whose draws for a row depend only on that row (e.g. per-row seeding), give
    identical results for any ``chunk_size``.

    Args:
        alpha: Miscoverage rate.
        critical_bins: Optional target bins for a local critical-value curve.
        covariance_type: ``"diagonal"`` or ``"full"``.
        n_samples: Number of density samples used per context at calibration
            and prediction time.
        chunk_size: Context rows per ``sample_fn`` call.  ``None`` (default)
            uses ``max(1, 2**20 // n_samples)`` rows (4096 for 256 samples).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        critical_bins: Optional[int] = None,
        critical_value_regressor: Optional[WaldoCriticalValueRegressor] = None,
        covariance_type: str = "full",
        n_samples: int = 256,
        chunk_size: Optional[int] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.n_samples = int(n_samples)
        self.covariance_type = covariance_type
        self.calibrator = WaldoConformal(
            alpha=alpha,
            critical_bins=critical_bins,
            critical_value_regressor=critical_value_regressor,
            covariance_type=covariance_type,
        )
        self._chunk_rows = _resolve_chunk_rows(chunk_size, self.n_samples)
        self.chunk_size = chunk_size

    @property
    def threshold(self) -> Optional[Tensor]:
        return self.calibrator.threshold

    def calibrate(
        self,
        sample_fn: Callable[[Tensor, int], Tensor],
        context: Tensor,
        target: Tensor,
    ) -> None:
        """Calibrate on held-out contexts and observed targets."""
        target_2d = target if target.dim() > 1 else target.unsqueeze(-1)
        scores = []
        for a, b in _row_chunks(context.shape[0], self._chunk_rows):
            mean, cov = _moments_for(sample_fn(context[a:b], self.n_samples), self.covariance_type)
            scores.append(self.calibrator.score(mean, target_2d[a:b], cov).detach())
        self.calibrator._fit_scores(torch.cat(scores), target_2d.detach())

    def contains(
        self,
        sample_fn: Callable[[Tensor, int], Tensor],
        context: Tensor,
        target: Tensor,
    ) -> Tensor:
        parts = []
        for a, b in _row_chunks(context.shape[0], self._chunk_rows):
            mean, cov = _moments_for(sample_fn(context[a:b], self.n_samples), self.covariance_type)
            parts.append(self.calibrator.contains(mean, target[a:b], cov))
        return torch.cat(parts)

    def predict_region(
        self,
        sample_fn: Callable[[Tensor, int], Tensor],
        context: Tensor,
        target_grid: Tensor,
    ) -> Tensor:
        parts = []
        for a, b in _row_chunks(context.shape[0], self._chunk_rows):
            mean, cov = _moments_for(sample_fn(context[a:b], self.n_samples), self.covariance_type)
            grid = target_grid if target_grid.dim() != 3 else target_grid[a:b]
            parts.append(self.calibrator.predict_region(mean, grid, cov))
        return torch.cat(parts)

    def predict_interval(
        self,
        sample_fn: Callable[[Tensor, int], Tensor],
        context: Tensor,
        *,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        grid_size: int = 500,
    ) -> tuple[Tensor, Tensor]:
        means, stds = [], []
        for a, b in _row_chunks(context.shape[0], self._chunk_rows):
            samples = sample_fn(context[a:b], self.n_samples)
            means.append(samples.mean(dim=1))
            stds.append(samples.std(dim=1))
        return self.calibrator.predict_interval(
            torch.cat(means),
            torch.cat(stds),
            y_min=y_min,
            y_max=y_max,
            grid_size=grid_size,
        )
