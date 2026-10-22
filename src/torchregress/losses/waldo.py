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
        self.global_value_ = torch.quantile(scores, 1.0 - self.alpha)
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
            values.append(torch.quantile(scores[mask], 1.0 - self.alpha))
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
        return values[left] + frac * (values[right] - values[left])


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
        return values[left] + frac * (values[right] - values[left])

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
        self._cal_scores = scores
        self._cal_targets = target_2d.detach()
        self.q_hat = torch.quantile(scores, 1.0 - self.alpha)
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
            values.append(torch.quantile(scores[mask], 1.0 - self.alpha))
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
        lower = torch.empty_like(m)
        upper = torch.empty_like(m)
        for i in range(m.numel()):
            grid = torch.linspace(y_min, y_max, int(grid_size), device=m.device, dtype=m.dtype)
            score = (grid - m[i]) ** 2 / (s[i] ** 2)
            crit = self.critical_value(grid.reshape(-1, 1))
            accepted = score <= crit
            if bool(accepted.any()):
                idx = accepted.nonzero(as_tuple=True)[0]
                lower[i] = grid[idx[0]]
                upper[i] = grid[idx[-1]]
            else:
                lower[i] = m[i]
                upper[i] = m[i]
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


class WaldoPosteriorCalibrator:
    """WALDO calibration from predictive samples (posterior-estimator mode).

    The conditional mean/covariance are estimated from samples (e.g. a flow or
    ensemble); the Wald statistic and critical values are then the same as in
    :class:`WaldoConformal`.  For multi-target targets, :meth:`contains` and
    :meth:`predict_region` define joint ellipsoidal Wald regions; scalar
    intervals are not manufactured.

    Args:
        alpha: Miscoverage rate.
        critical_bins: Optional target-binned critical values.
        covariance_type: ``"diagonal"`` or ``"full"``.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        critical_bins: Optional[int] = None,
        critical_value_regressor: Optional[WaldoCriticalValueRegressor] = None,
        covariance_type: str = "full",
    ) -> None:
        self.calibrator = WaldoConformal(
            alpha=alpha,
            critical_bins=critical_bins,
            critical_value_regressor=critical_value_regressor,
            covariance_type=covariance_type,
        )
        self.alpha = float(alpha)
        self.covariance_type = covariance_type

    @property
    def threshold(self) -> Optional[Tensor]:
        return self.calibrator.threshold

    def calibrate(self, samples: Tensor, target: Tensor) -> None:
        mean, cov = _sample_moments(samples)
        self.calibrator.calibrate(mean, target, cov)
        self._is_calibrated = True

    def contains(self, samples: Tensor, target: Tensor) -> Tensor:
        mean, cov = _sample_moments(samples)
        return self.calibrator.contains(mean, target, cov)

    def predict_region(self, samples: Tensor, target_grid: Tensor) -> Tensor:
        mean, cov = _sample_moments(samples)
        return self.calibrator.predict_region(mean, target_grid, cov)


class WaldoDensityConformal:
    """WALDO regions for any conditional density sampler, without a simulator.

    The conditional mean/covariance are estimated from samples of the density
    (flow, MDN, ensemble, or bootstrap), so WALDO's prediction/posterior score
    can be applied to a standard conditional-density model.  Calibration uses a
    held-out split of observed targets; this is a split-conformal guarantee
    (marginal), with optional target-binned critical values for a local
    approximation.  No simulator or parameter prior is required.

    Args:
        alpha: Miscoverage rate.
        critical_bins: Optional target bins for a local critical-value curve.
        covariance_type: ``"diagonal"`` or ``"full"``.
        n_samples: Number of density samples used per context at calibration
            and prediction time.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        *,
        critical_bins: Optional[int] = None,
        critical_value_regressor: Optional[WaldoCriticalValueRegressor] = None,
        covariance_type: str = "full",
        n_samples: int = 256,
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

    @property
    def threshold(self) -> Optional[Tensor]:
        return self.calibrator.threshold

    def _moments(
        self, sample_fn: Callable[[Tensor, int], Tensor], context: Tensor
    ) -> tuple[Tensor, Tensor]:
        samples = sample_fn(context, self.n_samples)
        return _sample_moments(samples)

    def calibrate(
        self,
        sample_fn: Callable[[Tensor, int], Tensor],
        context: Tensor,
        target: Tensor,
    ) -> None:
        """Calibrate on held-out contexts and observed targets."""
        mean, cov = self._moments(sample_fn, context)
        self.calibrator.calibrate(mean, target, cov)

    def contains(
        self,
        sample_fn: Callable[[Tensor, int], Tensor],
        context: Tensor,
        target: Tensor,
    ) -> Tensor:
        mean, cov = self._moments(sample_fn, context)
        return self.calibrator.contains(mean, target, cov)

    def predict_region(
        self,
        sample_fn: Callable[[Tensor, int], Tensor],
        context: Tensor,
        target_grid: Tensor,
    ) -> Tensor:
        mean, cov = self._moments(sample_fn, context)
        return self.calibrator.predict_region(mean, target_grid, cov)

    def predict_interval(
        self,
        sample_fn: Callable[[Tensor, int], Tensor],
        context: Tensor,
        *,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        grid_size: int = 500,
    ) -> tuple[Tensor, Tensor]:
        samples = sample_fn(context, self.n_samples)
        mean = samples.mean(dim=1)
        std = samples.std(dim=1)
        return self.calibrator.predict_interval(
            mean,
            std,
            y_min=y_min,
            y_max=y_max,
            grid_size=grid_size,
        )
