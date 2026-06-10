"""
Conformal prediction for regression.

Provides standalone conformal predictors (calibration + prediction) and
backward-compatible loss wrappers. All methods provide finite-sample
marginal coverage guarantees under exchangeability.

Predictors
----------
- SplitConformal: absolute residual scores, ŷ ± q_hat
- CQR: conformalized quantile regression with optional debiasing
- UACQR: CQR with scores normalized by predicted quantile band width (thin wrapper)
- CTI: conformal thresholded intervals (density-level sets, smallest intervals)
- DistributionalConformal: PIT-based approximate conditional coverage
- R2CConformal: regression-as-classification for multimodal targets
- MultiTargetConformal: per-dimension calibration for multi-output
- DensityConformal: density-adaptive split conformal for long-tail targets
- PrevalenceAdjustedCP: group-prevalence adjusted split conformal
- MonteCarloConformal: MC-sample conformal with uncertainty-normalized scores

Composable features (all predictors):
- Normalized scores (difficulty-adaptive intervals)
- Mondrian groups (group-conditional calibration)
- Weighted scores (covariate-shift robustness)

Loss wrappers (backward-compatible):
- ConformalLoss: training loss + calibration + prediction
- MultiDimensionalConformalLoss: multi-output variant
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from .base import RegressionLoss
from .loss_registry import register_regression_loss

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _weighted_quantile(
    scores: Tensor,
    q: float,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Compute (weighted) quantile of 1-D scores.

    For unweighted case, uses the standard ceil((n+1)*(1-alpha))/n finite-
    sample correction.  For weighted case, uses the weighted empirical CDF.

    Args:
        scores: 1-D tensor of nonconformity scores.
        q: Quantile level in [0, 1] (e.g. 1 - alpha).
        weights: Optional 1-D importance weights (need not sum to 1).

    Returns:
        Scalar tensor with the quantile value.
    """
    if weights is None:
        n = scores.numel()
        if n == 0:
            raise ValueError("Input scores tensor is empty.")
        q_adj = min(math.ceil((n + 1) * q) / n, 1.0)
        return torch.quantile(scores, q_adj, interpolation="higher")

    # Weighted quantile via sorted CDF
    sorted_idx = scores.argsort()
    sorted_scores = scores[sorted_idx]
    sorted_weights = weights[sorted_idx]
    if sorted_weights.numel() == 0:
        raise ValueError("Input weights tensor is empty.")
    if torch.any(sorted_weights < 0):
        raise ValueError("Sample weights must be non-negative.")
    total_weight = sorted_weights.sum()
    if total_weight <= 0:
        raise ValueError("Sum of sample weights must be positive.")
    cum_weights = torch.cumsum(sorted_weights, dim=0)
    cum_weights = cum_weights / total_weight  # normalize to [0, 1]
    # First index where cumulative weight >= q
    idx = torch.searchsorted(cum_weights, q)
    idx = idx.clamp(max=len(sorted_scores) - 1)
    return sorted_scores[idx]


# ---------------------------------------------------------------------------
# Base Conformal Predictor
# ---------------------------------------------------------------------------


class ConformalPredictor:
    """Base class for conformal prediction methods.

    Handles score normalization, Mondrian grouping, and weighted quantiles
    as composable features.  Subclasses implement ``_compute_scores`` and
    ``_build_intervals``.

    Args:
        alpha: Desired miscoverage rate (e.g. 0.1 for 90% coverage).
        normalize_fn: Optional callable ``(y_pred, x) -> difficulty`` that
            returns per-sample difficulty estimates.  Scores are divided by
            difficulty to produce adaptive intervals.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        normalize_fn: Optional[Callable[..., Tensor]] = None,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.normalize_fn = normalize_fn
        self._is_calibrated = False
        # Scalar quantile (no groups) or dict group_key -> quantile
        self.q_hat: Optional[Union[Tensor, Dict[Any, Tensor]]] = None
        self._groups_list: Optional[List[Any]] = None

    # -- Subclass hooks ------------------------------------------------------

    def _compute_scores(
        self,
        y_pred: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Return 1-D tensor of nonconformity scores.  Override in subclass."""
        raise NotImplementedError

    def _build_intervals(
        self,
        y_pred: Tensor,
        q: Tensor,
        difficulty: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return (lower, upper) from predictions and calibrated quantile."""
        raise NotImplementedError

    # -- Public API ----------------------------------------------------------

    def calibrate(
        self,
        y_pred: Tensor,
        target: Tensor,
        *,
        mask: Optional[Tensor] = None,
        groups: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> None:
        """Calibrate on held-out data.

        Args:
            y_pred: Predictions on calibration set.
            target: True values.
            mask: Optional boolean mask for valid samples.
            groups: Optional 1-D integer/categorical tensor for Mondrian CP.
                Separate quantiles are computed per unique group value.
            weights: Optional 1-D importance weights for weighted CP
                (covariate shift robustness).
            x: Optional input features, passed to ``normalize_fn``.
        """
        # Apply mask
        if mask is not None:
            mask_1d = mask.all(dim=-1) if mask.dim() > 1 else mask
            y_pred = y_pred[mask_1d]
            target = target[mask_1d]
            if groups is not None:
                groups = groups[mask_1d]
            if weights is not None:
                weights = weights[mask_1d]
            if x is not None:
                x = x[mask_1d]

        if y_pred.shape[0] == 0:
            raise ValueError(
                "Calibration set is empty (after masking, if applicable). "
                "Cannot calibrate with zero samples."
            )

        scores = self._compute_scores(y_pred, target)

        # Normalize by difficulty
        if self.normalize_fn is not None and x is not None:
            difficulty = self.normalize_fn(y_pred, x)
            scores = scores / difficulty.clamp(min=1e-8)

        q_level = 1.0 - self.alpha

        if groups is not None:
            # Mondrian: per-group quantiles
            unique_groups = groups.unique().tolist()
            self.q_hat = {}
            self._groups_list = unique_groups
            for g in unique_groups:
                g_mask = groups == g
                g_scores = scores[g_mask]
                g_weights = weights[g_mask] if weights is not None else None
                self.q_hat[g] = _weighted_quantile(g_scores, q_level, g_weights)
        else:
            self.q_hat = _weighted_quantile(scores, q_level, weights)
            self._groups_list = None

        self._is_calibrated = True

    def predict_interval(
        self,
        y_pred: Tensor,
        *,
        groups: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Predict intervals using calibrated quantile(s).

        Args:
            y_pred: Model predictions.
            groups: Group labels (must match calibration groups for Mondrian).
            x: Input features for normalized CP (passed to normalize_fn).

        Returns:
            Tuple of (lower_bound, upper_bound) tensors.
        """
        if not self._is_calibrated or self.q_hat is None:
            raise RuntimeError(
                "Predictor must be calibrated before making predictions. Call calibrate() first."
            )

        difficulty = None
        if self.normalize_fn is not None and x is not None:
            difficulty = self.normalize_fn(y_pred, x)

        if isinstance(self.q_hat, dict):
            # Mondrian: build per-sample q from group assignment
            if groups is None:
                raise ValueError("groups must be provided at prediction time for Mondrian CP")
            q_per_sample = torch.empty(y_pred.shape[0], device=y_pred.device, dtype=y_pred.dtype)
            # Vectorized group lookup: build a lookup tensor if groups
            # are integer-like, otherwise fall back to scatter
            group_keys = list(self.q_hat.keys())
            group_vals = torch.stack([self.q_hat[g].to(y_pred.device) for g in group_keys])
            if all(isinstance(g, int) for g in group_keys):
                max_g = max(group_keys)
                lut = torch.zeros(max_g + 1, device=y_pred.device, dtype=y_pred.dtype)
                gk = torch.tensor(group_keys, device=y_pred.device)
                lut[gk] = group_vals
                q_per_sample = lut[groups.long()]
            else:
                # Optimized vectorized lookup for non-integer keys (e.g., floats)
                # Fall back to loop if keys are incompatible with tensor operations
                try:
                    # Convert keys and values to tensors
                    keys = list(self.q_hat.keys())
                    keys_tensor = torch.tensor(keys, device=y_pred.device)
                    vals_tensor = torch.stack([self.q_hat[k].to(y_pred.device) for k in keys])

                    # Sort for searchsorted
                    sorted_idx = torch.argsort(keys_tensor)
                    sorted_keys = keys_tensor[sorted_idx]
                    sorted_vals = vals_tensor[sorted_idx]

                    # Find indices of groups in sorted keys
                    # Ensure groups has same dtype for comparison
                    groups_cast = groups.to(keys_tensor.dtype)
                    idx = torch.searchsorted(sorted_keys, groups_cast)
                    idx = idx.clamp(max=len(sorted_keys) - 1)

                    # Only update valid matches (mimic original behavior of skipping unknown groups)
                    matches = sorted_keys[idx] == groups_cast

                    if matches.all():
                        q_per_sample = sorted_vals[idx]
                    else:
                        q_per_sample[matches] = sorted_vals[idx[matches]]

                except (TypeError, RuntimeError, ValueError):
                    for g, q_val in self.q_hat.items():
                        g_mask = groups == g
                        q_per_sample[g_mask] = q_val.to(y_pred.device)
            # Reshape for broadcasting
            q = q_per_sample.view(-1, *([1] * (y_pred.dim() - 1)))
        else:
            q = self.q_hat.to(y_pred.device)

        return self._build_intervals(y_pred, q, difficulty)


# ---------------------------------------------------------------------------
# Split Conformal Prediction
# ---------------------------------------------------------------------------


class SplitConformal(ConformalPredictor):
    """Split conformal prediction using absolute residual scores.

    Score: |y - ŷ| (optionally divided by difficulty).
    Interval: ŷ ± q_hat (or ŷ ± q_hat * difficulty for normalized CP).

    Example:
        >>> cp = SplitConformal(alpha=0.1)
        >>> cp.calibrate(cal_preds, cal_targets)
        >>> lower, upper = cp.predict_interval(test_preds)
    """

    def _compute_scores(self, y_pred: Tensor, target: Tensor) -> Tensor:
        scores = torch.abs(y_pred - target)
        if scores.dim() > 1:
            scores = scores.max(dim=-1).values
        return scores

    def _build_intervals(
        self,
        y_pred: Tensor,
        q: Tensor,
        difficulty: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if difficulty is not None:
            width = q * difficulty.view(-1, *([1] * (y_pred.dim() - 1)))
        else:
            width = q
        return y_pred - width, y_pred + width


# ---------------------------------------------------------------------------
# Conformalized Quantile Regression (CQR)
# ---------------------------------------------------------------------------


class CQR(ConformalPredictor):
    """Conformalized Quantile Regression.

    Expects predictions as ``[lower_quantile, upper_quantile]`` concatenated
    along the last dimension.

    Score: max(q_lo - y, y - q_hi).
    Interval: [q_lo - q_hat, q_hi + q_hat].

    Args:
        alpha: Miscoverage rate.
        debias: If True, apply coverage bias correction (Gibbs, Cherian &
            Candès, 2025).  Adjusts the quantile level to account for
            finite-sample overfitting of the quantile regression model.
        normalize_fn: Optional difficulty normalization.

    Example:
        >>> cqr = CQR(alpha=0.1, debias=True)
        >>> cqr.calibrate(cal_quantile_preds, cal_targets)
        >>> lower, upper = cqr.predict_interval(test_quantile_preds)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        debias: bool = False,
        normalize_fn: Optional[Callable[..., Tensor]] = None,
    ) -> None:
        super().__init__(alpha=alpha, normalize_fn=normalize_fn)
        self.debias = debias

    def _compute_scores(self, y_pred: Tensor, target: Tensor) -> Tensor:
        # A 1-D target must gain a trailing feature axis before the
        # arithmetic below: [N] - [N, 1] would broadcast to [N, N] and
        # silently produce garbage scores (max over all targets per row).
        if target.dim() == 1 and y_pred.dim() > 1:
            target = target.unsqueeze(-1)
        n_feat = target.shape[-1] if target.dim() > 1 else 1
        lower_pred = y_pred[..., :n_feat]
        upper_pred = y_pred[..., n_feat:]
        scores = torch.maximum(target - upper_pred, lower_pred - target)
        if scores.dim() > 1:
            scores = scores.max(dim=-1).values
        return scores

    def calibrate(
        self,
        y_pred: Tensor,
        target: Tensor,
        *,
        mask: Optional[Tensor] = None,
        groups: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> None:
        """Calibrate CQR, optionally with coverage bias correction."""
        if self.debias:
            # Coverage bias correction: adjust alpha to account for QR
            # overfitting.  Uses the heuristic from Gibbs et al. (2025):
            # inflate alpha by the ratio p/n where p is the effective
            # dimensionality.  For neural networks, we use a simple LOO-style
            # correction: alpha_adj = alpha * n / (n - 1).
            if mask is not None:
                mask_1d = mask.all(dim=-1) if mask.dim() > 1 else mask
                n = mask_1d.sum().item()
            else:
                n = y_pred.shape[0]
            # Finite-sample correction: slightly tighter quantile
            alpha_orig = self.alpha
            self.alpha = self.alpha * n / (n + 1)
            super().calibrate(y_pred, target, mask=mask, groups=groups, weights=weights, x=x)
            self.alpha = alpha_orig
        else:
            super().calibrate(y_pred, target, mask=mask, groups=groups, weights=weights, x=x)

    def _build_intervals(
        self,
        y_pred: Tensor,
        q: Tensor,
        difficulty: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        n_feat = y_pred.shape[-1] // 2
        lower_pred = y_pred[..., :n_feat]
        upper_pred = y_pred[..., n_feat:]
        if difficulty is not None:
            width = q * difficulty.view(-1, *([1] * (lower_pred.dim() - 1)))
        else:
            width = q
        return lower_pred - width, upper_pred + width


class UACQR(CQR):
    """Uncertainty-aware CQR: normalize nonconformity scores by predicted band width.

    This is a thin wrapper over :class:`CQR` that sets ``normalize_fn`` to divide
    scores by the predicted quantile interval width :math:`(q_{\\mathrm{hi}} -
    q_{\\mathrm{lo}})` (clamped).  Wider model-predicted bands therefore receive
    proportionally wider conformal corrections—an uncertainty-aware adaptive
    variant of CQR without changing the score definition.

    Training and calibration use the same pinball / CQR conventions as
    :class:`CQR`.  If ``x`` is omitted on :meth:`calibrate` or
    :meth:`predict_interval`, a dummy tensor is supplied so normalization runs
    from ``y_pred`` alone (the base class only applies ``normalize_fn`` when
    ``x`` is not ``None``).

    Args:
        alpha: Miscoverage rate.
        debias: Same as :class:`CQR`.
        min_width: Floor on :math:`q_{\\mathrm{hi}} - q_{\\mathrm{lo}}` for division.
        aggregation: ``\"mean\"`` or ``\"max\"`` across output dimensions for multi-target.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        debias: bool = False,
        *,
        min_width: float = 1e-6,
        aggregation: str = "mean",
    ) -> None:
        self._uacqr_min_width = float(min_width)
        if aggregation not in ("mean", "max"):
            raise ValueError('aggregation must be "mean" or "max".')
        self._uacqr_aggregation = aggregation
        super().__init__(alpha=alpha, debias=debias, normalize_fn=self._uacqr_difficulty)

    def _uacqr_difficulty(self, y_pred: Tensor, x: Tensor) -> Tensor:
        del x
        n_feat = y_pred.shape[-1] // 2
        lower_pred = y_pred[..., :n_feat]
        upper_pred = y_pred[..., n_feat:]
        width = (upper_pred - lower_pred).clamp(min=self._uacqr_min_width)
        if self._uacqr_aggregation == "max":
            return width.max(dim=-1).values
        return width.mean(dim=-1)

    def calibrate(
        self,
        y_pred: Tensor,
        target: Tensor,
        *,
        mask: Optional[Tensor] = None,
        groups: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> None:
        if x is None:
            x = torch.zeros((y_pred.shape[0], 1), device=y_pred.device, dtype=y_pred.dtype)
        super().calibrate(y_pred, target, mask=mask, groups=groups, weights=weights, x=x)

    def predict_interval(
        self,
        y_pred: Tensor,
        *,
        groups: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if x is None:
            x = torch.zeros((y_pred.shape[0], 1), device=y_pred.device, dtype=y_pred.dtype)
        return super().predict_interval(y_pred, groups=groups, x=x)


# ---------------------------------------------------------------------------
# Conformal Thresholded Intervals (CTI)
# ---------------------------------------------------------------------------


class CTI(ConformalPredictor):
    """Conformal Thresholded Intervals — smallest prediction sets.

    Uses negative log-density as the nonconformity score.  The prediction
    set is the density level set ``{y : -log p(y|x) <= q_hat}``, which
    produces the smallest possible intervals (and naturally handles
    multimodal, skewed distributions).

    The caller must provide log-density values at calibration and a
    ``density_fn`` at prediction time.

    Args:
        alpha: Miscoverage rate.
        grid_size: Number of grid points for constructing level-set
            intervals at prediction time (default: 500).

    Example:
        >>> cti = CTI(alpha=0.1, grid_size=1000)
        >>> # log_density_cal[i] = log p(y_cal[i] | x_cal[i])
        >>> cti.calibrate(log_density_cal, y_cal)
        >>> # density_fn: (y_grid, x) -> log p(y_grid | x)
        >>> intervals = cti.predict_intervals_from_density(
        ...     density_fn, x_test, y_min=-5, y_max=5)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        grid_size: int = 500,
    ) -> None:
        super().__init__(alpha=alpha)
        self.grid_size = grid_size

    def _compute_scores(self, y_pred: Tensor, target: Tensor) -> Tensor:
        # y_pred here is log p(y | x) values from calibration
        return -y_pred.view(-1)  # negative log-density as score

    def calibrate(
        self,
        log_density_cal: Tensor,
        target: Tensor,
        *,
        mask: Optional[Tensor] = None,
        groups: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> None:
        """Calibrate CTI.

        Args:
            log_density_cal: Log-density values log p(y_cal | x_cal),
                shape ``(n_cal,)`` or ``(n_cal, 1)``.
            target: Calibration targets (used only for masking).
            mask: Optional mask.
            groups: Optional Mondrian groups.
            weights: Optional importance weights.
            x: Unused (density values already computed).
        """
        # Bypass normalize_fn — CTI uses density directly
        norm_fn_backup = self.normalize_fn
        self.normalize_fn = None
        super().calibrate(log_density_cal, target, mask=mask, groups=groups, weights=weights)
        self.normalize_fn = norm_fn_backup

    def _build_intervals(
        self,
        y_pred: Tensor,
        q: Tensor,
        difficulty: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # CTI intervals are built via predict_intervals_from_density instead
        raise NotImplementedError(
            "CTI intervals require a density function. "
            "Use predict_intervals_from_density() instead."
        )

    def predict_intervals_from_density(
        self,
        density_fn: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
        y_min: float,
        y_max: float,
    ) -> Tuple[Tensor, Tensor]:
        """Build prediction intervals from density level sets.

        For each test point, evaluates the density on a grid and returns
        the tightest [lower, upper] bounding box of the level set.

        Args:
            density_fn: ``(y_grid, x) -> log_density``.  ``y_grid`` has
                shape ``(grid_size,)`` and ``x`` has shape ``(n_features,)``.
                Returns ``(grid_size,)`` log-densities.
            x: Test inputs, shape ``(n_test, n_features)``.
            y_min: Lower bound of the evaluation grid.
            y_max: Upper bound of the evaluation grid.

        Returns:
            (lower, upper) tensors of shape ``(n_test, 1)``.
        """
        if not self._is_calibrated or self.q_hat is None:
            raise RuntimeError("Call calibrate() first.")

        q = self.q_hat if isinstance(self.q_hat, Tensor) else next(iter(self.q_hat.values()))
        y_grid = torch.linspace(y_min, y_max, self.grid_size, device=x.device, dtype=x.dtype)

        n_test = x.shape[0]
        lower = torch.full((n_test, 1), y_max, device=x.device, dtype=x.dtype)
        upper = torch.full((n_test, 1), y_min, device=x.device, dtype=x.dtype)

        # Try vectorized execution first
        try:
            log_dens_batch = density_fn(y_grid, x)  # (n_test, grid_size)
            if log_dens_batch.shape == (n_test, self.grid_size):
                neg_log_dens = -log_dens_batch
                in_set = neg_log_dens <= q  # (n_test, grid_size)

                # For valid rows (where in_set has any True), find first and last index
                has_valid = in_set.any(dim=1)

                # Vectorized min/max index lookup using argmax
                # First True from left gives start index
                start_indices = in_set.int().argmax(dim=1)

                # First True from right (flipped) gives end index
                end_indices = self.grid_size - 1 - in_set.flip(dims=[1]).int().argmax(dim=1)

                # Fill valid entries
                valid_mask = has_valid
                if valid_mask.any():
                    lower[valid_mask, 0] = y_grid[start_indices[valid_mask]]
                    upper[valid_mask, 0] = y_grid[end_indices[valid_mask]]

                # Fill invalid entries (fallback to mode)
                invalid_mask = ~has_valid
                if invalid_mask.any():
                    mode_indices = log_dens_batch[invalid_mask].argmax(dim=1)
                    lower[invalid_mask, 0] = y_grid[mode_indices]
                    upper[invalid_mask, 0] = y_grid[mode_indices]

                return lower, upper
        except Exception as e:
            logger.debug(f"CTI vectorized execution failed, falling back to loop: {e}")

        for i in range(n_test):
            log_dens = density_fn(y_grid, x[i])  # (grid_size,)
            neg_log_dens = -log_dens
            in_set = neg_log_dens <= q  # density level set
            if in_set.any():
                indices = in_set.nonzero(as_tuple=True)[0]
                lower[i, 0] = y_grid[indices[0]]
                upper[i, 0] = y_grid[indices[-1]]
            else:
                # Fallback: use the mode
                mode_idx = log_dens.argmax()
                lower[i, 0] = y_grid[mode_idx]
                upper[i, 0] = y_grid[mode_idx]

        return lower, upper


# ---------------------------------------------------------------------------
# Distributional Conformal Prediction
# ---------------------------------------------------------------------------


class DistributionalConformal(ConformalPredictor):
    """Distributional conformal prediction via probability integral transform.

    Achieves approximate conditional coverage by using PIT residuals as
    nonconformity scores.  Requires a model that outputs CDF values
    ``F(y | x)`` for each calibration/test point.

    Score: |F(y|x) - tau| for a random tau, or simply |2*F(y|x) - 1|
        (a commonly used deterministic variant).
    Interval: invert the CDF at ``[alpha/2, 1-alpha/2]`` after conformal
        adjustment.

    Args:
        alpha: Miscoverage rate.

    Example:
        >>> dcp = DistributionalConformal(alpha=0.1)
        >>> # F_cal[i] = CDF(y_cal[i] | x_cal[i]) from model
        >>> dcp.calibrate(F_cal, y_cal)
        >>> lower, upper = dcp.predict_intervals_from_cdf(
        ...     cdf_fn, icdf_fn, x_test)
    """

    def _compute_scores(self, y_pred: Tensor, target: Tensor) -> Tensor:
        # y_pred here = F(y | x) = CDF values, shape (n,) or (n, 1)
        pit = y_pred.view(-1)
        # Symmetric PIT score: how far from uniform
        return torch.abs(2 * pit - 1)

    def _build_intervals(
        self,
        y_pred: Tensor,
        q: Tensor,
        difficulty: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "Distributional CP intervals require CDF/ICDF functions. "
            "Use predict_intervals_from_cdf() instead."
        )

    def predict_intervals_from_cdf(
        self,
        icdf_fn: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Build prediction intervals by inverting the calibrated CDF.

        Args:
            icdf_fn: Inverse CDF function ``(quantile_levels, x) -> y``.
                Can accept either a single input ``x`` (shape ``(features,)``)
                returning ``(2,)``, or a batch of inputs ``x`` (shape
                ``(batch, features)``) returning ``(batch, 2)``.
                ``quantile_levels`` has shape ``(2,)`` containing
                ``[alpha_low, alpha_high]``.
            x: Test inputs, shape ``(n_test, n_features)``.

        Returns:
            (lower, upper) tensors of shape ``(n_test, 1)``.
        """
        if not self._is_calibrated or self.q_hat is None:
            raise RuntimeError("Call calibrate() first.")

        q = self.q_hat if isinstance(self.q_hat, Tensor) else next(iter(self.q_hat.values()))

        # Conformal adjustment: widen the PIT quantile levels
        # q_hat is the quantile of |2*F(y)-1|, so the adjusted band is
        # [(1-q_hat)/2, (1+q_hat)/2], clipped to [0, 1]
        q_val = q.item()
        alpha_low = max((1 - q_val) / 2, 0.0)
        alpha_high = min((1 + q_val) / 2, 1.0)
        levels = torch.tensor([alpha_low, alpha_high], device=x.device, dtype=x.dtype)

        n_test = x.shape[0]

        # Try vectorized execution first
        try:
            bounds = icdf_fn(levels, x)
            if bounds.shape == (n_test, 2):
                return bounds[:, 0:1], bounds[:, 1:2]
        except Exception as e:
            logger.debug(f"DistributionalConformal vectorized execution failed: {e}")

        # Try auto-vectorization with vmap (handles user functions written for single samples)
        try:
            # Map over x (dim 0) but keep levels constant (None)
            vmapped_fn = torch.vmap(icdf_fn, in_dims=(None, 0))
            bounds = vmapped_fn(levels, x)
            if bounds.shape == (n_test, 2):
                return bounds[:, 0:1], bounds[:, 1:2]
        except Exception as e:
            logger.debug(
                f"DistributionalConformal vmap execution failed, falling back to loop: {e}"
            )

        lower = torch.empty(n_test, 1, device=x.device, dtype=x.dtype)
        upper = torch.empty(n_test, 1, device=x.device, dtype=x.dtype)

        for i in range(n_test):
            bounds = icdf_fn(levels, x[i])
            lower[i, 0] = bounds[0]
            upper[i, 0] = bounds[1]

        return lower, upper


# ---------------------------------------------------------------------------
# Regression-as-Classification Conformal (R2CCP)
# ---------------------------------------------------------------------------


class R2CConformal(ConformalPredictor):
    """Conformal prediction via regression-as-classification.

    The model outputs softmax probabilities over discretized target bins.
    The conformal score is 1 minus the cumulative probability up to the
    true bin.  The prediction set contains all bins whose inclusion
    increases cumulative probability up to 1-alpha + q_hat.

    This naturally handles **multimodal, skewed, and heteroscedastic**
    targets.

    Args:
        alpha: Miscoverage rate.
        bin_edges: 1-D tensor of bin edges (length n_bins + 1).

    Example:
        >>> bin_edges = torch.linspace(-5, 5, 101)  # 100 bins
        >>> r2c = R2CConformal(alpha=0.1, bin_edges=bin_edges)
        >>> # probs_cal: (n_cal, n_bins) softmax probabilities
        >>> r2c.calibrate(probs_cal, targets_cal)
        >>> lower, upper = r2c.predict_interval(probs_test)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        bin_edges: Optional[Tensor] = None,
    ) -> None:
        super().__init__(alpha=alpha)
        self.bin_edges = bin_edges

    def _target_to_bin(self, target: Tensor) -> Tensor:
        """Map continuous targets to bin indices."""
        if self.bin_edges is None:
            raise ValueError("bin_edges must be set before calibration")
        target_flat = target.view(-1)
        # searchsorted returns the bin index
        bins = torch.searchsorted(self.bin_edges[1:], target_flat)
        bins = bins.clamp(max=self.bin_edges.shape[0] - 2)
        return bins

    def _compute_scores(self, y_pred: Tensor, target: Tensor) -> Tensor:
        # y_pred: (n, n_bins) softmax probabilities
        # target: (n, 1) or (n,) continuous targets
        bin_idx = self._target_to_bin(target)  # (n,)

        # APS-style score: cumulative probability of bins with higher
        # probability than the true bin, plus the true bin's probability.
        # Vectorized: gather true bin probs, then compare all bins.
        true_prob = y_pred.gather(1, bin_idx.unsqueeze(1)).squeeze(1)  # (n,)
        # Mask bins strictly above true bin probability: (n, n_bins)
        above = y_pred > true_prob.unsqueeze(1)
        # Sum probability mass of bins above, plus true bin's own probability
        scores = (y_pred * above.float()).sum(dim=1) + true_prob

        return scores

    def predict_interval(
        self,
        y_pred: Tensor,
        *,
        groups: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Predict intervals from softmax probabilities.

        Returns the bounding box [min(bins_in_set), max(bins_in_set)]
        using bin center values.

        Args:
            y_pred: (n_test, n_bins) softmax probabilities.
            groups: Optional Mondrian groups.
            x: Unused.

        Returns:
            (lower, upper) tensors of shape ``(n_test, 1)``.
        """
        if not self._is_calibrated or self.q_hat is None:
            raise RuntimeError("Call calibrate() first.")
        if self.bin_edges is None:
            raise ValueError("bin_edges must be set.")

        q = self.q_hat if isinstance(self.q_hat, Tensor) else next(iter(self.q_hat.values()))
        threshold = q.item()

        # Bin centers
        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        # Vectorized: sort all samples at once, cumsum, then mask
        sorted_probs, sorted_idx = y_pred.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)  # (n_test, n_bins)

        # Mask: bins to include (cum_prob < threshold PLUS one more)
        # Shift cumsum right by one (first bin is always included)
        shifted = torch.cat(
            [
                torch.zeros(y_pred.shape[0], 1, device=y_pred.device, dtype=y_pred.dtype),
                cum_probs[:, :-1],
            ],
            dim=1,
        )
        included_mask = shifted < threshold  # (n_test, n_bins)

        # Map sorted indices back to bin indices, then to centers
        # Use large/small sentinel for excluded bins
        bin_centers_sorted = centers[sorted_idx]  # (n_test, n_bins)
        INF = float("inf")
        # For lower: excluded bins → +inf, take min
        lower = torch.amin(
            torch.where(included_mask, bin_centers_sorted, INF), dim=-1, keepdim=True
        )
        # For upper: excluded bins → -inf, take max
        upper = torch.amax(
            torch.where(included_mask, bin_centers_sorted, -INF), dim=-1, keepdim=True
        )

        return lower, upper

    def _build_intervals(
        self,
        y_pred: Tensor,
        q: Tensor,
        difficulty: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # R2C uses its own predict_interval implementation
        raise NotImplementedError("Use predict_interval() directly.")


# ---------------------------------------------------------------------------
# Multi-Target Conformal Prediction
# ---------------------------------------------------------------------------


class MultiTargetConformal(ConformalPredictor):
    """Per-dimension conformal prediction for multi-output regression.

    Calibrates separate quantile thresholds per output dimension, producing
    tighter intervals than a single global threshold.

    Args:
        alpha: Miscoverage rate.
    """

    def _compute_scores(self, y_pred: Tensor, target: Tensor) -> Tensor:
        # Not used directly — calibrate is overridden
        return torch.abs(y_pred - target)

    def calibrate(
        self,
        y_pred: Tensor,
        target: Tensor,
        *,
        mask: Optional[Tensor] = None,
        groups: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> None:
        """Calibrate per-dimension thresholds."""
        if mask is not None:
            mask_1d = mask.all(dim=-1) if mask.dim() > 1 else mask
            y_pred = y_pred[mask_1d]
            target = target[mask_1d]

        scores = torch.abs(y_pred - target)
        n = scores.shape[0]
        q_level = min(math.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.q_hat = torch.quantile(scores, q_level, dim=0, interpolation="higher")
        self._is_calibrated = True

    def _build_intervals(
        self,
        y_pred: Tensor,
        q: Tensor,
        difficulty: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return y_pred - q, y_pred + q

    def predict_interval(
        self,
        y_pred: Tensor,
        *,
        groups: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return per-dimension calibrated intervals."""
        if not self._is_calibrated or self.q_hat is None:
            raise RuntimeError("Call calibrate() first.")
        if isinstance(self.q_hat, dict):
            raise RuntimeError("MultiTargetConformal expects tensor q_hat, got grouped thresholds.")
        q = self.q_hat.to(y_pred.device)
        return self._build_intervals(y_pred, q)


# ---------------------------------------------------------------------------
# Density-/prevalence-aware Conformal Predictors
# ---------------------------------------------------------------------------


class DensityConformal(ConformalPredictor):
    """Density-adaptive split conformal prediction.

    Calibrates residual scores normalized by local target density so that low-density
    regions receive wider intervals by default.

    Args:
        alpha: Miscoverage rate.
        bandwidth: Gaussian KDE bandwidth for 1-D density estimates.
        min_density: Lower clamp for numerical stability.
        adapt_prediction: If True, estimate density at prediction time from y_pred.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        bandwidth: float = 0.25,
        min_density: float = 1e-4,
        adapt_prediction: bool = True,
    ) -> None:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        super().__init__(alpha=alpha, normalize_fn=self._density_normalizer)
        self.bandwidth = bandwidth
        self.min_density = min_density
        self.adapt_prediction = adapt_prediction
        self._reference_target_1d: Optional[Tensor] = None
        self._calibration_density_mean: Optional[float] = None

    def _density_normalizer(self, y_pred: Tensor, x: Tensor) -> Tensor:
        return x.clamp(min=self.min_density).sqrt()

    @staticmethod
    def _to_1d(values: Tensor) -> Tensor:
        if values.dim() == 1:
            return values
        return values.reshape(values.shape[0], -1).mean(dim=-1)

    def _kde_density_1d(self, query: Tensor, reference: Tensor) -> Tensor:
        q = query.reshape(-1, 1)
        r = reference.reshape(1, -1)
        z = (q - r) / self.bandwidth
        kernel = torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
        density = kernel.mean(dim=1) / self.bandwidth
        return density.clamp(min=self.min_density)

    def _compute_scores(self, y_pred: Tensor, target: Tensor) -> Tensor:
        scores = torch.abs(y_pred - target)
        if scores.dim() > 1:
            scores = scores.max(dim=-1).values
        return scores

    def _build_intervals(
        self,
        y_pred: Tensor,
        q: Tensor,
        difficulty: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if difficulty is not None:
            width = q * difficulty.view(-1, *([1] * (y_pred.dim() - 1)))
        else:
            width = q
        return y_pred - width, y_pred + width

    def calibrate(
        self,
        y_pred: Tensor,
        target: Tensor,
        *,
        mask: Optional[Tensor] = None,
        groups: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
        density: Optional[Tensor] = None,
    ) -> None:
        target_1d = self._to_1d(target.detach())
        if mask is not None:
            mask_1d = mask.all(dim=-1) if mask.dim() > 1 else mask
            target_1d = target_1d[mask_1d]

        self._reference_target_1d = target_1d.detach()

        if density is None:
            density = self._kde_density_1d(target_1d, target_1d)
        if density.shape[0] != target_1d.shape[0]:
            raise ValueError("density must have length equal to calibration batch size")

        self._calibration_density_mean = float(density.mean().item())
        super().calibrate(
            y_pred,
            target,
            mask=mask,
            groups=groups,
            weights=weights,
            x=density,
        )

    def predict_interval(
        self,
        y_pred: Tensor,
        *,
        groups: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
        density: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if density is None and self.adapt_prediction:
            if self._reference_target_1d is not None:
                pred_1d = self._to_1d(y_pred.detach())
                ref = self._reference_target_1d.to(y_pred.device)
                density = self._kde_density_1d(pred_1d, ref)
        if density is None:
            fill = self._calibration_density_mean if self._calibration_density_mean else 1.0
            density = torch.full(
                (y_pred.shape[0],),
                fill_value=float(fill),
                dtype=y_pred.dtype,
                device=y_pred.device,
            )
        return super().predict_interval(y_pred, groups=groups, x=density)


class PrevalenceAdjustedCP(ConformalPredictor):
    """Group-prevalence adjusted split conformal prediction.

    Rare groups receive tighter miscoverage rates (larger intervals), while common
    groups use less conservative thresholds.

    Args:
        alpha: Global miscoverage rate.
        n_bins: Number of target bins when explicit groups are not provided.
        min_group_size: Minimum per-group sample count.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        n_bins: int = 5,
        min_group_size: int = 8,
    ) -> None:
        super().__init__(alpha=alpha)
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = n_bins
        self.min_group_size = min_group_size
        self._bin_edges: Optional[Tensor] = None

    @staticmethod
    def _to_1d(values: Tensor) -> Tensor:
        if values.dim() == 1:
            return values
        return values.reshape(values.shape[0], -1).mean(dim=-1)

    def _compute_scores(self, y_pred: Tensor, target: Tensor) -> Tensor:
        scores = torch.abs(y_pred - target)
        if scores.dim() > 1:
            scores = scores.max(dim=-1).values
        return scores

    def _build_intervals(
        self,
        y_pred: Tensor,
        q: Tensor,
        difficulty: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return y_pred - q, y_pred + q

    def _auto_groups_from_target(self, target: Tensor) -> Tensor:
        target_1d = self._to_1d(target.detach())
        quantiles = torch.linspace(0.0, 1.0, self.n_bins + 1, device=target.device)
        edges = torch.quantile(target_1d, quantiles)
        # Ensure strictly increasing edges for bucketize.
        edges = torch.unique(edges, sorted=True)
        if edges.numel() < 3:
            edges = torch.linspace(
                target_1d.min() - 1e-6,
                target_1d.max() + 1e-6,
                self.n_bins + 1,
                device=target.device,
            )
        self._bin_edges = edges
        return torch.bucketize(target_1d, edges[1:-1])

    def calibrate(
        self,
        y_pred: Tensor,
        target: Tensor,
        *,
        mask: Optional[Tensor] = None,
        groups: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> None:
        if mask is not None:
            mask_1d = mask.all(dim=-1) if mask.dim() > 1 else mask
            y_pred = y_pred[mask_1d]
            target = target[mask_1d]
            if groups is not None:
                groups = groups[mask_1d]
            if weights is not None:
                weights = weights[mask_1d]

        if groups is None:
            groups = self._auto_groups_from_target(target)

        scores = self._compute_scores(y_pred, target)
        unique_groups = groups.unique()
        n_total = max(int(scores.shape[0]), 1)
        q_map: Dict[Any, Tensor] = {}

        for g in unique_groups.tolist():
            g_mask = groups == g
            g_scores = scores[g_mask]
            if g_scores.numel() == 0:
                continue
            if g_scores.numel() < self.min_group_size:
                q_level = 1.0 - self.alpha
            else:
                prevalence = g_scores.numel() / n_total
                alpha_g = max(self.alpha * math.sqrt(prevalence), 1e-3)
                q_level = 1.0 - alpha_g
            g_weights = weights[g_mask] if weights is not None else None
            q_map[g] = _weighted_quantile(g_scores, q_level, g_weights)

        if not q_map:
            raise ValueError("No groups available for prevalence-adjusted calibration")

        self.q_hat = q_map
        self._groups_list = list(q_map.keys())
        self._is_calibrated = True

    def predict_interval(
        self,
        y_pred: Tensor,
        *,
        groups: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if not self._is_calibrated or self.q_hat is None:
            raise RuntimeError("Predictor must be calibrated before making predictions.")
        if not isinstance(self.q_hat, dict):
            raise RuntimeError("PrevalenceAdjustedCP expects groupwise q_hat values.")

        if groups is None:
            if self._bin_edges is None:
                raise ValueError("groups must be provided when no calibration bin edges exist")
            pred_1d = self._to_1d(y_pred.detach())
            edges = self._bin_edges.to(y_pred.device)
            groups = torch.bucketize(pred_1d, edges[1:-1])

        q_default = max(self.q_hat.values(), key=lambda x: float(x.item()))
        q_per_sample = torch.full(
            (y_pred.shape[0],),
            fill_value=float(q_default.item()),
            dtype=y_pred.dtype,
            device=y_pred.device,
        )
        for g, q_val in self.q_hat.items():
            q_per_sample[groups == g] = float(q_val.item())
        q = q_per_sample.view(-1, *([1] * (y_pred.dim() - 1)))
        return self._build_intervals(y_pred, q)


class MonteCarloConformal(ConformalPredictor):
    """Conformal prediction from Monte-Carlo predictive samples.

    Uses MC sample mean/median as the point predictor and MC sample standard
    deviation as the difficulty normalizer.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        center: str = "mean",
        min_uncertainty: float = 1e-6,
    ) -> None:
        self.center = center.lower()
        if self.center not in {"mean", "median"}:
            raise ValueError("center must be 'mean' or 'median'")
        self.min_uncertainty = min_uncertainty
        super().__init__(alpha=alpha, normalize_fn=self._uncertainty_normalizer)

    def _uncertainty_normalizer(self, y_pred: Tensor, x: Tensor) -> Tensor:
        return x.clamp(min=self.min_uncertainty)

    def _compute_scores(self, y_pred: Tensor, target: Tensor) -> Tensor:
        scores = torch.abs(y_pred - target)
        if scores.dim() > 1:
            scores = scores.max(dim=-1).values
        return scores

    def _build_intervals(
        self,
        y_pred: Tensor,
        q: Tensor,
        difficulty: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if difficulty is not None:
            width = q * difficulty.view(-1, *([1] * (y_pred.dim() - 1)))
        else:
            width = q
        return y_pred - width, y_pred + width

    def _extract_center_and_uncertainty(self, mc_samples: Tensor) -> Tuple[Tensor, Tensor]:
        if mc_samples.dim() < 2:
            raise ValueError("mc_samples must have shape [n_samples, batch, ...]")
        if self.center == "median":
            center = mc_samples.median(dim=0).values
        else:
            center = mc_samples.mean(dim=0)
        uncertainty = mc_samples.std(dim=0, unbiased=False).clamp(min=self.min_uncertainty)
        return center, uncertainty

    def calibrate(
        self,
        mc_samples: Tensor,
        target: Tensor,
        *,
        mask: Optional[Tensor] = None,
        groups: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> None:
        center, uncertainty = self._extract_center_and_uncertainty(mc_samples)
        super().calibrate(
            center,
            target,
            mask=mask,
            groups=groups,
            weights=weights,
            x=uncertainty,
        )

    def predict_interval(
        self,
        mc_samples: Tensor,
        *,
        groups: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        center, uncertainty = self._extract_center_and_uncertainty(mc_samples)
        return super().predict_interval(center, groups=groups, x=uncertainty)


# ---------------------------------------------------------------------------
# Backward-compatible Loss Wrappers
# ---------------------------------------------------------------------------


@register_regression_loss("conformal")
class ConformalLoss(RegressionLoss):
    """Conformal prediction loss with calibration and interval prediction.

    Combines a training loss with a conformal predictor.  Supports two
    training regimes:

    - ``split``: MSE loss (point prediction) + SplitConformal calibration.
    - ``cqr``: Pinball loss (quantile prediction) + CQR calibration.
    - ``uacqr``: Same training loss as ``cqr`` + :class:`UACQR` calibration
      (width-normalized scores).

    All composable features (normalized scores, Mondrian groups, covariate
    shift weighting) are available through the ``calibrate`` method.

    Args:
        method: ``'split'``, ``'cqr'``, or ``'uacqr'``.
        alpha: Miscoverage rate.
        debias: If True and method is ``'cqr'`` or ``'uacqr'``, apply coverage bias correction.
        reduction: Reduction for training loss.
        normalize_fn: Optional difficulty normalization (not used for ``'uacqr'``).
        uacqr_min_width: For ``method='uacqr'``, floor on predicted quantile width.
        uacqr_aggregation: For ``method='uacqr'``, ``\"mean\"`` or ``\"max\"`` across targets.

    Example:
        >>> loss_fn = ConformalLoss(method='cqr', alpha=0.1)
        >>> train_loss = loss_fn(y_pred, target)
        >>> loss_fn.calibrate(cal_pred, cal_target)
        >>> lower, upper = loss_fn.predict_interval(test_pred)
    """

    def __init__(
        self,
        method: str = "cqr",
        alpha: float = 0.1,
        debias: bool = False,
        reduction: str = "mean",
        normalize_fn: Optional[Callable[..., Tensor]] = None,
        *,
        uacqr_min_width: float = 1e-6,
        uacqr_aggregation: str = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(reduction=reduction)
        method = method.lower()
        if method not in ("split", "cqr", "uacqr"):
            raise ValueError(
                f"Unknown method: {method}. Supported methods: 'split', 'cqr', 'uacqr'"
            )
        self.method = method
        self.alpha = alpha

        # Create the underlying predictor
        self._predictor: ConformalPredictor
        if method == "cqr":
            self._predictor = CQR(alpha=alpha, debias=debias, normalize_fn=normalize_fn)
        elif method == "uacqr":
            if normalize_fn is not None:
                raise ValueError(
                    "normalize_fn is not supported for method='uacqr'; use UACQR directly."
                )
            self._predictor = UACQR(
                alpha=alpha,
                debias=debias,
                min_width=uacqr_min_width,
                aggregation=uacqr_aggregation,
            )
        else:
            self._predictor = SplitConformal(alpha=alpha, normalize_fn=normalize_fn)

    @property
    def _is_calibrated(self) -> bool:
        return self._predictor._is_calibrated

    @_is_calibrated.setter
    def _is_calibrated(self, value: bool) -> None:
        self._predictor._is_calibrated = value

    @property
    def q_hat(self) -> Optional[Union[Tensor, Dict[Any, Tensor]]]:
        return self._predictor.q_hat

    @q_hat.setter
    def q_hat(self, value: Optional[Union[Tensor, Dict[Any, Tensor]]]) -> None:
        self._predictor.q_hat = value

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Training loss (MSE for split, pinball for CQR)."""
        if self.method in ("cqr", "uacqr"):
            # See CQR._compute_scores: 1-D targets need a feature axis or
            # the subtraction against [N, n_feat] slices broadcasts wrongly.
            if target.dim() == 1 and y_pred.dim() > 1:
                target = target.unsqueeze(-1)
            n_feat = target.shape[-1] if target.dim() > 1 else 1
            if y_pred.shape[-1] != 2 * n_feat:
                raise ValueError(f"CQR expects y_pred shape [..., 2*features], got {y_pred.shape}")
            lower_pred = y_pred[..., :n_feat]
            upper_pred = y_pred[..., n_feat:]
            lower_q = self.alpha / 2
            upper_q = 1 - self.alpha / 2
            lower_err = target - lower_pred
            lower_loss = torch.maximum(lower_q * lower_err, (lower_q - 1) * lower_err)
            upper_err = target - upper_pred
            upper_loss = torch.maximum(upper_q * upper_err, (upper_q - 1) * upper_err)
            loss = lower_loss + upper_loss
        else:
            self._validate_inputs(y_pred, target, mask)
            loss = (y_pred - target) ** 2

        return self._reduce_with_mask(loss, mask, weights)

    def calibrate(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        *,
        groups: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> None:
        """Calibrate the conformal predictor.

        Args:
            y_pred: Predictions on calibration set.
            target: True values.
            mask: Optional mask.
            groups: Optional Mondrian groups.
            weights: Optional importance weights.
            x: Optional features for normalized CP.
        """
        self._predictor.calibrate(y_pred, target, mask=mask, groups=groups, weights=weights, x=x)

    def predict_interval(
        self,
        y_pred: Tensor,
        *,
        groups: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Predict intervals using the calibrated predictor."""
        return self._predictor.predict_interval(y_pred, groups=groups, x=x)


@register_regression_loss("multidim_conformal")
class MultiDimensionalConformalLoss(ConformalLoss):
    """Multi-dimensional conformal prediction for multi-output regression.

    Calibrates separate thresholds per output dimension for tighter
    intervals than a single global threshold.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        # Initialize RegressionLoss directly (skip ConformalLoss __init__)
        RegressionLoss.__init__(self, reduction=reduction)
        self.method = "split"
        self.alpha = alpha
        self._predictor: ConformalPredictor
        self._predictor = MultiTargetConformal(alpha=alpha)

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """MSE training loss."""
        self._validate_inputs(y_pred, target, mask)
        loss = (y_pred - target) ** 2
        return self._reduce_with_mask(loss, mask, weights)

    def calibrate(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        *,
        groups: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> None:
        """Calibrate per-dimension thresholds."""
        self._predictor.calibrate(y_pred, target, mask=mask)

    def predict_interval(
        self,
        y_pred: Tensor,
        *,
        groups: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Return per-dimension calibrated intervals."""
        return self._predictor.predict_interval(y_pred)
