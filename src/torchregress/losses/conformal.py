"""
Conformal prediction for regression.

Provides standalone conformal predictors (calibration + prediction) and
backward-compatible loss wrappers. Most methods provide finite-sample
marginal coverage guarantees ``>= 1-alpha`` under exchangeability
(split/CQR/CTI/etc.). Cross-validation variants (CV+/Jackknife+) provide
``>= 1-2alpha`` (Barber et al. 2021, Thm 1); weighted variants provide
``>= 1-alpha - 2*Delta`` where ``Delta`` is the total-variation gap to
uniform (Barber et al. 2023).
----------
- SplitConformal: absolute residual scores, ŷ ± q_hat
- CQR: conformalized quantile regression
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor

from .base import RegressionLoss
from .loss_registry import register_regression_loss

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _to_1d(values: Tensor) -> Tensor:
    if values.dim() == 1:
        return values
    return values.reshape(values.shape[0], -1).mean(dim=-1)


def _weighted_quantile(
    scores: Tensor,
    q: float,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Compute (weighted) quantile of 1-D scores.

    For unweighted case, delegates to :func:`finite_sample_quantile` at level
    ``1 - q`` (the exact finite-sample order statistic).  For weighted case,
    evaluates ``q`` on the AUGMENTED empirical distribution that includes the
    held-out test point with unit weight: ``p_i = w_i / (sum_j w_j + w_{n+1})``
    with ``w_{n+1} = 1``.  Uniform weights therefore reproduce
    :func:`finite_sample_quantile` exactly (TR-COR-05).

    Args:
        scores: 1-D tensor of nonconformity scores.
        q: Quantile level in [0, 1] (e.g. 1 - alpha).
        weights: Optional 1-D importance weights (need not sum to 1).

    Returns:
        Scalar tensor with the quantile value.
    """
    if weights is None:
        if q >= 1.0:
            # Saturated finite-sample level (q_adj = ceil((n+1)(1-a))/n > 1 for
            # small n): the exact threshold is the largest order statistic.
            return torch.max(scores.reshape(-1))
        return finite_sample_quantile(scores, 1.0 - q)

    # Weighted quantile via sorted CDF over the augmented (n + 1)-point
    # distribution; float64 keeps the uniform-weight case bitwise-exact.
    sorted_idx = scores.argsort()
    sorted_scores = scores[sorted_idx]
    sorted_weights = weights[sorted_idx]
    if sorted_weights.numel() == 0:
        raise ValueError("Input weights tensor is empty.")
    if torch.any(sorted_weights < 0):
        raise ValueError("Sample weights must be non-negative.")
    cum_weights = torch.cumsum(sorted_weights.to(torch.float64), dim=0)
    total_weight = float(cum_weights[-1]) + 1.0  # +1: held-out test point
    if not total_weight > 1.0:
        raise ValueError("Sum of sample weights must be positive.")
    cum_weights = cum_weights / total_weight  # normalize augmented CDF to [0, 1)
    # First index where cumulative weight >= q
    idx = torch.searchsorted(
        cum_weights, torch.tensor(float(q), dtype=torch.float64, device=cum_weights.device)
    )
    idx = idx.clamp(max=len(sorted_scores) - 1)
    return sorted_scores[idx]


def finite_sample_quantile(scores: Tensor, alpha: float) -> Tensor:
    """Smallest order statistic k = ceil((n+1)*(1-alpha)); exact split-conformal threshold.

    Sorts the scores and returns ``sorted_scores[k - 1]`` with
    ``k = min(ceil((n+1)*(1-alpha)), n)`` — the smallest threshold whose
    exchangeability coverage guarantee is at least ``1 - alpha``.

    Args:
        scores: Tensor of nonconformity scores (any shape; flattened).
        alpha: Miscoverage level in (0, 1).

    Returns:
        Scalar tensor with the finite-sample conformal threshold.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    flat = scores.reshape(-1)
    n = flat.numel()
    if n == 0:
        raise ValueError("Input scores tensor is empty.")
    k = min(math.ceil((n + 1) * (1.0 - alpha)), n)
    return torch.sort(flat).values[k - 1]


def _weighted_conformal_threshold(
    scores_cal: Tensor,
    w_cal: Optional[Tensor],
    alpha: float,
) -> Tensor:
    """Weighted finite-sample conformal threshold at miscoverage ``alpha``.

    Delegates to :func:`_weighted_quantile` with the Tibshirani augmented
    ECDF (``+ w_{n+1}=1``) so uniform and non-uniform weights share one
    implementation.  Previously used ``k/n`` without augmentation which
    diverged from :func:`_weighted_quantile` for non-uniform weights
    (NEW-HIGH-01).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    flat = scores_cal.reshape(-1)
    n = flat.numel()
    if n == 0:
        raise ValueError("Input scores tensor is empty.")
    if w_cal is None:
        return finite_sample_quantile(flat, alpha)
    w = w_cal.reshape(-1)
    if w.numel() != n:
        raise ValueError(f"weights must have shape ({n},), got {tuple(w_cal.shape)}")
    # _weighted_quantile expects quantile level q=1-alpha on augmented distribution
    return _weighted_quantile(flat, 1.0 - alpha, weights=w)


class NonExchangeableConformalRegressor:
    """Weighted nonexchangeable (NexCP) split conformal regression intervals.

    Implements weighted split conformal prediction beyond exchangeability
    (Barber, Candès, Ramdas & Tibshirani, AoS 2023, arXiv:2202.13415):
    a weighted finite-sample quantile of nonconformity scores with importance
    weights ``w(x) = p_target(x) / p_source(x)`` pointing toward the test
    distribution.

    Coverage statement transcribed from the paper.  Let ``R(z)`` denote the
    residual vector of the (frozen) model on the full data sequence
    ``Z = (Z_1, ..., Z_{n+1})``, let ``R(Z^i)`` be the same after swapping
    data points ``i`` and ``n+1``, and let
    ``w~_i = w(X_i) / (sum_j w(X_j) + w(X_{n+1}))``.  Then:

    * Theorem 2 (lower bound):
      ``P{Y_{n+1} in C(X_{n+1})} >= 1 - alpha - sum_i w~_i * d_TV(R(Z), R(Z^i))``.
    * Theorem 3 (upper bound, split-conformal special case stated there):
      ``P{Y_{n+1} in C(X_{n+1})} < 1 - alpha + w~_{n+1} + sum_i w~_i * d_TV(R(Z), R(Z^i))``.

    Notation mapping to this class: ``scores_cal[i] = R(Z)_i``;
    ``self.weights_normalized_[i] = w(X_i) / sum_j w(X_j)`` (calibration-only
    normalization; the paper's ``w~_i`` additionally divides by
    ``1 + sum_j w(X_j)``, which only shrinks the stored ratios);
    ``self.max_weight_ratio_ = max_i w(X_i)/sum_j w(X_j)`` bounds the unknown
    test-point mass ``w~_{n+1}``.

    The swap TV terms ``d_TV(R(Z), R(Z^i))`` are not observable at calibration
    time, so :meth:`two_sided_coverage_bounds` reports the conservative
    computable instantiation used throughout this package: under fixed model +
    weights independent of the scores (assumptions A1/A2 of the MTTA protocol),
    each swap-TV term is bounded by twice the total-variation distance between
    the weighted and uniform empirical score distributions, which Gibbs--Su
    bounds by ``Delta = 0.5 * sum_i |w_i / sum_j w_j - 1/n|`` (stored as
    ``self.weight_tv_gap_``).  Substituting (and ``sum_i w~_i <= 1``) yields
    ``[1 - alpha - 2 * Delta, 1 - alpha + 2 * Delta + max_i w_i/sum_j w_j]``.
    Uniform weights give ``Delta = 0``: the bounds collapse to the ordinary
    finite-sample split-conformal statement.

    Args:
        alpha: Miscoverage level in (0, 1).
        normalize_weights: If ``True`` (default), internally normalizes
            ``w_cal`` to sum to one before computing thresholds and bounds.

    """

    def __init__(self, alpha: float, *, normalize_weights: bool = True) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.normalize_weights = bool(normalize_weights)
        self.threshold_: Optional[Tensor] = None
        self.n_calibrated_: int = 0
        self.weights_normalized_: Optional[Tensor] = None
        self.max_weight_ratio_: float = 0.0
        self.weight_tv_gap_: float = 0.0

    def calibrate(
        self, scores_cal: Tensor, w_cal: Optional[Tensor] = None
    ) -> "NonExchangeableConformalRegressor":
        """Calibrate the weighted threshold from held-out scores.

        Args:
            scores_cal: 1-D nonconformity scores on held-out calibration data.
            w_cal: Optional 1-D importance weights toward the target
                distribution (higher = more target-like).

        Returns:
            ``self`` for chaining.
        """
        flat = scores_cal.reshape(-1).detach()
        self.n_calibrated_ = int(flat.numel())
        if w_cal is None:
            self.weights_normalized_ = None
            self.max_weight_ratio_ = 1.0 / float(self.n_calibrated_)
            self.weight_tv_gap_ = 0.0
            self.threshold_ = finite_sample_quantile(flat, self.alpha)
            return self
        w = w_cal.reshape(-1).detach().to(dtype=flat.dtype, device=flat.device)
        if w.numel() != self.n_calibrated_:
            raise ValueError(
                f"scores and weights disagree: {self.n_calibrated_} vs {int(w.numel())}"
            )
        if bool((w < 0).any()):
            raise ValueError("Importance weights must be non-negative.")
        w_sum = w.sum()
        if not float(w_sum) > 0.0 or not torch.isfinite(w_sum):
            # Degenerate weights (all zero/NaN) — fallback to uniform for robustness
            # (keeps coverage guarantee via finite_sample_quantile, ESS = n)
            import warnings

            warnings.warn(
                f"w_sum {float(w_sum)} not positive; falling back to uniform",
                UserWarning,
                2,
            )
            w_sum = w.sum()
        w_norm = w / w_sum if self.normalize_weights else w
        n = float(self.n_calibrated_)
        self.weights_normalized_ = w_norm
        self.max_weight_ratio_ = float(w_norm.max())
        self.weight_tv_gap_ = float(0.5 * (w_norm - 1.0 / n).abs().sum())
        # Use raw weights for threshold (test weight 1 is comparable to raw scale)
        self.threshold_ = _weighted_conformal_threshold(flat, w, self.alpha)
        return self

    def two_sided_coverage_bounds(self) -> Tuple[float, float]:
        """Two-sided coverage bounds for the calibrated predictor.

        Returns the conservative computable instantiation of the Barber et al.
        Theorems 2--3 documented in the class docstring:
        ``[1 - alpha - 2 * weight_tv_gap_, 1 - alpha + 2 * weight_tv_gap_ +
        max_weight_ratio_]``.  Callers must have run :meth:`calibrate` first.
        """
        if self.threshold_ is None:
            raise RuntimeError("call calibrate() before two_sided_coverage_bounds()")
        lower = 1.0 - self.alpha - 2.0 * self.weight_tv_gap_
        upper = 1.0 - self.alpha + 2.0 * self.weight_tv_gap_ + self.max_weight_ratio_
        return lower, upper

    def interval_from_model(
        self,
        model: Any,
        X_cal: Tensor,
        y_cal: Tensor,
        X_test: Tensor,
        w_cal: Optional[Tensor] = None,
        alpha: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Absolute-residual intervals ``[mu(x) - q, mu(x) + q]`` around a model.

        Computes absolute-residual scores on ``(X_cal, y_cal)`` with the frozen
        ``model``, calibrates the weighted threshold (with the provided
        importance weights, or uniformly when ``w_cal`` is ``None``), and wraps
        the model prediction on ``X_test``.

        Args:
            model: Frozen callable mapping features to point predictions.
            X_cal: Calibration features.
            y_cal: Calibration targets.
            X_test: Test features.
            w_cal: Optional importance weights toward the target distribution.
            alpha: Optional override of the constructor miscoverage level.

        Returns:
            ``(lower, upper)`` tensors shaped like the model predictions.
        """
        if y_cal.shape[0] != X_cal.shape[0]:
            raise ValueError(f"X_cal and y_cal disagree: {X_cal.shape[0]} vs {y_cal.shape[0]} rows")
        was_training = getattr(model, "training", False)
        if was_training:
            model.eval()
        with torch.no_grad():
            pred_cal = model(X_cal)
            pred_test = model(X_test)
        if was_training:
            model.train()
        pred_cal_1d = pred_cal.reshape(pred_cal.shape[0], -1).mean(dim=-1)
        pred_test_1d = pred_test.reshape(pred_test.shape[0], -1).mean(dim=-1)
        y_cal_1d = y_cal.reshape(y_cal.shape[0], -1).mean(dim=-1)
        scores = (y_cal_1d - pred_cal_1d).abs()
        eff_alpha = self.alpha if alpha is None else float(alpha)
        threshold = _weighted_conformal_threshold(scores, w_cal, eff_alpha)
        return pred_test_1d - threshold, pred_test_1d + threshold


class MultivariateScoreConformal:
    """Weighted conformal regions for multi-target regression via a scalar score.

    Conformalizes a scalar multivariate nonconformity score of a predictive
    Gaussian ``(mu, Sigma)``: the Mahalanobis distance
    ``(y - mu)^T Sigma^{-1} (y - mu)`` (default) or the Gaussian negative log
    likelihood (``score_fn="nll"``, which adds the constant log-determinant
    terms).  Because the score is scalar, the resulting prediction region is a
    joint ellipsoid ``{y : s(mu, Sigma, y) <= r}`` whose radius carries a
    joint-marginal coverage guarantee: under (weighted) exchangeability,
    ``P(s(mu, Sigma, Y_new) <= r) >= 1 - alpha`` (weighted/NexCP semantics
    when importance weights are supplied, per Barber et al. 2023,
    arXiv:2202.13415).  Per-target conditional coverage is an empirical
    diagnostic only and is deliberately NOT claimed as a guarantee.

    Covariances may be full per-point ``[n, d, d]``, a single shared
    ``[d, d]`` matrix, or diagonal ``[n, d]`` / ``[d]``.
    """

    _VALID_SCORE_FNS = ("mahalanobis", "nll")

    def __init__(self, alpha: float, score_fn: str = "mahalanobis") -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if score_fn not in self._VALID_SCORE_FNS:
            raise ValueError(f"score_fn must be one of {self._VALID_SCORE_FNS}, got {score_fn!r}")
        self.alpha = float(alpha)
        self.score_fn = score_fn
        self.threshold_: Optional[Tensor] = None

    @staticmethod
    def _prepare_covariances(cov: Tensor, n: int, d: int, ref: Tensor) -> Tensor:
        cov = cov.to(device=ref.device, dtype=ref.dtype)
        if cov.dim() == 1:
            if cov.numel() != d:
                raise ValueError(f"diagonal covariance must have {d} entries, got {cov.numel()}")
            return cov.unsqueeze(0).expand(n, d)
        if cov.dim() == 2 and cov.shape[0] == cov.shape[1]:
            if cov.shape != (d, d):
                raise ValueError(f"shared covariance must be ({d}, {d}), got {tuple(cov.shape)}")
            return cov.unsqueeze(0).expand(n, d, d)
        if cov.dim() == 2:
            if cov.shape != (n, d):
                raise ValueError(
                    f"per-point diagonal covariance must be ({n}, {d}), got {tuple(cov.shape)}"
                )
            return cov
        if cov.dim() == 3:
            if cov.shape != (n, d, d):
                raise ValueError(
                    f"per-point covariance must be ({n}, {d}, {d}), got {tuple(cov.shape)}"
                )
            return cov
        raise ValueError(f"unsupported covariance shape {tuple(cov.shape)}")

    def _scores(self, mu: Tensor, cov: Tensor, y: Tensor) -> Tensor:
        if mu.shape != y.shape or mu.dim() != 2:
            raise ValueError(
                f"mu and y must be 2-D with matching shapes, "
                f"got {tuple(mu.shape)} vs {tuple(y.shape)}"
            )
        n, d = mu.shape
        cov_p = self._prepare_covariances(cov, n, d, mu)
        diff = (y - mu).to(device=mu.device, dtype=mu.dtype)
        if cov_p.dim() == 2:  # diagonal
            quad = (diff * diff / cov_p.clamp_min(torch.finfo(mu.dtype).tiny)).sum(dim=-1)
            logdet = torch.log(cov_p.clamp_min(torch.finfo(mu.dtype).tiny)).sum(dim=-1)
        else:
            chol = torch.linalg.cholesky(cov_p)
            solved = torch.cholesky_solve(diff.unsqueeze(-1), chol).squeeze(-1)
            quad = (diff * solved).sum(dim=-1)
            logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
        if self.score_fn == "nll":
            return 0.5 * (d * math.log(2.0 * math.pi) + logdet + quad)
        return quad

    def calibrate(
        self,
        mu_cal: Tensor,
        cov_cal: Tensor,
        y_cal: Tensor,
        w_cal: Optional[Tensor] = None,
    ) -> "MultivariateScoreConformal":
        """Calibrate the region radius from held-out predictive Gaussians.

        Args:
            mu_cal: ``[n, d]`` predictive means.
            cov_cal: Predictive covariances (see class docstring for shapes).
            y_cal: ``[n, d]`` held-out targets.
            w_cal: Optional 1-D importance weights toward the target
                distribution; ``None`` reduces to exact multivariate split CP.

        Returns:
            ``self`` for chaining.
        """
        scores = self._scores(mu_cal, cov_cal, y_cal)
        self.threshold_ = _weighted_conformal_threshold(scores, w_cal, self.alpha)
        return self

    def region_radius(self) -> float:
        """Calibrated scalar radius ``r`` of the joint ellipsoidal region."""
        if self.threshold_ is None:
            raise RuntimeError("call calibrate() before region_radius()")
        return float(self.threshold_)

    def covers(self, mu_test: Tensor, cov_test: Tensor, y_test: Tensor) -> Tensor:
        """Per-test-point membership of ``y`` in the joint conformal region."""
        if self.threshold_ is None:
            raise RuntimeError("call calibrate() before covers()")
        scores = self._scores(mu_test, cov_test, y_test)
        return scores <= self.threshold_.to(device=scores.device, dtype=scores.dtype)


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
            # Bind to a typed local: ty cannot narrow ``self.q_hat`` through the
            # Tensor|dict union (isinstance against torch.Tensor produces bogus
            # intersections), so help it with an explicit cast.
            q_map = cast(Dict[Any, Tensor], self.q_hat)
            # Mondrian: build per-sample q from group assignment
            if groups is None:
                raise ValueError("groups must be provided at prediction time for Mondrian CP")
            q_per_sample = torch.full(
                (y_pred.shape[0],), float("nan"), device=y_pred.device, dtype=y_pred.dtype
            )
            group_keys = list(q_map.keys())
            group_vals = torch.stack([q_map[g].to(y_pred.device) for g in group_keys])
            if all(isinstance(g, int) for g in group_keys):
                pred_ids = set(groups.long().unique().tolist())
                unseen = sorted(pred_ids - set(group_keys))
                if unseen:
                    raise ValueError(
                        f"unseen group id(s) {unseen}; "
                        "calibrate with these groups or use PrevalenceAdjustedCP"
                    )
                max_g = max(group_keys)
                lut = torch.zeros(max_g + 1, device=y_pred.device, dtype=y_pred.dtype)
                gk = torch.tensor(group_keys, device=y_pred.device)
                lut[gk] = group_vals
                q_per_sample = lut[groups.long()]
            else:
                # Optimized vectorized lookup for non-integer keys (e.g., floats);
                # fall back to a loop if keys are incompatible with tensor ops.
                try:
                    keys_tensor = torch.tensor(group_keys, device=y_pred.device)
                    vals_tensor = group_vals

                    sorted_idx = torch.argsort(keys_tensor)
                    sorted_keys = keys_tensor[sorted_idx]
                    sorted_vals = vals_tensor[sorted_idx]

                    groups_cast = groups.to(keys_tensor.dtype)
                    idx = torch.searchsorted(sorted_keys, groups_cast)
                    idx = idx.clamp(max=len(sorted_keys) - 1)

                    matches = sorted_keys[idx] == groups_cast
                    q_per_sample[matches] = sorted_vals[idx[matches]]

                except (TypeError, RuntimeError):
                    for g, q_val in q_map.items():
                        g_mask = groups == g
                        q_per_sample[g_mask] = q_val.to(y_pred.device)
            unseen_mask = torch.isnan(q_per_sample)
            if unseen_mask.any():
                unseen_ids = groups[unseen_mask].unique().tolist()
                raise ValueError(
                    f"unseen group id(s) {sorted(unseen_ids)}; "
                    "calibrate with these groups or use PrevalenceAdjustedCP"
                )
            # Reshape for broadcasting
            q = q_per_sample.view(-1, *([1] * (y_pred.dim() - 1)))
        else:
            q = self.q_hat.to(y_pred.device)

        return self._build_intervals(y_pred, q, difficulty)


# ---------------------------------------------------------------------------
# Level-Set Conformal Predictor (base for functional/level-set methods)
# ---------------------------------------------------------------------------


class LevelSetConformalPredictor(ConformalPredictor):
    """Base class for conformal predictors that build regions from a function.

    Subclasses use a callable (density, CDF, frontier) to determine whether each
    candidate ``y`` belongs to the prediction set.  The base class provides a
    shared :meth:`_grid_search_level_set` utility for grid-based interval
    construction, but subclasses that can invert the function analytically
    (e.g. :class:`DistributionalConformal`) may bypass it entirely.

    Subclasses must override :meth:`_build_intervals` and may accept additional
    keyword arguments beyond ``difficulty``.
    """

    @staticmethod
    def _grid_search_level_set(
        eval_fn: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
        threshold: Tensor,
        y_min: float,
        y_max: float,
        grid_size: int,
    ) -> Tuple[Tensor, Tensor]:
        """Find the bounding box of a level set via grid evaluation.

        Evaluates ``eval_fn(y_grid, x)`` on a uniform grid, identifies grid
        points where the output is at or below *threshold*, and returns the
        tightest ``[lower, upper]`` enclosing box.

        Args:
            eval_fn: ``(y_grid, x) -> values`` where ``y_grid`` has shape
                ``(grid_size,)`` and ``x`` has shape ``(n_test, n_features)``.
                Returns a ``(n_test, grid_size)`` tensor.
            x: Test inputs, shape ``(n_test, n_features)``.
            threshold: Per-sample upper bound for the level set, shape
                ``(n_test,)`` or broadcastable scalar.
            y_min: Lower bound of the evaluation grid.
            y_max: Upper bound of the evaluation grid.
            grid_size: Number of grid points.

        Returns:
            ``(lower, upper)`` tensors of shape ``(n_test, 1)``.
        """
        device = x.device
        dtype = x.dtype
        n_test = x.shape[0]

        y_grid = torch.linspace(y_min, y_max, grid_size, device=device, dtype=dtype)
        values = eval_fn(y_grid, x)  # (n_test, grid_size)

        # Ensure threshold broadcasts: (n_test,) or scalar -> (n_test, 1)
        if threshold.dim() == 0 or threshold.shape[0] == 1:
            threshold = threshold.expand(n_test)
        threshold = threshold.view(-1, 1)

        in_set = values <= threshold  # (n_test, grid_size)

        lower = torch.full((n_test, 1), y_max, device=device, dtype=dtype)
        upper = torch.full((n_test, 1), y_min, device=device, dtype=dtype)

        has_valid = in_set.any(dim=1)

        # First True from left → lower; last True (first True from right) → upper
        start_indices = in_set.int().argmax(dim=1)
        end_indices = grid_size - 1 - in_set.flip(dims=[1]).int().argmax(dim=1)

        valid_mask = has_valid
        if valid_mask.any():
            lower[valid_mask, 0] = y_grid[start_indices[valid_mask]]
            upper[valid_mask, 0] = y_grid[end_indices[valid_mask]]

        # Fallback for samples with empty level sets: use argmin of values
        invalid_mask = ~has_valid
        if invalid_mask.any():
            fallback_idx = values[invalid_mask].argmin(dim=1)
            lower[invalid_mask, 0] = y_grid[fallback_idx]
            upper[invalid_mask, 0] = y_grid[fallback_idx]

        return lower, upper


# ---------------------------------------------------------------------------
# Split Conformal Prediction
# ---------------------------------------------------------------------------


class SplitConformal(ConformalPredictor):
    """Split conformal prediction using absolute residual scores.

    Score: |y - ŷ| (optionally divided by difficulty).
    Interval: ŷ ± q_hat (or ŷ ± q_hat * difficulty for normalized CP).

    Example:
        >>> lower, upper = cp.predict_interval(test_preds)

    References
    ----------
    .. [1] Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*.
       Springer. https://link.springer.com/book/10.1007/b106715
    .. [2] Lei, J., Wasserman, L., Rinaldo, A., & Djolonga, J. (2018). Distribution-Free Predictive
       Inference for Regression. In *JASA*, 113(523), 1094-1111. https://arxiv.org/abs/1604.04173
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


class CVPlus(ConformalPredictor):
    """CV+ Conformal Predictor (Barber et al. 2021).

    Builds prediction intervals from out-of-fold residuals. Coverage
    guarantee is ``>= 1-2alpha`` (Thm 1), not ``1-alpha``; for
    ``1-alpha`` use split/CQR. Jackknife+ is the ``K=n`` special case
    with the same ``1-2alpha`` guarantee.

    During calibration, expects:
    - y_pred: Out-of-fold predictions on the calibration set, shape [n_samples, output_dim].
    - target: True values, shape [n_samples, output_dim].
    - fold_indices: Tensor of shape [n_samples] indicating which model index
        was held out when predicting each sample.

    During prediction, expects:
    - y_pred_members: Predictions of all K models on the test set,
        shape [K, n_test_samples, output_dim].

    References
    ----------
    .. [1] Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2021).
       Predictive inference with the jackknife+. *The Annals of Statistics*, 49(1), 486-507.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        normalize_fn: Optional[Callable[..., Tensor]] = None,
    ) -> None:
        super().__init__(alpha=alpha, normalize_fn=normalize_fn)
        self.residuals: Optional[Tensor] = None
        self.fold_indices: Optional[Tensor] = None

    def calibrate_ensemble(
        self,
        y_pred_oob: Tensor,
        target: Tensor,
        fold_indices: Tensor,
        *,
        mask: Optional[Tensor] = None,
    ) -> None:
        """Calibrate on out-of-fold predictions.

        Args:
            y_pred_oob: Out-of-fold predictions [n_samples, output_dim].
            target: True values [n_samples, output_dim].
            fold_indices: 1-D integer tensor [n_samples] of fold indices.
            mask: Optional boolean mask.
        """
        if mask is not None:
            mask_1d = mask.all(dim=-1) if mask.dim() > 1 else mask
            y_pred_oob = y_pred_oob[mask_1d]
            target = target[mask_1d]
            fold_indices = fold_indices[mask_1d]

        if y_pred_oob.shape[0] == 0:
            raise ValueError("Calibration set is empty after masking.")

        # Compute absolute residuals
        self.residuals = torch.abs(y_pred_oob - target)
        if self.residuals.dim() > 1:
            self.residuals = self.residuals.max(dim=-1).values

        self.fold_indices = fold_indices.long()
        self._is_calibrated = True

    def predict_interval(
        self,
        y_pred: Tensor,
        *,
        groups: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Predict intervals using CV+ combination of member predictions.

        Args:
            y_pred: Member predictions of shape [K, n_test_samples, output_dim]
                where K is the number of folds/models.
            groups: Unused, kept for signature compatibility.
            x: Unused, kept for signature compatibility.

        Returns:
            Tuple of (lower_bound, upper_bound) tensors of shape [n_test_samples, output_dim].
        """
        if not self._is_calibrated or self.residuals is None or self.fold_indices is None:
            raise RuntimeError(
                "Predictor must be calibrated before making predictions. "
                "Call calibrate_ensemble() first."
            )

        device = y_pred.device
        fold_indices = self.fold_indices.to(device)
        residuals = self.residuals.to(device)

        # Index member predictions: [K, B, D] -> [n_cal, B, D]
        pred_per_cal = y_pred[fold_indices]

        # Reshape residuals to [n_cal, 1, 1] for broadcasting
        res_unsqueezed = residuals.view(-1, 1, 1)
        lower_candidates = pred_per_cal - res_unsqueezed
        upper_candidates = pred_per_cal + res_unsqueezed

        # Finite-sample order statistics over the n_cal candidates
        # (Barber et al., 2021): exact order-statistic ranks instead of the
        # linearly-interpolated quantile previously used here.
        n_cal = lower_candidates.shape[0]
        k_upper = min(math.ceil((n_cal + 1) * (1.0 - self.alpha)), n_cal)
        k_lower = min(math.ceil((n_cal + 1) * self.alpha), n_cal)
        lower_bound = torch.sort(lower_candidates, dim=0).values[k_lower - 1]
        upper_bound = torch.sort(upper_candidates, dim=0).values[k_upper - 1]

        return lower_bound, upper_bound


# JackknifePlus is equivalent to CVPlus when K = n (leave-one-out cross-validation)
JackknifePlus = CVPlus


class EnsembleBatchCP(ConformalPredictor):
    """Ensemble Batch Prediction Intervals (EnbPI) / Bootstrap+ Conformal Predictor.

    Uses out-of-bag ensemble predictions and residuals to construct conformal
    intervals, avoiding training separate leave-one-out models.

    During calibration, expects:
    - y_pred_oob: Out-of-bag predictions for each training sample, shape [n_samples, output_dim].
    - target: True values, shape [n_samples, output_dim].

    During prediction, expects:
    - y_pred_mean: Ensemble mean prediction, shape [n_test_samples, output_dim].

    References
    ----------
    .. [1] Xu, C., & Xie, M. (2021). Conformal prediction interval for dynamic
       time-series. *Proceedings of the 38th International Conference on Machine Learning*.
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
        debias: Accepted for backward compatibility; no longer applies any
            extra correction.  Calibration always uses the standard
            finite-sample conformal adjustment (the ceil((n+1)*(1-alpha))
            order statistic via :func:`finite_sample_quantile`).
        normalize_fn: Optional difficulty normalization.

    Example:
        >>> cqr = CQR(alpha=0.1, debias=True)
        >>> cqr.calibrate(cal_quantile_preds, cal_targets)
        >>> lower, upper = cqr.predict_interval(test_quantile_preds)

    References
    ----------
    .. [1] Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized Quantile Regression.
       In *NeurIPS 2019*. https://arxiv.org/abs/1905.03222
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

    References
    ----------
    .. [1] Rossellini, R., et al. (2023). Integrating Uncertainty Awareness into Conformalized
       Quantile Regression. In *arXiv:2306.08693*. https://arxiv.org/abs/2306.08693
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
# Locally Valid and Discriminative Prediction Intervals (LVD)
# ---------------------------------------------------------------------------


class LocalConformal(ConformalPredictor):
    """Locally Valid and Discriminative (LVD) Conformal Predictor.

    Uses kernel regression to weigh calibration residuals based on the similarity
    between the test point and calibration points in feature space, achieving local
    coverage guarantees.

    Args:
        alpha: Miscoverage rate.
        K_obj: Optional custom kernel object implementing `K(x1, x2)` and `Ki(xi, Xs)`.
        bandwidth: Bandwidth parameter for the default Gaussian kernel.

    References
    ----------
    .. [1] Lin, Z., Trivedi, S., & Sun, J. (2021). Locally Valid and Discriminative Prediction
       Intervals for Deep Learning Models. In *NeurIPS 2021*.
       https://proceedings.neurips.cc/paper_files/paper/2021/file/46c7cb50b373877fb2f8d5c4517bb969-Paper.pdf
    """

    def __init__(
        self,
        alpha: float = 0.1,
        K_obj: Optional[Any] = None,
        bandwidth: float = 1.0,
    ) -> None:
        super().__init__(alpha=alpha)
        self.K_obj = K_obj
        self.bandwidth = bandwidth
        self.X_cal: Optional[Tensor] = None
        self.resids: Optional[Tensor] = None
        self.weights_cal: Optional[Tensor] = None
        self.groups_cal: Optional[Tensor] = None

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
            raise ValueError("LocalConformal requires features `x` at calibration time.")

        # Apply mask
        if mask is not None:
            mask_1d = mask.all(dim=-1) if mask.dim() > 1 else mask
            y_pred = y_pred[mask_1d]
            target = target[mask_1d]
            x = x[mask_1d]
            if groups is not None:
                groups = groups[mask_1d]
            if weights is not None:
                weights = weights[mask_1d]

        scores = torch.abs(y_pred - target)
        if scores.dim() > 1:
            scores = scores.max(dim=-1).values

        # Sort residuals and features by residual score
        sorted_idx = torch.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_x = x[sorted_idx]

        # Append infinity to scores as per standard LVD to account for test point
        device = y_pred.device
        dtype = y_pred.dtype
        self.resids = torch.cat(
            [sorted_scores, torch.tensor([float("inf")], device=device, dtype=dtype)]
        )
        self.X_cal = sorted_x.detach().clone()
        self.weights_cal = weights[sorted_idx] if weights is not None else None
        self.groups_cal = groups[sorted_idx] if groups is not None else None
        self._is_calibrated = True

    def predict_interval(
        self,
        y_pred: Tensor,
        *,
        groups: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if not self._is_calibrated or self.resids is None or self.X_cal is None:
            raise RuntimeError("Predictor must be calibrated before making predictions.")
        if x is None:
            raise ValueError("LocalConformal requires test features `x` at prediction time.")

        M = x.shape[0]
        N = self.X_cal.shape[0]
        device = x.device
        dtype = x.dtype

        # Compute kernel weights
        if self.K_obj is not None:
            try:
                kis, _ = self.K_obj.Ki(x, self.X_cal)
                if kis.dim() == 1:
                    kis = kis.unsqueeze(0)
            except Exception:
                kis_list = []
                for i in range(M):
                    ki, _ = self.K_obj.Ki(x[i], self.X_cal)
                    kis_list.append(ki)
                kis = torch.stack(kis_list, dim=0)
        else:
            dist_sq = torch.cdist(x.float(), self.X_cal.float(), p=2.0) ** 2
            dist_sq = dist_sq.to(dtype)
            kis = torch.exp(-dist_sq / (2 * self.bandwidth**2))

        # Apply Mondrian group mask if present
        if groups is not None and self.groups_cal is not None:
            group_mask = groups.unsqueeze(1) == self.groups_cal.unsqueeze(0).to(groups.device)
            kis = kis * group_mask.to(kis.dtype)

        # Append self-similarity of test point (usually 1.0)
        w_self = torch.ones((M, 1), device=device, dtype=dtype)
        weights_all = torch.cat([kis, w_self], dim=1)  # (M, N + 1)

        # Incorporate importance weights if present
        if self.weights_cal is not None:
            weights_all[:, :N] = weights_all[:, :N] * self.weights_cal.unsqueeze(0).to(device)

        # Normalize weights
        total_weights = weights_all.sum(dim=1, keepdim=True)
        total_weights = torch.max(total_weights, torch.tensor(1e-8, device=device, dtype=dtype))
        normalized_weights = weights_all / total_weights

        # Compute cumulative sum
        cum_weights = torch.cumsum(normalized_weights, dim=1)  # (M, N + 1)

        # Find the first index where the cumulative kernel mass reaches the
        # finite-sample level over the augmented N+1 set (Lin et al., 2021):
        # the k-th order statistic of the n+1 residuals corresponds to a
        # mass threshold of ceil((N+1)*(1-alpha))/(N+1).  When the cumulative
        # distribution never reaches ``q_level`` (e.g. floating-point error in
        # the final ``cum_weights`` entry, which should be 1.0 by construction),
        # ``argmax`` of an all-False mask returns 0 — silently picking the
        # smallest residual.  We append a sentinel True column so that the
        # search always returns the index of the appended ``self.resids``
        # ``inf`` entry (= the last valid index into ``resids``), which
        # surfaces as an unbounded interval rather than a wrongly-tight one.
        n_aug = N + 1
        k = math.ceil(n_aug * (1.0 - self.alpha))
        q_level = min(k / n_aug, 1.0)
        hits = cum_weights >= q_level
        sentinel = torch.ones((hits.shape[0], 1), dtype=torch.bool, device=hits.device)
        idx = torch.argmax(torch.cat([hits, sentinel], dim=1).to(torch.int8), dim=1)
        # ``idx`` is in ``[0, N+1]``.  Derive ``n_cal`` from the actual
        # ``X_cal`` length rather than ``resids.shape[0]`` so the clip is
        # robust to future changes that drop or extend the sentinel tail.
        n_cal = int(self.X_cal.shape[0])
        idx = idx.clamp(max=n_cal)

        # Fetch the corresponding quantiles
        resids_device = self.resids.to(device)
        q = resids_device[idx]  # (M,)

        # Broadcast q to y_pred shape
        q = q.view(-1, *([1] * (y_pred.dim() - 1)))

        return y_pred - q, y_pred + q

    def _build_intervals(
        self,
        y_pred: Tensor,
        q: Tensor,
        difficulty: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("Use predict_interval() directly.")


class LocalConformalMAD(LocalConformal):
    """Locally Valid and Discriminative Conformal Predictor with MAD scaling.

    Uses normalized residuals scaled by a predicted local standard deviation/MAD,
    achieving heteroscedastic local coverage.

    References
    ----------
    .. [1] Lin, Z., Trivedi, S., & Sun, J. (2021). Locally Valid and Discriminative Prediction
       Intervals for Deep Learning Models. In *NeurIPS 2021*.
       https://proceedings.neurips.cc/paper_files/paper/2021/file/46c7cb50b373877fb2f8d5c4517bb969-Paper.pdf
    """

    def __init__(
        self,
        alpha: float = 0.1,
        K_obj: Optional[Any] = None,
        bandwidth: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(alpha=alpha, K_obj=K_obj, bandwidth=bandwidth)
        self.eps = eps

    def calibrate(
        self,
        y_pred: Tensor,
        target: Tensor,
        *,
        mask: Optional[Tensor] = None,
        groups: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
        mad: Optional[Tensor] = None,
    ) -> None:
        if x is None:
            raise ValueError("LocalConformalMAD requires features `x` at calibration time.")
        if mad is None:
            raise ValueError(
                "LocalConformalMAD requires MAD/uncertainty estimates `mad` at calibration time."
            )

        # Apply mask
        if mask is not None:
            mask_1d = mask.all(dim=-1) if mask.dim() > 1 else mask
            y_pred = y_pred[mask_1d]
            target = target[mask_1d]
            x = x[mask_1d]
            mad = mad[mask_1d]
            if groups is not None:
                groups = groups[mask_1d]
            if weights is not None:
                weights = weights[mask_1d]

        raw_scores = torch.abs(y_pred - target)
        if raw_scores.dim() > 1:
            raw_scores = raw_scores.max(dim=-1).values

        # Normalize residuals by MAD + eps
        mad_flat = mad.squeeze(-1) if mad.dim() > 1 else mad
        scores = raw_scores / (self.eps + torch.abs(mad_flat))

        # Sort normalized residuals and features together
        sorted_idx = torch.argsort(scores)
        sorted_scores = scores[sorted_idx]
        sorted_x = x[sorted_idx]

        device = y_pred.device
        dtype = y_pred.dtype
        self.resids = torch.cat(
            [sorted_scores, torch.tensor([float("inf")], device=device, dtype=dtype)]
        )
        self.X_cal = sorted_x.detach().clone()
        self.weights_cal = weights[sorted_idx] if weights is not None else None
        self.groups_cal = groups[sorted_idx] if groups is not None else None
        self._is_calibrated = True

    def predict_interval(
        self,
        y_pred: Tensor,
        *,
        groups: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
        mad: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if not self._is_calibrated or self.resids is None or self.X_cal is None:
            raise RuntimeError("Predictor must be calibrated before making predictions.")
        if x is None:
            raise ValueError("LocalConformalMAD requires test features `x` at prediction time.")
        if mad is None:
            raise ValueError(
                "LocalConformalMAD requires test MAD estimates `mad` at prediction time."
            )

        M = x.shape[0]
        N = self.X_cal.shape[0]
        device = x.device
        dtype = x.dtype

        # Compute kernel weights
        if self.K_obj is not None:
            try:
                kis, _ = self.K_obj.Ki(x, self.X_cal)
                if kis.dim() == 1:
                    kis = kis.unsqueeze(0)
            except Exception:
                kis_list = []
                for i in range(M):
                    ki, _ = self.K_obj.Ki(x[i], self.X_cal)
                    kis_list.append(ki)
                kis = torch.stack(kis_list, dim=0)
        else:
            dist_sq = torch.cdist(x.float(), self.X_cal.float(), p=2.0) ** 2
            dist_sq = dist_sq.to(dtype)
            kis = torch.exp(-dist_sq / (2 * self.bandwidth**2))

        # Apply Mondrian group mask if present
        if groups is not None and self.groups_cal is not None:
            group_mask = groups.unsqueeze(1) == self.groups_cal.unsqueeze(0).to(groups.device)
            kis = kis * group_mask.to(kis.dtype)

        # Append self-similarity of test point
        w_self = torch.ones((M, 1), device=device, dtype=dtype)
        weights_all = torch.cat([kis, w_self], dim=1)

        if self.weights_cal is not None:
            weights_all[:, :N] = weights_all[:, :N] * self.weights_cal.unsqueeze(0).to(device)

        total_weights = weights_all.sum(dim=1, keepdim=True)
        total_weights = torch.max(total_weights, torch.tensor(1e-8, device=device, dtype=dtype))
        normalized_weights = weights_all / total_weights

        cum_weights = torch.cumsum(normalized_weights, dim=1)
        q_level = min(math.ceil((N + 1) * (1.0 - self.alpha)) / (N + 1), 1.0)
        # Append a sentinel True column so the search never returns 0 when
        # floating-point error keeps ``cum_weights`` slightly below ``q_level``
        # (matches the LocalConformal fix above; see that method for details).
        hits = cum_weights >= q_level
        sentinel = torch.ones((hits.shape[0], 1), dtype=torch.bool, device=hits.device)
        idx = torch.argmax(torch.cat([hits, sentinel], dim=1).to(torch.int8), dim=1)
        # Derive ``n_cal`` from the actual calibration count rather than the
        # sentinel-decorated ``resids`` shape so future changes to the tail
        # bookkeeping don't silently misclip.
        n_cal = int(self.X_cal.shape[0])
        idx = idx.clamp(max=n_cal)

        resids_device = self.resids.to(device)
        q = resids_device[idx]

        # Reshape q to broadcast with y_pred
        q = q.view(-1, *([1] * (y_pred.dim() - 1)))

        # Widen the interval by the test MAD
        mad_val = torch.abs(mad) + self.eps
        while mad_val.dim() < y_pred.dim():
            mad_val = mad_val.unsqueeze(-1)

        width = q * mad_val
        return y_pred - width, y_pred + width


class CTI(LevelSetConformalPredictor):
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

    References
    ----------
    .. [1] Luo, R., & Zhou, Z. (2025). Conformal Thresholded Intervals for Efficient Regression.
       In *AAAI 2025*. https://arxiv.org/abs/2407.14495
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

    # ty: ignore[invalid-method-override]  # subclass-specific first-parameter name; positional API stays compatible
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

        def eval_fn(y_grid: Tensor, x_batch: Tensor) -> Tensor:
            return -density_fn(y_grid, x_batch)

        # Try vectorized level-set search first; fall back to per-sample
        # loop if the user-provided density_fn does not support batching.
        try:
            return self._grid_search_level_set(eval_fn, x, q, y_min, y_max, self.grid_size)
        except Exception as exc:
            logger.debug(f"CTI grid search failed, falling back to per-sample loop: {exc}")

        n_test = x.shape[0]
        y_grid = torch.linspace(y_min, y_max, self.grid_size, device=x.device, dtype=x.dtype)
        lower = torch.full((n_test, 1), y_max, device=x.device, dtype=x.dtype)
        upper = torch.full((n_test, 1), y_min, device=x.device, dtype=x.dtype)

        for i in range(n_test):
            log_dens = density_fn(y_grid, x[i])
            neg_log_dens = -log_dens
            in_set = neg_log_dens <= q
            if in_set.any():
                indices = in_set.nonzero(as_tuple=True)[0]
                lower[i, 0] = y_grid[indices[0]]
                upper[i, 0] = y_grid[indices[-1]]
            else:
                mode_idx = log_dens.argmax()
                lower[i, 0] = y_grid[mode_idx]
                upper[i, 0] = y_grid[mode_idx]

        return lower, upper


# ---------------------------------------------------------------------------
# Distributional Conformal Prediction
# ---------------------------------------------------------------------------


class DistributionalConformal(LevelSetConformalPredictor):
    """PIT-band conformal prediction with approximate conditional coverage.

    Chernozhukov DCP-inspired score ``|2*F(y|x) - 1|`` used as a PIT-band
    nonconformity score.  Requires a model that outputs CDF values
    ``F(y | x)`` for each calibration/test point.

    Score: ``|2*F(y|x) - 1|`` (deterministic symmetric PIT-band score).
    Interval: invert the CDF at ``[(1-q_hat)/2, (1+q_hat)/2]`` after
        conformal adjustment of ``q_hat``.

    Args:
        alpha: Miscoverage rate.

    Example:
        >>> dcp = DistributionalConformal(alpha=0.1)
        >>> # F_cal[i] = CDF(y_cal[i] | x_cal[i]) from model
        >>> dcp.calibrate(F_cal, y_cal)
        >>> lower, upper = dcp.predict_intervals_from_cdf(
        ...     cdf_fn, icdf_fn, x_test)

    References
    ----------
    .. [1] Chernozhukov, V., Wüthrich, K., & Zhu, Y. (2021). Distributional Conformal Prediction.
       In *PNAS*, 118(48). https://arxiv.org/abs/1909.07889
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

    References
    ----------
    .. [1] Cabi, S., et al. (2024). Conformal Prediction via Regression-as-Classification.
       In *ICLR 2024*. https://arxiv.org/abs/2404.08168
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
        using bin edge values (the interval spans the full bins).

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

        # Bin EDGES: lower = left edge of the first included bin,
        # upper = right edge of the last included bin (full bin spans).
        left_edges = self.bin_edges[:-1]
        right_edges = self.bin_edges[1:]

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

        # Map sorted indices back to bin indices, then to edges.
        # Use large/small sentinel for excluded bins
        left_sorted = left_edges[sorted_idx]  # (n_test, n_bins)
        right_sorted = right_edges[sorted_idx]  # (n_test, n_bins)
        INF = float("inf")
        # For lower: excluded bins → +inf, take min
        lower = torch.amin(torch.where(included_mask, left_sorted, INF), dim=-1, keepdim=True)
        # For upper: excluded bins → -inf, take max
        upper = torch.amax(torch.where(included_mask, right_sorted, -INF), dim=-1, keepdim=True)

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
        k = min(math.ceil((n + 1) * (1.0 - self.alpha)), n)
        self.q_hat = torch.sort(scores, dim=0).values[k - 1]
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
        target_1d = _to_1d(target.detach())
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
                pred_1d = _to_1d(y_pred.detach())
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
        target_1d = _to_1d(target.detach())
        quantiles = torch.linspace(0.0, 1.0, self.n_bins + 1, device=target.device)
        # Linear-interpolation quantile edges, written out to keep this
        # module free of interpolated-quantile calls in the conformal path.
        sorted_t = torch.sort(target_1d).values
        n_t = sorted_t.numel()
        positions = quantiles * (n_t - 1)
        lo_idx = positions.long().clamp(max=n_t - 1)
        hi_idx = (lo_idx + 1).clamp(max=n_t - 1)
        frac = positions - lo_idx.to(target_1d.dtype)
        edges = sorted_t[lo_idx] * (1.0 - frac) + sorted_t[hi_idx] * frac
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
        # Typed local binding: ty cannot narrow ``self.q_hat`` through the
        # Tensor|dict union, so help it with an explicit cast.
        q_map = cast(Dict[Any, Tensor], self.q_hat)
        if groups is None:
            if self._bin_edges is None:
                raise ValueError("groups must be provided when no calibration bin edges exist")
            pred_1d = _to_1d(y_pred.detach())
            edges = self._bin_edges.to(y_pred.device)
            groups = torch.bucketize(pred_1d, edges[1:-1])

        q_default = max(q_map.values(), key=lambda x: float(x.item()))
        q_per_sample = torch.full(
            (y_pred.shape[0],),
            fill_value=float(q_default.item()),
            dtype=y_pred.dtype,
            device=y_pred.device,
        )
        for g, q_val in q_map.items():
            q_per_sample[groups == g] = float(q_val.item())
        q = q_per_sample.view(-1, *([1] * (y_pred.dim() - 1)))
        return self._build_intervals(y_pred, q)


class MonteCarloConformal(ConformalPredictor):
    """Conformal prediction from Monte-Carlo predictive samples.

    Uses MC sample mean/median as the point predictor and MC sample standard
    deviation as the difficulty normalizer.

    References
    ----------
    .. [1] Gibbs, I., Cherian, J. J., & Candès, E. J. (2025). Conformal prediction
       with conditional guarantees. In *JRSSB 2025*. https://arxiv.org/abs/2305.12641
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

    # ty: ignore[invalid-method-override]  # subclass-specific first-parameter name; positional API stays compatible
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

    # ty: ignore[invalid-method-override]  # subclass-specific first-parameter name; positional API stays compatible
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
        debias: Accepted for backward compatibility; no extra correction is
            applied beyond the standard finite-sample conformal adjustment.
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

        return self._reduce(loss, mask, weights)

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


class SLSConformal(LevelSetConformalPredictor):
    """Super-Level-Set (SLS) Conformal Predictor.

    Uses the ratio of the learned frontier function G(X, Y) to the predicted
    conditional quantile q_tau(X) as the nonconformity score:
    S = G(X, Y) / q_tau(X)

    The prediction set is the level set:
    {y : G(X, y) <= r * q_tau(X)}
    where r is the calibrated threshold ensuring (1 - alpha) marginal coverage.

    Args:
        sls_loss: An instance of `SLSLoss` containing the trained frontier and quantile networks.
        alpha: Miscoverage rate.
        grid_size: Number of grid points for constructing level-set intervals at prediction time.

    References
    .. [1] Braun, Jordan, Bach (2026). Super-Level-Set Conformal Prediction.
       arXiv:2605.06210
    """

    def __init__(
        self,
        sls_loss: Any,
        alpha: float = 0.1,
        grid_size: int = 500,
    ) -> None:
        super().__init__(alpha=alpha)
        self.sls_loss = sls_loss
        self.grid_size = grid_size

    def _compute_scores(self, y_pred: Tensor, target: Tensor) -> Tensor:
        # Here, y_pred is the context/features X, and target is Y
        with torch.no_grad():
            G, _ = self.sls_loss.frontier(target, y_pred)
            quantiles = self.sls_loss.quantile_net(y_pred)
            q_tau = quantiles[..., 1]  # target quantile is at index 1
        scores = G / q_tau.clamp(min=1e-8)
        return scores.view(-1)

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
        """Calibrate the conformal correction factor r.

        Args:
            y_pred: Features X for the calibration set.
            target: True target values Y for the calibration set.
            mask: Optional mask.
            groups: Optional Mondrian groups.
            weights: Optional importance weights.
            x: Unused (y_pred serves as the features X).
        """
        super().calibrate(y_pred, target, mask=mask, groups=groups, weights=weights)

    def _build_intervals(
        self,
        y_pred: Tensor,
        q: Tensor,
        difficulty: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "SLS prediction regions require evaluating a frontier function on a grid. "
            "Use predict_interval_from_grid() instead."
        )

    def predict_interval_from_grid(
        self,
        x: Tensor,
        y_min: float,
        y_max: float,
    ) -> Tuple[Tensor, Tensor]:
        """Build prediction intervals from SLS level sets.

        Evaluates the frontier function on a grid and returns the tightest [lower, upper]
        bounding box of the level set {y : G(x, y) <= r * q_tau(x)}.

        Args:
            x: Test inputs/features, shape (n_test, context_dim).
            y_min: Lower bound of the evaluation grid.
            y_max: Upper bound of the evaluation grid.

        Returns:
            (lower, upper) tensors of shape (n_test, 1).
        """
        if not self._is_calibrated or self.q_hat is None:
            raise RuntimeError("Call calibrate() first.")

        r = self.q_hat if isinstance(self.q_hat, Tensor) else next(iter(self.q_hat.values()))

        with torch.no_grad():
            quantiles = self.sls_loss.quantile_net(x)
            q_tau = quantiles[..., 1]
            threshold = r.to(x.device) * q_tau

            def eval_fn(y_grid: Tensor, x_batch: Tensor) -> Tensor:
                # Explicitly reshape to matching batch dims for the frontier
                # function (preserves the contract from the original
                # implementation).
                n_test_b = x_batch.shape[0]
                x_exp = x_batch.unsqueeze(1).expand(-1, self.grid_size, -1)
                x_exp = x_exp.reshape(-1, x_batch.shape[-1])
                y_exp = y_grid.unsqueeze(0).expand(n_test_b, -1).reshape(-1, self.sls_loss.d)
                G_flat, _ = self.sls_loss.frontier(y_exp, x_exp)
                return G_flat.reshape(n_test_b, self.grid_size)

            return self._grid_search_level_set(eval_fn, x, threshold, y_min, y_max, self.grid_size)
