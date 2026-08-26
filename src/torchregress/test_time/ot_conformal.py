"""
Shift-aware **classification-style** conformal utilities using score CDF matching.

v1 learns nonnegative calibration weights on a simplex by aligning a **weighted**
empirical CDF of calibration nonconformity scores to the empirical CDF of unlabeled
target scores on a 1-D grid, with an entropy regulariser. This is a lightweight
surrogate for OT reweighting ideas from non-exchangeable conformal work; it does
**not** call an external optimal-transport solver.

See ``docs/test_time/ot_shift_conformal.md`` for assumptions and limitations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union, cast

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression  # type: ignore

from torchregress.losses.conformal import _weighted_quantile
from torchregress.prediction import PredictiveBatch


def _as_1d_scores(x: Union[torch.Tensor, Any], *, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    t = cast(torch.Tensor, x).reshape(-1).float()
    if t.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    return t


def _normalize_simplex(weights: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    w = weights.clamp(min=0.0)
    s = w.sum().clamp(min=eps)
    return w / s


def _weighted_ecdf_on_grid(
    scores: torch.Tensor,
    weights: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Weighted ECDF evaluated at each grid point; ``scores`` and ``weights`` are 1-D [n]."""
    w = _normalize_simplex(weights)
    le = scores.unsqueeze(-1) <= grid.unsqueeze(0)
    return (le.float() * w.unsqueeze(-1)).sum(dim=0)


def _uniform_ecdf_on_grid(scores: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    n = scores.shape[0]
    w = torch.full((n,), 1.0 / n, device=scores.device, dtype=scores.dtype)
    return _weighted_ecdf_on_grid(scores, w, grid)


def _effective_sample_size_inv_square(weights: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    w = _normalize_simplex(weights, eps=eps)
    return 1.0 / (w * w).sum().clamp(min=eps)


class OptimalTransportCoverageGap:
    """
    Diagnostics for score CDF mismatch between calibration and target samples.

    Computes a coarse ``L_2`` gap between uniform-weight calibration and target
    empirical CDFs on a shared 1-D grid, plus a simple Kolmogorov--Smirnov-style
    maximum absolute deviation proxy.
    """

    def __init__(self, n_grid: int = 129) -> None:
        if n_grid < 8:
            raise ValueError("n_grid must be at least 8")
        self.n_grid = n_grid

    def estimate(
        self,
        *,
        calibration_scores: torch.Tensor,
        target_score_summary: torch.Tensor,
    ) -> Dict[str, Any]:
        s_cal = _as_1d_scores(calibration_scores, name="calibration_scores")
        s_tgt = _as_1d_scores(target_score_summary, name="target_score_summary")
        lo = torch.minimum(s_cal.min(), s_tgt.min())
        hi = torch.maximum(s_cal.max(), s_tgt.max())
        if bool(hi <= lo):
            hi = lo + 1.0
        grid = torch.linspace(lo, hi, self.n_grid, device=s_cal.device, dtype=s_cal.dtype)
        f_cal = _uniform_ecdf_on_grid(s_cal, grid)
        f_tgt = _uniform_ecdf_on_grid(s_tgt, grid)
        diff = f_cal - f_tgt
        l2 = torch.sqrt((diff * diff).mean() + 1e-18)
        ks = torch.max(torch.abs(diff))
        return {
            "l2_cdf_gap": float(l2.item()),
            "ks_max_abs": float(ks.item()),
            "n_calibration": int(s_cal.numel()),
            "n_target": int(s_tgt.numel()),
        }


class ScoreCDFReweighter:
    """
    Learn simplex weights over calibration points by CDF matching on a 1-D score grid.

    Args:
        score_mode: Only ``\"classification\"`` is supported in v1 (1-D nonconformity scores).
        objective: Only ``\"weighted_cdf\"`` is implemented.
        weight_parameterization: Only ``\"free\"`` (softmax over unconstrained logits).
        entropy_penalty: Non-negative multiplier on :math:`\\sum_i w_i \\log w_i`
            (encourages higher entropy / more uniform weights when the CDF term alone
            would concentrate mass).
        n_grid: Number of grid points for the CDF ``L_2`` term.
        n_steps: Adam steps for the inner optimisation.
        learning_rate: Adam learning rate.
    """

    def __init__(
        self,
        *,
        entropy_penalty: float = 1e-3,
        n_grid: int = 129,
        n_steps: int = 200,
        learning_rate: float = 0.05,
    ) -> None:
        if entropy_penalty < 0:
            raise ValueError("entropy_penalty must be non-negative")
        self.entropy_penalty = float(entropy_penalty)
        self.n_grid = int(n_grid)
        self.n_steps = int(n_steps)
        self.learning_rate = float(learning_rate)

        self.weights_: Optional[torch.Tensor] = None
        self.objective_value_: Optional[float] = None
        self.diagnostics_: Dict[str, Any] = {}

    def fit(
        self,
        calibration_scores: torch.Tensor,
        target_unlabeled_scores: torch.Tensor,
    ) -> ScoreCDFReweighter:
        s_cal = _as_1d_scores(calibration_scores, name="calibration_scores")
        s_tgt = _as_1d_scores(target_unlabeled_scores, name="target_unlabeled_scores")
        lo = torch.minimum(s_cal.min(), s_tgt.min())
        hi = torch.maximum(s_cal.max(), s_tgt.max())
        if bool(hi <= lo):
            hi = lo + 1.0
        grid = torch.linspace(lo, hi, self.n_grid, device=s_cal.device, dtype=s_cal.dtype)
        f_tgt = _uniform_ecdf_on_grid(s_tgt, grid)

        n = s_cal.shape[0]
        raw = torch.zeros(n, device=s_cal.device, dtype=s_cal.dtype, requires_grad=True)
        opt = torch.optim.Adam([raw], lr=self.learning_rate)
        last_loss = torch.zeros((), device=s_cal.device, dtype=s_cal.dtype)
        for _ in range(self.n_steps):
            opt.zero_grad(set_to_none=True)
            w = torch.softmax(raw, dim=0)
            f_w = _weighted_ecdf_on_grid(s_cal, w, grid)
            loss_cdf = ((f_w - f_tgt) ** 2).mean()
            loss_ent = (w * (w.clamp(min=1e-12).log())).sum()
            loss = loss_cdf + self.entropy_penalty * loss_ent
            last_loss = loss
            loss.backward()
            opt.step()

        with torch.no_grad():
            w_final = torch.softmax(raw, dim=0)
            f_final = _weighted_ecdf_on_grid(s_cal, w_final, grid)
        self.weights_ = w_final.detach().clone()
        self.objective_value_ = float(last_loss.detach().item())
        ess = _effective_sample_size_inv_square(self.weights_)
        self.diagnostics_ = {
            "ess_inv_square": float(ess.item()),
            "cdf_l2_on_grid": float(((f_final - f_tgt) ** 2).mean().sqrt().item()),
        }
        return self


class WeightedSplitConformalAdapter:
    """
    Weighted split-conformal threshold for 1-D nonconformity scores (classification).

    Uses the same finite-sample quantile adjustment as unweighted split conformal
    when weights are uniform; for general nonnegative weights, delegates to
    :func:`torchregress.losses.conformal._weighted_quantile`.

    ``predict_from_test_scores`` expects per-example, per-class nonconformity scores
    ``[batch, n_classes]`` and returns a boolean mask of included labels.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.threshold_: Optional[torch.Tensor] = None

    def calibrate(
        self,
        calibration_scores: torch.Tensor,
        calibration_weights: torch.Tensor,
    ) -> WeightedSplitConformalAdapter:
        scores = _as_1d_scores(calibration_scores, name="calibration_scores")
        weights = _as_1d_scores(calibration_weights, name="calibration_weights")
        if weights.shape != scores.shape:
            raise ValueError("calibration_weights must match calibration_scores shape")
        # The (n+1) test-point mass is supplied by ``_weighted_quantile``'s
        # augmented distribution; pass the raw target level (TR-COR-05).
        q_level = 1.0 - self.alpha
        self.threshold_ = _weighted_quantile(scores, q_level, weights)
        return self

    def predict_from_test_scores(self, candidate_scores: torch.Tensor) -> torch.Tensor:
        if self.threshold_ is None:
            raise RuntimeError("Call calibrate before predict_from_test_scores.")
        if candidate_scores.dim() != 2:
            raise ValueError("candidate_scores must be 2-D [batch, n_classes]")
        return candidate_scores <= self.threshold_

    @torch.no_grad()
    def coverage_diagnostics(
        self,
        calibration_scores: torch.Tensor,
        calibration_weights: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Summarise calibration-set behaviour relative to the fitted weighted quantile threshold.

        Returns the **weighted empirical fraction** of calibration scores at or below
        ``threshold_`` (the split-conformal cutoff), ``nominal_coverage`` ``≈ 1 - alpha``,
        and their difference. Under exchangeability and correct weighting, the weighted
        fraction should be close to nominal for large ``n``.
        """
        if self.threshold_ is None:
            raise RuntimeError("Call calibrate before coverage_diagnostics.")
        scores = _as_1d_scores(calibration_scores, name="calibration_scores")
        weights = _as_1d_scores(calibration_weights, name="calibration_weights")
        if weights.shape != scores.shape:
            raise ValueError("calibration_weights must match calibration_scores shape")
        w = _normalize_simplex(weights)
        thr = self.threshold_
        if not torch.is_tensor(thr):
            thr_t = torch.as_tensor(thr, device=scores.device, dtype=scores.dtype)
        else:
            thr_t = thr.to(device=scores.device, dtype=scores.dtype)
        covered = (scores <= thr_t).float()
        weighted_frac = float((w * covered).sum().item())
        nominal = 1.0 - self.alpha
        ess = _effective_sample_size_inv_square(weights)
        return {
            "alpha": float(self.alpha),
            "threshold": float(thr_t.item()),
            "n_calibration": int(scores.numel()),
            "nominal_coverage": float(nominal),
            "weighted_empirical_coverage": weighted_frac,
            "coverage_gap": weighted_frac - nominal,
            "calibration_ess_inv_square": float(ess.item()),
        }


class WeightedConformalRegressionAdapter:
    """
    Weighted split-conformal regression adapter using classifier-based density ratio estimation.

    Estimates the covariate shift density ratio w(x) = p_target(x) / p_source(x)
    by training a classifier (defaulting to Logistic Regression) to distinguish
    between calibration and target features.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        classifier: Any | None = None,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)

        if classifier is None:
            classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.classifier = classifier
        self.scores_: torch.Tensor | None = None
        self.w_cal_: torch.Tensor | None = None
        self.X_cal_: torch.Tensor | None = None

    def fit_density_ratio(
        self,
        X_cal: torch.Tensor | np.ndarray,
        X_tgt: torch.Tensor | np.ndarray,
    ) -> WeightedConformalRegressionAdapter:
        """
        Fit the classifier to estimate the density ratio w(x) = p_tgt(x) / p_cal(x).
        """
        if torch.is_tensor(X_cal):
            X_cal_np = X_cal.detach().cpu().numpy()
        else:
            X_cal_np = np.asarray(X_cal)

        if torch.is_tensor(X_tgt):
            X_tgt_np = X_tgt.detach().cpu().numpy()
        else:
            X_tgt_np = np.asarray(X_tgt)

        n_cal = X_cal_np.shape[0]
        n_tgt = X_tgt_np.shape[0]

        X_comb = np.concatenate([X_cal_np, X_tgt_np], axis=0)
        y_comb = np.concatenate([np.zeros(n_cal), np.ones(n_tgt)], axis=0)

        self.classifier.fit(X_comb, y_comb)
        return self

    def compute_density_ratios(self, X: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Compute density ratio weights for given features.
        """
        if torch.is_tensor(X):
            X_np = X.detach().cpu().numpy()
            device = X.device
        else:
            X_np = np.asarray(X)
            device = torch.device("cpu")

        probs = self.classifier.predict_proba(X_np)[:, 1]
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        w = probs / (1.0 - probs)
        return torch.as_tensor(w, dtype=torch.float32, device=device)

    def calibrate(
        self,
        y_pred_cal: torch.Tensor | np.ndarray | PredictiveBatch | dict[str, Any],
        y_cal: torch.Tensor | np.ndarray,
        X_cal: torch.Tensor | np.ndarray,
        X_tgt: torch.Tensor | np.ndarray,
    ) -> WeightedConformalRegressionAdapter:
        """
        Calibrate the adapter on held-out calibration data.
        """
        self.fit_density_ratio(X_cal, X_tgt)

        if not torch.is_tensor(y_cal):
            y_cal_t = torch.as_tensor(y_cal, dtype=torch.float32)
        else:
            y_cal_t = y_cal

        std_cal = None
        if isinstance(y_pred_cal, PredictiveBatch):
            mean_cal = y_pred_cal.mean if y_pred_cal.mean is not None else y_pred_cal.point
            std_cal = y_pred_cal.std
        elif isinstance(y_pred_cal, dict):
            mean_cal = y_pred_cal.get("mean")
            if mean_cal is None:
                mean_cal = y_pred_cal.get("point")
            std_cal = y_pred_cal.get("std")
        else:
            mean_cal = y_pred_cal

        if not torch.is_tensor(mean_cal):
            mean_cal_t = torch.as_tensor(mean_cal, dtype=torch.float32)
        else:
            mean_cal_t = mean_cal

        std_cal_t: torch.Tensor | None = None
        if std_cal is not None:
            std_cal_t = torch.as_tensor(std_cal, dtype=torch.float32)

        # Align shapes
        if y_cal_t.dim() == 1 and mean_cal_t.dim() > 1:
            y_cal_t = y_cal_t.view_as(mean_cal_t)
        elif mean_cal_t.dim() == 1 and y_cal_t.dim() > 1:
            mean_cal_t = mean_cal_t.view_as(y_cal_t)

        scores = torch.abs(y_cal_t - mean_cal_t)
        if std_cal_t is not None:
            if std_cal_t.dim() == 1 and scores.dim() > 1:
                std_cal_t = std_cal_t.view_as(scores)
            elif scores.dim() == 1 and std_cal_t.dim() > 1:
                scores = scores.view_as(std_cal_t)
            scores = scores / torch.clamp(std_cal_t, min=1e-8)

        if scores.dim() > 1:
            scores = scores.max(dim=-1).values

        self.scores_ = scores.reshape(-1)
        self.w_cal_ = self.compute_density_ratios(X_cal)
        if torch.is_tensor(X_cal):
            self.X_cal_ = X_cal
        else:
            self.X_cal_ = torch.as_tensor(X_cal, dtype=torch.float32)

        return self

    def predict_interval(
        self,
        y_pred: torch.Tensor | np.ndarray | PredictiveBatch | dict[str, Any],
        X: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray]:
        """
        Compute weighted split conformal prediction intervals for test features X.
        """
        if self.scores_ is None or self.w_cal_ is None:
            raise RuntimeError("Adapter must be calibrated before calling predict_interval.")

        X_t = torch.as_tensor(X)

        device = X_t.device
        dtype = X_t.dtype

        w_test = self.compute_density_ratios(X_t).to(device=device, dtype=dtype)
        w_cal = self.w_cal_.to(device=device, dtype=dtype)
        scores = self.scores_.to(device=device, dtype=dtype)

        std_test = None
        if isinstance(y_pred, PredictiveBatch):
            mean_test = y_pred.mean if y_pred.mean is not None else y_pred.point
            std_test = y_pred.std
        elif isinstance(y_pred, dict):
            mean_test = y_pred.get("mean")
            if mean_test is None:
                mean_test = y_pred.get("point")
            std_test = y_pred.get("std")
        elif torch.is_tensor(y_pred):
            mean_test = y_pred
        elif isinstance(y_pred, np.ndarray):
            mean_test = torch.as_tensor(y_pred, dtype=dtype, device=device)
        else:
            raise TypeError("Unsupported prediction type.")

        if not torch.is_tensor(mean_test):
            mean_test = torch.as_tensor(mean_test, dtype=dtype, device=device)
        if std_test is not None and not torch.is_tensor(std_test):
            std_test = torch.as_tensor(std_test, dtype=dtype, device=device)

        sorted_idx = torch.argsort(scores)
        scores_sorted = scores[sorted_idx]
        w_cal_sorted = w_cal[sorted_idx]

        c_cum = torch.cumsum(w_cal_sorted, dim=0)
        s_w = w_cal_sorted.sum()

        targets = (1.0 - self.alpha) * (s_w + w_test)
        idx = torch.searchsorted(c_cum, targets)

        n_cal = scores.shape[0]
        idx_clamped = torch.clamp(idx, max=n_cal - 1)
        q_hat = scores_sorted[idx_clamped]

        if mean_test.dim() > 1:
            q_hat_broadcast = q_hat.view(-1, 1)
        else:
            q_hat_broadcast = q_hat

        if std_test is not None:
            if std_test.dim() > 1:
                width = q_hat_broadcast * std_test
            else:
                if mean_test.dim() > 1:
                    width = q_hat_broadcast * std_test.view(-1, 1)
                else:
                    width = q_hat_broadcast * std_test
        else:
            width = q_hat_broadcast

        lower_bound = mean_test - width
        upper_bound = mean_test + width

        return lower_bound, upper_bound
