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

import math
from typing import Any, Dict, Literal, Optional, Union, cast

import torch

from torchregress.losses.conformal import _weighted_quantile

ScoreMode = Literal["classification"]
Objective = Literal["weighted_cdf"]
WeightParameterization = Literal["free"]


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
        score_mode: ScoreMode = "classification",
        objective: Objective = "weighted_cdf",
        weight_parameterization: WeightParameterization = "free",
        entropy_penalty: float = 1e-3,
        n_grid: int = 129,
        n_steps: int = 200,
        learning_rate: float = 0.05,
    ) -> None:
        if score_mode != "classification":
            raise ValueError(f'score_mode must be "classification" in v1, got {score_mode!r}')
        if objective != "weighted_cdf":
            raise ValueError(f'objective must be "weighted_cdf" in v1, got {objective!r}')
        if weight_parameterization != "free":
            raise ValueError(
                f'weight_parameterization must be "free" in v1, got {weight_parameterization!r}'
            )
        if entropy_penalty < 0:
            raise ValueError("entropy_penalty must be non-negative")
        self.score_mode = score_mode
        self.objective = objective
        self.weight_parameterization = weight_parameterization
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
        n = scores.numel()
        q_level = min(math.ceil((n + 1) * (1.0 - self.alpha)) / n, 1.0)
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
