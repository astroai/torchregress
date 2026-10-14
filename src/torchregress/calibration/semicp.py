"""Semi-supervised and weighted conformal prediction calibration under shift."""

from __future__ import annotations

import math
from typing import Union

import torch

from torchregress.losses.conformal import finite_sample_quantile


class SemiConformalCalibrator:
    """Semi-supervised and weighted conformal calibration under covariate/label shift.

    Supports weighted split conformal prediction and SemiCP-style calibration using
    unlabeled target samples to estimate target-weighted nonconformity score thresholds.

    References
    ----------
    .. [1] Tibshirani, R. J., Foygel Barber, R., Candes, E., & Ramdas, A. (2019).
       Conformal Prediction Under Covariate Shift. In *NeurIPS 2019*.
    .. [2] Cauchois, M., Gupta, S., & Duchi, J. C. (2020). Knowing what you don't know:
       Unbiased calibration of conformal prediction. *arXiv preprint arXiv:2005.21147*.
    """

    def __init__(
        self,
        *,
        eps: float = 1e-8,
    ) -> None:
        self.eps = eps
        self.scores_cal_: torch.Tensor | None = None
        self.weights_cal_: torch.Tensor | None = None

    @staticmethod
    def _to_tensor(array: Union[torch.Tensor, float, int]) -> torch.Tensor:
        if isinstance(array, torch.Tensor):
            return array
        return torch.tensor(array, dtype=torch.float)

    def fit(
        self,
        nonconformity_scores_cal: Union[torch.Tensor],
        weights_cal: Union[torch.Tensor] | None = None,
    ) -> "SemiConformalCalibrator":
        """Fit the calibrator on calibration nonconformity scores and optional weights.

        Parameters
        ----------
        nonconformity_scores_cal : torch.Tensor
            Calibration set nonconformity scores, shape (N_cal,).
        weights_cal : torch.Tensor, optional
            Weights for calibration samples (e.g. prior ratio), shape (N_cal,).

        Notes
        -----
        Scores and weights are stored in float64 so that the cumulative
        weighted CDF selects the exact order statistic.
        """
        scores = self._to_tensor(nonconformity_scores_cal).reshape(-1).to(torch.float64)
        if weights_cal is not None:
            weights = self._to_tensor(weights_cal).reshape(-1).to(torch.float64)
            if weights.shape[0] != scores.shape[0]:
                raise ValueError("weights_cal must share shape with nonconformity_scores_cal")
            if torch.any(weights < 0.0):
                raise ValueError("weights_cal must be non-negative")
        else:
            weights = torch.ones_like(scores)

        # Sort scores and weights in ascending order of scores
        sort_idx = torch.argsort(scores)
        self.scores_cal_ = scores[sort_idx]
        self.weights_cal_ = weights[sort_idx]
        return self

    def compute_thresholds(
        self,
        weights_target: Union[torch.Tensor, float],
        alpha: float = 0.1,
    ) -> Union[torch.Tensor, float]:
        """Compute sample-specific conformal thresholds for target points.

        Parameters
        ----------
        weights_target : Union[torch.Tensor, float]
            Shift weights w(x) = p_target(x)/p_source(x) for target points, shape (N_target,).
        alpha : float
            Nominal coverage level is 1 - alpha (e.g. alpha = 0.1 for 90% coverage).
            Thresholds evaluate the finite-sample level ``ceil((n+1)*(1-alpha))/(n+1)``
            ONCE on the augmented empirical distribution ``sum_i p_i delta_{S_i} +
            w_target / (sum_j w_j + w_target) * delta_inf`` whose normalization
            denominator already contains the target pseudo-weight; the (n+1)/n
            inflation is not applied a second time (exact order statistic in the
            unweighted, zero-target-weight limit).

        Returns
        -------
        Union[torch.Tensor, float]
            Thresholds (float64), one per target point.  A threshold is
            ``+inf`` when the calibration mass cannot reach the level (the
            quantile falls on the target-point atom), e.g. with too few
            calibration points for the requested ``alpha`` or a large target
            weight; the corresponding interval is then infinite, as the
            finite-sample guarantee requires.
        """
        if self.scores_cal_ is None or self.weights_cal_ is None:
            raise RuntimeError("Calibrator must be fitted before computing thresholds")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")

        # Convert inputs to tensor
        if isinstance(weights_target, (float, int)):
            if weights_target < 0.0:
                raise ValueError("weights_target must be non-negative")
            w_tgt = torch.tensor([float(weights_target)], dtype=torch.float64)
        else:
            w_tgt = self._to_tensor(weights_target).reshape(-1).to(torch.float64)
            if torch.any(w_tgt < 0.0):
                raise ValueError("weights_target must be non-negative")

        device = w_tgt.device
        # float64 throughout: a float32 cumulative sum misses the exact order
        # statistic by one in ~1% of (n, alpha) combinations.
        scores = self.scores_cal_.to(device=device, dtype=torch.float64)
        weights = self.weights_cal_.to(device=device, dtype=torch.float64)
        n_cal = scores.shape[0]

        # Finite-sample correction (TR-COR-06): the (n+1) adjustment enters
        # exactly once -- through the target pseudo-mass already added to the
        # denominator -- so the level on the augmented distribution is
        # ceil((n+1)*(1-alpha))/(n+1), not the doubly-inflated
        # ceil((n+1)*(1-alpha))/n.
        level = math.ceil((n_cal + 1) * (1.0 - alpha)) / (n_cal + 1)
        uniform_weights = bool(torch.all(weights == weights[0]))

        # Weighted path (Tibshirani et al., 2019), vectorised over targets:
        # smallest score whose cumulative mass on the augmented distribution
        # reaches the level, i.e. the first index with
        # cum_w >= level * (sum_j w_j + w_target).  A relative tolerance
        # absorbs float64 round-off at exact hits; when the calibration mass
        # never reaches the level, the quantile is the test-point atom: +inf.
        cum_w = torch.cumsum(weights, dim=0)
        denom = cum_w[-1] + w_tgt
        need = level * denom
        need = need - 1e-12 * need.abs()
        idx = torch.searchsorted(cum_w, need)
        thresholds = torch.full((w_tgt.shape[0],), float("inf"), dtype=torch.float64, device=device)
        hit = (idx < n_cal) & (denom > 0.0)
        thresholds[hit] = scores[idx[hit]]

        # Unweighted path, zero target weight: exact finite-sample
        # split-conformal order statistic (+inf when ceil((n+1)(1-alpha)) > n).
        if uniform_weights and float(weights[0]) > 0.0:
            zero_tgt = w_tgt == 0.0
            if bool(zero_tgt.any()):
                thresholds[zero_tgt] = finite_sample_quantile(scores, alpha).to(torch.float64)

        if isinstance(weights_target, (float, int)):
            return float(thresholds[0])
        return thresholds

    def calibrate_interval(
        self,
        pred_lower: Union[torch.Tensor],
        pred_upper: Union[torch.Tensor],
        weights_target: Union[torch.Tensor, float],
        alpha: float = 0.1,
    ) -> tuple[Union[torch.Tensor], Union[torch.Tensor]]:
        """Calibrate lower and upper prediction intervals under shift.

        Parameters
        ----------
        pred_lower : torch.Tensor
            Uncalibrated lower predictions, shape (N_target,).
        pred_upper : torch.Tensor
            Uncalibrated upper predictions, shape (N_target,).
        weights_target : Union[torch.Tensor, float]
            Prior ratio weights for target samples, shape (N_target,).
        alpha : float
            Nominal significance level.
        """
        # Convert inputs to tensor
        pred_lower = self._to_tensor(pred_lower)
        pred_upper = self._to_tensor(pred_upper)

        # Compute thresholds
        q = self.compute_thresholds(weights_target, alpha=alpha)

        if isinstance(q, torch.Tensor):
            q_tensor = q.to(dtype=pred_lower.dtype)
            if q_tensor.ndim == 1 and pred_lower.ndim == 2:
                q_tensor = q_tensor.unsqueeze(1)
            lower_cal = pred_lower - q_tensor
            upper_cal = pred_upper + q_tensor
            return lower_cal, upper_cal
        else:
            # Scalar threshold
            return pred_lower - q, pred_upper + q


__all__ = ["SemiConformalCalibrator"]
