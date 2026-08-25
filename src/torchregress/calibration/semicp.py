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
        """
        scores = self._to_tensor(nonconformity_scores_cal).reshape(-1).float()
        if weights_cal is not None:
            weights = self._to_tensor(weights_cal).reshape(-1).float()
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
            Thresholds use the finite-sample correction: the smallest score whose
            normalized cumulative weight reaches ``ceil((n+1)*(1-alpha))/n``
            (exact order statistic in the unweighted case).
        """
        if self.scores_cal_ is None or self.weights_cal_ is None:
            raise RuntimeError("Calibrator must be fitted before computing thresholds")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")

        # Convert inputs to tensor
        if isinstance(weights_target, (float, int)):
            if weights_target < 0.0:
                raise ValueError("weights_target must be non-negative")
            w_tgt = torch.tensor([float(weights_target)])
        else:
            w_tgt = self._to_tensor(weights_target).reshape(-1).float()
            if torch.any(w_tgt < 0.0):
                raise ValueError("weights_target must be non-negative")

        n_tgt = w_tgt.shape[0]
        device = w_tgt.device
        scores = self.scores_cal_.to(device=device, dtype=torch.float)
        weights = self.weights_cal_.to(device=device, dtype=torch.float)
        thresholds = torch.zeros(n_tgt, device=device)

        # Total sum of calibration weights
        sum_w_cal = weights.sum()

        n_cal = scores.shape[0]
        # Finite-sample correction (TR-MET-12): target mass on the normalized
        # cumulative-weight curve is ceil((n+1)*(1-alpha))/n, not 1 - alpha.
        q_adj = min(math.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal, 1.0)
        uniform_weights = bool(torch.all(weights == weights[0]))

        for k in range(n_tgt):
            w_inf = w_tgt[k]
            denom = sum_w_cal + w_inf
            if denom <= 0.0:
                thresholds[k] = scores[-1]
                continue

            # Unweighted path: exact finite-sample split-conformal order statistic.
            if uniform_weights and float(w_inf) == 0.0:
                thresholds[k] = finite_sample_quantile(scores, alpha)
                continue

            # Weighted path (Tibshirani et al., 2019): the cumulative weight of
            # sorted calibration scores plus the target pseudo-weight, evaluated
            # at the (n+1)-corrected mass q_adj.
            p = weights / denom
            cum_p = torch.cumsum(p, dim=0)

            # Find the smallest index m such that cum_p >= q_adj
            mask = cum_p >= q_adj
            idx = torch.where(mask)[0]
            if idx.numel() > 0:
                thresholds[k] = scores[idx[0]]
            else:
                thresholds[k] = scores[-1]

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
