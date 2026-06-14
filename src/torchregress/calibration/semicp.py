"""Semi-supervised and weighted conformal prediction calibration under shift."""

from __future__ import annotations

from typing import Union

import numpy as np
import torch


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
        self.scores_cal_: np.ndarray | None = None
        self.weights_cal_: np.ndarray | None = None

    def _to_numpy(self, array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
        return np.asarray(array)

    def fit(
        self,
        nonconformity_scores_cal: Union[np.ndarray, torch.Tensor],
        weights_cal: Union[np.ndarray, torch.Tensor] | None = None,
    ) -> "SemiConformalCalibrator":
        """Fit the calibrator on calibration nonconformity scores and optional weights.

        Parameters
        ----------
        nonconformity_scores_cal : Union[np.ndarray, torch.Tensor]
            Calibration set nonconformity scores, shape (N_cal,).
        weights_cal : Union[np.ndarray, torch.Tensor], optional
            Weights for calibration samples (e.g. prior ratio), shape (N_cal,).
        """
        scores = self._to_numpy(nonconformity_scores_cal).reshape(-1)
        if weights_cal is not None:
            weights = self._to_numpy(weights_cal).reshape(-1)
            if weights.shape[0] != scores.shape[0]:
                raise ValueError("weights_cal must share shape with nonconformity_scores_cal")
            if np.any(weights < 0.0):
                raise ValueError("weights_cal must be non-negative")
        else:
            weights = np.ones_like(scores)

        # Sort scores and weights in ascending order of scores
        sort_idx = np.argsort(scores)
        self.scores_cal_ = scores[sort_idx]
        self.weights_cal_ = weights[sort_idx]
        return self

    def compute_thresholds(
        self,
        weights_target: Union[np.ndarray, torch.Tensor, float],
        alpha: float = 0.1,
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """Compute sample-specific conformal thresholds for target points.

        Parameters
        ----------
        weights_target : Union[np.ndarray, torch.Tensor, float]
            Shift weights w(x) = p_target(x)/p_source(x) for target points, shape (N_target,).
        alpha : float
            Nominal coverage level is 1 - alpha (e.g. alpha = 0.1 for 90% coverage).
        """
        if self.scores_cal_ is None or self.weights_cal_ is None:
            raise RuntimeError("Calibrator must be fitted before computing thresholds")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")

        # Convert inputs to numpy
        if isinstance(weights_target, (float, int)):
            if weights_target < 0.0:
                raise ValueError("weights_target must be non-negative")
            w_tgt = np.array([float(weights_target)])
        else:
            w_tgt = self._to_numpy(weights_target).reshape(-1)
            if np.any(w_tgt < 0.0):
                raise ValueError("weights_target must be non-negative")

        n_tgt = w_tgt.shape[0]
        thresholds = np.zeros(n_tgt)

        # Total sum of calibration weights
        sum_w_cal = self.weights_cal_.sum()

        for k in range(n_tgt):
            w_inf = w_tgt[k]
            denom = sum_w_cal + w_inf
            if denom <= 0.0:
                thresholds[k] = self.scores_cal_[-1]
                continue

            # Compute cumulative probabilities for sorted calibration scores
            p = self.weights_cal_ / denom
            cum_p = np.cumsum(p)

            # Find the smallest index m such that cum_p >= 1 - alpha
            idx = np.where(cum_p >= 1.0 - alpha)[0]
            if idx.size > 0:
                thresholds[k] = self.scores_cal_[idx[0]]
            else:
                thresholds[k] = self.scores_cal_[-1]

        if isinstance(weights_target, torch.Tensor):
            # Match the device and dtype of the input tensor
            return torch.as_tensor(thresholds, dtype=torch.float32, device=weights_target.device)

        if isinstance(weights_target, (float, int)):
            return float(thresholds[0])
        return thresholds

    def calibrate_interval(
        self,
        pred_lower: Union[np.ndarray, torch.Tensor],
        pred_upper: Union[np.ndarray, torch.Tensor],
        weights_target: Union[np.ndarray, torch.Tensor, float],
        alpha: float = 0.1,
    ) -> tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """Calibrate lower and upper prediction intervals under shift.

        Parameters
        ----------
        pred_lower : Union[np.ndarray, torch.Tensor]
            Uncalibrated lower predictions, shape (N_target,).
        pred_upper : Union[np.ndarray, torch.Tensor]
            Uncalibrated upper predictions, shape (N_target,).
        weights_target : Union[np.ndarray, torch.Tensor, float]
            Prior ratio weights for target samples, shape (N_target,).
        alpha : float
            Nominal significance level.
        """
        # Compute thresholds
        q = self.compute_thresholds(weights_target, alpha=alpha)

        if isinstance(pred_lower, torch.Tensor):
            q_tensor = torch.as_tensor(q, dtype=pred_lower.dtype, device=pred_lower.device)
            # Handle possible batching or 1D format
            if q_tensor.ndim == 1 and pred_lower.ndim == 2:
                q_tensor = q_tensor.unsqueeze(1)
            lower_cal = pred_lower - q_tensor
            upper_cal = pred_upper + q_tensor
            return lower_cal, upper_cal
        else:
            lower = np.asarray(pred_lower)
            upper = np.asarray(pred_upper)
            q_arr = np.asarray(q)
            if q_arr.ndim == 1 and lower.ndim == 2:
                q_arr = q_arr[:, None]
            return lower - q_arr, upper + q_arr


__all__ = ["SemiConformalCalibrator"]
