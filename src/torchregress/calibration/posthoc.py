"""Post-hoc calibration transforms for regression outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from torchregress.utils.distributions import normal_cdf


@dataclass
class VarianceTemperatureScaler:
    """Scalar variance-temperature calibration for Gaussian predictive variance.

    References
    ----------
    .. [1] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration
       of Modern Neural Networks. In *ICML 2017*. https://arxiv.org/abs/1706.04599
    """

    temperature: float = 1.0
    eps: float = 1e-8

    def fit(
        self,
        pred_mean: Tensor,
        pred_var: Tensor,
        target: Tensor,
        *,
        max_iter: int = 200,
        lr: float = 0.05,
    ) -> "VarianceTemperatureScaler":
        if pred_mean.shape != pred_var.shape or pred_mean.shape != target.shape:
            raise ValueError("pred_mean, pred_var, and target must share shape")

        mean = pred_mean.detach().float()
        var = pred_var.detach().float().clamp_min(self.eps)
        y = target.detach().float()

        log_t = torch.nn.Parameter(torch.tensor(np.log(self.temperature), dtype=torch.float32))
        optimizer = torch.optim.Adam([log_t], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad(set_to_none=True)
            t = torch.exp(log_t).clamp(min=0.05, max=20.0)
            scaled_var = (var * t).clamp_min(self.eps)
            nll = 0.5 * (
                torch.log(scaled_var) + ((y - mean) ** 2) / scaled_var + np.log(2.0 * np.pi)
            )
            loss = nll.mean()
            loss.backward()
            optimizer.step()

        self.temperature = float(torch.exp(log_t).item())
        return self

    def transform(self, pred_var: Tensor) -> Tensor:
        return (pred_var * self.temperature).clamp_min(self.eps)


@dataclass
class IsotonicMeanCalibrator:
    """Isotonic regression calibrator for point predictions.

    Implements the Pool Adjacent Violators Algorithm (PAVA) directly on
    PyTorch tensors so no scikit-learn dependency is required.

    References
    ----------
    .. [1] Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate
       multiclass probability estimates. In *KDD 2002*. https://doi.org/10.1145/775047.775151
    """

    out_of_bounds: str = "clip"

    def __post_init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    # ── PAVA (Pool Adjacent Violators Algorithm) ──────────────────────────
    @staticmethod
    def _pava(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit isotonic regression via PAVA.

        Returns (x_sorted, y_monotonic) where y_monotonic is non-decreasing
        and x_sorted is the corresponding block averages.
        """
        if x.size == 0:
            return np.array([], dtype=x.dtype), np.array([], dtype=y.dtype)

        order = np.argsort(x)
        x_s = x[order].astype(np.float64)
        y_s = y[order].astype(np.float64)

        # PAVA: merge blocks that violate monotonicity
        blocks_x: list[np.ndarray] = []
        blocks_y: list[np.ndarray] = []

        for i in range(len(x_s)):
            blocks_x.append(x_s[i : i + 1])
            blocks_y.append(y_s[i : i + 1])

            # merge backward while constraint violated
            while len(blocks_y) >= 2 and blocks_y[-2].mean() > blocks_y[-1].mean():
                bx = np.concatenate([blocks_x.pop(), blocks_x.pop()])
                by = np.concatenate([blocks_y.pop(), blocks_y.pop()])
                blocks_x.append(bx)
                blocks_y.append(by)

        # output block means
        result_x = np.array([b.mean() for b in blocks_x], dtype=x.dtype)
        result_y = np.array([b.mean() for b in blocks_y], dtype=y.dtype)
        return result_x, result_y

    def _interpolate(self, x_query: np.ndarray) -> np.ndarray:
        """Predict y-values for x_query using the fitted PAVA solution."""
        if self._x is None or self._y is None:
            raise ValueError("IsotonicMeanCalibrator must be fitted before transform")

        if self._x.size == 0:
            return np.full_like(x_query, 0.0)

        if self._x.size == 1:
            return np.full_like(x_query, self._y[0])

        idx = np.searchsorted(self._x, x_query)

        # clip to valid range
        idx = np.clip(idx, 1, len(self._x) - 1)

        # linear interpolation between neighbouring breakpoints
        x_left = self._x[idx - 1]
        x_right = self._x[idx]
        y_left = self._y[idx - 1]
        y_right = self._y[idx]

        denom = x_right - x_left
        t = np.where(np.abs(denom) < 1e-12, 0.0, (x_query - x_left) / denom)

        if self.out_of_bounds == "clip":
            t = np.clip(t, 0.0, 1.0)
            # also clip extrapolation beyond the fitted range
            y_out = y_left + t * (y_right - y_left)
            y_out = np.where(x_query <= self._x[0], self._y[0], y_out)
            y_out = np.where(x_query >= self._x[-1], self._y[-1], y_out)
            return y_out

        return y_left + t * (y_right - y_left)

    def fit(self, pred_mean: Tensor, target: Tensor) -> "IsotonicMeanCalibrator":
        x = pred_mean.detach().cpu().numpy().reshape(-1).astype(np.float64)
        y = target.detach().cpu().numpy().reshape(-1).astype(np.float64)
        if x.shape[0] != y.shape[0]:
            raise ValueError("pred_mean and target must share sample dimension")

        self._x, self._y = self._pava(x, y)
        return self

    def transform(self, pred_mean: Tensor) -> Tensor:
        x = pred_mean.detach().cpu().numpy().reshape(-1).astype(np.float64)
        y_hat = self._interpolate(x)
        return torch.tensor(y_hat, dtype=torch.float32, device=pred_mean.device).reshape(
            pred_mean.shape
        )


@dataclass
class PITCalibrator:
    """Monotonic PIT-value calibrator using empirical CDF mapping."""

    eps: float = 1e-6

    def __post_init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    @staticmethod
    def pit_from_gaussian(pred_mean: Tensor, pred_std: Tensor, target: Tensor) -> Tensor:
        std = pred_std.clamp_min(1e-8)
        z = (target - pred_mean) / std
        return normal_cdf(z).clamp(min=1e-6, max=1.0 - 1e-6)

    def fit(self, pit_values: Tensor) -> "PITCalibrator":
        pit = pit_values.detach().cpu().numpy().reshape(-1)
        pit = np.clip(pit, self.eps, 1.0 - self.eps)
        pit_sorted = np.sort(pit)
        n = pit_sorted.shape[0]
        targets = (np.arange(n, dtype=np.float64) + 0.5) / n
        self._x = pit_sorted
        self._y = targets
        return self

    def transform(self, pit_values: Tensor) -> Tensor:
        if self._x is None or self._y is None:
            raise ValueError("PITCalibrator must be fitted before transform")
        pit = pit_values.detach().cpu().numpy().reshape(-1)
        mapped = np.interp(pit, self._x, self._y)
        mapped = np.clip(mapped, self.eps, 1.0 - self.eps)
        return torch.tensor(mapped, dtype=torch.float32, device=pit_values.device).reshape(
            pit_values.shape
        )


__all__ = [
    "VarianceTemperatureScaler",
    "IsotonicMeanCalibrator",
    "PITCalibrator",
]
