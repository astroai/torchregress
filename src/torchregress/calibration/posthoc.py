"""Post-hoc calibration transforms for regression outputs."""

from __future__ import annotations

import math
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

        log_t = torch.nn.Parameter(torch.tensor(math.log(self.temperature), dtype=torch.float32))
        optimizer = torch.optim.Adam([log_t], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad(set_to_none=True)
            t = torch.exp(log_t).clamp(min=0.05, max=20.0)
            scaled_var = (var * t).clamp_min(self.eps)
            nll = 0.5 * (
                torch.log(scaled_var) + ((y - mean) ** 2) / scaled_var + math.log(2.0 * math.pi)
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
        self._x: Tensor | None = None
        self._y: Tensor | None = None

    @staticmethod
    def _pava(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        is_numpy = isinstance(x, np.ndarray)
        x_t = torch.as_tensor(x)
        y_t = torch.as_tensor(y)

        if x_t.numel() == 0:
            if is_numpy:
                return np.array([], dtype=x.dtype), np.array([], dtype=y.dtype)
            return x_t.clone(), y_t.clone()

        order = x_t.argsort()
        x_s = x_t[order].double()
        y_s = y_t[order].double()

        blocks_x: list[Tensor] = []
        blocks_y: list[Tensor] = []

        for i in range(len(x_s)):
            blocks_x.append(x_s[i : i + 1])
            blocks_y.append(y_s[i : i + 1])

            while len(blocks_y) >= 2 and blocks_y[-2].mean() > blocks_y[-1].mean():
                bx = torch.cat([blocks_x.pop(), blocks_x.pop()])
                by = torch.cat([blocks_y.pop(), blocks_y.pop()])
                blocks_x.append(bx)
                blocks_y.append(by)

        result_x = torch.tensor([b.mean() for b in blocks_x], dtype=x_t.dtype)
        result_y = torch.tensor([b.mean() for b in blocks_y], dtype=y_t.dtype)

        if is_numpy:
            return result_x.numpy(), result_y.numpy()
        return result_x, result_y

    def _interpolate(self, x_query: Tensor) -> Tensor:
        xq = torch.as_tensor(x_query)
        if self._x is None or self._y is None:
            raise ValueError("IsotonicMeanCalibrator must be fitted before transform")

        if self._x.numel() == 0:
            return torch.zeros_like(xq)

        if self._x.numel() == 1:
            return torch.full_like(xq, self._y[0])

        idx = torch.searchsorted(self._x, xq).clamp(1, len(self._x) - 1)
        x_left = self._x[idx - 1]
        x_right = self._x[idx]
        y_left = self._y[idx - 1]
        y_right = self._y[idx]

        denom = x_right - x_left
        t = torch.where(denom.abs() < 1e-12, 0.0, (xq - x_left) / denom)

        if self.out_of_bounds == "clip":
            t = t.clamp(0.0, 1.0)
            y_out = y_left + t * (y_right - y_left)
            y_out = torch.where(xq <= self._x[0], self._y[0], y_out)
            y_out = torch.where(xq >= self._x[-1], self._y[-1], y_out)
            return y_out

        return y_left + t * (y_right - y_left)

    def fit(self, pred_mean: Tensor, target: Tensor) -> "IsotonicMeanCalibrator":
        x = pred_mean.detach().reshape(-1).double()
        y = target.detach().reshape(-1).double()
        if x.shape[0] != y.shape[0]:
            raise ValueError("pred_mean and target must share sample dimension")

        self._x, self._y = self._pava(x, y)
        return self

    def transform(self, pred_mean: Tensor) -> Tensor:
        x = pred_mean.detach().reshape(-1).double()
        y_hat = self._interpolate(x)
        return y_hat.reshape(pred_mean.shape).to(pred_mean.dtype)


@dataclass
class PITCalibrator:
    """Monotonic PIT-value calibrator using empirical CDF mapping."""

    eps: float = 1e-6

    def __post_init__(self) -> None:
        self._x: Tensor | None = None
        self._y: Tensor | None = None

    @staticmethod
    def pit_from_gaussian(pred_mean: Tensor, pred_std: Tensor, target: Tensor) -> Tensor:
        std = pred_std.clamp_min(1e-8)
        z = (target - pred_mean) / std
        return normal_cdf(z).clamp(min=1e-6, max=1.0 - 1e-6)

    def fit(self, pit_values: Tensor) -> "PITCalibrator":
        pit = pit_values.detach().reshape(-1).clamp(self.eps, 1.0 - self.eps)
        pit_sorted = pit.sort().values
        n = pit_sorted.shape[0]
        targets = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        self._x = pit_sorted
        self._y = targets
        return self

    def transform(self, pit_values: Tensor) -> Tensor:
        if self._x is None or self._y is None:
            raise ValueError("PITCalibrator must be fitted before transform")
        pit = pit_values.detach().reshape(-1)
        idx = torch.searchsorted(self._x, pit).clamp(1, len(self._x) - 1)
        x_left = self._x[idx - 1]
        x_right = self._x[idx]
        y_left = self._y[idx - 1]
        y_right = self._y[idx]
        denom = x_right - x_left
        t = torch.where(denom.abs() < 1e-12, 0.0, (pit - x_left) / denom)
        mapped = y_left + t.clamp(0.0, 1.0) * (y_right - y_left)
        mapped = torch.where(pit <= self._x[0], self._y[0], mapped)
        mapped = torch.where(pit >= self._x[-1], self._y[-1], mapped)
        return mapped.clamp(self.eps, 1.0 - self.eps).reshape(pit_values.shape).to(pit_values.dtype)


__all__ = [
    "VarianceTemperatureScaler",
    "IsotonicMeanCalibrator",
    "PITCalibrator",
]
