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

    References
    ----------
    .. [1] Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate
       multiclass probability estimates. In *KDD 2002*. https://doi.org/10.1145/775047.775151
    """

    out_of_bounds: str = "clip"

    def __post_init__(self) -> None:
        self._model = None

    def fit(self, pred_mean: Tensor, target: Tensor) -> "IsotonicMeanCalibrator":
        from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]

        x = pred_mean.detach().cpu().numpy().reshape(-1)
        y = target.detach().cpu().numpy().reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("pred_mean and target must share sample dimension")

        model = IsotonicRegression(out_of_bounds=self.out_of_bounds)
        model.fit(x, y)
        self._model = model
        return self

    def transform(self, pred_mean: Tensor) -> Tensor:
        if self._model is None:
            raise ValueError("IsotonicMeanCalibrator must be fitted before transform")
        x = pred_mean.detach().cpu().numpy().reshape(-1)
        y_hat = self._model.predict(x)
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
