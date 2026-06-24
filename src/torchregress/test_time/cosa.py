"""COSA-style (Conformal Output Space Adaptation) test-time adaptation adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from torchregress.prediction import PredictiveBatch


class DelayedLabelResidualAdapter:
    """
    COSA-style (Conformal Output Space Adaptation) test-time adapter.

    Adapts predictions and/or standard deviations in a streaming fashion,
    incorporating delayed label feedback.
    """

    def __init__(
        self,
        base_model: Any,
        *,
        ema_beta: float = 0.1,
        scale_ema_beta: float = 0.1,
    ) -> None:
        """
        Initialize the adapter.

        Parameters
        ----------
        base_model : Any
            The base model or adapter to adapt. Must support
            `predict_distribution`, `predict`, or be callable.
        ema_beta : float
            Smoothing factor for the point prediction residual EMA.
        scale_ema_beta : float
            Smoothing factor for the variance inflation scale EMA.
        """
        self.base_model = base_model
        self.ema_beta = float(ema_beta)
        self.scale_ema_beta = float(scale_ema_beta)

        self.residual_mean_: torch.Tensor | None = None
        self.variance_inflation_: torch.Tensor | None = None

    def _get_base_prediction(self, X: torch.Tensor) -> PredictiveBatch:
        if hasattr(self.base_model, "predict_distribution"):
            return self.base_model.predict_distribution(X)
        elif hasattr(self.base_model, "predict"):
            pred = self.base_model.predict(X)
            if isinstance(pred, PredictiveBatch):
                return pred
            if isinstance(pred, dict):
                return PredictiveBatch(
                    point=pred.get("mean") if pred.get("mean") is not None else pred.get("point"),
                    std=pred.get("std"),
                    mean=pred.get("mean"),
                    quantiles=pred.get("quantiles"),
                    quantile_levels=pred.get("quantile_levels"),
                )
            if torch.is_tensor(pred):
                return PredictiveBatch(point=pred)
            if isinstance(pred, np.ndarray):
                return PredictiveBatch(point=torch.from_numpy(pred))
            raise TypeError(f"Unsupported prediction return type: {type(pred)}")
        elif callable(self.base_model):
            pred = self.base_model(X)
            if isinstance(pred, PredictiveBatch):
                return pred
            if torch.is_tensor(pred):
                return PredictiveBatch(point=pred)
        raise AttributeError(
            "base_model must implement predict_distribution, predict, or be callable."
        )

    def predict_distribution(self, X: np.ndarray | torch.Tensor) -> PredictiveBatch:
        """
        Predict adapted distribution parameters for the input.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Input features.

        Returns
        -------
        PredictiveBatch
            Adapted predictive batch with adjusted point, mean, std, and quantiles.
        """
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            X_tensor = torch.as_tensor(X, dtype=torch.float32)
        else:
            X_tensor = torch.as_tensor(X)

        pred = self._get_base_prediction(X_tensor)

        res_mean = self.residual_mean_
        var_infl = self.variance_inflation_

        # 1. Adapt point prediction and mean
        point_adapted: torch.Tensor | np.ndarray | None = None
        if pred.point is not None:
            point_t = torch.as_tensor(pred.point)
            if res_mean is not None:
                point_adapted = point_t + res_mean
            else:
                point_adapted = point_t

        mean_adapted: torch.Tensor | np.ndarray | None = None
        if pred.mean is not None:
            mean_t = torch.as_tensor(pred.mean)
            if res_mean is not None:
                mean_adapted = mean_t + res_mean
            else:
                mean_adapted = mean_t

        # 2. Adapt standard deviation
        std_adapted: torch.Tensor | np.ndarray | None = None
        if pred.std is not None:
            std_t = torch.as_tensor(pred.std)
            if var_infl is not None:
                std_adapted = std_t * torch.sqrt(var_infl)
            else:
                std_adapted = std_t

        # 3. Adapt quantiles
        quantiles_adapted: torch.Tensor | np.ndarray | None = None
        if pred.quantiles is not None:
            quantiles_t = torch.as_tensor(pred.quantiles)
            if res_mean is not None:
                if pred.mean is not None and var_infl is not None and mean_adapted is not None:
                    # Scale quantiles around the new mean
                    scale = torch.sqrt(var_infl)
                    mean_raw_t = torch.as_tensor(pred.mean)
                    mean_adapted_t = torch.as_tensor(mean_adapted)
                    quantiles_adapted = mean_adapted_t + scale * (quantiles_t - mean_raw_t)
                else:
                    # Simple shift
                    quantiles_adapted = quantiles_t + res_mean
            else:
                quantiles_adapted = quantiles_t

        if is_numpy:
            if point_adapted is not None:
                point_adapted = torch.as_tensor(point_adapted).detach().cpu().numpy()
            if mean_adapted is not None:
                mean_adapted = torch.as_tensor(mean_adapted).detach().cpu().numpy()
            if std_adapted is not None:
                std_adapted = torch.as_tensor(std_adapted).detach().cpu().numpy()
            if quantiles_adapted is not None:
                quantiles_adapted = torch.as_tensor(quantiles_adapted).detach().cpu().numpy()

        return PredictiveBatch(
            point=point_adapted,
            mean=mean_adapted,
            std=std_adapted,
            quantiles=quantiles_adapted,
            quantile_levels=pred.quantile_levels,
            extra=pred.extra,
        )

    def predict(self, X: np.ndarray | torch.Tensor) -> dict[str, Any]:
        """Predict point predictions and/or standard deviations for the input."""
        pred = self.predict_distribution(X)
        res: dict[str, Any] = {}
        if pred.point is not None:
            res["point"] = pred.point
        if pred.mean is not None:
            res["mean"] = pred.mean
        if pred.std is not None:
            res["std"] = pred.std
        if pred.quantiles is not None:
            res["quantiles"] = pred.quantiles
            res["quantile_levels"] = pred.quantile_levels
        return res

    def partial_fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor) -> None:
        """
        Update the running estimate of the residuals with newly observed labels.

        Parameters
        ----------
        X : np.ndarray | torch.Tensor
            Observed input features.
        y : np.ndarray | torch.Tensor
            Observed ground truth labels.
        """
        is_numpy = isinstance(X, np.ndarray)
        if is_numpy:
            X_tensor = torch.as_tensor(X, dtype=torch.float32)
            y_tensor = torch.as_tensor(y, dtype=torch.float32)
        else:
            X_tensor = torch.as_tensor(X)
            y_tensor = torch.as_tensor(y)

        pred = self._get_base_prediction(X_tensor)

        mean_raw = pred.mean if pred.mean is not None else pred.point
        if mean_raw is None:
            raise ValueError("Base model predictions must contain either point or mean.")

        mean_raw_t = torch.as_tensor(mean_raw)

        # Align shapes of y and prediction
        if y_tensor.dim() == 1 and mean_raw_t.dim() > 1:
            y_tensor = y_tensor.view_as(mean_raw_t)
        elif mean_raw_t.dim() == 1 and y_tensor.dim() > 1:
            mean_raw_t = mean_raw_t.view_as(y_tensor)

        error = y_tensor - mean_raw_t

        # Update point correction (EMA of residuals)
        batch_mean_error = error.mean(dim=0)
        if self.residual_mean_ is None:
            self.residual_mean_ = batch_mean_error
        else:
            self.residual_mean_ = (
                1.0 - self.ema_beta
            ) * self.residual_mean_ + self.ema_beta * batch_mean_error

        # Update variance inflation factor (EMA of normalized squared residuals)
        if pred.std is not None:
            std_t = torch.as_tensor(pred.std)
            std_clamped = torch.clamp(std_t, min=1e-8)
            z_squared = ((error - self.residual_mean_) / std_clamped) ** 2
            batch_mean_z_squared = z_squared.mean(dim=0)
            if self.variance_inflation_ is None:
                self.variance_inflation_ = batch_mean_z_squared
            else:
                self.variance_inflation_ = (
                    1.0 - self.scale_ema_beta
                ) * self.variance_inflation_ + self.scale_ema_beta * batch_mean_z_squared
            self.variance_inflation_ = torch.clamp(self.variance_inflation_, min=1e-5)
