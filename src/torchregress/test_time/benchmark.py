"""Causal streaming evaluation harness for regression test-time adaptation."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from torchregress.prediction import PredictiveBatch

from .transport import _native_interval


class CausalTTAHarness:
    """
    Causal streaming evaluation harness for test-time adaptation.

    Simulates a real-time stream where prediction happens first, followed by
    optional delayed label observations and adapter state updates.
    """

    def __init__(
        self,
        *,
        delay: int = 0,
        alpha: float = 0.1,
        predict_fn: Callable[[Any, np.ndarray | torch.Tensor], PredictiveBatch] | None = None,
        update_unlabeled_fn: Callable[[Any, np.ndarray | torch.Tensor], None] | None = None,
        update_labeled_fn: (
            Callable[[Any, np.ndarray | torch.Tensor, np.ndarray | torch.Tensor], None] | None
        ) = None,
    ) -> None:
        """
        Initialize the harness.

        Parameters
        ----------
        delay : int
            Feedback delay in steps (batches).
            - 0: Immediate feedback. Labels are observed right after predicting the current batch.
            - >0: Delayed feedback. Labels from batch `t` are observed at step `t + delay`.
            - <0 or None: Unlabeled TTA (no labels are ever observed/used for update).
        alpha : float
            Significance level for prediction intervals (coverage targets 1 - alpha).
        predict_fn : Callable, optional
            Custom prediction function (defaults to calling predict_distribution or predict).
        update_unlabeled_fn : Callable, optional
            Custom update function for new unlabeled data.
        update_labeled_fn : Callable, optional
            Custom update function when labeled data becomes available.
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.delay = delay
        self.alpha = alpha
        self.predict_fn = predict_fn or self._default_predict_fn
        self.update_unlabeled_fn = update_unlabeled_fn or self._default_update_unlabeled_fn
        self.update_labeled_fn = update_labeled_fn or self._default_update_labeled_fn

    @staticmethod
    def _default_predict_fn(adapter: Any, X: np.ndarray | torch.Tensor) -> PredictiveBatch:
        if hasattr(adapter, "predict_distribution"):
            return adapter.predict_distribution(X)
        elif hasattr(adapter, "predict"):
            pred = adapter.predict(X)
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
            if torch.is_tensor(pred) or isinstance(pred, np.ndarray):
                return PredictiveBatch(point=pred)
            raise TypeError(f"Unsupported prediction return type: {type(pred)}")
        elif callable(adapter):
            pred = adapter(X)
            if isinstance(pred, PredictiveBatch):
                return pred
        raise AttributeError(
            "Adapter must implement predict_distribution, predict, or be callable."
        )

    @staticmethod
    def _default_update_unlabeled_fn(adapter: Any, X: np.ndarray | torch.Tensor) -> None:
        if hasattr(adapter, "adapt_unlabeled_target"):
            adapter.adapt_unlabeled_target(X)

    @staticmethod
    def _default_update_labeled_fn(
        adapter: Any, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
    ) -> None:
        if hasattr(adapter, "partial_fit"):
            adapter.partial_fit(X, y)
        elif hasattr(adapter, "calibrate_target"):
            adapter.calibrate_target(X, y)

    def evaluate(
        self,
        adapter: Any,
        stream_X: list[np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor,
        stream_y: list[np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        """
        Execute the causal evaluation loop over the stream.

        Returns a dictionary of aggregated metrics.
        """
        batches_X: list[np.ndarray | torch.Tensor] = []
        batches_y: list[np.ndarray | torch.Tensor] = []

        if isinstance(stream_X, list):
            batches_X = stream_X
            if isinstance(stream_y, list):
                batches_y = stream_y
            else:
                raise TypeError("stream_y must be a list if stream_X is a list")
        else:
            if isinstance(stream_y, list):
                raise TypeError("stream_y cannot be a list if stream_X is a tensor/array")
            n_samples = len(stream_X)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batches_X.append(stream_X[start:end])
                batches_y.append(stream_y[start:end])

        n_steps = len(batches_X)
        all_y_true: list[np.ndarray] = []
        all_y_pred: list[np.ndarray] = []
        all_lower: list[np.ndarray] = []
        all_upper: list[np.ndarray] = []
        all_nll: list[np.ndarray] = []

        pending_labels: list[tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]] = []

        for t in range(n_steps):
            X_t = batches_X[t]
            y_t = batches_y[t]

            # 1. Predict first (causal)
            pred_batch = self.predict_fn(adapter, X_t)

            # 2. Record predictions
            y_pred_np = self._to_numpy(
                pred_batch.point
                if pred_batch.point is not None
                else (pred_batch.mean if pred_batch.mean is not None else 0.0)
            ).reshape(-1)
            y_true_np = self._to_numpy(y_t).reshape(-1)

            all_y_pred.append(y_pred_np)
            all_y_true.append(y_true_np)

            # Extract interval bounds
            lower, upper = _native_interval(pred_batch, alpha=self.alpha, eps=1e-8)
            all_lower.append(lower.reshape(-1))
            all_upper.append(upper.reshape(-1))

            # Extract NLL if mean and std are available
            if pred_batch.mean is not None and pred_batch.std is not None:
                mu = self._to_numpy(pred_batch.mean).reshape(-1)
                sigma = np.clip(self._to_numpy(pred_batch.std).reshape(-1), 1e-8, None)
                nll = 0.5 * np.log(2.0 * np.pi * (sigma**2)) + ((y_true_np - mu) ** 2) / (
                    2.0 * (sigma**2)
                )
                all_nll.append(nll)

            # 3. Update unlabeled features
            if self.update_unlabeled_fn is not None:
                self.update_unlabeled_fn(adapter, X_t)

            # 4. Feed labeled targets (with delay)
            if self.delay is not None and self.delay >= 0:
                pending_labels.append((X_t, y_t))
                if len(pending_labels) > self.delay:
                    X_old, y_old = pending_labels.pop(0)
                    if self.update_labeled_fn is not None:
                        self.update_labeled_fn(adapter, X_old, y_old)

        # Concatenate all steps
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        lower_bounds = np.concatenate(all_lower)
        upper_bounds = np.concatenate(all_upper)

        # Compute point metrics
        errors = y_pred - y_true
        mse = float(np.mean(errors**2))
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(mse))

        # Compute interval metrics
        within = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        coverage = float(np.mean(within))
        widths = upper_bounds - lower_bounds
        mean_width = float(np.mean(widths))

        # Average winkler/interval score
        below_lower = np.clip(lower_bounds - y_true, a_min=0, a_max=None)
        above_upper = np.clip(y_true - upper_bounds, a_min=0, a_max=None)
        winkler_scores = widths + (2.0 / self.alpha) * (below_lower + above_upper)
        mean_winkler = float(np.mean(winkler_scores))

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "coverage": coverage,
            "mean_width": mean_width,
            "winkler_score": mean_winkler,
        }

        if all_nll:
            metrics["NLL"] = float(np.mean(np.concatenate(all_nll)))

        return metrics

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
