"""Causal streaming evaluation harness for regression test-time adaptation."""

from __future__ import annotations

import math
from typing import Any, Callable

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
        predict_fn: Callable[[Any, torch.Tensor], PredictiveBatch] | None = None,
        update_unlabeled_fn: Callable[[Any, torch.Tensor], None] | None = None,
        update_labeled_fn: (Callable[[Any, torch.Tensor, torch.Tensor], None] | None) = None,
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
    def _default_predict_fn(adapter: Any, X: torch.Tensor) -> PredictiveBatch:
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
            if torch.is_tensor(pred):
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
    def _default_update_unlabeled_fn(adapter: Any, X: torch.Tensor) -> None:
        if hasattr(adapter, "adapt_unlabeled_target"):
            adapter.adapt_unlabeled_target(X)

    @staticmethod
    def _default_update_labeled_fn(adapter: Any, X: torch.Tensor, y: torch.Tensor) -> None:
        if hasattr(adapter, "partial_fit"):
            adapter.partial_fit(X, y)
        elif hasattr(adapter, "calibrate_target"):
            pred: Any
            if hasattr(adapter, "predict"):
                pred = adapter.predict(target_inputs=X)
            elif hasattr(adapter, "predict_distribution"):
                pred = adapter.predict_distribution(X)
            else:
                pred = adapter(X)
            if not isinstance(pred, PredictiveBatch):
                pred = PredictiveBatch(point=pred)
            adapter.calibrate_target(pred, y)

    @staticmethod
    def _to_tensor(x: Any) -> torch.Tensor:
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x)

    def evaluate(
        self,
        adapter: Any,
        stream_X: list[torch.Tensor] | torch.Tensor,
        stream_y: list[torch.Tensor] | torch.Tensor,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        """
        Execute the causal evaluation loop over the stream.

        Returns a dictionary of aggregated metrics.
        """
        batches_X: list[torch.Tensor] = []
        batches_y: list[torch.Tensor] = []

        if isinstance(stream_X, list):
            batches_X = stream_X
            if isinstance(stream_y, list):
                batches_y = stream_y
            else:
                raise TypeError("stream_y must be a list if stream_X is a list")
        else:
            if isinstance(stream_y, list):
                raise TypeError("stream_y cannot be a list if stream_X is a tensor")
            n_samples = len(stream_X)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batches_X.append(self._to_tensor(stream_X[start:end]))
                batches_y.append(self._to_tensor(stream_y[start:end]))

        n_steps = len(batches_X)
        all_y_true: list[torch.Tensor] = []
        all_y_pred: list[torch.Tensor] = []
        all_lower: list[torch.Tensor] = []
        all_upper: list[torch.Tensor] = []
        all_nll: list[torch.Tensor] = []

        pending_labels: list[tuple[torch.Tensor, torch.Tensor]] = []

        for t in range(n_steps):
            X_t = batches_X[t]
            y_t = batches_y[t]

            # 1. Predict first (causal)
            pred_batch = self.predict_fn(adapter, X_t)

            # 2. Record predictions
            y_pred = self._to_tensor(
                pred_batch.point
                if pred_batch.point is not None
                else (pred_batch.mean if pred_batch.mean is not None else 0.0)
            )
            all_y_pred.append(y_pred.reshape(-1))
            y_t_t = self._to_tensor(y_t)
            all_y_true.append(y_t_t.reshape(-1))

            # Extract interval bounds
            lower, upper = _native_interval(pred_batch, alpha=self.alpha, eps=1e-8)
            all_lower.append(self._to_tensor(lower.reshape(-1)))
            all_upper.append(self._to_tensor(upper.reshape(-1)))

            # Extract NLL if mean and std are available
            if pred_batch.mean is not None and pred_batch.std is not None:
                mu = self._to_tensor(pred_batch.mean).reshape(-1).float()
                sigma = self._to_tensor(pred_batch.std).reshape(-1).float().clamp(min=1e-8)
                diff = (y_t_t.reshape(-1).float() - mu) ** 2
                nll = 0.5 * (math.log(2.0 * math.pi) + 2.0 * sigma.log() + diff / (sigma**2))
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
        y_true = torch.cat(all_y_true)
        y_pred = torch.cat(all_y_pred)
        lower_bounds = torch.cat(all_lower)
        upper_bounds = torch.cat(all_upper)

        # Compute point metrics
        errors = y_pred - y_true
        mse = float(torch.mean(errors**2))
        mae = float(torch.mean(errors.abs()))
        rmse = float(torch.sqrt(torch.tensor(mse)))

        # Compute interval metrics
        within = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        coverage = float(within.float().mean())
        widths = upper_bounds - lower_bounds
        mean_width = float(widths.mean())

        # Average winkler/interval score
        below_lower = (lower_bounds - y_true).clamp(min=0.0)
        above_upper = (y_true - upper_bounds).clamp(min=0.0)
        winkler_scores = widths + (2.0 / self.alpha) * (below_lower + above_upper)
        mean_winkler = float(winkler_scores.mean())

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "coverage": coverage,
            "mean_width": mean_width,
            "winkler_score": mean_winkler,
        }

        if all_nll:
            metrics["NLL"] = float(torch.cat(all_nll).mean())

        return metrics
