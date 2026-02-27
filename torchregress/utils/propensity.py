"""Propensity estimation and inverse-probability weighting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PropensityEstimator:
    """Simple logistic propensity estimator for selection/missingness modeling."""

    clip_min: float = 0.01
    clip_max: float = 0.99
    max_iter: int = 1000
    random_state: int = 42

    def __post_init__(self) -> None:
        if not (0.0 < self.clip_min < self.clip_max < 1.0):
            raise ValueError("clip_min/clip_max must satisfy 0 < clip_min < clip_max < 1")
        self._model = None

    def fit(self, x: Tensor, observed: Tensor) -> "PropensityEstimator":
        """Fit propensity p(observed=1 | x)."""
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

        x_np = x.detach().cpu().numpy()
        obs_np = observed.detach().cpu().numpy().reshape(-1)
        if x_np.shape[0] != obs_np.shape[0]:
            raise ValueError("x and observed must share the sample dimension")

        model = LogisticRegression(
            max_iter=self.max_iter,
            class_weight="balanced",
            random_state=self.random_state,
        )
        model.fit(x_np, obs_np)
        self._model = model
        return self

    def predict_proba(self, x: Tensor) -> Tensor:
        """Predict clipped propensity scores p(observed=1 | x)."""
        if self._model is None:
            raise ValueError("PropensityEstimator must be fitted before predict_proba")

        x_np = x.detach().cpu().numpy()
        p = self._model.predict_proba(x_np)[:, 1]
        p_t = torch.tensor(p, dtype=torch.float32, device=x.device)
        return p_t.clamp(min=self.clip_min, max=self.clip_max)


def ipw_weights(
    propensity: Tensor,
    observed: Tensor | None = None,
    *,
    clip_min: float = 0.01,
    clip_max: float = 0.99,
    normalize: bool = True,
) -> Tensor:
    """Compute inverse-probability weights.

    If `observed` is provided, computes symmetric weights:
    - observed=1 -> 1 / p
    - observed=0 -> 1 / (1 - p)

    If `observed` is omitted, returns 1 / p.
    """
    if not (0.0 < clip_min < clip_max < 1.0):
        raise ValueError("clip_min/clip_max must satisfy 0 < clip_min < clip_max < 1")

    p = propensity.float().clamp(min=clip_min, max=clip_max)

    w: Tensor
    if observed is None:
        w = 1.0 / p
    else:
        obs = observed.float()
        if obs.shape != p.shape:
            raise ValueError("observed and propensity must have matching shapes")
        w = obs / p + (1.0 - obs) / (1.0 - p)

    if normalize:
        w = w / w.mean().clamp_min(1e-8)
    return w


__all__ = ["PropensityEstimator", "ipw_weights"]
