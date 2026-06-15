"""Propensity estimation and inverse-probability weighting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


@dataclass
class PropensityEstimator:
    """Simple logistic propensity estimator for selection/missingness modeling.

    Uses a torch-native linear model with binary cross-entropy loss so that
    the estimator integrates with GPU workflows.  Equivalent to a scikit-learn
    ``LogisticRegression`` with ``class_weight='balanced'``.
    """

    clip_min: float = 0.01
    clip_max: float = 0.99
    max_iter: int = 1000
    learning_rate: float = 0.1
    random_state: int = 42

    _weight: Tensor | None = field(default=None, init=False, repr=False)
    _bias: Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.clip_min < self.clip_max < 1.0):
            raise ValueError("clip_min/clip_max must satisfy 0 < clip_min < clip_max < 1")

    # ── balancing heuristic (matched to class_weight='balanced') ──────────
    @staticmethod
    def _balanced_class_weight(observed: Tensor) -> Tensor:
        """Return scalar weight for the positive class (1's)."""
        n = observed.numel()
        n_pos = observed.sum().clamp_min(1)
        n_neg = n - n_pos
        if n_neg == 0:
            return torch.tensor(1.0, device=observed.device, dtype=torch.float32)
        return (n_neg / n_pos).float()

    # ── training helpers ───────────────────────────────────────────────────
    def _predict_logits(self, x: Tensor) -> Tensor:
        if self._weight is None or self._bias is None:
            raise ValueError("PropensityEstimator must be fitted before predict_proba")
        return x @ self._weight + self._bias

    def fit(self, x: Tensor, observed: Tensor) -> "PropensityEstimator":
        """Fit propensity p(observed=1 | x) with a linear logistic model."""
        if x.shape[0] != observed.numel():
            raise ValueError("x and observed must share the sample dimension")

        obs = observed.float().reshape(-1, 1)

        # balanced class weight (applied in BCE loss)
        pos_weight = self._balanced_class_weight(obs).to(device=x.device, dtype=x.dtype)

        # initialise linear model
        in_features = x.shape[-1]
        _gen = torch.Generator(device=x.device)
        _gen.manual_seed(self.random_state)

        weight = torch.zeros(in_features, 1, device=x.device, dtype=x.dtype)
        bias = torch.zeros(1, device=x.device, dtype=x.dtype)
        weight.requires_grad_(True)
        bias.requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [weight, bias],
            lr=self.learning_rate,
            max_iter=self.max_iter,
            line_search_fn="strong_wolfe",
        )

        def _closure() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            logits = x @ weight + bias
            loss = F.binary_cross_entropy_with_logits(
                logits, obs, pos_weight=pos_weight.expand_as(logits)
            )
            loss.backward()
            return loss

        optimizer.step(_closure)

        self._weight = weight.detach().cpu()
        self._bias = bias.detach().cpu()
        return self

    def predict_proba(self, x: Tensor) -> Tensor:
        """Predict clipped propensity scores p(observed=1 | x)."""
        logits = self._predict_logits(x)
        p = torch.sigmoid(logits).squeeze(-1)
        return p.clamp(min=self.clip_min, max=self.clip_max)


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
