"""Density-ratio weight estimators for shift-aware conformal calibration.

Estimates per-point importance weights ``w(x) = p_target(x) / p_source(x)``
consumed by :class:`~torchregress.losses.conformal.NonExchangeableConformalRegressor`
and :class:`~torchregress.losses.conformal.MultivariateScoreConformal`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .label_shift import estimate_target_prior_em
from .ot_conformal import ScoreCDFReweighter

logger = logging.getLogger(__name__)


def _as_2d_float(x: Tensor, *, name: str) -> Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)!r}")
    if x.dim() != 2:
        raise ValueError(f"{name} must be 2-D [n, d], got {tuple(x.shape)}")
    return x


class DomainClassifierRatioEstimator:
    """Density-ratio ``w(x) = p_target(x)/p_source(x)`` via a domain classifier.

    Fits a probabilistic MLP discriminator between source and target features
    (BCE loss); the posterior ``p(target | x)`` converts to the density ratio as
    ``w(x) = p / (1 - p)``, clipped to ``[1/clip, clip]`` for stability.

    Args:
        hidden: Hidden layer widths of the discriminator MLP.
        epochs: Full-batch Adam steps.
        lr: Adam learning rate.
        clip: Clip bound applied to the density ratio (and its inverse).
    """

    def __init__(
        self,
        hidden: Tuple[int, ...] = (64, 64),
        epochs: int = 200,
        lr: float = 1e-3,
        clip: float = 20.0,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if clip <= 0:
            raise ValueError("clip must be positive")
        self.hidden = tuple(int(h) for h in hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.clip = float(clip)
        self._net: Optional[nn.Module] = None
        self._mean: Optional[Tensor] = None
        self._std: Optional[Tensor] = None
        self.diagnostics_: Dict[str, Any] = {}

    def fit(self, X_source: Tensor, X_target: Tensor) -> "DomainClassifierRatioEstimator":
        xs = _as_2d_float(X_source, name="X_source").detach()
        xt = _as_2d_float(X_target, name="X_target").detach()
        if xs.shape[1] != xt.shape[1]:
            raise ValueError(
                f"feature dimension mismatch: source {xs.shape[1]} vs target {xt.shape[1]}"
            )
        device, dtype = xs.device, xs.dtype
        mean = xs.mean(dim=0)
        std = xs.std(dim=0).clamp_min(torch.finfo(dtype).tiny)
        zs = (xs - mean) / std
        zt = ((xt - mean) / std).to(device=device, dtype=dtype)

        feats = torch.cat([zs, zt], dim=0)
        labels = torch.cat([torch.zeros(xs.shape[0]), torch.ones(zt.shape[0])], dim=0).to(
            device=device, dtype=dtype
        )
        layers: list[nn.Module] = []
        in_dim = xs.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        net = nn.Sequential(*layers).to(device=device, dtype=dtype)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        last = torch.zeros((), device=device, dtype=dtype)
        for _ in range(self.epochs):
            opt.zero_grad(set_to_none=True)
            logits = net(feats).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            last = loss.detach()
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = float(((net(feats).squeeze(-1) > 0).float() == labels).float().mean())
        self._net = net.eval()
        self._mean = mean
        self._std = std
        # ponytail: full-batch training keeps the estimator deterministic and
        # simple; ceiling is memory on very large pooled sets — upgrade path is
        # mini-batching over permutations.
        self.diagnostics_ = {"domain_bce": float(last), "domain_accuracy": acc}
        return self

    def weights_for(self, X: Tensor) -> Tensor:
        """Density-ratio weights for arbitrary feature rows."""
        if self._net is None or self._mean is None or self._std is None:
            raise RuntimeError("call fit() before requesting weights")
        x = _as_2d_float(X, name="X").to(device=self._mean.device, dtype=self._mean.dtype)
        if x.shape[1] != self._mean.shape[0]:
            raise ValueError(
                f"feature dimension mismatch: expected {self._mean.shape[0]}, got {x.shape[1]}"
            )
        with torch.no_grad():
            logit = self._net((x - self._mean) / self._std).squeeze(-1)
        ratio = torch.exp(logit.clamp(min=-self.clip, max=self.clip))
        return ratio.clamp(min=1.0 / self.clip, max=self.clip)

    def weights(self, X_cal: Tensor) -> Tensor:
        """Importance weights for calibration rows (alias of :meth:`weights_for`)."""
        return self.weights_for(X_cal)


class OTScoreWeightEstimator(ScoreCDFReweighter):
    """``ScoreCDFReweighter`` adapter exposing feature-space ``.weights(X)``.

    Projects source/target features onto the first principal component of the
    pooled data (via ``torch.pca_lowrank``) and fits the 1-D score-CDF matching
    reweighter on those projections verbatim.

    ponytail: the fitted simplex is tied to the calibration rows seen at fit
    time; ``weights()`` therefore only accepts the same number of rows and
    returns the stored simplex aligned by row order. Upgrade path: refit or
    out-of-sample extension via kernel smoothing.
    """

    def fit(
        self,
        calibration_scores: Tensor,
        target_unlabeled_scores: Tensor,
    ) -> "OTScoreWeightEstimator":
        """Fit on 1-D PCA projections of source/target feature matrices."""
        xc = _as_2d_float(calibration_scores, name="calibration_scores").detach()
        xt = _as_2d_float(target_unlabeled_scores, name="target_unlabeled_scores").detach()
        if xc.shape[1] != xt.shape[1]:
            raise ValueError(
                f"feature dimension mismatch: cal {xc.shape[1]} vs target {xt.shape[1]}"
            )
        pooled = torch.cat([xc, xt], dim=0)
        centered = pooled - pooled.mean(dim=0, keepdim=True)
        q = min(1, min(centered.shape))
        try:
            _, _, v = torch.pca_lowrank(centered, q=q, niter=4)
            direction = v[:, 0]
        except Exception as exc:  # degenerate (constant) features
            logger.warning("pca_lowrank failed (%s); falling back to first axis", exc)
            direction = torch.zeros(centered.shape[1], device=centered.device, dtype=centered.dtype)
            direction[0] = 1.0
        proj_cal = ((xc - pooled.mean(dim=0)) @ direction).reshape(-1)
        proj_tgt = ((xt - pooled.mean(dim=0)) @ direction).reshape(-1)
        super().fit(proj_cal, proj_tgt)
        self._n_cal = int(xc.shape[0])
        return self

    def weights(self, X_cal: Tensor) -> Tensor:
        """Simplex weights over calibration rows (row-aligned with ``fit`` input)."""
        if getattr(self, "_n_cal", None) is None or self.weights_ is None:
            raise RuntimeError("call fit() before weights()")
        if not torch.is_tensor(X_cal) or X_cal.dim() != 2 or X_cal.shape[0] != self._n_cal:
            raise ValueError(
                f"weights() requires the [{self._n_cal}, d] calibration matrix from fit()"
            )
        return self.weights_


def estimate_label_shift_weights(
    mu_source_probs: np.ndarray,
    mu_target_probs: np.ndarray,
    *,
    clip: float = 20.0,
) -> tuple[np.ndarray, Any]:
    """Per-point importance weights under label shift (BBSE-per-point identity).

    Estimates target priors ``pi_t`` from unlabeled target posteriors via EM,
    then returns ``w(x) = sum_k p_s(k|x) * pi_t(k)/pi_s(k)`` — the standard
    identity ``E_{p_s}[w(x) | x] = sum_k p_s(y=k|x) * pi_t(k)/pi_s(k)``.

    Args:
        mu_source_probs: ``[n_source, K]`` predicted posteriors on source data.
        mu_target_probs: ``[n_target, K]`` predicted posteriors on target data.
        clip: Upper clip applied to class prior ratios.

    Returns:
        ``(w_target, label_shift_estimate)`` where ``w_target`` has shape
        ``[n_target]`` and the second element is the EM
        :class:`~torchregress.test_time.label_shift.LabelShiftEstimate`.
    """
    src = np.asarray(mu_source_probs, dtype=float)
    tgt = np.asarray(mu_target_probs, dtype=float)
    if src.ndim != 2 or tgt.ndim != 2 or src.shape[1] != tgt.shape[1]:
        raise ValueError(
            f"posteriors must be [n, K] with matching K, got {src.shape} vs {tgt.shape}"
        )
    src = np.clip(src, 1e-12, None)
    src = src / src.sum(axis=1, keepdims=True)
    source_prior = np.clip(src.mean(axis=0), 1e-12, None)
    source_prior = source_prior / source_prior.sum()
    est = estimate_target_prior_em(tgt, source_prior=source_prior)
    ratios = np.clip(est.target_prior / est.source_prior, 1e-12, float(clip))
    w = (tgt * ratios[None, :]).sum(axis=1)
    return w.astype(float), est
