"""Noise-aware likelihoods and binned CDF density heads.

This module separates two different uncertainties that are often conflated in
regression:

* the intrinsic conditional density :math:`p(y \\mid x)`, and
* the measurement process that turns the latent :math:`y` into the recorded
  label :math:`\\tilde y`.

For Gaussian label noise, :class:`NoiseAwareDensityLoss` maximizes

.. math::

    p(\\tilde y \\mid x, \\sigma)
    = \\int p(y \\mid x)\\,\\mathcal N(\\tilde y; y, \\sigma^2)\\,dy .

It wraps any torchregress density loss (normalizing flow, MDN, or a custom
negative-log-density callable).  Scalar flows use Gaussian--Hermite quadrature;
diagonal MDNs use the exact analytic convolution; everything else falls back to
Monte Carlo label-noise averaging.  An optional contamination component handles
grossly wrong labels rather than merely noisy measurements.

:class:`BinnedDensityLoss` is the corresponding smoothed histogram / ordinal
CDF head: a categorical model over fixed target bins, trained with Gaussian bin
masses (or one-hot labels) and a cross-entropy score.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from .base import BaseLoss

_MIN_SIGMA = 1.0e-8


def _as_sigma(sigma: Tensor | float | None, target: Tensor) -> Optional[Tensor]:
    """Broadcast a scalar / per-row / per-element sigma to ``target`` shape."""
    if sigma is None:
        return None
    if not isinstance(sigma, Tensor):
        sigma = torch.as_tensor(sigma, device=target.device, dtype=target.dtype)
    sigma = sigma.to(device=target.device, dtype=target.dtype)
    if not torch.isfinite(sigma).all() or (sigma < 0).any():
        raise ValueError("sigma must contain finite nonnegative standard deviations")
    if sigma.numel() == 1:
        return sigma.reshape(()).expand_as(target)
    if sigma.shape == target.shape + (1,):
        sigma = sigma.squeeze(-1)
    if sigma.dim() == 1:
        if sigma.shape[0] == target.shape[0]:
            sigma = sigma.reshape(-1, *([1] * (target.dim() - 1)))
        elif sigma.shape[0] == target.shape[-1]:
            sigma = sigma.reshape(*([1] * (target.dim() - 1)), -1)
        else:
            raise ValueError(
                f"sigma shape {tuple(sigma.shape)} is neither one-per-sample nor one-per-feature"
            )
    try:
        return sigma.expand_as(target)
    except RuntimeError as exc:
        raise ValueError(
            f"sigma shape {tuple(sigma.shape)} cannot broadcast to target shape "
            f"{tuple(target.shape)}"
        ) from exc


def _scalar_log_density_from_loss(
    base_loss: Any,
    y_pred: Tensor,
    target: Tensor,
    *,
    mask: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Return a per-sample clean log-density from any supported head."""
    # NormalizingFlowLoss / distribution objects with a true log_prob.
    if hasattr(base_loss, "log_prob"):
        out = base_loss.log_prob(y_pred, target)
        return out.reshape(out.shape[0], -1).sum(dim=-1)

    # MDN diagonal: exact mixture log density.
    if hasattr(base_loss, "_extract_distribution_parameters") and hasattr(
        base_loss, "covariance_type"
    ):
        if str(base_loss.covariance_type).lower() == "diagonal":
            return _mdn_diagonal_log_prob(base_loss, y_pred, target)

    # Generic NLL callable: temporarily force reduction='none'.
    previous = getattr(base_loss, "reduction", None)
    if previous is not None:
        base_loss.reduction = "none"
    try:
        nll = base_loss(y_pred, target, mask=mask, weights=weights)
    finally:
        if previous is not None:
            base_loss.reduction = previous
    nll = nll.reshape(nll.shape[0], -1)
    return -nll.sum(dim=-1)


def _mdn_diagonal_log_prob(loss_fn: Any, y_pred: Tensor, target: Tensor) -> Tensor:
    """Exact diagonal-mixture log density (unconvolved)."""
    log_weights, means, stds = loss_fn._extract_distribution_parameters(y_pred)
    target_expanded = target.unsqueeze(-2)
    stds = stds.clamp_min(loss_fn.min_std)
    normalized = (target_expanded - means) / stds
    log_component = -0.5 * (normalized**2).sum(dim=-1)
    log_component = log_component - torch.log(stds).sum(dim=-1)
    log_component = log_component - 0.5 * loss_fn.n_features * loss_fn.log_2pi
    return torch.logsumexp(log_weights + log_component, dim=-1)


def _mdn_diagonal_convolved_log_prob(
    loss_fn: Any,
    y_pred: Tensor,
    target: Tensor,
    sigma: Tensor,
) -> Tensor:
    """Exact diagonal-mixture convolution with Gaussian label noise."""
    log_weights, means, stds = loss_fn._extract_distribution_parameters(y_pred)
    target_expanded = target.unsqueeze(-2)
    stds = stds.clamp_min(loss_fn.min_std)
    obs_var = stds**2 + sigma.unsqueeze(-2) ** 2
    obs_std = obs_var.sqrt()
    normalized = (target_expanded - means) / obs_std
    log_component = -0.5 * (normalized**2).sum(dim=-1)
    log_component = log_component - torch.log(obs_std).sum(dim=-1)
    log_component = log_component - 0.5 * loss_fn.n_features * loss_fn.log_2pi
    return torch.logsumexp(log_weights + log_component, dim=-1)


def _gauss_hermite(order: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    nodes, weights = np.polynomial.hermite.hermgauss(int(order))
    return (
        torch.as_tensor(nodes, device=device, dtype=dtype),
        torch.as_tensor(weights, device=device, dtype=dtype),
    )


class NoiseAwareDensityLoss(BaseLoss):
    """Marginal likelihood for a density head with noisy target labels.

    Given a density head ``p_base(y | x)`` and per-label standard deviations
    ``sigma``, this loss maximizes ``log E_{y ~ p_base(.|x)}[N(y; y_obs, sigma^2)]``.
    The clean density is never fitted to the noisy label directly; the
    measurement model accounts for the extra variance.

    Parameters
    ----------
    base_loss:
        A torchregress density loss.  Supported natively: ``NormalizingFlowLoss``
        (Gaussian--Hermite quadrature for a scalar target), diagonal
        ``MixtureDensityLoss`` (exact analytic convolution), and any callable
        returning per-sample negative log density (Monte Carlo fallback).
    sigma:
        Optional default label-noise standard deviation.  Can also be passed
        per call, with shape ``[B]``, ``[B, D]``, ``[D]``, or scalar.
    mode:
        ``"auto"`` (quadrature/analytic where available), ``"quadrature"``,
        ``"analytic"``, or ``"mc"``.
    n_samples:
        Number of Monte Carlo label-noise draws in the fallback path.
    contamination_fraction:
        Optional probability of a grossly wrong label, mixed as
        ``(1-eps) p_clean + eps p_bad``.  This is not the same as measurement
        noise: use it for swapped/mismatched labels, and keep it small unless
        the contaminating process is known.
    contamination_scale:
        Scale of the Student-t(3) contamination density in target units.  Should
        usually be on the order of the target range.
    reduction:
        ``"mean"``, ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        base_loss: Module | Any,
        *,
        sigma: Optional[Tensor | float] = None,
        mode: str = "auto",
        n_samples: int = 64,
        gh_order: int = 32,
        contamination_fraction: float = 0.0,
        contamination_scale: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        if not callable(base_loss):
            raise TypeError("base_loss must be a torchregress density loss or callable")
        self.base_loss: Any = base_loss
        self.sigma = sigma
        if mode not in {"auto", "quadrature", "analytic", "mc"}:
            raise ValueError("mode must be 'auto', 'quadrature', 'analytic', or 'mc'")
        self.mode = mode
        self.n_samples = int(n_samples)
        self.gh_order = int(gh_order)
        if self.n_samples < 1 or self.gh_order < 2:
            raise ValueError("n_samples must be positive and gh_order at least two")
        if not 0.0 <= float(contamination_fraction) < 1.0:
            raise ValueError("contamination_fraction must be in [0, 1)")
        self.contamination_fraction = float(contamination_fraction)
        if not math.isfinite(float(contamination_scale)) or float(contamination_scale) <= 0.0:
            raise ValueError("contamination_scale must be positive")
        self.contamination_scale = float(contamination_scale)

    def _clean_log_prob(self, y_pred: Tensor, target: Tensor, sigma: Tensor) -> Tensor:
        n_features = target.shape[-1]
        flow_log_prob = hasattr(self.base_loss, "log_prob")
        mdn_diag = hasattr(self.base_loss, "covariance_type") and (
            str(self.base_loss.covariance_type).lower() == "diagonal"
        )

        if self.mode in {"auto", "quadrature"} and flow_log_prob and n_features == 1:
            nodes, weights = _gauss_hermite(self.gh_order, target.device, target.dtype)
            # y = target + sqrt(2) * sigma * t; the Gaussian weight includes 1/sqrt(pi).
            # Keep the target feature axis: [B, K, 1] for a scalar flow.
            samples = target.unsqueeze(1) + math.sqrt(2.0) * sigma.unsqueeze(1) * nodes.reshape(
                1, -1, 1
            )
            b, k = samples.shape[:2]
            flat_context = (
                y_pred.unsqueeze(1)
                .expand(b, k, *y_pred.shape[1:])
                .reshape(b * k, *y_pred.shape[1:])
            )
            logp = self.base_loss.log_prob(flat_context, samples.reshape(b * k, -1))
            logp = logp.reshape(b, k)
            log_norm = math.log(math.sqrt(math.pi))
            return torch.logsumexp(logp + torch.log(weights).reshape(1, -1), dim=-1) - log_norm

        if self.mode in {"auto", "analytic"} and mdn_diag:
            return _mdn_diagonal_convolved_log_prob(self.base_loss, y_pred, target, sigma)

        if self.mode in {"analytic", "quadrature"}:
            raise ValueError(
                "requested integration mode is unsupported for this density/target dimension"
            )
        # Generic Monte Carlo: draw latent target noise and average the clean
        # density over the Gaussian measurement kernel.
        eps = torch.randn(self.n_samples, *target.shape, device=target.device, dtype=target.dtype)
        samples = target.unsqueeze(0) - eps * sigma.unsqueeze(0)
        flat_samples = samples.reshape(-1, *target.shape[1:])
        flat_pred = (
            y_pred.unsqueeze(0).expand(self.n_samples, *y_pred.shape).reshape(-1, *y_pred.shape[1:])
        )
        logp = _scalar_log_density_from_loss(self.base_loss, flat_pred, flat_samples)
        return torch.logsumexp(logp.reshape(self.n_samples, -1), dim=0) - math.log(self.n_samples)

    def _contamination_log_prob(self, target: Tensor) -> Tensor:
        dist = torch.distributions.StudentT(
            df=torch.tensor(3.0, device=target.device, dtype=target.dtype),
            loc=torch.zeros((), device=target.device, dtype=target.dtype),
            scale=torch.tensor(self.contamination_scale, device=target.device, dtype=target.dtype),
        )
        return dist.log_prob(target).sum(dim=-1)

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        sigma: Optional[Tensor | float] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(-1)
        if target.dim() != 2 or target.shape[0] != y_pred.shape[0]:
            raise ValueError("target must be [B,D] and share the prediction batch")
        row_mask = None
        if mask is not None:
            mask = mask.to(device=target.device, dtype=torch.bool)
            if mask.numel() == target.shape[0]:
                row_mask = mask.reshape(-1)
            elif mask.shape == target.shape:
                if ((mask.any(-1)) != mask.all(-1)).any():
                    raise ValueError(
                        "partial target masks require marginal densities, "
                        "not a joint-density row loss"
                    )
                row_mask = mask.all(-1)
            else:
                raise ValueError("mask must contain one entry per row or match target")
            target = torch.where(row_mask[:, None], target, torch.zeros_like(target))
            keep_context = row_mask.reshape((-1,) + (1,) * (y_pred.dim() - 1))
            y_pred = torch.where(keep_context, y_pred, torch.zeros_like(y_pred))
        if weights is not None:
            if weights.numel() != target.shape[0]:
                raise ValueError("joint-density weights must contain one entry per row")
            weights = weights.reshape(-1)
        sigma_value = self.sigma if sigma is None else sigma
        if isinstance(sigma_value, Tensor) and row_mask is not None:
            sigma_value = sigma_value.to(target)
            # Invalid rows may contain unavailable uncertainties, just like labels.
            if sigma_value.numel() == target.shape[0]:
                sigma_value = torch.where(row_mask, sigma_value.reshape(-1), 0)
            elif sigma_value.shape == target.shape:
                sigma_value = torch.where(row_mask[:, None], sigma_value, 0)
        sigma_t = _as_sigma(sigma_value, target)
        if sigma_t is None:
            # Apply masking/weighting once, outside the underlying density score.
            log_clean = _scalar_log_density_from_loss(self.base_loss, y_pred, target)
        else:
            log_clean = self._clean_log_prob(y_pred, target, sigma_t)
        if self.contamination_fraction > 0.0:
            log_bad = self._contamination_log_prob(target)
            eps = self.contamination_fraction
            log_mix = torch.logsumexp(
                torch.stack(
                    [
                        math.log1p(-eps) + log_clean,
                        math.log(eps) + log_bad,
                    ],
                    dim=0,
                ),
                dim=0,
            )
        else:
            log_mix = log_clean
        return self._reduce(-log_mix, row_mask, weights)


class BinnedDensityLoss(BaseLoss):
    """Smoothed histogram / ordinal CDF head for scalar targets.

    The model outputs one logit per target bin.  Hard labels are one-hot
    categorical targets; with ``sigma`` the target is replaced by its Gaussian
    measurement distribution integrated over each bin, yielding a soft-target
    cross-entropy score. This is not a deconvolved measurement likelihood.
    Sampling and the CDF use uniform density within each finite bin.

    Parameters
    ----------
    bin_edges:
        Strictly increasing bin boundaries, length ``K + 1`` for ``K`` logits.
    label_smoothing:
        Uniform categorical smoothing in ``[0, 1)``.
    reduction:
        ``"mean"``, ``"sum"``, or ``"none"``.
    """

    bin_edges: Tensor

    def __init__(
        self,
        bin_edges: Tensor | list[float] | tuple[float, ...],
        *,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        edges = torch.as_tensor(bin_edges, dtype=torch.float32)
        if edges.dim() != 1 or edges.numel() < 2:
            raise ValueError("bin_edges must be a 1D sequence with at least two edges")
        if not torch.isfinite(edges).all() or not bool(torch.all(edges[1:] > edges[:-1])):
            raise ValueError("bin_edges must be finite and strictly increasing")
        self.register_buffer("bin_edges", edges)
        if not 0.0 <= float(label_smoothing) < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.label_smoothing = float(label_smoothing)

    @property
    def n_bins(self) -> int:
        return int(self.bin_edges.numel() - 1)

    def bin_probabilities(
        self,
        target: Tensor,
        sigma: Optional[Tensor | float] = None,
    ) -> Tensor:
        """Gaussian bin masses (or hard one-hot with smoothing) for each row."""
        if target.dim() > 1:
            if target.numel() != target.shape[0]:
                raise ValueError("BinnedDensityLoss supports one scalar target per row")
            target = target.reshape(-1)
        target = target.reshape(-1).to(dtype=self.bin_edges.dtype, device=self.bin_edges.device)
        edges: Tensor = self.bin_edges
        sigma_t = _as_sigma(sigma, target)
        if sigma_t is None or bool(torch.all(sigma_t <= _MIN_SIGMA * 10)):
            idx = torch.searchsorted(edges, target.reshape(-1)).clamp(1, self.n_bins) - 1
            probs = torch.zeros(
                target.shape[0], self.n_bins, device=target.device, dtype=edges.dtype
            )
            probs.scatter_(1, idx.unsqueeze(1), 1.0)
        else:
            normal = torch.distributions.Normal(
                loc=target.reshape(-1, 1), scale=sigma_t.reshape(-1, 1).clamp_min(_MIN_SIGMA)
            )
            cdf = normal.cdf(edges.reshape(1, -1))
            probs = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(1.0e-12)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        if self.label_smoothing > 0.0:
            probs = (1.0 - self.label_smoothing) * probs + self.label_smoothing / self.n_bins
        return probs

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        sigma: Optional[Tensor | float] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        del kwargs
        if y_pred.dim() != 2 or y_pred.shape[-1] != self.n_bins:
            raise ValueError(f"y_pred must be [B, {self.n_bins}] bin logits")
        if target.numel() != y_pred.shape[0]:
            raise ValueError("BinnedDensityLoss supports one scalar target per row")
        target = target.reshape(-1)
        if mask is not None:
            if mask.numel() != len(target):
                raise ValueError("mask must contain one entry per scalar target")
            mask = mask.reshape(-1).to(device=target.device, dtype=torch.bool)
            target = torch.where(mask, target, torch.zeros_like(target))
            if isinstance(sigma, Tensor) and sigma.numel() == len(target):
                sigma = torch.where(mask, sigma.reshape(-1).to(target), 0)
        if weights is not None:
            if weights.numel() != len(target):
                raise ValueError("weights must contain one entry per scalar target")
            weights = weights.reshape(-1)
        probs = self.bin_probabilities(target, sigma).to(device=y_pred.device, dtype=y_pred.dtype)
        log_probs = torch.log_softmax(y_pred, dim=-1)
        nll = -(probs * log_probs).sum(dim=-1)
        return self._reduce(nll, mask, weights)

    def cdf(self, y_pred: Tensor, values: Tensor) -> Tensor:
        """CDF of the uniform-within-bin distribution, one query per row."""
        if y_pred.dim() != 2 or y_pred.shape[-1] != self.n_bins:
            raise ValueError(f"y_pred must be [B, {self.n_bins}] bin logits")
        if values.numel() != y_pred.shape[0]:
            raise ValueError("values must contain one query per prediction row")
        edges = self.bin_edges.to(y_pred)
        query = values.reshape(-1, 1).to(y_pred)
        fraction = ((query - edges[:-1]) / (edges[1:] - edges[:-1])).clamp(0, 1)
        return (torch.softmax(y_pred, dim=-1) * fraction).sum(-1).reshape(values.shape)

    def sample(self, y_pred: Tensor, n_samples: int = 1) -> Tensor:
        """Draw bin indices and uniform intra-bin target values."""
        probs = torch.softmax(y_pred, dim=-1)
        idx = torch.distributions.Categorical(probs=probs).sample((n_samples,)).transpose(0, 1)
        edges: Tensor = self.bin_edges
        left = edges[:-1][idx]
        right = edges[1:][idx]
        return left + (right - left) * torch.rand_like(left)
