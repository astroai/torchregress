"""Noise-adaptive Balanced MSE via batch-based Monte Carlo (BMC).

This module implements the **Batch-based Monte-Carlo (BMC)** option of Balanced
MSE from Ren et al. (CVPR 2022, arXiv:2203.16427).  It is distinct from
:class:`torchregress.losses.BalancedMSELoss`, which is inverse bin-frequency
weighting and is *not* the noise-adaptive estimator of the paper.

Balanced MSE models the balanced predictive distribution
``p_bal(y | x) = N(y; f(x), sigma**2)`` and converts it to the training
distribution with Bayes' rule.  The resulting negative log-likelihood, written
for one training target ``y_i`` and a batch of targets ``{y_j}``, is

.. math::

    L_i = -\\log p(y_i)
          + \\frac{(y_i - f_i)^2}{2\\sigma^2}
          + \\log \\sum_j p(y_j)
            \\exp\\!\\left(-\\frac{(f_i - y_j)^2}{2\\sigma^2}\\right),

up to constants that are identical for every candidate label and therefore
cancel.  Here ``p`` is the label prior estimated by :meth:`BalancedMSEBMC.fit`
(or the uniform prior); with a uniform prior the expression reduces exactly to
the paper's BMC loss (their Eq. 3.14), a softmax with temperature
``tau = 2 * sigma**2``.

The loss is computed per sample and then handed to the shared
:class:`~torchregress.losses.base.BaseLoss` reduction, so masks and sample
weights follow the usual torchregress contract.  Masked (missing) targets are
removed from the batch support of the Monte Carlo estimate, not merely zeroed
afterwards.

Notes
-----
BMC is inherently **batch-based**: the normaliser is a Monte Carlo estimate over
the targets in the current mini-batch.  Its value and variance therefore depend
on the batch size and on which samples happen to share a batch.  The paper uses
a batch size of 256 for its synthetic benchmark.  Do not compare absolute loss
values across batch sizes, and prefer reasonably large batches; the pairwise
tensor is ``O(B**2)``.

Only scalar (single-output) regression targets are supported.  A target with
shape ``[B, 1]`` is accepted and treated as one scalar per sample.  For
multi-output targets, model each coordinate with its own prior and noise scale,
or use the GMM-based analytical option of the paper; silently averaging
coordinates would change the objective.

References
----------
.. [1] J. Ren, M. Zhang, C. Yu, Z. Liu. "Balanced MSE for Imbalanced Visual
   Regression." *CVPR*, 2022. https://arxiv.org/abs/2203.16427
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal, Optional, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import BaseLoss, RegressionLoss
from .loss_registry import register_regression_loss

__all__ = ["BalancedMSEBMC", "balanced_mse_bmc_loss"]

PriorKind = Literal["histogram", "kde", "uniform"]

#: Consistency factor converting a Gaussian median absolute deviation (MAD)
#: to a standard-deviation estimate.
_MAD_TO_SIGMA = 1.4826


def _scalar_targets(t: Tensor, name: str) -> tuple[Tensor, bool]:
    """Return ``(t_flat, was_2d)`` for scalar-per-sample targets.

    ``[B]`` stays ``[B]``; ``[B, 1]`` is squeezed to ``[B]``.  Anything else is
    rejected so that a multi-output objective is never silently reduced to the
    mean coordinate.
    """
    if not torch.is_floating_point(t):
        raise TypeError(f"{name} must be a floating-point tensor, got dtype {t.dtype}.")
    if t.dim() == 0:
        raise ValueError(f"{name} must have a batch dimension, got a scalar tensor.")
    if t.dim() == 1:
        return t, False
    if t.dim() == 2 and t.shape[-1] == 1:
        return t.squeeze(-1), True
    raise ValueError(
        f"{name} must be single-output with shape [B] or [B, 1]; got {tuple(t.shape)}. "
        "BalancedMSEBMC does not average multi-output targets because that changes the "
        "objective; use one loss per coordinate with its own prior and sigma."
    )


def _collapse_mask(mask: Tensor, batch_size: int) -> Tensor:
    """Return a boolean ``[B]`` validity mask, collapsing trailing dims with ``all``."""
    if mask.shape[0] != batch_size:
        raise ValueError(
            f"Mask batch dimension {mask.shape[0]} does not match target batch {batch_size}."
        )
    if mask.dim() == 1:
        return mask.to(dtype=torch.bool)
    return mask.to(dtype=torch.bool).reshape(batch_size, -1).all(dim=-1)


def _compute_dtype(*tensors: Tensor) -> torch.dtype:
    """Promoted floating dtype, widened to float32 for half-precision inputs."""
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _mad_scale(x: Tensor) -> Tensor:
    """Robust Gaussian scale estimate ``1.4826 * median(|x - median(x)|).``"""
    x = x.reshape(-1)
    center = x.median()
    return _MAD_TO_SIGMA * (x - center).abs().median()


def _bmc_per_sample(
    y_pred: Tensor,
    target: Tensor,
    support: Tensor,
    log_prior: Tensor,
    sigma: Tensor,
    valid: Tensor,
) -> Tensor:
    """Core BMC per-sample loss on scalar ``[B]`` tensors.

    ``support`` is the detached target batch used for the Monte Carlo
    normaliser; ``target`` is the (possibly differentiable) target used in the
    squared residual.  ``log_prior`` is ``-inf`` for invalid support entries.
    """
    if not bool(valid.any()):
        return torch.zeros_like(y_pred)
    sigma_sq = sigma * sigma
    # Pairwise logits: log p(y_j) - (f_i - y_j)^2 / (2 sigma^2).
    logits = log_prior.unsqueeze(0) - (y_pred.unsqueeze(1) - support.unsqueeze(0)).square() / (
        2.0 * sigma_sq
    )
    logits = logits.masked_fill(~valid.unsqueeze(0), -torch.inf)
    log_normaliser = torch.logsumexp(logits, dim=1)
    residual = (y_pred - target).square() / (2.0 * sigma_sq)
    per_sample = -log_prior + residual + log_normaliser
    # Rows that are masked off may contain inf/nan from the -inf prior; zero
    # them before the shared reduction so 0 * nan can never leak into a mean.
    return torch.where(valid, per_sample, torch.zeros_like(per_sample))


@register_regression_loss("balanced_mse_bmc")
class BalancedMSEBMC(RegressionLoss):
    """Batch-based Monte-Carlo Balanced MSE (BMC) for scalar targets.

    Parameters
    ----------
    sigma:
        Label-noise standard deviation.  If omitted, :meth:`fit` estimates it
        robustly (see :meth:`fit`).  It sets the softmax temperature and
        controls the frequent-vs-rare trade-off; tune or estimate it for your
        data.  The paper learns ``sigma`` jointly with the model instead.
    prior:
        Label-prior estimator used by :meth:`fit`:

        * ``"histogram"`` (default): Gaussian-smoothed histogram of the
          training targets, evaluated by linear interpolation of the log
          density at the bin centres.  Fast and scalable.
        * ``"kde"``: Gaussian kernel density estimate with Silverman's rule
          (or an explicit ``prior_bandwidth``).  The reference set is
          deterministically subsampled to ``kde_points`` points.
        * ``"uniform"``: ``p(y)`` is constant.  With this choice the loss is
          exactly the paper's BMC loss (their Eq. 3.14), because the ``log p``
          terms cancel.
    num_bins:
        Number of histogram bins for ``prior="histogram"`` (at least 2).
    prior_bandwidth:
        Smoothing bandwidth in target units.  For the histogram it defaults to
        1.5 bin widths; for KDE it defaults to Silverman's rule.  Must be
        positive when supplied.
    kde_points:
        Maximum number of KDE reference points (deterministic subsample).
    prior_floor:
        Lower floor on the estimated density, as a fraction of its maximum.
        Keeps ``log p`` finite in empty histogram bins and far KDE tails.
        Must lie in ``(0, 1]``.
    min_sigma:
        Positive floor applied to an estimated ``sigma``.
    reduction:
        ``'mean'`` | ``'sum'`` | ``'none'``; see
        :class:`~torchregress.losses.base.BaseLoss`.

    Notes
    -----
    Call :meth:`fit` once on the training targets before the optimization loop.
    The fitted prior is a buffer and follows the module's device via ``.to()``.
    If ``forward`` is called before ``fit``, it falls back to the paper's
    uniform-prior BMC with the supplied ``sigma`` (or ``sigma=1.0``) and emits a
    ``UserWarning``; ``prior="uniform"`` with an explicit ``sigma`` needs no
    fitting at all.

    The batch normaliser makes the loss **non-decomposable across samples**:
    the loss of sample ``i`` depends on the other targets in its batch.  Use a
    batch size large enough for the Monte Carlo estimate to be stable (the
    paper uses 256) and keep it fixed between runs being compared.

    Example
    -------
    >>> import torch
    >>> from torchregress.losses.balanced_mse_bmc import BalancedMSEBMC
    >>> loss_fn = BalancedMSEBMC(sigma=0.5).fit(train_targets)
    >>> loss = loss_fn(model(x_batch), y_batch)
    """

    _sigma_init: Optional[float]
    _warned_no_fit: bool
    _fitted_flag: Tensor
    _sigma: Tensor
    _prior_centers: Tensor
    _prior_log_density: Tensor
    _prior_bin_width: Tensor
    _prior_log_floor: Tensor
    _kde_points: Tensor
    _kde_bandwidth: Tensor
    _fitted_median: Tensor

    def __init__(
        self,
        *,
        sigma: Optional[float] = None,
        prior: PriorKind = "histogram",
        num_bins: int = 64,
        prior_bandwidth: Optional[float] = None,
        kde_points: int = 2048,
        prior_floor: float = 1e-6,
        min_sigma: float = 1e-6,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)

        if sigma is not None:
            sigma = float(sigma)
            if not math.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(f"sigma must be a positive finite number, got {sigma!r}.")
        self._sigma_init = sigma
        self._warned_no_fit = False

        if prior not in ("histogram", "kde", "uniform"):
            raise ValueError("prior must be 'histogram', 'kde', or 'uniform'.")
        self.prior = cast(PriorKind, prior)

        if int(num_bins) < 2:
            raise ValueError("num_bins must be at least 2.")
        self.num_bins = int(num_bins)

        if prior_bandwidth is not None:
            prior_bandwidth = float(prior_bandwidth)
            if not math.isfinite(prior_bandwidth) or prior_bandwidth <= 0.0:
                raise ValueError("prior_bandwidth must be positive and finite when supplied.")
        self.prior_bandwidth = prior_bandwidth

        if int(kde_points) < 1:
            raise ValueError("kde_points must be at least 1.")
        self.kde_points = int(kde_points)

        prior_floor = float(prior_floor)
        if not 0.0 < prior_floor <= 1.0:
            raise ValueError("prior_floor must lie in (0, 1].")
        self.prior_floor = prior_floor

        min_sigma = float(min_sigma)
        if not math.isfinite(min_sigma) or min_sigma <= 0.0:
            raise ValueError("min_sigma must be a positive finite number.")
        self.min_sigma = min_sigma

        self.register_buffer(
            "_sigma", torch.tensor(0.0 if sigma is None else sigma, dtype=torch.float32)
        )
        self.register_buffer("_fitted_flag", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_prior_centers", torch.empty(0, dtype=torch.float32))
        self.register_buffer("_prior_log_density", torch.empty(0, dtype=torch.float32))
        self.register_buffer("_prior_bin_width", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("_prior_log_floor", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("_kde_points", torch.empty(0, dtype=torch.float32))
        self.register_buffer("_kde_bandwidth", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("_fitted_median", torch.tensor(0.0, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not bool(self._fitted_flag):
            raise RuntimeError("Call fit(train_targets) before using BalancedMSEBMC.")

    def _sigma_for_forward(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """No-fit fallback: supplied sigma if available, otherwise one."""
        if bool(self._fitted_flag):
            return self.sigma.to(device=device, dtype=dtype)
        if self._sigma_init is not None:
            return torch.tensor(self._sigma_init, device=device, dtype=dtype)
        return torch.ones((), device=device, dtype=dtype)

    def _warn_missing_fit(self, sigma: float) -> None:
        if self._warned_no_fit:
            return
        warnings.warn(
            "BalancedMSEBMC.forward called before fit(train_targets); falling back to "
            f"the paper's uniform-prior BMC with sigma={sigma:g}. Call fit(...) to use "
            "the fitted label prior and the automatic noise-scale estimate.",
            UserWarning,
            stacklevel=2,
        )
        self._warned_no_fit = True

    @property
    def sigma(self) -> Tensor:
        """Label-noise standard deviation (fitted or supplied)."""
        if not bool(self._fitted_flag) and self._sigma_init is None:
            raise RuntimeError("Call fit(train_targets) before using BalancedMSEBMC.")
        return self._sigma

    def fit(
        self,
        train_targets: Tensor,
        train_predictions: Optional[Tensor] = None,
    ) -> BalancedMSEBMC:
        """Estimate the label prior and, when ``sigma`` was omitted, the noise scale.

        Parameters
        ----------
        train_targets:
            Training targets with shape ``[N]`` or ``[N, 1]``.  Non-finite rows
            are ignored (they carry no prior information).
        train_predictions:
            Optional predictions on the same training rows.  When ``sigma`` is
            omitted *and* predictions are supplied, ``sigma`` is estimated from
            the robust scale (MAD) of the residuals ``train_targets -
            train_predictions``.  This is the intended noise-adaptive estimate;
            use out-of-sample (or cross-fitted) predictions, because in-sample
            residuals underestimate the noise scale.  Without predictions, only
            the marginal target distribution is
            available, so ``sigma`` falls back to a robust **target-scale
            proxy**: it is not identifiable label noise, and in general it is
            an upper-bound-style heuristic rather than the paper's learned
            value.  Supply ``sigma`` when the noise scale is known.

        Returns
        -------
        BalancedMSEBMC
            ``self``, for chaining.

        Notes
        -----
        The paper makes ``sigma`` a single learnable parameter optimized with
        the model (Appendix B.5) and initializes it near the true noise scale;
        it does not select ``sigma`` by a minimum-variance-over-batch-sizes
        search.  This implementation uses a one-shot robust MAD estimate
        instead, which avoids adding a parameter to the optimizer but is not
        end-to-end optimal.  The MAD is a one-shot statistic (O(N) memory, no
        pairwise costs); it is not a minimum-variance search over batch sizes.
        """
        targets, _ = _scalar_targets(train_targets, "train_targets")
        targets = targets.detach()
        finite = torch.isfinite(targets)
        if not bool(finite.any()):
            raise ValueError("train_targets must contain at least one finite target.")
        targets_all = targets
        targets = targets[finite]

        fit_dtype = torch.float64 if targets.dtype == torch.float64 else torch.float32
        targets = targets.to(dtype=fit_dtype)

        if self.prior == "histogram":
            self._fit_histogram(targets)
        elif self.prior == "kde":
            self._fit_kde(targets)
        else:
            self._prior_centers = torch.empty(0, dtype=fit_dtype, device=targets.device)
            self._prior_log_density = torch.empty(0, dtype=fit_dtype, device=targets.device)
            self._kde_points = torch.empty(0, dtype=fit_dtype, device=targets.device)

        self._fitted_median = targets.median().to(dtype=torch.float32).clone()

        if self._sigma_init is not None:
            self._sigma = torch.tensor(self._sigma_init, dtype=torch.float32, device=targets.device)
        else:
            predictions: Optional[Tensor] = None
            if train_predictions is not None:
                predictions, _ = _scalar_targets(train_predictions, "train_predictions")
                predictions = predictions.detach()
                if predictions.numel() != targets_all.numel():
                    raise ValueError(
                        "train_predictions must have the same number of elements as train_targets."
                    )
                predictions = predictions.to(device=targets.device, dtype=fit_dtype)[finite]
                residuals = targets - predictions
                residuals = residuals[torch.isfinite(residuals)]
                if residuals.numel() == 0:
                    raise ValueError("No finite residuals available to estimate sigma.")
                sigma = _mad_scale(residuals)
            else:
                # Target-scale proxy: a robust marginal scale, not label noise.
                sigma = _mad_scale(targets)

            if not torch.isfinite(sigma) or float(sigma) <= 0.0:
                # Degenerate/constant targets: any sigma is admissible; use a
                # neutral scale instead of the 1e-6 floor.
                sigma = torch.clamp(targets.std(unbiased=False), min=1.0)
            sigma = torch.clamp(sigma, min=self.min_sigma)
            self._sigma = sigma.to(dtype=torch.float32).clone()

        self._fitted_flag.fill_(True)
        return self

    def _fit_histogram(self, y: Tensor) -> None:
        """Gaussian-smoothed, edge-corrected histogram label prior."""
        n_bins = self.num_bins
        lo = y.min()
        hi = y.max()
        span = hi - lo
        if not torch.isfinite(span) or float(span) <= 0.0:
            pad = torch.clamp(y.abs().max(), min=1.0)
            lo = lo - pad
            hi = hi + pad
        bin_width = (hi - lo) / n_bins
        edges = lo + bin_width * torch.arange(n_bins + 1, device=y.device, dtype=y.dtype)
        # ``bucketize`` with right=True gives bins in [0, n_bins]; the outer
        # clamp only matters for numerical ties at the upper edge.
        indices = (torch.bucketize(y, edges, right=True) - 1).clamp(0, n_bins - 1)
        counts = torch.zeros(n_bins, device=y.device, dtype=y.dtype)
        counts.scatter_add_(0, indices, torch.ones_like(y))

        centers = lo + (torch.arange(n_bins, device=y.device, dtype=y.dtype) + 0.5) * bin_width
        bandwidth = (
            1.5 * bin_width
            if self.prior_bandwidth is None
            else torch.as_tensor(self.prior_bandwidth, device=y.device, dtype=y.dtype)
        )
        if float(bandwidth) > 0.0:
            # Reflect the counts at both boundaries before convolving so edge
            # bins keep the kernel's mass and a flat histogram stays flat.
            pad = int(math.ceil(3.0 * float(bandwidth) / float(bin_width)))
            pad = max(1, min(pad, n_bins - 1))
            padded = F.pad(counts.view(1, 1, -1), (pad, pad), mode="reflect").view(-1)
            offsets = torch.arange(-pad, pad + 1, device=y.device, dtype=y.dtype)
            kernel = torch.exp(-0.5 * (offsets / (bandwidth / bin_width)).square())
            kernel = kernel / kernel.sum().clamp_min(torch.finfo(y.dtype).tiny)
            smoothed = F.conv1d(padded.view(1, 1, -1), kernel.view(1, 1, -1)).view(-1)
        else:
            smoothed = counts

        density = smoothed / (smoothed.sum() * bin_width).clamp_min(torch.finfo(y.dtype).tiny)
        floor = density.max() * self.prior_floor
        density = density.clamp_min(floor.clamp_min(torch.finfo(y.dtype).tiny))
        self._prior_centers = centers
        self._prior_bin_width = bin_width.clone()
        self._prior_log_density = density.log()
        self._prior_log_floor = floor.log().clone()
        self._kde_points = torch.empty(0, dtype=y.dtype, device=y.device)

    def _fit_kde(self, y: Tensor) -> None:
        """Gaussian KDE label prior with a deterministic reference subsample."""
        n = y.numel()
        if n > self.kde_points:
            indices = torch.linspace(0, n - 1, self.kde_points, device=y.device)
            points = y[indices.round().long().clamp(0, n - 1)]
        else:
            points = y
        if self.prior_bandwidth is None:
            std = points.std(unbiased=False)
            q25 = torch.quantile(points, 0.25)
            q75 = torch.quantile(points, 0.75)
            iqr_scale = (q75 - q25) / 1.349
            scale = torch.minimum(std, iqr_scale) if float(iqr_scale) > 0.0 else std
            bandwidth = 0.9 * scale * points.numel() ** (-0.2)
            if not torch.isfinite(bandwidth) or float(bandwidth) <= 0.0:
                bandwidth = torch.ones((), device=y.device, dtype=y.dtype)
        else:
            bandwidth = torch.as_tensor(self.prior_bandwidth, device=y.device, dtype=y.dtype)

        z = (points.unsqueeze(1) - points.unsqueeze(0)) / bandwidth
        log_density = (
            torch.logsumexp(-0.5 * z.square(), dim=1)
            - math.log(points.numel())
            - bandwidth.log()
            - 0.5 * math.log(2.0 * math.pi)
        )
        floor = log_density.max() + math.log(self.prior_floor)
        self._kde_points = points
        self._kde_bandwidth = bandwidth.clone()
        self._prior_log_floor = floor.clone()
        self._prior_centers = torch.empty(0, dtype=y.dtype, device=y.device)
        self._prior_log_density = torch.empty(0, dtype=y.dtype, device=y.device)

    # ------------------------------------------------------------------
    # Prior evaluation
    # ------------------------------------------------------------------
    def _log_prior_scalar(self, y: Tensor) -> Tensor:
        """Evaluate the fitted log prior at scalar targets ``y`` (no shaping)."""
        if self.prior == "uniform":
            return torch.zeros_like(y)
        if self.prior == "histogram":
            centers = self._prior_centers.to(device=y.device, dtype=y.dtype)
            log_density = self._prior_log_density.to(device=y.device, dtype=y.dtype)
            bin_width = self._prior_bin_width.to(device=y.device, dtype=y.dtype)
            # Linear interpolation of the log density between bin centres;
            # outside the fitted range the edge density is held constant.
            position = ((y - centers[0]) / bin_width).clamp(0.0, centers.numel() - 1.0)
            left = position.floor().long().clamp(0, centers.numel() - 2)
            fraction = position - left
            log_prior = (1.0 - fraction) * log_density[left] + fraction * log_density[left + 1]
            return torch.clamp_min(
                log_prior, self._prior_log_floor.to(device=y.device, dtype=y.dtype)
            )

        points = self._kde_points.to(device=y.device, dtype=y.dtype)
        bandwidth = self._kde_bandwidth.to(device=y.device, dtype=y.dtype)
        z = (y.unsqueeze(1) - points.unsqueeze(0)) / bandwidth
        log_prior = (
            torch.logsumexp(-0.5 * z.square(), dim=1)
            - math.log(points.numel())
            - bandwidth.log()
            - 0.5 * math.log(2.0 * math.pi)
        )
        return torch.clamp_min(log_prior, self._prior_log_floor.to(device=y.device, dtype=y.dtype))

    def log_prior(self, target: Tensor) -> Tensor:
        """Fitted log label prior evaluated at ``target``.

        The result has the same shape as ``target`` (``[B]`` or ``[B, 1]``).
        This is useful for diagnostics and for comparing the ``"histogram"``,
        ``"kde"``, and ``"uniform"`` choices.  It is detached and carries no
        gradient.
        """
        y, was_2d = _scalar_targets(target, "target")
        y = y.detach()
        if not bool(self._fitted_flag):
            if self.prior != "uniform":
                self._check_fitted()
            zeros = torch.zeros_like(y)
            return zeros.unsqueeze(-1) if was_2d else zeros
        dtype = _compute_dtype(y)
        log_prior = self._log_prior_scalar(y.to(dtype=dtype))
        return log_prior.unsqueeze(-1) if was_2d else log_prior

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the BMC loss.

        Parameters
        ----------
        y_pred:
            Predictions with shape ``[B]`` or ``[B, 1]``.
        target:
            Targets with the same shape.  Missing entries must be marked by
            ``mask``; non-finite targets without a mask raise ``ValueError``.
        mask:
            Boolean tensor with the shape of ``target``; ``False`` marks a
            missing entry.  Missing targets are excluded from the batch
            normaliser, so the loss of the remaining samples changes when a
            target is masked out.
        weights:
            Optional non-negative sample weights, following the shared
            reduction contract: ``reduction='mean'`` returns
            ``sum_i w_i L_i / sum_i w_i``.

        Returns
        -------
        Tensor
            Scalar for ``'mean'``/``'sum'``; per-sample loss for ``'none'``
            (shape ``[B]`` for 1-D inputs and ``[B, 1]`` for column inputs).
        """
        self._validate_inputs(y_pred, target, mask)
        pred, pred_was_2d = _scalar_targets(y_pred, "y_pred")
        tgt, tgt_was_2d = _scalar_targets(target, "target")
        if pred_was_2d != tgt_was_2d:
            raise ValueError("y_pred and target must have the same shape.")
        if pred.numel() == 0:
            raise ValueError("y_pred and target must be non-empty.")

        dtype = _compute_dtype(pred, tgt)
        pred_c = pred.to(dtype=dtype)
        tgt_c = tgt.to(dtype=dtype)

        valid = torch.isfinite(tgt_c)
        if mask is not None:
            valid = valid & _collapse_mask(mask, pred.numel()).to(device=valid.device)
        elif not bool(valid.all()):
            raise ValueError(
                "target contains non-finite values; mark missing entries with mask=..."
            )

        if bool(self._fitted_flag):
            fill = self._fitted_median.to(device=tgt_c.device, dtype=dtype)
            support = torch.where(valid, tgt_c, fill).detach()
            log_prior = self._log_prior_scalar(support)
        else:
            sigma_fallback = self._sigma_for_forward(device=pred_c.device, dtype=dtype)
            if not (self.prior == "uniform" and self._sigma_init is not None):
                self._warn_missing_fit(float(sigma_fallback))
            fill = torch.zeros((), device=tgt_c.device, dtype=dtype)
            support = torch.where(valid, tgt_c, fill).detach()
            log_prior = torch.zeros_like(support)
        log_prior = torch.where(valid, log_prior, torch.full_like(log_prior, -torch.inf))

        sigma = self._sigma_for_forward(device=pred_c.device, dtype=dtype)
        per_sample = _bmc_per_sample(pred_c, tgt_c, support, log_prior, sigma, valid)

        reduce_mask: Optional[Tensor] = (
            valid if (mask is not None or not bool(valid.all())) else None
        )
        if pred_was_2d:
            return self._reduce(per_sample.unsqueeze(-1), reduce_mask, weights)
        return self._reduce(per_sample, reduce_mask, weights)


def balanced_mse_bmc_loss(
    y_pred: Tensor,
    target: Tensor,
    log_prior: Tensor,
    sigma: float | Tensor,
    *,
    mask: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """Functional BMC loss with a precomputed log label prior.

    This is a stateless counterpart of :class:`BalancedMSEBMC` for callers that
    already have ``log p(y)`` for the batch targets (for example from an
    external density model).  The per-sample objective is

    .. math::

        L_i = -\\log p(y_i) + \\frac{(y_i - f_i)^2}{2\\sigma^2}
              + \\log \\sum_j p(y_j)
                \\exp\\!\\left(-\\frac{(f_i - y_j)^2}{2\\sigma^2}\\right).

    Parameters
    ----------
    y_pred, target:
        Scalar-per-sample tensors with shape ``[B]`` or ``[B, 1]``.
    log_prior:
        ``log p(y_j)`` for every batch target, shape ``[B]`` or ``[B, 1]``.
        Pass zeros for a uniform prior (the paper's BMC loss).
    sigma:
        Positive label-noise standard deviation.
    mask, weights, reduction:
        As in :meth:`BalancedMSEBMC.forward`.

    Returns
    -------
    Tensor
        Reduced loss following the shared torchregress mask/weight contract.
    """
    if y_pred.shape != target.shape:
        raise ValueError(f"target shape {target.shape} must match y_pred shape {y_pred.shape}.")
    pred, pred_was_2d = _scalar_targets(y_pred, "y_pred")
    tgt, tgt_was_2d = _scalar_targets(target, "target")
    prior, prior_was_2d = _scalar_targets(log_prior, "log_prior")
    if pred.numel() == 0:
        raise ValueError("y_pred and target must be non-empty.")
    if prior.numel() != pred.numel() or (prior_was_2d != pred_was_2d):
        raise ValueError("log_prior must have the same shape as target.")

    dtype = _compute_dtype(pred, tgt, prior)
    pred_c = pred.to(dtype=dtype)
    tgt_c = tgt.to(dtype=dtype)
    prior_c = prior.to(dtype=dtype)
    if bool(torch.isnan(prior_c).any()):
        raise ValueError("log_prior must not contain NaN.")

    sigma_tensor = torch.as_tensor(sigma, device=pred.device, dtype=dtype)
    if sigma_tensor.numel() != 1:
        raise ValueError("sigma must be a scalar.")
    sigma_tensor = sigma_tensor.reshape(())
    if not bool(torch.isfinite(sigma_tensor)) or float(sigma_tensor) <= 0.0:
        raise ValueError("sigma must be a positive finite number.")

    valid = torch.isfinite(tgt_c)
    if mask is not None:
        valid = valid & _collapse_mask(mask, pred.numel()).to(device=valid.device)
    elif not bool(valid.all()):
        raise ValueError("target contains non-finite values; mark missing entries with mask=...")

    fill = (
        tgt_c[valid].median()
        if bool(valid.any())
        else torch.zeros((), dtype=dtype, device=tgt_c.device)
    )
    support = torch.where(valid, tgt_c, fill).detach()
    log_prior_c = torch.where(valid, prior_c, torch.full_like(prior_c, -torch.inf))
    per_sample = _bmc_per_sample(pred_c, tgt_c, support, log_prior_c, sigma_tensor, valid)

    reduce_mask: Optional[Tensor] = valid if (mask is not None or not bool(valid.all())) else None
    reducer = BaseLoss(reduction=reduction)
    if pred_was_2d:
        return reducer._reduce(per_sample.unsqueeze(-1), reduce_mask, weights)
    return reducer._reduce(per_sample, reduce_mask, weights)
