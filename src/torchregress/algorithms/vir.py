r"""Variational imbalanced regression with probabilistic smoothing (VIR).

Regression on a continuous target whose training distribution is long-tailed
fails in two different ways: the bulk of the labels dominates the objective
(so predictions in the rare tails are poor), and a deterministic model gives
no usable uncertainty estimate where it is least reliable.  **Variational
Imbalanced Regression** (Wang & Wang, NeurIPS 2023) addresses both by coupling
a *probabilistic* latent representation to an *evidential* Normal--Inverse--Gamma
(NIG) predictive head and by reweighting each sample with the smoothed label
density.

Notation
--------
Let :math:`\{(x_i, y_i)\}_{i=1}^N` be the training data and partition the
target range into :math:`B` equal-width bins with edges
:math:`e_0 < \dots < e_B` and centres :math:`c_b = (e_{b-1}+e_b)/2`.  Write
:math:`b(i)` for the bin of :math:`y_i` and :math:`N_b` for its population.

1.  **Probabilistic representation.**  An encoder :math:`\mathcal I(\cdot)`
    maps :math:`x_i` to a diagonal Gaussian

    .. math::

        z_i \sim q_\phi(z_i \mid x_i)
            = \mathcal N\!\left(z_i;\, z_i^\mu,\, \operatorname{diag}(z_i^\Sigma)\right).

    Unlike a standard VAE, the representation of a *rare* label is allowed to
    borrow information from data with *neighbouring labels* (the N.I.D.
    assumption).

2.  **Probabilistic statistics and smoothing.**  For every bin,

    .. math::

        \mu_b^\mu = \frac{1}{N_b}\sum_{i \in b} z_i^\mu,
        \qquad
        \mu_b^\Sigma = \frac{1}{N_b^2}\sum_{i \in b} z_i^\Sigma,

    .. math::

        \Sigma_b^\mu = \frac{1}{N_b}\sum_{i \in b}
            \left[z_i^\Sigma + (z_i^\mu)^2\right]
            - \mu_b^\Sigma - (\mu_b^\mu)^2 .

    A symmetric, row-normalised Gaussian kernel :math:`k_{bb'}` over bin
    centres smooths these statistics across neighbouring labels:

    .. math::

        \tilde\mu_b^\mu = \sum_{b'} k_{bb'}\,\mu_{b'}^\mu,
        \qquad
        \tilde\Sigma_b^\mu = \sum_{b'} k_{bb'}\,\Sigma_{b'}^\mu,
        \qquad
        \tilde\mu_b^\Sigma = \sum_{b'} \bar k_{bb'}^2\,\mu_{b'}^\Sigma,

    where :math:`\bar k^2` is the squared kernel renormalised to sum to one
    (the variance of a weighted mean is the squared-weight sum).

3.  **Probabilistic whitening and recolouring.**  Each representation is
    whitened with its own bin's statistics and recoloured with the smoothed
    ones:

    .. math::

        \tilde z_i^\mu = (z_i^\mu - \mu_b^\mu)
            \sqrt{\tilde\Sigma_b^\mu / \Sigma_b^\mu} + \tilde\mu_b^\mu,

    .. math::

        \tilde z_i^\Sigma = (z_i^\Sigma + \mu_b^\Sigma)
            \sqrt{\tilde\Sigma_b^\mu / \Sigma_b^\mu} + \tilde\mu_b^\Sigma .

    During training :math:`z_i \sim \mathcal N(\tilde z_i^\mu,
    \operatorname{diag}(\tilde z_i^\Sigma))` is reparameterised.  The running
    statistics are refreshed from the current encoder at the start of every
    epoch and treated as constants; they are only needed at training time.
    Empty bins are excluded from the kernel (their statistics carry no
    information), and a count-dependent shrinkage protects singleton bins from
    collapsing onto their neighbours.

4.  **Probabilistic reweighting of the NIG posterior.**  The smoothed label
    density :math:`\tilde p_b = \sum_{b'} k_{bb'}\hat p_{b'}` defines the
    importance weight :math:`w_i = \tilde p_{b(i)}^{-1/2}`.  A small linear
    head emits a pseudo-count :math:`n_i`, a location :math:`\psi_i` and a
    positive scale :math:`\phi_i`.  With prior
    :math:`\mathrm{NIG}(\gamma_0, \nu_0, \alpha_0, \beta_0)` the reweighted
    posterior parameters are

    .. math::

        \nu_i^* = \nu_0 + w_i n_i,
        \qquad
        \gamma_i^* = \frac{\gamma_0\nu_0 + w_i n_i \psi_i}{\nu_i^*},

    .. math::

        \alpha_i^* = \alpha_0 + \tfrac12 w_i n_i,
        \qquad
        \beta_i^* = \beta_0 + \tfrac12\gamma_0^2\nu_0 + \phi_i .

    Because :math:`w_i` grows as the (smoothed) density falls, rare labels
    receive a larger pseudo-count and therefore a larger weight in the
    likelihood.  Marginalising the Gaussian likelihood over the NIG prior
    gives a Student-t predictive
    :math:`y_i \sim t_{2\alpha_i^*}(\gamma_i^*, \beta_i^*(1+\nu_i^*)/(\alpha_i^*\nu_i^*))`
    with negative log-likelihood

    .. math::

        \mathcal L_i^{\mathrm{NLL}}
            = \tfrac12 \log\frac{\pi}{\nu_i^*}
              - \alpha_i^* \log \Omega_i
              + (\alpha_i^* + \tfrac12)\log\!\left(
                    (y_i-\gamma_i^*)^2\nu_i^* + \Omega_i\right)
              + \log\frac{\Gamma(\alpha_i^*)}{\Gamma(\alpha_i^*+\tfrac12)},

    where :math:`\Omega_i = 2\beta_i^*(1+\nu_i^*)`.  The training objective
    adds the evidential regulariser of the paper,

    .. math::

        \mathcal L_i = \mathcal L_i^{\mathrm{NLL}}
            + \lambda\,(\nu_i^* + 2\alpha_i^*)\,|y_i-\gamma_i^*| .

The implementation here is the *discriminative core* of VIR: it keeps the
probabilistic representation, the neighbouring-label smoothing and the
reweighted NIG head, but it omits the input-reconstruction term of the full
variational ELBO.  This is deliberate -- the reconstruction term needs a
decoder and a likelihood for :math:`x`, which are task-specific, and does not
affect the predictive distribution used here.  :class:`VIRRegressor` exposes a
small optional KL weight on the latent representation to keep its scale
stable in the absence of the reconstruction term.

References
----------
.. [1] Wang, Z., & Wang, H. (2023). Variational Imbalanced Regression: Fair
       Uncertainty Quantification via Probabilistic Smoothing. In *NeurIPS
       2023*. https://arxiv.org/abs/2306.06599
.. [2] Amini, A., Schwarting, W., Soleimany, A., & Rus, D. (2020). Deep
       Evidential Regression. In *NeurIPS 2020*.
       https://arxiv.org/abs/1910.02600
.. [3] Yang, Y., Zha, K., Chen, Y.-C., Wang, H., & Katabi, D. (2021).
       Delving into Deep Imbalanced Regression. In *ICML 2021*.
       https://arxiv.org/abs/2102.09554
.. [4] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X.,
       Botvinick, M., Mohamed, S., & Lerchner, A. (2017). beta-VAE: Learning
       Basic Visual Concepts with a Constrained Variational Framework.
       In *ICLR 2017*.  (Reparameterisation and whitening/recolouring.)
"""

from __future__ import annotations

import copy
import math
from typing import Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from ..losses.base import DistributionLoss
from ..utils.validation import check_tensor

__all__ = ["VIRLoss", "VIRRegressor"]

#: Lower bound on the pseudo-count ``n`` used by the paper (``n >= 2``).
_MIN_PSEUDO_COUNT = 2.0
#: Numerical floor for variances.
_VARIANCE_FLOOR = 1e-8
#: Clamp for the whitening/recolouring scale ratio (as in FDS).
_RECOLOUR_RATIO_RANGE = (0.1, 10.0)


class _NIGPosterior(NamedTuple):
    """Reweighted Normal--Inverse--Gamma posterior parameters (per sample)."""

    gamma: Tensor
    nu: Tensor
    alpha: Tensor
    beta: Tensor
    importance_weight: Tensor


def _validate_prior(prior: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Validate the NIG prior ``(gamma0, nu0, alpha0, beta0)``."""
    if len(prior) != 4:
        raise ValueError(
            f"prior must be a 4-tuple (gamma0, nu0, alpha0, beta0), got {len(prior)} entries"
        )
    gamma0, nu0, alpha0, beta0 = (float(value) for value in prior)
    if not math.isfinite(gamma0):
        raise ValueError(f"prior gamma0 must be finite, got {gamma0}")
    if not (nu0 > 0.0 and math.isfinite(nu0)):
        raise ValueError(f"prior nu0 must be positive and finite, got {nu0}")
    if not (alpha0 > 1.0 and math.isfinite(alpha0)):
        raise ValueError(f"prior alpha0 must be > 1 for a finite predictive variance, got {alpha0}")
    if not (beta0 >= 0.0 and math.isfinite(beta0)):
        raise ValueError(f"prior beta0 must be non-negative and finite, got {beta0}")
    return gamma0, nu0, alpha0, beta0


def _gaussian_smoothing_matrix(
    n_bins: int,
    kernel_sigma: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Row-normalised Gaussian smoothing matrix over equally spaced bins.

    Args:
        n_bins: Number of label bins ``B``.
        kernel_sigma: Kernel bandwidth in **bin-width units**.
        dtype: Output dtype.
        device: Output device.

    Returns:
        ``(B, B)`` tensor whose rows sum to one.
    """
    index = torch.arange(n_bins, dtype=dtype, device=device)
    distance = index.reshape(-1, 1) - index.reshape(1, -1)
    kernel = torch.exp(-0.5 * (distance / kernel_sigma) ** 2)
    return kernel / kernel.sum(dim=1, keepdim=True)


def _bin_index_from_edges(edges: Tensor, values: Tensor, n_bins: int) -> Tensor:
    """Map target values to bins with right-closed interior edges."""
    cast_edges = edges.to(device=values.device, dtype=values.dtype)
    index = torch.bucketize(values, cast_edges, right=True) - 1
    return index.clamp(0, n_bins - 1)


def _flatten_target(target: Tensor) -> Tensor:
    """Return a per-sample float target vector, validating the layout."""
    if target.dim() == 1:
        return target
    if target.dim() == 2 and target.shape[1] == 1:
        return target.reshape(-1)
    raise ValueError(
        "VIR supports a single regression target; target must have shape (N,) or "
        f"(N, 1), got {tuple(target.shape)}"
    )


def _split_raw_output(y_pred: Tensor | tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, ...]:
    """Split raw NIG-head output into ``(raw_n, psi, raw_phi)`` vectors."""
    if isinstance(y_pred, (tuple, list)):
        if len(y_pred) != 3:
            raise ValueError(
                "tuple y_pred must contain exactly 3 tensors (raw_n, psi, raw_phi), "
                f"got {len(y_pred)}"
            )
        raw_n, psi, raw_phi = y_pred
    else:
        if y_pred.dim() != 2 or y_pred.shape[1] != 3:
            raise ValueError(
                "y_pred must have shape (N, 3) = [raw_n, psi, raw_phi] or be a "
                f"3-tuple of (N,) tensors, got {tuple(y_pred.shape)}"
            )
        raw_n = y_pred[:, 0]
        psi = y_pred[:, 1]
        raw_phi = y_pred[:, 2]
    raw_n = raw_n.reshape(-1)
    psi = psi.reshape(-1)
    raw_phi = raw_phi.reshape(-1)
    if not (raw_n.shape == psi.shape == raw_phi.shape):
        raise ValueError(
            "raw_n, psi and raw_phi must have the same number of samples, got "
            f"{tuple(raw_n.shape)}, {tuple(psi.shape)} and {tuple(raw_phi.shape)}"
        )
    return raw_n, psi, raw_phi


class VIRLoss(DistributionLoss):
    r"""Reweighted Normal--Inverse--Gamma evidential loss with label smoothing.

    The loss consumes the **raw** output of a 3-unit NIG head.  It turns that
    output into positive pseudo-counts and scales, reweights them with the
    smoothed label density and evaluates the Student-t marginal NLL plus the
    evidential regulariser:

    .. math::

        \mathcal L_i = \mathcal L_i^{\mathrm{NLL}}
            + \lambda\,(\nu_i^* + 2\alpha_i^*)\,|y_i-\gamma_i^*|.

    Labels are binned into ``n_bins`` equal-width bins by :meth:`fit`.  Call
    :meth:`fit` once on the training targets; if it is skipped the importance
    weights are all one and the loss reduces to a plain (DER-style) evidential
    NLL.  Bins are half-open ``[e_b, e_{b+1})`` and out-of-range values are
    clamped into the boundary bins, matching
    :class:`~torchregress.algorithms.group_dro.ContinuousGroupDRO`.

    Args:
        prior: ``(gamma0, nu0, alpha0, beta0)`` of the NIG prior.  Requires
            ``nu0 > 0``, ``alpha0 > 1`` (finite predictive variance) and
            ``beta0 >= 0``.
        reg_weight: Weight :math:`\lambda` of the evidential regulariser.
        n_bins: Number of equal-width label bins ``B``.
        kernel_sigma: Label-smoothing bandwidth in bin-width units.  ``1.0``
            means the neighbouring bin receives ``exp(-0.5)`` of the centre
            weight before row normalisation.
        min_nu: Additive floor on the pseudo-count ``n`` (the paper uses 2).
        min_beta: Additive floor on the positive scale ``phi``.
        max_importance_weight: Upper clamp on
            :math:`\tilde p_b^{-1/2}`.  The clamp keeps a nearly empty tail
            bin from dominating the objective; it also makes the loss robust
            to a badly estimated density.
        reduction: ``"mean"``, ``"sum"`` or ``"none"`` (standard loss
            contract).

    Attributes:
        bin_edges (Tensor): ``(B + 1,)`` label-bin edges (empty until
            :meth:`fit`).
        bin_weights (Tensor): ``(B,)`` importance weights per bin.
        bin_density (Tensor): ``(B,)`` smoothed training label density.

    Example:
        >>> import torch
        >>> from torchregress.algorithms.vir import VIRLoss
        >>>
        >>> loss_fn = VIRLoss(n_bins=8, kernel_sigma=1.0).fit(torch.randn(256))
        >>> y = torch.randn(16)
        >>> raw = torch.randn(16, 3, requires_grad=True)
        >>> loss = loss_fn(raw, y)
        >>> loss.backward()
        >>> torch.isfinite(loss)
        tensor(True)

    Notes:
        * The importance weights are computed from the **training targets**
          and detached; gradients flow only through the NIG head output.
        * :math:`\\lambda` trades the NLL against the evidence penalty.  A
          large :math:`\\lambda` makes the model under-confident; the default
          ``0.01`` follows the evidential-regression literature.
        * Only a single regression target is supported.  Multi-output targets
          need one NIG head per output.
    """

    bin_edges: Tensor
    bin_weights: Tensor
    bin_density: Tensor

    def __init__(
        self,
        *,
        prior: tuple[float, float, float, float] = (0.0, 1.0, 2.0, 0.1),
        reg_weight: float = 0.01,
        n_bins: int = 20,
        kernel_sigma: float = 1.0,
        min_nu: float = _MIN_PSEUDO_COUNT,
        min_beta: float = 1e-6,
        max_importance_weight: float = 10.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.prior = _validate_prior(prior)
        if not (float(reg_weight) >= 0.0 and math.isfinite(float(reg_weight))):
            raise ValueError(f"reg_weight must be non-negative and finite, got {reg_weight}")
        if int(n_bins) < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        if not (float(kernel_sigma) > 0.0 and math.isfinite(float(kernel_sigma))):
            raise ValueError(f"kernel_sigma must be positive and finite, got {kernel_sigma}")
        if not (float(min_nu) >= 0.0 and math.isfinite(float(min_nu))):
            raise ValueError(f"min_nu must be non-negative and finite, got {min_nu}")
        if not (float(min_beta) > 0.0 and math.isfinite(float(min_beta))):
            raise ValueError(f"min_beta must be positive and finite, got {min_beta}")
        if not (float(max_importance_weight) >= 1.0):
            raise ValueError(f"max_importance_weight must be >= 1, got {max_importance_weight}")

        self.reg_weight = float(reg_weight)
        self.n_bins = int(n_bins)
        self.kernel_sigma = float(kernel_sigma)
        self.min_nu = float(min_nu)
        self.min_beta = float(min_beta)
        self.max_importance_weight = float(max_importance_weight)

        self.register_buffer("bin_edges", torch.empty(0, dtype=torch.float64))
        self.register_buffer("bin_weights", torch.empty(0, dtype=torch.float64))
        self.register_buffer("bin_density", torch.empty(0, dtype=torch.float64))

    # ── label density ──────────────────────────────────────────────────
    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has estimated the label density."""
        return self.bin_edges.numel() == self.n_bins + 1

    def fit(self, targets: Tensor) -> VIRLoss:
        """Estimate equal-width label bins and the smoothed label density.

        Args:
            targets: Training targets of shape ``(N,)`` or ``(N, 1)``.
                Non-finite entries are ignored for the density estimate.

        Returns:
            ``self``, fitted.
        """
        values = _flatten_target(targets.detach()).to(device="cpu", dtype=torch.float64)
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            raise ValueError("VIRLoss.fit() needs at least one finite training target")
        low = float(finite.min().item())
        high = float(finite.max().item())
        if high <= low:
            pad = 1e-3 * max(1.0, abs(low))
            low, high = low - pad, high + pad

        edges = torch.linspace(low, high, self.n_bins + 1, dtype=torch.float64)
        index = _bin_index_from_edges(edges, finite, self.n_bins)
        counts = torch.bincount(index, minlength=self.n_bins).to(torch.float64)
        density = counts / counts.sum().clamp_min(1.0)

        kernel = _gaussian_smoothing_matrix(
            self.n_bins, self.kernel_sigma, dtype=torch.float64, device=torch.device("cpu")
        )
        smoothed = kernel @ density
        smoothed = smoothed.clamp_min(torch.finfo(torch.float64).tiny)
        weights = smoothed.rsqrt().clamp(max=self.max_importance_weight)

        self.bin_edges = edges
        self.bin_density = smoothed
        self.bin_weights = weights
        return self

    def bin_index(self, target: Tensor) -> Tensor:
        """Assign per-sample targets to label bins (no side effects).

        Args:
            target: Targets of shape ``(N,)`` or ``(N, 1)``.

        Returns:
            Integer tensor of shape ``(N,)`` in ``[0, B)``.  Non-finite
            targets are reported as bin 0 (they should be masked in the loss).
        """
        if not self.is_fitted:
            raise RuntimeError("VIRLoss must be fit() before calling bin_index()")
        values = _flatten_target(target.detach())
        finite = torch.isfinite(values)
        safe = torch.where(
            finite, values, torch.zeros((), dtype=values.dtype, device=values.device)
        )
        index = _bin_index_from_edges(self.bin_edges, safe, self.n_bins)
        return torch.where(finite, index, torch.zeros_like(index))

    def importance_weights(self, target: Tensor) -> Tensor:
        """Per-sample importance weights :math:`\\tilde p_{b(i)}^{-1/2}`.

        Args:
            target: Targets of shape ``(N,)`` or ``(N, 1)``.

        Returns:
            ``(N,)`` tensor of positive weights, detached.  Non-finite
            targets receive the neutral weight ``1``.  Before :meth:`fit` all
            weights are one.
        """
        values = _flatten_target(target.detach())
        if not self.is_fitted:
            return torch.ones_like(values)
        finite = torch.isfinite(values)
        index = self.bin_index(values)
        weights = self.bin_weights.to(device=values.device, dtype=values.dtype)[index]
        return torch.where(finite, weights, torch.ones_like(weights))

    def smoothing_matrix(self) -> Tensor:
        """``(B, B)`` row-normalised Gaussian smoothing matrix.

        The matrix follows the dtype and device of the fitted label bins.
        """
        return _gaussian_smoothing_matrix(
            self.n_bins,
            self.kernel_sigma,
            dtype=self.bin_edges.dtype,
            device=self.bin_edges.device,
        )

    # ── NIG posterior and predictive distribution ──────────────────────
    def posterior_parameters(
        self,
        y_pred: Tensor | tuple[Tensor, Tensor, Tensor],
        importance_weight: Tensor | None = None,
    ) -> _NIGPosterior:
        """Map raw NIG-head output to reweighted posterior parameters.

        Args:
            y_pred: Raw head output ``(N, 3)`` (or a 3-tuple).  Columns are
                an unconstrained pseudo-count, a location and an unconstrained
                positive scale.
            importance_weight: Optional per-sample weights overriding the
                smoothed label density (useful in a custom loop or for
                ablations).  Shape ``(N,)``.

        Returns:
            A :class:`_NIGPosterior` with ``gamma``, ``nu``, ``alpha``,
            ``beta`` and the effective ``importance_weight``; all are
            ``(N,)`` tensors.
        """
        raw_n, psi, raw_phi = _split_raw_output(y_pred)
        reference = raw_n
        if importance_weight is None:
            weight = torch.ones_like(reference)
        else:
            weight = importance_weight.reshape(-1).to(
                device=reference.device, dtype=reference.dtype
            )
            if weight.shape != reference.shape:
                raise ValueError(
                    f"importance_weight must have {reference.numel()} entries, got {weight.numel()}"
                )

        pseudo_count = F.softplus(raw_n) + self.min_nu
        phi = F.softplus(raw_phi) + self.min_beta
        gamma0, nu0, alpha0, beta0 = self.prior

        weight = weight.clamp_min(torch.finfo(reference.dtype).tiny)
        weighted_count = weight * pseudo_count
        nu = nu0 + weighted_count
        gamma = (gamma0 * nu0 + weighted_count * psi) / nu
        alpha = alpha0 + 0.5 * weighted_count
        beta = beta0 + 0.5 * gamma0 * gamma0 * nu0 + phi
        return _NIGPosterior(gamma=gamma, nu=nu, alpha=alpha, beta=beta, importance_weight=weight)

    @staticmethod
    def predictive_mean_std(
        posterior: _NIGPosterior,
    ) -> tuple[Tensor, Tensor]:
        """Mean and total standard deviation of the Student-t predictive.

        The NIG marginal predictive is
        :math:`t_{2\\alpha}(\\gamma, \\beta(1+\\nu)/(\\alpha\\nu))`; its
        variance is :math:`\\beta(1+\\nu)/(\\nu(\\alpha-1))`, i.e. the sum of
        the aleatoric and epistemic variances.  ``alpha > 1`` is guaranteed by
        the prior validation.

        Returns:
            ``(mean, std)`` with shapes ``(N,)`` and ``(N,)``.
        """
        variance = posterior.beta * (1.0 + posterior.nu) / (posterior.nu * (posterior.alpha - 1.0))
        return posterior.gamma, variance.clamp_min(_VARIANCE_FLOOR).sqrt()

    @staticmethod
    def _nig_nll(target: Tensor, posterior: _NIGPosterior) -> Tensor:
        r"""Student-t marginal NLL of the reweighted NIG posterior."""
        residual_sq = (target - posterior.gamma) ** 2
        omega = 2.0 * posterior.beta * (1.0 + posterior.nu)
        return (
            0.5 * torch.log(math.pi / posterior.nu)
            - posterior.alpha * torch.log(omega)
            + (posterior.alpha + 0.5) * torch.log(posterior.nu * residual_sq + omega)
            + torch.lgamma(posterior.alpha)
            - torch.lgamma(posterior.alpha + 0.5)
        )

    def _elementwise_loss(
        self,
        y_pred: Tensor | tuple[Tensor, Tensor, Tensor],
        target: Tensor,
        importance_weight: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(loss, nll)`` as ``(N,)`` tensors."""
        flat_target = _flatten_target(target)
        posterior = self.posterior_parameters(y_pred, importance_weight)
        if posterior.gamma.shape != flat_target.shape:
            raise ValueError(
                "y_pred and target must contain the same number of samples, got "
                f"{posterior.gamma.numel()} and {flat_target.numel()}"
            )
        nll = self._nig_nll(flat_target, posterior)
        regulariser = (posterior.nu + 2.0 * posterior.alpha) * (flat_target - posterior.gamma).abs()
        return nll + self.reg_weight * regulariser, nll

    # ── loss interface ─────────────────────────────────────────────────
    def forward(
        self,
        y_pred: Tensor | tuple[Tensor, Tensor, Tensor],
        target: Tensor,
        importance_weight: Tensor | None = None,
        mask: Tensor | None = None,
        weights: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the VIR evidential loss.

        Args:
            y_pred: Raw NIG-head output of shape ``(N, 3)`` (or a 3-tuple of
                ``(N,)`` tensors).
            target: Targets of shape ``(N,)`` or ``(N, 1)``.
            importance_weight: Optional per-sample reweighting overriding the
                fitted smoothed density.  Shape ``(N,)``.
            mask: Optional boolean mask (``False`` = missing).  Masked entries
                contribute exactly zero and are excluded from the mean.
            weights: Optional non-negative user sample weights following the
                shared reduction contract.
            **kwargs: Ignored (accepted for API compatibility).

        Returns:
            Scalar under ``reduction="mean"``/``"sum"``; per-sample losses
            with the same shape as ``target`` under ``"none"``.
        """
        if importance_weight is None and self.is_fitted:
            # Use the smoothed training label density by default.  The weights
            # are detached (they depend only on the target), so gradients flow
            # through the NIG head alone.
            importance_weight = self.importance_weights(target)
        loss, _ = self._elementwise_loss(y_pred, target, importance_weight)
        shaped = loss.reshape(target.shape) if target.dim() > 1 else loss
        return self._reduce(shaped, mask, weights)

    def predict(
        self,
        y_pred: Tensor | tuple[Tensor, Tensor, Tensor],
        *,
        importance_weight: Tensor | None = None,
        return_std: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Predictive mean (and optionally standard deviation) from raw output.

        This is a convenience for a custom training loop; :class:`VIRRegressor`
        wraps it together with the encoder and latent smoothing.

        Args:
            y_pred: Raw NIG-head output ``(N, 3)`` (or a 3-tuple).
            importance_weight: Optional per-sample reweighting.  Defaults to
                one, which is the natural choice when the label density is
                unavailable; :class:`VIRRegressor` evaluates the density at its
                own predicted label instead.
            return_std: When ``True`` return ``(mean, std)``.
        """
        posterior = self.posterior_parameters(y_pred, importance_weight)
        mean, std = self.predictive_mean_std(posterior)
        return (mean, std) if return_std else mean


class _VIRNetwork(nn.Module):
    """Encoder + probabilistic latent head + NIG head."""

    def __init__(self, encoder: nn.Module, feature_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_head = nn.Linear(feature_dim, 2 * latent_dim)
        self.nig_head = nn.Linear(latent_dim, 3)

    def encode(self, X: Tensor) -> tuple[Tensor, Tensor]:
        """Return the diagonal-Gaussian latent ``(mean, variance)``."""
        features = self.encoder(X)
        if not isinstance(features, Tensor):
            raise TypeError(
                "encoder must return a torch.Tensor, got "
                f"{type(features).__name__}; wrap the feature extractor in a module "
                "with a tensor forward()"
            )
        if features.dim() != 2:
            raise ValueError(
                f"encoder output must be 2-D (N, F), got shape {tuple(features.shape)}"
            )
        latent = self.latent_head(features)
        mean, raw_var = latent.chunk(2, dim=-1)
        variance = F.softplus(raw_var) + _VARIANCE_FLOOR
        return mean, variance

    def head(self, latent: Tensor) -> Tensor:
        """Raw NIG parameters from a latent representation."""
        raw: Tensor = self.nig_head(latent)
        return raw


class VIRRegressor:
    r"""End-to-end variational imbalanced regressor with an NIG predictive head.

    The caller supplies a **feature encoder** (any ``nn.Module`` mapping
    ``(N, D)`` inputs to ``(N, F)`` features).  :class:`VIRRegressor` appends a
    probabilistic latent head and a 3-unit NIG head, applies the
    neighbouring-label smoothing of VIR during training and fits everything
    end-to-end with Adam.

    Training proceeds in epochs.  During the first epoch smoothing is the
    identity; from the second epoch on, the per-bin statistics (overall and
    kernel-smoothed) are recomputed from the current encoder at the start of
    each epoch and used to whiten and recolour each representation, as in the
    paper's "statistics of statistics".  The NIG head is trained on the
    recoloured representation with the label-density reweighting of
    :class:`VIRLoss`.

    At prediction time the training labels are unavailable, so the bin of a
    test point is unknown.  With ``inference_smoothing="two_pass"`` (the
    default) a first pass on the raw representation predicts a label, its bin
    selects the stored training statistics, and the NIG head is evaluated on
    the recoloured representation; ``n_refine`` controls how many times the
    bin is refreshed.  ``inference_smoothing="none"`` skips this step and is
    kept for ablations -- it creates a train/test mismatch because the head
    was trained on recoloured features.

    Args:
        encoder: Feature encoder.  It is **deep-copied** before fitting, so
            the caller's module is never modified.
        feature_dim: Output width of ``encoder``.  When ``None``, an
            ``out_features`` attribute is used if present, otherwise one
            forward pass on a small probe batch infers it.
        latent_dim: Width of the probabilistic latent representation.  When
            ``None`` it equals ``feature_dim``.
        n_bins: Number of equal-width label bins.
        kernel_sigma: Label-smoothing bandwidth in bin-width units.
        smoothing_shrinkage: Pseudo-count :math:`\\kappa` of the
            empirical-Bayes blend between the raw and the recoloured
            representation, ``N_b / (N_b + kappa)``.  ``0`` gives the paper's
            full recolouring; a positive value protects sparse/singleton bins
            from collapsing onto their neighbours.
        prior: ``(gamma0, nu0, alpha0, beta0)`` NIG prior.
        reg_weight: Evidential regulariser weight :math:`\lambda`.
        max_importance_weight: Clamp on the density reweighting.
        kl_weight: Weight of :math:`\mathrm{KL}(q(z\mid x)\,\|\,\mathcal N(0,I))`
            on the raw probabilistic representation.  The paper's full ELBO
            balances this KL with an input-reconstruction term; without the
            decoder it is a scale regulariser, so ``0`` is allowed but can let
            the latent scale drift.
        epochs: Number of training epochs.
        batch_size: Mini-batch size (clipped to the training-set size).
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        inference_smoothing: ``"two_pass"`` or ``"none"``.
        n_refine: Number of bin-refresh passes in ``"two_pass"`` mode.
        seed: Seed for head initialisation, shuffling and latent noise.  With
            a seed, repeated fits on the same encoder/data are deterministic
            and no global RNG state is consumed.
        device: Optional device; defaults to the device of ``X``.
        verbose: Print the mean training loss every epoch.

    Attributes:
        model_: Fitted ``nn.Module`` (encoder + latent head + NIG head), in
            ``eval`` mode.
        loss_fn_: Fitted :class:`VIRLoss` (its density is reused at inference).
        history_: One dict per epoch with the mean batch loss.
        bin_counts_: ``(B,)`` training sample count per label bin.
        bin_mean_, bin_cov_, bin_mean_var_: Per-bin latent statistics
            :math:`\mu_b^\mu`, :math:`\Sigma_b^\mu`,
            :math:`\mu_b^\Sigma`.
        smoothed_bin_mean_, smoothed_bin_cov_, smoothed_bin_mean_var_:
            Kernel-smoothed counterparts used for recolouring.
        smoothing_used_: Whether training ever used the recoloured
            representation (``epochs > 1``).  Inference smoothing is skipped
            when it is ``False`` to avoid a train/test mismatch.

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from torchregress.algorithms.vir import VIRRegressor
        >>>
        >>> _ = torch.manual_seed(0)
        >>> y = torch.cat([0.4 * torch.randn(500), 2.5 * torch.randn(50)])
        >>> X = torch.stack([torch.sin(y), torch.cos(y)], dim=1)
        >>> X = X + 0.05 * torch.randn_like(X)
        >>> encoder = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 32))
        >>>
        >>> reg = VIRRegressor(encoder, n_bins=12, epochs=60, seed=0)
        >>> _ = reg.fit(X, y)
        >>> mean, std = reg.predict(X[:5], return_std=True)
        >>> mean.shape, std.shape
        (torch.Size([5, 1]), torch.Size([5, 1]))
        >>> _ = reg.predict_interval(X[:5], confidence=0.9)

    Notes:
        * The encoder is cloned at :meth:`fit`; retrain by calling
          :meth:`fit` again (a fresh copy is taken each time).
        * This class implements the discriminative core of VIR.  It does not
          model ``p(x | z)`` and therefore has no reconstruction/KL term; the
          latent smoothing is a training-time regulariser, not a posterior
          over the encoder weights.  Uncertainty is the NIG predictive
          uncertainty (aleatoric + evidential epistemic); it is **not** a
          Bayesian model-uncertainty decomposition.
        * Only single-output regression is supported and non-finite
          inputs/targets are rejected.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        feature_dim: int | None = None,
        latent_dim: int | None = None,
        n_bins: int = 20,
        kernel_sigma: float = 1.0,
        smoothing_shrinkage: float = 5.0,
        prior: tuple[float, float, float, float] = (0.0, 1.0, 2.0, 0.1),
        reg_weight: float = 0.01,
        max_importance_weight: float = 10.0,
        kl_weight: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        inference_smoothing: str = "two_pass",
        n_refine: int = 1,
        seed: int | None = None,
        device: str | torch.device | None = None,
        verbose: bool = False,
    ) -> None:
        if not isinstance(encoder, nn.Module):
            raise TypeError(f"encoder must be an nn.Module, got {type(encoder).__name__}")
        if feature_dim is not None and int(feature_dim) < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
        if latent_dim is not None and int(latent_dim) < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if int(epochs) < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not (float(lr) > 0.0 and math.isfinite(float(lr))):
            raise ValueError(f"lr must be positive and finite, got {lr}")
        if not (float(weight_decay) >= 0.0):
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        smoothing = str(inference_smoothing).lower()
        if smoothing not in {"two_pass", "none"}:
            raise ValueError(
                f"inference_smoothing must be 'two_pass' or 'none', got {inference_smoothing!r}"
            )
        if int(n_refine) < 0:
            raise ValueError(f"n_refine must be >= 0, got {n_refine}")
        if not (float(kl_weight) >= 0.0 and math.isfinite(float(kl_weight))):
            raise ValueError(f"kl_weight must be non-negative and finite, got {kl_weight}")
        if not (float(smoothing_shrinkage) >= 0.0 and math.isfinite(float(smoothing_shrinkage))):
            raise ValueError(
                f"smoothing_shrinkage must be non-negative and finite, got {smoothing_shrinkage}"
            )

        self.encoder = encoder
        self.feature_dim = None if feature_dim is None else int(feature_dim)
        self.latent_dim = None if latent_dim is None else int(latent_dim)
        self.prior = _validate_prior(prior)
        self.reg_weight = float(reg_weight)
        self.max_importance_weight = float(max_importance_weight)
        self.kl_weight = float(kl_weight)
        self.n_bins = int(n_bins)
        self.kernel_sigma = float(kernel_sigma)
        self.smoothing_shrinkage = float(smoothing_shrinkage)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.inference_smoothing = smoothing
        self.n_refine = int(n_refine)
        self.seed = None if seed is None else int(seed)
        self.device = device
        self.verbose = bool(verbose)

        # Validate the remaining hyper-parameters through VIRLoss, which owns
        # the same semantics.  A throw-away instance is cheap.
        VIRLoss(
            prior=prior,
            reg_weight=self.reg_weight,
            n_bins=self.n_bins,
            kernel_sigma=self.kernel_sigma,
            max_importance_weight=self.max_importance_weight,
        )

        self.model_: _VIRNetwork | None = None
        self.loss_fn_: VIRLoss | None = None
        self.history_: list[dict[str, Any]] = []
        self.bin_counts_: Tensor | None = None
        self.bin_mean_: Tensor | None = None
        self.bin_cov_: Tensor | None = None
        self.bin_mean_var_: Tensor | None = None
        self.smoothed_bin_mean_: Tensor | None = None
        self.smoothed_bin_cov_: Tensor | None = None
        self.smoothed_bin_mean_var_: Tensor | None = None
        self.smoothing_used_ = False
        self._fitted = False

    # ── helpers ────────────────────────────────────────────────────────
    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return self._fitted

    def _resolve_device(self, X: Tensor) -> torch.device:
        return torch.device(self.device) if self.device is not None else X.device

    def _infer_feature_dim(self, X: Tensor, device: torch.device, dtype: torch.dtype) -> int:
        if self.feature_dim is not None:
            return self.feature_dim
        out_features = getattr(self.encoder, "out_features", None)
        if isinstance(out_features, int) and out_features > 0:
            return out_features
        probe_encoder = copy.deepcopy(self.encoder).to(device=device, dtype=dtype)
        probe_encoder.eval()
        with torch.no_grad():
            probe = probe_encoder(X[: min(8, X.shape[0])])
        if not isinstance(probe, Tensor) or probe.dim() != 2:
            raise ValueError(
                "could not infer feature_dim from the encoder; pass feature_dim explicitly"
            )
        return int(probe.shape[1])

    @staticmethod
    def _validate_fit_inputs(X: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        check_tensor(X, "X")
        check_tensor(y, "y")
        if X.dim() != 2:
            raise ValueError(f"X must be a 2-D tensor (N, D), got shape {tuple(X.shape)}")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        elif y.dim() != 2 or y.shape[1] != 1:
            raise ValueError(f"y must have shape (N,) or (N, 1), got shape {tuple(y.shape)}")
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}"
            )
        if not X.is_floating_point() or not y.is_floating_point():
            raise TypeError("X and y must be floating-point tensors")
        if X.dtype not in {torch.float32, torch.float64}:
            raise TypeError(
                f"VIR supports float32/float64 inputs, got dtype {X.dtype}; cast X and y first"
            )
        if not bool(torch.isfinite(X).all()):
            raise ValueError("X contains NaN or Inf; VIR does not accept missing features")
        if not bool(torch.isfinite(y).all()):
            raise ValueError("y contains NaN or Inf; VIR does not accept missing targets")
        return X, y

    def _build_model(self, X: Tensor, device: torch.device, dtype: torch.dtype) -> _VIRNetwork:
        feature_dim = self._infer_feature_dim(X, device, dtype)
        latent_dim = feature_dim if self.latent_dim is None else self.latent_dim
        # Local RNG scope: deterministic head init without touching global state.
        with torch.random.fork_rng(devices=[]):
            if self.seed is not None:
                torch.manual_seed(self.seed)
            model = _VIRNetwork(copy.deepcopy(self.encoder), feature_dim, latent_dim)
        # Start with a small latent variance: the default softplus(Linear)
        # initialisation gives a variance of ~0.7, so the reparameterisation
        # noise would overwhelm the signal and the NIG head would underfit.
        with torch.no_grad():
            model.latent_head.bias[latent_dim:].fill_(-4.0)
        return model.to(device=device, dtype=dtype)

    @staticmethod
    def _summarise_bin_statistics(
        sum_mean: Tensor,
        sum_variance: Tensor,
        sum_second: Tensor,
        counts: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Per-bin ``(mean, mean_variance, covariance)`` with empty-bin imputation."""
        occupied = counts > 0
        safe_counts = counts.clamp_min(1.0)
        mean = sum_mean / safe_counts.reshape(-1, 1)
        mean_variance = sum_variance / (safe_counts * safe_counts).reshape(-1, 1)
        covariance = sum_second / safe_counts.reshape(-1, 1) - mean_variance - mean * mean
        covariance = covariance.clamp_min(_VARIANCE_FLOOR)
        mean_variance = mean_variance.clamp_min(0.0)

        total = counts.sum().clamp_min(1.0)
        global_mean = sum_mean.sum(dim=0) / total
        global_mean_variance = sum_variance.sum(dim=0) / (total * total)
        global_covariance = (
            sum_second.sum(dim=0) / total - global_mean_variance - global_mean * global_mean
        ).clamp_min(_VARIANCE_FLOOR)

        mean = torch.where(occupied.reshape(-1, 1), mean, global_mean)
        mean_variance = torch.where(occupied.reshape(-1, 1), mean_variance, global_mean_variance)
        covariance = torch.where(occupied.reshape(-1, 1), covariance, global_covariance)
        return mean, mean_variance, covariance

    @staticmethod
    def _smooth_bin_statistics(
        stats: tuple[Tensor, Tensor, Tensor],
        kernel: Tensor,
        counts: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Kernel-smooth ``(mean, mean_variance, covariance)`` across occupied bins.

        Empty bins are imputed with the global statistics for *indexing*
        reasons, but they must not contribute to the smoothing: a zero-count
        bin carries no information and would otherwise drag the extreme bins
        toward the global (bulk-dominated) mean.  The kernel is therefore
        renormalised over the occupied bins of each row.
        """
        mean, mean_variance, covariance = stats
        occupancy = (counts > 0).to(kernel.dtype).reshape(1, -1)
        weights = kernel * occupancy
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(kernel.dtype).tiny
        )
        squared = weights * weights
        squared = squared / squared.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(kernel.dtype).tiny
        )
        smoothed_mean = weights @ mean
        smoothed_mean_variance = squared @ mean_variance
        smoothed_covariance = weights @ covariance
        return smoothed_mean, smoothed_mean_variance, smoothed_covariance

    @staticmethod
    def _recolour(
        mean: Tensor,
        variance: Tensor,
        bin_index: Tensor,
        stats: tuple[Tensor, Tensor, Tensor],
        smoothed: tuple[Tensor, Tensor, Tensor],
        counts: Tensor,
        shrinkage: float,
    ) -> tuple[Tensor, Tensor]:
        """Probabilistic whitening and recolouring with stored bin statistics.

        The recoloured representation is blended with the raw one using the
        empirical-Bayes weight :math:`N_b/(N_b+\\kappa)`.  With large bins
        (the paper's regime) this is the identity on the recoloured value; for
        a singleton bin it keeps most of the sample's own representation
        instead of collapsing it onto the neighbour-smoothed mean.
        """
        bin_mean, bin_mean_variance, bin_covariance = stats
        smooth_mean, smooth_mean_variance, smooth_covariance = smoothed
        mean_b = bin_mean.index_select(0, bin_index)
        mean_variance_b = bin_mean_variance.index_select(0, bin_index)
        covariance_b = bin_covariance.index_select(0, bin_index)
        smooth_mean_b = smooth_mean.index_select(0, bin_index)
        smooth_mean_variance_b = smooth_mean_variance.index_select(0, bin_index)
        smooth_covariance_b = smooth_covariance.index_select(0, bin_index)

        ratio = (smooth_covariance_b / covariance_b).clamp(*_RECOLOUR_RATIO_RANGE)
        root = ratio.sqrt()
        recoloured_mean = (mean - mean_b) * root + smooth_mean_b
        recoloured_variance = (variance + mean_variance_b) * root + smooth_mean_variance_b

        if shrinkage > 0.0:
            count_b = counts.index_select(0, bin_index).to(mean.dtype)
            weight = (count_b / (count_b + shrinkage)).reshape(-1, 1)
            recoloured_mean = (1.0 - weight) * mean + weight * recoloured_mean
            recoloured_variance = (1.0 - weight) * variance + weight * recoloured_variance
        return recoloured_mean, recoloured_variance.clamp_min(_VARIANCE_FLOOR)

    def _collect_statistics(
        self,
        model: _VIRNetwork,
        X: Tensor,
        y: Tensor,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], Tensor]:
        """Run the encoder over the training set and aggregate per-bin statistics."""
        model.eval()
        latent_dim = model.latent_head.out_features // 2
        device = X.device
        dtype = X.dtype
        sum_mean = torch.zeros(self.n_bins, latent_dim, dtype=dtype, device=device)
        sum_variance = torch.zeros_like(sum_mean)
        sum_second = torch.zeros_like(sum_mean)
        counts = torch.zeros(self.n_bins, dtype=dtype, device=device)
        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                X_batch = X[start : start + self.batch_size]
                mean, variance = model.encode(X_batch)
                index = self._bin_index(y[start : start + self.batch_size])
                sum_mean.index_add_(0, index, mean)
                sum_variance.index_add_(0, index, variance)
                sum_second.index_add_(0, index, variance + mean * mean)
                counts.index_add_(0, index, torch.ones_like(index, dtype=dtype))
        stats = self._summarise_bin_statistics(sum_mean, sum_variance, sum_second, counts)
        return stats, counts

    def _bin_index(self, target: Tensor) -> Tensor:
        if self.loss_fn_ is None:
            raise RuntimeError("VIRRegressor must be fit before binning targets")
        return self.loss_fn_.bin_index(target)

    # ── training ───────────────────────────────────────────────────────
    def fit(self, X: Tensor, y: Tensor) -> VIRRegressor:
        """Fit the encoder, latent smoother and NIG head with Adam.

        Args:
            X: Inputs of shape ``(N, D)``; finite floating-point.
            y: Targets of shape ``(N,)`` or ``(N, 1)``; finite.

        Returns:
            ``self``, fitted.
        """
        X, y = self._validate_fit_inputs(X, y)
        device = self._resolve_device(X)
        X = X.to(device=device)
        y = y.to(device=device, dtype=X.dtype)
        dtype = X.dtype

        model = self._build_model(X, device, dtype)
        loss_fn = VIRLoss(
            prior=self.prior,
            reg_weight=self.reg_weight,
            n_bins=self.n_bins,
            kernel_sigma=self.kernel_sigma,
            max_importance_weight=self.max_importance_weight,
        ).fit(y)
        loss_fn = loss_fn.to(device=device)
        kernel = _gaussian_smoothing_matrix(
            self.n_bins, self.kernel_sigma, dtype=dtype, device=device
        )

        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n_samples = X.shape[0]
        batch_size = min(self.batch_size, n_samples)

        shuffle_generator = torch.Generator(device="cpu")
        noise_generator = torch.Generator(device=device)
        if self.seed is not None:
            shuffle_generator.manual_seed(self.seed + 1)
            noise_generator.manual_seed(self.seed + 2)

        self.loss_fn_ = loss_fn
        previous_stats: tuple[Tensor, Tensor, Tensor] | None = None
        previous_smoothed: tuple[Tensor, Tensor, Tensor] | None = None
        previous_counts: Tensor | None = None
        self.history_ = []

        for epoch in range(self.epochs):
            model.train()
            # Refresh the statistics-of-statistics with the *current* encoder
            # at the start of every epoch (no gradient, one extra forward pass).
            # The paper updates them across epochs; recomputing on the current
            # model avoids training the NIG head against stale recolouring.
            if epoch > 0:
                model.eval()
                previous_stats, previous_counts = self._collect_statistics(model, X, y)
                previous_smoothed = self._smooth_bin_statistics(
                    previous_stats, kernel, previous_counts
                )
                model.train()
            order = torch.randperm(n_samples, generator=shuffle_generator).to(device)
            running_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                index = order[start : start + batch_size]
                X_batch = X[index]
                y_batch = y[index]
                optimizer.zero_grad(set_to_none=True)

                mean, variance = model.encode(X_batch)
                raw_mean, raw_variance = mean, variance
                bin_index = loss_fn.bin_index(y_batch)
                recoloured_variance = variance
                if (
                    previous_stats is not None
                    and previous_smoothed is not None
                    and previous_counts is not None
                ):
                    mean, recoloured_variance = self._recolour(
                        mean,
                        variance,
                        bin_index,
                        previous_stats,
                        previous_smoothed,
                        previous_counts,
                        self.smoothing_shrinkage,
                    )
                noise = torch.randn(
                    mean.shape, generator=noise_generator, device=device, dtype=dtype
                )
                latent = mean + recoloured_variance.clamp_min(_VARIANCE_FLOOR).sqrt() * noise
                loss = loss_fn(model.head(latent), y_batch)
                if self.kl_weight > 0.0:
                    # KL(q(z | x) || N(0, I)) on the *raw* probabilistic
                    # representation.  The full VIR ELBO also has an input
                    # reconstruction term that balances this KL; without it,
                    # use a small weight as a scale regulariser.
                    kl = (
                        0.5 * (raw_variance + raw_mean * raw_mean - 1.0 - raw_variance.log()).mean()
                    )
                    loss = loss + self.kl_weight * kl
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach())
                n_batches += 1

            epoch_loss = running_loss / max(n_batches, 1)
            self.history_.append({"loss": epoch_loss, "smoothed": epoch > 0})
            if self.verbose:
                print(f"epoch {epoch + 1:4d}/{self.epochs}  loss={epoch_loss:.6g}")

        # Final statistics used at inference, computed on the fitted encoder.
        final_stats, final_counts = self._collect_statistics(model, X, y)
        final_smoothed = self._smooth_bin_statistics(final_stats, kernel, final_counts)

        model.eval()
        self.model_ = model
        self.bin_counts_ = final_counts.detach()
        self.bin_mean_, self.bin_mean_var_, self.bin_cov_ = (
            tensor.detach() for tensor in final_stats
        )
        (
            self.smoothed_bin_mean_,
            self.smoothed_bin_mean_var_,
            self.smoothed_bin_cov_,
        ) = (tensor.detach() for tensor in final_smoothed)
        # With a single epoch the heads never saw a recoloured representation;
        # applying the transform at inference would be a train/test mismatch.
        self.smoothing_used_ = self.epochs > 1
        self._fitted = True
        return self

    # ── inference ──────────────────────────────────────────────────────
    def _check_fitted(self) -> None:
        if not self._fitted or self.model_ is None or self.loss_fn_ is None:
            raise RuntimeError("VIRRegressor must be fit before calling predict")
        assert self.bin_counts_ is not None
        assert self.bin_mean_ is not None
        assert self.bin_mean_var_ is not None
        assert self.bin_cov_ is not None
        assert self.smoothed_bin_mean_ is not None
        assert self.smoothed_bin_mean_var_ is not None
        assert self.smoothed_bin_cov_ is not None

    def _stored_statistics(
        self,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], tuple[Tensor, Tensor, Tensor]]:
        self._check_fitted()
        assert self.bin_mean_ is not None
        assert self.bin_mean_var_ is not None
        assert self.bin_cov_ is not None
        assert self.smoothed_bin_mean_ is not None
        assert self.smoothed_bin_mean_var_ is not None
        assert self.smoothed_bin_cov_ is not None
        stats = (self.bin_mean_, self.bin_mean_var_, self.bin_cov_)
        smoothed = (
            self.smoothed_bin_mean_,
            self.smoothed_bin_mean_var_,
            self.smoothed_bin_cov_,
        )
        return stats, smoothed

    def _posterior_for(self, X: Tensor) -> _NIGPosterior:
        """Validate ``X`` and return the plug-in NIG posterior used for inference."""
        self._check_fitted()
        assert self.model_ is not None
        assert self.loss_fn_ is not None
        check_tensor(X, "X")
        if X.dim() == 1:
            X = X.unsqueeze(0)
        if X.dim() != 2:
            raise ValueError(f"X must be 2-D (N, D), got shape {tuple(X.shape)}")
        if not X.is_floating_point():
            raise TypeError("X must be floating-point")
        device = self._resolve_device(X)
        X = X.to(device=device, dtype=self._dtype())
        model = self.model_.eval()

        with torch.no_grad():
            mean, variance = model.encode(X)
            raw = model.head(mean)
            stats, smoothed = self._stored_statistics()
            counts = self.bin_counts_
            assert counts is not None
            # The density weight is a function of the (unknown) test label.
            # Evaluate it at the model's own prediction, so inference needs no
            # labels and remains a legitimate plug-in procedure.
            posterior = self.loss_fn_.posterior_parameters(raw)
            weight = self.loss_fn_.importance_weights(posterior.gamma)
            if (
                self.inference_smoothing == "two_pass"
                and self.n_refine > 0
                and self.smoothing_used_
            ):
                bin_index = self._bin_index(posterior.gamma)
                for _ in range(self.n_refine):
                    recoloured_mean, _ = self._recolour(
                        mean,
                        variance,
                        bin_index,
                        stats,
                        smoothed,
                        counts,
                        self.smoothing_shrinkage,
                    )
                    raw = model.head(recoloured_mean)
                    posterior = self.loss_fn_.posterior_parameters(raw)
                    weight = self.loss_fn_.importance_weights(posterior.gamma)
                    bin_index = self._bin_index(posterior.gamma)
            return self.loss_fn_.posterior_parameters(raw, weight)

    def predict(self, X: Tensor, *, return_std: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Predict the target and, optionally, its predictive standard deviation.

        Args:
            X: Inputs of shape ``(N, D)`` (or ``(D,)`` for a single sample).
            return_std: When ``True`` return ``(mean, std)`` instead of the
                mean alone.

        Returns:
            ``(N, 1)`` predictions, or ``(mean, std)`` both of shape
            ``(N, 1)``.  ``std`` is the total Student-t predictive standard
            deviation (aleatoric + evidential epistemic), not a Bayesian
            posterior standard deviation.
        """
        self._check_fitted()
        assert self.loss_fn_ is not None
        posterior = self._posterior_for(X)
        prediction, std = self.loss_fn_.predictive_mean_std(posterior)
        prediction = prediction.reshape(-1, 1)
        if return_std:
            return prediction, std.reshape(-1, 1)
        return prediction

    def predict_interval(
        self,
        X: Tensor,
        *,
        confidence: float = 0.9,
    ) -> tuple[Tensor, Tensor]:
        """Central Student-t prediction interval from the NIG posterior.

        Args:
            X: Inputs of shape ``(N, D)`` (or ``(D,)``).
            confidence: Nominal coverage in ``(0, 1)``.

        Returns:
            ``(lower, upper)`` tensors of shape ``(N, 1)``.
        """
        if not 0.0 < float(confidence) < 1.0:
            raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
        self._check_fitted()
        assert self.loss_fn_ is not None
        posterior = self._posterior_for(X)
        scale = (
            (posterior.beta * (1.0 + posterior.nu) / (posterior.alpha * posterior.nu))
            .clamp_min(_VARIANCE_FLOOR)
            .sqrt()
        )
        from scipy.stats import t as student_t

        degrees_of_freedom = (2.0 * posterior.alpha).detach().cpu().numpy()
        quantile_np = student_t.ppf((1.0 + float(confidence)) / 2.0, degrees_of_freedom)
        quantile = torch.as_tensor(
            quantile_np, dtype=posterior.gamma.dtype, device=posterior.gamma.device
        )
        lower = (posterior.gamma - quantile * scale).reshape(-1, 1)
        upper = (posterior.gamma + quantile * scale).reshape(-1, 1)
        return lower, upper

    def _dtype(self) -> torch.dtype:
        assert self.model_ is not None
        for parameter in self.model_.parameters():
            return parameter.dtype
        return torch.float32
