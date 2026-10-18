r"""Errors-in-variables losses for repeated measurements and corrected scores.

Two objectives that correct the *attenuation bias* of ordinary least squares
when the regression inputs are measured with additive noise, using nothing but
the noise model itself (no latent-variable sampler and no inner optimisation
loop):

1. :class:`SplitMeasurementLoss` -- for **paired repeated measurements** of the
   same latent input.  It is the product of the residuals of two independent
   noisy views and needs no knowledge of the noise covariance at all.
2. :class:`CorrectedScoreEIVLoss` -- for a **known additive Gaussian input
   noise covariance**.  It subtracts the noise-induced variance of the model
   output (the Stefanski/Nakamura corrected score), unlike TLS/ODR style
   gradient penalties that *add* a penalty and flatten the objective.

Both losses inherit from :class:`torchregress.losses.eiv.BaseEIVLoss` and follow
its calling convention: ``y_pred`` carries the **observed noisy inputs** and
``target`` the observed labels, because the loss evaluates ``model`` itself.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor

from ..utils.validation import validate_weights
from .eiv import BaseEIVLoss

__all__ = [
    "CorrectedScoreEIVLoss",
    "SplitMeasurementLoss",
    "corrected_score_eiv_loss",
    "split_measurement_loss",
]

ModelFn = Callable[[Tensor], Tensor]
"""A model callable mapping a feature batch ``[N, D]`` to a prediction."""

_STABILIZERS = ("none", "clamp")


def _validate_feature_matrix(features: Tensor, name: str) -> None:
    """Require observed features to be a non-empty ``[N, D]`` matrix."""
    if features.dim() != 2:
        raise ValueError(f"{name} must have shape [N, D], got {tuple(features.shape)}")
    if features.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample, got {tuple(features.shape)}")


def _validate_target_and_mask(
    target: Tensor,
    mask: Optional[Tensor],
    batch_size: int,
    owner: str,
) -> None:
    """Validate a ``[N]`` or ``[N, M]`` target and a sample-level ``[N]`` mask."""
    if target.dim() == 0:
        raise ValueError(f"{owner}: target must have shape [N] or [N, M], got a scalar")
    if target.shape[0] != batch_size:
        raise ValueError(
            f"{owner}: batch size mismatch: y_pred has {batch_size} rows, "
            f"target has {target.shape[0]} rows."
        )
    if mask is None:
        return
    if mask.shape != target.shape and mask.shape != (batch_size,):
        raise ValueError(
            f"{owner}: mask must be sample-level [N] or match target shape "
            f"{tuple(target.shape)}, got {tuple(mask.shape)}."
        )


def _validate_forward_inputs(
    y_pred: Tensor,
    target: Tensor,
    mask: Optional[Tensor],
    weights: Optional[Tensor],
    owner: str,
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """Shared forward-argument validation for the two losses in this module."""
    _validate_feature_matrix(y_pred, f"{owner}: y_pred")
    batch_size = y_pred.shape[0]
    _validate_target_and_mask(target, mask, batch_size, owner)
    validated_weights = validate_weights(weights, batch_size)
    return mask, validated_weights


class SplitMeasurementLoss(BaseEIVLoss):
    r"""Errors-in-variables loss for paired (repeated) measurements.

    Let :math:`x_i` be the latent clean input, :math:`y_i = f(x_i)` the label
    and :math:`x^{(1)}_i = x_i + e^{(1)}_i`, :math:`x^{(2)}_i = x_i + e^{(2)}_i`
    two conditionally independent measurements with
    :math:`\mathbb{E}[e^{(1)}] = \mathbb{E}[e^{(2)}] = 0` and
    :math:`\mathbb{E}[e^{(1)} (e^{(2)})^\top] = 0`.

    The per-sample loss is the product of the residuals of the two views,

    .. math::

        \mathcal{L}_i = \frac{1}{M} \sum_{m=1}^{M}
        \bigl(y_{i,m} - f_m(x^{(1)}_i)\bigr)
        \bigl(y_{i,m} - f_m(x^{(2)}_i)\bigr),

    which is unbiased for the *noiseless* squared error of any model that is
    linear in the features:

    .. math::

        \mathbb{E}_{e^{(1)}, e^{(2)}}[\mathcal{L}_i]
        = \frac{1}{M}\bigl\| y_i - f(x_i) \bigr\|^2 .

    Because the cross term vanishes only in expectation, the noise contributes
    no systematic attenuation to the gradient -- no estimate of the input noise
    covariance is required, only a second independent measurement.  For
    non-linear :math:`f` the expectation retains the squared *smoothing bias*
    :math:`\frac{1}{M}\|\mathbb{E}[f(x+e)] - f(x)\|^2`, so the correction is
    exact only to first order; see the module documentation and
    ``docs/losses/split_eiv.md``.

    Independent target noise adds :math:`\sigma_y^2` to the expected loss, as
    it does for the ordinary squared error; the loss is therefore unbiased for
    the noisy-target risk, not for the noise-free risk.

    Args:
        model: Model function :math:`f` mapping ``[N, D]`` features to
            predictions.  It is evaluated internally on both measurement views.
        reduction: One of ``'none'``, ``'mean'``, ``'sum'``.
        eps: Small numerical-stability value forwarded to the base EIV class.
            It is unused by this loss and kept for API symmetry.

    Shape:
        - y_pred: :math:`(N, D)` first noisy measurement :math:`x^{(1)}`
        - target: :math:`(N)` or :math:`(N, M)` labels
        - y_pred_second: :math:`(N, D)` second noisy measurement :math:`x^{(2)}`
        - mask: :math:`(N)` (sample-level; ``True`` keeps a sample), or the
          target shape (a partially masked row is dropped)
        - weights: :math:`(N)` sample weights
        - Output: scalar if ``reduction`` is ``'mean'`` or ``'sum'``, else
          :math:`(N)`

    Example:
        >>> import torch
        >>> from torchregress.losses.split_eiv import SplitMeasurementLoss
        >>> model = torch.nn.Linear(1, 1)
        >>> x1 = torch.randn(8, 1)
        >>> x2 = x1 + 0.1 * torch.randn(8, 1)
        >>> y = model(x1).detach() + 0.05 * torch.randn(8, 1)
        >>> loss_fn = SplitMeasurementLoss(model)
        >>> loss = loss_fn(x1, y, y_pred_second=x2)
        >>> loss.ndim
        0
    """

    def __init__(
        self,
        model: ModelFn,
        reduction: str = "mean",
        eps: float = 1e-3,
    ) -> None:
        # The split loss does not need the input noise covariance: the two views
        # provide it implicitly through their independence.  sigma_x is fixed at
        # zero so BaseEIVLoss' plumbing stays valid.
        super().__init__(model=model, sigma_x=0.0, sigma_y=None, reduction=reduction, eps=eps)

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        y_pred_second: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the split-measurement loss.

        Args:
            y_pred: First noisy measurement view :math:`x^{(1)}` with shape
                ``[N, D]``.
            target: Labels with shape ``[N]`` or ``[N, M]``.
            y_pred_second: Second noisy measurement view :math:`x^{(2)}` with
                shape ``[N, D]``.  Required -- without it the loss is undefined.
            mask: Optional sample-level boolean mask ``[N]`` (``True`` keeps a
                sample).  A target-shaped mask is also accepted and collapsed
                to sample level.
            weights: Optional non-negative sample weights ``[N]``.
            **kwargs: Ignored; accepted for the shared loss-forward contract.

        Returns:
            Reduced loss tensor.

        Raises:
            ValueError: If ``y_pred_second`` is missing, or if the two views,
                target, mask, or weights have incompatible shapes.
        """
        del kwargs
        owner = type(self).__name__
        if y_pred_second is None:
            raise ValueError(
                f"{owner} requires the second measurement view: pass the paired "
                "noisy features as y_pred_second=..."
            )
        mask, weights = _validate_forward_inputs(y_pred, target, mask, weights, owner)
        _validate_feature_matrix(y_pred_second, f"{owner}: y_pred_second")
        if y_pred.shape != y_pred_second.shape:
            raise ValueError(
                f"{owner}: paired views must have the same shape, got "
                f"{tuple(y_pred.shape)} and {tuple(y_pred_second.shape)}."
            )

        pred_first = self.model(y_pred)
        pred_second = self.model(y_pred_second)
        if pred_first.shape != pred_second.shape:
            raise ValueError(
                f"{owner}: model output shape differs between the two views, got "
                f"{tuple(pred_first.shape)} and {tuple(pred_second.shape)}."
            )
        if pred_first.dim() == 0 or pred_first.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"{owner}: model must return one prediction per sample with a leading "
                f"batch dimension, got {tuple(pred_first.shape)}."
            )

        try:
            residual_first = target - pred_first
            residual_second = target - pred_second
        except RuntimeError as exc:  # pragma: no cover - defensive error translation
            raise ValueError(
                f"{owner}: target shape {tuple(target.shape)} is not broadcast-compatible "
                f"with model output shape {tuple(pred_first.shape)}."
            ) from exc

        # Mean over output dimensions; [N] for a single output, [N, M] otherwise.
        elementwise = residual_first * residual_second
        per_sample = elementwise.reshape(y_pred.shape[0], -1).mean(dim=1)
        return self._reduce(per_sample, mask, weights)


def split_measurement_loss(
    model: ModelFn,
    x_first: Tensor,
    target: Tensor,
    x_second: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Functional form of :class:`SplitMeasurementLoss`.

    Args:
        model: Model function ``f`` evaluated internally on both views.
        x_first: First noisy measurement :math:`x^{(1)}` with shape ``[N, D]``.
        target: Labels with shape ``[N]`` or ``[N, M]``.
        x_second: Second, independently noisy measurement :math:`x^{(2)}` with
            shape ``[N, D]``.
        reduction: ``'none'``, ``'mean'``, or ``'sum'``.

    Returns:
        Reduced loss tensor.
    """
    loss_fn = SplitMeasurementLoss(model=model, reduction=reduction)
    return loss_fn(x_first, target, y_pred_second=x_second)


class CorrectedScoreEIVLoss(BaseEIVLoss):
    r"""Unbiased corrected-score errors-in-variables loss.

    The observed inputs are :math:`w_i = x_i + \varepsilon_i` with
    :math:`\varepsilon_i \sim \mathcal{N}(0, \Sigma_x)` independent of
    :math:`(x_i, y_i)`.  The naive residual :math:`\|y_i - f(w_i)\|^2` is
    biased upward by the noise propagated through the model.  The corrected
    score of Stefanski (1989) and Nakamura (1990) subtracts exactly that
    expectation:

    .. math::

        \mathcal{L}_i = \|y_i - f(w_i)\|^2 - \operatorname{tr}\!\bigl(
        J_i \, \Sigma_x \, J_i^\top \bigr),
        \qquad J_i = \left.\frac{\partial f}{\partial w}\right|_{w_i},

    where for a vector output :math:`J_i \in \mathbb{R}^{M \times D}` and the
    trace equals :math:`\sum_{m=1}^{M} J_{i,m}^\top \Sigma_x J_{i,m}`.  With a
    diagonal covariance this is
    :math:`\sum_{m,d} \sigma_d^2 (\partial f_m / \partial w_d)^2`.  Because the
    correction is **subtracted**, it counters the variance inflation of the
    residual and the minimiser is not attenuated -- unlike TLS/ODR style
    gradient penalties, which add a penalty and flatten the objective.

    Important caveats:

    - The target noise (if any) contributes :math:`+\sigma_y^2` to the
      expected loss, exactly as for the ordinary squared error.  It is not
      corrected here.
    - The model is differentiated twice per step (a Jacobian-vector product
      backward pass through ``torch.func.jacrev``), so it must be
      ``torch.func``- and ``vmap``-safe: no data-dependent control flow, no
      batch-dependent BatchNorm statistics and no stochastic dropout in the
      forward pass.  Put the model in evaluation mode before training with
      this loss, and keep module parameters in the autograd graph (as usual).
    - The corrected score is an *unbiased estimating equation*, not a
      non-negative loss: individual samples (and even the batch mean) can be
      negative.  This is expected.  Use ``stabilize='clamp'`` only as a
      numerical crutch for unstable optimisation; it biases the objective.

    Args:
        model: Model function :math:`f` mapping **noisy observed** features
            ``[N, D]`` to predictions.  Evaluated internally.
        sigma_x: Standard deviation (scalar, ``(D,)``) or covariance matrix
            (``(D, D)``) of the additive input noise, with optional per-sample
            variants ``(N, D)`` / ``(N, D, D)``.  Interpreted by
            :meth:`BaseEIVLoss._prepare_covariance_from_sigma`: vectors are
            standard deviations (squared internally), matrices are already
            covariance matrices.
        reduction: One of ``'none'``, ``'mean'``, ``'sum'``.
        eps: Small numerical-stability value forwarded to the base EIV class.
        stabilize: ``'none'`` (statistically unbiased default) or ``'clamp'``
            (clamp each per-sample value at zero; biased, only for unstable
            optimisation).
        jacobian_chunk_size: Optional chunk size for the per-sample Jacobian
            computation.  ``None`` computes the whole batch at once; a positive
            integer reduces peak memory on large batches.

    Shape:
        - y_pred: :math:`(N, D)` observed noisy features :math:`w`
        - target: :math:`(N)` or :math:`(N, M)` labels
        - mask: :math:`(N)` (sample-level), or the target shape
        - weights: :math:`(N)` sample weights
        - Output: scalar if ``reduction`` is ``'mean'`` or ``'sum'``, else
          :math:`(N)`

    Example:
        >>> import torch
        >>> from torchregress.losses.split_eiv import CorrectedScoreEIVLoss
        >>> model = torch.nn.Linear(1, 1)
        >>> w = torch.randn(8, 1)  # noisy observed input
        >>> y = model(w).detach() + 0.05 * torch.randn(8, 1)
        >>> loss_fn = CorrectedScoreEIVLoss(model, sigma_x=0.2)
        >>> loss = loss_fn(w, y)
        >>> loss.ndim
        0
    """

    def __init__(
        self,
        model: ModelFn,
        sigma_x: Union[float, Tensor],
        reduction: str = "mean",
        eps: float = 1e-3,
        stabilize: str = "none",
        jacobian_chunk_size: Optional[int] = None,
    ) -> None:
        if sigma_x is None:  # type: ignore[comparison-overlap]
            raise ValueError("sigma_x is required: pass the input-noise std/covariance.")
        if stabilize not in _STABILIZERS:
            raise ValueError(f"stabilize must be one of {_STABILIZERS}, got {stabilize!r}")
        if jacobian_chunk_size is not None:
            if isinstance(jacobian_chunk_size, bool) or not isinstance(jacobian_chunk_size, int):
                raise ValueError(
                    "jacobian_chunk_size must be a positive integer or None, got "
                    f"{jacobian_chunk_size!r}"
                )
            if jacobian_chunk_size < 1:
                raise ValueError(f"jacobian_chunk_size must be positive, got {jacobian_chunk_size}")
        super().__init__(model=model, sigma_x=sigma_x, reduction=reduction, eps=eps)
        self.stabilize = stabilize
        self.jacobian_chunk_size = jacobian_chunk_size

    def _single_sample_output(self, w: Tensor) -> Tensor:
        """Evaluate the model on one sample (vmap-compatible, no batch dim)."""
        return self.model(w.unsqueeze(0)).squeeze(0)

    def _jacobian_vmap(self, features: Tensor) -> Tensor:
        """Per-sample Jacobian ``[n, M, D]`` of the model at ``features``."""
        jacobian = torch.func.vmap(torch.func.jacrev(self._single_sample_output))(features)
        if jacobian.dim() == 2:
            # Model returns a scalar per sample: vmap stacks to [n, D].
            jacobian = jacobian.unsqueeze(1)
        if jacobian.dim() != 3:
            raise ValueError(
                "CorrectedScoreEIVLoss: the model must return predictions with shape "
                f"[N, M] or [N], got per-sample Jacobian shaped {tuple(jacobian.shape)}."
            )
        return jacobian

    def _jacobian(self, features: Tensor) -> Tensor:
        """Per-sample Jacobian, optionally computed in chunks to bound memory."""
        chunk_size = self.jacobian_chunk_size
        n_samples = features.shape[0]
        if chunk_size is None or chunk_size >= n_samples:
            return self._jacobian_vmap(features)
        chunks = [
            self._jacobian_vmap(features[start : start + chunk_size])
            for start in range(0, n_samples, chunk_size)
        ]
        return torch.cat(chunks, dim=0)

    def _trace_correction(self, jacobian: Tensor, sigma_cov: Tensor) -> Tensor:
        """Per-sample ``tr(J Sigma J^T)``.

        ``jacobian`` is ``[N, M, D]`` and ``sigma_cov`` is either ``[D, D]``
        (shared covariance) or ``[N, D, D]`` (per-sample covariance).
        """
        if sigma_cov.dim() == 2:
            return torch.einsum("nmd,de,nme->n", jacobian, sigma_cov, jacobian)
        if sigma_cov.dim() == 3:
            return torch.einsum("nmd,nde,nme->n", jacobian, sigma_cov, jacobian)
        raise ValueError(
            "CorrectedScoreEIVLoss: prepared sigma covariance must be [D, D] or "
            f"[N, D, D], got {tuple(sigma_cov.shape)}."
        )

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the corrected-score EIV loss.

        Args:
            y_pred: Observed noisy features :math:`w` with shape ``[N, D]``.
            target: Labels with shape ``[N]`` or ``[N, M]``.
            mask: Optional sample-level boolean mask ``[N]`` (``True`` keeps a
                sample).  A target-shaped mask is also accepted and collapsed
                to sample level.
            weights: Optional non-negative sample weights ``[N]``.
            **kwargs: May carry ``sigma_x=`` to override the covariance for this
                call (same accepted forms as the constructor argument); other
                keys are ignored.

        Returns:
            Reduced loss tensor.

        Raises:
            ValueError: If input shapes are incompatible or ``sigma_x`` is not
                preparable for the given number of features.
        """
        owner = type(self).__name__
        mask, weights = _validate_forward_inputs(y_pred, target, mask, weights, owner)
        sigma_x = kwargs.pop("sigma_x", self.sigma_x)
        del kwargs

        batch_size, n_features = y_pred.shape
        predictions = self.model(y_pred)
        if predictions.dim() == 0:
            raise ValueError(f"{owner}: model must return a batch of predictions, got a scalar.")
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)
        if predictions.shape[0] != batch_size:
            raise ValueError(
                f"{owner}: model must return one prediction per sample, got "
                f"{tuple(predictions.shape)} for a batch of {batch_size}."
            )

        target_matrix = target if target.dim() == 2 else target.unsqueeze(1)
        try:
            squared_terms = (target_matrix - predictions) ** 2
        except RuntimeError as exc:  # pragma: no cover - defensive error translation
            raise ValueError(
                f"{owner}: target shape {tuple(target.shape)} is not broadcast-compatible "
                f"with model output shape {tuple(predictions.shape)}."
            ) from exc
        squared_terms = squared_terms.reshape(batch_size, -1)
        # Sum over output dimensions: the correction below subtracts the full
        # trace, which sums over output dimensions as well, so the objective is
        # unbiased for sum-of-squares targets.
        squared_residual = squared_terms.sum(dim=1)

        jacobian = self._jacobian(y_pred)  # [N, M, D]
        if jacobian.shape[0] != batch_size or jacobian.shape[2] != n_features:
            raise ValueError(
                f"{owner}: Jacobian shape {tuple(jacobian.shape)} is inconsistent with "
                f"y_pred shape {tuple(y_pred.shape)}."
            )
        if jacobian.shape[1] != squared_terms.shape[1]:
            raise ValueError(
                f"{owner}: the model output has {squared_terms.shape[1]} elements per "
                f"sample but the per-sample Jacobian has {jacobian.shape[1]} rows; "
                "note that model output shapes are checked after broadcasting."
            )

        sigma_cov = self._prepare_covariance_from_sigma(
            sigma_x,
            n_features,
            y_pred.device,
            batch_size=batch_size,
            dtype=y_pred.dtype,
        )
        sigma_cov = sigma_cov.to(device=jacobian.device, dtype=jacobian.dtype)
        correction = self._trace_correction(jacobian, sigma_cov)

        per_sample = squared_residual - correction
        if self.stabilize == "clamp":
            # Biased, optimisation-only safeguard: the unbiased corrected score
            # is allowed to be negative, clamping changes its stationary points.
            per_sample = per_sample.clamp_min(0.0)
        return self._reduce(per_sample, mask, weights)


def corrected_score_eiv_loss(
    model: ModelFn,
    y_pred: Tensor,
    target: Tensor,
    sigma_x: Union[float, Tensor],
    reduction: str = "mean",
    eps: float = 1e-3,
    stabilize: str = "none",
    jacobian_chunk_size: Optional[int] = None,
) -> Tensor:
    """Functional form of :class:`CorrectedScoreEIVLoss`.

    Args:
        model: Model function ``f`` evaluated internally on the noisy features.
        y_pred: Observed noisy features :math:`w` with shape ``[N, D]``.
        target: Labels with shape ``[N]`` or ``[N, M]``.
        sigma_x: Input-noise standard deviation or covariance (scalar, ``(D,)``,
            ``(D, D)``, ``(N, D)``, or ``(N, D, D)``).
        reduction: ``'none'``, ``'mean'``, or ``'sum'``.
        eps: Numerical-stability value forwarded to the loss.
        stabilize: ``'none'`` or ``'clamp'``.
        jacobian_chunk_size: Optional chunk size for the Jacobian computation.

    Returns:
        Reduced loss tensor.
    """
    loss_fn = CorrectedScoreEIVLoss(
        model=model,
        sigma_x=sigma_x,
        reduction=reduction,
        eps=eps,
        stabilize=stabilize,
        jacobian_chunk_size=jacobian_chunk_size,
    )
    return loss_fn(y_pred, target)
