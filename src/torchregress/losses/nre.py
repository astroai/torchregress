"""
Neural Ratio Estimation (NRE) losses for likelihood-ratio based regression.

NRE (Cranmer, Pavez & Louppe 2015; Durkan, Murray & Papamakarios 2020;
Miller, Weniger & Forre 2022) estimates the likelihood-to-evidence ratio

.. math::

    r(x, \\theta) = \\frac{p(x \\mid \\theta)}{p(x)}
    = \\frac{p(x, \\theta)}{p(x)\\,p(\\theta)}

by training a classifier to separate **joint** pairs ``(theta, x) ~ p(theta, x)``
from **marginal** pairs ``(theta, x) ~ p(theta) p(x)``.  A binary classifier
trained with balanced labels has the Bayes-optimal solution

.. math::

    \\sigma\\bigl(s(\\theta, x)\\bigr) = \\frac{p(x, \\theta)}{p(x, \\theta) + p(\\theta)p(x)},
    \\qquad s(\\theta, x) = \\log r(x, \\theta),

so the classifier logit is a direct estimator of the log likelihood ratio.

Because :math:`\\log r(x, \\theta) = \\log p(x \\mid \\theta) - \\log p(x)` and
:math:`p(x)` does not depend on :math:`\\theta`, ranking candidate parameters by
:math:`s(\\theta, x)` recovers the maximum-likelihood estimate
:math:`\\arg\\max_\\theta p(x \\mid \\theta)` *without* knowing the training prior
``p(theta)``: the prior cancels between the joint and marginal samples.  This is
the property that distinguishes NRE from a conditional-density surrogate that
has to model the target distribution directly.

Two classifier objectives are supported:

- ``"bce"``: balanced binary NRE / likelihood-ratio trick,
  :math:`-\\log\\sigma(s_+) - \\frac{1}{K}\\sum_k \\log\\sigma(-s^-_k)`.
- ``"multiclass"``: categorical (NRE-B / InfoNCE-style) NRE,
  :math:`-\\log\\mathrm{softmax}([s_+, s^-_1, \\ldots, s^-_K])_0`.

The balanced binary objective (NRE-A in the literature) targets
:math:`\\log r(x, \\theta)` exactly at its optimum when the negatives follow
the same reference prior.  The multiclass objective (NRE-B) is a contrastive
variant: it has an intrinsic bias at finite ``K`` and only approaches the same
ratio as the number of negatives grows; Miller, Weniger & Forre (2022) propose
a bias-free multiclass variant (CNRE) that is not implemented here.  Prefer
``"bce"`` when a calibrated absolute log-ratio is required.

The classifier is trained jointly with the rest of the model by minimising this
loss; the returned value is reduced per sample exactly like other torchregress
losses (mask, sample weights, ``'none' | 'mean' | 'sum'``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils.validation import validate_weights
from .base import BaseLoss

__all__ = [
    "NeuralRatioLoss",
    "build_ratio_classifier",
    "neural_ratio_logits",
    "neural_ratio_loss",
]

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}

_CRITERIA = ("bce", "multiclass")


def build_ratio_classifier(
    context_dim: int,
    observable_dim: int,
    hidden_dims: Sequence[int] = (64, 64),
    activation: str = "silu",
) -> nn.Module:
    """Build an MLP classifier over ``concat([context, observable])`` logits.

    The returned module maps a pair ``(context, observable)`` to a single
    scalar logit, i.e. it is the classifier used by :class:`NeuralRatioLoss` to
    estimate ``s(theta, x) = log p(x|theta) - log p(x)``.

    Args:
        context_dim: Number of context / parameter dimensions ``C``.
        observable_dim: Number of observable / target dimensions ``D``.
        hidden_dims: Hidden layer widths.  ``()`` yields a single linear layer
            (useful for exact analytic checks); the default ``(64, 64)`` is two
            hidden layers.
        activation: Activation name, one of ``"silu"`` (default), ``"relu"``,
            ``"gelu"``, ``"tanh"``, ``"elu"``, ``"leaky_relu"``.

    Returns:
        ``nn.Sequential`` with input width ``context_dim + observable_dim`` and
        one output logit per pair.  The module is tagged with ``context_dim``,
        ``observable_dim`` and ``input_dim`` attributes so
        :class:`NeuralRatioLoss` can infer its shapes without a first forward
        pass.

    Example:
        >>> classifier = build_ratio_classifier(2, 1, hidden_dims=(32, 32))
        >>> classifier(torch.zeros(5, 3)).shape
        torch.Size([5, 1])
    """
    context_dim = int(context_dim)
    observable_dim = int(observable_dim)
    if context_dim < 1:
        raise ValueError(f"context_dim must be a positive integer, got {context_dim}")
    if observable_dim < 1:
        raise ValueError(f"observable_dim must be a positive integer, got {observable_dim}")
    if isinstance(hidden_dims, int):
        raise TypeError("hidden_dims must be a sequence of layer widths, not an int")
    widths = [int(width) for width in hidden_dims]
    if any(width < 1 for width in widths):
        raise ValueError(f"hidden_dims must be positive integers, got {list(widths)}")

    activation_key = str(activation).lower()
    if activation_key not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}")
    activation_cls = _ACTIVATIONS[activation_key]

    layers: list[nn.Module] = []
    in_features = context_dim + observable_dim
    for width in widths:
        layers.append(nn.Linear(in_features, width))
        layers.append(activation_cls())
        in_features = width
    layers.append(nn.Linear(in_features, 1))

    classifier = nn.Sequential(*layers)
    # Plain attributes (not parameters/buffers): shape metadata survives .to(),
    # state_dict round-trips and deep copies, and lets NeuralRatioLoss validate
    # the concatenated input width lazily.
    setattr(classifier, "context_dim", context_dim)
    setattr(classifier, "observable_dim", observable_dim)
    setattr(classifier, "input_dim", context_dim + observable_dim)
    return classifier


def _as_2d(tensor: Tensor, name: str) -> Tensor:
    """Reshape a vector to a ``[N, 1]`` feature matrix; require 1-D or 2-D input."""
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    if tensor.dim() == 1:
        return tensor.unsqueeze(-1)
    if tensor.dim() == 2:
        return tensor
    raise ValueError(f"{name} must be 1-D or 2-D, got shape {tuple(tensor.shape)}")


def _validate_floating(tensor: Tensor, name: str) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor, got dtype {tensor.dtype}")


def neural_ratio_logits(classifier: nn.Module, context: Tensor, observable: Tensor) -> Tensor:
    """Evaluate a ratio classifier on matched ``(context, observable)`` pairs.

    This is the low-level scoring primitive: it concatenates each context with
    its observable and returns the classifier logit, which is the estimated log
    likelihood-to-evidence ratio ``s(theta, x)`` when the classifier was trained
    with :class:`NeuralRatioLoss`.

    Args:
        classifier: Module mapping ``concat([context, observable])`` to one logit.
            :func:`build_ratio_classifier` produces a suitable module.
        context: ``[N, C]`` candidate parameters/hypotheses (1-D ``[N]`` is
            treated as ``[N, 1]``).
        observable: ``[N, D]`` observations (1-D ``[N]`` is treated as
            ``[N, 1]``).

    Returns:
        ``[N]`` classifier logits.

    Raises:
        TypeError: If ``classifier`` is not an ``nn.Module`` or an input is not
            a floating-point tensor.
        ValueError: If batch sizes differ, the concatenated width does not match
            a tagged classifier, or the classifier returns more than one logit
            per pair.
    """
    if not isinstance(classifier, nn.Module):
        raise TypeError(f"classifier must be a torch.nn.Module, got {type(classifier)}")
    context_2d = _as_2d(context, "context")
    observable_2d = _as_2d(observable, "observable")
    _validate_floating(context_2d, "context")
    _validate_floating(observable_2d, "observable")
    if context_2d.shape[0] != observable_2d.shape[0]:
        raise ValueError(
            "context and observable must have the same batch dimension, got "
            f"{context_2d.shape[0]} and {observable_2d.shape[0]}"
        )
    if context_2d.device != observable_2d.device:
        raise ValueError(
            f"context and observable must be on the same device, got "
            f"{context_2d.device} and {observable_2d.device}"
        )
    if context_2d.dtype != observable_2d.dtype:
        raise ValueError(
            f"context and observable must have the same dtype, got "
            f"{context_2d.dtype} and {observable_2d.dtype}"
        )
    expected_width = getattr(classifier, "input_dim", None)
    if expected_width is not None and context_2d.shape[-1] + observable_2d.shape[-1] != int(
        expected_width
    ):
        raise ValueError(
            "context_dim + observable_dim does not match the classifier input width: "
            f"{context_2d.shape[-1]} + {observable_2d.shape[-1]} != {int(expected_width)}"
        )

    logits = classifier(torch.cat([context_2d, observable_2d], dim=-1))
    if not isinstance(logits, Tensor):
        raise TypeError(f"classifier must return a torch.Tensor, got {type(logits)}")
    if logits.dim() == 2 and logits.shape[-1] == 1:
        logits = logits.squeeze(-1)
    if logits.dim() != 1:
        raise ValueError(
            "classifier must return one logit per pair with shape [N] or [N, 1], "
            f"got shape {tuple(logits.shape)}"
        )
    if logits.shape[0] != context_2d.shape[0]:
        raise ValueError(
            "classifier output batch dimension must match the input batch dimension, "
            f"got {logits.shape[0]} and {context_2d.shape[0]}"
        )
    return logits


class NeuralRatioLoss(BaseLoss):
    """Classifier-based Neural Ratio Estimation loss.

    Trains a ratio classifier ``s(theta, x)`` to distinguish true
    ``(theta, x)`` pairs from mismatched ones.  For the balanced binary
    objective the optimal logit is
    ``log r(x, theta) = log p(x|theta) - log p(x)``, so ranking candidate
    parameters by the logit recovers the maximum-likelihood estimate while the
    training prior cancels between the joint and marginal classes.  The
    multiclass objective is a contrastive (InfoNCE-style, NRE-B) variant that
    approaches the same ratio as the number of negatives grows.

    Args:
        classifier: Module mapping ``concat([context, observable])`` to one logit,
            e.g. from :func:`build_ratio_classifier`.  The classifier parameters
            are owned (and registered) by this loss, so create the optimiser
            after constructing the loss or optimise ``loss_fn.classifier``.
        criterion: ``"bce"`` for balanced binary NRE (NRE-A;
            :math:`-\\log\\sigma(s_+) - \\frac{1}{K}\\sum_k\\log\\sigma(-s^-_k)`)
            or ``"multiclass"`` for categorical NRE (NRE-B;
            :math:`-\\log\\mathrm{softmax}([s_+, s^-_1,\\ldots,s^-_K])_0`).
        reduction: ``'none' | 'mean' | 'sum'`` over samples.

    Shapes:
        ``y_pred``: positive context ``[N, C]`` (1-D ``[N]`` is ``[N, 1]``).
        ``target``: observations ``[N, D]`` (1-D ``[N]`` is ``[N, 1]``).
        ``negative_context``: negative candidate parameters, ``[N, K, C]``,
            ``[1, K, C]`` (shared bank) or ``[K, C]`` (shared bank, ambiguous
            when ``K == N`` — disambiguate with ``shared_negative_context``).
        ``negative_target``: optional negative observations ``[N, K, D]``,
            ``[1, K, D]`` or ``[K, D]``.  When omitted the negatives reuse the
            positive observations, estimating likelihood ratios *between
            hypotheses* for the same observation; when provided each negative
            is a full mismatched ``(theta, x)`` pair.
        ``mask`` / ``weights``: sample-level ``[N]`` (or ``[N, 1]``).

    Reference:
        Cranmer, Pavez & Louppe, "Approximating Likelihood Ratios with
        Calibrated Probabilistic Classifiers", arXiv:1506.02169 (2015).
        Durkan, Murray & Papamakarios, "On Contrastive Learning for Likelihood-free
        Inference", arXiv:2002.03712 (2020).
        Miller, Weniger & Forre, "Contrastive Neural Ratio Estimation for
        Simulation-based Inference", arXiv:2210.06170 (2022).
        Miller, Cole, Forre, Louppe & Weniger, "Truncated Marginal Neural Ratio
        Estimation", arXiv:2107.01214 (2021).
    """

    def __init__(
        self,
        classifier: nn.Module,
        *,
        criterion: str = "bce",
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        if not isinstance(classifier, nn.Module):
            raise TypeError(f"classifier must be a torch.nn.Module, got {type(classifier)}")
        criterion = str(criterion).lower()
        if criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {list(_CRITERIA)}, got {criterion!r}")
        self.classifier = classifier
        self.criterion = criterion

        inferred_context = getattr(classifier, "context_dim", None)
        inferred_observable = getattr(classifier, "observable_dim", None)
        self._context_dim: Optional[int] = (
            None if inferred_context is None else int(inferred_context)
        )
        self._observable_dim: Optional[int] = (
            None if inferred_observable is None else int(inferred_observable)
        )

    @property
    def context_dim(self) -> Optional[int]:
        """Positive-context dimension ``C``, inferred from the classifier or first batch."""
        return self._context_dim

    @property
    def observable_dim(self) -> Optional[int]:
        """Observable dimension ``D``, inferred from the classifier or first batch."""
        return self._observable_dim

    def _infer_dims(self, context: Tensor, observable: Tensor) -> tuple[int, int]:
        """Infer/validate ``(context_dim, observable_dim)`` from a batch."""
        context_dim = int(context.shape[-1])
        observable_dim = int(observable.shape[-1])
        if self._context_dim is None:
            if context.numel() == 0:
                raise ValueError(
                    "cannot infer context_dim from an empty batch; pass a non-empty batch "
                    "or build the classifier with build_ratio_classifier"
                )
            self._context_dim = context_dim
        elif context_dim != self._context_dim:
            raise ValueError(
                f"y_pred context dimension changed: expected {self._context_dim}, got {context_dim}"
            )
        if self._observable_dim is None:
            if observable.numel() == 0:
                raise ValueError(
                    "cannot infer observable_dim from an empty batch; pass a non-empty batch "
                    "or build the classifier with build_ratio_classifier"
                )
            self._observable_dim = observable_dim
        elif observable_dim != self._observable_dim:
            raise ValueError(
                "target observable dimension changed: expected "
                f"{self._observable_dim}, got {observable_dim}"
            )
        return self._context_dim, self._observable_dim

    @staticmethod
    def _prepare_negative_bank(
        tensor: Tensor,
        batch_size: int,
        feature_dim: int,
        *,
        name: str,
        shared: Optional[bool],
        shared_arg: str,
    ) -> Tensor:
        """Normalise a negative bank to ``[batch, K, feature_dim]``.

        Mirrors :class:`~torchregress.losses.nflows.ContrastiveFlowLoss`
        conventions: a 2-D tensor is a shared bank unless it is explicitly
        marked per-sample, and is rejected as ambiguous when its leading
        dimension equals the batch size.
        """
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None, got {type(tensor)}")
        if shared is not None and not isinstance(shared, bool):
            raise TypeError(f"{shared_arg} must be a bool or None, got {type(shared)}")
        if tensor.dim() == 2:
            if tensor.shape[0] == 0:
                raise ValueError(f"{name} must include at least one negative candidate")
            if int(tensor.shape[-1]) != feature_dim:
                raise ValueError(
                    f"{name} last dimension must match the positive feature dimension "
                    f"({feature_dim}), got {int(tensor.shape[-1])}"
                )
            n_candidates = int(tensor.shape[0])
            if shared is None and n_candidates == batch_size:
                raise ValueError(
                    f"{name} with shape [N, F] is ambiguous when N matches the batch size. "
                    f"Pass [batch, 1, F] for per-sample candidates, [1, K, F] for a shared "
                    f"bank, or set {shared_arg} explicitly."
                )
            if shared is False:
                if n_candidates != batch_size:
                    raise ValueError(
                        f"per-sample {name} must have batch dimension matching y_pred, "
                        f"got {n_candidates} and {batch_size}"
                    )
                return tensor.unsqueeze(1)
            return tensor.unsqueeze(0).expand(batch_size, -1, -1)
        if tensor.dim() == 3:
            if int(tensor.shape[1]) == 0:
                raise ValueError(f"{name} must include at least one negative candidate")
            if int(tensor.shape[-1]) != feature_dim:
                raise ValueError(
                    f"{name} last dimension must match the positive feature dimension "
                    f"({feature_dim}), got {int(tensor.shape[-1])}"
                )
            if int(tensor.shape[0]) == batch_size:
                return tensor
            if int(tensor.shape[0]) == 1:
                return tensor.expand(batch_size, -1, -1)
            raise ValueError(
                f"3D {name} must have shape [batch, n_negatives, F] or [1, n_negatives, F], "
                f"got {tuple(tensor.shape)}"
            )
        raise ValueError(
            f"{name} must have shape [batch, n_negatives, F], [1, n_negatives, F] "
            f"or [n_negatives, F], got shape {tuple(tensor.shape)}"
        )

    @staticmethod
    def _zero_masked_rows(tensor: Tensor, sample_mask: Tensor) -> Tensor:
        """Replace fully masked samples by zeros so ignored rows never propagate NaNs."""
        rank = tensor.dim() - 1
        view_shape = (sample_mask.shape[0],) + (1,) * rank
        return torch.where(sample_mask.reshape(view_shape), tensor, torch.zeros_like(tensor))

    def _logits(
        self,
        y_pred: Tensor,
        target: Tensor,
        negative_context: Tensor,
        negative_target: Optional[Tensor] = None,
        *,
        shared_negative_context: Optional[bool] = None,
        shared_negative_target: Optional[bool] = None,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Return positive and negative classifier logits ``([N], [N, K])``."""
        context = _as_2d(y_pred, "y_pred")
        observable = _as_2d(target, "target")
        _validate_floating(context, "y_pred")
        _validate_floating(observable, "target")
        if context.shape[0] != observable.shape[0]:
            raise ValueError(
                "y_pred and target must have the same batch dimension, got "
                f"{context.shape[0]} and {observable.shape[0]}"
            )
        batch_size = int(context.shape[0])
        context_dim, observable_dim = self._infer_dims(context, observable)

        negative_context_3d = self._prepare_negative_bank(
            negative_context,
            batch_size,
            context_dim,
            name="negative_context",
            shared=shared_negative_context,
            shared_arg="shared_negative_context",
        )
        n_negatives = int(negative_context_3d.shape[1])
        if negative_target is None:
            negative_observable_3d = observable.unsqueeze(1).expand(
                batch_size, n_negatives, observable_dim
            )
        else:
            negative_observable_3d = self._prepare_negative_bank(
                negative_target,
                batch_size,
                observable_dim,
                name="negative_target",
                shared=shared_negative_target,
                shared_arg="shared_negative_target",
            )
            if int(negative_observable_3d.shape[1]) != n_negatives:
                raise ValueError(
                    "negative_target must have the same number of negatives as "
                    f"negative_context: got {int(negative_observable_3d.shape[1])} and "
                    f"{n_negatives}"
                )

        if mask is not None:
            context = self._zero_masked_rows(context, mask)
            observable = self._zero_masked_rows(observable, mask)
            negative_context_3d = self._zero_masked_rows(negative_context_3d, mask)
            negative_observable_3d = self._zero_masked_rows(negative_observable_3d, mask)

        positive_logits = neural_ratio_logits(self.classifier, context, observable)
        negative_logits = neural_ratio_logits(
            self.classifier,
            negative_context_3d.reshape(batch_size * n_negatives, context_dim),
            negative_observable_3d.reshape(batch_size * n_negatives, observable_dim),
        ).reshape(batch_size, n_negatives)
        return positive_logits, negative_logits

    def _criterion_loss(self, positive_logits: Tensor, negative_logits: Tensor) -> Tensor:
        """Per-sample NRE classification loss (unreduced)."""
        if self.criterion == "bce":
            # Balanced NRE: the positive term and the average over negatives
            # carry equal total weight, matching a balanced binary classifier.
            return -F.logsigmoid(positive_logits) - F.logsigmoid(-negative_logits).mean(dim=1)
        logits = torch.cat([positive_logits.unsqueeze(1), negative_logits], dim=1)
        return -F.log_softmax(logits, dim=1)[:, 0]

    def log_likelihood_ratio(
        self,
        y_pred: Tensor,
        target: Tensor,
        negative_context: Tensor,
        negative_target: Optional[Tensor] = None,
        *,
        shared_negative_context: Optional[bool] = None,
        shared_negative_target: Optional[bool] = None,
    ) -> Tensor:
        """Estimated log ratio ``s(theta+, x+) - s(theta-_k, x-_k)`` ``[N, K]``.

        With ``negative_target=None`` the same observation is scored under each
        negative hypothesis, so the result estimates the pairwise likelihood
        ratio ``log p(x | theta+) / p(x | theta-_k)``.  With ``negative_target``
        given, each negative is a full mismatched pair and the result compares
        joint scores.

        Args:
            y_pred: Positive context ``[N, C]``.
            target: Observations ``[N, D]``.
            negative_context: Negative candidate parameters, see class shapes.
            negative_target: Optional negative observations.
            shared_negative_context: Force a 2-D ``negative_context`` to be read
                as a shared bank (``True``) or as per-sample candidates
                (``False``).
            shared_negative_target: Same for ``negative_target``.

        Returns:
            ``[N, K]`` tensor of estimated log likelihood ratios.
        """
        positive_logits, negative_logits = self._logits(
            y_pred,
            target,
            negative_context,
            negative_target,
            shared_negative_context=shared_negative_context,
            shared_negative_target=shared_negative_target,
        )
        return positive_logits.unsqueeze(1) - negative_logits

    @staticmethod
    def _prepare_sample_mask(mask: Tensor, batch_size: int) -> Tensor:
        if not isinstance(mask, Tensor):
            raise TypeError(f"mask must be a torch.Tensor or None, got {type(mask)}")
        if mask.numel() != batch_size:
            raise ValueError(
                "NeuralRatioLoss only supports sample-level masks: mask must have "
                f"{batch_size} elements, got shape {tuple(mask.shape)}"
            )
        return mask.reshape(batch_size).to(torch.bool)

    @staticmethod
    def _prepare_sample_weights(weights: Optional[Tensor], batch_size: int) -> Optional[Tensor]:
        validated = validate_weights(weights, batch_size)
        if validated is None:
            return None
        if validated.numel() != batch_size:
            raise ValueError(
                "NeuralRatioLoss only supports sample-level weights: weights must have "
                f"{batch_size} elements, got shape {tuple(validated.shape)}"
            )
        return validated.reshape(batch_size)

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        negative_context: Optional[Tensor] = None,
        negative_target: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the NRE classification loss.

        Args:
            y_pred: Positive context, i.e. candidate parameters/hypotheses
                ``[N, C]``.
            target: Observations ``[N, D]``.
            negative_context: Negative candidate parameters.  Required; see the
                class docstring for the accepted shapes.
            negative_target: Optional negative observations.  ``None`` (default)
                reuses ``target`` and estimates likelihood ratios between
                hypotheses for the same observation.
            mask: Optional sample-level boolean mask, ``False`` = invalid sample.
                Masked samples are zero-filled and excluded from the reduction.
            weights: Optional non-negative sample weights ``[N]``.

        Keyword Args:
            shared_negative_context: ``True`` forces a 2-D ``negative_context``
                to be a shared bank, ``False`` forces per-sample candidates;
                required to disambiguate a 2-D bank whose leading dimension
                equals the batch size.
            shared_negative_target: Same for ``negative_target``.
            negative_y_pred: Alias for ``negative_context``.

        Returns:
            Loss reduced according to ``self.reduction``.

        Raises:
            ValueError: If ``negative_context`` is missing or the negative bank
                shapes are invalid.
        """
        shared_negative_context = kwargs.pop("shared_negative_context", None)
        shared_negative_target = kwargs.pop("shared_negative_target", None)
        if negative_context is None:
            negative_context = kwargs.pop("negative_y_pred", None)
        if negative_context is None:
            raise ValueError("NeuralRatioLoss requires negative_context in forward()")

        # Validate the batch dimension before touching mask/weights so malformed
        # inputs raise the same shape errors as _logits does below.
        batch_size = int(_as_2d(y_pred, "y_pred").shape[0])
        sample_mask = None if mask is None else self._prepare_sample_mask(mask, batch_size)
        sample_weights = self._prepare_sample_weights(weights, batch_size)

        positive_logits, negative_logits = self._logits(
            y_pred,
            target,
            negative_context,
            negative_target,
            shared_negative_context=shared_negative_context,
            shared_negative_target=shared_negative_target,
            mask=sample_mask,
        )
        per_sample = self._criterion_loss(positive_logits, negative_logits)
        return self._reduce(per_sample, sample_mask, sample_weights)


def neural_ratio_loss(
    classifier: nn.Module,
    context: Tensor,
    observable: Tensor,
    negative_context: Tensor,
    negative_target: Optional[Tensor] = None,
    criterion: str = "bce",
    reduction: str = "mean",
) -> Tensor:
    """Functional NRE loss: train ``classifier`` as a likelihood-ratio estimator.

    Convenience wrapper around :class:`NeuralRatioLoss`; the classifier is used
    as given (its parameters are returned through ``classifier.parameters()``).

    Args:
        classifier: Module mapping ``concat([context, observable])`` to one logit.
        context: Positive context ``[N, C]``.
        observable: Observations ``[N, D]``.
        negative_context: Negative candidate parameters ``[N, K, C]``,
            ``[1, K, C]`` or ``[K, C]`` (shared bank).
        negative_target: Optional mismatched observations ``[N, K, D]``,
            ``[1, K, D]`` or ``[K, D]``.
        criterion: ``"bce"`` (default) or ``"multiclass"``.
        reduction: ``'none' | 'mean' | 'sum'``.

    Returns:
        Loss tensor.

    Example:
        >>> classifier = build_ratio_classifier(1, 1, hidden_dims=(16, 16))
        >>> context = torch.randn(8, 1)
        >>> observable = context + 0.1 * torch.randn(8, 1)
        >>> negative = torch.randn(8, 4, 1)  # alternative hypotheses from the prior
        >>> loss = neural_ratio_loss(classifier, context, observable, negative)
        >>> loss.backward()
    """
    loss_fn = NeuralRatioLoss(
        classifier,
        criterion=criterion,
        reduction=reduction,
    )
    return loss_fn(
        context,
        observable,
        negative_context=negative_context,
        negative_target=negative_target,
    )
