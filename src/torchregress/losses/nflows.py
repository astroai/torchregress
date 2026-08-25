"""
Normalizing Flow loss functions for regression tasks.

This module provides loss functions for regression models that use
normalizing flows to model complex output distributions.
Uses the zuko package for efficient implementation of various flows.
"""

from collections.abc import Sequence
from typing import Any, Callable, Optional, cast

import torch
from torch import Tensor
from torch.nn import Module

try:
    from zuko.flows import MAF, NSF, RealNVP  # type: ignore[import-untyped]

    HAS_ZUKO = True
except ImportError:
    HAS_ZUKO = False

    MAF = NSF = RealNVP = None  # ty: ignore[invalid-assignment]  # zuko is optional; call sites are guarded by HAS_ZUKO

from ..utils.tensor_ops import masked_reduction
from .base import DistributionLoss
from .loss_registry import register_regression_loss


def create_flow_model(
    n_features: int,
    context_dim: int = 0,
    flow_type: str = "nsf",
    n_transforms: int = 5,
    hidden_features: Optional[int | Sequence[int]] = None,
    n_hidden_layers: int | None = None,
    **kwargs: Any,
) -> Module:
    """
    Create a zuko flow model with a torchregress-friendly interface.

    This helper exists primarily for examples/docs and keeps naming consistent with
    the rest of the library (``n_features``, ``context_dim``, ``n_transforms``).

    Parameters
    ----------
    n_features : int
        Target dimensionality (number of output dimensions).
    context_dim : int, default=0
        Context / conditioning dimension from the backbone model.
    flow_type : str, default="nsf"
        Flow architecture: ``"nsf"`` (Neural Spline Flow), ``"maf"`` (Masked
        Autoregressive Flow), or ``"realnvp"`` (RealNVP).  NSF provides the
        best expressivity-per-parameter and is the recommended default.
    n_transforms : int, default=5
        Number of coupling/autoregressive transforms in the flow.

        The zuko default is 5; the torchregress-harness benchmarks found that
        3 transforms is the sweet spot for small/medium tabular datasets
        (< 10k samples) — more transforms can overfit and produce overconfident
        densities.  5 is the safe library default for general use across
        dataset sizes and dimensionalities.
    hidden_features : int or sequence of int, optional
        Hidden layer widths for the coupling networks.  If ``None`` (default),
        resolves to ``[64, 64]`` — two hidden layers of 64 units each.

        **Recommended minimum: [64, 64].**  Reducing to [32, 32] under-expresses
        the coupling networks, producing overconfident (too-peaked) densities
        that cause severe under-coverage.  On diabetes at 60 epochs, [32, 32]
        achieves coverage 0.59 vs 0.66 for [64, 64] (+12% relative improvement).
    n_hidden_layers : int, optional
        Number of hidden layers when ``hidden_features`` is a single integer.
        Ignored when ``hidden_features`` is a sequence.  Default: 2.
    **kwargs : Any
        Additional keyword arguments forwarded to the zuko flow constructor.

    Returns
    -------
    torch.nn.Module
        A zuko normalizing flow instance with a ``context`` attribute set
        to ``context_dim`` (used by :class:`NormalizingFlowLoss`).

    Notes
    -----
    The ``context`` attribute is injected on the flow instance after
    construction so that :class:`NormalizingFlowLoss` can discover the
    context dimension.  This is a torchregress convention, not a zuko API.
    """
    if not HAS_ZUKO:
        raise ImportError("zuko is required for normalizing flows. Install torchregress[flows].")

    flow_type_key = flow_type.lower()
    flow_cls_map = {
        "nsf": NSF,
        "maf": MAF,
        "realnvp": RealNVP,
    }
    if flow_type_key not in flow_cls_map:
        raise ValueError(
            f"Unsupported flow_type {flow_type!r}. Expected one of {sorted(flow_cls_map)}."
        )

    hidden = _resolve_hidden_features(
        hidden_features=hidden_features,
        n_hidden_layers=n_hidden_layers,
    )
    flow_cls = flow_cls_map[flow_type_key]
    assert flow_cls is not None
    flow = flow_cls(
        features=n_features,
        context=context_dim,
        transforms=n_transforms,
        hidden_features=hidden,
        **kwargs,
    )
    # Store context dim in a stable attribute used by NormalizingFlowLoss.
    setattr(flow, "context", context_dim)
    return cast(Module, flow)


def _resolve_hidden_features(
    *,
    hidden_features: Optional[int | Sequence[int]],
    n_hidden_layers: int | None,
) -> list[int]:
    if hidden_features is None:
        width = 64
        depth = 2 if n_hidden_layers is None else int(n_hidden_layers)
        return [width] * max(depth, 1)

    if isinstance(hidden_features, Sequence) and not isinstance(hidden_features, (str, bytes)):
        resolved = [int(w) for w in cast(Sequence[int], hidden_features)]
        if not resolved:
            raise ValueError("hidden_features sequence must contain at least one layer width")
        return resolved

    width = int(hidden_features)
    depth = 1 if n_hidden_layers is None else int(n_hidden_layers)
    return [width] * max(depth, 1)


def create_flow_loss(*, reduction: str = "mean", **flow_kwargs: Any) -> "NormalizingFlowLoss":
    """Create a normalizing-flow loss from flow constructor arguments."""
    flow = create_flow_model(**flow_kwargs)
    return NormalizingFlowLoss(flow=flow, reduction=reduction)


def create_contrastive_flow_loss(
    *,
    reduction: str = "mean",
    temperature: float = 1.0,
    margin: float = 0.0,
    **flow_kwargs: Any,
) -> "ContrastiveFlowLoss":
    """Create a contrastive normalizing-flow loss from flow constructor arguments."""
    flow = create_flow_model(**flow_kwargs)
    return ContrastiveFlowLoss(
        flow=flow,
        reduction=reduction,
        temperature=temperature,
        margin=margin,
    )


@register_regression_loss("nflow")
class NormalizingFlowLoss(DistributionLoss):
    """
    Negative Log-Likelihood loss for conditional normalizing flow models using zuko.

    This loss allows modeling complex multi-dimensional target distributions
    for regression tasks using various normalizing flow architectures. The flow
    is conditioned on the model's output, allowing it to learn target distributions
    that depend on the input.

    Parameters
    ----------
    flow : torch.nn.Module
        A zuko Flow instance (RealNVP, MAF, NSF, etc.).  The flow must be
        created with a context dimension matching the model output.  Use
        :func:`create_flow_model` for a torchregress-friendly constructor
        with sensible defaults.
    reduction : str, default="mean"
        Reduction to apply to the per-sample losses:
        ``"mean"``, ``"sum"``, or ``"none"``.

    Architecture Guidance
    ---------------------
    The flow architecture is created externally and passed to the loss.
    Use :func:`create_flow_model` to construct flows with sensible
    defaults; see its docstring for detailed sweep evidence.

    Recommended minimums (summarised):

    | Parameter | Minimum | Notes |
    |---|---|---|
    | ``hidden_features`` | ``[64, 64]`` | 32-unit layers → overconfident densities |
    | | | under-coverage (0.59→0.66 on diabetes) |
    | ``n_transforms`` | 3 (tabular) | zuko default is 5; 3 is the sweet spot |
    | | | for datasets < 10k samples |
    | ``flow_type`` | ``"nsf"`` | Neural Spline Flow — best per-parameter |

    Scalar vs. multivariate tradeoff
        The [64, 64] / 3-transform configuration improves scalar
        calibration, but the more expressive flow learns tighter joint
        distributions on multivariate data, which can reduce
        JointCoverage.  For joint multivariate intervals, consider
        :class:`GaussianNLLLoss <torchregress.losses.GaussianNLLLoss>`
        if joint calibration is the priority.

    Mathematical Formulation
    ------------------------
    Normalizing flows transform a simple base distribution into a complex target
    distribution through a series of invertible transformations. For conditional
    flows, the transformation depends on context c (model output):

    .. math::

        \\text{NLL} = -\\log p_X(x|c) = -\\log p_Z(f(x|c)) - \\log|\\det(df/dx)|

    where p_Z is the density of the base distribution, f is the invertible
    transformation conditioned on c, and |det(df/dx)| is the absolute determinant
    of the Jacobian.

    Notes
    -----
    - Requires the ``zuko`` package (``pip install torchregress[flows]``).
    - The flow must be a trainable ``nn.Module`` — its parameters are
      trained alongside the model via backpropagation.
    - The model should output context vectors that condition the flow.
    - Different flow types (RealNVP, MAF, NSF) have different modeling
      capacities and computational characteristics.
    - Use :func:`create_flow_model` to construct flows with
      torchregress-friendly naming and validated defaults.
    - :func:`create_flow_loss` is a convenience that creates both the
      flow and the loss in one call.

    Examples
    --------
    >>> import torch
    >>> from torch import nn
    >>> from torchregress.losses.nflows import NormalizingFlowLoss, create_flow_model
    >>>
    >>> # Create a conditional flow with recommended defaults
    >>> flow = create_flow_model(
    ...     n_features=2,
    ...     context_dim=10,
    ...     n_transforms=3,
    ...     hidden_features=[64, 64],
    ... )
    >>> loss_fn = NormalizingFlowLoss(flow=flow)
    >>>
    >>> # Model outputs context vectors
    >>> class MyModel(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.net = nn.Linear(5, 10)
    ...     def forward(self, x):
    ...         return self.net(x)
    >>>
    >>> model = MyModel()
    >>> x = torch.randn(32, 5)
    >>> context = model(x)  # [32, 10]
    >>> target = torch.randn(32, 2)  # [32, 2]
    >>> loss = loss_fn(context, target)
    >>> loss.backward()
    """

    def __init__(
        self,
        flow: Module,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)

        if not isinstance(flow, Module):
            raise TypeError(f"flow must be a torch.nn.Module (zuko Flow), got {type(flow)}")

        self.flow = flow

        # Extract flow configuration for validation
        # Zuko flows store dimensions in the base distribution
        try:
            base_fn = cast(Callable[[], Any], getattr(flow, "base"))
            base_dist = base_fn()
            self.n_features = base_dist.event_shape[0] if len(base_dist.event_shape) > 0 else 1
        except Exception as e:
            raise ValueError(f"Could not extract feature dimension from flow: {e}")

        # Try to get context dimension - check if it was added by create_flow_model
        self.context_dim = getattr(flow, "context", None)  # May be None if not set

    def _extract_distribution_parameters(self, y_pred: Tensor) -> Tensor:
        """
        Extract context from model predictions.

        Args:
            y_pred: Model predictions serving as context for the flow
                    Shape: [batch_size, context_dim]

        Returns:
            Context tensor for conditioning the flow
        """
        return y_pred

    def distribution(self, y_pred: Tensor) -> Any:
        """Return the conditional distribution induced by ``y_pred`` context."""
        context = self._extract_distribution_parameters(y_pred)
        if self.context_dim is None and context.numel() > 0:
            self.context_dim = context.shape[-1] if context.dim() > 0 else 0
        if self.context_dim is not None and self.context_dim > 0:
            return cast(Any, self.flow(context))
        return cast(Any, self.flow())

    def log_prob(self, y_pred: Tensor, target: Tensor) -> Tensor:
        """Return per-sample log probability for ``target`` under the flow."""
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        if target.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features in target, got {target.shape[-1]}"
            )
        dist = self.distribution(y_pred)
        return cast(Tensor, dist.log_prob(target))

    def _sample_tensor(self, y_pred: Tensor, n_samples: int) -> Tensor:
        samples = self.sample(y_pred, n_samples=n_samples)
        if samples.dim() == 2:
            return samples.unsqueeze(1)
        return samples

    def quantile(
        self,
        y_pred: Tensor,
        levels: Tensor | list[float] | tuple[float, ...],
        *,
        n_samples: int = 512,
    ) -> Tensor:
        """Approximate scalar predictive quantiles using flow samples."""
        if self.n_features != 1:
            raise ValueError("quantile currently only supports scalar normalizing-flow targets")
        level_tensor = torch.as_tensor(levels, dtype=y_pred.dtype, device=y_pred.device).reshape(-1)
        level_tensor = level_tensor.clamp(0.0, 1.0)
        samples = self._sample_tensor(y_pred, n_samples=n_samples)[..., 0]
        sorted_samples = torch.sort(samples, dim=1).values
        n_draws = sorted_samples.shape[1]
        if n_draws <= 1:
            return sorted_samples.expand(-1, level_tensor.numel())
        positions = level_tensor * float(n_draws - 1)
        lower_idx = torch.floor(positions).long().clamp(0, n_draws - 1)
        upper_idx = torch.ceil(positions).long().clamp(0, n_draws - 1)
        weight = (positions - lower_idx.to(positions.dtype)).view(1, -1)
        lower = sorted_samples.index_select(1, lower_idx)
        upper = sorted_samples.index_select(1, upper_idx)
        return lower + (upper - lower) * weight

    def cdf(
        self,
        y_pred: Tensor,
        values: Tensor,
        *,
        n_samples: int = 512,
    ) -> Tensor:
        """Approximate scalar predictive CDF values using flow samples."""
        if self.n_features != 1:
            raise ValueError("cdf currently only supports scalar normalizing-flow targets")
        value_tensor = values
        squeeze_last = False
        if value_tensor.dim() == 1:
            value_tensor = value_tensor.unsqueeze(-1)
            squeeze_last = True
        elif value_tensor.dim() == 2 and value_tensor.shape[-1] == 1:
            squeeze_last = True
        samples = self._sample_tensor(y_pred, n_samples=n_samples)[..., 0]
        cdf = (samples.unsqueeze(-1) <= value_tensor.unsqueeze(1)).to(samples.dtype).mean(dim=1)
        return cdf.squeeze(-1) if squeeze_last else cdf

    def _calculate_nll(
        self,
        target: Tensor,
        context: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calculate negative log-likelihood for the conditional flow model.

        Args:
            target: Target values [batch_size, n_features]
            context: Context vectors from model [batch_size, context_dim]
            mask: Optional mask [batch_size, n_features]

        Returns:
            Negative log-likelihood [batch_size]
        """
        del mask
        return cast(Tensor, -self.log_prob(context, target))

    @staticmethod
    def _sample_mask(mask: Optional[Tensor], target: Tensor) -> Optional[Tensor]:
        """Collapse masks to the sample dimension and reject partial feature masking."""
        if mask is None:
            return None
        if mask.shape[0] != target.shape[0]:
            raise ValueError("mask batch dimension must match target batch dimension")
        mask_bool = mask.bool()
        if mask_bool.dim() == 1:
            return mask_bool
        flattened = mask_bool.reshape(mask_bool.shape[0], -1)
        row_all = flattened.all(dim=1)
        row_any = flattened.any(dim=1)
        if not torch.equal(row_all, row_any):
            raise ValueError(
                "NormalizingFlowLoss only supports sample-level masks; "
                "feature-wise masking is not valid for joint flow densities"
            )
        return row_all

    @staticmethod
    def _mask_invalid_samples(target: Tensor, sample_mask: Optional[Tensor]) -> Tensor:
        """Replace fully masked samples with zeros to avoid NaNs during ignored evaluations."""
        if sample_mask is None:
            return target
        expand_shape = (sample_mask.shape[0],) + (1,) * max(target.dim() - 1, 0)
        expanded = sample_mask.reshape(expand_shape)
        return torch.where(expanded, target, torch.zeros_like(target))

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculate normalizing flow negative log-likelihood loss.

        Args:
            y_pred: Context from model [batch_size, context_dim]
            target: Ground truth values [batch_size, n_features]
            mask: Optional boolean mask [batch_size, n_features]
            weights: Optional sample weights [batch_size]

        Returns:
            Negative log-likelihood loss

        Raises:
            ValueError: If shapes don't match expected dimensions
        """
        # Validate target shape
        target_feature_dim = 1 if target.dim() == 1 else target.shape[-1]
        if target_feature_dim != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features in target, got {target_feature_dim}"
            )

        # Infer and store context_dim on first forward pass
        if self.context_dim is None and y_pred.numel() > 0:
            self.context_dim = y_pred.shape[-1] if y_pred.dim() > 0 else 0

        sample_mask = self._sample_mask(mask, target)
        target_eval = self._mask_invalid_samples(target, sample_mask)

        # Extract context
        context = self._extract_distribution_parameters(y_pred)

        # Calculate negative log-likelihood
        nll = self._calculate_nll(target_eval, context, sample_mask)

        # Apply weights if provided
        if weights is not None:
            # Handle different weight shapes
            if weights.dim() > 1 and weights.shape[1] > 1:
                # Average across features if weights are per-feature
                weights = weights.mean(dim=1)
            nll = nll * weights

        # Apply reduction
        return masked_reduction(nll, sample_mask, self.reduction)

    def sample(self, y_pred: Tensor, n_samples: int = 1) -> Tensor:
        """
        Generate samples from the conditional flow distribution.

        Args:
            y_pred: Context from model [batch_size, context_dim]
            n_samples: Number of samples to generate per input
                Default: 1

        Returns:
            Samples [batch_size, n_samples, n_features] or [batch_size, n_features] if n_samples=1
        """
        batch_size = y_pred.shape[0]
        dist = self.distribution(y_pred)

        if self.context_dim is not None and self.context_dim > 0:
            if n_samples == 1:
                samples = cast(Tensor, dist.sample((1,))).transpose(0, 1).squeeze(1)
            else:
                samples = cast(Tensor, dist.sample((n_samples,))).transpose(0, 1)
        else:
            if n_samples == 1:
                samples = cast(Tensor, dist.sample((batch_size,)))
            else:
                samples = cast(Tensor, dist.sample((batch_size, n_samples)))

        return cast(Tensor, samples)


@register_regression_loss("contrastive_nflow")
class ContrastiveFlowLoss(NormalizingFlowLoss):
    """
    Contrastive likelihood-ratio loss for parameter-conditioned normalizing flows.

    This loss extends :class:`NormalizingFlowLoss` from plain conditional density
    estimation to ranking-style training. Each observed target is scored under:

    - a ``positive`` context corresponding to the generating parameter setting, and
    - one or more ``negative`` contexts corresponding to alternative hypotheses.

    The loss is a multiclass InfoNCE-style objective over flow log-likelihoods:

    .. math::

        \\mathcal{L}(x, c^+, \\{c^-_k\\}) =
        -\\log \\frac{\\exp((\\log p(x \\mid c^+) - m)/T)}
        {\\exp((\\log p(x \\mid c^+) - m)/T) + \\sum_k \\exp(\\log p(x \\mid c^-_k)/T)}

    where ``T`` is the temperature and ``m`` is an optional positive-class margin.

    This is useful when the downstream task is parameter estimation or robust
    hypothesis ranking under nuisance/domain-shift conditions rather than generic
    density modeling alone.
    """

    def __init__(
        self,
        flow: Module,
        reduction: str = "mean",
        *,
        temperature: float = 1.0,
        margin: float = 0.0,
    ) -> None:
        super().__init__(flow=flow, reduction=reduction)
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if margin < 0:
            raise ValueError(f"margin must be >= 0, got {margin}")
        self.temperature = float(temperature)
        self.margin = float(margin)

    def _prepare_negative_context(
        self,
        negative_context: Tensor,
        batch_size: int,
        *,
        shared_negative_context: bool | None = None,
    ) -> Tensor:
        """Normalize negative-context inputs to shape ``[batch, n_negatives, context_dim]``."""
        context_dim = 0 if self.context_dim is None else int(self.context_dim)
        if negative_context.dim() == 2:
            if negative_context.shape[0] == 0:
                raise ValueError("negative_context must include at least one negative hypothesis")
            if negative_context.shape[-1] != context_dim:
                raise ValueError(
                    "negative_context last dimension must match the positive context dimension"
                )
            if shared_negative_context is None and negative_context.shape[0] == batch_size:
                raise ValueError(
                    "negative_context with shape [N, context_dim] is ambiguous when N matches "
                    "the batch size. Pass [batch, 1, context_dim] for per-sample negatives, "
                    "[1, n_negatives, context_dim] for a shared bank, or set "
                    "shared_negative_context explicitly."
                )
            if shared_negative_context is False:
                if negative_context.shape[0] != batch_size:
                    raise ValueError(
                        "Per-sample negative_context must have batch dimension matching y_pred"
                    )
                return negative_context.unsqueeze(1)
            return negative_context.unsqueeze(0).expand(batch_size, -1, -1)
        if negative_context.dim() == 3:
            if negative_context.shape[1] == 0:
                raise ValueError("negative_context must include at least one negative hypothesis")
            if negative_context.shape[-1] != context_dim:
                raise ValueError(
                    "negative_context last dimension must match the positive context dimension"
                )
            if negative_context.shape[0] == batch_size:
                return negative_context
            if negative_context.shape[0] == 1:
                return negative_context.expand(batch_size, -1, -1)
            raise ValueError(
                "3D negative_context must have shape [batch, n_negatives, context_dim] "
                "or [1, n_negatives, context_dim]"
            )
        raise ValueError(
            "negative_context must have shape [batch, context_dim], "
            "[n_negatives, context_dim], [1, n_negatives, context_dim], "
            "or [batch, n_negatives, context_dim]"
        )

    def negative_log_likelihoods(
        self,
        y_pred: Tensor,
        target: Tensor,
        negative_context: Tensor,
        *,
        shared_negative_context: bool | None = None,
    ) -> Tensor:
        """Return per-sample log-likelihoods under negative contexts."""
        if self.context_dim is None and y_pred.numel() > 0:
            self.context_dim = y_pred.shape[-1] if y_pred.dim() > 0 else 0

        normalized_neg = self._prepare_negative_context(
            negative_context,
            batch_size=y_pred.shape[0],
            shared_negative_context=shared_negative_context,
        )
        batch_size, n_negatives, _ = normalized_neg.shape
        flat_context = normalized_neg.reshape(batch_size * n_negatives, -1)
        target_matrix = target if target.dim() > 1 else target.unsqueeze(-1)
        flat_target = (
            target_matrix.unsqueeze(1)
            .expand(-1, n_negatives, -1)
            .reshape(batch_size * n_negatives, -1)
        )
        return self.log_prob(flat_context, flat_target).reshape(batch_size, n_negatives)

    def log_likelihood_ratio(
        self,
        y_pred: Tensor,
        target: Tensor,
        negative_context: Tensor,
        *,
        shared_negative_context: bool | None = None,
    ) -> Tensor:
        """Return ``log p(target|positive) - log p(target|negative)`` for each negative context."""
        positive = self.log_prob(y_pred, target).unsqueeze(1)
        negative = self.negative_log_likelihoods(
            y_pred,
            target,
            negative_context,
            shared_negative_context=shared_negative_context,
        )
        return positive - negative

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculate a contrastive flow loss using positive and negative contexts.

        Keyword Args:
            negative_context: Alternative contexts/hypotheses with shape
                ``[batch, context_dim]``, ``[n_negatives, context_dim]``, or
                ``[batch, n_negatives, context_dim]``. Prefer
                ``[batch, n_negatives, context_dim]`` or
                ``[1, n_negatives, context_dim]`` to avoid ambiguity.
            negative_y_pred: Alias for ``negative_context``.
            shared_negative_context: Set to ``True`` to interpret a 2-D
                ``negative_context`` as a shared bank.
        """
        negative_context = cast(
            Optional[Tensor],
            kwargs.pop("negative_context", kwargs.pop("negative_y_pred", None)),
        )
        shared_negative_context = cast(
            Optional[bool],
            kwargs.pop("shared_negative_context", None),
        )
        if negative_context is None:
            raise ValueError("ContrastiveFlowLoss requires negative_context in forward()")

        target_feature_dim = 1 if target.dim() == 1 else target.shape[-1]
        if target_feature_dim != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features in target, got {target_feature_dim}"
            )

        if self.context_dim is None and y_pred.numel() > 0:
            self.context_dim = y_pred.shape[-1] if y_pred.dim() > 0 else 0

        sample_mask = self._sample_mask(mask, target)
        target_eval = self._mask_invalid_samples(target, sample_mask)
        positive_log_prob = self.log_prob(y_pred, target_eval)
        negative_log_prob = self.negative_log_likelihoods(
            y_pred,
            target_eval,
            negative_context,
            shared_negative_context=shared_negative_context,
        )

        pos_score = (positive_log_prob - self.margin) / self.temperature
        neg_score = negative_log_prob / self.temperature
        logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
        loss = -pos_score + torch.logsumexp(logits, dim=1)

        if weights is not None:
            if weights.dim() > 1 and weights.shape[1] > 1:
                weights = weights.mean(dim=1)
            loss = loss * weights

        return masked_reduction(loss, sample_mask, self.reduction)
