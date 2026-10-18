"""
Representation-level losses for deep imbalanced regression.

Both losses in this module shape the **feature space** of a regression model
instead of reweighting the output-space loss:

* :class:`PrimeLoss` -- PRIME (Lim et al., ICML 2025): learnable feature proxies
  whose targets tile the target range uniformly.  A proxy-geometry loss
  (t-SNE-style KL between label-space and feature-space affinity distributions)
  orders the proxies by target value, and a soft-label alignment loss pulls every
  sample feature towards the proxies whose targets are close to its label.
  Because the proxies are persistent parameters, every mini-batch (including
  batches that contain only majority targets) contributes a gradient to the
  under-represented target regions.
* :class:`ConR` -- ConR (Keramati et al., ICLR 2024): a contrastive regularizer
  with label-similarity positive pairs, prediction-similarity negative pairs and
  inverse-label-density relative pushing.

Feature Distribution Smoothing -- the other representation-level family for
imbalanced regression in this library -- lives in
:mod:`torchregress.losses.imbalanced`
(:class:`~torchregress.losses.imbalanced.FeatureDistributionSmoother`); it is a
feature-statistics calibrator, not a loss, and is not duplicated here.

See ``docs/losses/prime_conr.md`` for the full derivations, assumptions and a
comparison with output-space reweighting (LDS, Balanced MSE) and Rank-N-Contrast.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..utils.validation import validate_positive
from .base import BaseLoss
from .loss_registry import register_regression_loss

__all__ = ["ConR", "PrimeLoss", "conr_loss", "uniform_proxy_targets"]

#: Effective-number parameter used by ``PrimeLoss(class_balanced=True)`` (paper value).
_CB_BETA = 0.99


# ── shared helpers ────────────────────────────────────────────────────────────


def _inverse_frequency_weights(counts: Tensor, smoothing: float = 0.0) -> Tensor:
    """Per-bin weights proportional to ``1 / (count + smoothing)``.

    The weights are scaled so that their mean over the *training samples* is one
    (``sum_b count_b * w_b / N == 1``), which keeps a downstream loss on the same
    scale as its unweighted version.  Bins with zero (smoothed) count get weight
    zero instead of a huge ``1 / smoothing``.
    """
    smooth = counts + smoothing
    tiny = torch.finfo(smooth.dtype).tiny
    inv = torch.where(smooth > 0, 1.0 / smooth.clamp_min(tiny), torch.zeros_like(smooth))
    n_train = counts.sum()
    per_sample_mean = (counts * inv).sum() / n_train.clamp_min(1.0)
    return inv / per_sample_mean.clamp_min(tiny)


def _effective_number_weights(counts: Tensor, beta: float = _CB_BETA) -> Tensor:
    """Class-balanced effective-number weights (Cui et al., 2019).

    ``w_b = (1 - beta) / (1 - beta ** count_b)`` for populated bins and zero for
    empty bins, normalised so that the mean weight over the training samples is
    one.  The normalisation is an implementation choice (the PRIME paper tunes
    the trade-off coefficient instead).
    """
    beta_t = torch.tensor(beta, dtype=counts.dtype, device=counts.device)
    denom = (1.0 - beta_t.pow(counts)).clamp_min(torch.finfo(counts.dtype).tiny)
    w = torch.where(counts > 0, (1.0 - beta_t) / denom, torch.zeros_like(counts))
    n_train = counts.sum()
    per_sample_mean = (counts * w).sum() / n_train.clamp_min(1.0)
    return w / per_sample_mean.clamp_min(torch.finfo(counts.dtype).tiny)


def _scalar_labels(labels: Tensor, n_samples: int, name: str = "target") -> Tensor:
    """Return ``[N]`` scalar labels from ``[N]`` or ``[N, 1]`` input."""
    if labels.dim() == 2 and labels.shape[1] == 1:
        labels = labels.reshape(n_samples)
    if labels.dim() != 1 or labels.shape[0] != n_samples:
        raise ValueError(
            f"{name} must be scalar labels with shape [N] or [N, 1] "
            f"where N = {n_samples}; got {tuple(labels.shape)}"
        )
    return labels


def _labels_for_binning(targets: Tensor) -> Tensor:
    """Map arbitrary targets to one scalar per sample for histogram binning."""
    if targets.dim() <= 1:
        return targets.reshape(-1)
    # Multi-output: bin by the mean coordinate (documented simplification).
    return targets.mean(dim=-1)


def _bin_indices(values: Tensor, edges: Tensor) -> Tensor:
    """Bin indices in ``[0, n_bins - 1]`` for bins with ``n_bins + 1`` edges."""
    n_bins = edges.numel() - 1
    idx = torch.searchsorted(edges, values, right=True) - 1
    return idx.clamp(0, n_bins - 1)


# ── PRIME ─────────────────────────────────────────────────────────────────────


def uniform_proxy_targets(y_min: float, y_max: float, n_proxies: int) -> Tensor:
    """Uniformly spaced proxy targets inside ``[y_min, y_max]``.

    PRIME places the proxy targets uniformly across the target range: the
    ``n_proxies`` points are the interior points of the uniform grid that splits
    ``[y_min, y_max]`` into ``n_proxies + 1`` equal-width intervals (i.e. the
    ``(n_proxies + 1)``-quantiles of the range, endpoints excluded).

    Args:
        y_min: Lower end of the target range.
        y_max: Upper end of the target range (must be larger than ``y_min``).
        n_proxies: Number of proxy targets ``C`` (at least one).

    Returns:
        ``[C]`` float tensor of uniformly spaced proxy targets.
    """
    if n_proxies < 1:
        raise ValueError(f"n_proxies must be >= 1, got {n_proxies}")
    y_min, y_max = float(y_min), float(y_max)
    if not y_max > y_min:
        raise ValueError(f"y_max ({y_max}) must be larger than y_min ({y_min})")
    return torch.linspace(y_min, y_max, n_proxies + 2)[1:-1]


@register_regression_loss("prime")
class PrimeLoss(BaseLoss):
    """PRIME proxy-alignment loss for imbalanced regression (Lim et al., 2025).

    PRIME introduces ``C`` learnable proxies ``{(z_j^p, y_j^p)}_{j=1..C}`` with
    *feature* points ``z_j^p`` and *targets* ``y_j^p`` that tile the training
    target range uniformly.  Two terms are added to the task regression loss:

    **Proxy geometry** (paper Eq. 1-4).  With ``d_t(y_i, y_j) = |y_i - y_j|`` and
    ``d_f(z_i, z_j) = ||z_i - z_j||_2``:

    .. code-block:: text

        p_ij = exp(-tau_t d_t(y_i^p, y_j^p)) / sum_{k != l} exp(-tau_t d_t(y_k^p, y_l^p))
        q_ij = exp(-tau_f d_f(z_i^p, z_j^p)) / sum_{k != l} exp(-tau_f d_f(z_k^p, z_l^p))
        L_proxy = sum_{i != j} [ p_ij log(p_ij / q_ij)
                                 - alpha * d_t(y_i^p, y_j^p) * (1 - cos(z_i^p, z_j^p))^2 ]

    The KL term orders the proxy features by target value (t-SNE style); the
    negative weighted cosine-distance term (``alpha = spread_weight``) stops the
    trivial collapse of all proxies onto one point.

    **Alignment** (paper Eq. 5-7).  For a sample feature ``z`` with label ``y``:

    .. code-block:: text

        T_j = exp(-tau_t |y - y_j^p|) / sum_k exp(-tau_t |y - y_k^p|)
        A_j = exp(-tau_f ||z - z_j^p||) / sum_k exp(-tau_f ||z - z_k^p||)
        L_align = - sum_j c_j T_j log A_j

    i.e. a soft-label cross-entropy between the target association ``T`` and the
    feature association ``A``.  As ``tau_t -> 0`` this tends to an InfoNCE-style
    term whose positive is the single nearest proxy.  With
    ``class_balanced=True`` the per-proxy coefficients ``c_j`` are the
    class-balanced effective-number weights (paper Eq. 10, ``beta = 0.99``),
    normalised to mean one over the training samples; otherwise ``c_j = 1``.

    The total contribution is ``proxy_weight * L_proxy + alignment_weight *
    mean_batch(L_align)`` (``lambda_p`` / ``lambda_a`` in paper Eq. 8), designed
    to be added to a task regression loss.  The proxies are persistent
    parameters, so they should be passed to the optimizer together with the model
    parameters.

    Args:
        feature_dim: Dimensionality ``D`` of the feature vectors ``y_pred``.
        n_proxies: Number of proxies ``C`` (default 100).
        label_range: Optional ``(y_min, y_max)`` target range.  When given the
            proxy targets are initialised in ``__init__`` and ``forward`` can be
            used before :meth:`fit`; :meth:`fit` then only computes the
            class-balanced proxy weights.  When omitted the range is taken from
            the training targets passed to :meth:`fit`.
        temperature: Feature-space temperature ``tau_f`` used by ``A`` (paper
            Eq. 5) and ``Q`` (paper Eq. 2).  Default 0.1.
        proxy_temperature: Label-space temperature ``tau_t`` used by ``T`` (paper
            Eq. 6) and ``P`` (paper Eq. 1), in target units.  Default 1.0.
        alignment_weight: Weight ``lambda_a`` of the alignment term.  Default 1.0.
        proxy_weight: Weight ``lambda_p`` of the proxy-geometry term.  Default 1.0.
        spread_weight: Coefficient ``alpha`` of the proxy-separation term.
            Default 0.001 (same order of magnitude as the paper's tuned values).
        class_balanced: Apply class-balanced (effective-number) proxy weights to
            the alignment term.  Default ``False``.
        reduction: ``'none' | 'mean' | 'sum'``.  The scalar proxy term is added
            to every per-anchor alignment value, so the reductions are mutually
            consistent and ``'mean'`` equals the training objective.
        seed: Optional seed for the proxy initialisation (He normal, as in the
            paper's implementation details).  ``None`` uses the global RNG.

    Shapes:
        ``y_pred``: ``[N, D]`` feature vectors.  ``target``: ``[N]`` or ``[N, 1]``
        scalar labels.  ``mask``: ``[N]`` or ``[N, 1]`` boolean/0-1, per-sample
        (masked samples are dropped by the reduction).  ``weights``: ``[N]`` or
        ``[N, 1]`` non-negative sample weights, following the shared
        :meth:`BaseLoss._reduce` contract.

    Raises:
        RuntimeError: If ``forward`` is called before :meth:`fit` and no
            ``label_range`` was given.
        ValueError: For non-scalar labels, mismatched feature dimension, or
            invalid constructor arguments.

    Example:
        >>> import torch
        >>> from torchregress.losses.prime import PrimeLoss
        >>>
        >>> loss_fn = PrimeLoss(feature_dim=8, n_proxies=20, seed=0).fit(y_train)
        >>> # proxies are parameters: optimise them with the model
        >>> params = list(model.parameters()) + list(loss_fn.parameters())
        >>> optimizer = torch.optim.Adam(params, lr=1e-3)
        >>> features = model.encode(x)                  # [B, 8]
        >>> loss = mse(model.head(features), y) + loss_fn(features, y)
        >>> loss.backward()
        >>> optimizer.step()

    Notes:
        Implementation choices (the core design follows the paper):

        * Proxy targets are the uniform grid over ``[y_min, y_max]`` from
          :meth:`fit` (paper: ``{y_j^p}`` "uniformly distributed across the
          target values", Sec. 3.2 and Table 9's ``Unif.`` ablation).  Passing
          ``label_range`` pins the grid, which is useful when the training split
          does not cover the full test range.
        * The feature-space distance ``d_f`` is Euclidean; the paper leaves the
          metric generic and uses a cosine-distance separation term (reproduced
          exactly).
        * Multi-dimensional targets are **not** supported: the paper defines
          multi-dimensional proxy targets via K-means cluster centres, which this
          implementation does not include.  Use scalar labels (``[N]`` /
          ``[N, 1]``).
        * With ``reduction='none'`` the proxy term is broadcast to every anchor;
          the training objective is the ``'mean'`` reduction.

    Reference:
        Lim, J., Lee, S., Um, D., Park, S.-U., & Shin, J.  "PRIME: Deep
        Imbalanced Regression with Proxies."  ICML 2025.
        https://proceedings.mlr.press/v267/lim25a.html
    """

    proxies: Tensor
    proxy_targets: Tensor
    proxy_weights: Tensor

    def __init__(
        self,
        feature_dim: int,
        n_proxies: int = 100,
        *,
        label_range: Optional[tuple[float, float]] = None,
        temperature: float = 0.1,
        proxy_temperature: float = 1.0,
        alignment_weight: float = 1.0,
        proxy_weight: float = 1.0,
        spread_weight: float = 0.001,
        class_balanced: bool = False,
        reduction: str = "mean",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(reduction=reduction)
        if int(feature_dim) < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
        if int(n_proxies) < 1:
            raise ValueError(f"n_proxies must be >= 1, got {n_proxies}")

        self.feature_dim = int(feature_dim)
        self.n_proxies = int(n_proxies)
        self.temperature = float(validate_positive(temperature, "temperature"))
        self.proxy_temperature = float(validate_positive(proxy_temperature, "proxy_temperature"))
        self.alignment_weight = float(
            validate_positive(alignment_weight, "alignment_weight", allow_zero=True)
        )
        self.proxy_weight = float(validate_positive(proxy_weight, "proxy_weight", True))
        self.spread_weight = float(validate_positive(spread_weight, "spread_weight", True))
        self.class_balanced = bool(class_balanced)

        if label_range is not None:
            lo, hi = float(label_range[0]), float(label_range[1])
            if not hi > lo:
                raise ValueError(f"label_range must satisfy y_max > y_min, got ({lo}, {hi})")
            self.label_range: Optional[tuple[float, float]] = (lo, hi)
        else:
            self.label_range = None

        # Proxy features: He normal init, jointly optimised with the model.
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
        proxies = torch.empty(self.n_proxies, self.feature_dim)
        nn.init.kaiming_normal_(proxies, nonlinearity="leaky_relu", generator=generator)
        self.proxies = nn.Parameter(proxies)

        self.register_buffer("proxy_targets", torch.zeros(0))
        self.register_buffer("proxy_weights", torch.ones(self.n_proxies))
        if self.label_range is not None:
            self.proxy_targets = uniform_proxy_targets(*self.label_range, self.n_proxies)

    # -- fitting ----------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether proxy targets are available (from ``fit`` or ``label_range``)."""
        return self.proxy_targets.numel() == self.n_proxies

    def fit(self, train_targets: Tensor) -> PrimeLoss:
        """Set the proxy targets (and class-balanced proxy weights) from training labels.

        The proxy targets are the uniform grid of ``C`` interior points that
        splits the observed target range into ``C + 1`` equal-width intervals.
        If ``label_range`` was given to the constructor the grid is pinned to
        that range; only the proxy-frequency weights are recomputed here.

        Args:
            train_targets: Training labels, ``[N]`` or ``[N, 1]`` (scalar
                targets, as required by the PRIME proxy design).

        Returns:
            ``self``, so calls can be chained.
        """
        labels = _labels_for_binning(train_targets.detach().float())
        finite = torch.isfinite(labels)
        if not bool(finite.any()):
            raise ValueError("fit() needs at least one finite training target")
        labels = labels[finite]

        if self.label_range is not None:
            proxy_targets = uniform_proxy_targets(*self.label_range, self.n_proxies)
        else:
            y_min, y_max = float(labels.min()), float(labels.max())
            if not y_max > y_min:
                raise ValueError(
                    "train_targets must contain at least two distinct finite values "
                    f"(got min == max == {y_min}); pass label_range to pin the proxy range"
                )
            proxy_targets = uniform_proxy_targets(y_min, y_max, self.n_proxies)
        self.proxy_targets = proxy_targets.to(device=self.proxies.device, dtype=torch.float32)

        if self.class_balanced:
            # Assign every training sample to its nearest proxy target and apply
            # the paper's class-balanced (PRIME + CB) effective-number weights.
            proxy_targets = proxy_targets.to(device=labels.device, dtype=labels.dtype)
            assignments = (labels[:, None] - proxy_targets[None, :]).abs().argmin(dim=1)
            counts = torch.bincount(assignments, minlength=self.n_proxies).to(labels.dtype)
            weights = _effective_number_weights(counts)
            self.proxy_weights = weights.to(dtype=torch.float32)
        else:
            self.proxy_weights = torch.ones(self.n_proxies, dtype=torch.float32)
        return self

    # -- loss terms -------------------------------------------------------

    def _proxy_geometry(self, proxy_targets: Tensor, proxy_features: Tensor) -> Tensor:
        """``L_proxy`` of paper Eq. (4): KL(P || Q) minus the spread term."""
        n = self.n_proxies
        pair_offdiag = ~torch.eye(n, dtype=torch.bool, device=proxy_features.device)

        dt = (proxy_targets[:, None] - proxy_targets[None, :]).abs()
        df = torch.cdist(proxy_features, proxy_features, p=2)

        logits_p = (-self.proxy_temperature * dt).masked_fill(~pair_offdiag, float("-inf"))
        logits_q = (-self.temperature * df).masked_fill(~pair_offdiag, float("-inf"))
        log_p = torch.log_softmax(logits_p.reshape(-1), dim=0)[pair_offdiag.reshape(-1)]
        log_q = torch.log_softmax(logits_q.reshape(-1), dim=0)[pair_offdiag.reshape(-1)]
        p = log_p.exp()
        # KL(P || Q) = sum p log(p / q), with 0 log 0 = 0 handled by xlogy.
        kl = (torch.special.xlogy(p, p) - p * log_q).sum()

        cos = F.normalize(proxy_features, dim=-1) @ F.normalize(proxy_features, dim=-1).T
        spread = (dt * (1.0 - cos) ** 2).masked_fill(~pair_offdiag, 0.0).sum()
        return kl - self.spread_weight * spread

    def _alignment(
        self, features: Tensor, labels: Tensor, proxy_targets: Tensor, proxy_features: Tensor
    ) -> Tensor:
        """Per-anchor alignment term ``L_align`` of paper Eq. (7)."""
        dt = (labels[:, None] - proxy_targets[None, :]).abs()
        df = torch.cdist(features, proxy_features, p=2)
        log_t = F.log_softmax(-self.proxy_temperature * dt, dim=-1)
        log_a = F.log_softmax(-self.temperature * df, dim=-1)
        target_assoc = log_t.exp()
        coeff = self.proxy_weights.to(dtype=features.dtype, device=features.device)
        return -(coeff[None, :] * target_assoc * log_a).sum(dim=-1)

    # -- forward ----------------------------------------------------------

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the PRIME representation loss.

        Args:
            y_pred: Feature vectors ``[N, D]`` (the encoder output, not the
                regression head output).
            target: Scalar labels ``[N]`` or ``[N, 1]``.
            mask: Optional per-sample validity mask (``[N]`` / ``[N, 1]``);
                masked samples contribute nothing to the reduction.
            weights: Optional per-sample weights following the shared reduction
                contract.
            **kwargs: Accepted for API consistency and ignored.

        Returns:
            Scalar loss for ``reduction='mean'`` / ``'sum'``, or the
            ``[N]`` per-anchor tensor for ``reduction='none'``.

        Raises:
            RuntimeError: If proxy targets are not available yet.
            ValueError: For malformed shapes or non-scalar labels.
        """
        if y_pred.dim() != 2:
            raise ValueError(f"y_pred must be [N, D] feature vectors, got {tuple(y_pred.shape)}")
        n_samples, feature_dim = y_pred.shape
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"y_pred feature dim {feature_dim} does not match "
                f"PrimeLoss(feature_dim={self.feature_dim})"
            )
        if not y_pred.is_floating_point():
            raise ValueError("y_pred must be a floating-point tensor of features")
        if not self.is_fitted:
            raise RuntimeError(
                "PrimeLoss has no proxy targets yet: call fit(train_targets) before "
                "forward(), or pass label_range=(y_min, y_max) to the constructor."
            )

        labels = _scalar_labels(target, n_samples).to(dtype=y_pred.dtype)
        mask_flat: Optional[Tensor] = None
        if mask is not None:
            if mask.numel() != n_samples:
                raise ValueError(
                    f"mask must have one entry per sample ({n_samples}), "
                    f"got shape {tuple(mask.shape)}"
                )
            mask_flat = mask.reshape(n_samples).to(device=y_pred.device, dtype=torch.bool)
            # Keep masked (possibly NaN) labels out of the softmax: the zero-fill
            # policy of ``_reduce`` cannot repair NaNs created upstream.
            labels = torch.where(mask_flat, labels, torch.zeros_like(labels))
        if weights is not None and weights.numel() != n_samples:
            raise ValueError(
                f"weights must have one entry per sample ({n_samples}), "
                f"got shape {tuple(weights.shape)}"
            )

        proxy_targets = self.proxy_targets.to(device=y_pred.device, dtype=y_pred.dtype)
        proxy_features = self.proxies.to(device=y_pred.device, dtype=y_pred.dtype)

        alignment = self._alignment(y_pred, labels, proxy_targets, proxy_features)
        proxy_loss = self._proxy_geometry(proxy_targets, proxy_features)
        per_anchor = self.alignment_weight * alignment + self.proxy_weight * proxy_loss
        return self._reduce(per_anchor, mask_flat, weights)


# ── ConR ──────────────────────────────────────────────────────────────────────


def conr_loss(
    features: Tensor,
    labels: Tensor,
    predictions: Tensor,
    *,
    label_threshold: float = 0.2,
    prediction_threshold: Optional[float] = None,
    temperature: float = 0.07,
    push_scale: float = 0.2,
    density_weights: Optional[Tensor] = None,
) -> Tensor:
    """Per-anchor ConR regularizer (Keramati et al., 2024, Eq. 1-2).

    For anchor ``i`` with feature ``z_i``, label ``y_i`` and prediction
    ``p_i``, with ``y ~ z`` denoting L2 label distance (absolute difference for
    scalar labels) and ``p_dist`` the absolute difference of the scalar
    predictions (multi-output predictions are averaged over the last dim):

    .. code-block:: text

        pos_i = { j != i : |y_i - y_j| <= w }
        neg_i = { j != i : |y_i - y_j| >  w and |p_i - p_j| <= w }
        s_ij  = (z_i / ||z_i||) . (z_j / ||z_j||) / tau
        S_ij  = eta_i * e * |y_i - y_j|                     (pushing weight)
        L_i   = 0                                            if pos_i or neg_i empty
        L_i   = -1/|pos_i| * sum_{j in pos_i} log(
                    exp(s_ij) / ( sum_{k in pos_i} exp(s_ik)
                                  + sum_{q in neg_i} S_iq exp(s_iq) ) )

    which is evaluated in a numerically stable log-sum-exp form (the reference
    implementation evaluates the denominator with ``exp`` directly and can
    overflow).  The returned tensor contains ``L_i`` for every anchor; anchors
    without a positive or negative pair are exactly zero.

    Args:
        features: ``[N, D]`` feature vectors (L2-normalised internally).
        labels: ``[N]`` or ``[N, L]`` continuous labels.
        predictions: ``[N]`` or ``[N, T]`` regression outputs of the model
            (averaged over the last dim when multi-output).
        label_threshold: Similarity threshold ``w`` in label space; pairs with
            label distance ``<= w`` are positive pairs.  Default 0.2.
        prediction_threshold: Similarity threshold ``w`` in prediction space;
            defaults to ``label_threshold`` (as in the reference implementation).
        temperature: Contrastive temperature ``tau``.  Default 0.07.
        push_scale: Scale ``e`` of the relative pushing weight.  Default 0.2.
        density_weights: Optional ``[N]`` per-anchor pushing power ``eta_i``
            (inverse-label-density weights, see :meth:`ConR.fit`).  Defaults to
            ones.

    Returns:
        ``[N]`` tensor of per-anchor ConR values.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")
    if prediction_threshold is None:
        prediction_threshold = label_threshold
    if label_threshold < 0.0:
        raise ValueError(f"label_threshold must be non-negative, got {label_threshold}")
    if prediction_threshold < 0.0:
        raise ValueError(f"prediction_threshold must be non-negative, got {prediction_threshold}")

    n_samples = features.shape[0]
    labels_2d = labels.reshape(n_samples, -1)
    preds = predictions.reshape(n_samples, -1)
    if labels_2d.shape[0] != n_samples or preds.shape[0] != n_samples:
        raise ValueError("features, labels and predictions must share the batch dimension")

    if labels_2d.shape[1] == 1:
        label_dist = (labels_2d - labels_2d.T).abs()
    else:
        label_dist = torch.cdist(labels_2d, labels_2d, p=2)
    pred_scalar = preds.mean(dim=-1)
    pred_dist = (pred_scalar[:, None] - pred_scalar[None, :]).abs()

    off_diag = ~torch.eye(n_samples, dtype=torch.bool, device=features.device)
    pos = (label_dist <= label_threshold) & off_diag
    neg = (label_dist > label_threshold) & (pred_dist <= prediction_threshold) & off_diag
    has_pos = pos.any(dim=1)
    has_neg = neg.any(dim=1)

    sim = (F.normalize(features, dim=-1) @ F.normalize(features, dim=-1).T) / temperature

    eta = torch.ones(n_samples, dtype=sim.dtype, device=sim.device)
    if density_weights is not None:
        if density_weights.numel() != n_samples:
            raise ValueError(
                f"density_weights must have one entry per sample ({n_samples}), "
                f"got shape {tuple(density_weights.shape)}"
            )
        eta = density_weights.reshape(n_samples).to(dtype=sim.dtype, device=sim.device)

    tiny = torch.finfo(sim.dtype).tiny
    log_push = torch.log((eta[:, None] * label_dist * push_scale).clamp_min(tiny))

    log_den_pos = torch.logsumexp(sim.masked_fill(~pos, float("-inf")), dim=1)
    log_den_neg = torch.logsumexp((sim + log_push).masked_fill(~neg, float("-inf")), dim=1)
    log_denom = torch.logaddexp(log_den_pos, log_den_neg)
    n_pos = pos.sum(dim=1).clamp_min(1)
    positive_mean = sim.masked_fill(~pos, 0.0).sum(dim=1) / n_pos
    per_anchor = log_denom - positive_mean
    per_anchor = torch.where(has_pos & has_neg, per_anchor, torch.zeros_like(per_anchor))
    return torch.where(torch.isfinite(per_anchor), per_anchor, torch.zeros_like(per_anchor))


class ConR(BaseLoss):
    """Contrastive Regularizer for deep imbalanced regression (Keramati et al., 2024).

    ConR shapes the feature space with a supervised contrastive term whose pairs
    are selected by *both* label and prediction similarity: samples with
    label-similar labels are positives, while label-distant samples whose
    predictions fell together are negatives (the model collapsed them and should
    be separated).  Negative pairs are pushed with a weight proportional to the
    per-anchor inverse-label-density weight ``eta_i`` and to the label distance
    ``|y_i - y_q|``, so minority anchors are pushed harder.  See
    :func:`conr_loss` for the exact equation.

    The regularizer is designed to be added to a task regression loss:

    ``L = alpha * L_reg + beta * L_conr``  (paper Eq. 4).  Feed it the encoder
    features ``z``, the labels ``y`` **and** the model's own predictions
    ``p = g(z)`` (the prediction is required: it defines the negative pairs).

    Args:
        label_threshold: Label-similarity threshold ``w`` for positive pairs.
            Default 0.2.
        prediction_threshold: Prediction-similarity threshold for negative
            pairs; defaults to ``label_threshold`` (reference implementation).
        temperature: Contrastive temperature ``tau``.  Default 0.07.
        push_scale: Scale ``e`` of the relative pushing weight.  Default 0.2.
        reduction: ``'none' | 'mean' | 'sum'`` over anchors.

    Shapes:
        ``y_pred``: ``[N, D]`` features.  ``target``: ``[N]`` or ``[N, L]``
        continuous labels.  ``prediction``: ``[N]`` or ``[N, T]`` model
        predictions (averaged over the last dim).  ``mask`` / ``weights``:
        ``[N]`` / ``[N, 1]``; masked samples are removed from the anchor set and
        from every positive/negative denominator (same semantics as
        :class:`~torchregress.losses.contrastive.RankNContrastLoss`), and
        ``weights`` override the fitted inverse-density pushing weights.

    Raises:
        ValueError: If ``prediction`` is not supplied.

    Example:
        >>> import torch
        >>> from torchregress.losses.prime import ConR
        >>>
        >>> conr = ConR(label_threshold=0.2).fit(y_train)   # optional density weights
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> features = model.encode(x)
        >>> pred = model.head(features)
        >>> loss = mse(pred, y) + 4.0 * conr(features, y, prediction=pred)
        >>> loss.backward()
        >>> optimizer.step()

    Notes:
        Implementation notes, including the deviations from the reference
        implementation and the parameters that are not re-optimised:

        * Pair selection is the reference code's thresholding of the **absolute**
          label difference (L2 for multi-dimensional labels) together with the
          absolute prediction difference; multi-output predictions are averaged
          over the output dimension exactly as in the reference code.
        * The positive denominator sums only over positive pairs and is divided
          by the true number of positive pairs (paper Eq. 1).  The reference
          implementation additionally adds ``exp(0) = 1`` for every non-positive
          entry (and counts the self-pair in the denominator); that masking
          artefact is not reproduced.
        * ``eta_i`` comes from :meth:`fit` (inverse label-frequency bin weights,
          normalised to mean one over the training samples).  The paper uses
          ``eta_j = c * w_j`` with a tuned scale ``c``; the scale is absorbed by
          ``push_scale`` here.
        * The paper applies ConR to augmented pairs; augmentation is the caller's
          responsibility (feed both views of a sample to the same batch).
        * ConR is **not** registered with ``register_regression_loss`` because
          ``forward`` requires the model prediction, which the generic loss
          factories do not supply.

    Reference:
        Keramati, M., Meng, Y., & Evans, B.  "ConR: Contrastive Regularizer for
        Deep Imbalanced Regression."  ICLR 2024.
        https://arxiv.org/abs/2309.06651
    """

    bin_edges: Tensor
    density_bin_weights: Tensor

    def __init__(
        self,
        *,
        label_threshold: float = 0.2,
        prediction_threshold: Optional[float] = None,
        temperature: float = 0.07,
        push_scale: float = 0.2,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.label_threshold = float(label_threshold)
        self.prediction_threshold = (
            None if prediction_threshold is None else float(prediction_threshold)
        )
        if self.label_threshold < 0.0:
            raise ValueError(f"label_threshold must be non-negative, got {label_threshold}")
        if self.prediction_threshold is not None and self.prediction_threshold < 0.0:
            raise ValueError(
                f"prediction_threshold must be non-negative, got {prediction_threshold}"
            )
        self.temperature = float(validate_positive(temperature, "temperature"))
        self.push_scale = float(validate_positive(push_scale, "push_scale"))

        self.register_buffer("bin_edges", torch.zeros(0))
        self.register_buffer("density_bin_weights", torch.zeros(0))

    @property
    def is_fitted(self) -> bool:
        """Whether inverse-density pushing weights have been fitted."""
        return self.bin_edges.numel() > 1

    @property
    def bin_weights(self) -> Tensor:
        """Per-bin inverse-frequency weights ``eta`` (raises before :meth:`fit`)."""
        if not self.is_fitted:
            raise RuntimeError("Call fit(train_targets) before using ConR.bin_weights")
        return self.density_bin_weights

    def fit(self, train_targets: Tensor, n_bins: int = 100) -> ConR:
        """Estimate per-sample inverse-label-density pushing powers ``eta``.

        Bins the training targets into ``n_bins`` equal-width bins and assigns
        each bin the inverse of its (smoothed) count, normalised so the mean
        weight over the training samples is one.  Bins are built on the training
        target range; labels outside it are clamped to the edge bins.

        Args:
            train_targets: All training labels, ``[N]`` or ``[N, L]``
                (multi-output labels are binned by their mean coordinate).
            n_bins: Number of histogram bins.  Default 100.

        Returns:
            ``self``, so calls can be chained.
        """
        if int(n_bins) < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        labels = _labels_for_binning(train_targets.detach().float())
        finite = torch.isfinite(labels)
        if not bool(finite.any()):
            raise ValueError("fit() needs at least one finite training target")
        labels = labels[finite]

        y_min, y_max = float(labels.min()), float(labels.max())
        if not y_max > y_min:
            y_max = y_min + 1.0
        edges = torch.linspace(y_min, y_max, int(n_bins) + 1)
        assignments = _bin_indices(labels.cpu(), edges)
        counts = torch.bincount(assignments, minlength=int(n_bins)).to(labels.dtype)
        weights = _inverse_frequency_weights(counts)
        self.bin_edges = edges.to(dtype=torch.float32)
        self.density_bin_weights = weights.to(dtype=torch.float32)
        return self

    def _density_weights_for(self, labels: Tensor) -> Tensor:
        """Look up ``eta`` for arbitrary scalar labels from the fitted bins."""
        if not self.is_fitted:
            return torch.ones(labels.shape[0], dtype=labels.dtype, device=labels.device)
        edges = self.bin_edges.to(device=labels.device, dtype=labels.dtype)
        weights = self.density_bin_weights.to(device=labels.device, dtype=labels.dtype)
        return weights[_bin_indices(labels, edges)]

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        prediction: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the ConR regularizer.

        Args:
            y_pred: Feature vectors ``[N, D]`` (encoder output).
            target: Continuous labels ``[N]`` or ``[N, L]``.
            prediction: Model regression outputs ``[N]`` or ``[N, T]``
                (required -- they define the negative pairs).
            mask: Optional per-sample validity mask (``[N]`` / ``[N, 1]``).
                Masked samples are removed from the anchor set and from all
                positive/negative denominators.
            weights: Optional per-sample weights; when given they **override**
                the fitted inverse-density pushing weights ``eta`` (and are also
                used as the sample weights of the reduction, per the shared
                contract).
            **kwargs: Accepted for API consistency and ignored.

        Returns:
            Scalar loss for ``reduction='mean'`` / ``'sum'``, or the ``[N]``
            per-anchor tensor for ``reduction='none'``.

        Raises:
            ValueError: If ``prediction`` is missing or the shapes disagree.
        """
        if prediction is None:
            raise ValueError(
                "ConR requires the model predictions: call "
                "loss(features, target, prediction=model(head(features)))."
            )
        if y_pred.dim() != 2:
            raise ValueError(f"y_pred must be [N, D] feature vectors, got {tuple(y_pred.shape)}")
        n_samples = y_pred.shape[0]
        if target.shape[0] != n_samples or prediction.shape[0] != n_samples:
            raise ValueError(
                "y_pred, target and prediction must share the batch dimension "
                f"(got {n_samples}, {target.shape[0]}, {prediction.shape[0]})"
            )
        if not y_pred.is_floating_point():
            raise ValueError("y_pred must be a floating-point tensor of features")

        labels = target.reshape(n_samples, -1).detach()
        preds = prediction.reshape(n_samples, -1)
        mask_flat: Optional[Tensor] = None
        if mask is not None:
            if mask.numel() != n_samples:
                raise ValueError(
                    f"mask must have one entry per sample ({n_samples}), "
                    f"got shape {tuple(mask.shape)}"
                )
            mask_flat = mask.reshape(n_samples).to(device=y_pred.device, dtype=torch.bool)
        if weights is not None and weights.numel() != n_samples:
            raise ValueError(
                f"weights must have one entry per sample ({n_samples}), "
                f"got shape {tuple(weights.shape)}"
            )

        # ``weights`` override the fitted inverse-density pushing powers.
        if weights is not None:
            eta = weights.reshape(n_samples).detach().to(device=y_pred.device)
        else:
            eta = self._density_weights_for(_labels_for_binning(labels)).to(device=y_pred.device)

        kwargs_loss: dict[str, Any] = dict(
            label_threshold=self.label_threshold,
            prediction_threshold=self.prediction_threshold,
            temperature=self.temperature,
            push_scale=self.push_scale,
        )
        if mask_flat is not None:
            keep = mask_flat
            keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
            per_anchor = y_pred.new_zeros(n_samples)
            if keep_idx.numel() >= 2:
                per_anchor[keep_idx] = conr_loss(
                    y_pred[keep_idx],
                    labels[keep_idx],
                    preds[keep_idx],
                    density_weights=eta[keep_idx],
                    **kwargs_loss,
                )
            reduced_weights = weights.reshape(n_samples) if weights is not None else None
            return self._reduce(per_anchor, keep, reduced_weights)

        per_anchor = conr_loss(
            y_pred,
            labels,
            preds,
            density_weights=eta,
            **kwargs_loss,
        )
        reduced_weights = weights.reshape(n_samples) if weights is not None else None
        return self._reduce(per_anchor, None, reduced_weights)
