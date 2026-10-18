"""
RankSim: ranking-similarity regularization for deep imbalanced regression.

RankSim (Gong, Mori & Tung, ICML 2022) is a *representation-level* regularizer
for regression with continuous, ordered targets.  It encodes the inductive bias
that samples that are close in label space should also be close in feature
space: for every anchor it matches the ranking of the other samples by label
distance to their ranking by feature similarity.  The term is added to a task
regression loss and is complementary to output-space reweighting (LDS, FDS,
density weighting) and to other representation losses (Rank-N-Contrast, ConR,
PRIME).

This module implements the regularizer of the paper's Eq. (4) together with the
blackbox interpolation gradient of Eq. (6)-(7) used to differentiate through the
ranking operation.  It is distinct from
:class:`~torchregress.losses.contrastive.RankNContrastLoss`, which uses a
different (contrastive, N-pair) objective; see ``docs/losses/rank_sim.md`` for a
side-by-side comparison.

Reference:
    Gong, Y., Mori, G., & Tung, F.  "RankSim: Ranking Similarity Regularization
    for Deep Imbalanced Regression."  ICML 2022.
    https://arxiv.org/abs/2205.15236
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..utils.validation import validate_positive, validate_weights
from .base import BaseLoss
from .loss_registry import register_regression_loss

__all__ = ["RankSimLoss", "rank_sim_regularizer"]

_SIMILARITIES = ("cosine", "l2")
_DUPLICATE_POLICIES = ("mean", "first", "keep")


# ── ranking ──────────────────────────────────────────────────────────────────


@torch.no_grad()
def _rank_normalised(scores: Tensor) -> Tensor:
    """Normalised competition rank of every row of ``scores``.

    Implements the paper's Eq. (1): ``rk(a)_i = 1 + |{j : a_j > a_i}|``, so the
    largest score receives rank 1, equal scores receive the same rank, and the
    ranks are divided by the row length ``M`` (as in the reference
    implementation).  The result lies in ``(0, 1]`` and is invariant to the
    order of the columns.  Ranking is a piecewise-constant (non-differentiable)
    operation; gradients are supplied by :class:`_NormalisedRank`.
    """
    n = scores.shape[-1]
    if n == 0:
        return scores
    sorted_scores, order = torch.sort(scores, dim=-1, descending=True, stable=True)
    # ``starts[..., p]`` marks the first element of a run of equal scores.  The
    # rank of a run is one plus the number of elements before it: in descending
    # order every preceding element is strictly greater.
    starts = torch.ones_like(sorted_scores, dtype=torch.bool)
    if n > 1:
        starts[..., 1:] = sorted_scores[..., 1:] != sorted_scores[..., :-1]
    positions = torch.arange(n, device=scores.device, dtype=torch.long).expand_as(sorted_scores)
    run_start = torch.cummax(torch.where(starts, positions, torch.zeros_like(positions)), dim=-1)
    ranks_sorted = (run_start.values + 1).to(dtype=scores.dtype)
    ranks = torch.empty_like(sorted_scores)
    ranks.scatter_(-1, order, ranks_sorted)
    return ranks / n


class _NormalisedRank(torch.autograd.Function):
    """Ranking with the blackbox-interpolation gradient of RankSim Eq. (6)-(7).

    The forward value is the ordinary normalised rank (piecewise constant).  The
    backward replaces the (almost everywhere zero) true derivative with the
    gradient of the continuous interpolation of Vlastelica et al. (2020):

    .. code-block:: text

        dL/ds = -(rk(s) - rk(s + lambda * dL/drk)) / lambda

    where ``dL/drk`` is the incoming gradient (the MSE derivative).  Ranking the
    perturbed scores costs one extra sort per backward pass.
    """

    @staticmethod
    def forward(ctx: Any, scores: Tensor, interpolation_strength: float) -> Tensor:
        ctx.interpolation_strength = float(interpolation_strength)
        ranks = _rank_normalised(scores)
        ctx.save_for_backward(scores, ranks)
        return ranks

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[Tensor, None]:
        grad_output: Tensor = grad_outputs[0]
        scores, ranks = ctx.saved_tensors
        lam = ctx.interpolation_strength
        perturbed = scores + lam * grad_output
        rank_perturbed = _rank_normalised(perturbed)
        gradient = -(ranks - rank_perturbed) / (lam + 1e-8)
        return gradient, None


# ── shared helpers ───────────────────────────────────────────────────────────


def _validate_options(
    interpolation_strength: float, similarity: str, duplicate_policy: str
) -> None:
    if similarity not in _SIMILARITIES:
        raise ValueError(f"similarity must be one of {_SIMILARITIES}, got {similarity!r}")
    if duplicate_policy not in _DUPLICATE_POLICIES:
        raise ValueError(
            f"duplicate_policy must be one of {_DUPLICATE_POLICIES}, got {duplicate_policy!r}"
        )
    validate_positive(float(interpolation_strength), "interpolation_strength")


def _feature_similarity(features: Tensor, similarity: str) -> Tensor:
    """Pairwise feature similarity matrix ``[M, M]``."""
    if similarity == "cosine":
        normalised = F.normalize(features, dim=-1, eps=1e-12)
        return normalised @ normalised.transpose(-1, -2)
    return -torch.cdist(features, features, p=2)


def _scalar_targets(target: Tensor, n_samples: int, reference: Tensor) -> Tensor:
    """Return one scalar target per sample from ``[N]`` or ``[N, 1]`` input."""
    scalar_shape = target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1)
    if not scalar_shape or target.shape[0] != n_samples:
        raise ValueError(
            f"target must be scalar labels with shape [N] or [N, 1] where N = "
            f"{n_samples}; got {tuple(target.shape)}"
        )
    return target.reshape(n_samples).detach().to(device=reference.device, dtype=reference.dtype)


def _deduplicate(
    features: Tensor, labels: Tensor, duplicate_policy: str
) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Reduce repeated labels to one representative per distinct target.

    Returns ``(rep_features, rep_labels, rep_positions, inverse)`` where
    ``rep_positions[k]`` is the batch index of the ``k``-th representative and
    ``inverse`` maps every sample to its representative (``None`` for
    ``duplicate_policy='keep'``).
    """
    n = features.shape[0]
    if duplicate_policy == "keep" or n < 2:
        positions = torch.arange(n, device=features.device)
        return features, labels, positions, None

    unique_labels, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    n_unique = unique_labels.numel()
    positions = torch.full((n_unique,), n, dtype=torch.long, device=features.device)
    sample_index = torch.arange(n, device=features.device)
    # First occurrence of every distinct target: deterministic and independent
    # of the random tie sampling used by the reference implementation.
    positions.scatter_reduce_(0, inverse, sample_index, reduce="amin")

    if duplicate_policy == "mean":
        counts = torch.bincount(inverse, minlength=n_unique).to(features.dtype)
        summed = features.new_zeros(n_unique, features.shape[1])
        summed.index_add_(0, inverse, features)
        rep_features = summed / counts.unsqueeze(-1)
    else:  # "first"
        rep_features = features[positions]
    return rep_features, unique_labels, positions, inverse


def _rank_sim_terms(
    features: Tensor,
    labels: Tensor,
    *,
    interpolation_strength: float,
    similarity: str,
    duplicate_policy: str,
) -> tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
    """RankSim per-anchor terms over one batch (without mask/weights).

    Returns ``(terms, keep, inverse, positions)`` where ``terms`` is ``[N]``
    with zeros for samples that are not representatives, ``keep`` is the
    ``[N]`` boolean mask of retained anchor representatives, and ``inverse`` /
    ``positions`` are the deduplication mapping (``inverse=None`` for
    ``duplicate_policy='keep'``).
    """
    n = features.shape[0]
    rep_features, rep_labels, positions, inverse = _deduplicate(features, labels, duplicate_policy)
    terms = features.new_zeros(n)
    keep = torch.zeros(n, dtype=torch.bool, device=features.device)
    keep[positions] = True
    if positions.numel() < 2:
        # A single distinct label has no ordering information; the loss is zero.
        return terms, keep, inverse, positions

    feature_ranks = _NormalisedRank.apply(
        _feature_similarity(rep_features, similarity), float(interpolation_strength)
    )
    label_distance = (rep_labels[:, None] - rep_labels[None, :]).abs()
    label_ranks = _rank_normalised(-label_distance)
    per_rep = ((feature_ranks - label_ranks) ** 2).mean(dim=-1)
    terms[positions] = per_rep
    return terms, keep, inverse, positions


def _check_finite(features: Tensor, labels: Tensor) -> None:
    if not torch.isfinite(features).all():
        raise ValueError("features must be finite (mask out samples with missing features)")
    if not torch.isfinite(labels).all():
        raise ValueError("target must be finite (mask out samples with missing targets)")


# ── functional form ──────────────────────────────────────────────────────────


def rank_sim_regularizer(
    features: Tensor,
    target: Tensor,
    *,
    interpolation_strength: float = 2.0,
    similarity: str = "cosine",
    duplicate_policy: str = "mean",
    reduction: str = "mean",
) -> Tensor:
    """Unweighted RankSim regularizer for one batch of features and labels.

    Implements Eq. (4) of Gong et al. (2022) on a batch of feature vectors and
    scalar targets, including the paper's reduction of repeated labels to one
    representative per distinct target:

    .. code-block:: text

        S^y_ij = -|y_i - y_j|
        S^z_ij = cos(z_i, z_j)                 (or -||z_i - z_j||_2)
        rk(a)_i = 1 + |{j : a_j > a_i}|        (Eq. 1)
        L_i = (1/M) sum_j ( rk(S^z_i)_j/M - rk(S^y_i)_j/M )^2
        L = reduce_i(L_i)                      (Eq. 4 uses the sum over i)

    Args:
        features: ``[N, D]`` feature vectors (the encoder output, not the
            regression head output).
        target: Scalar labels ``[N]`` or ``[N, 1]``.
        interpolation_strength: Interpolation strength ``lambda > 0`` of the
            blackbox ranking gradient (paper Eq. 6-7).  The paper uses
            ``lambda = 2``.
        similarity: ``"cosine"`` (paper default) or ``"l2"`` (negative Euclidean
            distance).
        duplicate_policy: How repeated targets are reduced to one anchor each:

            * ``"mean"`` (default): one representative per distinct target whose
              feature is the mean feature of its group.  Deterministic and
              permutation-invariant; replaces the reference implementation's
              random choice of one sample per label.
            * ``"first"``: one representative per distinct target, the first
              occurrence in the batch (deterministic version of the reference
              implementation's random sample).
            * ``"keep"``: keep every sample as an anchor; equal label distances
              receive equal (competition) ranks.

        reduction: ``'mean'`` | ``'sum'`` | ``'none'``.  ``'none'`` returns the
            per-representative terms as a ``[M]`` tensor (``M <= N`` distinct
            targets), ordered by increasing target under a deduplicating policy
            and in batch order with ``duplicate_policy='keep'``; use
            :class:`RankSimLoss` for a per-batch ``[N]`` layout.

    Returns:
        The regularizer value: a scalar for ``'mean'`` / ``'sum'``, or the
        ``[M]`` per-representative terms for ``'none'``.  This is the
        *unweighted* ``L_RankSim``; the paper adds ``gamma * L_RankSim`` to the
        task loss (use ``gamma=1.0`` with this function if you scale at the call
        site).

    Raises:
        ValueError: For malformed shapes, non-finite valid inputs, or invalid
            constructor-style arguments.

    Example:
        >>> import torch
        >>> from torchregress.losses.rank_sim import rank_sim_regularizer
        >>>
        >>> features = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
        >>> target = torch.tensor([0.0, 1.0, 3.0])  # distinct pairwise distances
        >>> rank_sim_regularizer(features, target)  # ordered features -> 0
        tensor(0.)

    Note:
        The batch cost is ``O(N^2)`` memory for the similarity and rank
        matrices and ``O(N^2 log N)`` time for the per-row sorting (plus one
        extra sort per backward pass).  The regularizer is batch-composition
        dependent by construction: ranks are relative to the other samples in
        the batch.
    """
    if features.dim() != 2:
        raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")
    if not features.is_floating_point():
        raise ValueError("features must be a floating-point tensor")
    if features.shape[1] < 1:
        raise ValueError("features must have at least one feature dimension")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
    _validate_options(interpolation_strength, similarity, duplicate_policy)

    n = features.shape[0]
    labels = _scalar_targets(target, n, features)
    _check_finite(features, labels)
    terms, keep, _, _ = _rank_sim_terms(
        features,
        labels,
        interpolation_strength=interpolation_strength,
        similarity=similarity,
        duplicate_policy=duplicate_policy,
    )
    retained = terms[keep]
    if reduction == "none":
        return retained
    if reduction == "sum":
        return retained.sum()
    return retained.mean()


# ── module ───────────────────────────────────────────────────────────────────


@register_regression_loss("rank_sim")
class RankSimLoss(BaseLoss):
    """Ranking-similarity regularizer for imbalanced regression (RankSim).

    RankSim (Gong et al., ICML 2022) regularises the *feature space* so that,
    for every anchor, the ordering of the other samples by feature similarity
    matches their ordering by label distance.  Feed it the encoder features
    (``[N, D]``), **not** the regression head output, together with the scalar
    targets:

    .. code-block:: text

        L_total = L_task + gamma * L_RankSim(features, target)

    The paper's regularizer, with one representative per distinct target and
    ``M`` retained anchors, is

    .. code-block:: text

        S^y_ij = -|y_i - y_j|,     S^z_ij = cos(z_i, z_j)
        rk(a)_i = 1 + |{j : a_j > a_i}|                      (Eq. 1)
        L_RankSim = sum_i (1/M) sum_j ( rk(S^z_i)_j/M - rk(S^y_i)_j/M )^2
                  = sum_i L_i                                 (Eq. 4)

    i.e. a sum over anchors of the mean-squared difference between the
    normalised label ranks and the normalised feature ranks.  Ranking is not
    differentiable, so the backward pass uses the continuous interpolation of
    Vlastelica et al. (2020), controlled by ``interpolation_strength`` ``lambda``:

    .. code-block:: text

        dL/dS^z = -(rk(S^z) - rk(S^z + lambda * dL/drk)) / lambda      (Eq. 6-7)

    ``gamma`` is folded into the returned value, so the composition is simply
    ``task_loss + RankSimLoss(...)(features, target)``.  If you prefer to keep
    the balancing weight at the call site, construct the module with
    ``gamma=1.0`` and write ``task_loss + gamma * loss_fn(features, target)``;
    the functional :func:`rank_sim_regularizer` returns the unweighted term.
    Do not apply ``gamma`` twice.

    Args:
        gamma: Balancing weight ``gamma >= 0`` of the regularizer in the total
            objective.  Default ``1.0``.  The paper tunes ``gamma`` per
            benchmark (e.g. 100 for IMDB-WIKI-DIR / AgeDB-DIR with batch size 256
            and a sum over the retained anchors; 3e-4 for STS-B-DIR).  Because
            this class averages over anchors under ``reduction='mean'``,
            matching a paper run that sums over ``M`` anchors needs
            ``gamma_mean = gamma_paper * M``; use ``reduction='sum'`` to
            reproduce Eq. (4) directly.
        interpolation_strength: Interpolation strength ``lambda > 0`` of the
            blackbox ranking gradient.  Default ``2.0`` (paper value).  Small
            values make the surrogate gradient sparse (it changes only when the
            perturbation crosses a rank boundary) and large values smooth it at
            the cost of fidelity.
        similarity: Feature similarity ``sigma^z``: ``"cosine"`` (paper default)
            or ``"l2"`` (negative Euclidean distance).
        duplicate_policy: Reduction of repeated targets to one anchor each:
            ``"mean"`` (default; group-mean feature, deterministic and
            permutation-invariant), ``"first"`` (first occurrence, deterministic
            version of the reference implementation's random sample), or
            ``"keep"`` (no reduction; equal distances get equal ranks).
        reduction: ``'mean'`` | ``'sum'`` | ``'none'`` over the retained anchors
            (Eq. 4 is the ``'sum'`` reduction).  ``'none'`` returns a ``[N]``
            tensor with zeros for masked samples and for non-representative
            duplicates; because those duplicates are not anchors, its plain mean
            can differ from the ``'mean'`` reduction when labels repeat.

    Shapes:
        ``y_pred``: ``[N, D]`` feature vectors.  ``target``: scalar labels
        ``[N]`` or ``[N, 1]``.  ``mask``: ``[N]`` or ``[N, 1]`` boolean/0-1;
        masked samples are removed from the anchor set and from every ranking.
        ``weights``: ``[N]`` or ``[N, 1]`` non-negative sample weights following
        the shared :meth:`BaseLoss._reduce` contract; under a deduplicating
        policy a representative inherits the mean weight of its label group.

    Example:
        >>> import torch
        >>> from torchregress.losses.rank_sim import RankSimLoss
        >>>
        >>> rank_sim = RankSimLoss(gamma=0.5, interpolation_strength=2.0)
        >>> loss = torch.nn.MSELoss()
        >>> features = encoder(x)                    # [B, D]
        >>> pred = head(features).squeeze(-1)        # [B]
        >>> total = loss(pred, y) + rank_sim(features, y)
        >>> total.backward()

    Raises:
        ValueError: For malformed shapes, non-finite valid inputs, invalid
            hyperparameters, or non-scalar targets.

    Notes:
        Implementation choices and deviations from the reference code:

        * **Ties.**  Eq. (1) assigns equal ranks to equal scores (competition
          ranking), whereas the reference ``rank_normalised`` breaks ties
          arbitrarily through ``argsort``.  Competition ranking makes the value
          deterministic and invariant to the order of equal-label samples.
        * **Duplicate labels.**  The paper samples one instance per distinct
          label at random; randomness would violate this library's deterministic
          contract.  ``duplicate_policy="mean"`` (default) uses the group-mean
          feature, which is deterministic and permutation-invariant;
          ``"first"`` reproduces the single-sample choice deterministically from
          the batch order; ``"keep"`` disables the reduction.
        * **Normalisation.**  Ranks are divided by the number of retained
          anchors ``M`` and, under ``reduction='mean'``, the per-anchor terms are
          averaged, so the default value is on the same scale as an average task
          loss.  Eq. (4) is the ``'sum'`` reduction; the two differ by a factor
          of ``M`` (see ``gamma`` above).
        * **Scalar targets only.**  The paper defines ``sigma^y`` for continuous
          scalar labels; multi-output targets are rejected rather than
          silently averaged.
        * **Prediction-free.**  Unlike ConR, RankSim needs only features and
          labels (no model predictions), and unlike PRIME it adds no parameters.
          Like Rank-N-Contrast it is batch-contrastive: its cost and its value
          depend on the composition of the batch, and a batch with a single
          distinct label carries no signal.

    Reference:
        Gong, Y., Mori, G., & Tung, F.  "RankSim: Ranking Similarity
        Regularization for Deep Imbalanced Regression."  ICML 2022.
        https://arxiv.org/abs/2205.15236
    """

    def __init__(
        self,
        gamma: float = 1.0,
        *,
        interpolation_strength: float = 2.0,
        similarity: str = "cosine",
        duplicate_policy: str = "mean",
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)
        self.gamma = float(validate_positive(gamma, "gamma", allow_zero=True))
        _validate_options(interpolation_strength, similarity, duplicate_policy)
        self.interpolation_strength = float(interpolation_strength)
        self.similarity = similarity
        self.duplicate_policy = duplicate_policy

    def forward(
        self,
        y_pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the weighted RankSim regularizer.

        Args:
            y_pred: Feature vectors ``[N, D]`` (encoder output, not predictions).
            target: Scalar labels ``[N]`` or ``[N, 1]``.
            mask: Optional per-sample validity mask (``[N]`` / ``[N, 1]``);
                masked samples are removed from anchors and rankings and receive
                a zero per-anchor value.
            weights: Optional non-negative per-sample weights following the
                shared reduction contract (see class docstring).
            **kwargs: Accepted for API consistency and ignored.

        Returns:
            ``gamma * L_RankSim`` reduced according to ``self.reduction``:
            a scalar for ``'mean'`` / ``'sum'``, or a ``[N]`` tensor for
            ``'none'`` (zeros at masked and non-representative entries).

        Raises:
            ValueError: For malformed shapes, non-finite valid inputs, or
                non-scalar targets.
        """
        if y_pred.dim() != 2:
            raise ValueError(f"y_pred must be [N, D] feature vectors, got {tuple(y_pred.shape)}")
        if not y_pred.is_floating_point():
            raise ValueError("y_pred must be a floating-point tensor of features")
        if y_pred.shape[1] < 1:
            raise ValueError("y_pred must have at least one feature dimension")
        n_samples = y_pred.shape[0]
        labels = _scalar_targets(target, n_samples, y_pred)

        valid: Optional[Tensor] = None
        if mask is not None:
            if mask.numel() != n_samples:
                raise ValueError(
                    f"mask must have one entry per sample ({n_samples}), "
                    f"got shape {tuple(mask.shape)}"
                )
            valid = mask.reshape(n_samples).to(device=y_pred.device, dtype=torch.bool)
        weight_vector: Optional[Tensor] = None
        if weights is not None:
            checked = validate_weights(weights, n_samples)
            assert checked is not None  # validate_weights only returns None for None input
            weight_vector = checked.reshape(n_samples).to(device=y_pred.device, dtype=y_pred.dtype)

        if valid is not None and int(valid.sum()) < 2:
            per_anchor = y_pred.new_zeros(n_samples)
            keep = valid
            used_weights = weight_vector
            return self.gamma * self._reduce(per_anchor, keep, used_weights)

        if valid is None:
            features = y_pred
            valid_labels = labels
            valid_weight = weight_vector
        else:
            features = y_pred[valid]
            valid_labels = labels[valid]
            valid_weight = None if weight_vector is None else weight_vector[valid]

        _check_finite(features, valid_labels)
        terms, keep_sub, inverse, positions = _rank_sim_terms(
            features,
            valid_labels,
            interpolation_strength=self.interpolation_strength,
            similarity=self.similarity,
            duplicate_policy=self.duplicate_policy,
        )

        per_anchor = y_pred.new_zeros(n_samples)
        keep = torch.zeros(n_samples, dtype=torch.bool, device=y_pred.device)
        if valid is None:
            per_anchor = terms
            keep = keep_sub
        else:
            per_anchor[valid] = terms
            keep[valid] = keep_sub

        used_weights = weight_vector
        if valid_weight is not None:
            if inverse is not None:
                # A representative inherits the mean weight of its label group.
                n_unique = positions.numel()
                counts = torch.bincount(inverse, minlength=n_unique).to(valid_weight.dtype)
                sums = valid_weight.new_zeros(n_unique)
                sums.index_add_(0, inverse, valid_weight)
                group_weight = sums / counts
                valid_weight = valid_weight.new_zeros(valid_weight.shape)
                valid_weight[positions] = group_weight
            if valid is None:
                used_weights = valid_weight
            else:
                used_weights = y_pred.new_zeros(n_samples)
                used_weights[valid] = valid_weight

        return self.gamma * self._reduce(per_anchor, keep, used_weights)
