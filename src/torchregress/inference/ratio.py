r"""Posterior reweighting from estimated likelihood ratios.

Numerical recipes that turn candidate likelihood ratios -- the output of neural
ratio estimation (NRE), contrastive learning, or a grid likelihood scan -- into
normalized posterior weights, effective-sample-size diagnostics, and MLE/MAP
point estimates.

Let :math:`x` be an observation and :math:`\theta` a parameter.  A ratio
estimator models

.. math::

    r(x, \theta) = \frac{p(x \mid \theta)}{p_{\mathrm{ref}}(x)},
    \qquad
    \log r(x, \theta) = \log p(x \mid \theta) - \log p_{\mathrm{ref}}(x),

where :math:`p_{\mathrm{ref}}(x)` is a reference density over the data that does
not depend on :math:`\theta` (for example the marginal data distribution used to
build the training classes).  For a finite candidate bank
:math:`\{\theta_k\}_{k=1}^{K}` drawn from a proposal :math:`q(\theta)`,
self-normalized importance sampling approximates the posterior as

.. math::

    p(\theta \mid x) \;\propto\; \frac{r(x, \theta)\, p(\theta)}{q(\theta)},

so the candidate log weights are
:math:`\log r(x, \theta_k) + \log p(\theta_k) - \log q(\theta_k)`.
:func:`posterior_log_weights` normalizes them with a ``log_softmax``; when the
candidates are drawn from the prior, :math:`q = p` and the two corrections
cancel, leaving the log ratio alone.

Conventions
-----------
* ``log_ratio`` is always :math:`\log p(x \mid \theta) - \log p_{\mathrm{ref}}(x)`,
  i.e. a ratio relative to the reference/candidate distribution -- not a
  posterior density.
* ``log_prior`` is the *target* prior :math:`\log p(\theta)`.
* ``log_reference`` is the proposal :math:`\log q(\theta)` actually used to draw
  the candidates.
* ``[K]`` inputs are a single candidate bank, ``[N, K]`` inputs are ``N``
  independent banks, and ``dim`` selects the candidate axis.  Leading and
  trailing batch dimensions are supported as long as ``dim`` is valid.
* Floating dtypes are preserved (mixed dtypes are promoted) and autograd is
  preserved.  All inputs must be finite.

References
----------
.. [1] Papamakarios, G., Sterratt, D., & Murray, I. (2019). Sequential Neural
   Likelihood Inference. *AISTATS*. https://arxiv.org/abs/1805.07226
.. [2] Hermans, J., Begy, V., & Louppe, G. (2020). Likelihood-free MCMC with
   Amortized Approximate Ratio Estimators. *ICML*. https://arxiv.org/abs/1903.04057
.. [3] Cranmer, K., Brehmer, J., & Louppe, G. (2020). The frontier of
   simulation-based inference. *PNAS*, 117(48), 30055-30065.
   https://arxiv.org/abs/1911.01429
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "effective_sample_size",
    "grid_log_posterior",
    "maximum_likelihood_estimate",
    "posterior_log_weights",
]

#: Array-like inputs: tensors or (nested) float sequences.
_TensorLike = Tensor | Sequence[float] | Sequence[Sequence[float]] | float
#: Inputs that can be a single tensor or a Python float; both are converted to
#: tensors, so a float is a scalar correction.
_CorrectionLike = Tensor | Sequence[float] | float


def _as_float_tensor(name: str, value: _TensorLike) -> Tensor:
    """Convert an array-like input to a floating-point tensor.

    Floating dtypes (and devices) are preserved; integer and boolean inputs are
    cast to the default floating dtype so that log-space arithmetic is valid.
    Nothing is detached: the returned tensor stays connected to the autograd
    graph of the input.
    """
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    return tensor


def _check_finite(name: str, tensor: Tensor) -> None:
    """Raise ``ValueError`` unless every entry of ``tensor`` is finite."""
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite; found NaN or infinite entries")


def _resolve_dim(dim: int, ndim: int, name: str) -> int:
    """Normalize a possibly negative ``dim`` against ``ndim`` dimensions."""
    if not -ndim <= dim < ndim:
        raise ValueError(
            f"dim must be in [{-ndim}, {ndim - 1}] for {name} with {ndim} dimension(s), got {dim}"
        )
    return dim % ndim


def _prepare_correction(
    name: str,
    value: _CorrectionLike | None,
    shape: torch.Size,
) -> Tensor | None:
    """Validate an optional prior/proposal correction against a score shape.

    Corrections follow PyTorch broadcasting: they may be scalars, ``[K]``
    vectors, or full ``[N, K]`` tensors, but must not expand the score shape
    (a correction that broadcasts to a *larger* shape is a user error, because
    it would silently change the candidate axis).
    """
    if value is None:
        return None
    correction = _as_float_tensor(name, value)
    _check_finite(name, correction)
    try:
        broadcast = tuple(torch.broadcast_shapes(tuple(shape), tuple(correction.shape)))
    except RuntimeError:
        broadcast = None
    if broadcast is None or broadcast != tuple(shape):
        raise ValueError(
            f"{name} with shape {tuple(correction.shape)} is not broadcastable to "
            f"{tuple(shape)} without expanding it; corrections must act on the "
            "candidate axis (e.g. a scalar, [K], or [N, K])"
        )
    return correction


def _promoted_dtype(*tensors: Tensor | None) -> torch.dtype:
    """Promoted floating dtype of the non-``None`` tensors."""
    dtype: torch.dtype | None = None
    for tensor in tensors:
        if tensor is None:
            continue
        dtype = tensor.dtype if dtype is None else torch.promote_types(dtype, tensor.dtype)
    assert dtype is not None  # callers always pass the score tensor
    return dtype


def posterior_log_weights(
    log_ratio: _TensorLike,
    *,
    log_prior: _CorrectionLike | None = None,
    log_reference: _CorrectionLike | None = None,
    dim: int = -1,
) -> Tensor:
    r"""Normalize candidate log ratios into log posterior weights.

    The candidate log weights are

    .. math::

        \log \tilde{w}_k = \log r(x, \theta_k) + \log p(\theta_k)
                           - \log q(\theta_k),

    and they are normalized so that the posterior weights sum to one:

    .. math::

        \log w_k = \log \tilde{w}_k
                   - \operatorname{logsumexp}_j \log \tilde{w}_j,

    which is exactly ``log_softmax(log_ratio + log_prior - log_reference,
    dim=dim)``.  All three terms may be batched over independent observations.

    Args:
        log_ratio: Estimated log ratios
            :math:`\log r(x, \theta_k) = \log p(x \mid \theta_k)
            - \log p_{\mathrm{ref}}(x)`, with the :math:`K` candidates along
            ``dim``.  A single bank has shape ``[K]``, a batch ``[N, K]``; any
            number of extra batch dimensions is allowed.  Values must be finite.
        log_prior: Target prior :math:`\log p(\theta_k)`, broadcastable to
            ``log_ratio`` without expanding it: a scalar, ``[K]``, or ``[N, K]``
            tensor (a Python float is treated as a scalar).  Defaults to
            ``None``, i.e. a flat target prior.
        log_reference: Proposal density :math:`\log q(\theta_k)` of the
            distribution from which the candidates were drawn, with the same
            broadcasting rules as ``log_prior``.  Omit it when candidates were
            sampled from the prior: then :math:`q = p` and the correction
            cancels against ``log_prior``.  Defaults to ``None``.
        dim: Candidate axis of ``log_ratio``.  Defaults to ``-1``.

    Returns:
        Log posterior weights with the same shape as ``log_ratio``.  ``exp`` of
        the result sums to one along ``dim`` (up to floating-point rounding) and
        every entry is finite.

    Raises:
        ValueError: If ``log_ratio`` is a scalar or empty along ``dim``, if any
            input contains NaN or infinity, if a correction is not broadcastable
            to ``log_ratio`` without expanding it, or if ``dim`` is out of range.

    Warning:
        The weights are a *self-normalized importance-sampling* approximation
        built from a finite candidate bank.  They are meaningful only when the
        candidates cover the posterior mass: a bank with a small effective sample
        size (see :func:`effective_sample_size`) gives a high-variance posterior
        estimate even though the weights still sum to one.  Hard prior
        exclusions must be encoded with a large finite penalty (for example
        ``-1e30``) because ``-inf`` inputs are rejected.

    Examples:
        A flat likelihood on two candidates plus prior odds 1:3 gives posterior
        probabilities 1/4 and 3/4::

            log_prior = torch.tensor([0.0, math.log(3.0)])
            weights = posterior_log_weights(torch.zeros(2), log_prior=log_prior).exp()
            # tensor([0.2500, 0.7500])

        For candidates drawn from the prior the corrections cancel, so the
        posterior is simply the normalized ratio::

            weights = posterior_log_weights(log_ratio)  # log_reference omitted
    """
    ratio = _as_float_tensor("log_ratio", log_ratio)
    if ratio.ndim == 0:
        raise ValueError(
            "log_ratio must have at least one dimension (a [K] candidate bank or a "
            "[N, K] batch); got a scalar with shape ()"
        )
    _check_finite("log_ratio", ratio)
    axis = _resolve_dim(dim, ratio.ndim, "log_ratio")
    if ratio.shape[axis] == 0:
        raise ValueError(
            f"log_ratio must be non-empty along dim {dim}; got shape {tuple(ratio.shape)}"
        )

    prior = _prepare_correction("log_prior", log_prior, ratio.shape)
    reference = _prepare_correction("log_reference", log_reference, ratio.shape)
    dtype = _promoted_dtype(ratio, prior, reference)

    adjusted = ratio.to(dtype)
    if prior is not None:
        adjusted = adjusted + prior.to(dtype)
    if reference is not None:
        adjusted = adjusted - reference.to(dtype)
    return F.log_softmax(adjusted, dim=axis)


def effective_sample_size(log_weights: _TensorLike, *, dim: int = -1) -> Tensor:
    r"""Kish effective sample size of (log) candidate weights.

    For weights :math:`w_k = \exp(\log w_k)` normalized as
    :math:`\sum_k w_k = 1`, the Kish effective sample size is

    .. math::

        \mathrm{ESS} = \frac{\left(\sum_k w_k\right)^2}{\sum_k w_k^2}
            = \exp\!\Big(-\operatorname{logsumexp}_k\, 2 \log w_k\Big),

    evaluated in log space so that extreme ratios do not over- or underflow.
    Mathematically :math:`1 \le \mathrm{ESS} \le K`: :math:`K` for uniform
    weights and :math:`1` when a single candidate carries all the mass.  This is
    the standard candidate-bank quality diagnostic for NRE and importance
    sampling: the variance of importance-weighted summaries such as the
    posterior mean scales roughly as :math:`1 / \mathrm{ESS}`, so an ESS close to
    one means the summary is controlled by a single candidate and is unreliable.

    Unnormalized log weights are also accepted: the normalization is applied
    internally, so shifting all log weights by a constant leaves the ESS
    unchanged (for already-normalized inputs this is the formula above).

    Args:
        log_weights: Log candidate weights with the :math:`K` candidates along
            ``dim``, e.g. the output of :func:`posterior_log_weights` or
            :func:`grid_log_posterior`.  Shape ``[K]`` or any batch of banks
            such as ``[N, K]``.  Values must be finite.
        dim: Candidate axis.  Defaults to ``-1``.

    Returns:
        Effective sample size with ``dim`` removed: a 0-dim tensor for a single
        bank ``[K]`` and ``[N]`` for a batch ``[N, K]``.  Floating-point rounding
        can move the result a few ulps outside :math:`[1, K]`.

    Raises:
        ValueError: If ``log_weights`` is a scalar or empty along ``dim``,
            contains NaN or infinity, or if ``dim`` is out of range.

    Examples:
        A healthy bank should keep most of its candidates alive::

            log_w = posterior_log_weights(log_ratio)
            ess = effective_sample_size(log_w)  # in [1, K]
            healthy = ess > 0.5 * log_ratio.shape[-1]

        Use the ESS to flag a poorly covering proposal, then widen
        ``log_reference``/``q`` and redraw the bank.
    """
    weights = _as_float_tensor("log_weights", log_weights)
    if weights.ndim == 0:
        raise ValueError(
            "log_weights must have at least one dimension (a [K] candidate bank or a "
            "[N, K] batch); got a scalar with shape ()"
        )
    _check_finite("log_weights", weights)
    axis = _resolve_dim(dim, weights.ndim, "log_weights")
    if weights.shape[axis] == 0:
        raise ValueError(
            f"log_weights must be non-empty along dim {dim}; got shape {tuple(weights.shape)}"
        )

    normalized = weights - torch.logsumexp(weights, dim=axis, keepdim=True)
    return torch.exp(-torch.logsumexp(2.0 * normalized, dim=axis))


def grid_log_posterior(
    log_scores: _TensorLike,
    *,
    log_prior: _CorrectionLike | None = None,
    log_reference: _CorrectionLike | None = None,
    dim: int = -1,
) -> Tensor:
    r"""Log posterior weights for a likelihood scan (parameter grid).

    This is :func:`posterior_log_weights` under the naming used by likelihood
    scan / profile-likelihood workflows.  ``log_scores[n, k]`` holds the log
    likelihood (or an already-estimated log ratio) of observation :math:`x_n` at
    grid hypothesis :math:`\theta_k`, and the returned log weights are

    .. math::

        \log \tilde{w}_{nk} = \log p(x_n \mid \theta_k) + \log p(\theta_k)
                             - \log q_k,
        \qquad
        \log w_{nk} = \log \tilde{w}_{nk}
                      - \operatorname{logsumexp}_j \log \tilde{w}_{nj}.

    For a *uniform* grid over the parameter volume, :math:`q_k` is constant over
    the grid points and cancels in the softmax; subtract ``log_reference`` only
    for a non-uniform grid (for example importance-sampled grid points or points
    representing cells of unequal volume).

    Args:
        log_scores: Log likelihoods :math:`\log p(x \mid \theta_k)` (or log
            ratios) with the :math:`K` grid points along ``dim``.  Shape ``[K]``
            for a single scan, ``[N, K]`` for a batch.  Values must be finite.
        log_prior: Target prior :math:`\log p(\theta_k)` evaluated on the grid,
            broadcastable to ``log_scores`` without expanding it.  Defaults to
            ``None`` (flat prior).
        log_reference: Log proposal density / cell-volume term
            :math:`\log q_k` of the grid, broadcastable to ``log_scores``.
            Defaults to ``None`` (uniform grid).
        dim: Grid axis of ``log_scores``.  Defaults to ``-1``.

    Returns:
        Log posterior weights with the same shape as ``log_scores``; ``exp`` of
        the result sums to one along ``dim``.

    Raises:
        ValueError: If ``log_scores`` is a scalar or empty along ``dim``, if any
            input contains NaN or infinity, if a correction is not broadcastable
            to ``log_scores`` without expanding it, or if ``dim`` is out of
            range.

    Warning:
        The weights are posterior probabilities *within the supplied grid*: they
        ignore posterior mass outside the grid and assume each grid point
        represents an equal-volume cell.  A coarse or truncated grid therefore
        biases posterior summaries.  Represent non-uniform cells by adding
        ``log V_k`` (the log cell volume) to ``log_reference``.

    Examples:
        A one-dimensional likelihood scan with a Gaussian prior::

            theta = torch.linspace(-4.0, 4.0, 161)
            log_like = -0.5 * ((x - theta) / sigma) ** 2
            log_prior = -0.5 * (theta / prior_scale) ** 2  # standard normal, up to const
            weights = grid_log_posterior(log_like, log_prior=log_prior).exp()
            posterior_mean = (weights * theta).sum()
    """
    return posterior_log_weights(
        log_scores,
        log_prior=log_prior,
        log_reference=log_reference,
        dim=dim,
    )


def maximum_likelihood_estimate(
    scores: _TensorLike,
    hypotheses: _TensorLike,
    *,
    log_prior: _CorrectionLike | None = None,
    dim: int = -1,
) -> tuple[Tensor, Tensor]:
    r"""Maximum-likelihood or maximum-a-posteriori estimate from scored hypotheses.

    For each bank of :math:`K` scored hypotheses, picks the index maximizing the
    score, optionally plus a log prior:

    .. math::

        \hat{k}_n = \arg\max_k \Big( s_{nk} + \log p(\theta_k) \Big),
        \qquad
        \hat{\theta}_n = \theta_{\hat{k}_n},

    where :math:`s_{nk} = \log p(x_n \mid \theta_k)` are log likelihoods (or log
    likelihood ratios; the :math:`\theta`-independent reference cancels in the
    argmax).  Without ``log_prior`` this is the maximum-likelihood estimate
    (MLE); supplying the *target* prior turns it into the maximum-a-posteriori
    estimate (MAP).

    Args:
        scores: Log likelihoods or log ratios with the :math:`K` hypotheses
            along ``dim``.  Shape ``[K]`` or ``[N, K]``; extra batch dimensions
            are allowed.  Values must be finite.
        hypotheses: Hypothesis coordinates, shape ``[K, D]`` or ``[K]``.  For a
            1-D tensor a single coordinate column is assumed (``[K, 1]``), so a
            grid of scalar parameters needs no reshaping.  Values must be
            finite; non-floating hypotheses are cast to the default float dtype.
        log_prior: Optional target log prior, broadcastable to ``scores``
            without expanding it (a scalar, ``[K]``, or ``[N, K]`` tensor).
            ``None`` gives the MLE; a prior gives the MAP.  Defaults to ``None``.
        dim: Hypothesis axis of ``scores``.  Defaults to ``-1``.

    Returns:
        A pair ``(estimates, best_index)``:

        * ``best_index`` -- integer tensor with ``dim`` removed from
          ``scores.shape``; the selected hypothesis index for every bank.
        * ``estimates`` -- ``hypotheses[best_index]``, shape
          ``scores.shape`` with ``dim`` removed plus ``[D]``.  For 1-D ``scores``
          of shape ``[K]`` this is ``[D]``.

    Raises:
        ValueError: If ``scores`` is a scalar or empty along ``dim``, if any
            input contains NaN or infinity, if ``hypotheses`` is not 1-D or 2-D,
            if its first dimension does not equal ``scores.shape[dim]``, if
            ``log_prior`` is not broadcastable to ``scores`` without expanding
            it, or if ``dim`` is out of range.

    Notes:
        * Ties are broken by ``torch.argmax``: the first maximal index along
          ``dim`` wins.
        * The MLE is invariant to the prior used to *train or sample* the ratio
          estimator: no prior term enters the likelihood argmax and the
          :math:`\theta`-independent reference density cancels.  Supplying
          ``log_prior`` is what turns the MLE into a MAP under the target prior,
          which may differ from any training prior.
        * This returns a point estimate from a finite candidate bank, not the
          posterior mean.  For the minimum-MSE posterior summary, reweight the
          hypotheses with :func:`posterior_log_weights` and take the weighted
          average.
        * No ``log_reference`` argument is accepted because importance-sampling
          corrections change posterior *mass*, not the likelihood argmax.  For
          the mode of an importance-weighted posterior density, pass
          ``scores - log_reference`` explicitly.

    Examples:
        Recover a known maximum on an analytic grid::

            theta = torch.linspace(-3.0, 3.0, 121)
            x = torch.tensor([0.7, -1.4])
            scores = -((x[:, None] - theta[None, :]) ** 2)
            estimates, best = maximum_likelihood_estimate(scores, theta)
            # estimates.squeeze(-1) == x, best == tensor([74, 32])

        Add a prior to turn the MLE into a MAP::

            estimates, best = maximum_likelihood_estimate(
                scores, theta, log_prior=torch.log(prior_weights)
            )
    """
    log_scores = _as_float_tensor("scores", scores)
    if log_scores.ndim == 0:
        raise ValueError(
            "scores must have at least one dimension (a [K] hypothesis bank or a "
            "[N, K] batch); got a scalar with shape ()"
        )
    _check_finite("scores", log_scores)
    axis = _resolve_dim(dim, log_scores.ndim, "scores")
    n_hypotheses = log_scores.shape[axis]
    if n_hypotheses == 0:
        raise ValueError(
            f"scores must be non-empty along dim {dim}; got shape {tuple(log_scores.shape)}"
        )
    prior = _prepare_correction("log_prior", log_prior, log_scores.shape)

    coordinates = _as_float_tensor("hypotheses", hypotheses)
    _check_finite("hypotheses", coordinates)
    if coordinates.ndim not in (1, 2):
        raise ValueError(
            f"hypotheses must be 1-D [K] or 2-D [K, D]; got shape {tuple(coordinates.shape)}"
        )
    if coordinates.shape[0] != n_hypotheses:
        raise ValueError(
            f"hypotheses must have one row per scored hypothesis ({n_hypotheses}); "
            f"got {coordinates.shape[0]} row(s)"
        )
    if coordinates.ndim == 1:
        coordinates = coordinates.unsqueeze(-1)

    dtype = _promoted_dtype(log_scores, prior)
    adjusted = log_scores.to(dtype)
    if prior is not None:
        adjusted = adjusted + prior.to(dtype)

    best_index = torch.argmax(adjusted, dim=axis)
    return coordinates[best_index], best_index
