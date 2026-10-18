"""Simulation-based calibration (SBC) and TARP posterior diagnostics.

These diagnostics answer a different question from the predictive/interval
calibration metrics in :mod:`torchregress.calibration.metrics`: not *"are the
predictive intervals calibrated for the observed targets?"* but *"does the
posterior inference algorithm recover the parameters that generated the data?"*
They require simulated data for which the ground-truth parameters are known, which
is the natural setting for simulation-based inference (SBI): approximate Bayesian
computation, neural posterior estimation, MCMC/VI validation, and any
forward-model inversion.

SBC
---
Simulation-based calibration (Cook, Gelman & Rubin 2006; Talts et al. 2018)
draws ``N`` parameter values from the prior, simulates one data set per draw,
runs the inference algorithm, and computes, for every parameter dimension ``d``,
the rank of the true value among ``S`` posterior draws::

    r[n, d] = #{s : theta[n, s, d] < theta_star[n, d]}
              + 0.5 * #{s : theta[n, s, d] == theta_star[n, d]}

For a calibrated posterior the true value is exchangeable with the posterior
draws, so ``r[n, d]`` is discrete-uniform on ``{0, 1, ..., S}``. Any systematic
deviation (U-shape from over-confidence, skew from a biased posterior, pile-up
at the extremes from under-confidence) is a symptom of miscalibration. The
``0.5`` tie credit is the expectation of the standard randomised rank used for
discrete parameters (Talts et al. 2018, Appendix A).

TARP
----
The TARP test (Lemos et al. 2023) constructs *random* credible regions by
picking a reference point ``theta_ref``, computing distances
``d_s = ||theta_s - theta_ref||`` from the posterior draws, and taking the
``alpha``-quantile ``q_alpha`` of those distances. The random region is the
ball of radius ``q_alpha`` around ``theta_ref``. For a calibrated posterior the
true parameter is exchangeable with the posterior draws *for any* reference
point independent of it, so the expected coverage is exactly ``alpha``::

    coverage(alpha) = (1 / N) * sum_n 1[d_true[n] <= q_alpha[n]]

TARP therefore tests *joint* calibration of the posterior (dispersion,
correlation structure and biases) rather than marginal interval coverage. The
standard choice is to draw the reference points from the prior.

References
----------
.. [1] Cook, S. R., Gelman, A., & Rubin, D. B. (2006). Validation of software
   for Bayesian models. *Bayesian Analysis*, 1(3), 409-434.
.. [2] Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A.
   (2018). Validating Bayesian inference algorithms with simulation-based
   calibration. arXiv:1804.06788.
.. [3] Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023).
   Sampling-based accuracy testing of posterior estimators for general
   inference. In *ICML 2023*. arXiv:2302.03026.
.. [4] Geweke, J. (2004). Getting it right: joint distribution tests of
   posterior simulators. *Journal of the American Statistical Association*,
   99(467), 799-804.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch

from torchregress.utils.tensor_ops import convert_to_tensor, float_dtype

__all__ = [
    "sbc_rank_histogram",
    "sbc_uniformity_pvalue",
    "simulation_based_calibration_ranks",
    "tarp_coverage",
]

ArrayLike = Union[torch.Tensor, np.ndarray]

# Below this number of expected ranks per bin the chi-square approximation in
# :func:`sbc_uniformity_pvalue` is not reliable (standard rule of thumb).
_MIN_EXPECTED_PER_BIN = 5.0


def simulation_based_calibration_ranks(
    posterior_samples: ArrayLike,
    true_values: ArrayLike,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Tie-aware SBC ranks of the true parameters among posterior draws.

    For every observation ``n`` and parameter dimension ``d``, the rank is the
    number of posterior draws strictly below the true value plus half the number
    of draws exactly equal to it:

    .. math::

        r_{nd} = \\sum_{s=1}^{S} \\mathbb{1}
        \\left[\\theta^{(s)}_{nd} < \\theta^{\\star}_{nd}\\right]
        + \\tfrac{1}{2} \\sum_{s=1}^{S} \\mathbb{1}
        \\left[\\theta^{(s)}_{nd} = \\theta^{\\star}_{nd}\\right]

    For a calibrated posterior (one that is exact for the prior/likelihood pair
    used to simulate the data) the true value is exchangeable with the posterior
    draws and :math:`r_{nd}` is discrete-uniform on :math:`\\{0, \\dots, S\\}`,
    i.e. :math:`\\Pr(r_{nd} = k) = 1 / (S + 1)`.

    Parameters
    ----------
    posterior_samples : Tensor or ndarray
        Posterior draws. Canonical shape is ``[N, S, D]`` (``N`` simulated
        observations, ``S`` posterior draws each, ``D`` parameter dimensions);
        a 2-D ``[S, D]`` array is treated as a single observation and yields a
        ``[1, D]`` rank tensor.
    true_values : Tensor or ndarray
        Ground-truth parameters, shape ``[N, D]`` or ``[D]`` for a single
        observation.
    dim : int, default -1
        Axis of ``posterior_samples`` that indexes the parameter dimension
        (the last axis for the canonical ``[N, S, D]`` layout). The sample axis
        is the other non-batch axis, e.g. ``dim=1`` reads ``[N, D, S]`` input.
        The batch axis is always the first axis and cannot be ``dim``.

    Returns
    -------
    Tensor
        Ranks of shape ``[N, D]`` in the promoted floating dtype of the inputs.
        With the ``0.5`` tie convention the values are multiples of ``0.5`` in
        ``[0, S]``.

    Raises
    ------
    ValueError
        If shapes are incompatible, ``dim`` is invalid, or any input contains
        NaN/infinite values.

    Notes
    -----
    SBC validates the *marginal* rank distribution of each parameter dimension
    separately. It is a necessary but not sufficient condition for a calibrated
    posterior: a posterior can have uniform marginals and still be wrong about
    correlations or modes, and a non-uniform rank histogram rejects the
    combination "prior + likelihood + inference algorithm" rather than
    pinpointing which component is at fault. The rank definition also assumes
    that ``S`` draws are (approximately) independent posterior draws; strongly
    auto-correlated MCMC output does not satisfy the sampler assumptions of SBC.

    Examples
    --------
    >>> import torch
    >>> g = torch.Generator().manual_seed(0)
    >>> truth = torch.randn(3, 2, generator=g)
    >>> samples = truth.unsqueeze(1) + torch.randn(3, 100, 2, generator=g)
    >>> ranks = simulation_based_calibration_ranks(samples, truth)
    >>> ranks.shape
    torch.Size([3, 2])
    """
    samples, truths = _normalize_posterior_and_truth(
        posterior_samples, true_values, dim=dim, name="posterior_samples"
    )
    rank_dtype = float_dtype(samples, truths)

    below = (samples < truths.unsqueeze(1)).sum(dim=1)
    equal = (samples == truths.unsqueeze(1)).sum(dim=1)
    return below.to(rank_dtype) + 0.5 * equal.to(rank_dtype)


def sbc_rank_histogram(
    ranks: ArrayLike,
    *,
    n_bins: Optional[int] = None,
    n_posterior_samples: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Per-dimension SBC rank histogram over ``[0, S + 1]``.

    For a calibrated posterior the rank histogram is approximately flat; an
    over-confident posterior produces a U-shape (ranks pile up at both ends,
    because the true value often lies outside the posterior draws), an
    under-confident posterior produces a central peak, and a biased posterior
    produces a skewed histogram.

    Parameters
    ----------
    ranks : Tensor or ndarray
        Ranks from :func:`simulation_based_calibration_ranks`: shape ``[N, D]``,
        or ``[N]`` for a single parameter dimension. Values must be in
        ``[0, S]`` (half-integers are allowed for tied draws).
    n_bins : int, optional
        Number of histogram bins. Defaults to ``S + 1`` bins, one per possible
        integer rank, with integer bin edges ``0, 1, ..., S + 1``. Coarser
        grids use ``n_bins`` strictly increasing integer edges spread as evenly as
        possible over the same range (``1 <= n_bins <= S + 1``).
    n_posterior_samples : int, optional
        Number of posterior draws ``S`` behind each rank. If omitted it is
        inferred as ``ceil(max(ranks))``, which is exact only when a rank at the
        top of the range was observed; pass ``S`` explicitly when
        ``N`` is not much larger than ``S``.

    Returns
    -------
    dict
        ``"counts"``: histogram counts, shape ``[n_bins]`` for a single
        parameter dimension and ``[D, n_bins]`` otherwise; ``"bin_edges"``:
        ``n_bins + 1`` bin edges.

    Raises
    ------
    ValueError
        If ranks are empty/negative/non-finite, ``S`` cannot be inferred, ranks
        exceed ``S``, or ``n_bins`` is not in ``1 <= n_bins <= S + 1``.

    Examples
    --------
    >>> import torch
    >>> g = torch.Generator().manual_seed(0)
    >>> truth = torch.randn(500, 1, generator=g)
    >>> samples = truth.unsqueeze(1) + torch.randn(500, 50, 1, generator=g)
    >>> ranks = simulation_based_calibration_ranks(samples, truth)
    >>> hist = sbc_rank_histogram(ranks)
    >>> hist["counts"].sum()
    tensor(500.)
    """
    rank_matrix = _normalize_rank_matrix(ranks)
    native_samples = _resolve_n_posterior_samples(rank_matrix, n_posterior_samples)
    n_bins = _resolve_n_bins(n_bins, native_samples)
    edges = _integer_bin_edges(native_samples, n_bins, rank_matrix)
    counts = _histogram_counts(rank_matrix, edges).to(rank_matrix.dtype)

    if counts.shape[0] == 1:
        counts = counts[0]
    result: dict[str, torch.Tensor] = {"counts": counts, "bin_edges": edges}
    return result


def sbc_uniformity_pvalue(
    ranks: ArrayLike,
    *,
    n_bins: Optional[int] = None,
    n_posterior_samples: Optional[int] = None,
) -> dict[str, object]:
    """Chi-square goodness-of-fit test of SBC ranks against discrete uniformity.

    For each parameter dimension the ranks are binned and compared with the
    exact discrete-uniform expectation under a calibrated posterior:

    .. math::

        T_d = \\sum_{b=1}^{B} \\frac{(O_{db} - E_b)^2}{E_b},
        \\qquad E_b = N \\, \\frac{w_b}{S + 1},

    where :math:`w_b` is the number of integer rank values in bin ``b`` (equal
    to one for the default ``S + 1`` bins) and :math:`T_d \\sim \\chi^2_{B-1}`
    under the null hypothesis. Small p-values indicate miscalibration.

    Parameters
    ----------
    ranks : Tensor or ndarray
        Ranks from :func:`simulation_based_calibration_ranks`, shape ``[N, D]``
        or ``[N]`` for a single parameter dimension.
    n_bins : int, optional
        Number of bins, default ``S + 1`` (one per possible integer rank).
        Coarser grids use ``n_bins`` strictly increasing integer edges spread as
        evenly as possible over ``[0, S + 1]``. Must satisfy
        ``2 <= n_bins <= S + 1``.
    n_posterior_samples : int, optional
        Number of posterior draws ``S``; inferred as ``ceil(max(ranks))`` when
        omitted (pass it explicitly when ``N`` is not much larger than ``S``).

    Returns
    -------
    dict
        ``"p_values"``: one p-value per parameter dimension; ``"min_p_value"``:
        smallest of them; ``"chi2"``: chi-square statistics per dimension;
        ``"dof"``: degrees of freedom (``n_bins - 1``); ``"n_ranks"``: number
        of ranks per dimension ``N``; ``"n_bins"``: number of bins used;
        ``"n_posterior_samples"``: ``S`` used; ``"bonferroni_min_p_value"``:
        ``min_p_value`` multiplied by the number of dimensions and capped at 1
        (conservative family-wise correction for testing ``D`` dimensions).

    Raises
    ------
    ValueError
        If the inputs are invalid or there are fewer than ~5 expected ranks per
        bin, in which case the chi-square approximation is unreliable: use a
        coarser ``n_bins`` or more simulated observations.

    Notes
    -----
    The p-value is a stochastic diagnostic, not a calibration certificate: under
    the null it is uniform on ``[0, 1]``, so with many parameter dimensions (or
    many repeated runs) small p-values occur by chance. Use
    ``"bonferroni_min_p_value"`` for a conservative multiple-testing control,
    and complement the test with the rank histogram. Rejecting the null only
    says that the prior/likelihood/posterior triple is inconsistent with
    uniformity somewhere, not where. The chi-square null assumes
    (near-)continuous parameters; heavy ties make the discrete-uniform
    assumption of the tie-aware rank only approximate.
    """
    # scipy is a core dependency but heavy to import; keep the metrics package
    # import light (same convention as the local scipy imports elsewhere).
    from scipy.stats import chisquare

    rank_matrix = _normalize_rank_matrix(ranks)
    n_obs, n_dims = rank_matrix.shape
    native_samples = _resolve_n_posterior_samples(rank_matrix, n_posterior_samples)
    n_bins = _resolve_n_bins(n_bins, native_samples, minimum=2)
    edges = _integer_bin_edges(native_samples, n_bins, rank_matrix)
    counts = _histogram_counts(rank_matrix, edges)

    widths = torch.diff(edges).to(torch.float64)
    expected = n_obs * widths / float(native_samples + 1)
    min_expected = float(expected.min().item())
    if min_expected < _MIN_EXPECTED_PER_BIN:
        raise ValueError(
            "chi-square uniformity test requires at least "
            f"{_MIN_EXPECTED_PER_BIN:.0f} expected ranks per bin; got "
            f"{min_expected:.3g} for n_bins={n_bins}, n_ranks={n_obs}, "
            f"S={native_samples}. Reduce n_bins or simulate more observations."
        )

    expected_np = expected.cpu().numpy()
    chi2: list[float] = []
    p_values: list[float] = []
    for dim_idx in range(n_dims):
        observed_np = counts[dim_idx].to(torch.float64).cpu().numpy()
        statistic, p_value = chisquare(observed_np, f_exp=expected_np)
        chi2.append(float(statistic))
        p_values.append(float(p_value))

    min_p_value = float(min(p_values))
    result: dict[str, object] = {
        "p_values": p_values,
        "min_p_value": min_p_value,
        "chi2": chi2,
        "dof": n_bins - 1,
        "n_ranks": int(n_obs),
        "n_bins": int(n_bins),
        "n_posterior_samples": int(native_samples),
        "bonferroni_min_p_value": float(min(1.0, min_p_value * max(n_dims, 1))),
    }
    return result


def tarp_coverage(
    posterior_samples: ArrayLike,
    true_values: ArrayLike,
    reference_values: ArrayLike,
    *,
    alphas: Optional[ArrayLike] = None,
) -> dict[str, Union[torch.Tensor, float, int]]:
    """TARP coverage curve of random credible regions (Lemos et al. 2023).

    For each observation ``n`` the distances from the posterior draws to a
    reference point are

    .. math::

        d_{ns} = \\lVert \\theta^{(s)}_n - \\tilde\\theta_n \\rVert_2,
        \\qquad
        d^\\star_n = \\lVert \\theta^\\star_n - \\tilde\\theta_n \\rVert_2,

    the ``alpha``-quantile of ``{d_ns}`` defines the random credible region
    ``R_n(alpha) = {theta : ||theta - theta_ref_n|| <= q_n(alpha)}``, and the
    empirical coverage is

    .. math::

        \\hat c(\\alpha) = \\frac{1}{N} \\sum_{n=1}^{N}
        \\mathbb{1}\\left[d^\\star_n \\le q_n(\\alpha)\\right].

    For a calibrated posterior and reference points that are independent of the
    true parameters and of the posterior draws, ``E[c(alpha)] = alpha`` for every
    ``alpha`` in ``[0, 1]``. The TARP error is
    ``max_alpha |c(alpha) - alpha|``; calibrated posteriors give a small error,
    biased or mis-dispersed posteriors give a large one. Unlike SBC, TARP tests
    the *joint* posterior (including correlations and dispersion) rather than
    one marginal at a time.

    Parameters
    ----------
    posterior_samples : Tensor or ndarray
        Posterior draws, shape ``[N, S, D]`` (or ``[S, D]`` for a single
        observation).
    true_values : Tensor or ndarray
        Ground-truth parameters, shape ``[N, D]`` (or ``[D]`` for a single
        observation).
    reference_values : Tensor or ndarray
        Reference points ``theta_ref`` used to build the random regions: shape
        ``[N, D]`` for one point per observation, or ``[D]`` for a single point
        shared by all observations. Draw them independently from the prior (in
        the same parameter space and units as the posterior); never pass the
        true generating parameters, which would trivially cover the truth.
    alphas : Tensor or ndarray, optional
        Coverage levels, 1-D with values in ``[0, 1]``. Defaults to
        ``linspace(0, 1, 101)``. The implementation is vectorised over
        ``alphas`` (no Python loop).

    Returns
    -------
    dict
        ``"alpha"``: coverage levels ``[A]``; ``"coverage"``: empirical
        coverage per level ``[A]``; ``"tarp_error"``: ``max |coverage - alpha|``
        as a float; ``"n_samples"``: number of simulated observations ``N``
        used for the coverage estimate; ``"n_posterior_samples"``: number of
        posterior draws ``S`` per observation.

    Raises
    ------
    ValueError
        If shapes are incompatible, fewer than two posterior draws are given,
        ``alphas`` is invalid, or any input contains NaN/infinite values.

    Notes
    -----
    The coverage property is an expectation over the data *and* the random
    reference points, so ``tarp_error`` is a noisy diagnostic: with finite
    ``N`` it is biased upward by the maximum over the ``alpha`` grid (it is not
    a p-value and has no calibrated null threshold), and with finite ``S`` the
    empirical distance quantiles add an ``O(1 / S)`` bias to the coverage curve.
    Compare models/estimators on the same simulated data set and with large
    ``N`` (e.g. thousands) and ``S`` (e.g. hundreds).
    Distances are Euclidean, so parameter dimensions with very different units
    or scales should be standardised/whitened first; otherwise TARP is dominated
    by the largest-scale parameters. Coverage here refers to the random-ball
    regions, not to shortest/highest-density credible sets of fixed mass.

    Examples
    --------
    >>> import torch
    >>> g = torch.Generator().manual_seed(0)
    >>> truth = torch.randn(200, 2, generator=g)
    >>> noise = torch.randn(200, 2, generator=g)
    >>> samples = 0.5 * (truth + noise).unsqueeze(1) + 0.7071 * torch.randn(
    ...     200, 100, 2, generator=g
    ... )
    >>> refs = torch.randn(200, 2, generator=torch.Generator().manual_seed(1))
    >>> out = tarp_coverage(samples, truth, refs)
    >>> float(out["tarp_error"]) < 0.1
    True
    """
    samples, truths = _normalize_posterior_and_truth(
        posterior_samples, true_values, dim=-1, name="posterior_samples"
    )
    n_obs, n_samples, n_dims = samples.shape
    if n_samples < 2:
        raise ValueError("tarp_coverage requires at least 2 posterior samples per observation")

    # Distances require floating point: integer/discrete parameters are promoted.
    compute_dtype = float_dtype(samples, truths)
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    samples = samples.to(compute_dtype)
    truths = truths.to(compute_dtype)

    refs = convert_to_tensor(reference_values)
    if refs.dim() == 1:
        if refs.shape[0] != n_dims:
            raise ValueError(
                f"reference_values has {refs.shape[0]} dimensions but the posterior has {n_dims}"
            )
        refs = refs.unsqueeze(0).expand(n_obs, -1)
    elif refs.dim() == 2:
        if refs.shape[1] != n_dims:
            raise ValueError(
                f"reference_values has {refs.shape[1]} dimensions but the posterior has {n_dims}"
            )
        if refs.shape[0] != n_obs:
            if refs.shape[0] == 1:
                refs = refs.expand(n_obs, -1)
            else:
                raise ValueError(
                    "reference_values must have shape [N, D] matching the observations "
                    f"(N={n_obs}) or [D] for a shared reference point; got {tuple(refs.shape)}"
                )
    else:
        raise ValueError(
            f"reference_values must be 1-D [D] or 2-D [N, D]; got {refs.dim()} dimensions"
        )
    refs = refs.to(device=samples.device, dtype=samples.dtype)
    if not torch.isfinite(refs).all():
        raise ValueError("reference_values contains NaN or infinite values")

    if alphas is None:
        alpha = torch.linspace(0.0, 1.0, 101, device=samples.device, dtype=samples.dtype)
    else:
        alpha = convert_to_tensor(alphas).to(device=samples.device, dtype=samples.dtype)
        if alpha.dim() != 1 or alpha.numel() == 0:
            raise ValueError("alphas must be a non-empty 1-D tensor of coverage levels")
        if not torch.isfinite(alpha).all():
            raise ValueError("alphas contains NaN or infinite values")
        if bool((alpha < 0.0).any()) or bool((alpha > 1.0).any()):
            raise ValueError("alphas must lie in [0, 1]")

    sample_offsets = samples - refs.unsqueeze(1)
    distances = torch.linalg.vector_norm(sample_offsets, dim=-1)  # [N, S]
    truth_offsets = truths - refs
    true_distances = torch.linalg.vector_norm(truth_offsets, dim=-1)  # [N]

    # Vectorised empirical quantiles over the alpha grid: linear interpolation
    # between the two nearest order statistics (matches torch.quantile's
    # default "linear" interpolation) for every observation and alpha at once.
    sorted_distances = torch.sort(distances, dim=1).values
    position = alpha * (n_samples - 1)
    lower = torch.floor(position).long().clamp_(0, n_samples - 1)
    upper = torch.ceil(position).long().clamp_(0, n_samples - 1)
    fraction = (position - lower.to(alpha.dtype)).unsqueeze(0)
    thresholds = sorted_distances[:, lower] * (1.0 - fraction) + sorted_distances[:, upper] * (
        fraction
    )
    covered = true_distances.unsqueeze(1) <= thresholds
    coverage = covered.to(alpha.dtype).mean(dim=0)

    if coverage.numel() == 0:  # defensive: unreachable for a non-empty alpha grid
        raise ValueError("alphas must contain at least one coverage level")
    tarp_error = float(torch.abs(coverage.double() - alpha.double()).max().item())

    result: dict[str, Union[torch.Tensor, float, int]] = {
        "alpha": alpha,
        "coverage": coverage,
        "tarp_error": tarp_error,
        "n_samples": int(n_obs),
        "n_posterior_samples": int(n_samples),
    }
    return result


def _normalize_posterior_and_truth(
    posterior_samples: ArrayLike,
    true_values: ArrayLike,
    *,
    dim: int,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bring posterior draws to ``[N, S, D]`` and truths to ``[N, D]``.

    Validates shapes/dtypes/finiteness and applies ``dim`` as the parameter
    dimension axis of the (3-D) sample tensor.
    """
    samples = convert_to_tensor(posterior_samples)
    truths = convert_to_tensor(true_values)

    if samples.dim() == 2:
        param_axis = dim % 2
        if param_axis != 1:
            raise ValueError(
                f"dim={dim} is not the parameter-dimension axis of 2-D {name} [S, D]; pass dim=-1"
            )
        samples = samples.unsqueeze(0)
    elif samples.dim() == 3:
        param_axis = dim % 3
        if param_axis == 0:
            raise ValueError(
                f"dim={dim} selects the batch axis of {name}; expected the parameter "
                "dimension axis (default -1)"
            )
        sample_axis = 1 if param_axis == 2 else 2
        if sample_axis != 1:
            samples = samples.permute(0, sample_axis, param_axis)
    else:
        raise ValueError(
            f"{name} must be 2-D [S, D] or 3-D [N, S, D]; got shape {tuple(samples.shape)}"
        )

    if samples.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one posterior sample")
    if not torch.isfinite(samples).all():
        raise ValueError(f"{name} contains NaN or infinite values")

    if truths.dim() == 1:
        if samples.shape[0] != 1:
            raise ValueError(
                "true_values of shape [D] is only valid for a single observation; "
                f"got N={samples.shape[0]}, use [N, D]"
            )
        truths = truths.unsqueeze(0)
    elif truths.dim() != 2:
        raise ValueError(
            f"true_values must be 1-D [D] or 2-D [N, D]; got shape {tuple(truths.shape)}"
        )
    if truths.shape[0] != samples.shape[0]:
        raise ValueError(
            f"batch mismatch: {name} has {samples.shape[0]} observations but "
            f"true_values has {truths.shape[0]}"
        )
    if truths.shape[1] != samples.shape[2]:
        raise ValueError(
            f"dimension mismatch: {name} has {samples.shape[2]} parameter dimensions but "
            f"true_values has {truths.shape[1]}"
        )
    if not torch.isfinite(truths).all():
        raise ValueError("true_values contains NaN or infinite values")

    truths = truths.to(device=samples.device)
    common_dtype = torch.promote_types(samples.dtype, truths.dtype)
    samples = samples.to(common_dtype)
    truths = truths.to(common_dtype)
    if samples.dtype in (torch.float16, torch.bfloat16):
        samples = samples.to(torch.float32)
        truths = truths.to(torch.float32)
    return samples, truths


def _normalize_rank_matrix(ranks: ArrayLike) -> torch.Tensor:
    """Return ranks as a 2-D floating tensor ``[N, D]`` in at least float32."""
    rank_tensor = convert_to_tensor(ranks)
    if rank_tensor.dim() == 1:
        rank_tensor = rank_tensor.unsqueeze(-1)
    elif rank_tensor.dim() != 2:
        raise ValueError(f"ranks must be 1-D [N] or 2-D [N, D]; got {rank_tensor.dim()} dimensions")
    if rank_tensor.numel() == 0:
        raise ValueError("ranks must contain at least one value")

    dtype = float_dtype(rank_tensor)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    rank_tensor = rank_tensor.to(dtype)
    if not torch.isfinite(rank_tensor).all():
        raise ValueError("ranks contain NaN or infinite values")
    if bool((rank_tensor < 0).any()):
        raise ValueError("ranks must be non-negative")
    return rank_tensor


def _resolve_n_posterior_samples(
    rank_matrix: torch.Tensor, n_posterior_samples: Optional[int]
) -> int:
    """Return ``S`` from the explicit argument or infer it from the max rank."""
    max_rank = float(rank_matrix.max().item())
    if n_posterior_samples is None:
        inferred = int(math.ceil(max_rank))
        if inferred < 1:
            raise ValueError(
                "cannot infer the number of posterior samples from ranks that are all 0; "
                "pass n_posterior_samples explicitly"
            )
        return inferred
    if isinstance(n_posterior_samples, bool) or not isinstance(n_posterior_samples, int):
        raise ValueError("n_posterior_samples must be a positive integer")
    if n_posterior_samples < 1:
        raise ValueError("n_posterior_samples must be a positive integer")
    if max_rank > n_posterior_samples + 1e-6:
        raise ValueError(
            f"observed rank {max_rank} exceeds n_posterior_samples={n_posterior_samples}"
        )
    return n_posterior_samples


def _resolve_n_bins(n_bins: Optional[int], n_posterior_samples: int, *, minimum: int = 1) -> int:
    """Validate ``n_bins`` (default ``S + 1``, one bin per integer rank)."""
    if n_bins is None:
        return n_posterior_samples + 1
    if isinstance(n_bins, bool) or not isinstance(n_bins, int):
        raise ValueError("n_bins must be an integer")
    if n_bins < minimum:
        raise ValueError(f"n_bins must be at least {minimum}; got {n_bins}")
    if n_bins > n_posterior_samples + 1:
        raise ValueError(
            f"n_bins={n_bins} exceeds the {n_posterior_samples + 1} possible rank values; "
            "bins without integer ranks are not meaningful"
        )
    return n_bins


def _integer_bin_edges(
    n_posterior_samples: int, n_bins: int, reference: torch.Tensor
) -> torch.Tensor:
    """Integer bin edges over ``[0, S + 1]`` with bin widths as equal as possible."""
    range_size = n_posterior_samples + 1
    grid = torch.arange(n_bins + 1, device=reference.device, dtype=torch.float64)
    edges = torch.div(grid * range_size, n_bins, rounding_mode="floor")
    return edges.to(reference.dtype)


def _histogram_counts(rank_matrix: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Per-dimension histogram counts of ranks, shape ``[D, n_bins]`` (long)."""
    n_dims = rank_matrix.shape[1]
    n_bins = edges.numel() - 1
    values = rank_matrix if rank_matrix.is_contiguous() else rank_matrix.contiguous()
    indices = torch.bucketize(values, edges, right=True) - 1
    indices.clamp_(0, n_bins - 1)
    offsets = torch.arange(n_dims, device=rank_matrix.device).unsqueeze(0) * n_bins
    flat_indices = (indices + offsets).reshape(-1)
    flat_counts = torch.zeros(n_dims * n_bins, dtype=torch.long, device=rank_matrix.device)
    flat_counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices))
    return flat_counts.view(n_dims, n_bins)
