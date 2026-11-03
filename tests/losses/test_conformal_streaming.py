"""Chunked (streaming) density-conformal calibration and prediction.

Each test compares the chunked implementation with a reference copy of the
pre-chunking code path (one call over the whole split, per-row Python loops).
Scores are per row, so chunking is exact whenever the score is a row-wise
function built from IEEE-exact arithmetic (``+ - * /``).  The test densities
below are written that way so the comparisons can be bitwise on every device.
Batched moment/solve kernels (``bmm``, ``linalg.solve``) may pick a different
kernel for a different batch size on accelerators, so the Waldo sample-moment
comparisons are bitwise on CPU and use the float64 default tolerance elsewhere.
"""

from __future__ import annotations

import math
from typing import Callable

import pytest
import torch
from scipy import stats
from torch import Tensor

import torchregress.losses.waldo as waldo_mod
from torchregress.losses import (
    CTI,
    ConformalizedDensity,
    ContrastiveFlowConformal,
    WaldoConformal,
    WaldoDensityConformal,
    WaldoPosteriorCalibrator,
)
from torchregress.losses.waldo import _sample_moments

N_CAL = 103  # not divisible by the chunk sizes used below
CHUNK = 7


# --------------------------------------------------------------------------- #
# Row-wise test densities (rational arithmetic only)
# --------------------------------------------------------------------------- #
def _mean_scale(ctx: Tensor) -> tuple[Tensor, Tensor]:
    mean = 0.8 * ctx[:, 0:1] + 0.3 * ctx[:, 1:2] / (1.0 + ctx[:, 1:2] * ctx[:, 1:2])
    c2 = ctx[:, 2:3] * ctx[:, 2:3]
    scale = 0.3 + 0.4 * c2 / (1.0 + c2)
    return mean, scale


def _nld(ctx: Tensor, y: Tensor) -> Tensor:
    """Gaussian-like negative log-density score ``[N]`` for ``y`` of shape ``[N, D]``."""
    mean, scale = _mean_scale(ctx)
    z = (y.reshape(y.shape[0], -1) - mean) / scale
    terms = 0.5 * z * z + scale
    out = terms[:, 0]
    for j in range(1, terms.shape[1]):
        out = out + terms[:, j]
    return out


def _log_prob(ctx: Tensor, y: Tensor) -> Tensor:
    return -_nld(ctx, y)


class _Flow:
    def log_prob(self, context: Tensor, target: Tensor) -> Tensor:
        return _log_prob(context, target)


def _data(device: str, n: int, d: int, seed: int) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randn(n, 3, generator=g, dtype=torch.float64)
    mean, scale = _mean_scale(ctx)
    y = mean + scale * torch.randn(n, d, generator=g, dtype=torch.float64)
    return ctx.to(device), y.to(device)


# --------------------------------------------------------------------------- #
# Reference implementations (pre-chunking code paths)
# --------------------------------------------------------------------------- #
def _ref_density_threshold(
    log_prob: Callable[[Tensor, Tensor], Tensor],
    ctx: Tensor,
    y: Tensor,
    alpha: float,
    groups: Tensor | None = None,
) -> CTI:
    cti = CTI(alpha=alpha)
    cti.calibrate(log_prob(ctx, y).detach().reshape(-1), y.reshape(y.shape[0], -1), groups=groups)
    return cti


def _ref_density_interval(
    nld: Callable[[Tensor, Tensor], Tensor],
    cti: CTI,
    ctx: Tensor,
    y_min: float,
    y_max: float,
    grid_size: int,
    groups: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-row grid level search, as the pre-chunking implementation did it."""
    n = ctx.shape[0]
    q = cti._per_sample_q(n, groups=groups, device=ctx.device, dtype=ctx.dtype)
    y_grid = torch.linspace(y_min, y_max, grid_size, device=ctx.device, dtype=ctx.dtype)
    lower = torch.empty(n, 1, device=ctx.device, dtype=ctx.dtype)
    upper = torch.empty(n, 1, device=ctx.device, dtype=ctx.dtype)
    for i in range(n):
        values = -(-nld(ctx[i : i + 1].expand(grid_size, -1), y_grid.reshape(-1, 1)))
        if torch.isposinf(q[i]):
            lower[i, 0], upper[i, 0] = float("-inf"), float("inf")
            continue
        in_set = values <= q[i]
        if bool(in_set.any()):
            idx = in_set.nonzero(as_tuple=True)[0]
            lower[i, 0], upper[i, 0] = y_grid[idx[0]], y_grid[idx[-1]]
        else:
            lower[i, 0] = upper[i, 0] = y_grid[values.argmin()]
    return lower, upper


def _ref_waldo_interval(
    cp: WaldoConformal, mean: Tensor, std: Tensor, y_min: float, y_max: float, grid_size: int
) -> tuple[Tensor, Tensor]:
    """The pre-vectorisation per-row loop of ``WaldoConformal.predict_interval``."""
    m = mean.reshape(-1)
    s = std.reshape(-1).clamp_min(math.sqrt(cp.min_variance))
    lower = torch.empty_like(m)
    upper = torch.empty_like(m)
    for i in range(m.numel()):
        grid = torch.linspace(y_min, y_max, int(grid_size), device=m.device, dtype=m.dtype)
        score = (grid - m[i]) ** 2 / (s[i] ** 2)
        crit = cp.critical_value(grid.reshape(-1, 1))
        accepted = score <= crit
        if bool(accepted.any()):
            idx = accepted.nonzero(as_tuple=True)[0]
            lower[i] = grid[idx[0]]
            upper[i] = grid[idx[-1]]
        else:
            lower[i] = m[i]
            upper[i] = m[i]
    return lower.reshape(-1, 1), upper.reshape(-1, 1)


def _assert_equal(got: Tensor, want: Tensor) -> None:
    assert got.device == want.device
    torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True)


def _assert_moments_close(got: Tensor, want: Tensor) -> None:
    """Bitwise on CPU; batched accelerator kernels may round differently."""
    assert got.device == want.device
    if got.device.type == "cpu":
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    else:
        torch.testing.assert_close(got, want)


def _assert_membership(got: Tensor, want: Tensor, score: Tensor, crit: Tensor) -> None:
    """Exact on CPU; elsewhere a flip is allowed only where ``score == crit`` to rounding."""
    assert got.shape == want.shape and got.device == want.device
    if got.device.type == "cpu":
        _assert_equal(got, want)
        return
    near = (score - crit).abs() <= 1e-9 * (1.0 + crit.abs())
    assert bool(((got == want) | near).all())


def _pair_scores(cp: WaldoConformal, mean: Tensor, cov: Tensor, grid: Tensor) -> Tensor:
    """Wald scores and critical values ``[N, G]`` for a shared ``[G, D]`` grid."""
    n, g = mean.shape[0], grid.shape[0]
    score = cp.score(
        mean.repeat_interleave(g, dim=0),
        grid.repeat(n, 1),
        cov.repeat_interleave(g, dim=0),
    ).reshape(n, g)
    return score, cp.critical_value(grid).reshape(1, g).expand(n, g)


# --------------------------------------------------------------------------- #
# ConformalizedDensity / ContrastiveFlowConformal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("chunk_size", [CHUNK, 131_072])
def test_conformalized_density_calibrate_and_contains_match_reference(
    device: str, d: int, chunk_size: int
) -> None:
    ctx, y = _data(device, N_CAL, d, seed=0)
    test_ctx, test_y = _data(device, 61, d, seed=1)
    ref = _ref_density_threshold(_log_prob, ctx, y, alpha=0.1)

    cp = ConformalizedDensity(alpha=0.1, chunk_size=chunk_size)
    cp.calibrate(_nld, ctx, y)
    assert cp.threshold is not None
    _assert_equal(cp.threshold, ref.q_hat)

    cp_lp = ConformalizedDensity(alpha=0.1, chunk_size=chunk_size)
    cp_lp.calibrate_log_prob(_log_prob, ctx, y)
    _assert_equal(cp_lp.threshold, ref.q_hat)

    got = cp.contains(_nld, test_ctx, test_y)
    want = _nld(test_ctx, test_y) <= ref.q_hat
    assert got.dtype == torch.bool and got.shape == (61,)
    _assert_equal(got, want)


@pytest.mark.parametrize("chunk_size", [1, 3 * 37 + 5, 131_072])
def test_conformalized_density_interval_matches_per_row_reference(
    device: str, chunk_size: int
) -> None:
    grid_size = 37
    ctx, y = _data(device, N_CAL, 1, seed=2)
    test_ctx, _ = _data(device, 29, 1, seed=3)
    # Push some contexts far outside the grid so their level set is empty there.
    test_ctx[:4, 0] = 40.0
    ref = _ref_density_threshold(_log_prob, ctx, y, alpha=0.1)
    cp = ConformalizedDensity(alpha=0.1, grid_size=grid_size, chunk_size=chunk_size)
    cp.calibrate_log_prob(_log_prob, ctx, y)
    lo, hi = cp.predict_interval(_nld, test_ctx, -3.0, 3.0)
    ref_lo, ref_hi = _ref_density_interval(_nld, ref, test_ctx, -3.0, 3.0, grid_size)
    assert lo.shape == hi.shape == (29, 1)
    _assert_equal(lo, ref_lo)
    _assert_equal(hi, ref_hi)
    # Empty level sets collapse to the grid argmin (the upper grid edge here).
    assert bool((lo[:4] == 3.0).all()) and bool((hi[:4] == 3.0).all())


def test_conformalized_density_mondrian_and_infinite_threshold(device: str) -> None:
    grid_size = 21
    ctx, y = _data(device, N_CAL, 1, seed=4)
    # Group 2 has 3 calibration rows: ceil(4 * 0.9) = 4 > 3 -> infinite threshold.
    groups = (ctx[:, 0] > 0).long()
    groups[:3] = 2
    test_ctx, _ = _data(device, 17, 1, seed=5)
    test_groups = torch.tensor([0, 1, 2] * 5 + [0, 2], device=device)
    ref = _ref_density_threshold(_log_prob, ctx, y, alpha=0.1, groups=groups)
    cp = ConformalizedDensity(alpha=0.1, grid_size=grid_size, chunk_size=2 * grid_size + 3)
    cp.calibrate_log_prob(_log_prob, ctx, y, groups=groups)
    lo, hi = cp.predict_interval(_nld, test_ctx, -3.0, 3.0, groups=test_groups)
    ref_lo, ref_hi = _ref_density_interval(
        _nld, ref, test_ctx, -3.0, 3.0, grid_size, groups=test_groups
    )
    _assert_equal(lo, ref_lo)
    _assert_equal(hi, ref_hi)
    inf_rows = test_groups == 2
    assert bool(torch.isneginf(lo[inf_rows]).all()) and bool(torch.isposinf(hi[inf_rows]).all())


def test_contrastive_flow_conformal_chunked_matches_unchunked(device: str) -> None:
    ctx, y = _data(device, N_CAL, 1, seed=6)
    test_ctx, test_y = _data(device, 23, 1, seed=7)
    full = ContrastiveFlowConformal(_Flow(), alpha=0.2, grid_size=31)
    chunked = ContrastiveFlowConformal(_Flow(), alpha=0.2, grid_size=31, chunk_size=CHUNK)
    for cp in (full, chunked):
        cp.calibrate(ctx, y)
    _assert_equal(chunked.threshold, full.threshold)
    _assert_equal(chunked.contains(test_ctx, test_y), full.contains(test_ctx, test_y))
    for got, want in zip(
        chunked.predict_interval(test_ctx, -3.0, 3.0), full.predict_interval(test_ctx, -3.0, 3.0)
    ):
        _assert_equal(got, want)


def test_conformalized_density_chunks_callable_inputs(device: str) -> None:
    ctx, y = _data(device, N_CAL, 2, seed=8)
    sizes: list[int] = []

    def log_prob(c: Tensor, v: Tensor) -> Tensor:
        sizes.append(c.shape[0])
        assert c.device == ctx.device and v.device == ctx.device
        return _log_prob(c, v)

    cp = ConformalizedDensity(alpha=0.1, chunk_size=CHUNK)
    cp.calibrate_log_prob(log_prob, ctx, y)
    assert sizes == [CHUNK] * (N_CAL // CHUNK) + [N_CAL % CHUNK]
    sizes.clear()
    cp1 = ConformalizedDensity(alpha=0.1, grid_size=10, chunk_size=35)
    cp1.calibrate_log_prob(_log_prob, ctx, y[:, :1].contiguous())
    cp1.predict_interval(lambda c, v: -log_prob(c, v), ctx[:8], -2.0, 2.0)
    assert sizes == [30, 30, 20]  # 3 contexts x 10 grid points per call
    with pytest.raises(ValueError, match="chunk_size"):
        ConformalizedDensity(chunk_size=0)


# --------------------------------------------------------------------------- #
# WaldoConformal.predict_interval / predict_region
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("critical_bins", [None, 5])
def test_waldo_interval_matches_per_row_reference(
    device: str, critical_bins: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    g = torch.Generator().manual_seed(9)
    mean = torch.randn(400, 1, generator=g, dtype=torch.float64)
    std = 0.3 + torch.rand(400, 1, generator=g, dtype=torch.float64)
    target = mean + std * torch.randn(400, 1, generator=g, dtype=torch.float64)
    mean, std, target = mean.to(device), std.to(device), target.to(device)
    cp = WaldoConformal(alpha=0.1, critical_bins=critical_bins)
    cp.calibrate(mean, target, std**2)
    test_mean, test_std = mean[:53] * 2.0, std[:53]
    test_std[:3] = 1e-6  # tiny spread: acceptance set misses every grid point
    ref = _ref_waldo_interval(cp, test_mean, test_std, -4.0, 4.0, 41)
    # 41 grid points * 5 rows per chunk: 53 rows -> 11 chunks, the last one partial.
    monkeypatch.setattr(waldo_mod, "_ELEMENT_BUDGET", 41 * 5)
    got = cp.predict_interval(test_mean, test_std, y_min=-4.0, y_max=4.0, grid_size=41)
    for a, b in zip(got, ref):
        _assert_equal(a, b)
    assert bool((got[0][:3, 0] == test_mean[:3, 0]).all())
    # Automatic grid bounds follow the same path.
    auto = cp.predict_interval(test_mean, test_std, grid_size=41)
    monkeypatch.setattr(waldo_mod, "_ELEMENT_BUDGET", 1 << 22)
    for a, b in zip(auto, cp.predict_interval(test_mean, test_std, grid_size=41)):
        _assert_equal(a, b)


@pytest.mark.parametrize("covariance_type", ["diagonal", "full"])
@pytest.mark.parametrize("grid_dim", [2, 3])
def test_waldo_region_chunked_matches_unchunked(
    device: str, covariance_type: str, grid_dim: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    g = torch.Generator().manual_seed(10)
    n, d, n_grid = 31, 2, 9
    mean = torch.randn(n, d, generator=g, dtype=torch.float64)
    if covariance_type == "diagonal":
        cov = 0.5 + torch.rand(n, d, generator=g, dtype=torch.float64)
    else:
        a = torch.randn(n, d, d, generator=g, dtype=torch.float64)
        cov = a @ a.transpose(1, 2) + 0.5 * torch.eye(d, dtype=torch.float64)
    target = mean + torch.randn(n, d, generator=g, dtype=torch.float64)
    if grid_dim == 2:
        grid = torch.randn(n_grid, d, generator=g, dtype=torch.float64)
    else:
        grid = torch.randn(n, n_grid, d, generator=g, dtype=torch.float64)
    mean, cov, target, grid = (t.to(device) for t in (mean, cov, target, grid))
    cp = WaldoConformal(alpha=0.2, covariance_type=covariance_type, critical_bins=3)
    cp.calibrate(mean, target, cov)
    want = cp.predict_region(mean, grid, cov)
    per_row = n_grid * d * (d if covariance_type == "full" else 1)
    monkeypatch.setattr(waldo_mod, "_ELEMENT_BUDGET", 4 * per_row)  # 4 rows per chunk
    got = cp.predict_region(mean, grid, cov)
    assert got.shape == (n, n_grid) and got.dtype == torch.bool
    _assert_equal(got, want)


# --------------------------------------------------------------------------- #
# WaldoDensityConformal / WaldoPosteriorCalibrator
# --------------------------------------------------------------------------- #
def _det_sampler(context: Tensor, n: int) -> Tensor:
    """Deterministic row-wise 'sampler' (quasi-random offsets) -> ``[N, S, 2]``."""
    mean, scale = _mean_scale(context)
    k = torch.arange(n, device=context.device, dtype=context.dtype).reshape(1, n, 1)
    u = (k * 0.6180339887 + context[:, 2:3].unsqueeze(1) * 0.1) % 1.0 - 0.5
    v = (k * 0.7548776662 + context[:, 1:2].unsqueeze(1) * 0.1) % 1.0 - 0.5
    noise = torch.cat([u, u * 0.3 + v], dim=-1) * scale.unsqueeze(1) * 3.0
    return mean.unsqueeze(1) + noise


def _ref_waldo_density(
    sample_fn: Callable[[Tensor, int], Tensor],
    ctx: Tensor,
    target: Tensor,
    n_samples: int,
    covariance_type: str,
    critical_bins: int | None,
) -> WaldoConformal:
    mean, cov = _sample_moments(sample_fn(ctx, n_samples))
    if covariance_type == "diagonal":
        cov = cov.diagonal(dim1=-2, dim2=-1)  # per-target sample variances
    ref = WaldoConformal(alpha=0.1, critical_bins=critical_bins, covariance_type=covariance_type)
    ref.calibrate(mean, target, cov)
    return ref


@pytest.mark.parametrize("covariance_type", ["diagonal", "full"])
@pytest.mark.parametrize("critical_bins", [None, 4])
def test_waldo_density_chunked_matches_reference(
    device: str, covariance_type: str, critical_bins: int | None
) -> None:
    ctx, _ = _data(device, N_CAL, 1, seed=11)
    target = _det_sampler(ctx, 1)[:, 0] + 0.05 * ctx[:, :2]
    ref = _ref_waldo_density(_det_sampler, ctx, target, 32, covariance_type, critical_bins)
    cp = WaldoDensityConformal(
        alpha=0.1,
        critical_bins=critical_bins,
        covariance_type=covariance_type,
        n_samples=32,
        chunk_size=CHUNK,
    )
    cp.calibrate(_det_sampler, ctx, target)
    _assert_moments_close(cp.threshold, ref.q_hat)
    _assert_moments_close(cp.calibrator._cal_scores, ref._cal_scores)

    test_ctx, _ = _data(device, 40, 1, seed=12)
    test_target = _det_sampler(test_ctx, 1)[:, 0]
    mean, cov = _sample_moments(_det_sampler(test_ctx, 32))
    if covariance_type == "diagonal":
        cov = cov.diagonal(dim1=-2, dim2=-1)
    _assert_membership(
        cp.contains(_det_sampler, test_ctx, test_target),
        ref.contains(mean, test_target, cov),
        ref.score(mean, test_target, cov),
        ref.critical_value(test_target),
    )
    grid = test_target[:6]
    _assert_membership(
        cp.predict_region(_det_sampler, test_ctx, grid),
        ref.predict_region(mean, grid, cov),
        *_pair_scores(ref, mean, cov, grid),
    )
    _assert_membership(
        cp.predict_region(_det_sampler, test_ctx, test_target.unsqueeze(1)),
        ref.predict_region(mean, test_target.unsqueeze(1), cov),
        ref.score(mean, test_target, cov).unsqueeze(1),
        ref.critical_value(test_target).unsqueeze(1),
    )


def test_waldo_density_interval_chunked_matches_reference(device: str) -> None:
    def sampler_1d(context: Tensor, n: int) -> Tensor:
        return _det_sampler(context, n)[..., :1]

    ctx, _ = _data(device, N_CAL, 1, seed=13)
    target = sampler_1d(ctx, 1)[:, 0]
    cp = WaldoDensityConformal(alpha=0.1, covariance_type="diagonal", n_samples=16, chunk_size=5)
    cp.calibrate(sampler_1d, ctx, target)
    test_ctx, _ = _data(device, 23, 1, seed=14)
    samples = sampler_1d(test_ctx, 16)
    ref = _ref_waldo_interval(cp.calibrator, samples.mean(dim=1), samples.std(dim=1), -3.0, 3.0, 25)
    got = cp.predict_interval(sampler_1d, test_ctx, y_min=-3.0, y_max=3.0, grid_size=25)
    for a, b in zip(got, ref):
        _assert_moments_close(a, b)


def test_waldo_density_random_sampler_single_chunk_is_bit_identical(device: str) -> None:
    """With ``N <= chunk`` the sampler is called once, exactly as before."""

    def sample_fn(context: Tensor, n: int) -> Tensor:
        noise = torch.randn(context.shape[0], n, 2, device=context.device, dtype=context.dtype)
        return context[:, :2].unsqueeze(1) + 0.5 * noise

    ctx, _ = _data(device, N_CAL, 1, seed=15)
    target = ctx[:, :2] + 0.5 * torch.randn(N_CAL, 2, device=device, dtype=ctx.dtype)
    torch.manual_seed(123)
    ref = _ref_waldo_density(sample_fn, ctx, target, 64, "full", None)
    torch.manual_seed(123)
    cp = WaldoDensityConformal(alpha=0.1, covariance_type="full", n_samples=64)
    cp.calibrate(sample_fn, ctx, target)
    _assert_equal(cp.threshold, ref.q_hat)
    _assert_equal(cp.calibrator._cal_scores, ref._cal_scores)


def test_waldo_density_random_sampler_chunked_is_statistically_equivalent(device: str) -> None:
    """A ``[S, N, D]``-order sampler draws differently per chunk; the law is unchanged."""

    def sample_fn(context: Tensor, n: int) -> Tensor:
        noise = torch.randn(n, context.shape[0], 2, device=context.device, dtype=context.dtype)
        return context[:, :2].unsqueeze(1) + 0.5 * noise.transpose(0, 1)

    n = 3000
    g = torch.Generator().manual_seed(16)
    ctx = torch.randn(n, 3, generator=g, dtype=torch.float64).to(device)
    target = ctx[:, :2] + 0.5 * torch.randn(n, 2, generator=g, dtype=torch.float64).to(device)
    torch.manual_seed(321)
    full = WaldoDensityConformal(alpha=0.1, covariance_type="full", n_samples=64)
    full.calibrate(sample_fn, ctx, target)
    torch.manual_seed(321)
    chunked = WaldoDensityConformal(alpha=0.1, covariance_type="full", n_samples=64, chunk_size=97)
    chunked.calibrate(sample_fn, ctx, target)
    s_full = full.calibrator._cal_scores
    s_chunk = chunked.calibrator._cal_scores
    assert s_full is not None and s_chunk is not None
    assert not torch.equal(s_full, s_chunk)  # different draws ...
    ks = stats.ks_2samp(s_full.cpu().numpy(), s_chunk.cpu().numpy())
    assert ks.pvalue > 1e-3  # ... from the same score distribution
    # Exact moments give tau ~ chi2(2) scaled by S/(S-1)-type MC noise; both
    # thresholds estimate the same 0.9 quantile.
    q_exact = stats.chi2.ppf(0.9, df=2)
    for cp in (full, chunked):
        assert cp.threshold is not None
        assert abs(float(cp.threshold) / q_exact - 1.0) < 0.15


def test_waldo_posterior_chunked_matches_reference(device: str) -> None:
    g = torch.Generator().manual_seed(17)
    n, s, d = N_CAL, 24, 3
    samples = (
        torch.randn(n, s, d, generator=g, dtype=torch.float64) * 0.5
        + torch.randn(n, 1, d, generator=g, dtype=torch.float64)
    ).to(device)
    target = samples.mean(dim=1) + 0.5 * torch.randn(n, d, generator=g, dtype=torch.float64).to(
        device
    )
    mean, cov = _sample_moments(samples)
    ref = WaldoConformal(alpha=0.1, covariance_type="full", critical_bins=3)
    ref.calibrate(mean, target, cov)
    cp = WaldoPosteriorCalibrator(
        alpha=0.1, covariance_type="full", critical_bins=3, chunk_size=CHUNK
    )
    cp.calibrate(samples, target)
    _assert_moments_close(cp.threshold, ref.q_hat)
    _assert_moments_close(cp.calibrator._cal_scores, ref._cal_scores)
    grid = target[:5]
    _assert_membership(
        cp.contains(samples, target),
        ref.contains(mean, target, cov),
        ref.score(mean, target, cov),
        ref.critical_value(target),
    )
    _assert_membership(
        cp.predict_region(samples, grid),
        ref.predict_region(mean, grid, cov),
        *_pair_scores(ref, mean, cov, grid),
    )
    with pytest.raises(ValueError, match="chunk_size"):
        WaldoPosteriorCalibrator(chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
        WaldoDensityConformal(chunk_size=-1)
