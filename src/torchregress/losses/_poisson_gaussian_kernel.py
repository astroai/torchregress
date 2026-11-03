"""Finite-support kernel for the Poisson--Gaussian convolution likelihood.

Detector model: ``K ~ Poisson(rate)``, ``y = gain * K + offset + eps`` with
``eps ~ N(0, read_var)``, so

    p(y) = sum_k Pois(k; rate) N(y; gain * k + offset, read_var).

The sum is evaluated on a per-element count window that retains at least
``tail_sigma`` standard deviations of both the Poisson prior and the
observation-conditioned posterior over ``k``. Ported from torchsky's
``poisson_gaussian_nll`` (``torchsky.map.operators.noise``); torchregress does
not import torchsky. Differences from that implementation are listed in the
docstring of :func:`poisson_gaussian_nll`.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import Tensor

# Pixel x count terms per chunk; peak memory is independent of batch size.
_CHUNK_TERMS = 1 << 22
# Bisection brackets the posterior mode to within this many counts. The support
# is taken from the bracket ends, so the window bound stays rigorous.
_MODE_TOLERANCE = 0.25
_LOG_2PI = math.log(2.0 * math.pi)


def _support_window(
    value: Tensor,
    rate: Tensor,
    read_var: Tensor,
    gain: Tensor,
    offset: Tensor,
    *,
    tail_sigma: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-element inclusive count window ``(lower, upper)`` and a ``far`` flag.

    The prior window is ``rate +- (t sqrt(rate + 1) + t^2)``. The
    observation-conditioned weight ``Pois(n; rate) N(value; gain n + offset,
    read_var)`` is log-concave in ``n``, so its maximiser lies between the
    Poisson mode and the Gaussian centre ``(value - offset) / gain`` and is
    located by bisection on the monotone score. The posterior standard
    deviation is at most ``sqrt(1 / (1 / (m + 1) + gain^2 / read_var))``, so
    the same ``t`` radius around the mode keeps the posterior tail beyond
    ``t`` sigmas. Overlapping windows are merged. A posterior window disjoint
    from the prior window is used on its own and flagged ``far``; its terms use
    the exact, unrenormalised Poisson log pmf.
    """
    dtype = value.dtype
    eps = torch.finfo(dtype).eps
    tail = float(tail_sigma)
    rate = rate.detach()
    value, read_var, gain, offset = (t.detach() for t in (value, read_var, gain, offset))
    radius = tail * torch.sqrt(rate + 1.0) + tail**2
    plo = torch.floor((rate - radius).clamp_min(0.0))
    phi = torch.ceil(rate + radius)
    positive = rate > 0
    cap = 2.0 / eps
    centre = (value - offset) / gain
    lo = (torch.minimum(rate, centre) - 1.0).clamp_min(0.0)
    hi = torch.maximum(rate, centre).clamp(0.0, cap)
    lo = torch.minimum(lo, hi)
    log_rate = torch.where(positive, rate, 1.0).log()
    slope = gain / read_var
    resid = value - offset
    active = positive & ((hi - lo) > _MODE_TOLERANCE)
    if bool(active.any()):
        idx = active.nonzero(as_tuple=True)
        a, b = lo[idx], hi[idx]
        lr, sl, rs, g = log_rate[idx], slope[idx], resid[idx], gain[idx]

        def score_at(x: Tensor) -> Tensor:
            return lr - torch.digamma(x + 1.0) + sl * (rs - g * x)

        # The score is monotone decreasing: a boundary root is located exactly.
        a = torch.where(score_at(b) >= 0, b, a)
        b = torch.where(score_at(a) <= 0, a, b)
        span = max(float((b - a).max()), _MODE_TOLERANCE)
        for _ in range(int(math.ceil(math.log2(span / _MODE_TOLERANCE)))):
            mid = 0.5 * (a + b)
            up = score_at(mid) > 0
            a = torch.where(up, mid, a)
            b = torch.where(up, b, mid)
        lo = lo.clone()
        hi = hi.clone()
        lo[idx], hi[idx] = a, b
    # Posterior variance is largest at the upper bracket end (psi'(x) >= 1/x).
    post_var = 1.0 / (1.0 / (hi + 1.0) + gain.square() / read_var)
    post_radius = tail * torch.sqrt(post_var + 1.0) + tail**2
    mlo = torch.floor((lo - post_radius).clamp_min(0.0))
    mhi = torch.ceil(hi + post_radius)
    above = positive & (mlo > phi + 1.0)
    below = positive & (mhi < plo - 1.0)
    far = above | below
    lower = torch.where(positive, torch.where(far, mlo, torch.minimum(plo, mlo)), plo)
    upper = torch.where(positive, torch.where(far, mhi, torch.maximum(phi, mhi)), phi)
    top = torch.maximum(upper, phi)
    if not bool(torch.isfinite(top).all()) or bool((top > cap).any()):
        raise ValueError(
            "Poisson-Gaussian support exceeds the exact integer range of the dtype; "
            "use float64 or the Gaussian limit at such rates"
        )
    return lower, upper, far


def _mode_anchored_log_weights(rate: Tensor, counts_i: Tensor, included: Tensor) -> Tensor:
    """Window-normalised Poisson log weights as cumulative log-ratios from the mode.

    ``log p(n) - log p(m)`` is a sum of ``log1p`` ratios whose error scales with
    the distance from the mode rather than with ``lgamma(n)``, which keeps
    float32 accurate at large rates. Values only (no autograd use).
    """
    counts = counts_i.to(rate.dtype)
    mode = rate.floor().unsqueeze(-1)
    positive = rate.unsqueeze(-1) > 0
    lam = torch.where(positive, rate.unsqueeze(-1), 1.0)
    above = counts > mode
    below = counts < mode
    safe_c = counts.clamp_min(1.0)
    near = (lam - safe_c).abs() < 0.5 * safe_c
    right_term = torch.where(near, torch.log1p((lam - safe_c) / safe_c), lam.log() - safe_c.log())
    next_c = counts + 1.0
    near_l = (next_c - lam).abs() < 0.5 * lam
    left_term = torch.where(near_l, torch.log1p((next_c - lam) / lam), next_c.log() - lam.log())
    right = torch.where(above, right_term, 0.0).cumsum(-1)
    left = torch.where(below, left_term, 0.0).flip(-1).cumsum(-1).flip(-1)
    log_w = (right + left).masked_fill(~included, -torch.inf)
    # A zero-rate row is the point mass at count zero.
    point = torch.where(counts_i == 0, 0.0, -torch.inf).to(rate.dtype)
    log_w = torch.where(positive, log_w, point.masked_fill(~included, -torch.inf))
    return log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)


def _count_grid(lower: Tensor, upper: Tensor, max_terms: int) -> tuple[Tensor, Tensor]:
    width = int((upper - lower + 1).max().item())
    if width > int(max_terms):
        raise ValueError(
            f"Poisson-Gaussian support requires {width} count terms, exceeding "
            f"max_terms={int(max_terms)}; raise max_terms or use the Gaussian "
            "limit N(gain*rate + offset, gain^2*rate + read_var) at such rates"
        )
    offsets = torch.arange(width, device=lower.device, dtype=torch.int64)
    counts_i = lower.to(torch.int64).unsqueeze(-1) + offsets
    included = counts_i <= upper.to(torch.int64).unsqueeze(-1)
    return counts_i, included


def _gaussian_log(obs: Tensor, var: Tensor, gain: Tensor, offset: Tensor, counts: Tensor) -> Tensor:
    mean = gain.unsqueeze(-1) * counts + offset.unsqueeze(-1)
    z = (obs.unsqueeze(-1) - mean) / var.sqrt().unsqueeze(-1)
    return -0.5 * (var.log().unsqueeze(-1) + _LOG_2PI + z.square())


def _rows_moments(
    obs: Tensor,
    rate: Tensor,
    var: Tensor,
    gain: Tensor,
    offset: Tensor,
    lower: Tensor,
    upper: Tensor,
    *,
    far: bool,
    max_terms: int,
) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """NLL and analytic gradients ``(y, rate, var, gain, offset)`` per row.

    With posterior ``q_n`` over the retained counts and ``r_n = y - g n - b``:
    ``dy = E[r]/s``, ``db = -dy``, ``dg = (g Var(n) - E[n] E[r]) / s``,
    ``ds = 1/(2 s) - (E[r]^2 + g^2 Var(n)) / (2 s^2)`` (``s`` the read
    variance), and ``drate = -(E_q[n] - E_prior[n]) / rate`` for the
    renormalised window (``1 - E_q[n] / rate`` for unrenormalised far rows).
    Moments are taken relative to ``lower`` to avoid cancellation.
    """
    counts_i, included = _count_grid(lower, upper, max_terms)
    counts = counts_i.to(obs.dtype)
    if far:
        safe = torch.where(rate > 0, rate, 1.0).unsqueeze(-1)
        lw = counts * safe.log() - rate.unsqueeze(-1) - torch.lgamma(counts + 1.0)
        lw = lw.masked_fill(~included, -torch.inf)
    else:
        lw = _mode_anchored_log_weights(rate, counts_i, included)
    joint = lw + _gaussian_log(obs, var, gain, offset, counts)
    lse = torch.logsumexp(joint, dim=-1)
    post = (joint - lse.unsqueeze(-1)).exp()
    c = (counts_i - lower.to(torch.int64).unsqueeze(-1)).to(obs.dtype)
    cm = (post * c).sum(-1)
    var_n = (post * (c - cm.unsqueeze(-1)).square()).sum(-1)
    nbar = lower + cm
    mr = (obs - offset - gain * lower) - gain * cm
    dy = mr / var
    d_gain = (gain * var_n - nbar * mr) / var
    d_var = 0.5 / var - (mr.square() + gain.square() * var_n) / (2.0 * var.square())
    safe_rate = torch.where(rate > 0, rate, 1.0)
    if far:
        d_rate = 1.0 - nbar / safe_rate
    else:
        prior_cm = (lw.exp() * c).sum(-1)
        d_rate = -(cm - prior_cm) / safe_rate
    return -lse, (dy, d_rate, d_var, d_gain, -dy)


def _rows_autograd(
    obs: Tensor,
    rate: Tensor,
    var: Tensor,
    gain: Tensor,
    offset: Tensor,
    lower: Tensor,
    upper: Tensor,
    far: Tensor,
    *,
    max_terms: int,
) -> Tensor:
    """The same retained law as a plain differentiable log-sum-exp.

    Used for double backward. Near rows renormalise over the window, far rows
    use the exact pmf. Zero-rate rows give the exact value but no rate
    derivatives (the first-order right derivative is supplied by the moment
    path only).
    """
    counts_i, included = _count_grid(lower, upper, max_terms)
    counts = counts_i.to(obs.dtype)
    zero = (rate == 0).unsqueeze(-1)
    safe = torch.where(zero, 1.0, rate.unsqueeze(-1))
    lw = counts * safe.log() - safe - torch.lgamma(counts + 1.0)
    point = torch.where(counts_i == 0, 0.0, -torch.inf).to(obs.dtype)
    lw = torch.where(zero, point, lw).masked_fill(~included, -torch.inf)
    normalised = lw - torch.logsumexp(lw, dim=-1, keepdim=True)
    lw = torch.where(far.unsqueeze(-1), lw, normalised)
    return -torch.logsumexp(lw + _gaussian_log(obs, var, gain, offset, counts), dim=-1)


def _chunk_size(rate: Tensor, tail_sigma: float) -> int:
    """Rows per chunk from the widest prior window (merged-window slack included)."""
    if rate.numel() == 0:
        return 1
    radius = float(tail_sigma) * math.sqrt(float(rate.detach().max()) + 1.0)
    terms = 3.0 * (radius + float(tail_sigma) ** 2) + 2.0
    return max(1, int(_CHUNK_TERMS // terms))


def _forward_moments(
    flat: tuple[Tensor, ...], need: tuple[bool, ...], tail_sigma: float, max_terms: int
) -> tuple[Tensor, list[Optional[Tensor]]]:
    obs = flat[0]
    count = obs.numel()
    out = obs.new_empty(count)
    grads: list[Optional[Tensor]] = [obs.new_empty(count) if n else None for n in need]
    chunk = _chunk_size(flat[1], tail_sigma)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        part = tuple(t[start:stop] for t in flat)
        lower, upper, far = _support_window(*part, tail_sigma=tail_sigma)
        size = stop - start
        c_out = obs.new_empty(size)
        c_grads = [obs.new_empty(size) for _ in range(5)]
        for mask, is_far in ((~far, False), (far, True)):
            idx = mask.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            args = tuple(t[idx] for t in (*part, lower, upper))
            value, gs = _rows_moments(*args, far=is_far, max_terms=max_terms)
            c_out[idx] = value
            for dst, src in zip(c_grads, gs):
                dst[idx] = src
        zero = part[1] == 0
        if need[1] and bool(zero.any()):
            # Exact right derivative in the rate at zero rate.
            o, _, v, g, b = part
            score = -torch.expm1(g * ((o - b) - 0.5 * g) / v)
            if not bool(torch.isfinite(score[zero]).all()):
                raise ValueError(
                    "zero-rate derivative exceeds the dtype range; the value is "
                    "available without rate gradients"
                )
            c_grads[1] = torch.where(zero, score, c_grads[1])
        out[start:stop] = c_out
        for dst, src in zip(grads, c_grads):
            if dst is not None:
                dst[start:stop] = src
    return out, grads


def _forward_autograd(flat: tuple[Tensor, ...], tail_sigma: float, max_terms: int) -> Tensor:
    obs = flat[0]
    chunk = _chunk_size(flat[1], tail_sigma)
    pieces = []
    for start in range(0, obs.numel(), chunk):
        part = tuple(t[start : start + chunk] for t in flat)
        lower, upper, far = _support_window(*part, tail_sigma=tail_sigma)
        pieces.append(_rows_autograd(*part, lower, upper, far, max_terms=max_terms))
    return torch.cat(pieces) if pieces else obs.new_empty(0)


class _PoissonGaussianNLL(torch.autograd.Function):
    """NLL with analytic first derivatives from posterior count moments.

    The forward sweep saves only O(elements) tensors. Double backward
    (``create_graph=True``) recomputes through a differentiable log-sum-exp of
    the same retained law, which keeps the per-count graph.
    """

    @staticmethod
    def forward(ctx: Any, tail_sigma: float, max_terms: int, *flat: Tensor) -> Tensor:
        need = tuple(bool(n) for n in ctx.needs_input_grad[2:])
        out, grads = _forward_moments(flat, need, tail_sigma, max_terms)
        ctx.tail_sigma, ctx.max_terms, ctx.need = tail_sigma, max_terms, need
        ctx.n_inputs = len(flat)
        ctx.save_for_backward(*flat, *[g for g in grads if g is not None])
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_out = grad_outputs[0]
        saved = ctx.saved_tensors
        flat, stored = saved[: ctx.n_inputs], saved[ctx.n_inputs :]
        if torch.is_grad_enabled():
            with torch.enable_grad():
                out = _forward_autograd(tuple(flat), ctx.tail_sigma, ctx.max_terms)
                wrt = [t for t, n in zip(flat, ctx.need) if n]
                got = iter(
                    torch.autograd.grad(out, wrt, grad_out, create_graph=True, allow_unused=True)
                )
            result = [next(got) if n else None for n in ctx.need]
        else:
            it = iter(stored)
            result = [grad_out * next(it) if n else None for n in ctx.need]
        return (None, None, *result)


def poisson_gaussian_nll(
    target: Tensor,
    rate: Tensor,
    read_noise_variance: Tensor | float,
    gain: Tensor | float = 1.0,
    offset: Tensor | float = 0.0,
    *,
    tail_sigma: float = 12.0,
    max_terms: int = 4096,
) -> Tensor:
    """Elementwise NLL of the Poisson--Gaussian convolution (no reduction).

    Port of torchsky's ``poisson_gaussian_nll``: same support window
    (prior plus observation-conditioned window, far-outlier rows scored with
    the exact unrenormalised pmf), explicit renormalisation of near rows,
    chunking over ``2**22`` count terms, and analytic streaming first
    derivatives from posterior moments. Dropped here: the modal-polynomial
    double-backward path (replaced by a plain log-sum-exp, which is less
    accurate for curvature at tiny rates in float32), the zero-rate second-order
    jet, forward-mode AD, the CDF, and activation checkpointing of the
    double-backward graph.
    """
    tail = float(tail_sigma)
    if not math.isfinite(tail) or tail <= 0.0:
        raise ValueError("tail_sigma must be finite and positive")
    if int(max_terms) < 2:
        raise ValueError("max_terms must be at least 2")
    if target.dtype not in (torch.float32, torch.float64):
        raise TypeError("target must be float32 or float64")
    # ``as_tensor(dtype=...)`` directly: a Python float must not pass through float32.
    params = [
        torch.as_tensor(p, dtype=target.dtype, device=target.device)
        for p in (rate, read_noise_variance, gain, offset)
    ]
    tensors = torch.broadcast_tensors(target, *params)
    obs, rate_t, var_t, gain_t, offset_t = tensors
    if not bool(torch.isfinite(obs).all()):
        raise ValueError("target must be finite (mask invalid entries)")
    if not bool((torch.isfinite(rate_t) & (rate_t >= 0)).all()):
        raise ValueError("rate must be finite and non-negative")
    if not bool((torch.isfinite(var_t) & (var_t > 0)).all()):
        raise ValueError("read_noise_variance must be finite and positive")
    if not bool((torch.isfinite(gain_t) & (gain_t > 0)).all()):
        raise ValueError("gain must be finite and positive")
    if not bool(torch.isfinite(offset_t).all()):
        raise ValueError("offset must be finite")
    shape = obs.shape
    flat = tuple(t.reshape(-1) for t in tensors)
    nll = _PoissonGaussianNLL.apply(tail, int(max_terms), *flat)
    return nll.reshape(shape)
