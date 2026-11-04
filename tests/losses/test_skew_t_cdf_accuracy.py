"""Accuracy and sync-freedom of the skew-t Student-t CDF.

``_student_t_log_cdf_vec`` (the ``T_{nu+1}`` factor of the skew-t density)
evaluates a continued fraction for the regularised incomplete beta. The
early-exit Lentz loop ``_betacf`` tests convergence with ``bool(...)`` every
term, which is a device-to-host sync on accelerators (and blocks CUDA-graph
capture). Off the CPU it now uses ``_betacf_fixed`` (same continued fraction,
fixed depth, tree-reduced convergents); on the CPU the early-exit loop is kept
(``_cf_fixed_depth``). Both paths are exercised here on whatever device runs
the suite. These tests pin:

* the log-CDF against :func:`scipy.stats.t.logcdf` in float64 over a wide grid
  of ``x`` and ``df`` (``df`` from ``1 + 1e-6`` to ``1e3``), including the
  centre/tail branch switch, where the continued fraction converges slowest;
* agreement of the fixed-depth and early-exit evaluations (values and
  gradients) up to ``df = 1e6``;
* gradients against closed forms (``d log T / dx = t_df(x) / T(x)``) and
  finite differences of scipy (``d log T / d df``);
* skew-t NLL values against the scipy closed form and NLL gradients in all four
  heads, fixed depth vs early exit;
* absence of host synchronisation (meta-device forward/backward, and CUDA sync
  debug mode when the suite runs on CUDA).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest
import scipy.stats as st
import torch

import torchregress.losses.families as fam
from torchregress.losses import SkewTLoss, skew_t_nll

F64 = torch.float64

_DFS = [1.0 + 1e-6, 1.0001, 1.1, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1e3]
_PATHS = ["fixed", "lentz"]


@contextmanager
def _cf_path(monkeypatch: pytest.MonkeyPatch, path: str) -> Iterator[None]:
    """Force the continued-fraction evaluation regardless of device."""
    with monkeypatch.context() as m:
        m.setattr(fam, "_cf_fixed_depth", lambda device: path == "fixed")
        yield


def _x_grid(df: float) -> np.ndarray:
    """Dense centre, log-spaced tails, and points around the branch switch."""
    half = 0.5 * df
    u_switch = 1.5 / (half + 2.5)
    x_switch = math.sqrt(df * u_switch / (1.0 - u_switch))
    around = x_switch * (1.0 + np.linspace(-1e-2, 1e-2, 201))
    return np.concatenate(
        [
            np.linspace(-30.0, 30.0, 1201),
            np.logspace(-8, 6, 281),
            -np.logspace(-8, 6, 281),
            around,
            -around,
            [0.0],
        ]
    )


def test_cf_fixed_depth_default_dispatch() -> None:
    assert fam._cf_fixed_depth(torch.device("cpu")) is False
    assert fam._cf_fixed_depth(torch.device("meta")) is True
    assert fam._cf_fixed_depth(torch.device("cuda")) is True


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("df", _DFS)
def test_student_t_log_cdf_matches_scipy_wide_grid(
    df: float, path: str, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = _x_grid(df)
    ref = st.t.logcdf(x, df)
    keep = ref > -700.0  # scipy's own log-CDF is exact to float64 here
    xt = torch.from_numpy(x[keep]).to(device=device)
    with _cf_path(monkeypatch, path):
        got = fam._student_t_log_cdf_vec(xt, torch.full_like(xt, df))
    torch.testing.assert_close(got.cpu(), torch.from_numpy(ref[keep]), rtol=1e-10, atol=1e-14)


# For df >= 1e4 both evaluations carry the same ~eps * df float64 error from the
# lgamma differences of the normaliser (each is within 9e-10 relative of an
# mpmath 40-digit reference at df = 1e6), so they agree to that level only.
@pytest.mark.parametrize(
    ("df", "rtol"),
    [*((df, 1e-12) for df in _DFS), (1e4, 1e-10), (1e5, 1e-9), (1e6, 1e-9)],
)
def test_student_t_log_cdf_fixed_depth_matches_early_exit_lentz(
    df: float, rtol: float, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = torch.from_numpy(_x_grid(df)).to(device=device)
    dft = torch.full_like(x, df)
    vals, gx, gdf = {}, {}, {}
    for path in _PATHS:
        xg = x.clone().requires_grad_(True)
        dg = dft.clone().requires_grad_(True)
        with _cf_path(monkeypatch, path):
            out = fam._student_t_log_cdf_vec(xg, dg)
        out.sum().backward()
        assert xg.grad is not None and dg.grad is not None
        vals[path], gx[path], gdf[path] = out.detach(), xg.grad, dg.grad
    torch.testing.assert_close(vals["fixed"], vals["lentz"], rtol=rtol, atol=1e-300)
    torch.testing.assert_close(gx["fixed"], gx["lentz"], rtol=100 * rtol, atol=1e-300)
    # d log T / d df is O(1/df^2) at large df; its float64 roundoff is absolute.
    torch.testing.assert_close(gdf["fixed"], gdf["lentz"], rtol=1e4 * rtol, atol=1e-4 * rtol)


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("df", _DFS)
def test_student_t_log_cdf_x_gradient_closed_form(
    df: float, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = _x_grid(df)
    x = x[st.t.logcdf(x, df) > -700.0]
    xg = torch.from_numpy(x).requires_grad_(True)
    with _cf_path(monkeypatch, path):
        fam._student_t_log_cdf_vec(xg, torch.full_like(xg, df)).sum().backward()
    assert xg.grad is not None
    ref = np.exp(st.t.logpdf(x, df) - st.t.logcdf(x, df))
    torch.testing.assert_close(xg.grad, torch.from_numpy(ref), rtol=1e-9, atol=1e-300)


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("df", [1.0 + 1e-3, 1.5, 2.0, 5.0, 30.0, 300.0, 1e3])
def test_student_t_log_cdf_df_gradient_finite_differences(
    df: float, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    x = np.array([-1e3, -40.0, -6.0, -2.0, -0.7, -1e-3, 0.0, 1e-3, 0.7, 2.0, 6.0, 40.0])
    x = x[st.t.logcdf(x, df) > -700.0]
    dg = torch.full((len(x),), df, dtype=F64, requires_grad=True)
    with _cf_path(monkeypatch, path):
        fam._student_t_log_cdf_vec(torch.from_numpy(x), dg).sum().backward()
    assert dg.grad is not None
    h = 1e-5 * df
    fd = (st.t.logcdf(x, df + h) - st.t.logcdf(x, df - h)) / (2.0 * h)
    torch.testing.assert_close(dg.grad, torch.from_numpy(fd), rtol=1e-5, atol=1e-11)


def _inv_softplus(v: float) -> float:
    return v + math.log(-math.expm1(-v))


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("alpha", [-4.0, -0.5, 0.0, 1.0, 6.0])
@pytest.mark.parametrize("nu", [0.05, 1.0, 4.0, 40.0, 1e3])
def test_skew_t_nll_matches_scipy_closed_form(
    alpha: float, nu: float, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    xi, omega = 0.3, 0.8
    y = np.concatenate([np.linspace(-20.0, 20.0, 81), [xi]])
    raw = torch.tensor([[xi, _inv_softplus(omega), alpha, _inv_softplus(nu)]], dtype=F64)
    with _cf_path(monkeypatch, path):
        got = skew_t_nll(raw.expand(len(y), 4), torch.from_numpy(y), eps=0.0, reduction="none")
    z = (y - xi) / omega
    w = alpha * z * np.sqrt((nu + 1.0) / (nu + z * z))
    ref = -(math.log(2.0) + st.t.logpdf(z, nu) - math.log(omega) + st.t.logcdf(w, nu + 1.0))
    finite = np.isfinite(ref)  # scipy's logcdf underflows to -inf in the far tail
    assert finite.sum() > 40
    torch.testing.assert_close(got[finite], torch.from_numpy(ref[finite]), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_skew_t_nll_gradients_fixed_depth_match_early_exit_lentz(
    dtype: torch.dtype, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    gen = torch.Generator().manual_seed(0)
    n = 512
    raw = torch.randn(n, 4, generator=gen, dtype=F64) * torch.tensor(
        [1.0, 1.0, 3.0, 3.0], dtype=F64
    )
    raw[:, 3] += torch.linspace(-3.0, 8.0, n, dtype=F64)  # nu from ~0.05 to ~3e3
    y = torch.randn(n, generator=gen, dtype=F64) * 4.0
    raw, y = raw.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype)

    losses, grads = {}, {}
    for path in _PATHS:
        p = raw.clone().requires_grad_(True)
        with _cf_path(monkeypatch, path):
            loss = skew_t_nll(p, y, reduction="none")
        loss.sum().backward()
        assert p.grad is not None
        losses[path], grads[path] = loss.detach(), p.grad
    tol = 1e-10 if dtype == F64 else 1e-6
    torch.testing.assert_close(losses["fixed"], losses["lentz"], rtol=tol, atol=tol)
    torch.testing.assert_close(grads["fixed"], grads["lentz"], rtol=tol, atol=tol)


def test_skew_t_nll_has_no_host_sync_meta_device() -> None:
    """Value-dependent Python control flow (``bool``/``.item()``) raises on meta tensors."""
    raw = torch.zeros(64, 4, device="meta", requires_grad=True)
    y = torch.zeros(64, device="meta")
    SkewTLoss()(raw, y).backward()
    assert raw.grad is not None and raw.grad.shape == raw.shape


def test_skew_t_nll_has_no_host_sync_on_device(device: str) -> None:
    gen = torch.Generator().manual_seed(1)
    raw = torch.randn(256, 4, generator=gen).to(device=device).requires_grad_(True)
    y = torch.randn(256, generator=gen).to(device=device)
    loss_fn = SkewTLoss()
    loss_fn(raw, y).backward()  # warm-up (lazy init)
    if torch.device(device).type == "cuda":
        previous = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            loss_fn(raw, y).backward()
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    else:
        loss_fn(raw, y).backward()
    assert raw.grad is not None and bool(torch.isfinite(raw.grad).all())


def test_betacf_fixed_chunking_matches_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = torch.Generator().manual_seed(2)
    a = 0.5 + 50.0 * torch.rand(3, 7, generator=gen, dtype=F64)
    b = torch.full((1, 7), 0.5, dtype=F64)
    x = torch.rand(3, 1, generator=gen, dtype=F64) * (a + 1.0) / (a + b + 2.0)
    whole = fam._betacf_fixed(a, b, x)
    with monkeypatch.context() as m:
        m.setattr(fam, "_CF_CHUNK", 4)
        chunked = fam._betacf_fixed(a, b, x)
    torch.testing.assert_close(chunked, whole, rtol=0.0, atol=0.0)
    torch.testing.assert_close(whole, fam._betacf(a, b, x), rtol=1e-12, atol=0.0)
