"""Prediction-powered inference utilities.

These helpers provide practical confidence intervals that combine:
- a small labeled set with trusted outcomes, and
- a larger unlabeled set with model predictions.

The implementation is intentionally lightweight and frequentist-first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor


@dataclass(frozen=True)
class PPIConfig:
    """Configuration for Prediction-Powered Inference.

    Attributes:
        alpha: Target error rate (e.g., 0.1 for 90% confidence).
        method: Method to compute CI (default: "bootstrap").
        n_boot: Number of bootstrap samples. Ignored by :func:`ppi_quantile_ci`,
            whose interval comes from inverting the rectified CDF.
        seed: Random seed for reproducibility.

    Notes:
        Bootstrap means are computed in chunks, so memory stays bounded
        (O(N) rather than O(n_boot * N)). For samples larger than
        ``_BOOTSTRAP_CLT_THRESHOLD`` (100,000) points, the bootstrap sample mean of
        that component is drawn from its normal (CLT) approximation
        ``N(mean, var / N)``, which is what the bootstrap converges to at that size
        and is what Angelopoulos et al. (2023) use for the unlabeled term.
    """

    alpha: float = 0.1
    method: str = "bootstrap"
    n_boot: int = 2000
    seed: int | None = None


# Above this many points the bootstrap distribution of a sample mean is replaced by
# its normal (CLT) approximation; below it the exact bootstrap is run in chunks.
_BOOTSTRAP_CLT_THRESHOLD = 100_000
# Maximum number of resampled elements materialised at once by the chunked bootstrap.
_BOOTSTRAP_CHUNK_ELEMENTS = 2**22


def _to_float_tensor(x: Tensor | list[float]) -> Tensor:
    """Detached tensor keeping a floating input dtype (float64 stays float64)."""
    t = x.detach() if isinstance(x, Tensor) else torch.as_tensor(x)
    if not t.is_floating_point():
        t = t.to(torch.get_default_dtype())
    return t


def _to_1d_tensor(x: Tensor | list[float]) -> Tensor:
    return _to_float_tensor(x).reshape(-1)


def _common_float(*tensors: Tensor) -> tuple[Tensor, ...]:
    """Cast tensors to their promoted floating dtype."""
    dtype = tensors[0].dtype
    for t in tensors[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    return tuple(t.to(dtype) for t in tensors)


def _bootstrap_indices(
    n: int,
    *,
    n_boot: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    return torch.randint(low=0, high=n, size=(n_boot, n), device=device, generator=generator)


def _bootstrap_means(
    v: Tensor,
    *,
    n_boot: int,
    generator: torch.Generator | None,
) -> Tensor:
    """Bootstrap distribution of ``v.mean()`` with bounded memory.

    For ``v.numel() <= _BOOTSTRAP_CLT_THRESHOLD`` the nonparametric bootstrap is
    computed exactly, in chunks of at most ``_BOOTSTRAP_CHUNK_ELEMENTS`` resampled
    elements. Larger samples use the CLT draw ``mean + std / sqrt(n) * Z``.
    """
    n = v.numel()
    if n > _BOOTSTRAP_CLT_THRESHOLD:
        z = torch.randn(n_boot, device=v.device, dtype=v.dtype, generator=generator)
        return v.mean() + v.std(unbiased=False) / math.sqrt(n) * z
    chunk = max(1, _BOOTSTRAP_CHUNK_ELEMENTS // max(n, 1))
    out = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = _bootstrap_indices(n, n_boot=b, device=v.device, generator=generator)
        out.append(v[idx].mean(dim=1))
    return torch.cat(out)


def _percentile_ci(samples: Tensor, alpha: float) -> tuple[float, float]:
    lo = float(torch.quantile(samples, alpha / 2).item())
    hi = float(torch.quantile(samples, 1.0 - alpha / 2).item())
    return lo, hi


def _rectified_mean_point(y_l: Tensor, p_l: Tensor, p_u: Tensor) -> Tensor:
    """PPI rectified mean: mean(unlabeled score) + mean(labeled residual)."""
    return p_u.mean() + (y_l - p_l).mean()


def _rectified_mean_bootstrap(
    y_l: Tensor,
    p_l: Tensor,
    p_u: Tensor,
    *,
    n_boot: int,
    alpha: float,
    generator: torch.Generator | None,
) -> tuple[Tensor, float, float]:
    """Nonparametric bootstrap for rectified mean with fixed calibrated scores."""
    boot_est = _bootstrap_means(y_l - p_l, n_boot=n_boot, generator=generator)
    boot_est = boot_est + _bootstrap_means(p_u, n_boot=n_boot, generator=generator)
    ci_lower, ci_upper = _percentile_ci(boot_est, alpha)
    return boot_est, ci_lower, ci_upper


def _linear_calibration_coefs(m_fit: Tensor, y_fit: Tensor) -> tuple[float, float]:
    """``(intercept, slope)`` of the least-squares affine map ``m -> y``."""
    mf = m_fit.reshape(-1)
    yf = y_fit.reshape(-1).to(mf.dtype)
    m_cent = mf - mf.mean()
    denom = (m_cent * m_cent).sum()
    if float(denom.item()) < 1e-20:
        return float(yf.mean().item()), 0.0
    slope = float(((m_cent * (yf - yf.mean())).sum() / denom).item())
    intercept = float((yf.mean() - mf.mean() * slope).item())
    return intercept, slope


def _linear_calibrate_apply(m_fit: Tensor, y_fit: Tensor, m_apply: Tensor) -> Tensor:
    """Affine map minimizing squared error on (m_fit, y_fit); applied to m_apply."""
    intercept, slope = _linear_calibration_coefs(m_fit, y_fit)
    return intercept + slope * m_apply


def ppi_mean_ci(
    y_labeled: Tensor | list[float],
    pred_labeled: Tensor | list[float],
    pred_unlabeled: Tensor | list[float],
    *,
    config: PPIConfig | None = None,
) -> dict[str, Any]:
    """Prediction-powered CI for a population mean.

    Estimator:
        E[Y] ≈ mean(pred_unlabeled) + mean(y_labeled - pred_labeled)

    References
    ----------
    .. [1] Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T. (2023).
       Prediction-Powered Inference. In *Science*, 382(6673), 903-907.
       https://arxiv.org/abs/2301.09633
    """
    cfg = config or PPIConfig()

    if not 0 < cfg.alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {cfg.alpha}")
    if cfg.n_boot < 10:
        raise ValueError(f"n_boot must be >= 10, got {cfg.n_boot}")
    if cfg.method not in {"bootstrap"}:
        raise ValueError(f"Unsupported method: {cfg.method}")

    y_l, p_l, p_u = _common_float(
        _to_1d_tensor(y_labeled), _to_1d_tensor(pred_labeled), _to_1d_tensor(pred_unlabeled)
    )
    if y_l.numel() != p_l.numel():
        raise ValueError("y_labeled and pred_labeled must have the same number of samples")
    if y_l.numel() < 2 or p_u.numel() < 2:
        raise ValueError("ppi_mean_ci requires at least 2 labeled and 2 unlabeled samples")

    residual = y_l - p_l
    point = _rectified_mean_point(y_l, p_l, p_u)
    estimate = float(point.item())

    # Asymptotic-style standard error (for diagnostics).
    se = float(
        torch.sqrt(
            residual.var(unbiased=True) / max(y_l.numel(), 1)
            + p_u.var(unbiased=True) / max(p_u.numel(), 1)
        ).item()
    )

    bootstrap_gen: torch.Generator | None = None
    if cfg.seed is not None:
        bootstrap_gen = torch.Generator(device=y_l.device)
        bootstrap_gen.manual_seed(cfg.seed)

    boot_est, ci_lower, ci_upper = _rectified_mean_bootstrap(
        y_l,
        p_l,
        p_u,
        n_boot=cfg.n_boot,
        alpha=cfg.alpha,
        generator=bootstrap_gen,
    )

    return {
        "method": "ppi_mean_ci",
        "estimate": estimate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": cfg.alpha,
        "n_labeled": int(y_l.numel()),
        "n_unlabeled": int(p_u.numel()),
        "bootstrap_samples": int(cfg.n_boot),
    }


def ppi_calibrated_mean_ci(
    y_labeled: Tensor | list[float],
    pred_labeled: Tensor | list[float],
    pred_unlabeled: Tensor | list[float],
    *,
    config: PPIConfig | None = None,
) -> dict[str, Any]:
    """Prediction-powered CI for a population mean with affine post-hoc calibration.

    Fits an affine map :math:`m^\\star(x) = \\hat a + \\hat b\\, m(x)` by ordinary
    least squares on labeled pairs :math:`(m(X_i), Y_i)`, then applies the usual
    rectified PPI mean
    :math:`\\mathbb{E}[m^\\star(\\tilde X)] + \\mathbb{E}[Y - m^\\star(X)]`
    with paired bootstrap that **refits** :math:`(\\hat a, \\hat b)` on each labeled
    resample.

    This is the linearly calibrated ("calibeating") PPI mean of van der Laan & van
    der Laan (arXiv:2604.21260); they relate it to prognostic-score style
    adjustment and to PPI++ at first order.

    Parameters
    ----------
    y_labeled, pred_labeled, pred_unlabeled
        Same semantics as :func:`ppi_mean_ci`.
    config
        Same as :class:`PPIConfig` for :func:`ppi_mean_ci`.

    References
    ----------
    .. [1] van der Laan, L., & van der Laan, M. (2026). Calibeating Prediction-Powered
       Inference. In *arXiv:2604.21260*. https://arxiv.org/abs/2604.21260
    """
    cfg = config or PPIConfig()
    if not 0 < cfg.alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {cfg.alpha}")
    if cfg.n_boot < 10:
        raise ValueError(f"n_boot must be >= 10, got {cfg.n_boot}")
    if cfg.method not in {"bootstrap"}:
        raise ValueError(f"Unsupported method: {cfg.method}")

    y_l, p_l, p_u = _common_float(
        _to_1d_tensor(y_labeled), _to_1d_tensor(pred_labeled), _to_1d_tensor(pred_unlabeled)
    )
    if y_l.numel() != p_l.numel():
        raise ValueError("y_labeled and pred_labeled must have the same number of samples")
    if y_l.numel() < 3 or p_u.numel() < 2:
        raise ValueError(
            "ppi_calibrated_mean_ci requires at least 3 labeled and 2 unlabeled samples"
        )

    p_l_cal = _linear_calibrate_apply(p_l, y_l, p_l)
    p_u_cal = _linear_calibrate_apply(p_l, y_l, p_u)

    point = _rectified_mean_point(y_l, p_l_cal, p_u_cal)
    estimate = float(point.item())
    residual_cal = y_l - p_l_cal
    se = float(
        torch.sqrt(
            residual_cal.var(unbiased=True) / max(y_l.numel(), 1)
            + p_u_cal.var(unbiased=True) / max(p_u.numel(), 1)
        ).item()
    )

    bootstrap_gen: torch.Generator | None = None
    if cfg.seed is not None:
        bootstrap_gen = torch.Generator(device=y_l.device)
        bootstrap_gen.manual_seed(cfg.seed)
    # The calibrated unlabeled term is affine in the raw scores, so its bootstrap mean
    # is ``a_b + b_b * mean(p_u[resample])``: only the bootstrap means of p_u are needed.
    boot_u_mean = _bootstrap_means(p_u, n_boot=cfg.n_boot, generator=bootstrap_gen)
    boot_est = torch.empty(cfg.n_boot, device=y_l.device, dtype=y_l.dtype)
    for b in range(cfg.n_boot):
        li = _bootstrap_indices(y_l.numel(), n_boot=1, device=y_l.device, generator=bootstrap_gen)[
            0
        ]
        m_lb, y_lb = p_l[li], y_l[li]
        intercept, slope = _linear_calibration_coefs(m_lb, y_lb)
        p_lb = intercept + slope * m_lb
        boot_est[b] = intercept + slope * boot_u_mean[b] + (y_lb - p_lb).mean()
    ci_lower, ci_upper = _percentile_ci(boot_est, cfg.alpha)

    return {
        "method": "ppi_calibrated_mean_ci",
        "estimate": estimate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": cfg.alpha,
        "n_labeled": int(y_l.numel()),
        "n_unlabeled": int(p_u.numel()),
        "bootstrap_samples": int(cfg.n_boot),
    }


def _strided_sorted(v: Tensor, max_points: int) -> Tensor:
    """Sorted values of ``v``, thinned to at most ``max_points`` evenly spaced order statistics."""
    sv = torch.sort(v).values
    if sv.numel() <= max_points:
        return sv
    idx = (
        torch.linspace(0, sv.numel() - 1, max_points, device=v.device, dtype=torch.float64)
        .round()
        .long()
    )
    return sv[idx]


def _rectified_cdf(
    grid: Tensor, y_l: Tensor, p_l: Tensor, p_u: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Rectified CDF on ``grid`` and the two variance components of its estimate.

    Returns ``(F, F_u, var_rect)`` with
    ``F(t) = F_u(t) + mean_l(1{y <= t} - 1{f <= t})``,
    ``F_u(t) = mean_u 1{f_u <= t}`` and ``var_rect`` the unbiased variance of the
    labeled rectifier. Uses sorted arrays and ``searchsorted`` so memory is
    O(n + N + len(grid)).
    """
    n = y_l.numel()
    big_n = p_u.numel()
    cnt_u = torch.searchsorted(torch.sort(p_u).values, grid, right=True).to(grid.dtype)
    cnt_y = torch.searchsorted(torch.sort(y_l).values, grid, right=True).to(grid.dtype)
    cnt_f = torch.searchsorted(torch.sort(p_l).values, grid, right=True).to(grid.dtype)
    # 1{y<=t} * 1{f<=t} = 1{max(y, f) <= t}
    cnt_both = torch.searchsorted(torch.sort(torch.maximum(y_l, p_l)).values, grid, right=True).to(
        grid.dtype
    )
    f_u = cnt_u / big_n
    rect_mean = (cnt_y - cnt_f) / n
    # rect in {-1, 0, 1}; rect^2 = 1{y<=t} xor 1{f<=t}
    rect_sq_mean = (cnt_y + cnt_f - 2.0 * cnt_both) / n
    var_rect = (rect_sq_mean - rect_mean.pow(2)).clamp_min(0.0) * (n / (n - 1.0))
    return f_u + rect_mean, f_u, var_rect


def ppi_quantile_ci(
    y_labeled: Tensor | list[float],
    pred_labeled: Tensor | list[float],
    pred_unlabeled: Tensor | list[float],
    *,
    q: float,
    config: PPIConfig | None = None,
) -> dict[str, Any]:
    """Prediction-powered CI for a target quantile.

    Implements the PPI quantile estimator of Angelopoulos et al. (2023) by
    inverting the rectified CDF

    .. math::
        \\hat F(\\theta) = \\frac{1}{N} \\sum_{j=1}^N 1\\{\\tilde f_j \\le \\theta\\}
            + \\frac{1}{n} \\sum_{i=1}^n \\big(1\\{Y_i \\le \\theta\\}
            - 1\\{f_i \\le \\theta\\}\\big),

    where :math:`\\tilde f_j` are the unlabeled predictions and :math:`(f_i, Y_i)`
    the labeled pairs. The point estimate is
    :math:`\\hat\\theta = \\inf\\{\\theta : \\hat F(\\theta) \\ge q\\}` and the
    :math:`1-\\alpha` confidence set is

    .. math::
        \\Big\\{\\theta : |\\hat F(\\theta) - q| \\le z_{1-\\alpha/2}
            \\sqrt{\\hat F_u(\\theta)(1-\\hat F_u(\\theta))/N
            + \\widehat{\\mathrm{Var}}_l(\\text{rectifier}(\\theta))/n}\\Big\\},

    evaluated on a grid made of the sorted labeled outcomes, labeled predictions
    and unlabeled predictions (each thinned to at most 4096 evenly spaced order
    statistics). The reported interval is the hull of that set, widened if
    needed to contain the point estimate. Memory is O(n + N).

    Parameters
    ----------
    y_labeled, pred_labeled, pred_unlabeled
        Labeled outcomes, labeled predictions and unlabeled predictions.
    q
        Target quantile level in ``(0, 1)``.
    config
        :class:`PPIConfig`; only ``alpha`` is used (the interval is the CLT
        inversion above, not a bootstrap, so ``n_boot`` and ``seed`` are ignored
        and ``bootstrap_samples`` is reported as 0).

    Returns
    -------
    dict
        ``estimate``, ``ci_lower``, ``ci_upper`` and ``se``, where ``se`` is the
        normal-equivalent half-width ``(ci_upper - ci_lower) / (2 z_{1-alpha/2})``.

    References
    ----------
    .. [1] Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T. (2023).
       Prediction-Powered Inference. In *Science*, 382(6673), 903-907.
       https://arxiv.org/abs/2301.09633
    """
    cfg = config or PPIConfig()

    if not 0 < q < 1:
        raise ValueError(f"q must be in (0, 1), got {q}")
    if not 0 < cfg.alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {cfg.alpha}")
    if cfg.n_boot < 10:
        raise ValueError(f"n_boot must be >= 10, got {cfg.n_boot}")
    if cfg.method not in {"bootstrap"}:
        raise ValueError(f"Unsupported method: {cfg.method}")

    y_l, p_l, p_u = _common_float(
        _to_1d_tensor(y_labeled), _to_1d_tensor(pred_labeled), _to_1d_tensor(pred_unlabeled)
    )
    if y_l.numel() != p_l.numel():
        raise ValueError("y_labeled and pred_labeled must have the same number of samples")
    if y_l.numel() < 2 or p_u.numel() < 2:
        raise ValueError("ppi_quantile_ci requires at least 2 labeled and 2 unlabeled samples")

    # The rectified CDF is a right-continuous step function that only jumps at the
    # observed values, so evaluating it at (a thinned set of) them is sufficient.
    max_points = 4096
    grid = torch.unique(
        torch.cat(
            [
                _strided_sorted(y_l, max_points),
                _strided_sorted(p_l, max_points),
                _strided_sorted(p_u, max_points),
            ]
        )
    )
    cdf, f_u, var_rect = _rectified_cdf(grid, y_l, p_l, p_u)
    # At the largest observed value every indicator is 1, so cdf[-1] == 1 >= q.
    first = int(torch.nonzero(cdf >= q)[0].item())
    estimate = float(grid[first].item())

    z = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - cfg.alpha / 2.0)))
    half_width = z * torch.sqrt(f_u * (1.0 - f_u) / p_u.numel() + var_rect / y_l.numel())
    inside = (cdf - q).abs() <= half_width
    if bool(inside.any()):
        ci_lower = min(float(grid[inside].min().item()), estimate)
        ci_upper = max(float(grid[inside].max().item()), estimate)
    else:
        ci_lower = ci_upper = estimate
    se = (ci_upper - ci_lower) / (2.0 * z)

    return {
        "method": "ppi_quantile_ci",
        "estimate": estimate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "q": q,
        "alpha": cfg.alpha,
        "n_labeled": int(y_l.numel()),
        "n_unlabeled": int(p_u.numel()),
        "bootstrap_samples": 0,
    }


def _as_2d(x: Tensor) -> Tensor:
    if x.dim() == 1:
        return x.unsqueeze(1)
    return x


def _add_intercept(x: Tensor) -> Tensor:
    ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
    return torch.cat([ones, x], dim=1)


def _ols_beta(x: Tensor, y: Tensor, ridge: float = 1e-6) -> Tensor:
    xtx = x.T @ x
    eye = torch.eye(xtx.shape[0], device=x.device, dtype=x.dtype)
    xty = x.T @ y
    return cast(Tensor, torch.linalg.solve(xtx + ridge * eye, xty))


def ppi_ols_ci(  # noqa: PLR0913
    x_labeled: Tensor,
    y_labeled: Tensor,
    x_unlabeled: Tensor,
    pred_labeled: Tensor,
    pred_unlabeled: Tensor,
    *,
    add_intercept: bool = True,
    config: PPIConfig | None = None,
) -> dict[str, Any]:
    """Prediction-powered CI for linear coefficients.

    Beta estimate combines:
    - plugin regression on unlabeled predictions, and
    - labeled residual correction.

    References
    ----------
    .. [1] Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T. (2023).
       Prediction-Powered Inference. In *Science*, 382(6673), 903-907.
       https://arxiv.org/abs/2301.09633
    """
    # Note: ppi_ols_ci historically used n_boot=1000 by default.
    # We create a specific default config for it if none provided.
    cfg = config or PPIConfig(n_boot=1000)

    if not 0 < cfg.alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {cfg.alpha}")
    if cfg.n_boot < 10:
        raise ValueError(f"n_boot must be >= 10, got {cfg.n_boot}")

    x_l, x_u, y_l, p_l, p_u = _common_float(
        _as_2d(_to_float_tensor(x_labeled)),
        _as_2d(_to_float_tensor(x_unlabeled)),
        _to_1d_tensor(y_labeled),
        _to_1d_tensor(pred_labeled),
        _to_1d_tensor(pred_unlabeled),
    )
    if x_l.shape[0] != y_l.numel() or x_l.shape[0] != p_l.numel():
        raise ValueError("x_labeled, y_labeled, and pred_labeled must align on sample dimension")
    if x_u.shape[0] != p_u.numel():
        raise ValueError("x_unlabeled and pred_unlabeled must align on sample dimension")

    if add_intercept:
        x_l = _add_intercept(x_l)
        x_u = _add_intercept(x_u)

    beta_pred = _ols_beta(x_u, p_u)
    beta_delta = _ols_beta(x_l, y_l - p_l)
    beta = beta_pred + beta_delta

    bootstrap_gen: torch.Generator | None = None
    if cfg.seed is not None:
        bootstrap_gen = torch.Generator(device=x_l.device)
        bootstrap_gen.manual_seed(cfg.seed)
    boot_beta = torch.empty((cfg.n_boot, beta.numel()), device=beta.device, dtype=beta.dtype)
    for i in range(cfg.n_boot):
        # One resample at a time keeps memory O(N) instead of O(n_boot * N).
        li = _bootstrap_indices(x_l.shape[0], n_boot=1, device=x_l.device, generator=bootstrap_gen)[
            0
        ]
        ui = _bootstrap_indices(x_u.shape[0], n_boot=1, device=x_u.device, generator=bootstrap_gen)[
            0
        ]
        b_pred = _ols_beta(x_u[ui], p_u[ui])
        b_delta = _ols_beta(x_l[li], (y_l - p_l)[li])
        boot_beta[i] = b_pred + b_delta

    se = torch.std(boot_beta, dim=0, unbiased=True)
    ci_lo = torch.quantile(boot_beta, cfg.alpha / 2, dim=0)
    ci_hi = torch.quantile(boot_beta, 1.0 - cfg.alpha / 2, dim=0)

    return {
        "method": "ppi_ols_ci",
        "coef": beta.detach().cpu().tolist(),
        "se": se.detach().cpu().tolist(),
        "ci_lower": ci_lo.detach().cpu().tolist(),
        "ci_upper": ci_hi.detach().cpu().tolist(),
        "alpha": cfg.alpha,
        "add_intercept": add_intercept,
        "n_labeled": int(x_labeled.shape[0]),
        "n_unlabeled": int(x_unlabeled.shape[0]),
        "bootstrap_samples": int(cfg.n_boot),
    }


def ppi_diagnostics(
    y_labeled: Tensor | list[float],
    pred_labeled: Tensor | list[float],
    pred_unlabeled: Tensor | list[float],
) -> dict[str, float]:
    """Compute practical diagnostics for PPI validity and usefulness."""
    y_l, p_l, p_u = _common_float(
        _to_1d_tensor(y_labeled), _to_1d_tensor(pred_labeled), _to_1d_tensor(pred_unlabeled)
    )
    if y_l.numel() != p_l.numel():
        raise ValueError("y_labeled and pred_labeled must have the same number of samples")

    residual = y_l - p_l
    if y_l.numel() > 1:
        y_std = float(y_l.std(unbiased=False).item())
        p_std = float(p_l.std(unbiased=False).item())
        if y_std > 0.0 and p_std > 0.0:
            corr = float(torch.corrcoef(torch.stack([y_l, p_l]))[0, 1].item())
        else:
            corr = 0.0
    else:
        corr = 0.0
    rmse = float(torch.sqrt(torch.mean((y_l - p_l) ** 2)).item())
    mean_shift = float((p_u.mean() - p_l.mean()).item())
    pred_l_min, pred_l_max = float(p_l.min().item()), float(p_l.max().item())
    pred_u_min, pred_u_max = float(p_u.min().item()), float(p_u.max().item())
    overlap = max(0.0, min(pred_l_max, pred_u_max) - max(pred_l_min, pred_u_min))
    denom = max(pred_l_max - pred_l_min, 1e-8)
    overlap_ratio = overlap / denom

    return {
        "n_labeled": float(y_l.numel()),
        "n_unlabeled": float(p_u.numel()),
        "prediction_label_correlation": corr,
        "residual_rmse_labeled": rmse,
        "residual_mean_labeled": float(residual.mean().item()),
        "prediction_mean_shift_unlabeled_vs_labeled": mean_shift,
        "prediction_range_overlap_ratio": float(overlap_ratio),
    }


def _var(x: Tensor) -> Tensor:
    return x.var(unbiased=True) if x.numel() > 1 else torch.zeros((), device=x.device)


def ppi_pp_mean_ci(  # noqa: PLR0912
    y_labeled: Tensor | list[float],
    pred_labeled: Tensor | list[float],
    pred_unlabeled: Tensor | list[float],
    *,
    lambdas: Tensor | list[float] | None = None,
    cross_fits: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """PPI++ confidence interval for a population mean (TR-INF-02).

    Selects the power-tuning parameter :math:`\\lambda` minimizing the PPI++
    first-order variance

    .. math::
        V(\\lambda) = \\frac{\\mathrm{Var}_l(y)}{n}
            + \\lambda^2 \\Big(\\frac{\\mathrm{Var}_u(\\hat y)}{N}
            + \\frac{\\mathrm{Var}_l(\\hat y)}{n}\\Big)
            - 2\\lambda \\frac{\\mathrm{Cov}_l(y, \\hat y)}{n},

    of the PPI++ estimator
    :math:`\\hat\\theta_\\lambda = \\bar y_l
    + \\lambda(\\bar{\\hat y}_u - \\bar{\\hat y}_l)`, over ``lambdas`` (default grid
    ``torch.linspace(0, 1, 21)``) per Angelopoulos et al. When
    ``cross_fits=k > 0``, labeled data is split into k folds and the affine
    calibration rectifier is refit out-of-fold before residuals are computed.

    References
    ----------
    .. [1] Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T.
       (2023). Prediction-Powered Inference. In *Science*, 382(6673), 903-907.
       https://arxiv.org/abs/2301.09633
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if cross_fits < 0:
        raise ValueError(f"cross_fits must be >= 0, got {cross_fits}")

    y_l, p_l, p_u = _common_float(
        _to_1d_tensor(y_labeled), _to_1d_tensor(pred_labeled), _to_1d_tensor(pred_unlabeled)
    )
    if y_l.numel() != p_l.numel():
        raise ValueError("y_labeled and pred_labeled must have the same number of samples")
    min_l = cross_fits + 2 if cross_fits > 0 else 2
    if y_l.numel() < min_l or p_u.numel() < 2:
        raise ValueError(
            f"ppi_pp_mean_ci requires at least {min_l} labeled and 2 unlabeled samples"
        )

    if lambdas is None:
        lambda_grid = torch.linspace(0.0, 1.0, 21, device=y_l.device, dtype=torch.float32)
    else:
        lambda_grid = torch.as_tensor(lambdas, dtype=torch.float32, device=y_l.device).reshape(-1)
        if lambda_grid.numel() == 0:
            raise ValueError("lambdas must be non-empty when provided")

    if cross_fits > 0:
        k = int(cross_fits)
        fold_idx = torch.arange(y_l.numel()) % k
        pl_cal_parts = []
        for f in range(k):
            train_mask = fold_idx != f
            test_mask = ~train_mask
            # Out-of-fold affine rectifier fit on the other folds only.
            p_l_f_cal = _linear_calibrate_apply(p_l[train_mask], y_l[train_mask], p_l[test_mask])
            pl_cal_parts.append(p_l_f_cal)
        pl_cal = torch.empty_like(p_l)
        for f in range(k):
            test_mask = fold_idx == f
            pl_cal[test_mask] = pl_cal_parts[f]
        p_u_eff = _linear_calibrate_apply(p_l, y_l, p_u)
        p_l_eff = pl_cal
    else:
        p_l_eff = p_l
        p_u_eff = p_u

    n = float(y_l.numel())
    big_n = float(p_u_eff.numel())

    # Unbiased PPI++ family: theta(lam) = mean_l(y) + lam*(mean_u(f_u) - mean_l(f_l)).
    # First-order variance of that linear combination (TR-INF-02):
    #   V(lam) = Var_l(y)/n + lam^2*(Var_u(f)/N + Var_l(f)/n) - 2*lam*Cov_l(y, f)/n
    var_y = _var(y_l)
    var_pl = _var(p_l_eff)
    var_pu = _var(p_u_eff)
    yc = y_l - y_l.mean()
    pc = p_l_eff - p_l_eff.mean()
    cov_yf = float(torch.mean(yc * pc).item())
    if n > 1:
        cov_yf *= n / (n - 1.0)  # unbiased covariance scaling

    lam = lambda_grid.to(torch.float64)
    v = (
        var_y.to(torch.float64) / n
        + lam.pow(2) * (var_pu.to(torch.float64) / big_n + var_pl.to(torch.float64) / n)
        - 2.0 * lam * cov_yf / n
    )
    best = int(torch.argmin(v).item())
    lambda_star = float(lambda_grid[best].item())
    variance = float(max(v[best].item(), 0.0))

    point = y_l.mean() + lambda_star * (p_u_eff.mean() - p_l_eff.mean())
    estimate = float(point.item())
    se = float(math.sqrt(variance))
    z = float(torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - alpha / 2.0)).item())
    ci_lower = estimate - z * se
    ci_upper = estimate + z * se

    return {
        "method": "ppi_pp_mean_ci",
        "estimate": estimate,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "alpha": float(alpha),
        "lambda": lambda_star,
        "variance": variance,
        "n_labeled": int(y_l.numel()),
        "n_unlabeled": int(p_u.numel()),
        "cross_fits": int(cross_fits),
    }


__all__ = [
    "PPIConfig",
    "ppi_calibrated_mean_ci",
    "ppi_diagnostics",
    "ppi_mean_ci",
    "ppi_ols_ci",
    "ppi_pp_mean_ci",
    "ppi_quantile_ci",
]
