"""Prediction-powered inference utilities.

These helpers provide practical confidence intervals that combine:
- a small labeled set with trusted outcomes, and
- a larger unlabeled set with model predictions.

The implementation is intentionally lightweight and frequentist-first.
"""

from __future__ import annotations

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
        n_boot: Number of bootstrap samples.
        seed: Random seed for reproducibility.
    """

    alpha: float = 0.1
    method: str = "bootstrap"
    n_boot: int = 2000
    seed: int | None = None


def _to_1d_tensor(x: Tensor | list[float]) -> Tensor:
    if isinstance(x, Tensor):
        t = x.detach().float()
    else:
        t = torch.as_tensor(x, dtype=torch.float32)
    return t.reshape(-1)


def _bootstrap_indices(
    n: int,
    *,
    n_boot: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    return torch.randint(low=0, high=n, size=(n_boot, n), device=device, generator=generator)


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
    boot_l_idx = _bootstrap_indices(
        y_l.numel(), n_boot=n_boot, device=y_l.device, generator=generator
    )
    boot_u_idx = _bootstrap_indices(
        p_u.numel(), n_boot=n_boot, device=p_u.device, generator=generator
    )
    boot_est = p_u[boot_u_idx].mean(dim=1) + (y_l - p_l)[boot_l_idx].mean(dim=1)
    ci_lower, ci_upper = _percentile_ci(boot_est, alpha)
    return boot_est, ci_lower, ci_upper


def _linear_calibrate_apply(m_fit: Tensor, y_fit: Tensor, m_apply: Tensor) -> Tensor:
    """Affine map minimizing squared error on (m_fit, y_fit); applied to m_apply."""
    mf = m_fit.reshape(-1).float()
    yf = y_fit.reshape(-1).float()
    ma = m_apply.reshape(-1).float()
    m_cent = mf - mf.mean()
    denom = (m_cent * m_cent).sum()
    if float(denom.item()) < 1e-20:
        return torch.full_like(ma, float(yf.mean().item())).reshape(m_apply.shape)
    slope = ((m_cent * (yf - yf.mean())).sum() / denom).item()
    intercept = (yf.mean() - mf.mean() * slope).item()
    return (intercept + slope * ma).reshape(m_apply.shape)


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

    y_l = _to_1d_tensor(y_labeled)
    p_l = _to_1d_tensor(pred_labeled)
    p_u = _to_1d_tensor(pred_unlabeled)
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

    This is the linearly calibrated PPI mean from Chen et al. (arXiv:2604.21260),
    Section 3.3; they relate it to prognostic-score style adjustment and to PPI++
    at first order.

    Parameters
    ----------
    y_labeled, pred_labeled, pred_unlabeled
        Same semantics as :func:`ppi_mean_ci`.
    config
        Same as :class:`PPIConfig` for :func:`ppi_mean_ci`.

    References
    ----------
    .. [1] Chen, et al. (2026). Linearly Calibrated Prediction-Powered Inference.
       In *arXiv:2604.21260*. https://arxiv.org/abs/2604.21260
    """
    cfg = config or PPIConfig()
    if not 0 < cfg.alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {cfg.alpha}")
    if cfg.n_boot < 10:
        raise ValueError(f"n_boot must be >= 10, got {cfg.n_boot}")
    if cfg.method not in {"bootstrap"}:
        raise ValueError(f"Unsupported method: {cfg.method}")

    y_l = _to_1d_tensor(y_labeled)
    p_l = _to_1d_tensor(pred_labeled)
    p_u = _to_1d_tensor(pred_unlabeled)
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
    boot_l_idx = _bootstrap_indices(
        y_l.numel(), n_boot=cfg.n_boot, device=y_l.device, generator=bootstrap_gen
    )
    boot_u_idx = _bootstrap_indices(
        p_u.numel(), n_boot=cfg.n_boot, device=p_u.device, generator=bootstrap_gen
    )

    boot_est = torch.empty(cfg.n_boot, device=y_l.device, dtype=torch.float32)
    for b in range(cfg.n_boot):
        li = boot_l_idx[b]
        ui = boot_u_idx[b]
        m_lb, y_lb = p_l[li], y_l[li]
        m_ub = p_u[ui]
        p_lb = _linear_calibrate_apply(m_lb, y_lb, m_lb)
        p_ub = _linear_calibrate_apply(m_lb, y_lb, m_ub)
        boot_est[b] = _rectified_mean_point(y_lb, p_lb, p_ub)
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


def ppi_quantile_ci(
    y_labeled: Tensor | list[float],
    pred_labeled: Tensor | list[float],
    pred_unlabeled: Tensor | list[float],
    *,
    q: float,
    config: PPIConfig | None = None,
) -> dict[str, Any]:
    """Prediction-powered CI for a target quantile.

    Uses a robust location correction based on labeled residual median:
        Q_q(Y) ≈ Q_q(pred_unlabeled) + median(y_labeled - pred_labeled)

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

    y_l = _to_1d_tensor(y_labeled)
    p_l = _to_1d_tensor(pred_labeled)
    p_u = _to_1d_tensor(pred_unlabeled)
    if y_l.numel() != p_l.numel():
        raise ValueError("y_labeled and pred_labeled must have the same number of samples")
    if y_l.numel() < 2 or p_u.numel() < 2:
        raise ValueError("ppi_quantile_ci requires at least 2 labeled and 2 unlabeled samples")

    residual = y_l - p_l
    shift = torch.median(residual)
    estimate = float((torch.quantile(p_u, q) + shift).item())

    bootstrap_gen: torch.Generator | None = None
    if cfg.seed is not None:
        bootstrap_gen = torch.Generator(device=y_l.device)
        bootstrap_gen.manual_seed(cfg.seed)
    boot_l_idx = _bootstrap_indices(
        y_l.numel(), n_boot=cfg.n_boot, device=y_l.device, generator=bootstrap_gen
    )
    boot_u_idx = _bootstrap_indices(
        p_u.numel(), n_boot=cfg.n_boot, device=p_u.device, generator=bootstrap_gen
    )
    boot_shift = torch.median(residual[boot_l_idx], dim=1).values
    boot_q = torch.quantile(p_u[boot_u_idx], q, dim=1)
    boot_est = boot_q + boot_shift
    ci_lower, ci_upper = _percentile_ci(boot_est, cfg.alpha)
    se = float(torch.std(boot_est, unbiased=True).item())

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
        "bootstrap_samples": int(cfg.n_boot),
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

    x_l = _as_2d(x_labeled.detach().float())
    x_u = _as_2d(x_unlabeled.detach().float())
    y_l = _to_1d_tensor(y_labeled)
    p_l = _to_1d_tensor(pred_labeled)
    p_u = _to_1d_tensor(pred_unlabeled)
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
    boot_l_idx = _bootstrap_indices(
        x_l.shape[0], n_boot=cfg.n_boot, device=x_l.device, generator=bootstrap_gen
    )
    boot_u_idx = _bootstrap_indices(
        x_u.shape[0], n_boot=cfg.n_boot, device=x_u.device, generator=bootstrap_gen
    )
    boot_beta = torch.empty((cfg.n_boot, beta.numel()), device=beta.device, dtype=beta.dtype)
    for i in range(cfg.n_boot):
        li = boot_l_idx[i]
        ui = boot_u_idx[i]
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
    y_l = _to_1d_tensor(y_labeled)
    p_l = _to_1d_tensor(pred_labeled)
    p_u = _to_1d_tensor(pred_unlabeled)
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


__all__ = [
    "PPIConfig",
    "ppi_calibrated_mean_ci",
    "ppi_mean_ci",
    "ppi_quantile_ci",
    "ppi_ols_ci",
    "ppi_diagnostics",
]
