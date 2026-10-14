"""Doubly-robust causal inference utilities for ATE/CATE."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Tuple

import torch
from torch import Tensor

from .diagnostics import causal_overlap_report

ModelFactory = Any


def _as_2d(x: Tensor) -> Tensor:
    if x.dim() == 1:
        return x.unsqueeze(-1)
    return x


def _as_1d(x: Tensor) -> Tensor:
    return x.reshape(-1)


def _float_dtype(*tensors: Tensor) -> torch.dtype:
    """Promoted floating dtype of the floating inputs (default dtype if none are floating).

    Keeps float64 inputs in float64 instead of silently downcasting to float32.
    """
    dtype: torch.dtype | None = None
    for t in tensors:
        if t.is_floating_point():
            dtype = t.dtype if dtype is None else torch.promote_types(dtype, t.dtype)
    return dtype if dtype is not None else torch.get_default_dtype()


def _build_model(factory_or_model: ModelFactory) -> Any:
    if isinstance(factory_or_model, type):
        return factory_or_model()
    if callable(factory_or_model) and not hasattr(factory_or_model, "fit"):
        return factory_or_model()
    return copy.deepcopy(factory_or_model)


def _fit_model(model: Any, x: Tensor, y: Tensor) -> Any:
    if not hasattr(model, "fit"):
        raise TypeError("Model must implement fit(X, y)")
    model.fit(x.detach().cpu().numpy(), y.detach().cpu().numpy().reshape(-1))
    return model


def _predict_outcome(model: Any, x: Tensor) -> Tensor:
    if not hasattr(model, "predict"):
        raise TypeError("Outcome model must implement predict(X)")
    pred = model.predict(x.detach().cpu().numpy())
    return torch.tensor(pred, dtype=x.dtype, device=x.device).reshape(-1)


def _predict_propensity(model: Any, x: Tensor, *, eps: float = 1e-4) -> Tensor:
    x_np = x.detach().cpu().numpy()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_np)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            out = proba[:, 1]
        else:
            out = proba.reshape(-1)
        return torch.tensor(out, dtype=x.dtype, device=x.device).clamp(eps, 1.0 - eps)
    if hasattr(model, "predict"):
        logit = model.predict(x_np).reshape(-1)
        out = torch.sigmoid(torch.tensor(logit, dtype=x.dtype, device=x.device))
        return out.clamp(eps, 1.0 - eps)
    raise TypeError("Propensity model must implement predict_proba(X) or predict(X)")


def _make_folds(n: int, folds: int, *, seed: int) -> list[Tuple[Tensor, Tensor]]:
    if folds < 2:
        raise ValueError("folds must be >= 2 for cross-fitting")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    fold_sizes = [n // folds for _ in range(folds)]
    for i in range(n % folds):
        fold_sizes[i] += 1

    split_indices: list[Tuple[Tensor, Tensor]] = []
    start = 0
    all_idx = torch.arange(n)
    for size in fold_sizes:
        stop = start + size
        test_idx = perm[start:stop]
        train_mask = torch.ones(n, dtype=torch.bool)
        train_mask[test_idx] = False
        train_idx = all_idx[train_mask]
        split_indices.append((train_idx, test_idx))
        start = stop
    return split_indices


def _crossfit_nuisances(
    x: Tensor,
    t: Tensor,
    y: Tensor,
    *,
    outcome_model: ModelFactory,
    propensity_model: ModelFactory,
    folds: int,
    seed: int,
    eps: float,
) -> Dict[str, Tensor]:
    n = x.shape[0]
    mu1_hat = torch.empty(n, dtype=x.dtype, device=x.device)
    mu0_hat = torch.empty(n, dtype=x.dtype, device=x.device)
    e_hat = torch.empty(n, dtype=x.dtype, device=x.device)
    fold_id = torch.empty(n, dtype=torch.long, device=x.device)

    for k, (train_idx, test_idx) in enumerate(_make_folds(n, folds, seed=seed)):
        x_train = x[train_idx]
        t_train = t[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]

        treated = t_train > 0.5
        control = ~treated
        if int(treated.sum().item()) == 0 or int(control.sum().item()) == 0:
            raise ValueError("Each fold must contain both treatment arms for DR estimation")

        m1 = _fit_model(_build_model(outcome_model), x_train[treated], y_train[treated])
        m0 = _fit_model(_build_model(outcome_model), x_train[control], y_train[control])
        mp = _fit_model(_build_model(propensity_model), x_train, t_train)

        mu1_hat[test_idx] = _predict_outcome(m1, x_test)
        mu0_hat[test_idx] = _predict_outcome(m0, x_test)
        e_hat[test_idx] = _predict_propensity(mp, x_test, eps=eps)
        fold_id[test_idx] = k

    return {"mu1_hat": mu1_hat, "mu0_hat": mu0_hat, "e_hat": e_hat, "fold_id": fold_id}


def _normal_ci(estimate: float, se: float, *, alpha: float) -> Tuple[float, float]:
    z = (
        1.959963984540054
        if abs(alpha - 0.05) < 1e-9
        else float(
            torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - alpha / 2.0)).item()
        )
    )
    return estimate - z * se, estimate + z * se


def _trim_scores(dr: Tensor, e_hat: Tensor, trim_threshold: float) -> Tuple[Tensor, int]:
    """Apply propensity trimming to the estimator (TR-CAU-01): drop scores
    whose cross-fitted propensity falls outside [trim_threshold, 1 - trim_threshold]."""
    keep = (e_hat >= trim_threshold) & (e_hat <= 1.0 - trim_threshold)
    n_kept = int(keep.sum().item())
    if n_kept < 2:
        raise ValueError(
            f"propensity trimming at trim_threshold={trim_threshold} keeps {n_kept} of "
            f"{keep.numel()} units (no overlap); lower trim_threshold or check positivity"
        )
    if bool(keep.all()):
        return dr, 0
    return dr[keep], int((~keep).sum().item())


def _fold_bootstrap_se(
    dr: Tensor,
    fold_id: Tensor,
    keep: Tensor,
    *,
    n_boot: int = 500,
    seed: int,
    max_elements: int = 2**22,
) -> float:
    """Fold-stratified bootstrap SE of the DR mean (TR-CAU-02).

    Resamples the kept DR scores with replacement *within* each cross-fit fold
    (fold sizes fixed) B times and returns the standard deviation of the pooled
    mean. Bootstrapping individual scores (rather than the K fold means) gives a
    consistent SE for any number of folds. Deterministic via ``seed``; memory is
    bounded by chunking the resamples to ``max_elements`` indices at a time.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    fold_ids = torch.unique(fold_id[keep]).tolist()
    if len(fold_ids) < 2:
        raise ValueError("fold_bootstrap requires at least 2 non-empty folds after trimming")
    total = int(keep.sum().item())
    boot = torch.zeros(n_boot, dtype=torch.float64)
    for k in fold_ids:
        scores_k = dr[(fold_id == k) & keep].detach().cpu().to(torch.float64)
        n_k = scores_k.numel()
        chunk = max(1, max_elements // n_k)
        for start in range(0, n_boot, chunk):
            b = min(chunk, n_boot - start)
            idx = torch.randint(0, n_k, (b, n_k), generator=gen)
            boot[start : start + b] += scores_k[idx].sum(dim=1) / total
    return float(boot.std(unbiased=True).item())


def _dr_scores(y: Tensor, t: Tensor, mu1_hat: Tensor, mu0_hat: Tensor, e_hat: Tensor) -> Tensor:
    dr: Tensor = (
        mu1_hat - mu0_hat + t * (y - mu1_hat) / e_hat - (1.0 - t) * (y - mu0_hat) / (1.0 - e_hat)
    )
    return dr


def dr_ate(
    x: Tensor,
    t: Tensor,
    y: Tensor,
    *,
    outcome_model: ModelFactory,
    propensity_model: ModelFactory,
    folds: int = 2,
    alpha: float = 0.05,
    seed: int = 42,
    trim_threshold: float = 0.05,
    eps: float = 1e-4,
    se_method: str = "analytic",
) -> Dict[str, Any]:
    """Cross-fitted doubly-robust ATE with robust SE/CI and overlap diagnostics.

    Propensity trimming acts on the estimator (TR-CAU-01): DR scores whose
    cross-fitted propensity lies outside ``[trim_threshold, 1 - trim_threshold]``
    are excluded before estimate/SE/CI are computed; the count is reported in
    ``diagnostics["n_trimmed"]``. A ``ValueError`` is raised if fewer than two
    units survive trimming. ``se_method="fold_bootstrap"`` uses a seeded B=500
    bootstrap of the individual DR scores, stratified by cross-fit fold (TR-CAU-02).
    Floating inputs keep their dtype (float64 stays float64); integer/bool inputs
    use the default dtype.

    References
    ----------
    .. [1] Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of Regression
       Coefficients When Some Regressors are Not Always Observed. In *JASA*, 89(427), 846-866.
       https://doi.org/10.1080/01621459.1994.10476818
    .. [2] Chernozhukov, V., et al. (2018). Double/debiased machine learning for
       treatment and structural parameters. In *The Econometrics Journal*, 21(1), C1-C68.
       https://arxiv.org/abs/1701.02036
    """
    dtype = _float_dtype(x, t, y)
    x2 = _as_2d(x).to(dtype)
    t1 = _as_1d(t).to(dtype)
    y1 = _as_1d(y).to(dtype)
    if not (x2.shape[0] == t1.shape[0] == y1.shape[0]):
        raise ValueError("x, t, and y must share sample dimension")

    nuisance = _crossfit_nuisances(
        x2,
        t1,
        y1,
        outcome_model=outcome_model,
        propensity_model=propensity_model,
        folds=folds,
        seed=seed,
        eps=eps,
    )
    e_hat = nuisance["e_hat"]
    fold_id = nuisance["fold_id"]
    dr = _dr_scores(y1, t1, nuisance["mu1_hat"], nuisance["mu0_hat"], e_hat)
    keep = (e_hat >= trim_threshold) & (e_hat <= 1.0 - trim_threshold)
    dr_kept, n_trimmed = _trim_scores(dr, e_hat, trim_threshold)
    n = int(keep.sum().item())
    ate = float(dr_kept.mean().item())
    if se_method == "analytic":
        se = float(dr_kept.std(unbiased=False).item() / math.sqrt(max(n, 1)))
    elif se_method == "fold_bootstrap":
        se = _fold_bootstrap_se(dr, fold_id, keep, seed=seed)
    else:
        raise ValueError(f"Unsupported se_method: {se_method}")
    ci_low, ci_high = _normal_ci(ate, se, alpha=alpha)
    overlap = causal_overlap_report(e_hat, t1, trim_threshold=trim_threshold, eps=eps)
    overlap["n_trimmed"] = n_trimmed
    overlap["trim_applied_to_estimator"] = not bool(keep.all())

    return {
        "estimate": ate,
        "se": se,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": alpha,
        "dr_scores": dr,
        "dr_scores_trimmed": dr_kept,
        "propensity": nuisance["e_hat"],
        "mu1_hat": nuisance["mu1_hat"],
        "mu0_hat": nuisance["mu0_hat"],
        "diagnostics": overlap,
    }


def dr_cate(
    x: Tensor,
    t: Tensor,
    y: Tensor,
    *,
    cate_model: ModelFactory,
    outcome_model: ModelFactory,
    propensity_model: ModelFactory,
    folds: int = 2,
    alpha: float = 0.05,
    seed: int = 42,
    trim_threshold: float = 0.05,
    eps: float = 1e-4,
    se_method: str = "analytic",
) -> Dict[str, Any]:
    """Cross-fitted DR CATE via pseudo-outcome regression.

    Trimming semantics match :func:`dr_ate` (TR-CAU-01): the pseudo-outcome
    regression and ATE/SE/CI use only scores kept by the propensity trim;
    ``diagnostics["n_trimmed"]`` reports the dropped count, and a ``ValueError``
    is raised if fewer than two units survive. ``se_method="fold_bootstrap"`` uses
    the same fold-stratified bootstrap of individual DR scores (TR-CAU-02).

    References
    ----------
    .. [1] Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of Regression
       Coefficients When Some Regressors are Not Always Observed. In *JASA*, 89(427), 846-866.
       https://doi.org/10.1080/01621459.1994.10476818
    .. [2] Chernozhukov, V., et al. (2018). Double/debiased machine learning for
       treatment and structural parameters. In *The Econometrics Journal*, 21(1), C1-C68.
       https://arxiv.org/abs/1701.02036
    """
    dtype = _float_dtype(x, t, y)
    x2 = _as_2d(x).to(dtype)
    t1 = _as_1d(t).to(dtype)
    y1 = _as_1d(y).to(dtype)
    if not (x2.shape[0] == t1.shape[0] == y1.shape[0]):
        raise ValueError("x, t, and y must share sample dimension")

    nuisance = _crossfit_nuisances(
        x2,
        t1,
        y1,
        outcome_model=outcome_model,
        propensity_model=propensity_model,
        folds=folds,
        seed=seed,
        eps=eps,
    )
    e_hat = nuisance["e_hat"]
    fold_id = nuisance["fold_id"]
    dr = _dr_scores(y1, t1, nuisance["mu1_hat"], nuisance["mu0_hat"], e_hat)
    keep = (e_hat >= trim_threshold) & (e_hat <= 1.0 - trim_threshold)
    dr_kept, n_trimmed = _trim_scores(dr, e_hat, trim_threshold)

    x_kept = x2[keep]
    cate = _fit_model(_build_model(cate_model), x_kept, dr_kept)
    cate_hat = _predict_outcome(cate, x2)

    ate = float(dr_kept.mean().item())
    if se_method == "analytic":
        se = float(dr_kept.std(unbiased=False).item() / math.sqrt(max(dr_kept.numel(), 1)))
    elif se_method == "fold_bootstrap":
        se = _fold_bootstrap_se(dr, fold_id, keep, seed=seed)
    else:
        raise ValueError(f"Unsupported se_method: {se_method}")
    ci_low, ci_high = _normal_ci(ate, se, alpha=alpha)
    overlap = causal_overlap_report(e_hat, t1, trim_threshold=trim_threshold, eps=eps)
    overlap["n_trimmed"] = n_trimmed
    overlap["trim_applied_to_estimator"] = not bool(keep.all())

    return {
        "ate_estimate": ate,
        "ate_se": se,
        "ate_ci_lower": ci_low,
        "ate_ci_upper": ci_high,
        "ate_ci_low": ci_low,
        "ate_ci_high": ci_high,
        "alpha": alpha,
        "cate_hat": cate_hat,
        "pseudo_outcome": dr,
        "pseudo_outcome_trimmed": dr_kept,
        "propensity": nuisance["e_hat"],
        "mu1_hat": nuisance["mu1_hat"],
        "mu0_hat": nuisance["mu0_hat"],
        "diagnostics": overlap,
    }


def dr_policy_value(
    x: Tensor,
    t: Tensor,
    y: Tensor,
    *,
    policy: Tensor,
    outcome_model: ModelFactory,
    propensity_model: ModelFactory,
    folds: int = 2,
    seed: int = 42,
    eps: float = 1e-4,
) -> Dict[str, float]:
    """AIPW value estimate for a binary treatment policy."""
    dtype = _float_dtype(x, t, y)
    x2 = _as_2d(x).to(dtype)
    t1 = _as_1d(t).to(dtype)
    y1 = _as_1d(y).to(dtype)
    pi = _as_1d(policy).to(dtype)
    if not (x2.shape[0] == t1.shape[0] == y1.shape[0] == pi.shape[0]):
        raise ValueError("x, t, y, and policy must share sample dimension")
    pi = (pi > 0.5).to(dtype)

    nuisance = _crossfit_nuisances(
        x2,
        t1,
        y1,
        outcome_model=outcome_model,
        propensity_model=propensity_model,
        folds=folds,
        seed=seed,
        eps=eps,
    )
    mu1 = nuisance["mu1_hat"]
    mu0 = nuisance["mu0_hat"]
    e = nuisance["e_hat"]
    ipw_term = pi * (t1 * (y1 - mu1) / e) + (1.0 - pi) * ((1.0 - t1) * (y1 - mu0) / (1.0 - e))
    outcome_term = pi * mu1 + (1.0 - pi) * mu0
    value_scores = outcome_term + ipw_term
    n = value_scores.numel()
    value = float(value_scores.mean().item())
    se = float(value_scores.std(unbiased=True).item() / math.sqrt(max(n, 1)))
    return {"estimate": value, "se": se, "n_samples": float(n)}


__all__ = ["dr_ate", "dr_cate", "dr_policy_value"]
