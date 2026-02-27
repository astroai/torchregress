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
    return torch.tensor(pred, dtype=torch.float32, device=x.device).reshape(-1)


def _predict_propensity(model: Any, x: Tensor, *, eps: float = 1e-4) -> Tensor:
    x_np = x.detach().cpu().numpy()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_np)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            out = proba[:, 1]
        else:
            out = proba.reshape(-1)
        return torch.tensor(out, dtype=torch.float32, device=x.device).clamp(eps, 1.0 - eps)
    if hasattr(model, "predict"):
        logit = model.predict(x_np).reshape(-1)
        out = torch.sigmoid(torch.tensor(logit, dtype=torch.float32, device=x.device))
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
    mu1_hat = torch.empty(n, dtype=torch.float32, device=x.device)
    mu0_hat = torch.empty(n, dtype=torch.float32, device=x.device)
    e_hat = torch.empty(n, dtype=torch.float32, device=x.device)

    for train_idx, test_idx in _make_folds(n, folds, seed=seed):
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

    return {"mu1_hat": mu1_hat, "mu0_hat": mu0_hat, "e_hat": e_hat}


def _normal_ci(estimate: float, se: float, *, alpha: float) -> Tuple[float, float]:
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else float(
        torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1.0 - alpha / 2.0)).item()
    )
    return estimate - z * se, estimate + z * se


def _dr_scores(y: Tensor, t: Tensor, mu1_hat: Tensor, mu0_hat: Tensor, e_hat: Tensor) -> Tensor:
    dr: Tensor = (
        mu1_hat
        - mu0_hat
        + t * (y - mu1_hat) / e_hat
        - (1.0 - t) * (y - mu0_hat) / (1.0 - e_hat)
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
) -> Dict[str, Any]:
    """Cross-fitted doubly-robust ATE with robust SE/CI and overlap diagnostics."""
    x2 = _as_2d(x).float()
    t1 = _as_1d(t).float()
    y1 = _as_1d(y).float()
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
    dr = _dr_scores(y1, t1, nuisance["mu1_hat"], nuisance["mu0_hat"], nuisance["e_hat"])
    n = dr.numel()
    ate = float(dr.mean().item())
    se = float(dr.std(unbiased=True).item() / math.sqrt(max(n, 1)))
    ci_low, ci_high = _normal_ci(ate, se, alpha=alpha)
    overlap = causal_overlap_report(nuisance["e_hat"], t1, trim_threshold=trim_threshold, eps=eps)

    return {
        "estimate": ate,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "alpha": alpha,
        "n_samples": n,
        "dr_scores": dr,
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
) -> Dict[str, Any]:
    """Cross-fitted DR CATE via pseudo-outcome regression."""
    x2 = _as_2d(x).float()
    t1 = _as_1d(t).float()
    y1 = _as_1d(y).float()
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
    dr = _dr_scores(y1, t1, nuisance["mu1_hat"], nuisance["mu0_hat"], nuisance["e_hat"])

    cate = _fit_model(_build_model(cate_model), x2, dr)
    cate_hat = _predict_outcome(cate, x2)

    ate = float(dr.mean().item())
    se = float(dr.std(unbiased=True).item() / math.sqrt(max(dr.numel(), 1)))
    ci_low, ci_high = _normal_ci(ate, se, alpha=alpha)
    overlap = causal_overlap_report(nuisance["e_hat"], t1, trim_threshold=trim_threshold, eps=eps)

    return {
        "ate_estimate": ate,
        "ate_se": se,
        "ate_ci_low": ci_low,
        "ate_ci_high": ci_high,
        "alpha": alpha,
        "cate_hat": cate_hat,
        "pseudo_outcome": dr,
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
    x2 = _as_2d(x).float()
    t1 = _as_1d(t).float()
    y1 = _as_1d(y).float()
    pi = _as_1d(policy).float()
    if not (x2.shape[0] == t1.shape[0] == y1.shape[0] == pi.shape[0]):
        raise ValueError("x, t, y, and policy must share sample dimension")
    pi = (pi > 0.5).float()

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
