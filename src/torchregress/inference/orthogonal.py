"""Neyman-orthogonal (double/debiased machine learning) estimating equations.

The canonical partially linear model is

    y = theta * x + eta(z) + eps,    E[eps | x, z] = 0,

where ``theta`` is the finite-dimensional target and ``eta`` an unknown
nuisance function. The orthogonal (Robinson) moment

    psi(theta) = (y - E[y|z]) - theta * (x - E[x|z])

identifies ``theta`` as the ratio of the two residual projections,

    theta_hat = sum_i v_i u_i / sum_i v_i x_i,   v = x - E[x|z],  u = y - E[y|z],

with the nuisances ``E[y|z]`` and ``E[x|z]`` estimated by **cross-fitting**
(``folds >= 2``): first-order nuisance errors are ignorable, so the estimator
stays consistent even when the nuisance learner is a flexible nonparametric
model that converges slower than ``sqrt(n)``. The influence-function variance
gives asymptotically valid confidence intervals.

This is the estimator class behind the cosmodist implicit-nuisance spine
(``docs/design/2026-09-16-operator-parameterization.md``, workstream W-M):
selection, population, instrument, and lensing kernels are learned from the
same real data, while cosmology is estimated from an orthogonal moment.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

__all__ = [
    "OrthogonalEstimate",
    "median_heuristic_bandwidth",
    "naive_linear_estimate",
    "orthogonal_partially_linear",
    "random_fourier_features",
]

#: Candidate penalties for ``ridge="gcv"`` / ``"loo"``, as multiples of the
#: largest squared singular value of the centred training design.
_RIDGE_GRID = tuple(10.0 ** (-k / 2.0) for k in range(0, 25))


@dataclass(frozen=True)
class OrthogonalEstimate:
    """Point estimate and influence-function inference for ``theta``."""

    theta: float
    sigma: float
    ci_low: float
    ci_high: float
    n: int
    folds: int
    nuisance_r2_x: float
    nuisance_r2_y: float
    cross_fitted: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "theta": self.theta,
            "sigma": self.sigma,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "folds": self.folds,
            "nuisance_r2_x": self.nuisance_r2_x,
            "nuisance_r2_y": self.nuisance_r2_y,
            "cross_fitted": self.cross_fitted,
        }


def _to_1d(name: str, value: Tensor | list[float]) -> Tensor:
    tensor = value.detach() if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.to(dtype=torch.float64).reshape(-1)
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _poly_features(z: Tensor, degree: int) -> Tensor:
    columns = [torch.ones_like(z[:, :1])]
    for power in range(1, degree + 1):
        columns.append(z**power)
    return torch.cat(columns, dim=1)


def median_heuristic_bandwidth(
    z: Tensor | list[float],
    *,
    n_subsample: int = 1000,
    seed: int = 0,
) -> float:
    """Median pairwise Euclidean distance of the rows of ``z`` (kernel bandwidth).

    The classic "median heuristic" for a Gaussian/RBF kernel
    ``k(a, b) = exp(-||a - b||^2 / (2 sigma^2))``.  The median is taken over the
    distinct pairs of a seeded random subsample of at most ``n_subsample`` rows
    so the cost stays ``O(n_subsample^2)`` for large ``z``.

    Parameters
    ----------
    z : Tensor or list of float
        Points, shape ``[N]`` or ``[N, K]``.  Standardise first if the columns
        have different scales (:func:`random_fourier_features` does).
    n_subsample : int, default 1000
        Maximum number of rows used.
    seed : int, default 0
        Seed of the subsample draw.

    Returns
    -------
    float
        The median pairwise distance, or ``1.0`` when all sampled points
        coincide (or there are fewer than two rows).

    Raises
    ------
    ValueError
        If ``z`` contains NaN or infinite values.
    """
    if n_subsample < 2:
        raise ValueError("n_subsample must be at least 2")
    points = z.detach() if isinstance(z, Tensor) else torch.as_tensor(z)
    points = points.to(dtype=torch.float64)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    if points.ndim != 2:
        raise ValueError("z must have shape [N] or [N,K]")
    n_rows = points.shape[0]
    if n_rows > n_subsample:
        generator = torch.Generator().manual_seed(seed)
        points = points[torch.randperm(n_rows, generator=generator)[:n_subsample]]
    if not bool(torch.isfinite(points).all()):
        raise ValueError("z must be finite")
    if points.shape[0] < 2:
        return 1.0
    # Distances are translation invariant: centre first and avoid the matmul
    # expansion of ``cdist``, which cancels catastrophically for offset data.
    points = points - points.mean(dim=0, keepdim=True)
    distances = torch.cdist(points, points, compute_mode="donot_use_mm_for_euclid_dist")
    upper = torch.triu_indices(points.shape[0], points.shape[0], offset=1)
    median = float(distances[upper[0], upper[1]].median())
    return median if median > 0.0 else 1.0


def random_fourier_features(
    z: Tensor | list[float],
    *,
    n_features: int = 256,
    bandwidth: float | Literal["median"] = "median",
    standardize: bool = True,
    polynomial_degree: int = 3,
    seed: int = 0,
) -> Tensor:
    """Random Fourier features of a Gaussian (RBF) kernel, for use as a nuisance basis.

    Returns ``sqrt(2/D) cos(z W + b)`` with ``W ~ N(0, 1/sigma^2)`` and
    ``b ~ U(0, 2 pi)``, so that ``phi(a) . phi(b)`` approximates
    ``exp(-||a - b||^2 / (2 sigma^2))`` (Rahimi and Recht, 2007).  Pass the
    result (or a ``lambda z: random_fourier_features(z, ...)``) as
    ``nuisance_features`` to :func:`orthogonal_partially_linear`.

    Parameters
    ----------
    z : Tensor or list of float
        Nuisance covariates, shape ``[N]`` or ``[N, K]``.
    n_features : int, default 256
        Number of random features ``D``.
    bandwidth : float or "median", default "median"
        Kernel bandwidth ``sigma`` in the (optionally standardised) units of
        ``z``.  ``"median"`` uses :func:`median_heuristic_bandwidth` on the
        standardised ``z`` (seeded subsample), which adapts to the number of
        covariates.  A fixed unit bandwidth on 20 standardised covariates is
        far too narrow to share strength across observations.  The older
        hand-rolled recipe (``W ~ N(0, 1/K)``, no linear terms) corresponds to
        ``bandwidth=sqrt(K), standardize=False, polynomial_degree=0``.
    standardize : bool, default True
        Centre and scale each column of ``z`` to unit variance (constant
        columns are only centred) before applying the kernel.
    polynomial_degree : int, default 3
        Append the per-covariate powers ``z, z^2, ..., z^degree`` of the
        (standardised) ``z`` to the random features; ``0`` returns the bare
        random features.  A purely periodic basis cannot represent the smooth
        global trends that nuisance functions usually contain.  Over five
        confounding designs (300 replications, n = 500) the cubic default keeps
        95% interval coverage within 0.90-0.95 everywhere; degree 1 drops to
        0.78 on cubic confounding and 0.86 on smooth interactions, while gaining
        only 1-2.5 points on linear and CCDDHNR-2018 designs.
    seed : int, default 0
        Seed of ``W``, ``b`` and the bandwidth subsample.

    Returns
    -------
    Tensor
        ``float64`` feature matrix of shape ``[N, n_features + K * polynomial_degree]`` (or
        ``[N, n_features]`` with ``polynomial_degree=0``).
    """
    if n_features < 1:
        raise ValueError("n_features must be at least 1")
    if polynomial_degree < 0:
        raise ValueError("polynomial_degree must be non-negative")
    points = z.detach() if isinstance(z, Tensor) else torch.as_tensor(z)
    points = points.to(dtype=torch.float64)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    if points.ndim != 2 or points.shape[0] < 1:
        raise ValueError("z must have shape [N] or [N,K]")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("z must be finite")
    if standardize:
        centred = points - points.mean(dim=0, keepdim=True)
        scale = centred.std(dim=0, unbiased=False, keepdim=True)
        points = centred / torch.where(scale > 0.0, scale, torch.ones_like(scale))
    if isinstance(bandwidth, str):
        if bandwidth != "median":
            raise ValueError("bandwidth must be a positive float or 'median'")
        sigma = median_heuristic_bandwidth(points, seed=seed)
    else:
        sigma = float(bandwidth)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("bandwidth must be positive and finite")
    generator = torch.Generator().manual_seed(seed)
    weights = (
        torch.randn(points.shape[1], n_features, generator=generator, dtype=torch.float64) / sigma
    )
    phase = torch.rand(n_features, generator=generator, dtype=torch.float64) * (2.0 * math.pi)
    features = math.sqrt(2.0 / n_features) * torch.cos(points @ weights + phase)
    if polynomial_degree == 0:
        return features
    powers = [points**power for power in range(1, polynomial_degree + 1)]
    return torch.cat([features, *powers], dim=1)


def _select_ridge(centred: Tensor, target: Tensor, *, criterion: str) -> float:
    """Ridge penalty minimising GCV or exact leave-one-out error (closed form).

    ``centred`` and ``target`` are the training design / target with the column
    / target means removed (an unpenalised intercept), solved through one SVD.
    """
    u, singular, _ = torch.linalg.svd(centred, full_matrices=False)
    s2 = singular**2
    projected = u.T @ target
    n_rows = centred.shape[0]
    top = float(s2.max())
    if top <= 0.0:
        return 1.0
    best_penalty, best_score = top * _RIDGE_GRID[0], math.inf
    for fraction in _RIDGE_GRID:
        penalty = fraction * top
        shrink = s2 / (s2 + penalty)
        residual = target - u @ (shrink * projected)
        # Intercept adds one degree of freedom (leverage 1/n per row).
        leverage = (u**2 * shrink).sum(dim=1) + 1.0 / n_rows
        if criterion == "loo":
            score = float(((residual / (1.0 - leverage).clamp_min(1.0e-12)) ** 2).mean())
        else:
            dof = float(leverage.sum())
            score = float((residual**2).mean()) / max(1.0 - dof / n_rows, 1.0e-12) ** 2
        if score < best_score:
            best_score, best_penalty = score, penalty
    return best_penalty


def _ridge_fit_predict(
    features: Tensor,
    target: Tensor,
    *,
    ridge: float | Literal["gcv", "loo"],
    train: Tensor,
    test: Tensor,
) -> tuple[Tensor, float]:
    x_train = features[train]
    if isinstance(ridge, str):
        # Data-driven penalty on the training fold only; unpenalised intercept.
        mean_x = x_train.mean(dim=0, keepdim=True)
        mean_y = target[train].mean()
        centred = x_train - mean_x
        penalty = _select_ridge(centred, target[train] - mean_y, criterion=ridge)
        gram = centred.T @ centred + penalty * torch.eye(
            centred.shape[1], dtype=centred.dtype, device=centred.device
        )
        weights = torch.linalg.solve(gram, centred.T @ (target[train] - mean_y))
        predictions = (features[test] - mean_x) @ weights + mean_y
    else:
        gram = x_train.T @ x_train + ridge * torch.eye(
            x_train.shape[1], dtype=x_train.dtype, device=x_train.device
        )
        weights = torch.linalg.solve(gram, x_train.T @ target[train])
        predictions = features[test] @ weights
    residual = target[test] - predictions
    total = target[test] - target[test].mean()
    variance = float((total**2).sum())
    r2 = 1.0 - float((residual**2).sum()) / variance if variance > 0.0 else 0.0
    return predictions, r2


def _cross_fitted_residuals(
    features: Tensor,
    target: Tensor,
    *,
    folds: int,
    ridge: float | Literal["gcv", "loo"],
    order: Tensor,
) -> tuple[Tensor, float]:
    """Out-of-fold residuals of ``target`` on ``features``.

    ``order`` is the fold assignment (a permutation of the rows).  Both
    nuisances of one estimate must share it: with independent splits for
    ``E[x|z]`` and ``E[y|z]`` the product of the two residuals picks up a
    non-vanishing cross term and the estimate is biased (about -0.03 on the
    DoubleML CCDDHNR-2018 design at n = 500, several standard errors).
    """
    n_samples = features.shape[0]
    if folds < 2:
        train = torch.arange(n_samples)
        predictions, r2 = _ridge_fit_predict(features, target, ridge=ridge, train=train, test=train)
        return target - predictions, r2
    residual = torch.empty_like(target)
    r2_folds: list[float] = []
    for fold in range(folds):
        test = order[fold::folds]
        mask = torch.ones(n_samples, dtype=torch.bool)
        mask[test] = False
        train = mask.nonzero().flatten()
        predictions, r2 = _ridge_fit_predict(features, target, ridge=ridge, train=train, test=test)
        residual[test] = target[test] - predictions
        r2_folds.append(r2)
    return residual, sum(r2_folds) / len(r2_folds)


def orthogonal_partially_linear(
    y: Tensor | list[float],
    x: Tensor | list[float],
    z: Tensor | list[float],
    *,
    folds: int = 5,
    ridge: float | Literal["gcv", "loo"] = 1.0e-6,
    nuisance_degree: int = 3,
    nuisance_features: Callable[[Tensor], Tensor] | None = None,
    confidence: float = 0.95,
    seed: int = 0,
) -> OrthogonalEstimate:
    """Cross-fitted orthogonal estimate of ``theta`` in ``y = theta x + eta(z) + eps``.

    Args:
        y: Outcome vector ``[N]``.
        x: Treatment/feature vector ``[N]``.
        z: Nuisance covariate(s): ``[N]`` or ``[N, K]``.
        folds: Number of cross-fitting folds (``>= 2`` recommended; ``1`` gives
            the non-cross-fitted variant for diagnostics only).
        ridge: Ridge penalty for the nuisance regressions: a float (the
            default ``1e-6`` is a numerical jitter, the right choice for a
            low-dimensional polynomial basis), or ``"gcv"`` / ``"loo"`` to
            pick the penalty separately for every nuisance and training fold
            by generalized cross-validation / exact leave-one-out error
            (closed form from one SVD, unpenalised intercept).  Use it with
            rich bases such as :func:`random_fourier_features`, where a fixed
            penalty over- or under-smooths.
        nuisance_degree: Degree of the polynomial nuisance basis on ``z``
            (ignored when ``nuisance_features`` is given).
        nuisance_features: Optional callable mapping ``z`` ``[N, K]`` to a
            feature matrix ``[N, P]``; use for categorical/per-method
            nuisances (one-hot) or richer bases (splines, RBFs, or
            ``lambda z: random_fourier_features(z, seed=seed)``).
        confidence: Two-sided confidence level for the reported interval.
        seed: Seed for the fold assignment.

    Returns:
        :class:`OrthogonalEstimate` with the point estimate, influence-function
        standard error, and interval.
    """
    y_vec = _to_1d("y", y)
    x_vec = _to_1d("x", x)
    if y_vec.shape != x_vec.shape:
        raise ValueError("y and x must have the same length")
    z_tensor = z.detach() if isinstance(z, Tensor) else torch.as_tensor(z)
    z_tensor = z_tensor.to(dtype=torch.float64)
    if z_tensor.ndim == 1:
        z_tensor = z_tensor.reshape(-1, 1)
    if z_tensor.ndim != 2 or z_tensor.shape[0] != y_vec.numel():
        raise ValueError("z must have shape [N] or [N,K] matching y")
    if not bool(torch.isfinite(z_tensor).all()):
        raise ValueError("z must be finite")
    if folds < 1:
        raise ValueError("folds must be at least 1")
    if nuisance_degree < 0:
        raise ValueError("nuisance_degree must be non-negative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if isinstance(ridge, str):
        if ridge not in ("gcv", "loo"):
            raise ValueError("ridge must be a non-negative float, 'gcv' or 'loo'")
    elif not ridge >= 0.0:
        raise ValueError("ridge must be a non-negative float, 'gcv' or 'loo'")

    generator = torch.Generator().manual_seed(seed)
    if nuisance_features is not None:
        features = nuisance_features(z_tensor)
        if not torch.is_tensor(features):
            features = torch.as_tensor(features)
        features = features.to(dtype=torch.float64)
        if features.ndim != 2 or features.shape[0] != y_vec.numel():
            raise ValueError(
                f"nuisance_features must return shape [N,P] matching y, got {tuple(features.shape)}"
            )
        if features.shape[1] < 1:
            raise ValueError("nuisance_features must return at least one column")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("nuisance_features must be finite")
    else:
        features = _poly_features(z_tensor, nuisance_degree)
    # One fold assignment for both nuisances (see _cross_fitted_residuals).
    order = torch.randperm(y_vec.numel(), generator=generator)
    residual_x, r2_x = _cross_fitted_residuals(
        features, x_vec, folds=folds, ridge=ridge, order=order
    )
    residual_y, r2_y = _cross_fitted_residuals(
        features, y_vec, folds=folds, ridge=ridge, order=order
    )

    denominator = float((residual_x * residual_x).sum())
    # Centred: a constant offset in x changes neither theta nor identification.
    x_centred = x_vec - x_vec.mean()
    treatment_scale = float((x_centred * x_centred).sum())
    if (
        denominator <= 0.0
        or denominator <= 1.0e-8 * treatment_scale
        or treatment_scale <= 1.0e-24 * float((x_vec * x_vec).sum())
    ):
        raise ValueError("x is perfectly explained by z; theta is not identified")
    theta = float((residual_x * residual_y).sum()) / denominator

    scores = residual_x * (residual_y - theta * residual_x)
    variance = float((scores**2).sum()) / (denominator**2)
    sigma = variance**0.5
    z_value = _normal_quantile(0.5 + 0.5 * confidence)
    return OrthogonalEstimate(
        theta=theta,
        sigma=sigma,
        ci_low=theta - z_value * sigma,
        ci_high=theta + z_value * sigma,
        n=int(y_vec.numel()),
        folds=int(folds),
        nuisance_r2_x=float(r2_x),
        nuisance_r2_y=float(r2_y),
        cross_fitted=folds >= 2,
    )


def naive_linear_estimate(
    y: Tensor | list[float],
    x: Tensor | list[float],
    *,
    confidence: float = 0.95,
) -> OrthogonalEstimate:
    """Ordinary least squares of ``y`` on ``x`` ignoring the nuisance ``z``.

    Provided as the biased baseline that the orthogonal estimator must beat on
    confounded problems.
    """
    y_vec = _to_1d("y", y)
    x_vec = _to_1d("x", x)
    if y_vec.shape != x_vec.shape:
        raise ValueError("y and x must have the same length")
    design = torch.stack([torch.ones_like(x_vec), x_vec], dim=1)
    gram = design.T @ design
    weights = torch.linalg.solve(gram, design.T @ y_vec)
    theta = float(weights[1])
    residual = y_vec - design @ weights
    sigma = float(residual.std(unbiased=True)) / float(
        torch.sqrt((x_vec**2).sum() - x_vec.numel() * x_vec.mean() ** 2)
    )
    z_value = _normal_quantile(0.5 + 0.5 * confidence)
    return OrthogonalEstimate(
        theta=theta,
        sigma=sigma,
        ci_low=theta - z_value * sigma,
        ci_high=theta + z_value * sigma,
        n=int(y_vec.numel()),
        folds=1,
        nuisance_r2_x=0.0,
        nuisance_r2_y=0.0,
        cross_fitted=False,
    )


def _normal_quantile(probability: float) -> float:
    return math.sqrt(2.0) * float(torch.erfinv(torch.tensor(2.0 * probability - 1.0)).item())
