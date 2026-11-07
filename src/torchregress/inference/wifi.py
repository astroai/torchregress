"""
wifi ensembles: frequentist uncertainties on neural density ratios.

Benevedes & Thaler (Phys. Rev. D 112, 056024, 2025; arXiv:2506.00113) model a
log density ratio ``log r(x) = log p(x) / q(x)`` as a linear combination of
``M`` frozen, independently trained ratio estimators ``f_i(x)`` plus a constant:

.. math::

    \\log r_w(x) = w_0 + \\sum_{i=1}^{M} w_i f_i(x).

The weights are fitted on data *not* used to train the ``f_i``, by minimising
the symmetrised maximum-likelihood-classifier (MLC) loss

.. math::

    L(w) = \\mathbb{E}_{p}\\bigl[-s_w(x) + e^{-s_w(x)}\\bigr]
         + \\mathbb{E}_{q}\\bigl[\\,s_w(x) + e^{s_w(x)}\\bigr],
    \\qquad s_w = \\log r_w,

whose pointwise minimiser is ``s = log p/q``.  ``L`` is convex in ``w``, so a
Newton solve is reliable.  ``w`` is an M-estimator, so its covariance is the
sandwich ``Sigma = H^{-1} V H^{-1}`` with ``H`` the Hessian of ``L`` and ``V``
the covariance of the per-sample score contributions of the two independent
samples.  The uncertainty on the log ratio at ``x`` follows as
``Var[log r(x)] = F(x)^T Sigma F(x)`` with ``F = [1, f_1, ..., f_M]``.

The intervals are asymptotically valid when the model is well specified (the
span of ``{1, f_i}`` contains the true log ratio), which is why the paper uses
large ensembles (M = 16-32) of well-trained, diverse members (bootstrap or
partitioned training data).

.. note::
    The equations above are a reconstruction from the paper's text (the
    symmetrised MLC objective, sandwich covariance and Gong-Samaniego
    propagation); the implementation is validated by coverage tests on
    problems with a known ratio, not by comparison with the authors' code.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

__all__ = ["MixtureFractionEstimate", "WifiRatioEnsemble"]


@dataclass(frozen=True)
class MixtureFractionEstimate:
    """Pseudo-MLE of a mixture fraction with its frequentist uncertainty.

    Attributes:
        fraction: Estimated mixture fraction ``kappa``.
        std: Total standard error, ``sqrt(std_statistical**2 + std_ratio**2)``.
        std_statistical: Standard error from the finite mixture sample.
        std_ratio: Additional standard error from the uncertainty of the
            learned ratio (Gong-Samaniego term).
        at_boundary: ``True`` when the estimate sits on the edge of the
            feasible interval (asymptotic normality then does not hold).
    """

    fraction: float
    std: float
    std_statistical: float
    std_ratio: float
    at_boundary: bool


def _as_features(features: Tensor, name: str, n_members: Optional[int] = None) -> Tensor:
    if not isinstance(features, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(features)}")
    if features.dim() == 1:
        features = features.unsqueeze(-1)
    if features.dim() != 2:
        raise ValueError(f"{name} must have shape [N, M], got {tuple(features.shape)}")
    if features.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row")
    if n_members is not None and features.shape[1] != n_members:
        raise ValueError(f"{name} must have {n_members} member columns, got {features.shape[1]}")
    features = features.detach().to(torch.float64)
    if not torch.isfinite(features).all():
        raise ValueError(f"{name} must be finite")
    return features


class WifiRatioEnsemble:
    """wifi ensemble: weighted log-ratio ensemble with frequentist error bars.

    Args:
        max_iter: Maximum Newton iterations for the weight fit.
        tol: Convergence tolerance on the Newton decrement.
        rcond: Relative cut-off for the pseudo-inverse of the Hessian.  Nearly
            collinear members (common with bootstrap ensembles) make individual
            weights poorly identified; predictions ``log r(x)`` and their
            variances stay well defined in the span of the members.

    Usage::

        # f_* are [N, M] log-ratio outputs of M frozen members (e.g. trained
        # NeuralRatioLoss classifiers on bootstrap resamples), evaluated on a
        # held-out numerator sample, denominator sample and test points.
        wifi = WifiRatioEnsemble().fit(f_numerator, f_denominator)
        log_r = wifi.log_ratio(f_test)
        log_r_std = wifi.log_ratio_std(f_test)
        kappa = wifi.mixture_fraction(f_mixture)

    Use :meth:`member_features` to stack member outputs into ``[N, M]``.
    """

    def __init__(self, *, max_iter: int = 100, tol: float = 1e-12, rcond: float = 1e-10) -> None:
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")
        if not tol > 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if not 0 < rcond < 1:
            raise ValueError(f"rcond must lie in (0, 1), got {rcond}")
        self.max_iter = max_iter
        self.tol = float(tol)
        self.rcond = float(rcond)
        self.weights_: Optional[Tensor] = None
        self.covariance_: Optional[Tensor] = None
        self.n_iter_: int = 0
        self.converged_: bool = False

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def member_features(members: Sequence[Callable[..., Tensor]], *inputs: Tensor) -> Tensor:
        """Stack member log-ratio outputs into ``[N, M]`` (float64, no grad).

        Each member is called as ``member(*inputs)`` and must return ``N``
        log-ratio values (``[N]`` or ``[N, 1]``).  For a parameter-conditioned
        ratio pass ``(theta, x)`` and members such as
        ``lambda c, x: neural_ratio_logits(classifier, c, x)``.
        """
        if len(members) < 1:
            raise ValueError("members must contain at least one estimator")
        with torch.no_grad():
            columns = [member(*inputs).reshape(-1).to(torch.float64) for member in members]
        lengths = {column.shape[0] for column in columns}
        if len(lengths) != 1:
            raise ValueError(f"members returned different numbers of rows: {sorted(lengths)}")
        return torch.stack(columns, dim=1)

    @property
    def n_members(self) -> int:
        """Number of ensemble members ``M`` (excluding the constant)."""
        self._check_fitted()
        assert self.weights_ is not None
        return int(self.weights_.shape[0]) - 1

    def _check_fitted(self) -> None:
        if self.weights_ is None or self.covariance_ is None:
            raise RuntimeError("WifiRatioEnsemble is not fitted; call fit() first")

    @staticmethod
    def _design(features: Tensor) -> Tensor:
        ones = torch.ones(features.shape[0], 1, dtype=features.dtype, device=features.device)
        return torch.cat([ones, features], dim=1)

    def _pinv(self, matrix: Tensor) -> Tensor:
        return torch.linalg.pinv(matrix, rtol=self.rcond, hermitian=True)

    # --------------------------------------------------------------------- fit

    @staticmethod
    def _loss(design_p: Tensor, design_q: Tensor, weights: Tensor) -> Tensor:
        s_p = design_p @ weights
        s_q = design_q @ weights
        return (torch.exp(-s_p) - s_p).mean() + (torch.exp(s_q) + s_q).mean()

    def fit(self, numerator_features: Tensor, denominator_features: Tensor) -> "WifiRatioEnsemble":
        """Fit the weights and their sandwich covariance.

        Args:
            numerator_features: Member outputs ``f_i(x)`` on held-out draws
                ``x ~ p`` (the ratio numerator), shape ``[N_p, M]``.
            denominator_features: Member outputs on held-out draws ``x ~ q``,
                shape ``[N_q, M]``.

        Returns:
            ``self``.
        """
        f_p = _as_features(numerator_features, "numerator_features")
        f_q = _as_features(denominator_features, "denominator_features", f_p.shape[1])
        x_p, x_q = self._design(f_p), self._design(f_q)
        n_p, n_q = x_p.shape[0], x_q.shape[0]

        # Start from the naive equal-weight ensemble (w_0 = 0, w_i = 1/M).
        n_members = f_p.shape[1]
        weights = torch.full(
            (n_members + 1,), 1.0 / n_members, dtype=torch.float64, device=f_p.device
        )
        weights[0] = 0.0
        loss = self._loss(x_p, x_q, weights)
        self.converged_ = False
        iteration = 0
        for iteration in range(1, self.max_iter + 1):
            e_p = torch.exp(-(x_p @ weights))
            e_q = torch.exp(x_q @ weights)
            gradient = x_p.T @ (-1.0 - e_p) / n_p + x_q.T @ (1.0 + e_q) / n_q
            hessian = (x_p.T * e_p) @ x_p / n_p + (x_q.T * e_q) @ x_q / n_q
            step = -self._pinv(hessian) @ gradient
            decrement = float(-(gradient @ step))
            if decrement / 2.0 <= self.tol:
                self.converged_ = True
                break
            # Backtracking line search (convex objective).
            scale = 1.0
            while scale > 1e-12:
                candidate = weights + scale * step
                candidate_loss = self._loss(x_p, x_q, candidate)
                if (
                    torch.isfinite(candidate_loss)
                    and candidate_loss <= loss - 0.25 * scale * decrement
                ):
                    break
                scale *= 0.5
            else:
                break
            weights, loss = candidate, candidate_loss
        self.n_iter_ = iteration

        # Sandwich covariance at the optimum.
        e_p = torch.exp(-(x_p @ weights))
        e_q = torch.exp(x_q @ weights)
        hessian = (x_p.T * e_p) @ x_p / n_p + (x_q.T * e_q) @ x_q / n_q
        score_p = x_p * (-1.0 - e_p).unsqueeze(1)
        score_q = x_q * (1.0 + e_q).unsqueeze(1)
        meat = _covariance(score_p) / n_p + _covariance(score_q) / n_q
        bread = self._pinv(hessian)
        covariance = bread @ meat @ bread
        self.weights_ = weights
        self.covariance_ = 0.5 * (covariance + covariance.T)
        return self

    # ---------------------------------------------------------------- predict

    def log_ratio(self, features: Tensor) -> Tensor:
        """Fitted ``log r(x) = w_0 + sum_i w_i f_i(x)``, shape ``[N]`` (float64)."""
        self._check_fitted()
        assert self.weights_ is not None
        design = self._design(_as_features(features, "features", self.n_members))
        return design @ self.weights_

    def log_ratio_std(self, features: Tensor) -> Tensor:
        """Standard error of :meth:`log_ratio` from the weight covariance, ``[N]``."""
        self._check_fitted()
        assert self.covariance_ is not None
        design = self._design(_as_features(features, "features", self.n_members))
        variance = ((design @ self.covariance_) * design).sum(dim=1)
        return variance.clamp_min(0.0).sqrt()

    def mixture_fraction(self, mixture_features: Tensor) -> MixtureFractionEstimate:
        """Estimate ``kappa`` in ``x ~ kappa p + (1 - kappa) q`` with error bars.

        The likelihood ratio of the mixture to ``q`` is ``1 + kappa (r(x) - 1)``,
        so the pseudo-MLE maximises ``sum_j log(1 + kappa (r_j - 1))`` (concave
        in ``kappa``) over the interval where every term is positive.  The
        standard error combines the sandwich variance of the mixture sample with
        the ratio uncertainty propagated by Gong & Samaniego (1981):

        .. math::

            \\mathrm{Var}(\\hat\\kappa) = \\frac{J + c^\\top \\Sigma_w c}{I^2},

        with ``I = sum_j (r_j - 1)^2 / D_j^2``, ``J = sum_j ((r_j - 1)/D_j)^2``,
        ``c = sum_j r_j F_j / D_j^2`` and ``D_j = 1 + kappa (r_j - 1)``.

        Args:
            mixture_features: Member outputs on the observed mixture sample,
                ``[N, M]``, disjoint from the data used by :meth:`fit`.
        """
        self._check_fitted()
        assert self.weights_ is not None and self.covariance_ is not None
        design = self._design(_as_features(mixture_features, "mixture_features", self.n_members))
        ratio = torch.exp(design @ self.weights_)
        delta = ratio - 1.0
        if bool((delta.abs() < 1e-15).all()):
            raise ValueError(
                "the fitted ratio is constant on the mixture sample; kappa is unidentified"
            )

        positive, negative = delta[delta > 0], delta[delta < 0]
        lower = float((-1.0 / positive).max()) if positive.numel() else -math.inf
        upper = float((-1.0 / negative).min()) if negative.numel() else math.inf

        def score(kappa: float) -> float:
            return float((delta / (1.0 + kappa * delta)).sum())

        # The score is strictly decreasing on (lower, upper).
        lo = lower + 1e-12 * max(1.0, abs(lower)) if math.isfinite(lower) else -1.0
        hi = upper - 1e-12 * max(1.0, abs(upper)) if math.isfinite(upper) else 1.0
        # With every r_j on one side of 1 the interval is unbounded on that side
        # and the likelihood can increase without limit; search no further than
        # |kappa| = _KAPPA_SEARCH_LIMIT and report the estimate as at_boundary.
        while not math.isfinite(lower) and score(lo) < 0 and lo > -_KAPPA_SEARCH_LIMIT:
            lo = max(2.0 * lo, -_KAPPA_SEARCH_LIMIT)
        while not math.isfinite(upper) and score(hi) > 0 and hi < _KAPPA_SEARCH_LIMIT:
            hi = min(2.0 * hi, _KAPPA_SEARCH_LIMIT)
        at_boundary = False
        if score(lo) <= 0:
            kappa, at_boundary = lo, True
        elif score(hi) >= 0:
            kappa, at_boundary = hi, True
        else:
            for _ in range(200):
                middle = 0.5 * (lo + hi)
                if score(middle) > 0:
                    lo = middle
                else:
                    hi = middle
            kappa = 0.5 * (lo + hi)

        denominator = 1.0 + kappa * delta
        information = float((delta.square() / denominator.square()).sum())
        meat = float((delta / denominator).square().sum())
        cross = (design * (ratio / denominator.square()).unsqueeze(1)).sum(dim=0)
        ratio_term = float(cross @ self.covariance_ @ cross)
        if information > 0.0:
            std_statistical = math.sqrt(meat) / information
            std_ratio = math.sqrt(max(ratio_term, 0.0)) / information
        else:  # no curvature left at the search limit: kappa is not determined
            std_statistical = std_ratio = math.inf
        return MixtureFractionEstimate(
            fraction=kappa,
            std=math.hypot(std_statistical, std_ratio),
            std_statistical=std_statistical,
            std_ratio=std_ratio,
            at_boundary=at_boundary,
        )


#: Largest |kappa| searched when the feasible interval is unbounded on a side.
_KAPPA_SEARCH_LIMIT = 1e6


def _covariance(rows: Tensor) -> Tensor:
    """Empirical covariance of the rows of ``[N, K]`` (``N - 1`` denominator)."""
    if rows.shape[0] < 2:
        return torch.zeros(rows.shape[1], rows.shape[1], dtype=rows.dtype, device=rows.device)
    centred = rows - rows.mean(dim=0, keepdim=True)
    return centred.T @ centred / (rows.shape[0] - 1)
