"""
Unit-integral B-spline (M-spline) basis for simplex-parameterised 1D densities.

A density on ``[lo, hi]`` is written as ``p(x) = sum_m c_m M_m(x)`` where the
coefficients ``c`` lie on the probability simplex and every basis function
integrates to one.  With that normalisation the simplex constraint alone
guarantees ``p >= 0`` and ``int p dx = 1``; no separate normalising pass is
needed after the network head.

Standard B-splines ``B_{m,k}`` of order ``k`` (degree ``k-1``) on a knot
vector ``t`` satisfy ``sum_m B_{m,k}(x) = 1`` (partition of unity), not unit
integral.  The unit-integral member of the family is the M-spline
(Ramsay 1988):

    M_{m,k}(x) = k * B_{m,k}(x) / (t_{m+k} - t_m),     int M_{m,k} dx = 1.

This module evaluates ``M_{m,k}`` with the Cox-de Boor recursion on an open
(clamped) knot vector, and integrates it exactly over arbitrary bins with
Gauss-Legendre quadrature applied piecewise between knots (a piecewise
polynomial of degree ``k-1`` is integrated exactly by ``ceil(k/2)`` nodes).

The class is differentiable with respect to the coefficients that multiply
the basis (that is the use-case: a network predicts ``c``), not with respect
to the knots.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

__all__ = ["BSplineDensityBasis"]


def _as_1d_tensor(values: Tensor | Sequence[float] | np.ndarray, *, name: str) -> Tensor:
    t = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    if t.numel() < 2:
        raise ValueError(f"{name} must contain at least two points")
    if not torch.isfinite(t).all():
        raise ValueError(f"{name} must be finite")
    if not torch.all(t[1:] > t[:-1]):
        raise ValueError(f"{name} must be strictly increasing")
    return t


class BSplineDensityBasis:
    """Unit-integral (M-spline) B-spline basis on a clamped knot vector.

    Args:
        breakpoints: Strictly increasing 1D sequence ``[lo, b_1, ..., hi]`` of
            distinct knot locations.  The open knot vector repeats ``lo`` and
            ``hi`` ``degree + 1`` times, so the basis spans exactly the interval
            ``[lo, hi]`` and every function vanishes outside it.
        degree: Polynomial degree (3 = cubic).  Order ``k = degree + 1``.

    The number of basis functions is ``n_basis = len(breakpoints) - 1 + degree``.

    Example:
        >>> basis = BSplineDensityBasis.from_uniform(0.0, 3.0, n_intervals=6)
        >>> basis.n_basis
        9
        >>> z = torch.linspace(0.0, 3.0, 5)
        >>> basis.evaluate(z).shape
        torch.Size([5, 9])
    """

    def __init__(self, breakpoints: Tensor | Sequence[float] | np.ndarray, degree: int = 3) -> None:
        if degree < 0:
            raise ValueError("degree must be non-negative")
        self.degree = int(degree)
        self.order = self.degree + 1
        self.breakpoints = _as_1d_tensor(breakpoints, name="breakpoints")
        lo = self.breakpoints[:1].repeat(self.degree)
        hi = self.breakpoints[-1:].repeat(self.degree)
        # Clamped knot vector: lo x (degree+1), interior, hi x (degree+1).
        self.knots = torch.cat([lo, self.breakpoints, hi])
        self.n_basis = self.knots.numel() - self.order
        # Unit-integral normalisation: int B_{m,k} = (t_{m+k} - t_m) / k.
        support = self.knots[self.order :] - self.knots[: -self.order]
        self._scale = self.order / support
        self._bin_integrals_cache: dict[tuple[float, ...], Tensor] = {}

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_uniform(
        cls, lo: float, hi: float, n_intervals: int, degree: int = 3
    ) -> BSplineDensityBasis:
        """Equally spaced breakpoints on ``[lo, hi]``."""
        if n_intervals < 1:
            raise ValueError("n_intervals must be >= 1")
        return cls(
            torch.linspace(float(lo), float(hi), int(n_intervals) + 1, dtype=torch.float64), degree
        )

    @classmethod
    def from_quantiles(
        cls,
        samples: Tensor | np.ndarray | Sequence[float],
        n_interior: int,
        lo: float,
        hi: float,
        degree: int = 3,
        min_spacing: float = 1e-6,
    ) -> BSplineDensityBasis:
        """Interior breakpoints at empirical quantiles of ``samples``.

        Quantile placement equalises the expected number of training labels
        per knot interval, which balances gradient flow across the support.
        Interior knots that fall outside ``(lo, hi)`` or closer than
        ``min_spacing`` to a neighbour are dropped.
        """
        if n_interior < 0:
            raise ValueError("n_interior must be >= 0")
        x = torch.as_tensor(np.asarray(samples), dtype=torch.float64).reshape(-1)
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            raise ValueError("samples must contain finite values")
        if n_interior == 0:
            interior = x.new_empty(0)
        else:
            q = torch.linspace(0.0, 1.0, n_interior + 2, dtype=torch.float64)[1:-1]
            interior = torch.quantile(x, q)
        pts = [float(lo)]
        for v in interior.tolist():
            if v - pts[-1] > min_spacing and hi - v > min_spacing:
                pts.append(v)
        pts.append(float(hi))
        return cls(torch.tensor(pts, dtype=torch.float64), degree)

    # ------------------------------------------------------------------ helpers
    @property
    def lo(self) -> float:
        return float(self.breakpoints[0])

    @property
    def hi(self) -> float:
        return float(self.breakpoints[-1])

    def _knots_like(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self.knots.to(device=x.device, dtype=x.dtype),
            self._scale.to(device=x.device, dtype=x.dtype),
        )

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, x: Tensor, *, normalized: bool = True) -> Tensor:
        """Evaluate the basis at ``x``.

        Args:
            x: Tensor of any shape ``(...)``.
            normalized: If ``True`` (default) return unit-integral M-splines;
                if ``False`` return the raw partition-of-unity B-splines.

        Returns:
            Tensor of shape ``(..., n_basis)``.  Zero outside ``[lo, hi]``;
            the right endpoint ``hi`` is included in the support.
        """
        t, scale = self._knots_like(x)
        shape = x.shape
        z = x.reshape(-1, 1)
        left = t[:-1]
        right = t[1:]
        # Degree-0 indicator on half-open intervals [t_i, t_{i+1}); closed at hi.
        b = ((z >= left) & (z < right)).to(z.dtype)
        at_hi = (z == t[-1]).squeeze(-1)
        if bool(at_hi.any()):
            last = int(torch.nonzero(right > left).max())
            b[at_hi, last] = 1.0
        for p in range(1, self.order):
            den_l = t[p:-1] - t[: -p - 1]
            den_r = t[p + 1 :] - t[1:-p]
            num_l = z - t[: -p - 1]
            num_r = t[p + 1 :] - z
            w_l = torch.where(
                den_l > 0,
                num_l / den_l.clamp_min(torch.finfo(z.dtype).tiny),
                torch.zeros_like(num_l),
            )
            w_r = torch.where(
                den_r > 0,
                num_r / den_r.clamp_min(torch.finfo(z.dtype).tiny),
                torch.zeros_like(num_r),
            )
            b = w_l * b[:, :-1] + w_r * b[:, 1:]
        if normalized:
            b = b * scale
        return b.reshape(*shape, self.n_basis)

    def density(self, coefficients: Tensor, x: Tensor) -> Tensor:
        """``p(x) = sum_m c_m M_m(x)`` for coefficients ``(..., n_basis)`` and grid ``x`` ``(G,)``.

        Returns shape ``(..., G)``.
        """
        basis = self.evaluate(x.to(coefficients.dtype)).to(coefficients.device)  # (G, M)
        return coefficients @ basis.transpose(-1, -2)

    # ------------------------------------------------------------------ integration
    def _gauss_legendre(
        self, dtype: torch.dtype, device: torch.device, extra_degree: int = 0
    ) -> tuple[Tensor, Tensor]:
        # Exact for polynomials of degree <= 2n - 1.
        n = max(1, math.ceil((self.degree + extra_degree + 1) / 2))
        nodes, weights = np.polynomial.legendre.leggauss(n)
        return (
            torch.as_tensor(nodes, dtype=dtype, device=device),
            torch.as_tensor(weights, dtype=dtype, device=device),
        )

    def _piece_integrals(self, edges: Tensor, moment: int = 0) -> tuple[Tensor, Tensor]:
        """Exact integrals of ``x^moment M_m(x)`` over the pieces of ``edges`` refined by the knots.

        Returns ``(pieces, integrals)`` with ``pieces`` of shape ``(P + 1,)`` and
        ``integrals`` of shape ``(P, n_basis)``.
        """
        t = self.breakpoints.to(device=edges.device, dtype=edges.dtype)
        inner = t[(t > edges[0]) & (t < edges[-1])]
        pieces = torch.unique(torch.cat([edges, inner]))  # sorted, unique
        a = pieces[:-1]
        b = pieces[1:]
        nodes, weights = self._gauss_legendre(edges.dtype, edges.device, extra_degree=moment)
        half = 0.5 * (b - a)  # (P,)
        mid = 0.5 * (b + a)
        x = mid[:, None] + half[:, None] * nodes[None, :]  # (P, n)
        vals = self.evaluate(x)  # (P, n, M)
        if moment:
            vals = vals * x.unsqueeze(-1) ** moment
        integrals = (vals * weights[None, :, None]).sum(dim=1) * half[:, None]
        return pieces, integrals

    def bin_integrals(self, edges: Tensor | Sequence[float]) -> Tensor:
        """``T[k, m] = int_{e_k}^{e_{k+1}} M_m(x) dx`` for sorted bin edges.

        Rows therefore map simplex coefficients to bin masses:
        ``mass = coefficients @ T.T``.  If the edges cover ``[lo, hi]`` every
        column sums to one.  Exact to floating-point round-off.
        """
        e = _as_1d_tensor(edges, name="edges")
        key = tuple(e.tolist())
        cached = self._bin_integrals_cache.get(key)
        if cached is not None:
            return cached
        pieces, integrals = self._piece_integrals(e)
        mids = 0.5 * (pieces[:-1] + pieces[1:])
        idx = torch.bucketize(mids, e) - 1  # piece -> bin
        T = torch.zeros(e.numel() - 1, self.n_basis, dtype=e.dtype)
        T.index_add_(0, idx, integrals)
        self._bin_integrals_cache[key] = T
        return T

    def cumulative_moment(self, x: Tensor, moment: int = 0) -> Tensor:
        """``int_{lo}^{x} u^moment M_m(u) du`` for a 1D tensor ``x``; shape ``(len(x), n_basis)``.

        ``moment=0`` is the CDF of each basis density.  Evaluation points are
        clamped to ``[lo, hi]`` first, so the result is 0 below the support and
        the full moment above it.  Exact (piecewise Gauss-Legendre).
        """
        if x.dim() != 1:
            raise ValueError("cumulative_moment expects a 1D tensor of evaluation points")
        xc = x.detach().to(torch.float64).clamp(self.lo, self.hi)
        uniq, inverse = torch.unique(xc, return_inverse=True)
        pts = torch.unique(torch.cat([self.breakpoints[:1], uniq]))
        pieces, integrals = self._piece_integrals(pts, moment=moment)
        cum = torch.cat(
            [integrals.new_zeros(1, self.n_basis), integrals.cumsum(0)]
        )  # (P+1, M) at pieces
        pos = torch.searchsorted(pieces, uniq)
        return cum[pos][inverse].to(device=x.device, dtype=x.dtype)

    def cdf(self, x: Tensor) -> Tensor:
        """``F_m(x) = int_{lo}^{x} M_m`` for a 1D tensor ``x``; shape ``(len(x), n_basis)``."""
        return self.cumulative_moment(x, moment=0)

    def basis_means(self) -> Tensor:
        """``int x M_m(x) dx`` for every basis function (the mean of each basis density)."""
        e = self.breakpoints[[0, -1]]
        _, integrals = self._piece_integrals(e, moment=1)
        return integrals.sum(0)

    def absolute_deviation(self, x: Tensor) -> Tensor:
        """``E_{M_m}|U - x| = int |u - x| M_m(u) du`` for 1D ``x``; shape ``(len(x), n_basis)``.

        For a simplex-weighted density ``p = sum_m c_m M_m`` this gives the exact
        1-Wasserstein distance between ``p`` and a point mass at ``x``:
        ``W1(p, delta_x) = E_p|U - x| = c @ absolute_deviation(x).T``.
        Uses ``E|U-x| = m1 - 2 F1(x) + x (2 F0(x) - 1)`` with the cumulative
        moments ``F0``, ``F1`` and the basis means ``m1``.
        """
        x64 = x.detach().to(torch.float64)
        F0 = self.cumulative_moment(x64, 0)
        F1 = self.cumulative_moment(x64, 1)
        m1 = self.basis_means().to(F0)
        xc = x64.clamp(self.lo, self.hi)
        out = m1[None, :] - 2.0 * F1 + xc[:, None] * (2.0 * F0 - 1.0)
        # Outside the support the identity still holds with clamped cumulatives
        # plus the extra distance to the support edge.
        out = out + (x64 - xc).abs()[:, None]
        return out.to(device=x.device, dtype=x.dtype)

    def __repr__(self) -> str:
        return (
            f"BSplineDensityBasis(degree={self.degree}, n_basis={self.n_basis}, "
            f"range=[{self.lo:g}, {self.hi:g}], n_intervals={self.breakpoints.numel() - 1})"
        )
