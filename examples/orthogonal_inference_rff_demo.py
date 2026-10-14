r"""
Orthogonal (double/debiased ML) inference with a random-Fourier-feature nuisance basis.

Partially linear model with a known target ``theta``:

.. math::

    y = \theta x + g(z) + \varepsilon, \qquad x = m(z) + v,

with five correlated covariates ``z`` and nuisances that share an interaction
``z_0 z_1`` (so ``x`` is confounded through a term an additive basis cannot fit):

.. math::

    m(z) = \tanh(z_0 z_1) + 0.5 \sin(2 z_2), \qquad
    g(z) = 1.5 \tanh(z_0 z_1) + 0.5 z_3^2 + 0.3 z_4 .

``orthogonal_partially_linear`` cross-fits ridge regressions for ``E[x|z]`` and
``E[y|z]`` and solves the Robinson moment
``theta_hat = sum v_i u_i / sum v_i^2`` with ``v = x - E[x|z]``, ``u = y - E[y|z]``.
Three estimators are run on the same replications:

* **naive OLS** - ``naive_linear_estimate(y, x)``, ignores ``z`` (biased baseline).
* **default call** - ``orthogonal_partially_linear(y, x, z)``: per-covariate cubic
  polynomial basis, ``ridge=1e-6``.
* **RFF + GCV** - ``nuisance_features`` = ``random_fourier_features`` of the
  standardised ``z`` with the ``median_heuristic_bandwidth``, plus per-covariate
  powers up to ``polynomial_degree=3``, and ``ridge="gcv"`` (penalty chosen per
  nuisance and training fold by generalized cross-validation).

Reported over the replications: bias of ``theta_hat``, its standard deviation,
mean 95% CI width, empirical 95% CI coverage and the cross-fitted nuisance R^2.

Run::

    python examples/orthogonal_inference_rff_demo.py --replications 40 --n 500 --seed 0

Caveats
-------
* Coverage from ``R`` replications has a binomial standard error of about
  ``sqrt(0.95 * 0.05 / R)`` (0.034 at ``R = 40``); raise ``--replications`` for a
  tighter estimate.
* The design is chosen so that the additive cubic basis is misspecified; on
  additive confounding the default call is already fine (see the CHANGELOG
  0.3.0 entry for the five-design study behind the defaults).
* The RFF draw, the bandwidth subsample and the fold assignment are seeded per
  replication, so each replication is reproducible.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from torchregress.comparison import print_comparison_summary
from torchregress.inference import (
    OrthogonalEstimate,
    median_heuristic_bandwidth,
    naive_linear_estimate,
    orthogonal_partially_linear,
    random_fourier_features,
)

THETA = 0.5


def simulate(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One draw of the partially linear design: ``(y, x, z)`` as float64 tensors."""
    rng = np.random.default_rng(seed)
    k = 5
    cov = 0.5 ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    z = rng.multivariate_normal(np.zeros(k), cov, size=n)
    inter = np.tanh(z[:, 0] * z[:, 1])
    m = inter + 0.5 * np.sin(2.0 * z[:, 2])
    g = 1.5 * inter + 0.5 * z[:, 3] ** 2 + 0.3 * z[:, 4]
    x = m + rng.standard_normal(n)
    y = THETA * x + g + rng.standard_normal(n)
    return torch.from_numpy(y), torch.from_numpy(x), torch.from_numpy(z)


def rff_estimate(
    y: torch.Tensor, x: torch.Tensor, z: torch.Tensor, *, seed: int, n_rff: int
) -> tuple[OrthogonalEstimate, float]:
    """Orthogonal estimate with an RFF + cubic nuisance basis and GCV ridge."""
    z_std = (z - z.mean(0)) / z.std(0, unbiased=False)
    # Bandwidth in standardised units (random_fourier_features standardises z too,
    # so this equals its bandwidth="median" default; computed here to show it).
    bandwidth = median_heuristic_bandwidth(z_std, seed=seed)

    def basis(zz: torch.Tensor) -> torch.Tensor:
        return random_fourier_features(
            zz, n_features=n_rff, bandwidth=bandwidth, polynomial_degree=3, seed=seed
        )

    est = orthogonal_partially_linear(y, x, z, nuisance_features=basis, ridge="gcv", seed=seed)
    return est, bandwidth


def summarise(name: str, estimates: list[OrthogonalEstimate], seconds: float) -> dict[str, object]:
    theta = np.array([e.theta for e in estimates])
    covered = np.array([e.ci_low <= THETA <= e.ci_high for e in estimates])
    return {
        "Method": name,
        "Bias": float(theta.mean() - THETA),
        "SD": float(theta.std(ddof=1)) if len(theta) > 1 else 0.0,
        "MeanSE": float(np.mean([e.sigma for e in estimates])),
        "CIWidth": float(np.mean([e.ci_high - e.ci_low for e in estimates])),
        "Coverage95": float(covered.mean()),
        "R2_x": float(np.mean([e.nuisance_r2_x for e in estimates])),
        "R2_y": float(np.mean([e.nuisance_r2_y for e in estimates])),
        "total_s": seconds,
    }


def main(
    replications: int = 40, n: int = 500, seed: int = 0, n_rff: int = 256
) -> list[dict[str, object]]:
    print(
        f"Partially linear design, theta = {THETA}, n = {n}, {replications} replications, "
        f"5 covariates, 5-fold cross-fitting\n"
    )
    naive, default, rff = [], [], []
    times = {"naive": 0.0, "default": 0.0, "rff": 0.0}
    bandwidths = []
    for r in range(replications):
        y, x, z = simulate(n, seed + r)
        t0 = time.perf_counter()
        naive.append(naive_linear_estimate(y, x))
        t1 = time.perf_counter()
        default.append(orthogonal_partially_linear(y, x, z, seed=seed + r))
        t2 = time.perf_counter()
        est, bw = rff_estimate(y, x, z, seed=seed + r, n_rff=n_rff)
        t3 = time.perf_counter()
        rff.append(est)
        bandwidths.append(bw)
        times["naive"] += t1 - t0
        times["default"] += t2 - t1
        times["rff"] += t3 - t2
        if r == 0:
            print("Replication 0:")
            print(f"  median_heuristic_bandwidth(z_std) = {bw:.3f}")
            for label, e in (("naive OLS", naive[0]), ("default", default[0]), ("RFF+GCV", est)):
                print(
                    f"  {label:<10} theta = {e.theta:+.3f}  "
                    f"95% CI [{e.ci_low:+.3f}, {e.ci_high:+.3f}]"
                )

    rows = [
        summarise("naive OLS (ignores z)", naive, times["naive"]),
        summarise("default (cubic poly, ridge=1e-6)", default, times["default"]),
        summarise(f"RFF({n_rff}) + median bw + cubic, ridge='gcv'", rff, times["rff"]),
    ]
    print(f"\nMean median-heuristic bandwidth: {np.mean(bandwidths):.3f}")
    print_comparison_summary(
        f"Orthogonal inference for theta = {THETA} over {replications} replications",
        rows,
        metric_order=["Bias", "SD", "MeanSE", "CIWidth", "Coverage95", "R2_x", "R2_y", "total_s"],
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orthogonal inference with an RFF nuisance basis")
    parser.add_argument("--replications", type=int, default=40)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--n-rff", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(replications=args.replications, n=args.n, seed=args.seed, n_rff=args.n_rff)
