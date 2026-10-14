"""Tests for Neyman-orthogonal (double/debiased ML) partially linear estimation."""

from __future__ import annotations

import math

import pytest
import torch

from torchregress.inference.orthogonal import (
    OrthogonalEstimate,
    median_heuristic_bandwidth,
    naive_linear_estimate,
    orthogonal_partially_linear,
    random_fourier_features,
)


def _confounded_sample(
    n: int = 4000, theta: float = 1.0, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    z = torch.rand(n, generator=generator, dtype=torch.float64)
    eta = z + z**2 + 0.5 * z**3
    x = 2.0 * eta + 0.5 * torch.randn(n, generator=generator, dtype=torch.float64)
    y = theta * x + eta + 0.5 * torch.randn(n, generator=generator, dtype=torch.float64)
    return y, x, z


def _smooth_nuisance_sample(
    n: int = 4000, theta: float = 1.0, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    z = torch.rand(n, generator=generator, dtype=torch.float64)
    eta = torch.sin(2.0 * math.pi * z) + z**2
    x = 2.0 * eta + 0.5 * torch.randn(n, generator=generator, dtype=torch.float64)
    y = theta * x + eta + 0.5 * torch.randn(n, generator=generator, dtype=torch.float64)
    return y, x, z


def test_orthogonal_estimator_beats_the_confounded_naive_estimator():
    y, x, z = _confounded_sample(theta=1.0, seed=0)
    naive = naive_linear_estimate(y, x)
    orthogonal = orthogonal_partially_linear(y, x, z, folds=5, seed=0)
    assert abs(naive.theta - 1.0) > 0.2
    assert abs(orthogonal.theta - 1.0) < 3.0 * orthogonal.sigma
    assert orthogonal.nuisance_r2_x > 0.5
    assert orthogonal.nuisance_r2_y > 0.5
    assert orthogonal.cross_fitted


def test_orthogonal_interval_covers_truth_across_seeds():
    covered = 0
    reps = 40
    for seed in range(reps):
        y, x, z = _confounded_sample(n=1500, theta=0.7, seed=seed)
        estimate = orthogonal_partially_linear(y, x, z, folds=5, seed=seed)
        if estimate.ci_low <= 0.7 <= estimate.ci_high:
            covered += 1
    assert covered >= 0.85 * reps


def test_misspecified_nuisance_basis_leaves_persistent_bias():
    rigid = orthogonal_partially_linear(
        *_smooth_nuisance_sample(n=24000, seed=13), folds=5, nuisance_degree=3, seed=13
    )
    assert abs(rigid.theta - 1.0) > 3.0 * rigid.sigma
    flexible = orthogonal_partially_linear(
        *_smooth_nuisance_sample(n=24000, seed=13), folds=5, nuisance_degree=8, seed=13
    )
    assert abs(flexible.theta - 1.0) < 3.0 * flexible.sigma
    assert flexible.ci_low <= 1.0 <= flexible.ci_high


def test_orthogonal_estimator_handles_multidimensional_nuisance():
    generator = torch.Generator().manual_seed(1)
    n = 3000
    z = torch.rand(n, 2, generator=generator, dtype=torch.float64)
    eta = torch.sin(2.0 * math.pi * z[:, 0]) + 0.5 * z[:, 1] ** 2
    x = 1.5 * eta + 0.4 * torch.randn(n, generator=generator, dtype=torch.float64)
    y = 0.5 * x + eta + 0.4 * torch.randn(n, generator=generator, dtype=torch.float64)
    estimate = orthogonal_partially_linear(y, x, z, folds=5, seed=3)
    assert abs(estimate.theta - 0.5) < 3.0 * estimate.sigma


def test_non_cross_fitted_variant_is_flagged():
    y, x, z = _confounded_sample(n=500, theta=1.0, seed=2)
    estimate = orthogonal_partially_linear(y, x, z, folds=1, seed=2)
    assert not estimate.cross_fitted
    assert estimate.folds == 1


def test_estimate_serialization_and_validation():
    y, x, z = _confounded_sample(n=200, seed=4)
    estimate = orthogonal_partially_linear(y, x, z, folds=3, seed=4)
    payload = estimate.to_dict()
    assert isinstance(estimate, OrthogonalEstimate)
    assert payload["n"] == 200
    assert payload["folds"] == 3
    assert payload["ci_low"] < payload["theta"] < payload["ci_high"]
    with pytest.raises(ValueError, match="same length"):
        orthogonal_partially_linear(y[:-1], x, z)
    with pytest.raises(ValueError, match="finite"):
        orthogonal_partially_linear(torch.full((4,), float("nan")), x[:4], z[:4])
    with pytest.raises(ValueError, match="shape"):
        orthogonal_partially_linear(y, x, z.reshape(-1, 1, 1))
    with pytest.raises(ValueError, match="folds"):
        orthogonal_partially_linear(y, x, z, folds=0)
    with pytest.raises(ValueError, match="confidence"):
        orthogonal_partially_linear(y, x, z, confidence=1.0)


def test_degenerate_treatment_is_rejected():
    z = torch.linspace(0.0, 1.0, 200, dtype=torch.float64)
    x = z.clone()
    y = 2.0 * x + torch.randn(200, dtype=torch.float64) * 0.01
    with pytest.raises(ValueError, match="not identified"):
        orthogonal_partially_linear(y, x, z, folds=4)


def test_naive_estimator_is_unbiased_without_confounding():
    generator = torch.Generator().manual_seed(5)
    n = 4000
    x = torch.randn(n, generator=generator, dtype=torch.float64)
    y = 0.3 * x + 0.5 * torch.randn(n, generator=generator, dtype=torch.float64)
    naive = naive_linear_estimate(y, x)
    assert abs(naive.theta - 0.3) < 3.0 * naive.sigma
    assert not naive.cross_fitted


def _one_hot(z: torch.Tensor, n_groups: int) -> torch.Tensor:
    index = z[:, 0].to(torch.long)
    return torch.nn.functional.one_hot(index, n_groups).to(torch.float64)


def test_categorical_per_method_nuisance_with_one_hot_features():
    generator = torch.Generator().manual_seed(11)
    n = 3000
    n_methods = 4
    method = torch.randint(0, n_methods, (n,), generator=generator).to(torch.float64)
    offsets = torch.tensor([0.0, 0.4, -0.3, 0.7], dtype=torch.float64)
    offset = offsets[method.to(torch.long)]
    x = offset + 0.5 * torch.randn(n, generator=generator, dtype=torch.float64)
    y = 1.2 * x + offset + 0.5 * torch.randn(n, generator=generator, dtype=torch.float64)

    naive = naive_linear_estimate(y, x)
    orthogonal = orthogonal_partially_linear(
        y,
        x,
        method,
        folds=5,
        nuisance_features=lambda z: _one_hot(z, n_methods),
        seed=0,
    )
    assert abs(naive.theta - 1.2) > 0.1
    assert abs(orthogonal.theta - 1.2) < 3.0 * orthogonal.sigma
    assert orthogonal.ci_low <= 1.2 <= orthogonal.ci_high


def test_nuisance_features_validation():
    y, x, z = _confounded_sample(n=200, seed=6)
    with pytest.raises(ValueError, match="matching y"):
        orthogonal_partially_linear(y, x, z, nuisance_features=lambda z: z[:10])
    with pytest.raises(ValueError, match="at least one column"):
        orthogonal_partially_linear(y, x, z, nuisance_features=lambda z: z[:, :0])
    with pytest.raises(ValueError, match="finite"):
        orthogonal_partially_linear(y, x, z, nuisance_features=lambda z: z * float("nan"))


def test_influence_function_variance_scales_as_one_over_sqrt_n():
    sigmas = []
    for n in (2000, 8000):
        y, x, z = _confounded_sample(n=n, theta=1.0, seed=12)
        estimate = orthogonal_partially_linear(y, x, z, folds=5, seed=12)
        sigmas.append(estimate.sigma)
    ratio = sigmas[0] / sigmas[1]
    assert 1.5 < ratio < 2.7, f"sigma should shrink like 1/sqrt(n), got ratio {ratio}"


def _ccddhnr(rng, n: int = 500, p: int = 20, theta: float = 0.5):
    """DoubleML's make_plr_CCDDHNR2018 design (Chernozhukov et al. 2018)."""
    import numpy as np

    idx = np.arange(p)
    x = rng.multivariate_normal(np.zeros(p), 0.7 ** np.abs(np.subtract.outer(idx, idx)), size=n)
    sig = 1.0 / (1.0 + np.exp(-x[:, 2]))
    d = x[:, 0] + 0.25 * sig + rng.normal(size=n)
    y = theta * d + 1.0 / (1.0 + np.exp(-x[:, 0])) + 0.25 * x[:, 2] + rng.normal(size=n)
    return torch.as_tensor(y), torch.as_tensor(d), torch.as_tensor(x)


def test_cross_fitting_shares_one_split_across_nuisances():
    """Regression: independent splits for E[x|z] and E[y|z] biased theta.

    Before the fix each nuisance drew its own permutation and the Monte Carlo
    mean over 200 replications of this design was 0.4736 (bias -0.026, about
    7.7 standard errors); with one shared split it is unbiased.
    """
    import numpy as np

    estimates = []
    for rep in range(200):
        y, d, x = _ccddhnr(np.random.default_rng(rep))
        estimates.append(orthogonal_partially_linear(y, d, x, folds=5, seed=rep).theta)
    est = np.asarray(estimates)
    standard_error = est.std(ddof=1) / np.sqrt(len(est))
    assert abs(est.mean() - 0.5) < 3.0 * standard_error


# ---------------------------------------------------------------------------
# Random Fourier feature nuisances, median-heuristic bandwidth, GCV / LOO ridge
# ---------------------------------------------------------------------------


def _pairwise_median(points: torch.Tensor) -> float:
    distances = torch.cdist(points, points)
    upper = torch.triu_indices(points.shape[0], points.shape[0], offset=1)
    return float(distances[upper[0], upper[1]].median())


def test_median_heuristic_bandwidth_is_the_median_pairwise_distance():
    generator = torch.Generator().manual_seed(0)
    points = torch.randn(60, 3, generator=generator, dtype=torch.float64)
    assert median_heuristic_bandwidth(points) == pytest.approx(_pairwise_median(points))
    line = torch.arange(5, dtype=torch.float64)  # distances 1,1,1,1,2,2,2,3,3,4
    assert median_heuristic_bandwidth(line) == pytest.approx(2.0)
    assert median_heuristic_bandwidth([[1.0, 1.0]] * 6) == 1.0  # coincident points
    assert median_heuristic_bandwidth([[1.0, 2.0]]) == 1.0  # a single row


def test_median_heuristic_bandwidth_subsample_is_seeded_and_close():
    generator = torch.Generator().manual_seed(1)
    points = torch.randn(3000, 4, generator=generator, dtype=torch.float64)
    first = median_heuristic_bandwidth(points, n_subsample=300, seed=7)
    assert first == median_heuristic_bandwidth(points, n_subsample=300, seed=7)
    assert first != median_heuristic_bandwidth(points, n_subsample=300, seed=8)
    # Full-data value for 4 standard-normal columns is about sqrt(2 * 4).
    assert first == pytest.approx(_pairwise_median(points[:1500]), rel=0.05)
    with pytest.raises(ValueError, match="n_subsample"):
        median_heuristic_bandwidth(points, n_subsample=1)


def test_random_fourier_features_shape_options_and_seed():
    z = torch.randn(50, 3, dtype=torch.float64)
    features = random_fourier_features(z, n_features=32, seed=3)
    assert features.shape == (50, 41)  # 32 random + 3 covariates x powers 1..3
    assert features.dtype == torch.float64
    assert torch.equal(features, random_fourier_features(z, n_features=32, seed=3))
    assert not torch.equal(features, random_fourier_features(z, n_features=32, seed=4))
    bare = random_fourier_features(z, n_features=32, seed=3, polynomial_degree=0)
    assert bare.shape == (50, 32)
    assert torch.equal(bare, features[:, :32])
    standardised = features[:, 32:35]  # power-1 block
    assert torch.allclose(standardised.mean(0), torch.zeros(3, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(standardised.std(0, unbiased=False), torch.ones(3, dtype=torch.float64))
    assert random_fourier_features(z[:, 0], n_features=8).shape == (50, 11)  # 1-D input
    assert random_fourier_features(z, n_features=32, polynomial_degree=1).shape == (50, 35)


def test_random_fourier_features_approximate_the_gaussian_kernel():
    generator = torch.Generator().manual_seed(2)
    z = torch.randn(40, 2, generator=generator, dtype=torch.float64)
    sigma = 1.7
    features = random_fourier_features(
        z, n_features=20000, bandwidth=sigma, standardize=False, polynomial_degree=0, seed=0
    )
    exact = torch.exp(-(torch.cdist(z, z) ** 2) / (2.0 * sigma**2))
    assert float((features @ features.T - exact).abs().max()) < 0.05


def test_random_fourier_features_default_bandwidth_is_the_median_heuristic():
    generator = torch.Generator().manual_seed(3)
    z = torch.randn(200, 5, generator=generator, dtype=torch.float64) * torch.tensor(
        [1.0, 10.0, 0.1, 1.0, 3.0], dtype=torch.float64
    )
    standardised = (z - z.mean(0)) / z.std(0, unbiased=False)
    sigma = median_heuristic_bandwidth(standardised, seed=5)
    by_default = random_fourier_features(z, n_features=16, seed=5)
    explicit = random_fourier_features(z, n_features=16, bandwidth=sigma, seed=5)
    assert torch.allclose(by_default, explicit)
    # Rescaling a column does not change standardised features (or the bandwidth).
    scaled = z.clone()
    scaled[:, 1] *= 1000.0
    assert torch.allclose(random_fourier_features(scaled, n_features=16, seed=5), by_default)


def test_random_fourier_features_validation():
    z = torch.randn(10, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="bandwidth"):
        random_fourier_features(z, bandwidth="mean")
    with pytest.raises(ValueError, match="bandwidth"):
        random_fourier_features(z, bandwidth=-1.0)
    with pytest.raises(ValueError, match="n_features"):
        random_fourier_features(z, n_features=0)
    with pytest.raises(ValueError, match="finite"):
        random_fourier_features(torch.full((4, 2), float("nan")))
    constant = random_fourier_features(torch.ones(8, 2), n_features=4)
    assert bool(torch.isfinite(constant).all())  # constant columns are only centred


def test_gcv_and_loo_penalty_follow_the_brute_force_leave_one_out_curve():
    from torchregress.inference import orthogonal as orth

    generator = torch.Generator().manual_seed(4)
    n, p = 40, 12
    features = torch.randn(n, p, generator=generator, dtype=torch.float64)
    target = (
        features[:, 0]
        - 0.5 * features[:, 1]
        + 0.7 * torch.randn(n, generator=generator, dtype=torch.float64)
    )
    centred = features - features.mean(0)
    target_c = target - target.mean()
    top = float(torch.linalg.svdvals(centred).max() ** 2)

    def brute_force_loo(penalty: float) -> float:
        errors = []
        for i in range(n):
            keep = torch.arange(n) != i
            x_tr, y_tr = features[keep], target[keep]
            mean_x, mean_y = x_tr.mean(0), y_tr.mean()
            xc = x_tr - mean_x
            gram = xc.T @ xc + penalty * torch.eye(p, dtype=torch.float64)
            weights = torch.linalg.solve(gram, xc.T @ (y_tr - mean_y))
            errors.append(float(target[i] - ((features[i] - mean_x) @ weights + mean_y)) ** 2)
        return sum(errors) / n

    grid = [fraction * top for fraction in orth._RIDGE_GRID]
    best = grid[min(range(len(grid)), key=lambda k: brute_force_loo(grid[k]))]
    assert orth._select_ridge(centred, target_c, criterion="loo") == pytest.approx(best)
    gcv = orth._select_ridge(centred, target_c, criterion="gcv")
    assert any(gcv == pytest.approx(g) for g in grid)
    # The data-driven penalty is a real penalty for noisy data, not the 1e-6 jitter.
    assert gcv > 1e-3 * top


@pytest.mark.parametrize("ridge", ["gcv", "loo"])
def test_data_driven_ridge_regularises_an_overparameterised_basis(ridge):
    """400 random features on 100 noisy rows: the near-unpenalised fit interpolates noise."""
    generator = torch.Generator().manual_seed(3)
    n = 100
    z = torch.rand(n, generator=generator, dtype=torch.float64)
    eta = torch.sin(2.0 * math.pi * z) + z**2
    x = 2.0 * eta + 1.5 * torch.randn(n, generator=generator, dtype=torch.float64)
    y = x + eta + 1.5 * torch.randn(n, generator=generator, dtype=torch.float64)

    def features(t):
        return random_fourier_features(
            t, n_features=400, bandwidth=0.15, standardize=False, polynomial_degree=0, seed=0
        )

    fixed = orthogonal_partially_linear(y, x, z, nuisance_features=features, ridge=1e-8, seed=1)
    tuned = orthogonal_partially_linear(y, x, z, nuisance_features=features, ridge=ridge, seed=1)
    assert fixed.nuisance_r2_x < 0.0  # worse than predicting the mean
    assert tuned.nuisance_r2_x > fixed.nuisance_r2_x + 0.1
    assert tuned.nuisance_r2_y > fixed.nuisance_r2_y + 0.02
    assert abs(tuned.theta - 1.0) < 4.0 * tuned.sigma


def test_ridge_argument_validation_and_default_behaviour_unchanged():
    y, x, z = _confounded_sample(n=300, seed=5)
    with pytest.raises(ValueError, match="ridge"):
        orthogonal_partially_linear(y, x, z, ridge="cv")
    with pytest.raises(ValueError, match="ridge"):
        orthogonal_partially_linear(y, x, z, ridge=-1.0)
    default = orthogonal_partially_linear(y, x, z, seed=2)
    explicit = orthogonal_partially_linear(y, x, z, ridge=1.0e-6, seed=2)
    assert default.to_dict() == explicit.to_dict()


def _plr_cubic_sample(seed: int, n: int = 500):
    """The harness ``plr_cubic`` design: 5 covariates, cubic confounding, theta = 1."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 5, generator=rng, dtype=torch.float64)
    noise = torch.randn(2, n, generator=rng, dtype=torch.float64)
    d = x[:, 0] + 0.5 * x[:, 1] ** 2 + 0.3 * x[:, 2] ** 3 + noise[0]
    y = d + 0.8 * x[:, 0] ** 3 + x[:, 1] ** 2 - 0.5 * x[:, 2] + 0.5 * x[:, 3] + noise[1]
    return y, d, x


def _plr_ccddhnr_sample(seed: int, n: int = 500, p: int = 20):
    """The CCDDHNR-2018 design (DoubleML ``make_plr_CCDDHNR2018``), theta = 0.5."""
    rng = torch.Generator().manual_seed(seed)
    index = torch.arange(p, dtype=torch.float64)
    cov = 0.7 ** (index[:, None] - index[None, :]).abs()
    x = torch.randn(n, p, generator=rng, dtype=torch.float64) @ torch.linalg.cholesky(cov).T
    noise = torch.randn(2, n, generator=rng, dtype=torch.float64)
    d = x[:, 0] + 0.25 * torch.sigmoid(x[:, 2]) + noise[0]
    y = 0.5 * d + torch.sigmoid(x[:, 0]) + 0.25 * x[:, 2] + noise[1]
    return y, d, x


@pytest.mark.parametrize(
    ("sampler", "theta"), [(_plr_ccddhnr_sample, 0.5), (_plr_cubic_sample, 1.0)]
)
def test_rff_nuisance_with_gcv_ridge_covers_on_the_harness_designs(sampler, theta):
    """Median-heuristic RFF + GCV ridge: honest intervals (200-replication harness
    numbers are in the CHANGELOG; this is a fast 40-replication guard)."""
    reps = 40
    covered, estimates = 0, []
    for seed in range(reps):
        y, d, x = sampler(seed)
        est = orthogonal_partially_linear(
            y,
            d,
            x,
            ridge="gcv",
            nuisance_features=lambda t, s=seed: random_fourier_features(t, seed=s),
            seed=seed,
        )
        covered += est.ci_low <= theta <= est.ci_high
        estimates.append(est.theta)
    spread = torch.tensor(estimates).std() / reps**0.5
    assert covered >= 0.85 * reps
    assert abs(float(torch.tensor(estimates).mean()) - theta) < 4.0 * float(spread)
