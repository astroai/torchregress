"""Unit tests for torchregress.metrics.sbi (SBC and TARP diagnostics)."""

from __future__ import annotations

import math

import pytest
import torch

from torchregress.metrics.sbi import (
    sbc_rank_histogram,
    sbc_uniformity_pvalue,
    simulation_based_calibration_ranks,
    tarp_coverage,
)

N_OBS = 4000
N_SAMPLES = 100
N_DIMS = 2


def _simulate_calibrated_gaussian(
    *,
    seed: int = 0,
    n_obs: int = N_OBS,
    n_samples: int = N_SAMPLES,
    n_dims: int = N_DIMS,
    scale: float = 1.0,
    shift: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate a data set with an exact Gaussian posterior (calibrated by construction).

    Prior ``theta* ~ N(0, 1)``, likelihood ``y = theta* + N(0, 1)``, so the exact
    posterior is ``N(y / 2, 1 / 2)``. ``scale`` multiplies the posterior standard
    deviation (``scale < 1`` is an over-confident posterior) and ``shift`` adds a
    constant bias to every posterior draw. Both miscalibrate the posterior.
    """
    generator = torch.Generator().manual_seed(seed)
    prior_var = 1.0
    noise_var = 1.0
    posterior_var = prior_var * noise_var / (prior_var + noise_var)
    truth = torch.randn(n_obs, n_dims, generator=generator) * math.sqrt(prior_var)
    y = truth + torch.randn(n_obs, n_dims, generator=generator) * math.sqrt(noise_var)
    posterior_mean = posterior_var / noise_var * y
    samples = posterior_mean.unsqueeze(1) + scale * math.sqrt(posterior_var) * torch.randn(
        n_obs, n_samples, n_dims, generator=generator
    )
    return samples + shift, truth


def _prior_reference_points(
    n_obs: int = N_OBS, n_dims: int = N_DIMS, *, seed: int = 12345
) -> torch.Tensor:
    """Reference points drawn from the prior with a RNG independent of the data.

    The reference point must be independent of both the true parameters and the
    posterior draws (TARP's coverage property); reusing the simulation seed would
    make the reference coincide with the truth and trivially inflate coverage.
    """
    return torch.randn(n_obs, n_dims, generator=torch.Generator().manual_seed(seed))


def _naive_tie_aware_ranks(samples: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Loop implementation of the tie-aware SBC rank, used as a reference."""
    n_obs, n_samples, n_dims = samples.shape
    ranks = torch.zeros(n_obs, n_dims, dtype=torch.float64)
    for n in range(n_obs):
        for d in range(n_dims):
            below = int((samples[n, :, d] < truth[n, d]).sum().item())
            equal = int((samples[n, :, d] == truth[n, d]).sum().item())
            ranks[n, d] = below + 0.5 * equal
    return ranks


# ═══════════════════════════════════════════════════════════════════════════════
# simulation_based_calibration_ranks
# ═══════════════════════════════════════════════════════════════════════════════


class TestSBCRanks:
    def test_calibrated_posterior_gives_uniform_ranks(self) -> None:
        """Exact Gaussian posterior: ranks are discrete-uniform on {0, ..., S}."""
        samples, truth = _simulate_calibrated_gaussian()
        ranks = simulation_based_calibration_ranks(samples, truth)

        assert ranks.shape == (N_OBS, N_DIMS)
        assert ranks.dtype == torch.float32
        assert float(ranks.min()) >= 0.0
        assert float(ranks.max()) <= N_SAMPLES
        # E[uniform rank] = S / 2 with sd ~ sqrt((S+1)^2 - 1) / sqrt(12 N) ~ 0.46.
        torch.testing.assert_close(
            ranks.mean(dim=0), torch.full((N_DIMS,), N_SAMPLES / 2.0), atol=3.0, rtol=0.0
        )

        histogram = sbc_rank_histogram(ranks)
        counts = histogram["counts"]
        expected = N_OBS / counts.shape[1]
        # Binomial fluctuation per bin is sqrt(expected); 4 sigma is a generous
        # deterministic tolerance for a fixed seed.
        assert float((counts - expected).abs().max()) < 4.0 * math.sqrt(expected)

    def test_overconfident_posterior_breaks_uniformity(self) -> None:
        """Posterior too narrow: ranks pile up in the extreme bins."""
        samples, truth = _simulate_calibrated_gaussian(scale=0.5)
        ranks = simulation_based_calibration_ranks(samples, truth)
        uniformity = sbc_uniformity_pvalue(ranks)

        assert uniformity["n_ranks"] == N_OBS
        assert float(uniformity["min_p_value"]) < 1e-6
        counts = sbc_rank_histogram(ranks)["counts"]
        edge_fraction = float((counts[0, 0] + counts[-1, -1]) / counts.sum())
        assert edge_fraction > 0.05  # calibrated value is ~1/(S+1) = 1% per edge

    def test_biased_posterior_skews_ranks(self) -> None:
        """Posterior shifted by +1: the true value sits below most draws."""
        samples, truth = _simulate_calibrated_gaussian(shift=1.0)
        ranks = simulation_based_calibration_ranks(samples, truth)

        assert float(ranks.mean()) < 0.4 * N_SAMPLES
        assert float(sbc_uniformity_pvalue(ranks)["min_p_value"]) < 1e-6

    def test_tie_aware_rank_hand_computed(self) -> None:
        """Ties get half credit: 2 below + 2 equal -> rank 2 + 0.5 * 2 = 3."""
        samples = torch.tensor([[[1.0], [3.0], [1.0], [2.0], [2.0]]])  # [1, S=5, D=1]
        assert simulation_based_calibration_ranks(samples, torch.tensor([[2.0]])).item() == 3.0

        truth_below = torch.tensor([[0.0]])
        truth_above = torch.tensor([[4.0]])
        assert simulation_based_calibration_ranks(samples, truth_below).item() == 0.0
        assert simulation_based_calibration_ranks(samples, truth_above).item() == 5.0

    def test_matches_naive_loop_with_ties(self) -> None:
        """Vectorised rank equals the loop implementation, including tied draws."""
        generator = torch.Generator().manual_seed(3)
        # Quantised draws create exact ties.
        samples = torch.round(torch.randn(4, 7, 3, generator=generator) * 2.0) / 2.0
        truth = torch.round(torch.randn(4, 3, generator=generator) * 2.0) / 2.0
        ranks = simulation_based_calibration_ranks(samples, truth)
        torch.testing.assert_close(ranks.double(), _naive_tie_aware_ranks(samples, truth))

    def test_single_observation_2d_input(self) -> None:
        """[S, D] posterior and [D] truth is a batch of one observation."""
        samples, truth = _simulate_calibrated_gaussian(n_obs=1, n_samples=50)
        single = simulation_based_calibration_ranks(samples[0], truth[0])
        batched = simulation_based_calibration_ranks(samples, truth)
        assert single.shape == (1, N_DIMS)
        torch.testing.assert_close(single, batched)

    def test_dim_one(self) -> None:
        """D = 1 keeps the trailing dimension: [N, S, 1] -> [N, 1]."""
        samples, truth = _simulate_calibrated_gaussian(n_obs=100, n_samples=20, n_dims=1)
        ranks = simulation_based_calibration_ranks(samples, truth)
        assert ranks.shape == (100, 1)
        assert ranks.dtype == torch.float32

    def test_dim_axis_parameter_dimension(self) -> None:
        """dim selects the parameter axis, so [N, D, S] with dim=1 matches canonical."""
        samples, truth = _simulate_calibrated_gaussian(n_obs=200, n_samples=30)
        canonical = simulation_based_calibration_ranks(samples, truth)
        transposed = simulation_based_calibration_ranks(samples.permute(0, 2, 1), truth, dim=1)
        torch.testing.assert_close(canonical, transposed)

    def test_float64_preserves_dtype(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(n_obs=20, n_samples=10)
        ranks = simulation_based_calibration_ranks(samples.double(), truth.double())
        assert ranks.dtype == torch.float64

    def test_numpy_input(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(n_obs=20, n_samples=10)
        ranks = simulation_based_calibration_ranks(samples.numpy(), truth.numpy())
        assert ranks.shape == (20, N_DIMS)
        assert isinstance(ranks, torch.Tensor)

    def test_integer_parameters(self) -> None:
        """Discrete parameters are supported (ties are common and half-credited)."""
        generator = torch.Generator().manual_seed(1)
        truth = torch.randint(0, 5, (30, 2), generator=generator)
        samples = truth.unsqueeze(1) + torch.randint(-1, 2, (30, 15, 2), generator=generator)
        ranks = simulation_based_calibration_ranks(samples, truth)
        assert ranks.shape == (30, 2)
        assert torch.all((ranks >= 0) & (ranks <= 15))

    def test_errors(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(n_obs=20, n_samples=10)

        with pytest.raises(ValueError, match="2-D"):
            simulation_based_calibration_ranks(torch.randn(10), truth[:1])
        with pytest.raises(ValueError, match="mismatch"):
            simulation_based_calibration_ranks(samples, truth[:10])
        with pytest.raises(ValueError, match="dimension mismatch"):
            simulation_based_calibration_ranks(samples, truth[:, :1])
        with pytest.raises(ValueError, match="single observation"):
            simulation_based_calibration_ranks(samples, truth[0])
        with pytest.raises(ValueError, match="batch axis"):
            simulation_based_calibration_ranks(samples, truth, dim=0)
        with pytest.raises(ValueError, match="parameter-dimension axis"):
            simulation_based_calibration_ranks(samples[0], truth[0], dim=0)
        with pytest.raises(ValueError, match="NaN or infinite"):
            bad = samples.clone()
            bad[0, 0, 0] = float("nan")
            simulation_based_calibration_ranks(bad, truth)
        with pytest.raises(ValueError, match="NaN or infinite"):
            simulation_based_calibration_ranks(samples, truth + torch.tensor(float("inf")))
        with pytest.raises(ValueError, match="shape"):
            simulation_based_calibration_ranks(torch.randn(2, 3, 4, 5), truth)


# ═══════════════════════════════════════════════════════════════════════════════
# sbc_rank_histogram
# ═══════════════════════════════════════════════════════════════════════════════


class TestSBCRankHistogram:
    def test_default_one_bin_per_rank(self) -> None:
        samples, truth = _simulate_calibrated_gaussian()
        ranks = simulation_based_calibration_ranks(samples, truth)
        histogram = sbc_rank_histogram(ranks)

        counts = histogram["counts"]
        edges = histogram["bin_edges"]
        assert counts.shape == (N_DIMS, N_SAMPLES + 1)
        assert edges.shape == (N_SAMPLES + 2,)
        torch.testing.assert_close(edges, torch.arange(0, N_SAMPLES + 2, dtype=edges.dtype))
        assert int(counts.sum().item()) == N_OBS * N_DIMS
        assert int(counts[0].sum().item()) == N_OBS

    def test_single_dimension_is_squeezed(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(n_obs=500, n_samples=20)
        ranks = simulation_based_calibration_ranks(samples, truth)
        assert sbc_rank_histogram(ranks[:, 0])["counts"].shape == (21,)
        assert sbc_rank_histogram(ranks[:, :1])["counts"].shape == (21,)
        assert sbc_rank_histogram(ranks)["counts"].shape == (2, 21)

    def test_coarse_histogram(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(n_obs=1000, n_samples=50)
        ranks = simulation_based_calibration_ranks(samples, truth)
        histogram = sbc_rank_histogram(ranks, n_bins=10)

        counts = histogram["counts"]
        edges = histogram["bin_edges"]
        assert counts.shape == (N_DIMS, 10)
        assert edges.shape == (11,)
        assert int(counts.sum().item()) == 1000 * N_DIMS
        assert float(edges[0]) == 0.0
        assert float(edges[-1]) == 51.0  # S = 50 samples in this simulation
        assert bool((torch.diff(edges) > 0).all())
        assert bool((torch.diff(edges) >= 1).all())  # integer edges

    def test_explicit_n_posterior_samples(self) -> None:
        """Ranks that do not reach S can still be binned over the full [0, S + 1]."""
        ranks = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        histogram = sbc_rank_histogram(ranks, n_posterior_samples=10)
        assert histogram["counts"].shape == (2, 11)
        assert histogram["bin_edges"].shape == (12,)
        assert float(histogram["bin_edges"][-1]) == 11.0

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            sbc_rank_histogram(torch.zeros(2, 3, 4))
        with pytest.raises(ValueError, match="non-negative"):
            sbc_rank_histogram(torch.tensor([[-1.0, 0.0]]))
        with pytest.raises(ValueError, match="NaN or infinite"):
            sbc_rank_histogram(torch.tensor([[0.0, float("nan")]]))
        with pytest.raises(ValueError, match="at least one value"):
            sbc_rank_histogram(torch.empty(0))
        with pytest.raises(ValueError, match="at least"):
            sbc_rank_histogram(torch.tensor([[0.0, 1.0, 2.0]]), n_bins=0)
        with pytest.raises(ValueError, match="exceeds"):
            sbc_rank_histogram(torch.tensor([[0.0, 3.0]]), n_posterior_samples=2)
        with pytest.raises(ValueError, match="cannot infer"):
            sbc_rank_histogram(torch.zeros(10, 1))
        with pytest.raises(ValueError, match="possible rank values"):
            sbc_rank_histogram(torch.tensor([[0.0, 1.0]]), n_posterior_samples=2, n_bins=10)


# ═══════════════════════════════════════════════════════════════════════════════
# sbc_uniformity_pvalue
# ═══════════════════════════════════════════════════════════════════════════════


class TestSBCUniformityPvalue:
    def test_calibrated_posterior_does_not_reject(self) -> None:
        samples, truth = _simulate_calibrated_gaussian()
        ranks = simulation_based_calibration_ranks(samples, truth)
        result = sbc_uniformity_pvalue(ranks)

        assert result["n_ranks"] == N_OBS
        assert result["n_bins"] == N_SAMPLES + 1
        assert result["n_posterior_samples"] == N_SAMPLES
        assert result["dof"] == N_SAMPLES
        p_values = result["p_values"]
        assert isinstance(p_values, list) and len(p_values) == N_DIMS
        assert all(0.0 <= p <= 1.0 for p in p_values)
        assert float(result["min_p_value"]) > 0.01
        assert float(result["bonferroni_min_p_value"]) == pytest.approx(
            min(1.0, 2.0 * float(result["min_p_value"]))
        )
        assert len(result["chi2"]) == N_DIMS

    def test_miscalibrated_posterior_is_rejected(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(scale=0.5)
        ranks = simulation_based_calibration_ranks(samples, truth)
        result = sbc_uniformity_pvalue(ranks)
        assert float(result["min_p_value"]) < 1e-6

    def test_inferred_n_posterior_samples(self) -> None:
        """When a rank equals S the number of posterior draws is recovered."""
        samples, truth = _simulate_calibrated_gaussian()
        ranks = simulation_based_calibration_ranks(samples, truth)
        inferred = sbc_uniformity_pvalue(ranks)
        explicit = sbc_uniformity_pvalue(ranks, n_posterior_samples=N_SAMPLES)
        assert inferred["n_bins"] == explicit["n_bins"]
        torch.testing.assert_close(torch.tensor(inferred["chi2"]), torch.tensor(explicit["chi2"]))

    def test_coarse_bins_reduce_dof(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(n_obs=1000, n_samples=50)
        ranks = simulation_based_calibration_ranks(samples, truth)
        result = sbc_uniformity_pvalue(ranks, n_bins=10)
        assert result["dof"] == 9
        assert result["n_bins"] == 10
        assert float(result["min_p_value"]) > 0.01

    def test_single_dimension_ranks(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(n_obs=1000, n_samples=50)
        ranks = simulation_based_calibration_ranks(samples, truth)
        one_dim = sbc_uniformity_pvalue(ranks[:, 0])
        two_dim = sbc_uniformity_pvalue(ranks)
        assert len(one_dim["p_values"]) == 1
        assert one_dim["p_values"][0] == pytest.approx(two_dim["p_values"][0])

    def test_requires_enough_samples_per_bin(self) -> None:
        with pytest.raises(ValueError, match="expected ranks per bin"):
            sbc_uniformity_pvalue(torch.zeros(50, 1), n_posterior_samples=100)

    def test_errors(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            sbc_uniformity_pvalue(torch.zeros(2, 3, 4))
        with pytest.raises(ValueError, match="cannot infer"):
            sbc_uniformity_pvalue(torch.zeros(10, 1))
        with pytest.raises(ValueError, match="at least"):
            sbc_uniformity_pvalue(torch.zeros(100, 1), n_posterior_samples=100, n_bins=1)
        with pytest.raises(ValueError, match="positive integer"):
            sbc_uniformity_pvalue(torch.zeros(100, 1), n_posterior_samples=0)


# ═══════════════════════════════════════════════════════════════════════════════
# tarp_coverage
# ═══════════════════════════════════════════════════════════════════════════════


class TestTARPCoverage:
    def test_calibrated_posterior_coverage(self) -> None:
        samples, truth = _simulate_calibrated_gaussian()
        result = tarp_coverage(samples, truth, _prior_reference_points())

        alpha = result["alpha"]
        coverage = result["coverage"]
        error = float(result["tarp_error"])
        assert alpha.shape == (101,)
        assert coverage.shape == (101,)
        torch.testing.assert_close(alpha, torch.linspace(0.0, 1.0, 101))
        assert error < 0.08
        assert error == pytest.approx(float((coverage - alpha).abs().max()), abs=1e-6)
        # Coverage is ~0 at alpha=0 and ~1 at alpha=1.
        assert float(coverage[0]) < 0.05
        assert float(coverage[-1]) > 0.95
        assert result["n_samples"] == N_OBS
        assert result["n_posterior_samples"] == N_SAMPLES

    def test_shared_reference_point(self) -> None:
        samples, truth = _simulate_calibrated_gaussian()
        shared = tarp_coverage(samples, truth, torch.tensor([0.3, -0.2]))
        broadcast = tarp_coverage(samples, truth, torch.tensor([[0.3, -0.2]]))
        assert float(shared["tarp_error"]) < 0.08
        torch.testing.assert_close(shared["coverage"], broadcast["coverage"])

    def test_overconfident_posterior_has_large_error(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(scale=0.5)
        result = tarp_coverage(samples, truth, _prior_reference_points())
        assert float(result["tarp_error"]) > 0.15

    def test_biased_posterior_has_large_error(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(shift=1.0)
        result = tarp_coverage(samples, truth, _prior_reference_points())
        assert float(result["tarp_error"]) > 0.15

    def test_custom_alpha_grid(self) -> None:
        samples, truth = _simulate_calibrated_gaussian()
        alphas = torch.tensor([0.1, 0.5, 0.9])
        result = tarp_coverage(samples, truth, _prior_reference_points(), alphas=alphas)
        torch.testing.assert_close(result["alpha"], alphas)
        assert result["coverage"].shape == (3,)
        torch.testing.assert_close(result["coverage"], alphas, atol=0.08, rtol=0.0)

    def test_matches_quantile_reference_implementation(self) -> None:
        """Vectorised quantiles agree with torch.quantile for every alpha."""
        generator = torch.Generator().manual_seed(7)
        samples = torch.randn(3, 11, 2, generator=generator)
        truth = torch.randn(3, 2, generator=generator)
        refs = torch.randn(3, 2, generator=generator)
        alphas = torch.tensor([0.0, 0.3, 0.5, 0.7, 1.0])
        result = tarp_coverage(samples, truth, refs, alphas=alphas)

        distances = torch.sqrt(((samples - refs.unsqueeze(1)) ** 2).sum(dim=-1))  # [N, S]
        true_distances = torch.sqrt(((truth - refs) ** 2).sum(dim=-1))  # [N]
        expected = torch.stack(
            [
                (true_distances <= torch.quantile(distances, float(a), dim=1))
                .to(torch.float32)
                .mean()
                for a in alphas
            ]
        )
        torch.testing.assert_close(result["coverage"], expected, atol=1e-5, rtol=0.0)

    def test_single_observation(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(n_obs=1, n_samples=50)
        result = tarp_coverage(samples[0], truth[0], _prior_reference_points(1)[0])
        assert result["alpha"].shape == (101,)
        assert result["coverage"].shape == (101,)
        assert result["n_samples"] == 1
        assert result["n_posterior_samples"] == 50
        assert math.isfinite(float(result["tarp_error"]))

    def test_deterministic(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(n_obs=100, n_samples=20)
        refs = _prior_reference_points(100, N_DIMS)
        first = tarp_coverage(samples, truth, refs)
        second = tarp_coverage(samples, truth, refs)
        torch.testing.assert_close(first["coverage"], second["coverage"])

    def test_numpy_input(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(n_obs=100, n_samples=20)
        result = tarp_coverage(samples.numpy(), truth.numpy(), _prior_reference_points(100).numpy())
        assert result["coverage"].shape == (101,)

    def test_errors(self) -> None:
        samples, truth = _simulate_calibrated_gaussian(n_obs=50, n_samples=10)
        refs = _prior_reference_points(50, N_DIMS)

        with pytest.raises(ValueError, match="2-D"):
            tarp_coverage(torch.randn(50), truth, refs)
        with pytest.raises(ValueError, match="batch mismatch"):
            tarp_coverage(samples, truth[:10], refs)
        with pytest.raises(ValueError, match="shape"):
            tarp_coverage(samples, truth, refs[:10])
        with pytest.raises(ValueError, match="dimensions"):
            tarp_coverage(samples, truth, torch.zeros(3))
        with pytest.raises(ValueError, match="1-D"):
            tarp_coverage(samples, truth, torch.zeros(2, 2, 2))
        with pytest.raises(ValueError, match="at least 2 posterior samples"):
            tarp_coverage(samples[:, :1], truth, refs)
        with pytest.raises(ValueError, match="1-D"):
            tarp_coverage(samples, truth, refs, alphas=torch.tensor([[0.5]]))
        with pytest.raises(ValueError, match="non-empty"):
            tarp_coverage(samples, truth, refs, alphas=torch.empty(0))
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            tarp_coverage(samples, truth, refs, alphas=torch.tensor([1.5]))
        with pytest.raises(ValueError, match="NaN or infinite"):
            bad = samples.clone()
            bad[0, 0, 0] = float("nan")
            tarp_coverage(bad, truth, refs)
        with pytest.raises(ValueError, match="NaN or infinite"):
            tarp_coverage(samples, truth, refs + torch.tensor(float("inf")))
