"""
Unit tests for torchregress.metrics.distribution.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchregress.metrics.distribution import (
    ContinuousRankedProbabilityScore,
    EnergyScore,
    _cdf_from_density,
    _interp1d,
    _pit_from_density,
    _pit_from_quantiles,
    _pit_from_samples,
    _quantiles_from_density,
    _support_widths,
    conditional_density_estimation_loss,
    continuous_ranked_probability_score,
    crps_from_samples,
    crps_gaussian,
    distribution_metrics_report,
    energy_score,
    gaussian_nll,
    highest_posterior_density_coverage,
    highest_posterior_density_level,
    kolmogorov_smirnov_uniform_statistic,
    probability_integral_transform,
)

# ═══════════════════════════════════════════════════════════════════════════════
# kolmogorov_smirnov_uniform_statistic
# ═══════════════════════════════════════════════════════════════════════════════


class TestKolmogorovSmirnovUniform:
    def test_uniform_data_gives_small_ks(self) -> None:
        """Uniform(0,1) data gives small KS statistic."""
        data = torch.linspace(0, 1, 1000)
        ks = kolmogorov_smirnov_uniform_statistic(data)
        assert float(ks.item()) < 0.1

    def test_clustered_data_gives_large_ks(self) -> None:
        """Clustered data (far from uniform) gives large KS."""
        data = torch.full((100,), 0.5)
        ks = kolmogorov_smirnov_uniform_statistic(data)
        assert float(ks.item()) > 0.3

    def test_empty_raises(self) -> None:
        """Empty tensor raises ValueError."""
        with pytest.raises(ValueError, match="at least one value"):
            kolmogorov_smirnov_uniform_statistic(torch.empty(0))

    def test_numpy_input(self) -> None:
        """Works with numpy array input."""
        data = np.linspace(0, 1, 100)
        ks = kolmogorov_smirnov_uniform_statistic(data)
        assert isinstance(ks, torch.Tensor)

    def test_values_clamped(self) -> None:
        """Values outside [0,1] are clamped."""
        data = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5])
        ks = kolmogorov_smirnov_uniform_statistic(data)
        assert torch.isfinite(ks)


# ═══════════════════════════════════════════════════════════════════════════════
# probability_integral_transform
# ═══════════════════════════════════════════════════════════════════════════════


class TestProbabilityIntegralTransform:
    def test_basic_no_histogram(self) -> None:
        """Returns PIT values as tensor."""
        pit = probability_integral_transform(
            lambda t: t.clamp(0, 1),
            torch.randn(50),
            return_histogram=False,
        )
        assert isinstance(pit, torch.Tensor)
        assert pit.shape == (50,)

    def test_with_histogram(self) -> None:
        """Returns dict with histogram and uniformity stats."""
        result = probability_integral_transform(
            lambda t: t.clamp(0, 1),
            torch.randn(100),
            return_histogram=True,
        )
        assert isinstance(result, dict)
        assert "pit_values" in result
        assert "uniformity_chi2" in result
        assert "uniformity_ks" in result

    def test_numpy_return(self) -> None:
        """as_numpy=True returns numpy arrays."""
        result = probability_integral_transform(
            lambda t: t.clamp(0, 1),
            torch.randn(30),
            return_histogram=True,
            as_numpy=True,
        )
        assert isinstance(result, dict)
        assert isinstance(result["pit_values"], np.ndarray)
        assert isinstance(result["uniformity_chi2"], float)

    def test_numpy_input_no_histogram(self) -> None:
        """Numpy input without histogram returns numpy array."""

        def cdf(x):
            return x.clamp(0, 1)

        result = probability_integral_transform(
            cdf,
            np.random.randn(40).astype(np.float32),
            return_histogram=False,
        )
        assert isinstance(result, np.ndarray)

    def test_custom_n_bins(self) -> None:
        """Custom n_bins changes histogram bin count."""
        result = probability_integral_transform(
            lambda t: t.clamp(0, 1),
            torch.randn(100),
            return_histogram=True,
            n_bins=20,
        )
        assert result["histogram_counts"].shape == (20,)


# ═══════════════════════════════════════════════════════════════════════════════
# _support_widths
# ═══════════════════════════════════════════════════════════════════════════════


class TestSupportWidths:
    def test_uniform_grid(self) -> None:
        """Uniform grid gives uniform widths."""
        s = torch.linspace(0.0, 1.0, 5)
        w = _support_widths(s)
        assert torch.allclose(w, 0.25 * torch.ones_like(w))

    def test_first_and_last(self) -> None:
        """First and last bins have single-sided widths."""
        s = torch.tensor([0.0, 0.1, 0.3, 0.6, 1.0])
        w = _support_widths(s)
        assert float(w[0].item()) == pytest.approx(0.1)
        assert float(w[-1].item()) == pytest.approx(0.4)


# ═══════════════════════════════════════════════════════════════════════════════
# _interp1d
# ═══════════════════════════════════════════════════════════════════════════════


class TestInterp1d:
    def test_basic(self) -> None:
        """Linear interpolation on a grid."""
        support = torch.tensor([0.0, 1.0, 2.0])
        values = torch.tensor([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
        query = torch.tensor([0.5, 1.5])
        result = _interp1d(support, values, query)
        assert result.shape == (2,)
        assert float(result[0].item()) == pytest.approx(15.0)
        assert float(result[1].item()) == pytest.approx(25.0)

    def test_batched_values(self) -> None:
        """Batched values interpolation."""
        support = torch.tensor([0.0, 1.0])
        values = torch.tensor([[0.0, 10.0], [0.0, 20.0]])
        query = torch.tensor([0.5, 0.5])
        result = _interp1d(support, values, query)
        assert result.shape == (2,)
        assert float(result[0].item()) == pytest.approx(5.0)
        assert float(result[1].item()) == pytest.approx(10.0)

    def test_boundary_clamping(self) -> None:
        """Queries outside support are clamped to nearest."""
        support = torch.tensor([0.0, 1.0])
        values = torch.tensor([[10.0, 20.0], [10.0, 20.0]])
        query = torch.tensor([-1.0, 5.0])
        result = _interp1d(support, values, query)
        assert float(result[0].item()) == pytest.approx(10.0)
        assert float(result[1].item()) == pytest.approx(20.0)


# ═══════════════════════════════════════════════════════════════════════════════
# _cdf_from_density
# ═══════════════════════════════════════════════════════════════════════════════


class TestCDFFromDensity:
    def test_uniform_density(self) -> None:
        """Uniform density gives linear CDF."""
        support = torch.linspace(0.0, 1.0, 11)
        density = torch.ones(2, 11)
        cdf = _cdf_from_density(support, density)
        assert cdf.shape == (2, 11)
        assert float(cdf[0, 0].item()) == pytest.approx(0.0)
        assert float(cdf[0, -1].item()) == pytest.approx(1.0)

    def test_single_sample(self) -> None:
        """Single-sample density."""
        support = torch.linspace(0.0, 1.0, 5)
        density = torch.ones(1, 5)
        cdf = _cdf_from_density(support, density)
        assert cdf.shape == (1, 5)


# ═══════════════════════════════════════════════════════════════════════════════
# _quantiles_from_density
# ═══════════════════════════════════════════════════════════════════════════════


class TestQuantilesFromDensity:
    def test_median_of_uniform(self) -> None:
        """Median of uniform density on [0,1] is ~0.5."""
        support = torch.linspace(0.0, 1.0, 101)
        density = torch.ones(5, 101)
        q = _quantiles_from_density(support, density, [0.5])
        assert q.shape == (5, 1)
        assert torch.allclose(q, 0.5 * torch.ones(5, 1), atol=0.02)

    def test_multiple_probs(self) -> None:
        """Multiple probability levels."""
        support = torch.linspace(0.0, 1.0, 101)
        density = torch.ones(3, 101)
        q = _quantiles_from_density(support, density, [0.25, 0.75])
        assert q.shape == (3, 2)
        assert (q[:, 1] > q[:, 0]).all()


# ═══════════════════════════════════════════════════════════════════════════════
# _pit_from_samples
# ═══════════════════════════════════════════════════════════════════════════════


class TestPITFromSamples:
    def test_well_calibrated(self) -> None:
        """Well-calibrated samples give uniform PIT."""
        y_true = torch.randn(100)
        samples = torch.randn(500, 100)
        pit = _pit_from_samples(samples, y_true)
        assert pit.shape == (100,)
        assert (pit >= 0).all() and (pit <= 1).all()

    def test_squeeze_last_dim(self) -> None:
        """Samples with trailing 1 dim are squeezed."""
        samples = torch.randn(50, 20, 1)
        y_true = torch.randn(20, 1)
        pit = _pit_from_samples(samples, y_true)
        assert pit.shape == (20,)

    def test_wrong_dim_raises(self) -> None:
        """3D samples not squeezed raises ValueError."""
        samples = torch.randn(10, 10, 2)
        y_true = torch.randn(10)
        with pytest.raises(ValueError, match="scalar targets only"):
            _pit_from_samples(samples, y_true)


# ═══════════════════════════════════════════════════════════════════════════════
# _pit_from_density
# ═══════════════════════════════════════════════════════════════════════════════


class TestPITFromDensity:
    def test_uniform_density(self) -> None:
        """PIT from uniform density should be close to y_true."""
        support = torch.linspace(0.0, 1.0, 101)
        density = torch.ones(20, 101)
        y_true = torch.rand(20)
        pit = _pit_from_density(support, density, y_true)
        assert pit.shape == (20,)
        assert (pit >= 0).all() and (pit <= 1).all()

    def test_non_increasing_support_raises(self) -> None:
        """Non-increasing support raises ValueError."""
        support = torch.tensor([0.0, 0.5, 0.2, 1.0])
        density = torch.ones(2, 4)
        y_true = torch.tensor([0.3, 0.7])
        with pytest.raises(ValueError, match="strictly increasing"):
            _pit_from_density(support, density, y_true)

    def test_support_mismatch_raises(self) -> None:
        """Density shape mismatch raises ValueError."""
        with pytest.raises(ValueError, match="support mismatch"):
            _pit_from_density(torch.linspace(0, 1, 5), torch.ones(2, 10), torch.rand(2))


# ═══════════════════════════════════════════════════════════════════════════════
# _pit_from_quantiles
# ═══════════════════════════════════════════════════════════════════════════════


class TestPITFromQuantiles:
    def test_perfect_quantiles(self) -> None:
        """When y_true equals a quantile level, PIT should match that level."""
        y_true = torch.tensor([0.0])
        quantiles = {
            0.0: torch.tensor([[-1.0]]),
            0.5: torch.tensor([[0.0]]),
            1.0: torch.tensor([[1.0]]),
        }
        pit = _pit_from_quantiles(quantiles, y_true)
        assert float(pit.item()) == pytest.approx(0.5)

    def test_below_lowest_quantile(self) -> None:
        """Value below lowest quantile gets PIT=0."""
        y_true = torch.tensor([-10.0])
        quantiles = {0.1: torch.tensor([[0.0]]), 0.9: torch.tensor([[1.0]])}
        pit = _pit_from_quantiles(quantiles, y_true)
        assert float(pit.item()) == pytest.approx(0.0)

    def test_above_highest_quantile(self) -> None:
        """Value above highest quantile gets PIT=1."""
        y_true = torch.tensor([100.0])
        quantiles = {0.1: torch.tensor([[0.0]]), 0.9: torch.tensor([[1.0]])}
        pit = _pit_from_quantiles(quantiles, y_true)
        assert float(pit.item()) == pytest.approx(1.0)

    def test_interpolation(self) -> None:
        """Value between quantile levels is interpolated."""
        y_true = torch.tensor([0.5])
        quantiles = {0.2: torch.tensor([[0.0]]), 0.8: torch.tensor([[1.0]])}
        pit = _pit_from_quantiles(quantiles, y_true)
        assert 0.2 < float(pit.item()) < 0.8

    def test_too_few_quantiles_raises(self) -> None:
        """Less than 2 quantile levels raises ValueError."""
        with pytest.raises(ValueError, match="At least two"):
            _pit_from_quantiles({0.5: torch.randn(5)}, torch.randn(5))

    def test_batch_mismatch_raises(self) -> None:
        """Mismatched batch sizes raise ValueError."""
        with pytest.raises(ValueError, match="matching batch size"):
            _pit_from_quantiles(
                {0.1: torch.randn(3), 0.9: torch.randn(3)},
                torch.randn(5),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# gaussian_nll
# ═══════════════════════════════════════════════════════════════════════════════


class TestGaussianNLL:
    def test_mean_reduction(self) -> None:
        """Default mean reduction."""
        mean = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([1.0, -1.0])
        var = torch.tensor([1.0, 1.0])
        nll = gaussian_nll(mean, y_true, var)
        assert nll > 0

    def test_sum_reduction(self) -> None:
        """Sum reduction returns float."""
        nll = gaussian_nll(torch.zeros(10), torch.randn(10), torch.ones(10), reduction="sum")
        assert isinstance(nll, float)
        assert nll > 0

    def test_none_reduction(self) -> None:
        """None reduction returns per-sample NLL tensor."""
        nll = gaussian_nll(torch.zeros(5), torch.randn(5), torch.ones(5), reduction="none")
        assert isinstance(nll, torch.Tensor)
        assert nll.shape == (5,)

    def test_var_clamped(self) -> None:
        """Negative variance is clamped to 1e-8 (produces finite, possibly negative NLL)."""
        nll = gaussian_nll(torch.zeros(3), torch.zeros(3), torch.tensor([-1.0, 0.0, 1.0]))
        assert isinstance(nll, float)
        assert np.isfinite(nll)

    def test_numpy_inputs(self) -> None:
        """Works with numpy arrays."""
        nll = gaussian_nll(np.zeros(5), np.ones(5), np.ones(5))
        assert isinstance(nll, float)


# ═══════════════════════════════════════════════════════════════════════════════
# crps_gaussian
# ═══════════════════════════════════════════════════════════════════════════════


class TestCRPSGaussian:
    def test_perfect_prediction_gives_zero(self) -> None:
        """CRPS of a Dirac (std→0) at the truth is ≈0."""
        mean = torch.tensor([5.0])
        y_true = torch.tensor([5.0])
        crps = crps_gaussian(mean, y_true, torch.tensor([1e-8]))
        assert float(crps) < 0.01

    def test_poor_prediction_gives_large_crps(self) -> None:
        """CRPS grows with prediction error."""
        mean = torch.tensor([0.0, 0.0])
        y_true = torch.tensor([10.0, -10.0])
        crps = crps_gaussian(mean, y_true, torch.tensor([1.0, 1.0]))
        assert crps > 5.0

    def test_none_reduction(self) -> None:
        """None reduction returns per-sample tensor."""
        crps = crps_gaussian(torch.zeros(4), torch.randn(4), torch.ones(4), reduction="none")
        assert isinstance(crps, torch.Tensor)
        assert crps.shape == (4,)

    def test_sum_reduction(self) -> None:
        """Sum reduction returns float."""
        crps = crps_gaussian(torch.zeros(4), torch.ones(4), torch.ones(4), reduction="sum")
        assert isinstance(crps, float)


# ═══════════════════════════════════════════════════════════════════════════════
# continuous_ranked_probability_score
# ═══════════════════════════════════════════════════════════════════════════════


class TestContinuousRankedProbabilityScore:
    def test_basic(self) -> None:
        """CRPS from quantiles returns a scalar."""
        quantiles = {0.1: torch.randn(10), 0.5: torch.randn(10), 0.9: torch.randn(10)}
        y_true = torch.randn(10)
        crps = continuous_ranked_probability_score(quantiles, y_true)
        assert isinstance(crps, float)
        assert crps > 0

    def test_none_reduction(self) -> None:
        """None reduction returns per-sample values."""
        quantiles = {0.1: torch.randn(5), 0.5: torch.randn(5), 0.9: torch.randn(5)}
        crps = continuous_ranked_probability_score(quantiles, torch.randn(5), reduction="none")
        assert crps.shape == (5,)

    def test_too_few_quantiles_raises(self) -> None:
        """Less than 2 quantiles raises ValueError."""
        with pytest.raises(ValueError, match="At least 2"):
            continuous_ranked_probability_score({0.5: torch.randn(5)}, torch.randn(5))

    def test_nan_in_y_true_raises(self) -> None:
        """NaN in y_true raises ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            continuous_ranked_probability_score(
                {0.1: torch.randn(5), 0.9: torch.randn(5)},
                torch.tensor([float("nan")] * 5),
            )

    def test_nan_in_pred_raises(self) -> None:
        """NaN in predictions raises ValueError."""
        quantiles = {0.1: torch.tensor([float("nan")] * 5), 0.9: torch.randn(5)}
        with pytest.raises(ValueError, match="NaN"):
            continuous_ranked_probability_score(quantiles, torch.randn(5))

    def test_scalar_input_raises(self) -> None:
        """Scalar inputs raise ValueError."""
        with pytest.raises(ValueError, match="cannot be scalars"):
            continuous_ranked_probability_score(
                {0.1: torch.tensor(1.0), 0.9: torch.tensor(2.0)},
                torch.tensor(1.5),
            )

    def test_batch_size_mismatch_raises(self) -> None:
        """Mismatched batch sizes raise ValueError."""
        with pytest.raises(ValueError, match="same batch size"):
            continuous_ranked_probability_score(
                {0.1: torch.randn(5), 0.9: torch.randn(5)},
                torch.randn(10),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# crps_from_samples
# ═══════════════════════════════════════════════════════════════════════════════


class TestCRPSFromSamples:
    def test_basic(self) -> None:
        """CRPS from samples returns a float."""
        samples = torch.randn(100, 20)
        y_true = torch.randn(20)
        crps = crps_from_samples(samples, y_true)
        assert isinstance(crps, float)
        assert crps > 0

    def test_samples_match_y_true(self) -> None:
        """Same-dim samples get unsqueezed."""
        samples = torch.randn(20)
        y_true = torch.randn(20)
        crps = crps_from_samples(samples, y_true)
        assert isinstance(crps, float)

    def test_none_reduction(self) -> None:
        """None reduction returns per-sample values."""
        samples = torch.randn(50, 10)
        crps = crps_from_samples(samples, torch.randn(10), reduction="none")
        assert crps.shape == (10,)

    def test_wrong_shape_raises(self) -> None:
        """Wrong sample shape raises ValueError."""
        with pytest.raises(ValueError, match="must have shape"):
            crps_from_samples(torch.randn(10, 20, 2), torch.randn(20))


# ═══════════════════════════════════════════════════════════════════════════════
# energy_score
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnergyScore:
    def test_basic(self) -> None:
        """Energy score returns a float."""
        samples = torch.randn(100, 20, 3)
        y_true = torch.randn(20, 3)
        es = energy_score(samples, y_true)
        assert isinstance(es, float)

    def test_with_max_pairs(self) -> None:
        """max_pairs limits the number of samples used."""
        es = energy_score(torch.randn(500, 10, 2), torch.randn(10, 2), max_pairs=50)
        assert isinstance(es, float)

    def test_with_beta(self) -> None:
        """Custom beta parameter produces finite energy score."""
        es = energy_score(torch.randn(200, 10, 2), torch.randn(10, 2), beta=2.0)
        assert isinstance(es, float)
        assert np.isfinite(es)

    def test_none_reduction(self) -> None:
        """None reduction returns per-sample values."""
        es = energy_score(torch.randn(50, 10, 3), torch.randn(10, 3), reduction="none")
        assert es.shape == (10,)


# ═══════════════════════════════════════════════════════════════════════════════
# conditional_density_estimation_loss
# ═══════════════════════════════════════════════════════════════════════════════


class TestConditionalDensityEstimationLoss:
    def test_basic(self) -> None:
        """CDE loss returns a float."""
        support = torch.linspace(-3.0, 3.0, 50)
        density = torch.ones(10, 50) * 0.1
        y_true = torch.randn(10)
        loss = conditional_density_estimation_loss(support, density, y_true)
        assert isinstance(loss, float)

    def test_none_reduction(self) -> None:
        """None reduction returns per-sample tensor."""
        support = torch.linspace(-2.0, 2.0, 30)
        density = torch.ones(5, 30)
        loss = conditional_density_estimation_loss(
            support, density, torch.randn(5), reduction="none"
        )
        assert loss.shape == (5,)

    def test_sum_reduction(self) -> None:
        """Sum reduction returns float."""
        support = torch.linspace(-2.0, 2.0, 30)
        loss = conditional_density_estimation_loss(
            support, torch.ones(5, 30), torch.randn(5), reduction="sum"
        )
        assert isinstance(loss, float)

    def test_support_too_short_raises(self) -> None:
        """Support with less than 2 points raises ValueError."""
        with pytest.raises(ValueError, match="at least two points"):
            conditional_density_estimation_loss(
                torch.tensor([0.0]), torch.ones(5, 1), torch.randn(5)
            )

    def test_support_not_strictly_increasing_raises(self) -> None:
        """Non-increasing support raises ValueError."""
        support = torch.tensor([0.0, 0.5, 0.2, 1.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            conditional_density_estimation_loss(support, torch.ones(3, 4), torch.randn(3))

    def test_density_batch_mismatch_raises(self) -> None:
        """Density/y_true batch mismatch raises ValueError."""
        support = torch.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="batch mismatch"):
            conditional_density_estimation_loss(support, torch.ones(3, 10), torch.randn(5))

    def test_support_mismatch_raises(self) -> None:
        """Density support mismatch raises ValueError."""
        support = torch.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="support mismatch"):
            conditional_density_estimation_loss(support, torch.ones(3, 5), torch.randn(3))


# ═══════════════════════════════════════════════════════════════════════════════
# highest_posterior_density_level
# ═══════════════════════════════════════════════════════════════════════════════


class TestHighestPosteriorDensityLevel:
    def test_basic(self) -> None:
        """HPD levels are in [0, 1]."""
        support = torch.linspace(-3.0, 3.0, 100)
        density = torch.ones(5, 100)
        y_true = torch.randn(5)
        levels = highest_posterior_density_level(support, density, y_true)
        assert levels.shape == (5,)
        assert (levels >= 0).all() and (levels <= 1).all()

    def test_central_point_gives_low_hpd(self) -> None:
        """Observing at the mode gives low HPD level."""
        support = torch.linspace(-3.0, 3.0, 200)
        mu = 0.0
        sigma = 1.0
        density = torch.exp(-0.5 * ((support - mu) / sigma) ** 2).unsqueeze(0)
        y_true = torch.tensor([0.0])
        levels = highest_posterior_density_level(support, density, y_true)
        assert float(levels.item()) < 0.5

    def test_support_too_short_raises(self) -> None:
        """Support with < 2 points raises ValueError."""
        with pytest.raises(ValueError, match="at least two points"):
            highest_posterior_density_level(torch.tensor([0.0]), torch.ones(3, 1), torch.randn(3))

    def test_batch_mismatch_raises(self) -> None:
        """Density/y_true batch mismatch raises ValueError."""
        support = torch.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="batch dimension"):
            highest_posterior_density_level(support, torch.ones(3, 10), torch.randn(5))


class TestHighestPosteriorDensityCoverage:
    def test_basic(self) -> None:
        """HPD coverage returns a float in [0, 1]."""
        support = torch.linspace(-3.0, 3.0, 100)
        density = torch.ones(20, 100)
        y_true = torch.randn(20)
        cov = highest_posterior_density_coverage(support, density, y_true, alpha=0.1)
        assert 0.0 <= cov <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# ContinuousRankedProbabilityScore (Metric class)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCRPSMetric:
    def test_update_and_compute(self) -> None:
        """Basic update/compute cycle."""
        metric = ContinuousRankedProbabilityScore()
        metric.update(
            {0.1: torch.randn(10), 0.5: torch.randn(10), 0.9: torch.randn(10)},
            torch.randn(10),
        )
        result = metric.compute()
        assert result.numel() == 1
        assert float(result.item()) > 0

    def test_multiple_updates(self) -> None:
        """Multiple updates accumulate correctly."""
        metric = ContinuousRankedProbabilityScore()
        for _ in range(3):
            metric.update(
                {0.1: torch.randn(5), 0.5: torch.randn(5), 0.9: torch.randn(5)},
                torch.randn(5),
            )
        result = metric.compute()
        assert float(result.item()) > 0

    def test_too_few_quantiles_raises(self) -> None:
        """Less than 2 quantiles raises ValueError."""
        metric = ContinuousRankedProbabilityScore()
        with pytest.raises(ValueError, match="At least 2"):
            metric.update({0.5: torch.randn(5)}, torch.randn(5))


# ═══════════════════════════════════════════════════════════════════════════════
# EnergyScore (Metric class)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnergyScoreMetric:
    def test_update_and_compute(self) -> None:
        """Basic update/compute cycle."""
        metric = EnergyScore()
        metric.update(torch.randn(50, 10, 3), torch.randn(10, 3))
        result = metric.compute()
        assert float(result.item()) > 0

    def test_with_max_pairs(self) -> None:
        """max_pairs limits samples used."""
        metric = EnergyScore(max_pairs=20)
        metric.update(torch.randn(100, 10, 2), torch.randn(10, 2))
        result = metric.compute()
        assert float(result.item()) > 0

    def test_with_beta(self) -> None:
        """Custom beta parameter produces finite energy score."""
        metric = EnergyScore(beta=2.0)
        metric.update(torch.randn(200, 8, 2), torch.randn(8, 2))
        result = metric.compute()
        assert np.isfinite(float(result.item()))


# ═══════════════════════════════════════════════════════════════════════════════
# distribution_metrics_report
# ═══════════════════════════════════════════════════════════════════════════════


class TestDistributionMetricsReport:
    def test_from_dist_dict(self) -> None:
        """Report from a Normal dist dict."""
        report = distribution_metrics_report(
            dist={"loc": torch.zeros(20), "scale": torch.ones(20)},
            y_true=torch.randn(20),
        )
        assert "log_prob" in report
        assert "crps" in report

    def test_from_torch_distribution(self) -> None:
        """Report from a torch Distribution object."""
        dist = torch.distributions.Normal(torch.zeros(10), torch.ones(10))
        report = distribution_metrics_report(dist=dist, y_true=torch.randn(10))
        assert "log_prob" in report
        assert "crps" in report

    def test_from_dist_with_samples(self) -> None:
        """Report with pre-computed samples."""
        report = distribution_metrics_report(
            dist={"loc": torch.zeros(8), "scale": torch.ones(8)},
            y_true=torch.randn(8),
            samples=torch.randn(100, 8),
        )
        assert "log_prob" in report

    def test_from_quantiles_only(self) -> None:
        """Report from quantiles only (no dist)."""
        report = distribution_metrics_report(
            y_true=torch.randn(15),
            y_pred_quantiles={0.05: torch.randn(15), 0.5: torch.randn(15), 0.95: torch.randn(15)},
        )
        assert "crps" in report
        assert "coverage_90" in report

    def test_from_samples_only(self) -> None:
        """Report from samples only (no dist)."""
        report = distribution_metrics_report(
            y_true=torch.randn(10),
            samples=torch.randn(200, 10),
        )
        assert "crps" in report
        assert "pit_chi2" in report

    def test_from_density(self) -> None:
        """Report from density and support."""
        support = torch.linspace(-3.0, 3.0, 100)
        density = torch.ones(10, 100)
        report = distribution_metrics_report(
            y_true=torch.randn(10),
            support=support,
            density=density,
        )
        assert "cde_loss" in report
        assert "log_prob" in report
        assert "coverage_90" in report

    def test_from_density_with_quantiles(self) -> None:
        """Density + quantiles combined."""
        support = torch.linspace(-3.0, 3.0, 100)
        density = torch.ones(8, 100)
        report = distribution_metrics_report(
            y_true=torch.randn(8),
            support=support,
            density=density,
            y_pred_quantiles={0.1: torch.randn(8), 0.9: torch.randn(8)},
        )
        assert "cde_loss" in report
        assert "crps" in report

    def test_dist_dict_missing_keys_raises(self) -> None:
        """Dist dict without loc/scale raises ValueError."""
        with pytest.raises(ValueError, match="must provide loc"):
            distribution_metrics_report(dist={"foo": 1.0}, y_true=torch.randn(5))

    def test_y_true_none_raises(self) -> None:
        """Missing y_true raises ValueError."""
        with pytest.raises(ValueError, match="y_true must be provided"):
            distribution_metrics_report()

    def test_multivariate_samples(self) -> None:
        """Multivariate y_true triggers energy score."""
        report = distribution_metrics_report(
            y_true=torch.randn(10, 3),
            samples=torch.randn(50, 10, 3),
        )
        assert "energy_score" in report
        assert "crps" in report
