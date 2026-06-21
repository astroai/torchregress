"""
Unit tests for torchregress.test_time.transport — config validation and utilities.

Existing smoke tests in test_prediction_and_test_time.py exercise the
ShiftFactoredPredictiveTransport orchestrator end-to-end. This file fills
the gap: dataclass integrity, config __post_init__ validation, and
internal helper functions that don't require a full transport pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchregress.prediction import PredictiveBatch
from torchregress.test_time.transport import (
    ShiftFactoredTransportConfig,
    ShiftFactoredTransportState,
    _as_1d,
    _as_2d,
    _density_to_probabilities,
    _discrete_tv,
    _finite_sample_quantile,
    _gaussian_density_on_support,
    _native_interval,
    _probability_moments,
    _resample_density,
    _support_edges,
    _support_grid_from_targets,
    _to_numpy,
    _uniform_dx,
)

# ═══════════════════════════════════════════════════════════════════════════════
# ShiftFactoredTransportConfig — validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestShiftFactoredTransportConfig:
    def test_defaults(self) -> None:
        """Defaults."""
        cfg = ShiftFactoredTransportConfig()
        assert cfg.n_support == 256
        assert cfg.alpha == 0.1
        assert cfg.top_fraction == 0.5
        assert cfg.enable_alignment is True

    def test_frozen(self) -> None:
        """Frozen."""
        cfg = ShiftFactoredTransportConfig()
        with pytest.raises(Exception):
            cfg.n_support = 100  # type: ignore[misc]

    def test_n_support_too_low(self) -> None:
        """N support too low."""
        with pytest.raises(ValueError, match="n_support"):
            ShiftFactoredTransportConfig(n_support=8)

    def test_support_margin_negative(self) -> None:
        """Support margin negative."""
        with pytest.raises(ValueError, match="support_margin"):
            ShiftFactoredTransportConfig(support_margin=-0.1)

    def test_alpha_out_of_range(self) -> None:
        """Alpha out of range."""
        with pytest.raises(ValueError, match="alpha"):
            ShiftFactoredTransportConfig(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            ShiftFactoredTransportConfig(alpha=1.0)

    def test_top_fraction_out_of_range(self) -> None:
        """Top fraction out of range."""
        with pytest.raises(ValueError, match="top_fraction"):
            ShiftFactoredTransportConfig(top_fraction=0.0)
        with pytest.raises(ValueError, match="top_fraction"):
            ShiftFactoredTransportConfig(top_fraction=1.5)

    def test_min_selection_count_non_positive(self) -> None:
        """Min selection count non positive."""
        with pytest.raises(ValueError, match="min_selection_count"):
            ShiftFactoredTransportConfig(min_selection_count=0)

    def test_local_consistency_k_non_positive(self) -> None:
        """Local consistency k non positive."""
        with pytest.raises(ValueError, match="local_consistency_k"):
            ShiftFactoredTransportConfig(local_consistency_k=0)

    def test_prior_estimation_rows_non_positive(self) -> None:
        """Prior estimation rows non positive."""
        with pytest.raises(ValueError, match="prior_estimation_rows"):
            ShiftFactoredTransportConfig(prior_estimation_rows=0)

    def test_prior_transport_strength_out_of_range(self) -> None:
        """Prior transport strength out of range."""
        with pytest.raises(ValueError, match="prior_transport_strength"):
            ShiftFactoredTransportConfig(prior_transport_strength=-0.1)
        with pytest.raises(ValueError, match="prior_transport_strength"):
            ShiftFactoredTransportConfig(prior_transport_strength=1.5)

    def test_prior_ratio_clip_below_one(self) -> None:
        """Prior ratio clip below one."""
        with pytest.raises(ValueError, match="prior_ratio_clip"):
            ShiftFactoredTransportConfig(prior_ratio_clip=0.5)

    def test_prior_transport_min_selected_fraction_negative(self) -> None:
        """Prior transport min selected fraction negative."""
        with pytest.raises(ValueError, match="prior_transport_min_selected_fraction"):
            ShiftFactoredTransportConfig(prior_transport_min_selected_fraction=-0.1)

    def test_prior_transport_max_prior_tv_out_of_range(self) -> None:
        """Prior transport max prior TV out of range."""
        with pytest.raises(ValueError, match="prior_transport_max_prior_tv"):
            ShiftFactoredTransportConfig(prior_transport_max_prior_tv=1.5)

    def test_uncertainty_base_temperature_non_positive(self) -> None:
        """Uncertainty base temperature non positive."""
        with pytest.raises(ValueError, match="uncertainty_base_temperature"):
            ShiftFactoredTransportConfig(uncertainty_base_temperature=0.0)

    def test_uncertainty_slope_negative(self) -> None:
        """Uncertainty slope negative."""
        with pytest.raises(ValueError, match="uncertainty_slope"):
            ShiftFactoredTransportConfig(uncertainty_slope=-0.1)

    def test_uncertainty_max_temp_below_base(self) -> None:
        """Uncertainty max temp below base."""
        with pytest.raises(ValueError, match="uncertainty_max_temperature"):
            ShiftFactoredTransportConfig(
                uncertainty_base_temperature=2.0, uncertainty_max_temperature=1.0
            )

    def test_uncertainty_clip_quantile_out_of_range(self) -> None:
        """Uncertainty clip quantile out of range."""
        with pytest.raises(ValueError, match="uncertainty_clip_quantile"):
            ShiftFactoredTransportConfig(uncertainty_clip_quantile=0.6)

    def test_eps_non_positive(self) -> None:
        """Eps non positive."""
        with pytest.raises(ValueError, match="eps"):
            ShiftFactoredTransportConfig(eps=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# ShiftFactoredTransportState — dataclass
# ═══════════════════════════════════════════════════════════════════════════════


class TestShiftFactoredTransportState:
    def test_minimal_fields(self) -> None:
        """Minimal fields."""
        support = np.linspace(0, 1, 5)
        prior = np.ones(4) / 4
        targets = np.array([0.2, 0.5, 0.8])
        state = ShiftFactoredTransportState(
            source_support=support, source_prior=prior, source_targets=targets
        )
        np.testing.assert_array_equal(state.source_support, support)
        assert state.source_inputs is None
        assert state.last_target_prior is None

    def test_frozen(self) -> None:
        """Frozen."""
        state = ShiftFactoredTransportState(
            source_support=np.arange(3, dtype=float),
            source_prior=np.ones(2) / 2,
            source_targets=np.array([1.0]),
        )
        with pytest.raises(Exception):
            state.source_prior = np.array([0.5])  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════════
# _to_numpy
# ═══════════════════════════════════════════════════════════════════════════════


class TestToNumpy:
    def test_ndarray_passthrough(self) -> None:
        """Ndarray passthrough."""
        x = np.array([1.0, 2.0])
        result = _to_numpy(x)
        assert result is x

    def test_tensor_converted(self) -> None:
        """Tensor converted."""
        x = torch.tensor([1.0, 2.0])
        result = _to_numpy(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_list_converted(self) -> None:
        """List converted."""
        result = _to_numpy([1.0, 2.0, 3.0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


# ═══════════════════════════════════════════════════════════════════════════════
# _as_1d / _as_2d
# ═══════════════════════════════════════════════════════════════════════════════


class TestAs1d:
    def test_1d_tensor(self) -> None:
        """1d tensor."""
        result = _as_1d(torch.tensor([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_2d_flattened(self) -> None:
        """2d flattened."""
        result = _as_1d(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert result.shape == (4,)


class TestAs2d:
    def test_1d_becomes_column(self) -> None:
        """1d becomes column."""
        result = _as_2d(np.array([1.0, 2.0]))
        assert result.shape == (2, 1)

    def test_2d_stays(self) -> None:
        """2d stays."""
        result = _as_2d(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert result.shape == (2, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# _support_grid_from_targets
# ═══════════════════════════════════════════════════════════════════════════════


class TestSupportGridFromTargets:
    def test_basic(self) -> None:
        """Basic."""
        targets = np.array([0.0, 10.0])
        grid = _support_grid_from_targets(targets, n_support=256, support_margin=0.1)
        assert len(grid) == 256
        assert grid[0] < 0.0
        assert grid[-1] > 10.0

    def test_hi_lo_equal_fallback(self) -> None:
        """Hi lo equal fallback."""
        targets = np.array([3.0, 3.0])
        grid = _support_grid_from_targets(targets, n_support=16, support_margin=0.0)
        assert len(grid) >= 16
        assert grid[-1] > grid[0]

    def test_nan_raises(self) -> None:
        """Nan raises."""
        with pytest.raises(ValueError, match="finite"):
            _support_grid_from_targets(np.array([np.nan, 1.0]), n_support=5, support_margin=0.1)

    def test_minimum_16_points(self) -> None:
        """Minimum 16 points."""
        targets = np.array([0.0, 1.0])
        grid = _support_grid_from_targets(targets, n_support=4, support_margin=0.0)
        assert len(grid) == 16


# ═══════════════════════════════════════════════════════════════════════════════
# _support_edges / _uniform_dx
# ═══════════════════════════════════════════════════════════════════════════════


class TestSupportEdges:
    def test_basic(self) -> None:
        """Basic."""
        support = np.array([0.0, 1.0, 2.0])
        edges = _support_edges(support)
        assert len(edges) == 4
        assert edges[0] < support[0]
        assert edges[-1] > support[-1]

    def test_single_midpoint(self) -> None:
        """Single midpoint."""
        support = np.array([0.0, 1.0])
        edges = _support_edges(support)
        assert len(edges) == 3
        assert edges[1] == 0.5

    def test_raises_on_single_point(self) -> None:
        """Raises on single point."""
        with pytest.raises(ValueError, match="at least two"):
            _support_edges(np.array([1.0]))


class TestUniformDx:
    def test_uniform(self) -> None:
        """Uniform."""
        dx = _uniform_dx(np.array([0.0, 1.0, 2.0]))
        assert dx == 1.0

    def test_non_uniform_mean(self) -> None:
        """Non uniform mean."""
        dx = _uniform_dx(np.array([0.0, 0.5, 2.0]))
        assert dx > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Density / probability utilities
# ═══════════════════════════════════════════════════════════════════════════════


class TestDensityToProbabilities:
    def test_basic(self) -> None:
        """Basic."""
        support = np.linspace(0, 1, 5)
        density = np.ones((2, 5))
        probs = _density_to_probabilities(support, density, eps=1e-8)
        assert probs.shape == (2, 5)
        np.testing.assert_array_almost_equal(probs.sum(axis=1), 1.0)


class TestGaussianDensityOnSupport:
    def test_normalized(self) -> None:
        """Normalized."""
        mean = np.array([0.0, 1.0])
        std = np.array([0.5, 0.5])
        support = np.linspace(-2, 3, 50)
        density = _gaussian_density_on_support(mean, std, support, eps=1e-8)
        assert density.shape == (2, 50)
        integrals = np.trapezoid(density, support, axis=1)
        np.testing.assert_array_almost_equal(integrals, 1.0)

    def test_zero_std_clamped(self) -> None:
        """Zero std clamped."""
        mean = np.array([0.0])
        std = np.array([0.0])
        support = np.linspace(-1, 1, 20)
        density = _gaussian_density_on_support(mean, std, support, eps=1e-8)
        assert np.all(np.isfinite(density))
        assert density.shape == (1, len(support))


class TestResampleDensity:
    def test_1d_support(self) -> None:
        """1d support."""
        support_in = np.linspace(0, 1, 10)
        density_in = np.ones((2, 10))
        support_out = np.linspace(0, 1, 5)
        result = _resample_density(support_in, density_in, support_out, eps=1e-8)
        assert result.shape == (2, 5)
        integrals = np.trapezoid(result, support_out, axis=1)
        np.testing.assert_array_almost_equal(integrals, 1.0)

    def test_2d_support(self) -> None:
        """2d support."""
        support_in = np.array(
            [np.linspace(0, 1, 10), np.linspace(0, 1, 10)],
        )
        density_in = np.ones((2, 10))
        support_out = np.linspace(0, 1, 5)
        result = _resample_density(support_in, density_in, support_out, eps=1e-8)
        assert result.shape == (2, 5)


class TestProbabilityMoments:
    def test_basic(self) -> None:
        """Basic."""
        support = np.array([0.0, 1.0, 2.0])
        probs = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
        mean, std = _probability_moments(support, probs, eps=1e-8)
        assert mean.shape == (2,)
        assert std.shape == (2,)
        np.testing.assert_array_almost_equal(mean[0], 0.5)
        np.testing.assert_array_almost_equal(mean[1], 1.5)

    def test_delta_spike(self) -> None:
        """Delta spike."""
        support = np.linspace(0, 10, 20)
        probs = np.zeros((1, 20))
        probs[0, 10] = 1.0
        mean, std = _probability_moments(support, probs, eps=1e-8)
        np.testing.assert_array_almost_equal(mean[0], support[10])
        assert std[0] < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# _finite_sample_quantile
# ═══════════════════════════════════════════════════════════════════════════════


class TestFiniteSampleQuantile:
    def test_basic(self) -> None:
        """Basic."""
        scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        q = _finite_sample_quantile(scores, alpha=0.4)
        # n=5, ceil((5+1)*(1-0.4))/5 = ceil(3.6)/5 = 4/5 = 0.8
        # np.quantile with method="higher": smallest x s.t. F(x) > 0.8 → 4
        assert q == 4.0

    def test_alpha_near_zero(self) -> None:
        """Alpha near zero."""
        scores = np.arange(10, dtype=float)
        q = _finite_sample_quantile(scores, alpha=0.01)
        assert q > 0
        assert q <= 9


# ═══════════════════════════════════════════════════════════════════════════════
# _discrete_tv
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiscreteTV:
    def test_identical(self) -> None:
        """Identical."""
        p = np.array([0.3, 0.7])
        q = np.array([0.3, 0.7])
        assert _discrete_tv(p, q) == 0.0

    def test_disjoint(self) -> None:
        """Disjoint."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        assert _discrete_tv(p, q) == 1.0

    def test_partial_overlap(self) -> None:
        """Partial overlap."""
        p = np.array([0.5, 0.5])
        q = np.array([0.2, 0.8])
        # 0.5 * sum(|p-q|) = 0.5 * (0.3 + 0.3) = 0.3
        assert _discrete_tv(p, q) == pytest.approx(0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# _native_interval
# ═══════════════════════════════════════════════════════════════════════════════


class TestNativeInterval:
    def test_gaussian_family(self) -> None:
        """Gaussian family."""
        batch = PredictiveBatch(
            mean=np.array([0.0, 1.0], dtype=np.float32),
            std=np.array([0.5, 0.5], dtype=np.float32),
            extra={"family": "gaussian"},
        )
        lower, upper = _native_interval(batch, alpha=0.1, eps=1e-8)
        assert lower.shape == (2,)
        assert upper.shape == (2,)
        assert np.all(lower < upper)
        # 90% interval: z ≈ 1.645
        np.testing.assert_array_almost_equal(lower, [-0.8224, 0.1776], decimal=3)
        np.testing.assert_array_almost_equal(upper, [0.8224, 1.8224], decimal=3)

    def test_point_family_fallback(self) -> None:
        """Point family fallback."""
        batch = PredictiveBatch(
            point=np.array([1.0, 2.0], dtype=np.float64),
        )
        lower, upper = _native_interval(batch, alpha=0.1, eps=1e-8)
        assert np.all(lower < upper)

    def test_quantile_family(self) -> None:
        """Quantile family."""
        batch = PredictiveBatch(
            quantiles=np.array([[0.2, 0.5, 0.8], [0.3, 0.6, 0.9]], dtype=np.float32),
            quantile_levels=[0.1, 0.5, 0.9],
        )
        lower, upper = _native_interval(batch, alpha=0.1, eps=1e-8)
        assert lower.shape == (2,)
        assert upper.shape == (2,)
        assert np.all(lower < upper)
