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
    ShiftFactoredPredictiveTransport,
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
        # n=5, k = ceil((5+1)*(1-0.4)) = ceil(3.6) = 4 -> the 4th smallest score (index 3).
        assert q == 3.0

    def test_alpha_near_zero(self) -> None:
        """Too few scores for the requested alpha: k = ceil(11 * 0.99) = 11 > n = 10 -> +inf."""
        scores = np.arange(10, dtype=float)
        q = _finite_sample_quantile(scores, alpha=0.01)
        assert q == float("inf")

    def test_largest_score_when_k_equals_n(self) -> None:
        """k = ceil(11 * 0.9) = 10 = n -> the maximum score (finite)."""
        scores = np.arange(10, dtype=float)
        assert _finite_sample_quantile(scores, alpha=0.1) == 9.0


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

    def test_monotonic_width_by_alpha(self) -> None:
        """Monotonic width by alpha."""
        batch = PredictiveBatch(
            mean=np.array([0.0], dtype=np.float32),
            std=np.array([1.0], dtype=np.float32),
            extra={"family": "gaussian"},
        )
        widths = {}
        for alpha in [0.01, 0.05, 0.1, 0.2]:
            lower, upper = _native_interval(batch, alpha=alpha, eps=1e-8)
            widths[alpha] = (upper - lower)[0]

        assert widths[0.01] > widths[0.05]
        assert widths[0.05] > widths[0.1]
        assert widths[0.1] > widths[0.2]

    def test_invalid_alpha_raises_value_error(self) -> None:
        """Invalid alpha raises ValueError."""
        batch = PredictiveBatch(
            mean=np.array([0.0], dtype=np.float32),
            std=np.array([1.0], dtype=np.float32),
            extra={"family": "gaussian"},
        )
        with pytest.raises(ValueError, match="alpha must be in"):
            _native_interval(batch, alpha=0.0, eps=1e-8)
        with pytest.raises(ValueError, match="alpha must be in"):
            _native_interval(batch, alpha=-0.1, eps=1e-8)
        with pytest.raises(ValueError, match="alpha must be in"):
            _native_interval(batch, alpha=1.0, eps=1e-8)
        with pytest.raises(ValueError, match="alpha must be in"):
            _native_interval(batch, alpha=1.5, eps=1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# Label shift: transport must not be inert, nor collapse past the source range
# (harness finding 8: PriorTV 1.0 / TransportApplied 0 for n_support 16-256)
# ═══════════════════════════════════════════════════════════════════════════════


def _ridge_gaussian(
    X_s: np.ndarray, y_s: np.ndarray, X_t: np.ndarray, alpha: float
) -> tuple[PredictiveBatch, PredictiveBatch, float]:
    """Ridge mean + residual std: a Gaussian predictive (calibrated for jointly Gaussian data)."""
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=alpha).fit(X_s, y_s)
    p_s, p_t = model.predict(X_s), model.predict(X_t)
    sigma = float(np.std(y_s - p_s))

    def batch(m: np.ndarray) -> PredictiveBatch:
        t = torch.from_numpy(m)
        return PredictiveBatch(point=t, mean=t, std=torch.full_like(t, sigma))

    return batch(p_s), batch(p_t), sigma


def _gaussian_label_shift(seed: int, shift: float) -> tuple[np.ndarray, ...]:
    """``y_s ~ N(0, 1)``, ``y_t ~ N(shift, 1)``, ``x | y ~ N(y, 0.5^2)`` (4 copies): pure label shift."""
    rng = np.random.default_rng(seed)
    y_s = rng.normal(size=3000)
    y_t = rng.normal(loc=shift, size=2000)
    X_s = y_s[:, None] + 0.5 * rng.normal(size=(3000, 4))
    X_t = y_t[:, None] + 0.5 * rng.normal(size=(2000, 4))
    return X_s, y_s, X_t, y_t


def _gauss_nll(mean: torch.Tensor, std: torch.Tensor, y: np.ndarray) -> float:
    m, s = mean.double().numpy(), std.double().numpy()
    return float(np.mean(0.5 * np.log(2 * np.pi * s**2) + 0.5 * ((y - m) / s) ** 2))


def _fit_transport(
    src: PredictiveBatch, y_s: np.ndarray, **cfg: object
) -> ShiftFactoredPredictiveTransport:
    config = ShiftFactoredTransportConfig(**cfg)  # type: ignore[arg-type]
    return ShiftFactoredPredictiveTransport(config).fit_source(src, torch.from_numpy(y_s))


class TestTransportUnderLabelShift:
    def test_margin_collapse_is_a_noop_in_the_stabiliser(self) -> None:
        """Why the pre-fix transport was inert: with all EM mass on a bin whose source
        prior is ``eps`` (an empty margin bin of the old label histogram), the shrunk
        prior is ``(1 - w) * src`` on every other bin, a uniform ratio that
        renormalises back to the source prior for any shrink weight ``w``."""
        from torchregress.test_time.transport import _stabilize_target_prior

        cfg = ShiftFactoredTransportConfig()
        src = np.array([1e-8, 0.2, 0.3, 0.3, 0.2 - 1e-8])
        tgt = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        confident = torch.eye(5, dtype=torch.float64)[[1, 2, 3]]
        stabilised, meta = _stabilize_target_prior(
            source_prior=src,
            target_prior=tgt,
            selected_probabilities=confident,
            converged=True,
            config=cfg,
        )
        assert meta["prior_shrink_weight"] > 0.0
        assert _discrete_tv(src, tgt) == pytest.approx(1.0)
        np.testing.assert_allclose(stabilised.numpy(), src, atol=1e-6)

    @pytest.mark.parametrize("n_support", [16, 32, 64, 128, 256])
    def test_ridge_on_diabetes_is_not_inert(self, n_support: int) -> None:
        """The harness case: Ridge on diabetes, logistic-in-y resample of the test split."""
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split

        X, y = load_diabetes(return_X_y=True)
        X_s, X_te, y_s, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        rng = np.random.default_rng(0)
        z = (y_te - y_te.mean()) / y_te.std()
        p = 1.0 / (1.0 + np.exp(-1.5 * z))
        idx = rng.choice(len(y_te), size=len(y_te), p=p / p.sum())
        src, tgt, _ = _ridge_gaussian(X_s, y_s, X_te[idx], alpha=1e-3)
        out = _fit_transport(src, y_s, n_support=n_support).adapt_unlabeled_target(
            target_predictions=tgt
        )
        meta = out.extra or {}
        assert meta["transport_applied"] is True
        # Used to be 1.0: the EM prior sat on the two empty margin bins.
        assert meta["prior_source_target_tv"] < 0.7
        assert meta["prior_mass_outside_source_range"] < 0.05
        # The stabilised prior moves toward the high labels the target over-represents.
        support = np.linspace(0, 1, n_support)  # monotone stand-in for the grid
        src_prior = np.asarray(meta["source_prior"])
        adapted = np.asarray(meta["target_prior"])
        assert float(adapted @ support) > float(src_prior @ support)

    @pytest.mark.parametrize("n_support", [16, 64, 256])
    def test_known_prior_shift_improves_target_nll(self, n_support: int) -> None:
        X_s, y_s, X_t, y_t = _gaussian_label_shift(seed=0, shift=0.8)
        src, tgt, sigma = _ridge_gaussian(X_s, y_s, X_t, alpha=1.0)
        transport = _fit_transport(
            src, y_s, n_support=n_support, enable_uncertainty_inflation=False
        )
        out = transport.adapt_unlabeled_target(target_predictions=tgt)
        meta = out.extra or {}
        assert meta["transport_applied"] is True
        # Raw EM prior vs the empirical target label prior, on 8 coarse bins.
        assert transport.state_ is not None
        support = transport.state_.source_support.numpy()
        coarse = np.linspace(support[0], support[-1], 9)
        est = np.histogram(support, bins=coarse, weights=np.asarray(meta["target_prior_raw"]))[0]
        src_c = np.histogram(support, bins=coarse, weights=np.asarray(meta["source_prior"]))[0]
        true = np.histogram(y_t, bins=coarse)[0] / len(y_t)
        tv_raw, tv_none = 0.5 * np.abs(est - true).sum(), 0.5 * np.abs(src_c - true).sum()
        assert tv_raw < 0.3 * tv_none
        assert out.mean is not None and out.std is not None and tgt.mean is not None
        base = _gauss_nll(tgt.mean, torch.full_like(tgt.mean, sigma), y_t)
        assert _gauss_nll(out.mean, out.std, y_t) <= base + 1e-3

    def test_near_identity_prior_keeps_the_gaussian_predictive(self) -> None:
        """A transported prior equal to the source prior up to a tiny change must return
        (almost) the input predictive: the Gaussian output used to be the moments of the
        discretised, grid-truncated predictive, which moved it even then."""
        X_s, y_s, X_t, _ = _gaussian_label_shift(seed=1, shift=1.5)
        src, tgt, sigma = _ridge_gaussian(X_s, y_s, X_t, alpha=1.0)
        out = _fit_transport(
            src,
            y_s,
            n_support=64,
            prior_transport_strength=0.01,
            enable_uncertainty_inflation=False,
        ).adapt_unlabeled_target(target_predictions=tgt)
        assert (out.extra or {})["transport_applied"] is True
        assert out.mean is not None and out.std is not None and tgt.mean is not None
        shift = (out.mean - tgt.mean).abs() / sigma
        # Before: mean 0.014, max 3.7 sigma (edge rows), std ratio off by up to 0.84.
        assert float(shift.mean()) < 0.01 and float(shift.max()) < 0.5
        assert float((out.std / sigma - 1.0).abs().max()) < 0.2

    @pytest.mark.parametrize("n_support", [16, 256])
    def test_target_past_source_range_warns_and_does_not_collapse(self, n_support: int) -> None:
        """Target labels beyond the source range (support violation): EM mass lands on the
        margin bins. Warn, and do not collapse the predictive onto the grid edge
        (std ~ grid step and NLL 1e2-1e4 before)."""
        X_s, y_s, X_t, y_t = _gaussian_label_shift(seed=0, shift=4.0)
        src, tgt, sigma = _ridge_gaussian(X_s, y_s, X_t, alpha=1.0)
        transport = _fit_transport(src, y_s, n_support=n_support)
        with pytest.warns(UserWarning, match="outside the source label range"):
            out = transport.adapt_unlabeled_target(target_predictions=tgt)
        meta = out.extra or {}
        assert meta["prior_mass_outside_source_range"] > 0.5
        assert out.mean is not None and out.std is not None and tgt.mean is not None
        assert float(out.std.min()) > 0.5 * sigma
        base = _gauss_nll(tgt.mean, torch.full_like(tgt.mean, sigma), y_t)
        assert _gauss_nll(out.mean, out.std, y_t) < base + 0.5

    def test_no_warning_without_shift(self) -> None:
        import warnings

        X_s, y_s, X_t, _ = _gaussian_label_shift(seed=2, shift=0.0)
        src, tgt, _ = _ridge_gaussian(X_s, y_s, X_t, alpha=1.0)
        transport = _fit_transport(src, y_s)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = transport.adapt_unlabeled_target(target_predictions=tgt)
        assert (out.extra or {})["prior_mass_outside_source_range"] < 0.05
