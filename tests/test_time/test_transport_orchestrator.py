"""
Unit tests for ShiftFactoredPredictiveTransport orchestrator methods.

Existing smoke tests in test_prediction_and_test_time.py exercise happy paths
end-to-end. This file fills gaps: edge cases, error handling, method dispatch,
and per-method behavior that the smoke tests don't cover.
"""

from __future__ import annotations

import numpy as np
import pytest

from torchregress.prediction import PredictiveBatch
from torchregress.test_time.transport import (
    ShiftFactoredPredictiveTransport,
    ShiftFactoredTransportConfig,
)


class _DummyPredictor:
    """Simple predictor returning gaussian mean/std from linear features."""

    def predict_distribution(self, X: np.ndarray, **kwargs: object) -> PredictiveBatch:
        del kwargs
        x = np.asarray(X, dtype=float)
        mean = (
            (1.2 * x[:, 0] - 0.4 * x[:, 1]).astype(np.float32)
            if x.ndim > 1
            else np.full(x.shape[0], 0.5, dtype=np.float32)
        )
        std = np.full(x.shape[0], 0.15, dtype=np.float32)
        return PredictiveBatch(mean=mean, std=std)


def _make_predictor() -> _DummyPredictor:
    return _DummyPredictor()


def _make_source_data(n: int = 48, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 0.8 * X[:, 0] - 0.3 * X[:, 1] + 0.05 * rng.normal(size=n)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# fit_source
# ═══════════════════════════════════════════════════════════════════════════════


class TestFitSource:
    def test_returns_self(self) -> None:
        """Returns self."""
        X, y = _make_source_data()
        predictor = _make_predictor()
        batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport()
        result = transport.fit_source(batch, y, source_inputs=X)
        assert result is transport

    def test_stores_state(self) -> None:
        """Stores state."""
        X, y = _make_source_data()
        predictor = _make_predictor()
        batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=128, random_state=0)
        ).fit_source(batch, y, source_inputs=X)
        state = transport.state_
        assert state is not None
        assert state.source_support is not None
        assert state.source_prior is not None
        np.testing.assert_array_almost_equal(state.source_prior.sum(), 1.0)
        np.testing.assert_array_equal(state.source_targets, y)
        assert state.source_inputs is not None
        assert state.source_representations is None
        assert state.metadata["n_support"] == 128

    def test_no_inputs_or_representations(self) -> None:
        """fit_source with no features — alignment and uncertainty are skipped."""
        X, y = _make_source_data()
        predictor = _make_predictor()
        batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport().fit_source(batch, y)
        assert transport.state_ is not None
        assert transport.state_.source_targets.shape == y.shape
        assert transport.state_.source_prior is not None

    def test_with_representations_only(self) -> None:
        """source_representations used as features for alignment/uncertainty."""
        X, y = _make_source_data()
        reprs = np.random.default_rng(1).normal(size=(len(X), 5))
        predictor = _make_predictor()
        batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport().fit_source(
            batch, y, source_representations=reprs
        )
        assert transport.state_ is not None
        assert transport.state_.source_representations is not None

    def test_alignment_disabled(self) -> None:
        """Alignment disabled."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(enable_alignment=False)
        ).fit_source(batch, y, source_inputs=X)
        assert transport.state_ is not None
        # Check that the subspace aligner was not created
        assert transport._subspace_aligner is None

    def test_uncertainty_inflation_disabled(self) -> None:
        """Uncertainty inflation disabled."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(enable_uncertainty_inflation=False)
        ).fit_source(batch, y, source_inputs=X)
        # calibrator is only set when enable_uncertainty_inflation=True and features present
        assert transport._shift_calibrator is None

    def test_single_source_target(self) -> None:
        """Single source target still produces a valid prior (1-bin)."""
        y = np.array([0.5])
        transport = ShiftFactoredPredictiveTransport().fit_source(
            PredictiveBatch(
                mean=np.array([0.0], dtype=np.float32), std=np.array([0.1], dtype=np.float32)
            ),
            y,
        )
        assert transport.state_ is not None
        assert len(transport.state_.source_prior) >= 1
        assert np.all(np.isfinite(transport.state_.source_prior))
        np.testing.assert_array_almost_equal(transport.state_.source_prior.sum(), 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# adapt_unlabeled_target
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdaptUnlabeledTarget:
    def test_raises_before_fit_source(self) -> None:
        """Raises before fit source."""
        transport = ShiftFactoredPredictiveTransport()
        with pytest.raises(RuntimeError, match="fit_source"):
            transport.adapt_unlabeled_target(
                target_predictions=PredictiveBatch(mean=np.zeros(4, dtype=np.float32))
            )

    def test_raises_when_no_predictions_and_no_predictor(self) -> None:
        """Raises when no predictions and no predictor."""
        X, y = _make_source_data()
        transport = ShiftFactoredPredictiveTransport().fit_source(
            _make_predictor().predict_distribution(X), y
        )
        with pytest.raises(ValueError, match="target_predictions"):
            transport.adapt_unlabeled_target()

    def test_predictor_provides_predictions(self) -> None:
        """When target_predictions is None, predictor is used to compute them."""
        X, y = _make_source_data()
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport().fit_source(
            predictor.predict_distribution(X), y, source_inputs=X
        )
        target_x = np.random.default_rng(2).normal(size=(10, 3))
        adapted = transport.adapt_unlabeled_target(
            target_inputs=target_x,
            predictor=predictor,
        )
        assert adapted.mean is not None
        assert adapted.std is not None

    def test_with_target_representations_as_features(self) -> None:
        """target_representations take priority over target_inputs as features."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        source_batch = predictor.predict_distribution(X)
        source_repr = np.random.default_rng(3).normal(size=(32, 5))

        transport = ShiftFactoredPredictiveTransport().fit_source(
            source_batch, y, source_representations=source_repr
        )

        target_repr = np.random.default_rng(4).normal(size=(10, 5))
        target_batch = predictor.predict_distribution(np.random.default_rng(5).normal(size=(10, 3)))
        adapted = transport.adapt_unlabeled_target(
            target_predictions=target_batch,
            target_representations=target_repr,
        )
        assert adapted.extra is not None
        assert "transport_applied" in adapted.extra
        assert adapted.mean is not None
        assert len(np.asarray(adapted.mean)) == 10

    def test_bar_logits_batch(self) -> None:
        """adapt_unlabeled_target with bar_logits PredictiveBatch."""
        X, y = _make_source_data(48)
        predictor = _make_predictor()
        source_batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, random_state=0)
        ).fit_source(source_batch, y, source_inputs=X)

        # Create a bar_logits batch
        edges = np.linspace(-1, 1, 9, dtype=float)
        logits = np.random.default_rng(6).normal(size=(10, 8)).astype(np.float32)
        bar_batch = PredictiveBatch(bar_logits=logits, bin_edges=edges)
        adapted = transport.adapt_unlabeled_target(
            target_predictions=bar_batch,
            target_inputs=np.random.default_rng(7).normal(size=(10, 3)),
        )
        assert adapted.extra is not None
        assert adapted.extra.get("family") == "bar"

    def test_point_only_batch(self) -> None:
        """adapt_unlabeled_target with point-only PredictiveBatch (no std)."""
        X, y = _make_source_data(48)
        predictor = _make_predictor()
        source_batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, random_state=0)
        ).fit_source(source_batch, y)

        point_batch = PredictiveBatch(
            point=np.random.default_rng(8).normal(size=10).astype(np.float32),
        )
        adapted = transport.adapt_unlabeled_target(
            target_predictions=point_batch,
        )
        assert adapted.extra is not None
        assert adapted.extra.get("family") == "point"

    def test_support_density_batch(self) -> None:
        """adapt_unlabeled_target with pre-computed support+density batch."""
        X, y = _make_source_data(48)
        predictor = _make_predictor()
        source_batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, random_state=0)
        ).fit_source(source_batch, y)

        support = np.linspace(-0.5, 1.5, 64)
        density = np.random.default_rng(9).uniform(size=(10, 64)).astype(np.float32)
        density = density / density.sum(axis=1, keepdims=True)
        dens_batch = PredictiveBatch(support=support, density=density)
        adapted = transport.adapt_unlabeled_target(
            target_predictions=dens_batch,
        )
        assert adapted.extra is not None
        assert adapted.mean is not None
        assert len(np.asarray(adapted.mean)) == 10

    def test_prior_transport_max_prior_tv_skips(self) -> None:
        """When prior_transport_max_prior_tv is 0.0, any TV deviation triggers a skip."""
        X, y = _make_source_data(48, seed=10)
        predictor = _make_predictor()
        source_batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(
                n_support=64,
                random_state=0,
                prior_transport_max_prior_tv=0.0,
            )
        ).fit_source(source_batch, y, source_inputs=X)
        target_x = np.random.default_rng(11).normal(size=(24, 3))
        target_batch = predictor.predict_distribution(target_x)
        adapted = transport.adapt_unlabeled_target(
            target_predictions=target_batch,
            target_inputs=target_x,
        )
        assert adapted.extra is not None
        assert adapted.extra.get("prior_transport_skip_reason") == "high_prior_tv"

    def test_no_target_features_no_weights(self) -> None:
        """When no target features provided, local consistency weights are skipped."""
        X, y = _make_source_data(48)
        predictor = _make_predictor()
        source_batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport().fit_source(source_batch, y)
        target_x = np.random.default_rng(12).normal(size=(10, 3))
        target_batch = predictor.predict_distribution(target_x)
        adapted = transport.adapt_unlabeled_target(
            target_predictions=target_batch,
            # no target_inputs or target_representations
        )
        assert adapted.extra is not None
        assert "transport_applied" in adapted.extra
        assert adapted.mean is not None
        assert np.all(np.isfinite(np.asarray(adapted.mean)))


# ═══════════════════════════════════════════════════════════════════════════════
# calibrate_target
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalibrateTarget:
    def test_raises_before_fit_source(self) -> None:
        """Raises before fit source."""
        transport = ShiftFactoredPredictiveTransport()
        with pytest.raises(RuntimeError, match="fit_source"):
            transport.calibrate_target(
                PredictiveBatch(mean=np.zeros(4, dtype=np.float32)),
                np.zeros(4),
            )

    def test_returns_self(self) -> None:
        """Returns self."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, random_state=0)
        ).fit_source(predictor.predict_distribution(X), y)
        result = transport.calibrate_target(
            predictor.predict_distribution(X),
            y,
        )
        assert result is transport

    def test_stores_conformal_state_and_method(self) -> None:
        """Stores conformal state and method."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, alpha=0.1, random_state=0)
        ).fit_source(predictor.predict_distribution(X), y)
        transport.calibrate_target(predictor.predict_distribution(X), y)
        assert transport._conformal_state is not None
        assert "method" in transport._conformal_state
        assert "q_hat" in transport._conformal_state
        assert transport.state_ is not None
        assert transport.state_.conformal_method is not None

    def test_gaussian_auto_selects_interval(self) -> None:
        """Gaussian families select 'interval' method when gaussian_conformal_uses_native_interval=True."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, gaussian_conformal_uses_native_interval=True)
        ).fit_source(batch, y)
        transport.calibrate_target(batch, y)
        assert transport._conformal_state is not None
        assert transport._conformal_state["method"] == "interval"

    def test_explicit_method_split(self) -> None:
        """Explicit method split."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64)
        ).fit_source(batch, y)
        transport.calibrate_target(batch, y, method="split")
        assert transport._conformal_state is not None
        assert transport._conformal_state["method"] == "split"

    def test_explicit_method_invalid_raises(self) -> None:
        """Explicit method invalid raises."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64)
        ).fit_source(predictor.predict_distribution(X), y)
        with pytest.raises(ValueError, match="conformal method"):
            transport.calibrate_target(predictor.predict_distribution(X), y, method="nonexistent")

    def test_quantile_batch_auto_selects_cqr(self) -> None:
        """Quantile batch auto selects CQR."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        source_batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64)
        ).fit_source(source_batch, y)

        # Create a quantile batch
        quantiles = np.stack(
            [np.linspace(-0.5, 0.5, 32), np.linspace(-0.25, 0.75, 32), np.linspace(0.0, 1.0, 32)],
            axis=1,
        ).astype(np.float32)
        quant_batch = PredictiveBatch(
            quantiles=quantiles, quantile_levels=[0.1, 0.5, 0.9], extra={"family": "quantile"}
        )
        transport.calibrate_target(quant_batch, y)
        assert transport._conformal_state is not None
        assert transport._conformal_state["method"] == "cqr"

    def test_density_batch_auto_selects_cti(self) -> None:
        """Density batch auto selects CTI."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        source_batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64)
        ).fit_source(source_batch, y)

        support = np.linspace(-0.5, 1.5, 64)
        density = np.random.default_rng(0).uniform(size=(32, 64)).astype(np.float32)
        density = density / density.sum(axis=1, keepdims=True)
        dens_batch = PredictiveBatch(support=support, density=density, extra={"family": "density"})
        transport.calibrate_target(dens_batch, y)
        assert transport._conformal_state is not None
        assert transport._conformal_state["method"] == "cti"


# ═══════════════════════════════════════════════════════════════════════════════
# predict
# ═══════════════════════════════════════════════════════════════════════════════


class TestPredict:
    def test_calls_adapt_unlabeled_target(self) -> None:
        """Calls adapt unlabeled target."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, random_state=0)
        ).fit_source(predictor.predict_distribution(X), y, source_inputs=X)

        target_x = np.random.default_rng(0).normal(size=(16, 3))
        result = transport.predict(
            target_predictions=predictor.predict_distribution(target_x),
            apply_conformal=False,
        )
        assert result.mean is not None
        assert result.extra is not None

    def test_apply_conformal_true_adds_intervals(self) -> None:
        """Apply conformal true adds intervals."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        source_batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, alpha=0.1, random_state=0)
        ).fit_source(source_batch, y, source_inputs=X)
        transport.calibrate_target(source_batch, y)

        target_x = np.random.default_rng(1).normal(size=(16, 3))
        result = transport.predict(
            target_inputs=target_x,
            predictor=predictor,
            apply_conformal=True,
        )
        assert result.extra is not None
        assert "interval_lower" in result.extra
        assert "interval_upper" in result.extra

    def test_predict_without_calibrate_is_noop(self) -> None:
        """Predict without calibrate is noop."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, random_state=0)
        ).fit_source(predictor.predict_distribution(X), y, source_inputs=X)

        target_x = np.random.default_rng(2).normal(size=(16, 3))
        # apply_conformal defaults to True, but without calibrate_target, it's a no-op
        result = transport.predict(
            target_inputs=target_x,
            predictor=predictor,
        )
        assert result.mean is not None
        assert "interval_lower" not in (result.extra or {})


# ═══════════════════════════════════════════════════════════════════════════════
# apply_conformal
# ═══════════════════════════════════════════════════════════════════════════════


class TestApplyConformal:
    def test_no_conformal_state_returns_unchanged(self) -> None:
        """No conformal state returns unchanged."""
        transport = ShiftFactoredPredictiveTransport()
        batch = PredictiveBatch(mean=np.array([0.0, 1.0], dtype=np.float32))
        result = transport.apply_conformal(batch)
        assert result is batch  # returns the same object

    def test_split_method_adds_intervals(self) -> None:
        """Split method adds intervals."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, alpha=0.1)
        ).fit_source(predictor.predict_distribution(X), y)
        transport.calibrate_target(predictor.predict_distribution(X), y, method="split")

        batch = predictor.predict_distribution(X)
        result = transport.apply_conformal(batch)
        assert result.extra is not None
        assert "interval_lower" in result.extra
        assert "interval_upper" in result.extra
        assert result.extra["conformal_method"] == "split"

    def test_interval_method_adds_predictions(self) -> None:
        """Interval method adds predictions."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, alpha=0.1)
        ).fit_source(predictor.predict_distribution(X), y)
        transport.calibrate_target(predictor.predict_distribution(X), y, method="interval")

        batch = predictor.predict_distribution(X)
        result = transport.apply_conformal(batch)
        assert result.extra is not None
        assert result.extra["conformal_method"] == "interval"
        lower = np.asarray(result.extra["interval_lower"], dtype=float)
        upper = np.asarray(result.extra["interval_upper"], dtype=float)
        assert np.all(upper >= lower)

    def test_cqr_method_on_quantile_batch(self) -> None:
        """CQR method on quantile batch."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        source_batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, alpha=0.1)
        ).fit_source(source_batch, y)

        quantiles = np.stack(
            [np.linspace(-0.5, 0.5, 32), np.linspace(-0.25, 0.75, 32), np.linspace(0.0, 1.0, 32)],
            axis=1,
        ).astype(np.float32)
        quant_batch = PredictiveBatch(
            quantiles=quantiles, quantile_levels=[0.1, 0.5, 0.9], extra={"family": "quantile"}
        )
        transport.calibrate_target(quant_batch, y, method="cqr")

        result = transport.apply_conformal(quant_batch)
        assert result.extra is not None
        assert result.extra["conformal_method"] == "cqr"
        lower = np.asarray(result.extra["interval_lower"], dtype=float)
        upper = np.asarray(result.extra["interval_upper"], dtype=float)
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))

    def test_cti_method_on_density_batch(self) -> None:
        """CTI method on density batch."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        source_batch = predictor.predict_distribution(X)
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, alpha=0.1)
        ).fit_source(source_batch, y)

        support = np.linspace(-0.5, 1.5, 64, dtype=float)
        density = np.random.default_rng(0).uniform(size=(32, 64)).astype(np.float32)
        density = density / density.sum(axis=1, keepdims=True)
        dens_batch = PredictiveBatch(support=support, density=density, extra={"family": "density"})
        transport.calibrate_target(dens_batch, y, method="cti")

        result = transport.apply_conformal(dens_batch)
        assert result.extra is not None
        assert result.extra["conformal_method"] == "cti"
        lower = np.asarray(result.extra["interval_lower"], dtype=float)
        upper = np.asarray(result.extra["interval_upper"], dtype=float)
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))
        assert np.all(upper >= lower)

    def test_invalid_method_raises(self) -> None:
        """Invalid method raises."""
        transport = ShiftFactoredPredictiveTransport()
        transport._conformal_state = {"method": "bogus", "q_hat": 1.0}
        batch = PredictiveBatch(
            mean=np.array([0.0], dtype=np.float32), std=np.array([0.1], dtype=np.float32)
        )
        with pytest.raises(ValueError, match="conformal method"):
            transport.apply_conformal(batch)

    def test_extra_stores_conformal_metadata(self) -> None:
        """Extra stores conformal metadata."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, alpha=0.1)
        ).fit_source(predictor.predict_distribution(X), y)
        transport.calibrate_target(predictor.predict_distribution(X), y, method="split")

        batch = predictor.predict_distribution(X)
        result = transport.apply_conformal(batch)
        assert result.extra is not None
        assert result.extra["conformal_method"] == "split"
        assert isinstance(result.extra["conformal_q_hat"], float)
        assert result.extra["conformal_q_hat"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# ppi_target_ci
# ═══════════════════════════════════════════════════════════════════════════════


class TestPPITargetCI:
    def test_mean_estimand(self) -> None:
        """Mean estimand."""
        X, y = _make_source_data(32, seed=20)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, random_state=0)
        ).fit_source(predictor.predict_distribution(X), y, source_inputs=X)

        labeled = transport.adapt_unlabeled_target(
            target_predictions=predictor.predict_distribution(X),
            target_inputs=X,
        )
        target_x = np.random.default_rng(21).normal(size=(24, 3))
        unlabeled = transport.adapt_unlabeled_target(
            target_inputs=target_x,
            predictor=predictor,
        )

        result = transport.ppi_target_ci(
            "mean",
            y,
            labeled,
            unlabeled,
            alpha=0.1,
            n_boot=200,
            seed=0,
        )
        assert result["method"] == "ppi_mean_ci"
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_quantile_estimand_with_q(self) -> None:
        """Quantile estimand with q."""
        X, y = _make_source_data(32, seed=22)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, random_state=0)
        ).fit_source(predictor.predict_distribution(X), y, source_inputs=X)

        labeled = transport.adapt_unlabeled_target(
            target_predictions=predictor.predict_distribution(X),
            target_inputs=X,
        )
        target_x = np.random.default_rng(23).normal(size=(24, 3))
        unlabeled = transport.adapt_unlabeled_target(
            target_inputs=target_x,
            predictor=predictor,
        )

        result = transport.ppi_target_ci(
            "quantile",
            y,
            labeled,
            unlabeled,
            q=0.5,
            alpha=0.1,
            n_boot=200,
            seed=0,
        )
        assert result["method"] == "ppi_quantile_ci"
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_quantile_estimand_without_q_raises(self) -> None:
        """Quantile estimand without q raises."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, random_state=0)
        ).fit_source(predictor.predict_distribution(X), y, source_inputs=X)

        batch = transport.adapt_unlabeled_target(
            target_predictions=predictor.predict_distribution(X),
            target_inputs=X,
        )
        with pytest.raises(ValueError, match="q is required"):
            transport.ppi_target_ci(
                "quantile",
                y,
                batch,
                batch,
                alpha=0.1,
                n_boot=200,
                seed=0,
            )

    def test_ols_estimand_with_x(self) -> None:
        """OLS estimand with x."""
        X, y = _make_source_data(32, seed=24)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, random_state=0)
        ).fit_source(predictor.predict_distribution(X), y, source_inputs=X)

        labeled = transport.adapt_unlabeled_target(
            target_predictions=predictor.predict_distribution(X),
            target_inputs=X,
        )
        target_x = np.random.default_rng(25).normal(size=(24, 3))
        unlabeled = transport.adapt_unlabeled_target(
            target_inputs=target_x,
            predictor=predictor,
        )

        result = transport.ppi_target_ci(
            "ols",
            y,
            labeled,
            unlabeled,
            x_labeled=X,
            x_unlabeled=target_x,
            alpha=0.1,
            n_boot=2000,
            seed=0,
        )
        assert result["method"] == "ppi_ols_ci"
        assert "ci_lower" in result
        assert "ci_upper" in result

    def test_ols_estimand_without_x_raises(self) -> None:
        """OLS estimand without x raises."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, random_state=0)
        ).fit_source(predictor.predict_distribution(X), y, source_inputs=X)

        batch = transport.adapt_unlabeled_target(
            target_predictions=predictor.predict_distribution(X),
            target_inputs=X,
        )
        with pytest.raises(ValueError, match="x_labeled"):
            transport.ppi_target_ci(
                "ols",
                y,
                batch,
                batch,
                alpha=0.1,
                n_boot=200,
                seed=0,
            )

    def test_invalid_estimand_raises(self) -> None:
        """Invalid estimand raises."""
        X, y = _make_source_data(32)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, random_state=0)
        ).fit_source(predictor.predict_distribution(X), y, source_inputs=X)

        batch = transport.adapt_unlabeled_target(
            target_predictions=predictor.predict_distribution(X),
            target_inputs=X,
        )
        with pytest.raises(ValueError, match="estimand"):
            transport.ppi_target_ci(
                "invalid_estimand",
                y,
                batch,
                batch,
                alpha=0.1,
                n_boot=200,
                seed=0,
            )

    def test_accepts_numpy_arrays_as_predictions(self) -> None:
        """ppi_target_ci accepts numpy arrays directly (not just PredictiveBatch)."""
        X, y = _make_source_data(32, seed=26)
        predictor = _make_predictor()
        transport = ShiftFactoredPredictiveTransport(
            ShiftFactoredTransportConfig(n_support=64, random_state=0)
        ).fit_source(predictor.predict_distribution(X), y, source_inputs=X)

        # labeled and unlabeled must have matching sample counts with labeled_targets
        result = transport.ppi_target_ci(
            "mean",
            y,
            y,  # labeled predictions (same size as labeled_targets)
            y,  # unlabeled predictions
            alpha=0.1,
            n_boot=200,
            seed=0,
        )
        assert result["method"] == "ppi_mean_ci"
