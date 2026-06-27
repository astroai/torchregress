from __future__ import annotations

import numpy as np
import pytest
import torch

import torchregress.test_time.transport as transport_mod
from torchregress.prediction import (
    PredictiveBatch,
    bars_to_density_grid,
    quantiles_to_density_grid,
    samples_to_density_grid,
)
from torchregress.test_time import (
    FeatureStatNormalizer,
    GaussianLabelShiftConfig,
    LocalConsistencyConfig,
    ParameterEMA,
    PosteriorLabelShiftAdapter,
    RepresentationShiftInflator,
    ShiftFactoredPredictiveTransport,
    ShiftFactoredTransportConfig,
    WeightedSubspaceMomentAligner,
    confidence_scores,
    correct_gaussian_predictions_for_label_shift,
    entropy_scores,
    local_consistency_weights,
    select_high_confidence,
)


def test_predictive_batch_density_helpers_normalize() -> None:
    quantiles = np.array([[0.1, 0.4, 0.8], [0.2, 0.5, 0.9]], dtype=float)
    support_q, density_q = quantiles_to_density_grid(quantiles, [0.1, 0.5, 0.9], n_support=64)
    assert support_q.shape == density_q.shape == (2, 64)
    assert np.allclose(np.trapezoid(density_q, support_q, axis=1), 1.0, atol=1.0e-3)

    logits = np.array([[1.0, 0.5, -0.2]], dtype=float)
    edges = np.array([0.0, 0.5, 1.0, 1.5], dtype=float)
    support_b, density_b = bars_to_density_grid(logits, edges, n_support=64)
    assert support_b.shape == density_b.shape == (1, 64)
    assert np.isclose(np.trapezoid(density_b[0], support_b[0]), 1.0, atol=1.0e-3)

    batch = PredictiveBatch(quantiles=quantiles, quantile_levels=[0.1, 0.5, 0.9]).with_density(
        n_support=64
    )
    assert batch.support is not None
    assert batch.density is not None

    rng = np.random.default_rng(0)
    samples = rng.normal(loc=np.array([[0.0], [1.0]]), scale=0.15, size=(2, 256))
    support_s, density_s = samples_to_density_grid(samples, n_support=64)
    assert support_s.shape == density_s.shape == (2, 64)
    assert np.allclose(np.trapezoid(density_s, support_s, axis=1), 1.0, atol=1.0e-3)

    sample_batch = PredictiveBatch(samples=samples.astype(np.float32), extra={"family": "mdn"})
    dense_sample_batch = sample_batch.with_density(n_support=64)
    assert dense_sample_batch.support is not None
    assert dense_sample_batch.density is not None


def test_shift_factored_transport_config_rejects_invalid_method_parameters() -> None:
    invalid_kwargs = [
        {"n_support": 8},
        {"support_margin": -0.1},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"top_fraction": 0.0},
        {"top_fraction": 1.1},
        {"min_selection_count": 0},
        {"local_consistency_k": 0},
        {"prior_estimation_rows": 0},
        {"prior_transport_strength": -0.1},
        {"prior_transport_strength": 1.1},
        {"prior_ratio_clip": 0.9},
        {"prior_transport_min_selected_fraction": -0.1},
        {"prior_transport_max_prior_tv": 1.1},
        {"uncertainty_base_temperature": 0.0},
        {"uncertainty_slope": -0.1},
        {"uncertainty_base_temperature": 2.0, "uncertainty_max_temperature": 1.0},
        {"uncertainty_clip_quantile": 0.5},
        {"eps": 0.0},
    ]

    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            ShiftFactoredTransportConfig(**kwargs)


def test_label_shift_adapter_estimates_and_corrects_target_prior() -> None:
    probs = np.array(
        [
            [0.85, 0.15],
            [0.80, 0.20],
            [0.30, 0.70],
            [0.25, 0.75],
        ],
        dtype=float,
    )
    adapter = PosteriorLabelShiftAdapter(source_prior=np.array([0.5, 0.5]))
    estimate = adapter.estimate(probs)
    corrected = adapter.transform(probs)
    assert estimate.target_prior.shape == (2,)
    assert np.isclose(estimate.target_prior.sum(), 1.0)
    assert corrected.shape == probs.shape
    assert np.allclose(corrected.sum(axis=1), 1.0)


def test_label_shift_adapter_supports_weighted_subsampled_estimation() -> None:
    probs = np.array(
        [
            [0.90, 0.10],
            [0.85, 0.15],
            [0.20, 0.80],
            [0.25, 0.75],
            [0.88, 0.12],
            [0.18, 0.82],
        ],
        dtype=float,
    )
    weights = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 2.0], dtype=float)
    adapter = PosteriorLabelShiftAdapter(
        source_prior=np.array([0.5, 0.5]),
        sample_size=4,
        random_state=0,
    )
    estimate = adapter.estimate(probs, sample_weights=weights)
    corrected = adapter.transform(probs)
    assert estimate.target_prior.shape == (2,)
    assert np.isclose(estimate.target_prior.sum(), 1.0)
    assert corrected.shape == probs.shape
    assert np.allclose(corrected.sum(axis=1), 1.0)


def test_gaussian_label_shift_correction_returns_finite_moments() -> None:
    mean = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    std = np.array([0.05, 0.08, 0.07, 0.06], dtype=float)
    source_targets = np.array([0.05, 0.15, 0.2, 0.75, 0.85, 0.95], dtype=float)
    features = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=float)
    corrected_mean, corrected_std, meta = correct_gaussian_predictions_for_label_shift(
        mean=mean,
        std=std,
        source_targets=source_targets,
        features=features,
        config=GaussianLabelShiftConfig(
            n_bins=8,
            estimation_rows=3,
            top_fraction=0.5,
            reference_size=2,
            seed=0,
        ),
    )
    assert corrected_mean.shape == mean.shape
    assert corrected_std.shape == std.shape
    assert np.all(np.isfinite(corrected_mean))
    assert np.all(np.isfinite(corrected_std))
    assert meta["estimate_converged"] in (True, False)


def test_confidence_and_local_consistency_utilities_are_finite() -> None:
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.55, 0.45], [0.2, 0.8]], dtype=float)
    features = np.array([[0.0, 0.0], [0.05, 0.0], [1.0, 1.0], [1.1, 1.0]], dtype=float)
    mask = select_high_confidence(probs, top_fraction=0.5, min_count=2)
    assert mask.sum() == 2
    assert np.all(confidence_scores(probs) <= 1.0)
    assert np.all(entropy_scores(probs) >= 0.0)
    weights = local_consistency_weights(features, probs, config=LocalConsistencyConfig(k=1))
    assert weights.shape == (4,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)


def test_local_consistency_weights_supports_approximate_reference_subset() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(128, 4))
    probs = rng.uniform(size=(128, 3))
    probs = probs / probs.sum(axis=1, keepdims=True)
    weights = local_consistency_weights(
        features,
        probs,
        config=LocalConsistencyConfig(
            k=3,
            reference_size=16,
            random_state=0,
        ),
    )
    assert weights.shape == (128,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)


def test_local_consistency_weights_chunked_matches_unchunked() -> None:
    rng = np.random.default_rng(2)
    features = rng.normal(size=(64, 5))
    probs = rng.uniform(size=(64, 4))
    probs = probs / probs.sum(axis=1, keepdims=True)
    full = local_consistency_weights(
        features, probs, config=LocalConsistencyConfig(k=3, query_chunk_size=None)
    )
    chunked = local_consistency_weights(
        features, probs, config=LocalConsistencyConfig(k=3, query_chunk_size=11)
    )
    assert np.allclose(full, chunked, atol=1.0e-10)


def test_significant_subspace_and_feature_stat_aligners_transform_shapes() -> None:
    rng = np.random.default_rng(0)
    X_source = rng.normal(size=(32, 4))
    y_source = 2.0 * X_source[:, 0] - 0.5 * X_source[:, 1]
    X_target = 1.5 * X_source + np.array([0.3, -0.2, 0.1, 0.0])

    ssa = WeightedSubspaceMomentAligner(rank=2).fit(X_source, y_source)
    X_aligned = ssa.transform(X_target)
    assert X_aligned.shape == X_target.shape
    assert np.all(np.isfinite(X_aligned))

    normalizer = FeatureStatNormalizer().fit(X_source)
    X_norm = normalizer.transform(X_target)
    assert X_norm.shape == X_target.shape
    assert np.all(np.isfinite(X_norm))


def test_aligners_support_subsampled_robust_target_stats() -> None:
    rng = np.random.default_rng(1)
    X_source = rng.normal(size=(64, 6))
    y_source = X_source[:, 0] - 0.3 * X_source[:, 1]
    X_target = 1.3 * X_source + 0.2
    X_target[:4, 0] += 50.0

    ssa = WeightedSubspaceMomentAligner(
        rank=3,
        target_sample_size=16,
        random_state=0,
        clip_quantile=0.05,
    ).fit(X_source, y_source)
    X_aligned = ssa.transform(X_target)
    assert X_aligned.shape == X_target.shape
    assert np.all(np.isfinite(X_aligned))

    normalizer = FeatureStatNormalizer(
        target_sample_size=16,
        random_state=0,
        clip_quantile=0.05,
    ).fit(X_source)
    X_norm = normalizer.transform(X_target)
    assert X_norm.shape == X_target.shape
    assert np.all(np.isfinite(X_norm))


def test_feature_stat_normalizer_clips_extreme_scale_ratios() -> None:
    X_source = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float)
    X_target = np.array([[100.0, 1.0], [100.0, 1.1], [100.0, 0.9]], dtype=float)
    normalizer = FeatureStatNormalizer(max_scale_ratio=2.0).fit(X_source)
    transformed = normalizer.transform(X_target)
    assert transformed.shape == X_target.shape
    assert np.all(np.isfinite(transformed))
    assert np.max(np.abs(transformed[:, 0])) < 500.0


def test_parameter_ema_tracks_parameter_updates() -> None:
    model = torch.nn.Linear(3, 2)
    ema = ParameterEMA(decay=0.5)
    ema.initialize(model)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
    ema.update(model)
    assert ema.shadow
    for name, value in ema.shadow.items():
        assert not torch.equal(value, before[name])


def test_representation_shift_calibrator_scales_probabilities_and_std() -> None:
    source = np.array([[0.0, 0.0], [0.1, -0.1], [-0.1, 0.05]], dtype=float)
    target = np.array([[0.05, 0.02], [2.0, 2.0]], dtype=float)
    probs = np.array([[0.8, 0.2], [0.8, 0.2]], dtype=float)
    std = np.array([0.1, 0.1], dtype=float)
    calibrator = RepresentationShiftInflator(slope=2.0, max_temperature=4.0).fit(source)
    calibrated_probs = calibrator.calibrate_probabilities(probs, target)
    calibrated_std = calibrator.calibrate_std(std, target)
    assert calibrated_probs.shape == probs.shape
    assert np.allclose(calibrated_probs.sum(axis=1), 1.0)
    assert calibrated_std[1] >= calibrated_std[0]


def test_representation_shift_calibrator_supports_subsampled_robust_fit() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(size=(128, 8))
    source[:4, 0] += 100.0
    target = rng.normal(loc=0.5, scale=1.5, size=(32, 8))
    std = np.full(target.shape[0], 0.2, dtype=float)
    calibrator = RepresentationShiftInflator(
        slope=1.5,
        max_temperature=3.0,
        source_sample_size=32,
        random_state=0,
        clip_quantile=0.05,
    ).fit(source)
    calibrated_std = calibrator.calibrate_std(std, target)
    assert calibrated_std.shape == std.shape
    assert np.all(np.isfinite(calibrated_std))
    assert np.all(calibrated_std >= std)


class _DummyPredictor:
    def predict_distribution(self, X: np.ndarray, **kwargs: object) -> PredictiveBatch:
        del kwargs
        x = np.asarray(X, dtype=float)
        mean = (1.2 * x[:, 0] - 0.4 * x[:, 1]).astype(np.float32)
        std = np.full(x.shape[0], 0.15, dtype=np.float32)
        return PredictiveBatch(mean=mean, std=std)


def _slice_predictive_batch(batch: PredictiveBatch, start: int, stop: int) -> PredictiveBatch:
    batch_size = None
    for value in (
        batch.mean,
        batch.point,
        batch.std,
        batch.quantiles,
        batch.bar_logits,
        batch.density,
        batch.samples,
    ):
        if value is not None:
            batch_size = int(np.asarray(value).shape[0])
            break

    def _slice(value: object) -> object:
        if value is None:
            return None
        arr = np.asarray(value)
        if batch_size is not None and arr.ndim >= 1 and arr.shape[0] == batch_size:
            return arr[start:stop]
        return value

    return PredictiveBatch(
        point=_slice(batch.point),
        mean=_slice(batch.mean),
        std=_slice(batch.std),
        quantiles=_slice(batch.quantiles),
        quantile_levels=batch.quantile_levels,
        bar_logits=_slice(batch.bar_logits),
        bin_edges=_slice(batch.bin_edges),
        samples=_slice(batch.samples),
        support=_slice(batch.support),
        density=_slice(batch.density),
        extra=dict(batch.extra or {}),
    )


def test_shift_factored_transport_adapts_gaussian_predictions() -> None:
    rng = np.random.default_rng(7)
    source_x = rng.normal(size=(48, 3))
    source_y = 0.8 * source_x[:, 0] - 0.3 * source_x[:, 1] + 0.05 * rng.normal(size=48)
    target_x = source_x + np.array([0.4, -0.2, 0.1])
    predictor = _DummyPredictor()

    source_batch = predictor.predict_distribution(source_x)
    target_batch = predictor.predict_distribution(target_x)

    transport = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(n_support=96, random_state=0)
    ).fit_source(
        source_batch,
        source_y,
        source_inputs=source_x,
    )
    adapted = transport.adapt_unlabeled_target(
        target_predictions=target_batch,
        target_inputs=target_x,
        predictor=predictor,
    )

    assert adapted.support is None
    assert adapted.density is None
    assert adapted.mean is not None
    assert adapted.std is not None
    assert np.all(np.isfinite(np.asarray(adapted.mean)))
    assert np.all(np.isfinite(np.asarray(adapted.std)))
    assert adapted.extra is not None
    assert "target_prior" in adapted.extra
    assert "target_prior_raw" in adapted.extra
    assert "prior_shrink_weight" in adapted.extra
    assert adapted.extra["alignment_applied"] is False


def test_shift_factored_transport_supports_opt_in_input_alignment_rerun() -> None:
    rng = np.random.default_rng(17)
    source_x = rng.normal(size=(48, 3))
    source_y = 0.8 * source_x[:, 0] - 0.3 * source_x[:, 1] + 0.05 * rng.normal(size=48)
    target_x = source_x + np.array([0.4, -0.2, 0.1])
    predictor = _DummyPredictor()

    source_batch = predictor.predict_distribution(source_x)
    target_batch = predictor.predict_distribution(target_x)

    transport = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(
            n_support=96,
            random_state=0,
            allow_input_alignment_rerun=True,
        )
    ).fit_source(
        source_batch,
        source_y,
        source_inputs=source_x,
    )
    adapted = transport.adapt_unlabeled_target(
        target_predictions=target_batch,
        target_inputs=target_x,
        predictor=predictor,
    )

    assert adapted.extra is not None
    assert adapted.extra["alignment_applied"] is True


def test_shift_factored_transport_can_disable_prior_shift_update() -> None:
    rng = np.random.default_rng(19)
    source_x = rng.normal(size=(48, 3))
    source_y = 0.8 * source_x[:, 0] - 0.3 * source_x[:, 1] + 0.05 * rng.normal(size=48)
    target_x = source_x + np.array([0.4, -0.2, 0.1])
    predictor = _DummyPredictor()

    source_batch = predictor.predict_distribution(source_x)
    target_batch = predictor.predict_distribution(target_x)
    transport = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(
            n_support=96,
            random_state=0,
            prior_transport_strength=0.0,
            enable_alignment=False,
            enable_uncertainty_inflation=False,
        )
    ).fit_source(
        source_batch,
        source_y,
        source_inputs=source_x,
    )
    adapted = transport.adapt_unlabeled_target(
        target_predictions=target_batch,
        target_inputs=target_x,
    )

    assert adapted.extra is not None
    assert np.allclose(adapted.extra["target_prior"], adapted.extra["source_prior"])
    assert float(adapted.extra["prior_shrink_weight"]) == 0.0
    assert adapted.extra["transport_applied"] is False
    assert np.allclose(np.asarray(adapted.mean), np.asarray(target_batch.mean))
    assert np.allclose(np.asarray(adapted.std), np.asarray(target_batch.std))


def test_shift_factored_transport_skips_nonconverged_prior_update_by_default() -> None:
    source_prior = np.array([0.5, 0.5], dtype=float)
    target_prior = np.array([0.1, 0.9], dtype=float)
    probs = np.array([[0.95, 0.05], [0.92, 0.08]], dtype=float)
    stabilized, meta = transport_mod._stabilize_target_prior(
        source_prior=source_prior,
        target_prior=target_prior,
        selected_probabilities=probs,
        converged=False,
        config=ShiftFactoredTransportConfig(),
    )
    assert np.allclose(stabilized, source_prior)
    assert float(meta["prior_shrink_weight"]) == 0.0
    assert float(meta["prior_transport_skipped"]) == 1.0
    assert float(meta["prior_evidence_scale"]) == 1.0


def test_stabilize_target_prior_respects_evidence_scale() -> None:
    source_prior = np.array([0.5, 0.5], dtype=float)
    target_prior = np.array([0.2, 0.8], dtype=float)
    probs = np.array([[0.6, 0.4], [0.55, 0.45]], dtype=float)
    stabilized, meta = transport_mod._stabilize_target_prior(
        source_prior=source_prior,
        target_prior=target_prior,
        selected_probabilities=probs,
        converged=True,
        config=ShiftFactoredTransportConfig(prior_transport_strength=1.0),
        evidence_scale=0.0,
    )
    assert np.allclose(stabilized, source_prior)
    assert float(meta["prior_shrink_weight"]) == 0.0
    assert float(meta["prior_evidence_scale"]) == 0.0


def test_shift_factored_transport_skips_prior_when_selection_fraction_too_low() -> None:
    rng = np.random.default_rng(31)
    source_x = rng.normal(size=(48, 3))
    source_y = 0.8 * source_x[:, 0] - 0.3 * source_x[:, 1] + 0.05 * rng.normal(size=48)
    target_x = source_x + np.array([0.4, -0.2, 0.1])
    predictor = _DummyPredictor()

    source_batch = predictor.predict_distribution(source_x)
    target_batch = predictor.predict_distribution(target_x)

    transport = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(
            n_support=96,
            random_state=0,
            prior_transport_min_selected_fraction=1.01,
        )
    ).fit_source(
        source_batch,
        source_y,
        source_inputs=source_x,
    )
    adapted = transport.adapt_unlabeled_target(
        target_predictions=target_batch,
        target_inputs=target_x,
        predictor=predictor,
    )
    assert adapted.extra is not None
    assert adapted.extra.get("prior_transport_skip_reason") == "low_selection_fraction"
    assert float(adapted.extra["prior_evidence_scale"]) == 0.0
    assert float(adapted.extra["prior_shrink_weight"]) == 0.0


def test_shift_factored_transport_preserves_quantile_family_outputs() -> None:
    rng = np.random.default_rng(11)
    source_y = rng.normal(size=40)
    quantiles = np.stack(
        [
            np.linspace(-0.5, 0.5, 24),
            np.linspace(-0.25, 0.75, 24),
            np.linspace(0.0, 1.0, 24),
        ],
        axis=1,
    )
    target_features = rng.normal(size=(24, 2))

    batch = PredictiveBatch(
        quantiles=quantiles.astype(np.float32),
        quantile_levels=[0.1, 0.5, 0.9],
    )
    transport = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(n_support=80, random_state=0)
    ).fit_source(
        batch,
        source_y,
        source_representations=rng.normal(size=(40, 2)),
    )
    adapted = transport.adapt_unlabeled_target(
        target_predictions=batch,
        target_representations=target_features,
    )

    assert adapted.quantiles is not None
    assert adapted.quantile_levels == [0.1, 0.5, 0.9]
    assert adapted.extra is not None
    assert adapted.extra["family"] == "quantile"


def test_shift_factored_transport_supports_sampled_predictive_batches() -> None:
    rng = np.random.default_rng(29)
    source_x = rng.normal(size=(32, 2))
    source_y = 0.8 * source_x[:, 0] - 0.3 * source_x[:, 1] + rng.normal(scale=0.2, size=32)
    target_x = rng.normal(loc=0.2, scale=1.1, size=(20, 2))
    target_mean = 0.7 * target_x[:, 0] - 0.2 * target_x[:, 1]
    target_samples = rng.normal(loc=target_mean[:, None], scale=0.3, size=(20, 128))

    transport = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(n_support=96, alpha=0.1, random_state=0)
    ).fit_source(PredictiveBatch(mean=source_y, std=np.full_like(source_y, 0.2)), source_y)

    adapted = transport.adapt_unlabeled_target(
        target_predictions=PredictiveBatch(
            samples=target_samples.astype(np.float32),
            extra={"family": "mdn"},
        ),
        target_inputs=target_x,
    )
    assert adapted.support is not None
    assert adapted.density is not None
    assert adapted.extra is not None
    assert adapted.extra["family"] == "mdn"

    cal_targets = target_mean[:10] + rng.normal(scale=0.2, size=10)
    transport.calibrate_target(_slice_predictive_batch(adapted, 0, 10), cal_targets)
    conformed = transport.apply_conformal(_slice_predictive_batch(adapted, 10, 20))
    assert conformed.support is not None
    assert conformed.density is not None


def test_shift_factored_transport_supports_conformal_and_ppi() -> None:
    rng = np.random.default_rng(13)
    source_x = rng.normal(size=(36, 2))
    source_y = 0.7 * source_x[:, 0] + 0.2 * rng.normal(size=36)
    predictor = _DummyPredictor()
    source_batch = predictor.predict_distribution(source_x)

    transport = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(alpha=0.1, n_support=96, random_state=0)
    ).fit_source(source_batch, source_y, source_inputs=source_x)

    cal_x = rng.normal(size=(18, 2))
    cal_batch = predictor.predict_distribution(cal_x)
    cal_y = 0.7 * cal_x[:, 0] + 0.2 * rng.normal(size=18)
    transport.calibrate_target(cal_batch, cal_y)

    pred_batch = transport.predict(target_inputs=cal_x, predictor=predictor)
    assert pred_batch.extra is not None
    assert "interval_lower" in pred_batch.extra
    assert "interval_upper" in pred_batch.extra

    labeled = pred_batch
    unlabeled = transport.adapt_unlabeled_target(
        target_inputs=rng.normal(size=(32, 2)), predictor=predictor
    )
    mean_ci = transport.ppi_target_ci(
        "mean",
        cal_y,
        labeled,
        unlabeled,
        alpha=0.1,
        n_boot=200,
        seed=0,
    )
    quantile_ci = transport.ppi_target_ci(
        "quantile",
        cal_y,
        labeled,
        unlabeled,
        q=0.9,
        alpha=0.1,
        n_boot=200,
        seed=0,
    )
    assert mean_ci["method"] == "ppi_mean_ci"
    assert quantile_ci["method"] == "ppi_quantile_ci"


def test_shift_factored_transport_preserves_gaussian_family_for_conformal() -> None:
    rng = np.random.default_rng(23)
    source_x = rng.normal(size=(36, 2))
    source_y = 0.7 * source_x[:, 0] + 0.2 * rng.normal(size=36)
    target_x = rng.normal(size=(18, 2))
    target_y = 0.7 * target_x[:, 0] + 0.2 * rng.normal(size=18)
    predictor = _DummyPredictor()

    transport = ShiftFactoredPredictiveTransport(
        ShiftFactoredTransportConfig(alpha=0.1, n_support=96, random_state=0)
    ).fit_source(
        predictor.predict_distribution(source_x),
        source_y,
        source_inputs=source_x,
    )
    adapted = transport.adapt_unlabeled_target(
        target_predictions=predictor.predict_distribution(target_x),
        target_inputs=target_x,
        predictor=predictor,
    )
    assert adapted.extra is not None
    assert adapted.extra["family"] == "gaussian"

    transport.calibrate_target(adapted, target_y)
    assert transport.state_ is not None
    assert transport.state_.conformal_method == "interval"

    native_lower, native_upper = transport_mod._native_interval(
        adapted,
        alpha=transport.config.alpha,
        eps=transport.config.eps,
        family_hint="gaussian",
    )
    conformed = transport.apply_conformal(adapted)
    assert conformed.extra is not None
    lower = np.asarray(conformed.extra["interval_lower"], dtype=float)
    upper = np.asarray(conformed.extra["interval_upper"], dtype=float)
    assert np.all(upper - lower >= native_upper - native_lower - 1.0e-8)
