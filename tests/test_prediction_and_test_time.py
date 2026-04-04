from __future__ import annotations

import numpy as np
import torch

from torchregress.prediction import PredictiveBatch, bars_to_density_grid, quantiles_to_density_grid
from torchregress.test_time import (
    FeatureStatNormalizer,
    ParameterEMA,
    PosteriorLabelShiftAdapter,
    RepresentationShiftCalibrator,
    SignificantSubspaceAligner,
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

    batch = PredictiveBatch(quantiles=quantiles, quantile_levels=[0.1, 0.5, 0.9]).with_density(n_support=64)
    assert batch.support is not None
    assert batch.density is not None


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
    corrected, estimate = adapter.fit_transform(probs)
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
    corrected, estimate = adapter.fit_transform(probs, sample_weights=weights)
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
        n_bins=8,
        estimation_rows=3,
        top_fraction=0.5,
        reference_size=2,
        seed=0,
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
    weights = local_consistency_weights(features, probs, k=1)
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
        k=3,
        reference_size=16,
        random_state=0,
    )
    assert weights.shape == (128,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)


def test_local_consistency_weights_chunked_matches_unchunked() -> None:
    rng = np.random.default_rng(2)
    features = rng.normal(size=(64, 5))
    probs = rng.uniform(size=(64, 4))
    probs = probs / probs.sum(axis=1, keepdims=True)
    full = local_consistency_weights(features, probs, k=3, query_chunk_size=None)
    chunked = local_consistency_weights(features, probs, k=3, query_chunk_size=11)
    assert np.allclose(full, chunked, atol=1.0e-10)


def test_significant_subspace_and_feature_stat_aligners_transform_shapes() -> None:
    rng = np.random.default_rng(0)
    X_source = rng.normal(size=(32, 4))
    y_source = 2.0 * X_source[:, 0] - 0.5 * X_source[:, 1]
    X_target = 1.5 * X_source + np.array([0.3, -0.2, 0.1, 0.0])

    ssa = SignificantSubspaceAligner(rank=2).fit(X_source, y_source)
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

    ssa = SignificantSubspaceAligner(
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
    calibrator = RepresentationShiftCalibrator(slope=2.0, max_temperature=4.0).fit(source)
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
    calibrator = RepresentationShiftCalibrator(
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
