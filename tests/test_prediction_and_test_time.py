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
