import numpy as np
import pytest

from torchregress.test_time.label_shift import (
    GaussianLabelShiftConfig,
    LabelShiftEMConfig,
    PosteriorLabelShiftAdapter,
    apply_label_shift_correction,
    correct_gaussian_predictions_for_label_shift,
    estimate_target_prior_em,
    gaussian_bin_edges_from_targets,
    gaussian_bin_probabilities,
    gaussian_moments_from_binned_probabilities,
)


def test_apply_label_shift_correction_valid():
    probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
    src_prior = np.array([0.5, 0.5])
    tgt_prior = np.array([0.2, 0.8])

    corrected = apply_label_shift_correction(probs, source_prior=src_prior, target_prior=tgt_prior)

    # Check shape
    assert corrected.shape == probs.shape

    # Check probabilities sum to 1
    np.testing.assert_allclose(corrected.sum(axis=1), np.ones(probs.shape[0]))

    # Manually compute expected for first row
    # p1 = 0.8 * (0.2/0.5) = 0.8 * 0.4 = 0.32
    # p2 = 0.2 * (0.8/0.5) = 0.2 * 1.6 = 0.32
    # p1_norm = 0.32 / (0.32 + 0.32) = 0.5
    # p2_norm = 0.32 / (0.32 + 0.32) = 0.5
    np.testing.assert_allclose(corrected[0], [0.5, 0.5])


def test_apply_label_shift_correction_mismatched_shapes():
    probs = np.array([[0.8, 0.2]])
    src_prior = np.array([0.5, 0.5, 0.0])
    tgt_prior = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="prior shapes must match"):
        apply_label_shift_correction(probs, source_prior=src_prior, target_prior=tgt_prior)

    src_prior2 = np.array([0.5, 0.5])
    tgt_prior2 = np.array([0.5, 0.5, 0.0])
    with pytest.raises(ValueError, match="prior shapes must match"):
        apply_label_shift_correction(probs, source_prior=src_prior2, target_prior=tgt_prior2)


def test_apply_label_shift_correction_zero_priors():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    src_prior = np.array([1.0, 0.0])
    tgt_prior = np.array([0.0, 1.0])

    # With default eps (1e-8), this should not raise, but clip
    corrected = apply_label_shift_correction(probs, source_prior=src_prior, target_prior=tgt_prior)
    assert corrected.shape == probs.shape
    np.testing.assert_allclose(corrected.sum(axis=1), np.ones(probs.shape[0]))


def test_estimate_target_prior_em_with_source_prior():
    probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
    src_prior = np.array([0.5, 0.5])

    estimate = estimate_target_prior_em(
        probs, source_prior=src_prior, config=LabelShiftEMConfig(max_iter=10)
    )

    assert estimate.source_prior.shape == (2,)
    assert estimate.target_prior.shape == (2,)
    np.testing.assert_allclose(estimate.source_prior.sum(), 1.0)
    np.testing.assert_allclose(estimate.target_prior.sum(), 1.0)
    assert estimate.iterations > 0
    assert isinstance(estimate.converged, bool)


def test_estimate_target_prior_em_invalid_source_prior_shape():
    probs = np.array([[0.8, 0.2]])
    src_prior = np.array([0.5, 0.5, 0.0])

    with pytest.raises(ValueError, match="source_prior must have shape"):
        estimate_target_prior_em(probs, source_prior=src_prior)


def test_posterior_label_shift_adapter():
    probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
    src_prior = np.array([0.5, 0.5])

    adapter = PosteriorLabelShiftAdapter(source_prior=src_prior)

    # Test estimate
    estimate = adapter.estimate(probs)
    assert estimate.target_prior is not None

    # Test transform
    corrected = adapter.transform(probs)
    assert corrected.shape == probs.shape
    np.testing.assert_allclose(corrected.sum(axis=1), np.ones(probs.shape[0]))

    # Test estimate + transform
    estimate2 = adapter.estimate(probs)
    corrected2 = adapter.transform(probs)
    assert corrected2.shape == probs.shape
    assert estimate2.target_prior is not None


def test_posterior_label_shift_adapter_missing_source_prior():
    probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'source_prior'"):
        PosteriorLabelShiftAdapter()

    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'source_prior'"):
        estimate_target_prior_em(probs)


def test_posterior_label_shift_adapter_explicit_target_prior():
    probs = np.array([[0.8, 0.2]])
    adapter = PosteriorLabelShiftAdapter(source_prior=np.array([0.5, 0.5]))

    corrected = adapter.transform(probs, target_prior=np.array([0.2, 0.8]))
    assert corrected.shape == probs.shape


def test_estimate_target_prior_em_with_sample_weights():
    probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]])
    weights = np.array([1.0, 2.0, 0.0])

    estimate = estimate_target_prior_em(
        probs,
        source_prior=np.array([0.5, 0.5]),
        sample_weights=weights,
        config=LabelShiftEMConfig(max_iter=10),
    )
    assert estimate.target_prior.shape == (2,)


def test_estimate_target_prior_em_with_subsampling():
    probs = np.random.rand(10, 2)
    probs = probs / probs.sum(axis=1, keepdims=True)

    estimate = estimate_target_prior_em(
        probs, source_prior=np.array([0.5, 0.5]), sample_size=5, random_state=42
    )
    assert estimate.target_prior.shape == (2,)


def test_gaussian_bin_edges_from_targets():
    targets = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    edges = gaussian_bin_edges_from_targets(targets, n_bins=4)
    assert edges.shape == (5,)
    assert edges[0] == 0.0
    assert edges[-1] == 9.0


def test_gaussian_bin_edges_fallback():
    # Only two unique values, should fallback to linspace
    targets = np.array([0.0, 0.0, 1.0, 1.0])
    edges = gaussian_bin_edges_from_targets(targets, n_bins=2)
    assert edges.shape == (3,)
    assert edges[0] == 0.0
    assert edges[-1] == 1.0


def test_gaussian_bin_probabilities():
    mean = np.array([0.0, 1.0])
    std = np.array([1.0, 0.5])
    edges = np.array([-1.0, 0.0, 1.0])

    probs = gaussian_bin_probabilities(mean, std, edges)
    assert probs.shape == (2, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_gaussian_moments_from_binned_probabilities():
    probs = np.array([[0.5, 0.5], [0.1, 0.9]])
    edges = np.array([0.0, 1.0, 2.0])

    mean, std = gaussian_moments_from_binned_probabilities(probs, edges)
    assert mean.shape == (2,)
    assert std.shape == (2,)


def test_correct_gaussian_predictions_for_label_shift():
    mean = np.array([0.0, 1.0, 2.0])
    std = np.array([1.0, 1.0, 1.0])
    source_targets = np.array([0.5, 1.5, 2.5, 0.1, 1.9, 2.1, 0.8, 1.2, 2.8])

    config = GaussianLabelShiftConfig(n_bins=3)
    c_mean, c_std, meta = correct_gaussian_predictions_for_label_shift(
        mean=mean, std=std, source_targets=source_targets, config=config
    )

    assert c_mean.shape == (3,)
    assert c_std.shape == (3,)
    assert "target_prior" in meta
    assert "source_prior" in meta
    assert "estimate_converged" in meta

    # Test with features
    features = np.random.rand(3, 5)
    c_mean_f, c_std_f, meta_f = correct_gaussian_predictions_for_label_shift(
        mean=mean, std=std, source_targets=source_targets, features=features, config=config
    )
    assert c_mean_f.shape == (3,)
    assert c_std_f.shape == (3,)


def test_gaussian_bin_edges_invalid_values():
    targets = np.array([np.inf, np.nan, np.inf])
    with pytest.raises(ValueError, match="finite values"):
        gaussian_bin_edges_from_targets(targets, n_bins=2)


def test_gaussian_bin_edges_constant():
    targets = np.array([5.0, 5.0, 5.0])
    edges = gaussian_bin_edges_from_targets(targets, n_bins=2)
    assert edges.shape == (3,)
    assert edges[-1] == 6.0  # hi = lo + 1.0


def test_correct_gaussian_predictions_top_fraction_none():
    mean = np.array([0.0, 1.0])
    std = np.array([1.0, 1.0])
    source_targets = np.array([0.5, 1.5, 2.5, 0.1])
    config = GaussianLabelShiftConfig(n_bins=3, top_fraction=None)

    c_mean, c_std, meta = correct_gaussian_predictions_for_label_shift(
        mean=mean, std=std, source_targets=source_targets, config=config
    )
    assert c_mean.shape == (2,)


def test_estimate_target_prior_em_with_subsampling_and_weights():
    probs = np.random.rand(10, 2)
    probs = probs / probs.sum(axis=1, keepdims=True)
    weights = np.ones(10)

    estimate = estimate_target_prior_em(
        probs,
        source_prior=np.array([0.5, 0.5]),
        sample_weights=weights,
        sample_size=5,
        random_state=42,
    )
    assert estimate.target_prior.shape == (2,)


def test_posterior_adapter_transform_without_target_prior_and_estimate_not_called():
    probs = np.array([[0.8, 0.2]])
    adapter = PosteriorLabelShiftAdapter(source_prior=np.array([0.5, 0.5]))

    # transform will call estimate because target_prior and last_estimate are None
    corrected = adapter.transform(probs)
    assert corrected.shape == probs.shape


def test_gaussian_bin_edges_validation_empty_or_nan() -> None:
    from torchregress.test_time.label_shift import gaussian_bin_edges_from_targets

    with pytest.raises(ValueError, match="non-empty"):
        gaussian_bin_edges_from_targets(np.array([]), n_bins=10)
    with pytest.raises(ValueError, match="finite values"):
        gaussian_bin_edges_from_targets(np.array([1.0, np.nan, 2.0]), n_bins=10)
    with pytest.raises(ValueError, match="finite values"):
        gaussian_bin_edges_from_targets(np.array([1.0, np.inf, 2.0]), n_bins=10)


def test_posterior_label_shift_target_recovery() -> None:
    # True target prior is [0.2, 0.8]
    # Source prior is [0.7, 0.3]
    rng = np.random.default_rng(42)
    n_samples = 5000
    true_tgt_prior = np.array([0.2, 0.8])
    src_prior = np.array([0.7, 0.3])

    classes = rng.choice([0, 1], p=true_tgt_prior, size=n_samples)

    x = np.zeros(n_samples)
    x[classes == 0] = rng.normal(-1.0, 1.0, size=(classes == 0).sum())
    x[classes == 1] = rng.normal(1.0, 1.0, size=(classes == 1).sum())

    from scipy.stats import norm

    f0 = norm.pdf(x, loc=-1.0, scale=1.0)
    f1 = norm.pdf(x, loc=1.0, scale=1.0)

    denom = f0 * src_prior[0] + f1 * src_prior[1]
    probs_s = np.zeros((n_samples, 2))
    probs_s[:, 0] = (f0 * src_prior[0]) / denom
    probs_s[:, 1] = (f1 * src_prior[1]) / denom

    adapter = PosteriorLabelShiftAdapter(source_prior=src_prior)
    est = adapter.estimate(probs_s)

    np.testing.assert_allclose(est.target_prior, true_tgt_prior, atol=0.03)
