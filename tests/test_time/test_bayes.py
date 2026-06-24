"""
Unit tests for torchregress.test_time.bayes — Bayesian linear regression heads.

Covers BayesianLinearHead, RecursiveBayesianHead, and module-level helpers
(_as_tensor, _augment_features, _posterior_covariance_from_precision).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchregress.test_time.bayes import (
    BayesianLinearHead,
    RecursiveBayesianHead,
    _as_tensor,
    _augment_features,
    _posterior_covariance_from_precision,
)

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def _make_head(**kwargs: object) -> BayesianLinearHead:
    defaults = {
        "in_features": 4,
        "out_features": 1,
        "fit_intercept": True,
        "prior_precision": 1.0,
        "noise_variance": 0.1,
    }
    return BayesianLinearHead(**{**defaults, **kwargs})


def _make_data(n: int = 32, d: int = 4, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    X = torch.randn(n, d, generator=gen)
    w = torch.randn(d, 1, generator=gen)
    y = X @ w + 0.1 * torch.randn(n, 1, generator=gen)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestAsTensor:
    def test_numpy_to_tensor(self) -> None:
        """Numpy to tensor."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        t = _as_tensor(x, device=DEVICE, dtype=DTYPE)
        assert isinstance(t, torch.Tensor)
        assert t.device == DEVICE
        assert t.dtype == DTYPE

    def test_tensor_preserved(self) -> None:
        """Tensor preserved."""
        x = torch.tensor([1.0, 2.0], device=DEVICE, dtype=DTYPE)
        t = _as_tensor(x, device=DEVICE, dtype=DTYPE)
        assert t is x  # same object when device/dtype match

    def test_tensor_moved(self) -> None:
        """Tensor moved."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        x = torch.tensor([1.0, 2.0])
        t = _as_tensor(x, device=torch.device("cpu"), dtype=torch.float64)
        assert t.device == torch.device("cpu")
        assert t.dtype == torch.float64


class TestAugmentFeatures:
    def test_no_intercept(self) -> None:
        """No intercept."""
        phi = torch.ones(5, 3)
        out = _augment_features(phi, fit_intercept=False)
        assert out.shape == (5, 3)
        assert out is phi

    def test_with_intercept(self) -> None:
        """With intercept."""
        phi = torch.ones(5, 3)
        out = _augment_features(phi, fit_intercept=True)
        assert out.shape == (5, 4)
        assert torch.all(out[:, -1] == 1.0)  # intercept column


class TestPosteriorCovarianceFromPrecision:
    def test_identity_precision(self) -> None:
        """Identity precision."""
        # No co-built tensor in this test — fall back to legacy device/dtype so
        # the fixture doesn't implicitly rely on the bayes module's defaults.
        prec = torch.eye(3, device=torch.device("cpu"), dtype=torch.get_default_dtype())
        cov = _posterior_covariance_from_precision(prec, jitter=1e-6)
        assert cov.shape == (3, 3)

    def test_diagonal_precision(self) -> None:
        """Diagonal precision."""
        prec = torch.diag(torch.tensor([2.0, 3.0, 4.0])).to(
            device=torch.device("cpu"), dtype=torch.get_default_dtype()
        )
        cov = _posterior_covariance_from_precision(prec, jitter=0.0)
        expected = torch.diag(1.0 / torch.tensor([2.0, 3.0, 4.0])).to(
            device=torch.device("cpu"), dtype=torch.get_default_dtype()
        )
        assert torch.allclose(cov, expected, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# BayesianLinearHead — constructor validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestBayesianLinearHeadInit:
    def test_default_construction(self) -> None:
        """Default construction."""
        head = _make_head()
        assert head.in_features == 4
        assert head.out_features == 1
        assert head.is_fitted is False

    def test_in_features_non_positive_raises(self) -> None:
        """In features non positive raises."""
        with pytest.raises(ValueError, match="in_features"):
            BayesianLinearHead(in_features=0, out_features=1)

    def test_out_features_non_positive_raises(self) -> None:
        """Out features non positive raises."""
        with pytest.raises(ValueError, match="out_features"):
            BayesianLinearHead(in_features=2, out_features=0)

    def test_noise_variance_non_positive_raises(self) -> None:
        """Noise variance non positive raises."""
        with pytest.raises(ValueError, match="noise_variance"):
            BayesianLinearHead(in_features=2, noise_variance=0.0)

    def test_prior_precision_non_positive_raises(self) -> None:
        """Prior precision non positive raises."""
        with pytest.raises(ValueError, match="prior_precision"):
            BayesianLinearHead(in_features=2, prior_precision=0.0)

    def test_auto_noise_not_bool_raises(self) -> None:
        """Auto noise not bool raises."""
        with pytest.raises(TypeError, match="auto_noise"):
            BayesianLinearHead(in_features=2, auto_noise=1)  # type: ignore[arg-type]

    def test_rbf_centers_non_positive_raises(self) -> None:
        """RBF centers non positive raises."""
        with pytest.raises(ValueError, match="rbf_centers"):
            BayesianLinearHead(in_features=2, rbf_centers=0)

    def test_prior_mean_scalar_broadcasts(self) -> None:
        """Prior mean scalar broadcasts."""
        head = BayesianLinearHead(in_features=3, fit_intercept=True, prior_mean=2.0)
        assert head._h0.numel() == 4  # 3 + intercept

    def test_prior_mean_vector_wrong_length_raises(self) -> None:
        """Prior mean vector wrong length raises."""
        with pytest.raises(ValueError, match="prior_mean"):
            BayesianLinearHead(
                in_features=3, fit_intercept=True, prior_mean=torch.tensor([1.0, 2.0])
            )

    def test_prior_mean_vector_correct_length(self) -> None:
        """Prior mean vector correct length."""
        head = BayesianLinearHead(
            in_features=2, fit_intercept=True, prior_mean=torch.tensor([1.0, 2.0, 3.0])
        )
        assert head._h0.numel() == 3

    def test_multi_output_buffer_shape(self) -> None:
        """Multi output buffer shape."""
        head = BayesianLinearHead(in_features=3, out_features=2)
        assert head._h.shape == (2, head._d_eff)
        assert torch.all(head._h[0] == head._h[1])  # same prior per output

    def test_rbf_enabled_buffers(self) -> None:
        """RBF enabled buffers."""
        head = BayesianLinearHead(in_features=3, rbf_centers=5)
        assert head.rbf_centers == 5
        assert head._d_eff == 5 + 1  # rbf_centers + intercept

    def test_no_intercept_effective_dim(self) -> None:
        """No intercept effective dim."""
        head = BayesianLinearHead(in_features=4, fit_intercept=False)
        assert head._d_eff == 4


# ═══════════════════════════════════════════════════════════════════════════════
# BayesianLinearHead — fit
# ═══════════════════════════════════════════════════════════════════════════════


class TestBayesianLinearHeadFit:
    def test_fit_returns_self(self) -> None:
        """Fit returns self."""
        head = _make_head()
        X, y = _make_data()
        result = head.fit(X, y)
        assert result is head

    def test_sets_is_fitted(self) -> None:
        """Sets is fitted."""
        head = _make_head()
        X, y = _make_data()
        head.fit(X, y)
        assert head.is_fitted is True

    def test_shape_mismatch_raises(self) -> None:
        """Shape mismatch raises."""
        head = _make_head()
        X = torch.randn(32, 4)
        y = torch.randn(16, 1)
        with pytest.raises(ValueError, match="same number of rows"):
            head.fit(X, y)

    def test_wrong_out_features_raises(self) -> None:
        """Wrong out features raises."""
        head = _make_head(out_features=1)
        X = torch.randn(32, 4)
        y = torch.randn(32, 3)
        with pytest.raises(ValueError, match="columns"):
            head.fit(X, y)

    def test_wrong_in_features_raises(self) -> None:
        """Wrong in features raises."""
        head = _make_head(in_features=4)
        X = torch.randn(32, 3)
        y = torch.randn(32, 1)
        with pytest.raises(ValueError, match="features"):
            head.fit(X, y)

    def test_accepts_numpy_inputs(self) -> None:
        """Accepts numpy inputs."""
        head = _make_head()
        X = np.random.default_rng(0).normal(size=(32, 4)).astype(np.float32)
        y = np.random.default_rng(1).normal(size=(32, 1)).astype(np.float32)
        head.fit(X, y)
        assert head.is_fitted

    def test_accepts_1d_y(self) -> None:
        """Accepts 1d y."""
        head = _make_head(out_features=1)
        X = torch.randn(32, 4)
        y = torch.randn(32)
        head.fit(X, y)
        assert head.is_fitted

    def test_updates_n_obs(self) -> None:
        """Updates n obs."""
        head = _make_head()
        X, y = _make_data(50)
        head.fit(X, y)
        assert head._n_obs.item() == 50

    def test_auto_noise_estimates_variance(self) -> None:
        """Auto noise estimates variance."""
        head = _make_head(auto_noise=True, noise_variance=1.0)
        X, y = _make_data(64)
        head.fit(X, y)
        # auto_noise overrides noise_variance: max((0.2*std_y)^2, 1e-4)
        expected_std = y.std().item() * 0.2
        expected_var = max(expected_std**2, 1e-4)
        assert head.noise_variance == pytest.approx(expected_var)

    def test_fit_resets_posterior(self) -> None:
        """Fit resets posterior."""
        head = _make_head()
        X, y = _make_data(32)
        head.fit(X, y)
        first_lambda = head._Lambda.clone()

        # Fit again with different data — should reset first
        X2, y2 = _make_data(64, seed=5)
        head.fit(X2, y2)
        # Lambda should be different (accumulated 64 samples vs 32)
        assert not torch.allclose(first_lambda, head._Lambda)

    def test_negative_sample_weight_raises(self) -> None:
        """Negative sample weight raises."""
        head = _make_head()
        X, y = _make_data(16)
        w = torch.tensor([1.0, -0.5, 1.0, 1.0] + [1.0] * 12)
        with pytest.raises(ValueError, match="sample_weight"):
            head.fit(X, y, sample_weight=w)


# ═══════════════════════════════════════════════════════════════════════════════
# BayesianLinearHead — predict
# ═══════════════════════════════════════════════════════════════════════════════


class TestBayesianLinearHeadPredict:
    def test_predict_before_fit_raises(self) -> None:
        """Predict before fit raises."""
        head = _make_head()
        X = torch.randn(8, 4)
        with pytest.raises(RuntimeError, match="fit before predict"):
            head.predict(X)

    def test_predict_returns_mean(self) -> None:
        """Predict returns mean."""
        head = _make_head()
        X, y = _make_data(32)
        head.fit(X, y)
        X_test = torch.randn(8, 4)
        out = head.predict(X_test)
        assert "mean" in out
        assert out["mean"].shape == (8, 1)

    def test_predict_return_std(self) -> None:
        """Predict return std."""
        head = _make_head()
        X, y = _make_data(32)
        head.fit(X, y)
        X_test = torch.randn(8, 4)
        out = head.predict(X_test, return_std=True)
        assert "variance" in out
        assert "std" in out
        assert out["std"].shape == (8, 1)
        assert torch.all(out["std"] >= 0)

    def test_predict_without_noise(self) -> None:
        """Predict without noise."""
        head = _make_head(noise_variance=0.5)
        X, y = _make_data(32)
        head.fit(X, y)
        X_test = torch.randn(8, 4)
        out_with = head.predict(X_test, return_std=True, include_noise=True)
        out_without = head.predict(X_test, return_std=True, include_noise=False)
        assert torch.all(out_without["variance"] <= out_with["variance"])

    def test_predict_multi_output(self) -> None:
        """Predict multi output."""
        head = _make_head(out_features=3)
        X = torch.randn(48, 4)
        y = torch.randn(48, 3)
        head.fit(X, y)
        X_test = torch.randn(8, 4)
        out = head.predict(X_test, return_std=True)
        assert out["mean"].shape == (8, 3)
        assert out["std"].shape == (8, 3)

    def test_predict_numpy_input(self) -> None:
        """Predict numpy input."""
        head = _make_head()
        X, y = _make_data(32)
        head.fit(X, y)
        X_np = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)
        out = head.predict(X_np)
        assert "mean" in out


# ═══════════════════════════════════════════════════════════════════════════════
# BayesianLinearHead — predictive_batch
# ═══════════════════════════════════════════════════════════════════════════════


class TestBayesianLinearHeadPredictiveBatch:
    def test_returns_predictive_batch(self) -> None:
        """Returns predictive batch."""
        head = _make_head()
        X, y = _make_data(32)
        head.fit(X, y)
        X_test = torch.randn(8, 4)
        batch = head.predictive_batch(X_test)
        assert batch.mean is not None
        assert batch.std is not None
        assert batch.point is not None

    def test_extra_contains_uncertainty_decomposition(self) -> None:
        """Extra contains uncertainty decomposition."""
        head = _make_head(noise_variance=0.25)
        X, y = _make_data(48)
        head.fit(X, y)
        X_test = torch.randn(8, 4)
        batch = head.predictive_batch(X_test)
        assert batch.extra is not None
        assert "epistemic_variance" in batch.extra
        assert "aleatoric_variance" in batch.extra
        assert "posterior_trace" in batch.extra
        assert "n_observations_seen" in batch.extra

        n_obs = batch.extra["n_observations_seen"]
        assert isinstance(n_obs, torch.Tensor)
        assert float(n_obs[0, 0].item()) == 48.0

    def test_aleatoric_zero_without_noise(self) -> None:
        """Aleatoric zero without noise."""
        head = _make_head(noise_variance=0.5)
        X, y = _make_data(32)
        head.fit(X, y)
        X_test = torch.randn(8, 4)
        batch = head.predictive_batch(X_test, include_noise=False)
        assert batch.extra is not None
        ale = batch.extra["aleatoric_variance"]
        assert torch.allclose(ale, torch.zeros_like(ale))


# ═══════════════════════════════════════════════════════════════════════════════
# BayesianLinearHead — sample_weights
# ═══════════════════════════════════════════════════════════════════════════════


class TestBayesianLinearHeadSampleWeights:
    def test_before_fit_raises(self) -> None:
        """Before fit raises."""
        head = _make_head()
        with pytest.raises(RuntimeError, match="fit before sample_weights"):
            head.sample_weights(10)

    def test_n_samples_non_positive_raises(self) -> None:
        """N samples non positive raises."""
        head = _make_head()
        X, y = _make_data(16)
        head.fit(X, y)
        with pytest.raises(ValueError, match="n_samples"):
            head.sample_weights(0)

    def test_shape(self) -> None:
        """Shape."""
        head = _make_head(out_features=2)
        X = torch.randn(32, 4)
        y = torch.randn(32, 2)
        head.fit(X, y)
        samples = head.sample_weights(5)
        assert samples.shape == (5, 2, head._d_eff)

    def test_reproducible_with_generator(self) -> None:
        """Reproducible with generator."""
        head = _make_head()
        X, y = _make_data(20)
        head.fit(X, y)
        gen = torch.Generator()
        gen.manual_seed(42)
        s1 = head.sample_weights(3, generator=gen)
        gen = torch.Generator()
        gen.manual_seed(42)
        s2 = head.sample_weights(3, generator=gen)
        assert torch.allclose(s1, s2)


# ═══════════════════════════════════════════════════════════════════════════════
# BayesianLinearHead — posterior properties
# ═══════════════════════════════════════════════════════════════════════════════


class TestBayesianLinearHeadPosterior:
    def test_posterior_precision_shape(self) -> None:
        """Posterior precision shape."""
        head = _make_head()
        X, y = _make_data(16)
        head.fit(X, y)
        prec = head.posterior_precision
        assert prec.shape == (head._d_eff, head._d_eff)

    def test_posterior_mean_shape(self) -> None:
        """Posterior mean shape."""
        head = _make_head(out_features=2)
        X = torch.randn(32, 4)
        y = torch.randn(32, 2)
        head.fit(X, y)
        mean = head.posterior_mean
        assert mean.shape == (2, head._d_eff)

    def test_posterior_covariance_shape(self) -> None:
        """Posterior covariance shape."""
        head = _make_head()
        X, y = _make_data(32)
        head.fit(X, y)
        cov = head.posterior_covariance
        assert cov.shape == (head._d_eff, head._d_eff)

    def test_mean_covariance_consistency(self) -> None:
        """posterior_covariance @ posterior_precision ≈ I (with jitter)."""
        head = _make_head(jitter=1e-4)
        X, y = _make_data(48)
        head.fit(X, y)
        prec = head.posterior_precision
        cov = head.posterior_covariance
        prod = cov @ prec
        # Pin dtype/device to ``prec`` so the fixture doesn't implicitly rely
        # on the bayes module handling dtype/device of input fixtures internally.
        expected = torch.eye(head._d_eff, device=prec.device, dtype=prec.dtype)
        assert torch.allclose(prod, expected, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# BayesianLinearHead — reset_posterior
# ═══════════════════════════════════════════════════════════════════════════════


class TestBayesianLinearHeadReset:
    def test_resets_to_prior(self) -> None:
        """Resets to prior."""
        head = _make_head()
        X, y = _make_data(64)
        head.fit(X, y)
        assert head.is_fitted
        assert not torch.allclose(head._Lambda, head._Lambda0)

        head.reset_posterior()
        assert torch.allclose(head._Lambda, head._Lambda0)
        assert torch.allclose(head._h[0], head._h0)
        assert head.is_fitted is False
        assert head._n_obs.item() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# BayesianLinearHead — RBF features
# ═══════════════════════════════════════════════════════════════════════════════


class TestBayesianLinearHeadRBF:
    def test_rbf_expands_features(self) -> None:
        """RBF expands features."""
        head = BayesianLinearHead(in_features=3, rbf_centers=5, fit_intercept=False)
        X = torch.randn(32, 3)
        y = torch.randn(32, 1)
        head.fit(X, y)
        assert head._rbf_centers.shape == (5, 3)
        assert head._rbf_gamma.item() > 0

    def test_rbf_predict_uses_same_centers(self) -> None:
        """RBF predict uses same centers."""
        head = BayesianLinearHead(in_features=3, rbf_centers=4, fit_intercept=False)
        X, y = _make_data(32, d=3)
        head.fit(X, y)
        centers_before = head._rbf_centers.clone()

        X_test = torch.randn(8, 3)
        out = head.predict(X_test)
        assert out["mean"].shape == (8, 1)
        # Centers should not change during predict
        assert torch.equal(head._rbf_centers, centers_before)

    def test_rbf_wrong_input_dim_raises(self) -> None:
        """RBF wrong input dim raises."""
        head = BayesianLinearHead(in_features=3, rbf_centers=4)
        X = torch.randn(32, 3)
        y = torch.randn(32, 1)
        head.fit(X, y)

        X_wrong = torch.randn(8, 5)
        with pytest.raises(ValueError, match="3 raw features"):
            head.predict(X_wrong)

    def test_rbf_user_gamma(self) -> None:
        """RBF user gamma."""
        head = BayesianLinearHead(in_features=3, rbf_centers=4, rbf_gamma=2.5, fit_intercept=False)
        X, y = _make_data(32, d=3)
        head.fit(X, y)
        assert head._rbf_gamma.item() == 2.5

    def test_rbf_fit_then_predict(self) -> None:
        """Full RBF workflow: fit on data, predict on new data."""
        head = BayesianLinearHead(
            in_features=2, rbf_centers=8, fit_intercept=True, noise_variance=0.01
        )
        gen = torch.Generator()
        gen.manual_seed(0)
        X = torch.randn(64, 2, generator=gen)
        w = torch.randn(9, 1, generator=gen)  # 8 RBF centres + 1 intercept = 9
        # We need to generate y from the RBF features for consistency
        head._init_rbf(X)
        phi_rbf = head._apply_rbf(X)
        phi = torch.cat([phi_rbf, torch.ones(64, 1)], dim=1)  # intercept
        y = phi @ w + 0.02 * torch.randn(64, 1, generator=gen)

        head.reset_posterior()
        head.fit(X, y)

        X_test = torch.randn(16, 2, generator=gen)
        out = head.predict(X_test, return_std=True)
        assert out["mean"].shape == (16, 1)
        assert torch.all(out["std"] >= 0)


# ═══════════════════════════════════════════════════════════════════════════════
# RecursiveBayesianHead
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecursiveBayesianHeadInit:
    def test_default_forgetting_factor(self) -> None:
        """Default forgetting factor."""
        head = RecursiveBayesianHead(in_features=3)
        assert head.forgetting_factor == 1.0

    def test_custom_forgetting_factor(self) -> None:
        """Custom forgetting factor."""
        head = RecursiveBayesianHead(in_features=3, forgetting_factor=0.9)
        assert head.forgetting_factor == 0.9

    def test_forgetting_factor_out_of_range_raises(self) -> None:
        """Forgetting factor out of range raises."""
        with pytest.raises(ValueError, match="forgetting_factor"):
            RecursiveBayesianHead(in_features=3, forgetting_factor=0.0)
        with pytest.raises(ValueError, match="forgetting_factor"):
            RecursiveBayesianHead(in_features=3, forgetting_factor=1.5)


class TestRecursiveBayesianHeadPartialFit:
    def test_partial_fit_returns_self(self) -> None:
        """Partial fit returns self."""
        head = RecursiveBayesianHead(in_features=4)
        X, y = _make_data(16)
        result = head.partial_fit(X, y)
        assert result is head

    def test_partial_fit_sets_fitted(self) -> None:
        """Partial fit sets fitted."""
        head = RecursiveBayesianHead(in_features=4)
        X, y = _make_data(8)
        head.partial_fit(X, y)
        assert head.is_fitted

    def test_multiple_partial_fits_accumulate(self) -> None:
        """Multiple partial fits accumulate."""
        head = RecursiveBayesianHead(in_features=4, forgetting_factor=1.0)
        X, y = _make_data(32, seed=10)
        # Split into 4 batches of 8
        n = 8
        for i in range(4):
            head.partial_fit(X[i * n : (i + 1) * n], y[i * n : (i + 1) * n])
        assert head._n_obs.item() == 32

        # Full fit should give similar posterior
        head2 = RecursiveBayesianHead(in_features=4, forgetting_factor=1.0)
        head2.fit(X, y)
        assert torch.allclose(head._Lambda, head2._Lambda, atol=1e-4)

    def test_forgetting_downweights_old_data(self) -> None:
        """Forgetting downweights old data."""
        head = RecursiveBayesianHead(in_features=4, forgetting_factor=0.5)
        X, y = _make_data(32, seed=20)
        # Fit first batch
        head.partial_fit(X[:16], y[:16])
        lam_after_first = head._Lambda.clone()

        # Fit second batch — old precision is halved first
        head.partial_fit(X[16:], y[16:])
        # Lambda should be NOT equal to lam_after_first + second contribution
        # because lam_after_first was scaled by 0.5
        assert not torch.allclose(head._Lambda, lam_after_first)

    def test_partial_fit_rbf_lazy_init(self) -> None:
        """Partial fit RBF lazy init."""
        head = RecursiveBayesianHead(in_features=3, rbf_centers=5, fit_intercept=False)
        X, y = _make_data(16, d=3, seed=30)
        head.partial_fit(X, y)
        assert head._rbf_centers.numel() == 15  # 5 centres × 3 features
        assert head._rbf_gamma.item() > 0

    def test_partial_fit_with_sample_weight(self) -> None:
        """Partial fit with sample weight."""
        head = RecursiveBayesianHead(in_features=4, forgetting_factor=1.0)
        X, y = _make_data(16)
        w = torch.ones(16) * 2.0
        head.partial_fit(X, y, sample_weight=w)
        assert head.is_fitted

    def test_partial_fit_shape_validation(self) -> None:
        """Partial fit shape validation."""
        head = RecursiveBayesianHead(in_features=4, out_features=2)
        X = torch.randn(16, 4)
        y = torch.randn(16, 3)  # wrong columns
        with pytest.raises(ValueError, match="columns"):
            head.partial_fit(X, y)

    def test_full_fit_resets_before_accumulate(self) -> None:
        """fit() should reset and accumulate fresh, unlike partial_fit()."""
        head = RecursiveBayesianHead(in_features=4, forgetting_factor=1.0)
        X, y = _make_data(16)
        head.partial_fit(X, y)
        lam_partial = head._Lambda.clone()
        n_partial = head._n_obs.item()

        # fit resets and re-accumulates
        head.fit(X, y)
        assert torch.allclose(head._Lambda, lam_partial, atol=1e-4)
        assert head._n_obs.item() == n_partial


class TestRecursiveBayesianHeadInvariants:
    def test_forgetting_factor_natural_parameter_discounting(self) -> None:
        # Check bounded mean under repeated updates with forgetting_factor < 1.0
        head = RecursiveBayesianHead(
            in_features=1,
            fit_intercept=False,
            prior_precision=1.0,
            forgetting_factor=0.9,
            noise_variance=1.0,
        )

        # Initial fit
        X1 = torch.ones(10, 1)
        y1 = torch.ones(10, 1)
        head.partial_fit(X1, y1)

        # The posterior mean should be close to 0.909 (approx 10 / (1 + 10) = 0.909)
        mean1 = head.posterior_mean
        assert 0.8 < mean1.item() < 1.0

        # Update repeatedly with y = 2.0.
        # Under the old incorrect rule (where Lambda shrunk but h stayed huge), the mean would blow up.
        # Under the correct rule, the mean should transition towards 2.0 and remain bounded.
        X2 = torch.ones(1, 1)
        y2 = torch.ones(1, 1) * 2.0
        for _ in range(50):
            head.partial_fit(X2, y2)

        mean2 = head.posterior_mean
        # Bounded and close to 2.0
        assert 1.5 < mean2.item() < 2.5

    def test_partial_fit_and_fit_equivalence_at_lambda_1(self) -> None:
        X = torch.randn(20, 3)
        y = torch.randn(20, 1)

        head_fit = RecursiveBayesianHead(in_features=3, forgetting_factor=1.0)
        head_fit.fit(X, y)

        head_partial = RecursiveBayesianHead(in_features=3, forgetting_factor=1.0)
        head_partial.partial_fit(X[:10], y[:10])
        head_partial.partial_fit(X[10:], y[10:])

        assert torch.allclose(head_fit._Lambda, head_partial._Lambda, atol=1e-5)
        assert torch.allclose(head_fit._h, head_partial._h, atol=1e-5)
        assert head_fit._n_obs.item() == head_partial._n_obs.item()

    def test_rbf_reset_on_refit(self) -> None:
        # fit() should reset RBF centers when called a second time
        head = RecursiveBayesianHead(in_features=3, rbf_centers=5)
        X1 = torch.randn(20, 3)
        y1 = torch.randn(20, 1)
        head.fit(X1, y1)
        centers1 = head._rbf_centers.clone()

        # Second fit on a new dataset
        X2 = torch.randn(20, 3) + 10.0  # very different range
        y2 = torch.randn(20, 1)
        head.fit(X2, y2)
        centers2 = head._rbf_centers.clone()

        # Check centers changed (they should be chosen from X2, not X1)
        assert not torch.allclose(centers1, centers2)

    def test_rbf_center_selection_generator(self) -> None:
        # Seeding generator should yield deterministic RBF selection
        X = torch.randn(30, 4)
        y = torch.randn(30, 1)

        gen1 = torch.Generator().manual_seed(42)
        head1 = RecursiveBayesianHead(in_features=4, rbf_centers=10)
        head1.fit(X, y, generator=gen1)

        gen2 = torch.Generator().manual_seed(42)
        head2 = RecursiveBayesianHead(in_features=4, rbf_centers=10)
        head2.fit(X, y, generator=gen2)

        assert torch.allclose(head1._rbf_centers, head2._rbf_centers)
