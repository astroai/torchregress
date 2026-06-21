"""
Unit tests for torchregress.causal.dr.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchregress.causal.dr import (
    _as_1d,
    _as_2d,
    _build_model,
    _crossfit_nuisances,
    _dr_scores,
    _fit_model,
    _make_folds,
    _normal_ci,
    _predict_outcome,
    _predict_propensity,
    dr_ate,
    dr_cate,
    dr_policy_value,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Simple sklearn-style mock models for DR testing
# ═══════════════════════════════════════════════════════════════════════════════


class MockRegressor:
    """Simple OLS regressor via pseudoinverse (numpy)."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return (X @ self.coef_).reshape(-1)


class MockClassifier:
    """Simple logistic regression via pseudoinverse (numpy)."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 0:
            y = y.reshape(1)
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        logits = X @ self.coef_
        # Return 2-column proba
        p1 = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - p1, p1])


# ═══════════════════════════════════════════════════════════════════════════════
# _as_2d / _as_1d
# ═══════════════════════════════════════════════════════════════════════════════


class TestAsShapes:
    def test_as_2d_from_1d(self) -> None:
        """1D tensor gets an extra dim."""
        x = torch.randn(10)
        result = _as_2d(x)
        assert result.shape == (10, 1)

    def test_as_2d_passthrough(self) -> None:
        """2D tensor passes through."""
        x = torch.randn(10, 3)
        result = _as_2d(x)
        assert result.shape == (10, 3)

    def test_as_1d_flattens(self) -> None:
        """Multi-dim tensor is flattened to 1D."""
        x = torch.randn(5, 2)
        result = _as_1d(x)
        assert result.shape == (10,)


# ═══════════════════════════════════════════════════════════════════════════════
# _build_model / _fit_model / _predict_outcome / _predict_propensity
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelHelpers:
    def test_build_model_from_class(self) -> None:
        """Class factory instantiates a new model."""
        m = _build_model(MockRegressor)
        assert isinstance(m, MockRegressor)

    def test_build_model_from_instance(self) -> None:
        """Instance is deep-copied."""
        m1 = MockRegressor()
        m1.coef_ = np.array([1.0, 2.0])
        m2 = _build_model(m1)
        assert m2.coef_ is not None
        assert np.array_equal(m2.coef_, m1.coef_)
        assert m2 is not m1

    def test_fit_model(self) -> None:
        """fit() is called with numpy arrays."""
        m = MockRegressor()
        x = torch.randn(20, 2)
        y = torch.randn(20)
        result = _fit_model(m, x, y)
        assert result.coef_ is not None
        assert result.coef_.shape == (2,)

    def test_fit_model_raises_without_fit(self) -> None:
        """Model without fit() raises TypeError."""

        class BadModel:
            pass

        with pytest.raises(TypeError, match="must implement fit"):
            _fit_model(BadModel(), torch.randn(10, 2), torch.randn(10))

    def test_predict_outcome(self) -> None:
        """predict() returns tensor on correct device."""
        m = MockRegressor()
        x = torch.randn(10, 2)
        m.fit(x.numpy(), torch.randn(10).numpy())
        result = _predict_outcome(m, x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (10,)

    def test_predict_outcome_raises_without_predict(self) -> None:
        """Model without predict() raises TypeError."""

        class BadModel:
            def fit(self, X, y):  # noqa: ANN001, ANN201
                pass

        with pytest.raises(TypeError, match="must implement predict"):
            _predict_outcome(BadModel(), torch.randn(10, 2))

    def test_predict_propensity_with_proba(self) -> None:
        """Model with predict_proba returns column 1 clamped."""
        m = MockClassifier()
        x = torch.randn(20, 2)
        t = (torch.rand(20) > 0.5).float()
        m.fit(x.numpy(), t.numpy())
        result = _predict_propensity(m, x)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (20,)
        assert (result >= 1e-4).all()
        assert (result <= 1 - 1e-4).all()

    def test_predict_propensity_raises_without_methods(self) -> None:
        """Model without predict_proba or predict raises TypeError."""

        class BadModel:
            def fit(self, X, y):  # noqa: ANN001, ANN201
                pass

        with pytest.raises(TypeError, match="must implement"):
            _predict_propensity(BadModel(), torch.randn(10, 2))


# ═══════════════════════════════════════════════════════════════════════════════
# _make_folds
# ═══════════════════════════════════════════════════════════════════════════════


class TestMakeFolds:
    def test_returns_correct_number_of_folds(self) -> None:
        """Returns exactly `folds` train/test splits."""
        splits = _make_folds(n=20, folds=4, seed=42)
        assert len(splits) == 4

    def test_train_test_disjoint(self) -> None:
        """Train and test indices are disjoint."""
        splits = _make_folds(n=30, folds=3, seed=42)
        all_test_indices = set()
        for train_idx, test_idx in splits:
            assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
            all_test_indices.update(test_idx.tolist())
        assert len(all_test_indices) == 30

    def test_folds_lt_2_raises(self) -> None:
        """folds < 2 raises ValueError."""
        with pytest.raises(ValueError, match="folds must be >= 2"):
            _make_folds(n=10, folds=1, seed=42)


# ═══════════════════════════════════════════════════════════════════════════════
# _normal_ci
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalCI:
    def test_ci_contains_estimate(self) -> None:
        """CI bounds bracket the estimate."""
        lo, hi = _normal_ci(estimate=5.0, se=1.0, alpha=0.05)
        assert lo < 5.0 < hi

    def test_wider_for_smaller_alpha(self) -> None:
        """Smaller alpha gives wider CI."""
        lo_10, hi_10 = _normal_ci(10.0, 1.0, alpha=0.10)
        lo_01, hi_01 = _normal_ci(10.0, 1.0, alpha=0.01)
        assert (hi_10 - lo_10) < (hi_01 - lo_01)

    def test_alpha_0_05_uses_z_1_96(self) -> None:
        """alpha=0.05 uses the hardcoded z value 1.96."""
        lo, hi = _normal_ci(0.0, 1.0, alpha=0.05)
        assert lo == pytest.approx(-1.959963984540054)
        assert hi == pytest.approx(1.959963984540054)


# ═══════════════════════════════════════════════════════════════════════════════
# _dr_scores
# ═══════════════════════════════════════════════════════════════════════════════


class TestDRScores:
    def test_shape(self) -> None:
        """Output shape matches input."""
        y = torch.randn(50)
        t = (torch.rand(50) > 0.5).float()
        mu1 = torch.randn(50)
        mu0 = torch.randn(50)
        e = torch.full((50,), 0.5)
        scores = _dr_scores(y, t, mu1, mu0, e)
        assert scores.shape == (50,)


# ═══════════════════════════════════════════════════════════════════════════════
# _crossfit_nuisances
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossfitNuisances:
    def test_returns_expected_keys(self) -> None:
        """Returns mu1_hat, mu0_hat, e_hat."""
        x = torch.randn(30, 2)
        t = (torch.rand(30) > 0.5).float()
        y = torch.randn(30)
        result = _crossfit_nuisances(
            x,
            t,
            y,
            outcome_model=MockRegressor,
            propensity_model=MockClassifier,
            folds=2,
            seed=42,
            eps=1e-4,
        )
        assert "mu1_hat" in result
        assert "mu0_hat" in result
        assert "e_hat" in result
        assert result["mu1_hat"].shape == (30,)
        assert result["mu0_hat"].shape == (30,)
        assert result["e_hat"].shape == (30,)


# ═══════════════════════════════════════════════════════════════════════════════
# dr_ate
# ═══════════════════════════════════════════════════════════════════════════════


class TestDRATE:
    def test_returns_expected_keys(self) -> None:
        """Returns ATE estimate, SE, CI, diagnostics."""
        n = 100
        x = torch.randn(n, 2)
        t = (torch.rand(n) > 0.5).float()
        y = 2.0 * t + x[:, 0] + 0.1 * torch.randn(n)
        result = dr_ate(
            x,
            t,
            y,
            outcome_model=MockRegressor,
            propensity_model=MockClassifier,
            folds=2,
            alpha=0.05,
            seed=42,
        )
        assert "estimate" in result
        assert "se" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "diagnostics" in result
        assert result["ci_lower"] <= result["ci_upper"]
        assert result["se"] > 0

    def test_mismatched_shapes_raises(self) -> None:
        """Mismatched x/t/y shapes raise ValueError."""
        with pytest.raises(ValueError, match="share sample dimension"):
            dr_ate(
                torch.randn(10, 2),
                torch.randn(20),
                torch.randn(10),
                outcome_model=MockRegressor,
                propensity_model=MockClassifier,
            )

    def test_single_fold_raises(self) -> None:
        """folds < 2 raises ValueError."""
        with pytest.raises(ValueError, match="folds must be >= 2"):
            dr_ate(
                torch.randn(20, 2),
                (torch.rand(20) > 0.5).float(),
                torch.randn(20),
                outcome_model=MockRegressor,
                propensity_model=MockClassifier,
                folds=1,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# dr_cate
# ═══════════════════════════════════════════════════════════════════════════════


class TestDRCATE:
    def test_returns_expected_keys(self) -> None:
        """Returns ATE + CATE estimates."""
        n = 80
        x = torch.randn(n, 2)
        t = (torch.rand(n) > 0.5).float()
        y = 2.0 * t + x[:, 0] + 0.1 * torch.randn(n)
        result = dr_cate(
            x,
            t,
            y,
            cate_model=MockRegressor,
            outcome_model=MockRegressor,
            propensity_model=MockClassifier,
            folds=2,
            alpha=0.05,
            seed=42,
        )
        assert "ate_estimate" in result
        assert "cate_hat" in result
        assert "pseudo_outcome" in result
        assert result["cate_hat"].shape == (80,)
        assert result["ate_ci_lower"] <= result["ate_ci_upper"]


# ═══════════════════════════════════════════════════════════════════════════════
# dr_policy_value
# ═══════════════════════════════════════════════════════════════════════════════


class TestDRPolicyValue:
    def test_returns_expected_keys(self) -> None:
        """Returns AIPW policy value estimate and SE."""
        n = 80
        x = torch.randn(n, 2)
        t = (torch.rand(n) > 0.5).float()
        y = 2.0 * t + 0.1 * torch.randn(n)
        policy = (torch.rand(n) > 0.5).float()
        result = dr_policy_value(
            x,
            t,
            y,
            policy=policy,
            outcome_model=MockRegressor,
            propensity_model=MockClassifier,
            folds=2,
            seed=42,
        )
        assert "estimate" in result
        assert "se" in result
        assert result["se"] >= 0
        assert result["n_samples"] == float(n)

    def test_mismatched_policy_shape_raises(self) -> None:
        """Policy with wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="share sample dimension"):
            dr_policy_value(
                torch.randn(20, 2),
                torch.randn(20),
                torch.randn(20),
                policy=torch.randn(10),
                outcome_model=MockRegressor,
                propensity_model=MockClassifier,
            )

    def test_policy_binarized(self) -> None:
        """Continuous policy is binarized at 0.5."""
        n = 40
        x = torch.randn(n, 2)
        t = (torch.rand(n) > 0.5).float()
        y = 2.0 * t + 0.1 * torch.randn(n)
        policy = torch.rand(n)  # continuous in [0,1]
        result = dr_policy_value(
            x,
            t,
            y,
            policy=policy,
            outcome_model=MockRegressor,
            propensity_model=MockClassifier,
            folds=2,
            seed=42,
        )
        assert "estimate" in result
