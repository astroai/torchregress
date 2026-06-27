from __future__ import annotations

import pytest
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression

from torchregress.causal import causal_overlap_report, dr_ate, dr_cate, dr_policy_value


def _sim_data(
    n: int = 800, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    torch.manual_seed(seed)
    x = torch.randn(n, 4)
    tau = 0.5 + 0.2 * torch.tanh(x[:, 0])
    p = torch.sigmoid(0.8 * x[:, 0] - 0.6 * x[:, 1]).clamp(0.05, 0.95)
    t = torch.bernoulli(p)
    y0 = 0.6 * x[:, 0] - 0.4 * x[:, 1] + 0.2 * x[:, 2] ** 2 + 0.3 * torch.randn(n)
    y = y0 + t * tau
    return x, t, y, float(tau.mean().item())


def test_causal_overlap_report_basic() -> None:
    x, t, _, _ = _sim_data(300, seed=1)
    p = torch.sigmoid(0.7 * x[:, 0] - 0.4 * x[:, 1]).clamp(0.05, 0.95)
    report = causal_overlap_report(p, t, trim_threshold=0.1)
    assert report["n_samples"] == 300.0
    assert 0.0 <= report["overlap_rate"] <= 1.0
    assert report["min_group_ess"] >= 0.0


def test_dr_ate_runs_and_returns_ci() -> None:
    x, t, y, _ = _sim_data(900, seed=2)
    out = dr_ate(
        x,
        t,
        y,
        outcome_model=LinearRegression,
        propensity_model=LogisticRegression(max_iter=1000),
        folds=3,
        alpha=0.05,
        seed=2,
    )
    assert "estimate" in out and "ci_lower" in out and "ci_upper" in out
    assert "ci_low" in out and "ci_high" in out
    assert out["ci_low"] == out["ci_lower"]
    assert out["ci_high"] == out["ci_upper"]
    assert out["ci_low"] <= out["estimate"] <= out["ci_high"]
    assert out["diagnostics"]["overlap_rate"] >= 0.0


def test_dr_cate_returns_vector_and_aggregate() -> None:
    x, t, y, _ = _sim_data(700, seed=3)
    out = dr_cate(
        x,
        t,
        y,
        cate_model=LinearRegression,
        outcome_model=LinearRegression,
        propensity_model=LogisticRegression(max_iter=1000),
        folds=3,
        alpha=0.05,
        seed=3,
    )
    cate_hat = out["cate_hat"]
    assert cate_hat.shape == (700,)
    assert out["ate_ci_low"] == out["ate_ci_lower"]
    assert out["ate_ci_high"] == out["ate_ci_upper"]
    assert out["ate_ci_low"] <= out["ate_estimate"] <= out["ate_ci_high"]


def test_dr_ate_reasonable_error_on_linear_data() -> None:
    x, t, y, true_ate = _sim_data(1200, seed=4)
    out = dr_ate(
        x,
        t,
        y,
        outcome_model=LinearRegression,
        propensity_model=LogisticRegression(max_iter=1000),
        folds=3,
        alpha=0.05,
        seed=4,
    )
    assert abs(float(out["estimate"]) - true_ate) < 0.2


def test_dr_policy_value_runs() -> None:
    x, t, y, _ = _sim_data(500, seed=5)
    policy = (x[:, 0] > 0).float()
    out = dr_policy_value(
        x,
        t,
        y,
        policy=policy,
        outcome_model=LinearRegression,
        propensity_model=LogisticRegression(max_iter=1000),
        folds=2,
        seed=5,
    )
    assert out["n_samples"] == 500.0
    assert out["se"] >= 0.0


def test_dr_ate_synthetic_exact() -> None:
    # 20 lines of synthetic data for a clean exact test
    torch.manual_seed(42)
    n = 1000
    x = torch.randn(n, 2)
    # Binary treatment assignment depends on x
    logits = x[:, 0] + x[:, 1]
    p = torch.sigmoid(logits)
    t = torch.bernoulli(p)

    # Simple linear outcomes with a constant treatment effect of 2.5
    true_ate = 2.5
    y0 = 1.0 + 2.0 * x[:, 0] - 1.5 * x[:, 1]
    y1 = y0 + true_ate
    y = torch.where(t > 0.5, y1, y0)

    # Ensure models are correctly specified
    out = dr_ate(
        x,
        t,
        y,
        outcome_model=LinearRegression,
        propensity_model=LogisticRegression(
            max_iter=1000,
            C=float("inf"),
        ),  # sklearn>=1.8: use C=inf instead of penalty=None for unregularized fit
        folds=2,
        seed=42,
    )

    # Since models are correctly specified and DR is unbiased, the estimate should be very close to true_ate
    assert abs(out["estimate"] - true_ate) < 0.15
    assert out["ci_low"] <= true_ate <= out["ci_high"]


def test_dr_ate_dimension_mismatch() -> None:
    x = torch.randn(100, 2)
    t = torch.randint(0, 2, (99,))
    y = torch.randn(100)
    with pytest.raises(ValueError, match="x, t, and y must share sample dimension"):
        dr_ate(
            x,
            t,
            y,
            outcome_model=LinearRegression,
            propensity_model=LogisticRegression(),
        )


def test_dr_ate_callable_and_instantiated_models() -> None:
    x, t, y, _ = _sim_data(200, seed=10)

    # Pass an instantiated model instead of a class
    instantiated_lr = LinearRegression()

    # Pass a factory lambda
    def factory_lr() -> LogisticRegression:
        return LogisticRegression(max_iter=1000)

    out = dr_ate(
        x,
        t,
        y,
        outcome_model=instantiated_lr,
        propensity_model=factory_lr,
        folds=2,
        seed=10,
    )
    assert out["estimate"] is not None


def test_dr_ate_missing_arms_in_fold() -> None:
    x = torch.randn(20, 2)
    y = torch.randn(20)

    # We need to force a ValueError by making t all zeros.
    t_all_zero = torch.zeros(20)
    with pytest.raises(
        ValueError, match="Each fold must contain both treatment arms for DR estimation"
    ):
        dr_ate(
            x,
            t_all_zero,
            y,
            outcome_model=LinearRegression,
            propensity_model=LogisticRegression(),
            folds=2,
            seed=0,
        )


def test_dr_ate_with_2d_x_and_callable_model_factory() -> None:
    x, t, y, _ = _sim_data(200, seed=6)

    # Pass a factory function instead of a class or an instance
    def make_lr():
        return LinearRegression()

    # Test `_as_2d` when it IS 1D by passing 1D x
    x_1d = x[:, 0]
    out = dr_ate(
        x_1d,
        t,
        y,
        outcome_model=make_lr,
        propensity_model=LogisticRegression(max_iter=100),
        folds=2,
    )
    assert "estimate" in out


class BadModelNoFit:
    pass


class BadModelNoPredict:
    def fit(self, x, y):
        pass


class BadModelNoPredictProba:
    def fit(self, x, y):
        pass


class BadModelPropensityFallback:
    def fit(self, x, y):
        pass

    def predict(self, x):
        import numpy as np

        return np.zeros(len(x))


class BadModelPredictProba1D:
    def fit(self, x, y):
        pass

    def predict_proba(self, x):
        import numpy as np

        return np.zeros(len(x))


def test_dr_exceptions() -> None:
    x, t, y, _ = _sim_data(100, seed=7)

    # TypeError in _fit_model
    with pytest.raises(TypeError, match="Model must implement fit"):
        dr_ate(x, t, y, outcome_model=BadModelNoFit, propensity_model=LogisticRegression)

    # TypeError in _predict_outcome
    with pytest.raises(TypeError, match="Outcome model must implement predict"):
        dr_ate(x, t, y, outcome_model=BadModelNoPredict, propensity_model=LogisticRegression)

    # TypeError in _predict_propensity
    with pytest.raises(
        TypeError, match="Propensity model must implement predict_proba.*or predict"
    ):
        dr_ate(x, t, y, outcome_model=LinearRegression, propensity_model=BadModelNoPredictProba)

    # Test _predict_propensity fallback to predict
    dr_ate(x, t, y, outcome_model=LinearRegression, propensity_model=BadModelPropensityFallback)

    # Test _predict_propensity with 1D predict_proba
    dr_ate(x, t, y, outcome_model=LinearRegression, propensity_model=BadModelPredictProba1D)

    # ValueError for folds < 2
    with pytest.raises(ValueError, match="folds must be >= 2"):
        dr_ate(
            x, t, y, outcome_model=LinearRegression, propensity_model=LogisticRegression, folds=1
        )

    # ValueError for fold with only one treatment arm
    t_all_treated = torch.ones_like(t)
    with pytest.raises(ValueError, match="Each fold must contain both treatment arms"):
        dr_ate(
            x, t_all_treated, y, outcome_model=LinearRegression, propensity_model=LogisticRegression
        )


def test_dr_shape_mismatches() -> None:
    x, t, y, _ = _sim_data(100, seed=8)

    # Mismatch in x, t, y
    with pytest.raises(ValueError, match="x, t, and y must share sample dimension"):
        dr_ate(x[:90], t, y, outcome_model=LinearRegression, propensity_model=LogisticRegression)

    with pytest.raises(ValueError, match="x, t, and y must share sample dimension"):
        dr_cate(
            x,
            t[:90],
            y,
            cate_model=LinearRegression,
            outcome_model=LinearRegression,
            propensity_model=LogisticRegression,
        )

    policy = torch.ones_like(t)
    with pytest.raises(ValueError, match="x, t, y, and policy must share sample dimension"):
        dr_policy_value(
            x,
            t,
            y,
            policy=policy[:90],
            outcome_model=LinearRegression,
            propensity_model=LogisticRegression,
        )


def test_internal_helpers() -> None:
    from torchregress.causal.dr import _as_1d, _as_2d, _build_model, _make_folds

    # Test _as_2d
    x_1d = torch.tensor([1, 2, 3])
    x_2d_expected = torch.tensor([[1], [2], [3]])
    assert torch.equal(_as_2d(x_1d), x_2d_expected)

    x_2d_input = torch.tensor([[1, 2], [3, 4]])
    assert torch.equal(_as_2d(x_2d_input), x_2d_input)

    # Test _as_1d
    x_2d_for_1d = torch.tensor([[1, 2], [3, 4]])
    x_1d_expected = torch.tensor([1, 2, 3, 4])
    assert torch.equal(_as_1d(x_2d_for_1d), x_1d_expected)

    # Test _build_model
    class DummyModel:
        def fit(self, x, y):
            pass

    # Factory returning instance
    def factory():
        return DummyModel()

    model1 = _build_model(factory)
    assert isinstance(model1, DummyModel)

    # Factory as class
    model2 = _build_model(DummyModel)
    assert isinstance(model2, DummyModel)

    # Existing instance
    existing_model = DummyModel()
    model3 = _build_model(existing_model)
    assert isinstance(model3, DummyModel)
    assert model3 is not existing_model  # should be deepcopy

    # Test _make_folds
    n = 10
    folds = 3
    seed = 42
    splits = _make_folds(n, folds, seed=seed)
    assert len(splits) == folds

    all_test_indices = []
    for train_idx, test_idx in splits:
        assert len(train_idx) + len(test_idx) == n
        # Check no overlap
        assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))
        all_test_indices.extend(test_idx.tolist())

    # Check all indices are present in exactly one test set
    assert sorted(all_test_indices) == list(range(n))
