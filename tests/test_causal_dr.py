from __future__ import annotations

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
    assert "estimate" in out and "ci_low" in out and "ci_high" in out
    assert "ci_lower" in out and "ci_upper" in out
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
