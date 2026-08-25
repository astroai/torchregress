"""Regression tests for audit workstream C causal/inference fixes.

TR-CAU-01 (trimming applied to the DR estimator), TR-CAU-02 (fold-bootstrap
SE), TR-INF-02 (PPI++ lambda tuning and cross-fitting).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

# ═══════════════════════════════════════════════════════════════════════════════
# Shared tiny sklearn-free models for cross-fitting
# ═══════════════════════════════════════════════════════════════════════════════


class _LinReg:
    def fit(self, X, y):
        Xb = torch.cat([_as_f(X), torch.ones(len(X), 1)], dim=1)
        self.w, *_ = torch.linalg.lstsq(Xb, _as_f(y).reshape(-1, 1))
        return self

    def predict(self, X):
        Xb = torch.cat([_as_f(X), torch.ones(len(X), 1)], dim=1)
        return (Xb @ self.w).squeeze(-1).numpy()


class _PropLogit:
    def fit(self, X, t):
        X = _as_f(X)
        t = _as_f(t).reshape(-1)
        Xb = torch.cat([X, torch.ones(len(X), 1)], dim=1)
        w = torch.zeros(Xb.shape[1], requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=50)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xb @ w, t)
            loss.backward()
            return loss

        opt.step(closure)
        self.w = w.detach()
        return self

    def predict_proba(self, X):
        Xt = _as_f(X)
        Xb = torch.cat([Xt, torch.ones(len(Xt), 1)], dim=1)
        p = torch.sigmoid(Xb @ self.w)
        return torch.stack([1 - p, p], dim=1).numpy()


def _as_f(x) -> Tensor:
    return x if isinstance(x, Tensor) else torch.as_tensor(x, dtype=torch.float32)


def _strong_selection_data(n: int = 300, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    e = torch.sigmoid(2.5 * x[:, 0])
    t = (torch.rand(n, generator=g) < e).float()
    y = 1.0 * t + x[:, 0] + 0.3 * torch.randn(n, generator=g)
    return x, t, y


KW = dict(outcome_model=_LinReg, propensity_model=_PropLogit, folds=4, seed=3)


# ═══════════════════════════════════════════════════════════════════════════════
# TR-CAU-01: trimming changes the estimator when extreme propensities exist
# ═══════════════════════════════════════════════════════════════════════════════


class TestDRTrimmingAppliedToEstimator:
    def test_estimates_differ_across_trim_thresholds(self) -> None:
        from torchregress.causal.dr import dr_ate

        x, t, y = _strong_selection_data()
        loose = dr_ate(x, t, y, trim_threshold=0.05, **KW)
        tight = dr_ate(x, t, y, trim_threshold=0.45, **KW)
        assert tight["diagnostics"]["n_trimmed"] > loose["diagnostics"]["n_trimmed"]
        assert abs(loose["estimate"] - tight["estimate"]) > 1e-9, (
            "trimming must act on the estimator, not only the report"
        )

    def test_n_trimmed_surfaced_and_noop_when_all_kept(self) -> None:
        from torchregress.causal.dr import dr_ate

        x, t, y = _strong_selection_data()
        r = dr_ate(x, t, y, trim_threshold=0.01, **KW)
        assert isinstance(r["diagnostics"]["n_trimmed"], int)
        if r["diagnostics"]["n_trimmed"] == 0:
            assert r["diagnostics"]["trim_applied_to_estimator"] is False

    def test_cate_regression_uses_trimmed_scores(self) -> None:
        from torchregress.causal.dr import dr_cate

        x, t, y = _strong_selection_data()
        r = dr_cate(x, t, y, cate_model=_LinReg, trim_threshold=0.4, **KW)
        kept = r["pseudo_outcome_trimmed"]
        full = r["pseudo_outcome"]
        assert kept.numel() < full.numel()

    def test_fold_bootstrap_se_runs_and_is_positive(self) -> None:
        from torchregress.causal.dr import dr_ate

        x, t, y = _strong_selection_data()
        dr_ate(x, t, y, trim_threshold=0.05, se_method="analytic", **KW)
        boot = dr_ate(x, t, y, trim_threshold=0.05, se_method="fold_bootstrap", **KW)
        assert math.isfinite(boot["se"]) and boot["se"] > 0.0
        # same fold-pair bootstrap is deterministic under the same seed
        boot2 = dr_ate(x, t, y, trim_threshold=0.05, se_method="fold_bootstrap", **KW)
        assert boot["se"] == pytest.approx(boot2["se"])

    def test_unknown_se_method_raises(self) -> None:
        from torchregress.causal.dr import dr_ate

        x, t, y = _strong_selection_data()
        with pytest.raises(ValueError, match="Unsupported se_method"):
            dr_ate(x, t, y, se_method="magic", **KW)


# ═══════════════════════════════════════════════════════════════════════════════
# TR-INF-02: PPI++ power tuning and cross-fitting
# ═══════════════════════════════════════════════════════════════════════════════


class TestPPIPPTuning:
    def test_never_worse_than_classical_ppi(self) -> None:
        """lambda=1 lies inside the family and equals the classical estimator."""
        from torchregress.inference.ppi import ppi_mean_ci, ppi_pp_mean_ci

        torch.manual_seed(0)
        n, N = 60, 5000
        y = torch.randn(N)
        pred = y + 0.2 * torch.randn(N)
        pp = ppi_pp_mean_ci(y[:n], pred[:n], pred[n:])
        classical = ppi_mean_ci(y[:n], pred[:n], pred[n:])
        assert pp["se"] <= classical["se"] + 1e-9
        assert pp["ci_lower"] < 0.0 < pp["ci_upper"]

    def test_lambda_shrinks_with_useless_predictions(self) -> None:
        from torchregress.inference.ppi import ppi_pp_mean_ci

        torch.manual_seed(0)
        n, N = 60, 5000
        y = torch.randn(N)
        noisy_pred = y + 3.0 * torch.randn(N) + 5.0
        res = ppi_pp_mean_ci(y[:n], noisy_pred[:n], noisy_pred[n:])
        assert res["lambda"] < 1.0
        assert res["ci_lower"] < 0.0 < res["ci_upper"]

    def test_explicit_lambdas_respected(self) -> None:
        from torchregress.inference.ppi import ppi_pp_mean_ci

        torch.manual_seed(1)
        n, N = 40, 2000
        y = torch.randn(N)
        pred = y + 0.5 * torch.randn(N)
        res = ppi_pp_mean_ci(y[:n], pred[:n], pred[n:], lambdas=[1.0])
        classical_style = pred[n:].mean() + (y[:n] - pred[:n]).mean()
        assert res["lambda"] == 1.0
        assert res["estimate"] == pytest.approx(float(classical_style), abs=1e-5)

    def test_cross_fit_path_runs_and_covers(self) -> None:
        from torchregress.inference.ppi import ppi_pp_mean_ci

        torch.manual_seed(2)
        n, N = 80, 4000
        y = torch.randn(N)
        pred = 0.5 * y + 0.7 * torch.randn(N)  # miscalibrated -> calibration helps
        plain = ppi_pp_mean_ci(y[:n], pred[:n], pred[n:])
        crossed = ppi_pp_mean_ci(y[:n], pred[:n], pred[n:], cross_fits=4)
        assert crossed["cross_fits"] == 4
        assert crossed["ci_lower"] < 0.0 < crossed["ci_upper"]
        assert crossed["variance"] > 0.0
        del plain

    def test_validation_errors(self) -> None:
        from torchregress.inference.ppi import ppi_pp_mean_ci

        y = torch.randn(50)
        pred = y + 0.1
        with pytest.raises(ValueError, match="alpha"):
            ppi_pp_mean_ci(y, pred, torch.randn(10), alpha=0.0)
        with pytest.raises(ValueError, match="cross_fits"):
            ppi_pp_mean_ci(y, pred, torch.randn(10), cross_fits=-1)
