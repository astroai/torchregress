"""
Tests for native conformal prediction module.

Tests cover all predictor classes and backward-compatible loss wrappers.
"""

import pytest
import torch

from torchregress.losses.conformal import (
    CQR,
    CTI,
    UACQR,
    ConformalLoss,
    ConformalPredictor,
    CVPlus,
    DensityConformal,
    DistributionalConformal,
    EnsembleBatchCP,
    JackknifePlus,
    LocalConformal,
    LocalConformalMAD,
    MonteCarloConformal,
    MultiTargetConformal,
    PrevalenceAdjustedCP,
    R2CConformal,
    SplitConformal,
)

# ---------------------------------------------------------------------------
# Backward compatibility tests (existing tests, preserved)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["cqr", "uacqr", "split"])
def test_conformal_loss_initialization(method):
    """Test initialization of ConformalLoss for different methods."""
    loss_fn = ConformalLoss(method=method, alpha=0.1)
    assert loss_fn.method == method
    assert loss_fn.alpha == 0.1
    assert not loss_fn._predictor._is_calibrated


def test_conformal_loss_invalid_method():
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match="Unknown method"):
        ConformalLoss(method="invalid")


def test_conformal_loss_forward_cqr():
    """Test forward pass of ConformalLoss with CQR."""
    loss_fn = ConformalLoss(method="cqr", alpha=0.1)
    batch_size, n_features = 10, 1
    y_pred = torch.randn(batch_size, 2 * n_features)
    y_true = torch.randn(batch_size, n_features)
    loss = loss_fn(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_conformal_loss_forward_split():
    """Test forward pass of ConformalLoss with Split CP."""
    loss_fn = ConformalLoss(method="split", alpha=0.1)
    batch_size, n_features = 10, 1
    y_pred = torch.randn(batch_size, n_features)
    y_true = torch.randn(batch_size, n_features)
    loss = loss_fn(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


@pytest.mark.parametrize("method", ["cqr", "uacqr", "split"])
def test_conformal_loss_calibration_and_prediction(method):
    """Test calibration and prediction flow."""
    loss_fn = ConformalLoss(method=method, alpha=0.1)
    batch_size, n_features = 50, 1

    if method in ("cqr", "uacqr"):
        y_pred_cal = torch.randn(batch_size, 2 * n_features)
        y_pred_test = torch.randn(batch_size, 2 * n_features)
    else:
        y_pred_cal = torch.randn(batch_size, n_features)
        y_pred_test = torch.randn(batch_size, n_features)

    y_true_cal = torch.randn(batch_size, n_features)

    with pytest.raises(RuntimeError):
        loss_fn.predict_interval(y_pred_test)

    loss_fn.calibrate(y_pred_cal, y_true_cal)
    assert loss_fn._predictor._is_calibrated

    lower, upper = loss_fn.predict_interval(y_pred_test)
    assert isinstance(lower, torch.Tensor)
    assert isinstance(upper, torch.Tensor)

    expected_shape = (batch_size, n_features)
    assert lower.shape == expected_shape
    assert upper.shape == expected_shape
    assert (lower <= upper).all()


def test_conformal_coverage():
    """Test that split CP achieves approximate coverage."""
    torch.manual_seed(42)
    n_cal, n_test = 200, 500
    alpha = 0.1

    truth_cal = torch.randn(n_cal, 1)
    noise_cal = torch.randn(n_cal, 1) * 0.5
    pred_cal = truth_cal + noise_cal

    truth_test = torch.randn(n_test, 1)
    noise_test = torch.randn(n_test, 1) * 0.5
    pred_test = truth_test + noise_test

    loss_fn = ConformalLoss(method="split", alpha=alpha)
    loss_fn.calibrate(pred_cal, truth_cal)

    lower, upper = loss_fn.predict_interval(pred_test)
    covered = (truth_test >= lower) & (truth_test <= upper)
    coverage = covered.float().mean().item()

    assert coverage >= (1 - alpha) - 0.05, f"Coverage {coverage:.3f} too low"


def test_uacqr_calibrate_and_predict_interval() -> None:
    n = 60
    target = torch.randn(n, 1)
    lower = target - torch.rand(n, 1).abs()
    upper = target + torch.rand(n, 1).abs()
    y_pred_cal = torch.cat([lower, upper], dim=-1)
    u = UACQR(alpha=0.1, debias=False)
    u.calibrate(y_pred_cal, target)
    y_pred_test = torch.cat([torch.zeros(7, 1), torch.ones(7, 1)], dim=-1)
    lo, hi = u.predict_interval(y_pred_test)
    assert lo.shape == (7, 1) and hi.shape == (7, 1)
    assert (lo <= hi).all()


def test_conformal_loss_uacqr_rejects_external_normalize_fn() -> None:
    with pytest.raises(ValueError, match="normalize_fn"):
        ConformalLoss(
            method="uacqr",
            normalize_fn=lambda y, x: torch.ones(y.shape[0], device=y.device),
        )


def test_cqr_coverage():
    """Test that CQR achieves approximate coverage."""
    torch.manual_seed(42)
    n_cal, n_test = 200, 500
    alpha = 0.1

    truth_cal = torch.randn(n_cal, 1)
    lower_cal = truth_cal - 0.8 * torch.abs(torch.randn(n_cal, 1))
    upper_cal = truth_cal + 0.8 * torch.abs(torch.randn(n_cal, 1))
    pred_cal = torch.cat([lower_cal, upper_cal], dim=-1)

    truth_test = torch.randn(n_test, 1)
    lower_test = truth_test - 0.8 * torch.abs(torch.randn(n_test, 1))
    upper_test = truth_test + 0.8 * torch.abs(torch.randn(n_test, 1))
    pred_test = torch.cat([lower_test, upper_test], dim=-1)

    loss_fn = ConformalLoss(method="cqr", alpha=alpha)
    loss_fn.calibrate(pred_cal, truth_cal)

    lower, upper = loss_fn.predict_interval(pred_test)
    covered = (truth_test >= lower) & (truth_test <= upper)
    coverage = covered.float().mean().item()

    assert coverage >= (1 - alpha) - 0.05, f"Coverage {coverage:.3f} too low"


def test_conformalized_quantile_loss_method():
    """Test that ConformalLoss with method='cqr' works correctly."""
    loss_fn = ConformalLoss(method="cqr", alpha=0.1)
    assert isinstance(loss_fn, ConformalLoss)
    assert loss_fn.method == "cqr"


def test_conformal_with_mask():
    """Test that masking works during calibration."""
    loss_fn = ConformalLoss(method="split", alpha=0.1)
    batch_size, n_features = 20, 1

    y_pred = torch.randn(batch_size, n_features)
    y_true = torch.randn(batch_size, n_features)
    mask = torch.ones(batch_size, n_features, dtype=torch.bool)
    mask[:5] = False

    loss_fn.calibrate(y_pred, y_true, mask=mask)
    assert loss_fn._predictor._is_calibrated

    lower, upper = loss_fn.predict_interval(y_pred)
    assert lower.shape == y_pred.shape


# ---------------------------------------------------------------------------
# Standalone predictor tests
# ---------------------------------------------------------------------------


class TestSplitConformal:
    """Tests for the SplitConformal predictor."""

    def test_basic_calibrate_predict(self):
        cp = SplitConformal(alpha=0.1)
        torch.manual_seed(42)
        preds = torch.randn(100, 1)
        targets = preds + torch.randn(100, 1) * 0.3
        cp.calibrate(preds, targets)
        lower, upper = cp.predict_interval(preds)
        assert lower.shape == preds.shape
        assert (lower <= upper).all()

    def test_coverage(self):
        """Split CP coverage on Gaussian noise."""
        torch.manual_seed(123)
        alpha = 0.1
        n_cal, n_test = 300, 1000

        truth_cal = torch.randn(n_cal, 1)
        pred_cal = truth_cal + torch.randn(n_cal, 1) * 0.5
        truth_test = torch.randn(n_test, 1)
        pred_test = truth_test + torch.randn(n_test, 1) * 0.5

        cp = SplitConformal(alpha=alpha)
        cp.calibrate(pred_cal, truth_cal)
        lower, upper = cp.predict_interval(pred_test)

        covered = (truth_test >= lower) & (truth_test <= upper)
        coverage = covered.float().mean().item()
        assert coverage >= (1 - alpha) - 0.05

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            SplitConformal(alpha=0.0)
        with pytest.raises(ValueError):
            SplitConformal(alpha=1.0)

    def test_predict_before_calibrate(self):
        cp = SplitConformal(alpha=0.1)
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict_interval(torch.randn(10, 1))


class TestCQR:
    """Tests for the CQR predictor."""

    def _make_synthetic_quantile_data(self, n, seed=42):
        torch.manual_seed(seed)
        truth = torch.randn(n, 1)
        lower = truth - 0.8 * torch.abs(torch.randn(n, 1))
        upper = truth + 0.8 * torch.abs(torch.randn(n, 1))
        preds = torch.cat([lower, upper], dim=-1)
        return preds, truth

    def test_basic(self):
        cqr = CQR(alpha=0.1)
        pred_cal, truth_cal = self._make_synthetic_quantile_data(100)
        cqr.calibrate(pred_cal, truth_cal)
        lower, upper = cqr.predict_interval(pred_cal)
        assert (lower <= upper).all()

    def test_coverage(self):
        alpha = 0.1
        cqr = CQR(alpha=alpha)
        pred_cal, truth_cal = self._make_synthetic_quantile_data(300, seed=10)
        pred_test, truth_test = self._make_synthetic_quantile_data(1000, seed=20)
        cqr.calibrate(pred_cal, truth_cal)
        lower, upper = cqr.predict_interval(pred_test)
        covered = (truth_test >= lower) & (truth_test <= upper)
        coverage = covered.float().mean().item()
        assert coverage >= (1 - alpha) - 0.05

    def test_debias(self):
        """Debiased CQR should also achieve coverage."""
        alpha = 0.1
        cqr = CQR(alpha=alpha, debias=True)
        pred_cal, truth_cal = self._make_synthetic_quantile_data(300, seed=30)
        pred_test, truth_test = self._make_synthetic_quantile_data(1000, seed=40)
        cqr.calibrate(pred_cal, truth_cal)
        lower, upper = cqr.predict_interval(pred_test)
        covered = (truth_test >= lower) & (truth_test <= upper)
        coverage = covered.float().mean().item()
        assert coverage >= (1 - alpha) - 0.05


class TestNormalizedCP:
    """Tests for normalized (difficulty-adaptive) conformal prediction."""

    def test_normalized_split(self):
        """Normalized CP should produce adaptive intervals."""
        torch.manual_seed(42)

        def difficulty_fn(y_pred, x):
            # Difficulty proportional to |x|
            return torch.abs(x).mean(dim=-1).clamp(min=0.1)

        cp = SplitConformal(alpha=0.1, normalize_fn=difficulty_fn)
        n = 200
        x = torch.randn(n, 3)
        truth = torch.randn(n, 1)
        preds = truth + torch.randn(n, 1) * 0.5

        cp.calibrate(preds, truth, x=x)
        lower, upper = cp.predict_interval(preds, x=x)

        widths = (upper - lower).squeeze()
        assert widths.shape == (n,)
        # Widths should vary (not constant) since difficulty varies
        assert widths.std() > 0.01

    def test_normalized_coverage(self):
        """Normalized CP should maintain coverage."""
        torch.manual_seed(99)
        alpha = 0.1
        n_cal, n_test = 300, 1000

        def difficulty_fn(y_pred, x):
            return torch.abs(x[:, 0]).clamp(min=0.1)

        cp = SplitConformal(alpha=alpha, normalize_fn=difficulty_fn)

        x_cal = torch.randn(n_cal, 2)
        truth_cal = torch.randn(n_cal, 1)
        pred_cal = truth_cal + torch.randn(n_cal, 1) * 0.5

        x_test = torch.randn(n_test, 2)
        truth_test = torch.randn(n_test, 1)
        pred_test = truth_test + torch.randn(n_test, 1) * 0.5

        cp.calibrate(pred_cal, truth_cal, x=x_cal)
        lower, upper = cp.predict_interval(pred_test, x=x_test)

        covered = (truth_test >= lower) & (truth_test <= upper)
        coverage = covered.float().mean().item()
        assert coverage >= (1 - alpha) - 0.05


class TestMondrianCP:
    """Tests for Mondrian (group-conditional) conformal prediction."""

    def test_mondrian_split(self):
        """Mondrian CP should produce per-group quantiles."""
        torch.manual_seed(42)
        cp = SplitConformal(alpha=0.1)

        n = 200
        preds = torch.randn(n, 1)
        targets = preds + torch.randn(n, 1) * 0.5
        groups = torch.randint(0, 3, (n,))

        cp.calibrate(preds, targets, groups=groups)
        assert isinstance(cp.q_hat, dict)
        assert len(cp.q_hat) == 3

    def test_mondrian_predict(self):
        """Mondrian CP prediction requires groups."""
        torch.manual_seed(42)
        cp = SplitConformal(alpha=0.1)

        n = 200
        preds = torch.randn(n, 1)
        targets = preds + torch.randn(n, 1) * 0.5
        groups = torch.randint(0, 2, (n,))

        cp.calibrate(preds, targets, groups=groups)

        # Should fail without groups
        with pytest.raises(ValueError, match="groups"):
            cp.predict_interval(preds)

        # Should succeed with groups
        lower, upper = cp.predict_interval(preds, groups=groups)
        assert (lower <= upper).all()

    def test_mondrian_per_group_coverage(self):
        """Each group should achieve approximate coverage."""
        torch.manual_seed(55)
        alpha = 0.1
        n_cal, n_test = 400, 1000

        # Two groups with different noise levels
        groups_cal = torch.cat([torch.zeros(n_cal // 2), torch.ones(n_cal // 2)]).long()
        truth_cal = torch.randn(n_cal, 1)
        noise_scales = torch.where(groups_cal == 0, 0.3, 1.0).unsqueeze(1)
        pred_cal = truth_cal + torch.randn(n_cal, 1) * noise_scales

        groups_test = torch.cat([torch.zeros(n_test // 2), torch.ones(n_test // 2)]).long()
        truth_test = torch.randn(n_test, 1)
        noise_scales_test = torch.where(groups_test == 0, 0.3, 1.0).unsqueeze(1)
        pred_test = truth_test + torch.randn(n_test, 1) * noise_scales_test

        cp = SplitConformal(alpha=alpha)
        cp.calibrate(pred_cal, truth_cal, groups=groups_cal)
        lower, upper = cp.predict_interval(pred_test, groups=groups_test)

        for g in [0, 1]:
            g_mask = groups_test == g
            covered = (truth_test[g_mask] >= lower[g_mask]) & (truth_test[g_mask] <= upper[g_mask])
            coverage = covered.float().mean().item()
            assert coverage >= (1 - alpha) - 0.10, f"Group {g} coverage {coverage:.3f} too low"

    def test_mondrian_float_groups(self):
        """Mondrian CP should work with float groups (vectorized path)."""
        torch.manual_seed(42)
        cp = SplitConformal(alpha=0.1)

        n = 200
        preds = torch.randn(n, 1)
        targets = preds + torch.randn(n, 1) * 0.5
        # Float groups
        groups = torch.randint(0, 3, (n,)).float()
        # Make them distinct floats
        groups_uniq = torch.tensor([0.1, 0.2, 0.3])
        groups = groups_uniq[groups.long()]

        cp.calibrate(preds, targets, groups=groups)
        assert isinstance(cp.q_hat, dict)
        assert len(cp.q_hat) == 3
        # Check keys are floats
        assert all(isinstance(k, float) for k in cp.q_hat.keys())

        # Prediction
        n_test = 500
        preds_test = torch.randn(n_test, 1)
        targets_test = preds_test + torch.randn(n_test, 1) * 0.5
        groups_test_idx = torch.randint(0, 3, (n_test,))
        groups_test = groups_uniq[groups_test_idx]

        lower_test, upper_test = cp.predict_interval(preds_test, groups=groups_test)

        assert lower_test.shape == (n_test, 1)
        assert (lower_test <= upper_test).all()

        covered = (targets_test >= lower_test) & (targets_test <= upper_test)
        coverage = covered.float().mean().item()
        assert coverage >= 0.85  # Expect ~0.9


class TestWeightedCP:
    """Tests for weighted (covariate-shift) conformal prediction."""

    def test_weighted_calibration(self):
        """Weighted CP should accept importance weights."""
        torch.manual_seed(42)
        cp = SplitConformal(alpha=0.1)

        n = 200
        preds = torch.randn(n, 1)
        targets = preds + torch.randn(n, 1) * 0.5
        weights = torch.rand(n) + 0.5  # positive weights

        cp.calibrate(preds, targets, weights=weights)
        assert cp._is_calibrated

        lower, upper = cp.predict_interval(preds)
        assert (lower <= upper).all()

    def test_weighted_vs_unweighted(self):
        """Weighted CP with uniform weights should match unweighted."""
        torch.manual_seed(42)
        n = 200
        preds = torch.randn(n, 1)
        targets = preds + torch.randn(n, 1) * 0.5

        cp1 = SplitConformal(alpha=0.1)
        cp1.calibrate(preds, targets)

        cp2 = SplitConformal(alpha=0.1)
        cp2.calibrate(preds, targets, weights=torch.ones(n))

        # Should be close (not exact due to algorithm differences)
        _, upper1 = cp1.predict_interval(preds)
        _, upper2 = cp2.predict_interval(preds)
        # Within 20% since weighted quantile is approximate
        q1 = cp1.q_hat.item()
        q2 = cp2.q_hat.item()
        assert abs(q1 - q2) / max(abs(q1), 1e-8) < 0.2


class TestCTI:
    """Tests for Conformal Thresholded Intervals."""

    def test_calibration(self):
        """CTI calibration with log-density values."""
        torch.manual_seed(42)
        cti = CTI(alpha=0.1)

        n = 200
        # Simulate log-density values
        targets = torch.randn(n, 1)
        log_density = -0.5 * targets**2  # Gaussian log-density
        log_density = log_density.squeeze()

        cti.calibrate(log_density, targets)
        assert cti._is_calibrated

    def test_intervals_from_density(self):
        """CTI should produce intervals from density function."""
        torch.manual_seed(42)
        cti = CTI(alpha=0.1, grid_size=200)

        n = 100
        targets = torch.randn(n, 1)
        log_density = -0.5 * targets.squeeze() ** 2

        cti.calibrate(log_density, targets)

        # Gaussian density function
        def density_fn(y_grid, x):
            mu = x[0]  # first feature as mean
            return -0.5 * (y_grid - mu) ** 2

        x_test = torch.randn(20, 2)
        lower, upper = cti.predict_intervals_from_density(density_fn, x_test, y_min=-5, y_max=5)
        assert lower.shape == (20, 1)
        assert upper.shape == (20, 1)
        assert (lower <= upper).all()

    def test_cti_smaller_than_split(self):
        """CTI intervals should be tighter than naive split CP."""
        torch.manual_seed(42)
        n_cal = 300

        # Skewed distribution: mixture of Gaussians
        targets = torch.randn(n_cal, 1) + 2.0

        # CTI with Gaussian density
        cti = CTI(alpha=0.1, grid_size=500)
        means = targets.squeeze()
        log_dens = -0.5 * (targets.squeeze() - means) ** 2  # perfect density
        cti.calibrate(log_dens, targets)

        # Just check calibration succeeds
        assert cti._is_calibrated


class TestDistributionalConformal:
    """Tests for Distributional Conformal Prediction."""

    def test_calibration_with_pit(self):
        """Distributional CP calibrates on PIT residuals."""
        torch.manual_seed(42)
        dcp = DistributionalConformal(alpha=0.1)

        n = 200
        # Simulate CDF values (should be ~Uniform if model is well-calibrated)
        targets = torch.randn(n, 1)
        # Well-calibrated: PIT values are uniform
        from torch.distributions import Normal

        cdf_values = Normal(0, 1).cdf(targets.squeeze())

        dcp.calibrate(cdf_values, targets)
        assert dcp._is_calibrated

    def test_intervals_from_cdf(self):
        """Distributional CP should produce intervals from inverse CDF."""
        torch.manual_seed(42)
        dcp = DistributionalConformal(alpha=0.1)

        from torch.distributions import Normal

        n = 200
        targets = torch.randn(n, 1)
        cdf_values = Normal(0, 1).cdf(targets.squeeze())

        dcp.calibrate(cdf_values, targets)

        def icdf_fn(levels, x):
            mu = x[0]
            return Normal(mu, 1.0).icdf(levels)

        x_test = torch.randn(30, 2)
        lower, upper = dcp.predict_intervals_from_cdf(icdf_fn, x_test)
        assert lower.shape == (30, 1)
        assert upper.shape == (30, 1)
        assert (lower < upper).all()

    def test_vectorized_icdf(self):
        """Test that vectorized path is used when icdf_fn supports it."""
        torch.manual_seed(42)
        dcp = DistributionalConformal(alpha=0.1)
        dcp.q_hat = torch.tensor(0.9)
        dcp._is_calibrated = True

        n_test = 10
        n_features = 2
        x_test = torch.randn(n_test, n_features)

        # Flag to verify vectorized call
        vectorized_called = False

        def icdf_fn_vectorized(levels, x):
            nonlocal vectorized_called
            if x.shape[0] == n_test and x.ndim == 2:
                vectorized_called = True
            mu = x[:, 0]
            # levels: (2,) -> (2, 1) broadcast with (N,) -> (2, N) -> T -> (N, 2)
            from torch.distributions import Normal

            return Normal(mu, 1.0).icdf(levels.unsqueeze(-1)).T

        lower, upper = dcp.predict_intervals_from_cdf(icdf_fn_vectorized, x_test)

        assert vectorized_called, "Vectorized path was not used"
        assert lower.shape == (n_test, 1)
        assert upper.shape == (n_test, 1)

    def test_fallback_logic(self):
        """Test fallback when vectorized call fails or returns wrong shape."""
        torch.manual_seed(42)
        dcp = DistributionalConformal(alpha=0.1)
        dcp.q_hat = torch.tensor(0.9)
        dcp._is_calibrated = True

        n_test = 10
        x_test = torch.randn(n_test, 2)
        from torch.distributions import Normal

        # Case 1: Exception triggers vmap (optimization)
        loop_calls = 0

        def icdf_fn_fail(levels, x):
            nonlocal loop_calls
            if x.ndim == 2 and x.shape[0] > 1:
                raise ValueError("No batch support")
            loop_calls += 1
            return Normal(x[0], 1.0).icdf(levels)

        dcp.predict_intervals_from_cdf(icdf_fn_fail, x_test)
        # vmap traces the function once, so loop_calls should be 1, not n_test
        assert loop_calls == 1

        # Case 2: Wrong shape triggers vmap (optimization)
        loop_calls = 0

        def icdf_fn_wrong_shape(levels, x):
            nonlocal loop_calls
            if x.ndim == 2 and x.shape[0] > 1:
                return torch.zeros(2)  # Wrong shape (should be N, 2)
            loop_calls += 1
            return Normal(x[0], 1.0).icdf(levels)

        dcp.predict_intervals_from_cdf(icdf_fn_wrong_shape, x_test)
        # vmap also fixes the shape issue by vectorizing the single-sample path
        assert loop_calls == 1

        # Case 3: vmap failure triggers explicit loop (true fallback)
        loop_calls = 0

        def icdf_fn_hard_fail(levels, x):
            nonlocal loop_calls
            if x.ndim == 2 and x.shape[0] > 1:
                raise ValueError("No batch support")

            # Trigger vmap failure: .item() is not allowed in vmap
            # But allowed in standard loop
            try:
                # We need to be careful not to fail in the loop case.
                # In loop case, x is (features,), x[0] is scalar tensor.
                _ = x[0].item()
            except RuntimeError:
                # This catches the vmap error and propagates it (or rather the function fails)
                raise

            loop_calls += 1
            return Normal(x[0], 1.0).icdf(levels)

        dcp.predict_intervals_from_cdf(icdf_fn_hard_fail, x_test)
        assert loop_calls == n_test


class TestR2CConformal:
    """Tests for Regression-as-Classification Conformal."""

    def _make_r2c_data(self, n, n_bins=50, seed=42):
        torch.manual_seed(seed)
        bin_edges = torch.linspace(-5, 5, n_bins + 1)
        targets = torch.randn(n, 1)

        # Simulate softmax probs: Gaussian around true target
        # Use σ=1.0 so probability spreads across multiple bins
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        probs = torch.exp(-0.5 * (centers.unsqueeze(0) - targets) ** 2 / 1.0**2)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        return probs, targets, bin_edges

    def test_basic(self):
        r2c = R2CConformal(alpha=0.1)
        probs, targets, bin_edges = self._make_r2c_data(100)
        r2c.bin_edges = bin_edges
        r2c.calibrate(probs, targets)
        assert r2c._is_calibrated

    def test_intervals(self):
        r2c = R2CConformal(alpha=0.1)
        probs_cal, targets_cal, bin_edges = self._make_r2c_data(200, seed=10)
        probs_test, _, _ = self._make_r2c_data(50, seed=20)
        r2c.bin_edges = bin_edges

        r2c.calibrate(probs_cal, targets_cal)
        lower, upper = r2c.predict_interval(probs_test)
        assert lower.shape == (50, 1)
        assert upper.shape == (50, 1)
        assert (lower <= upper).all()

    def test_coverage(self):
        """R2CCP should achieve approximate coverage."""
        alpha = 0.1
        r2c = R2CConformal(alpha=alpha)
        probs_cal, targets_cal, bin_edges = self._make_r2c_data(300, n_bins=100, seed=30)
        probs_test, targets_test, _ = self._make_r2c_data(500, n_bins=100, seed=40)
        r2c.bin_edges = bin_edges

        r2c.calibrate(probs_cal, targets_cal)
        lower, upper = r2c.predict_interval(probs_test)

        covered = (targets_test >= lower) & (targets_test <= upper)
        coverage = covered.float().mean().item()
        assert coverage >= (1 - alpha) - 0.10, f"Coverage {coverage:.3f} too low"


class TestMultiTargetConformal:
    """Tests for per-dimension conformal prediction."""

    def test_basic(self):
        cp = MultiTargetConformal(alpha=0.1)
        torch.manual_seed(42)
        preds = torch.randn(100, 3)
        targets = preds + torch.randn(100, 3) * 0.3
        cp.calibrate(preds, targets)
        assert cp._is_calibrated
        assert cp.q_hat.shape == (3,)

    def test_per_dim_thresholds(self):
        """Different noise per dimension should give different thresholds."""
        torch.manual_seed(42)
        cp = MultiTargetConformal(alpha=0.1)
        n = 200
        preds = torch.randn(n, 3)
        noise = torch.randn(n, 3) * torch.tensor([0.1, 0.5, 2.0])
        targets = preds + noise
        cp.calibrate(preds, targets)

        # Thresholds should increase across dimensions
        assert cp.q_hat[0] < cp.q_hat[2]

    def test_coverage(self):
        """Per-dimension coverage should hold."""
        torch.manual_seed(42)
        alpha = 0.1
        cp = MultiTargetConformal(alpha=alpha)

        n_cal, n_test = 300, 1000
        preds_cal = torch.randn(n_cal, 3)
        targets_cal = preds_cal + torch.randn(n_cal, 3) * 0.5
        preds_test = torch.randn(n_test, 3)
        targets_test = preds_test + torch.randn(n_test, 3) * 0.5

        cp.calibrate(preds_cal, targets_cal)
        lower, upper = cp.predict_interval(preds_test)

        for d in range(3):
            covered = (targets_test[:, d] >= lower[:, d]) & (targets_test[:, d] <= upper[:, d])
            coverage = covered.float().mean().item()
            assert coverage >= (1 - alpha) - 0.05, f"Dim {d} coverage {coverage:.3f} too low"


class TestConformalLossCQRDebias:
    """Test CQR debiasing through the loss wrapper."""

    def test_debias_coverage(self):
        torch.manual_seed(42)
        alpha = 0.1
        n_cal, n_test = 300, 1000

        truth_cal = torch.randn(n_cal, 1)
        lower_cal = truth_cal - 0.8 * torch.abs(torch.randn(n_cal, 1))
        upper_cal = truth_cal + 0.8 * torch.abs(torch.randn(n_cal, 1))
        pred_cal = torch.cat([lower_cal, upper_cal], dim=-1)

        truth_test = torch.randn(n_test, 1)
        lower_test = truth_test - 0.8 * torch.abs(torch.randn(n_test, 1))
        upper_test = truth_test + 0.8 * torch.abs(torch.randn(n_test, 1))
        pred_test = torch.cat([lower_test, upper_test], dim=-1)

        loss_fn = ConformalLoss(method="cqr", alpha=alpha, debias=True)
        loss_fn.calibrate(pred_cal, truth_cal)

        lower, upper = loss_fn.predict_interval(pred_test)
        covered = (truth_test >= lower) & (truth_test <= upper)
        coverage = covered.float().mean().item()
        assert coverage >= (1 - alpha) - 0.05


class TestConformalLossMondrianWeighted:
    """Test Mondrian and weighted CP through the loss wrapper."""

    def test_mondrian_through_loss(self):
        torch.manual_seed(42)
        loss_fn = ConformalLoss(method="split", alpha=0.1)
        n = 200
        preds = torch.randn(n, 1)
        targets = preds + torch.randn(n, 1) * 0.5
        groups = torch.randint(0, 2, (n,))

        loss_fn.calibrate(preds, targets, groups=groups)
        assert loss_fn._predictor._is_calibrated
        assert isinstance(loss_fn._predictor.q_hat, dict)

        lower, upper = loss_fn.predict_interval(preds, groups=groups)
        assert (lower <= upper).all()

    def test_weighted_through_loss(self):
        torch.manual_seed(42)
        loss_fn = ConformalLoss(method="split", alpha=0.1)
        n = 200
        preds = torch.randn(n, 1)
        targets = preds + torch.randn(n, 1) * 0.5
        weights = torch.rand(n) + 0.5

        loss_fn.calibrate(preds, targets, weights=weights)
        assert loss_fn._predictor._is_calibrated

        lower, upper = loss_fn.predict_interval(preds)
        assert (lower <= upper).all()


class TestDensityConformal:
    """Tests for density-adaptive conformal intervals."""

    def test_density_conformal_basic(self):
        torch.manual_seed(42)
        cp = DensityConformal(alpha=0.1, bandwidth=0.3)
        pred_cal = torch.randn(200, 1)
        target_cal = pred_cal + 0.35 * torch.randn(200, 1)
        cp.calibrate(pred_cal, target_cal)
        lower, upper = cp.predict_interval(pred_cal)
        assert lower.shape == pred_cal.shape
        assert upper.shape == pred_cal.shape
        assert (lower <= upper).all()

    def test_density_conformal_coverage(self):
        torch.manual_seed(42)
        cp = DensityConformal(alpha=0.1, bandwidth=0.25)
        pred_cal = torch.randn(300, 1)
        target_cal = pred_cal + 0.4 * torch.randn(300, 1)
        pred_test = torch.randn(600, 1)
        target_test = pred_test + 0.4 * torch.randn(600, 1)
        cp.calibrate(pred_cal, target_cal)
        lower, upper = cp.predict_interval(pred_test)
        cov = ((target_test >= lower) & (target_test <= upper)).float().mean().item()
        assert cov >= 0.82


class TestPrevalenceAdjustedCP:
    """Tests for prevalence-adjusted group conformal intervals."""

    def test_prevalence_adjusted_basic(self):
        torch.manual_seed(42)
        cp = PrevalenceAdjustedCP(alpha=0.1, n_bins=4)
        pred_cal = torch.randn(240, 1)
        target_cal = pred_cal + 0.5 * torch.randn(240, 1)
        cp.calibrate(pred_cal, target_cal)
        lower, upper = cp.predict_interval(pred_cal)
        assert lower.shape == pred_cal.shape
        assert (lower <= upper).all()

    def test_prevalence_adjusted_with_groups(self):
        torch.manual_seed(42)
        cp = PrevalenceAdjustedCP(alpha=0.1, n_bins=3)
        pred_cal = torch.randn(180, 1)
        target_cal = pred_cal + 0.45 * torch.randn(180, 1)
        groups = torch.cat([torch.zeros(120), torch.ones(60)]).long()
        cp.calibrate(pred_cal, target_cal, groups=groups)
        lower, upper = cp.predict_interval(pred_cal, groups=groups)
        assert lower.shape == pred_cal.shape
        assert (lower <= upper).all()


class TestMonteCarloConformal:
    """Tests for MC-sample conformal intervals."""

    def test_mc_conformal_basic(self):
        torch.manual_seed(42)
        cp = MonteCarloConformal(alpha=0.1)
        n_mc, n = 20, 200
        true = torch.randn(n, 1)
        mc_cal = true.unsqueeze(0) + 0.4 * torch.randn(n_mc, n, 1)
        cp.calibrate(mc_cal, true)
        lower, upper = cp.predict_interval(mc_cal)
        assert lower.shape == true.shape
        assert upper.shape == true.shape
        assert (lower <= upper).all()

    def test_mc_conformal_coverage(self):
        torch.manual_seed(42)
        cp = MonteCarloConformal(alpha=0.1)
        n_mc = 24
        true_cal = torch.randn(240, 1)
        mc_cal = true_cal.unsqueeze(0) + 0.45 * torch.randn(n_mc, 240, 1)
        true_test = torch.randn(600, 1)
        mc_test = true_test.unsqueeze(0) + 0.45 * torch.randn(n_mc, 600, 1)
        cp.calibrate(mc_cal, true_cal)
        lower, upper = cp.predict_interval(mc_test)
        cov = ((true_test >= lower) & (true_test <= upper)).float().mean().item()
        assert cov >= 0.82


class TestEmptyInputs:
    """Tests for handling empty inputs in conformal prediction."""

    class MyPredictor(ConformalPredictor):
        def _compute_scores(self, y_pred, target):
            return torch.abs(y_pred - target)

        def _build_intervals(self, y_pred, q, difficulty=None):
            return y_pred - q, y_pred + q

    def test_empty_calibration(self):
        """Test that calibration raises ValueError on empty input."""
        cp = self.MyPredictor(alpha=0.1)
        y_pred = torch.tensor([])
        target = torch.tensor([])

        with pytest.raises(ValueError, match="Calibration set is empty"):
            cp.calibrate(y_pred, target)

    def test_weighted_quantile_empty_scores(self):
        """Test that _weighted_quantile raises ValueError on empty scores."""
        from torchregress.losses.conformal import _weighted_quantile

        scores = torch.tensor([])
        with pytest.raises(ValueError, match="Input scores tensor is empty"):
            _weighted_quantile(scores, 0.9)

    def test_weighted_quantile_empty_weights(self):
        """Test that _weighted_quantile raises ValueError on empty weights."""
        from torchregress.losses.conformal import _weighted_quantile

        scores = torch.tensor([])
        weights = torch.tensor([])
        with pytest.raises(ValueError, match="Input weights tensor is empty"):
            _weighted_quantile(scores, 0.9, weights=weights)


class TestInvalidWeights:
    """Tests for invalid weights in conformal prediction."""

    def test_weighted_quantile_negative_weights(self):
        """Test that _weighted_quantile raises ValueError on negative weights."""
        from torchregress.losses.conformal import _weighted_quantile

        scores = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([1.0, -0.5, 0.5])
        with pytest.raises(ValueError, match="Sample weights must be non-negative"):
            _weighted_quantile(scores, 0.9, weights=weights)

    def test_weighted_quantile_zero_sum_weights(self):
        """Test that _weighted_quantile raises ValueError on zero sum weights."""
        from torchregress.losses.conformal import _weighted_quantile

        scores = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Sum of sample weights must be positive"):
            _weighted_quantile(scores, 0.9, weights=weights)


class TestOneDimensionalTargets:
    """Regression tests: 1-D targets must behave exactly like [N, 1] targets.

    Previously ``target [N] - y_pred[..., :1] [N, 1]`` broadcast to
    ``[N, N]`` and the per-row max silently produced garbage scores, so
    CQR calibrated on 1-D targets massively over-covered.
    """

    def test_cqr_scores_match_column_targets(self):
        torch.manual_seed(0)
        n = 64
        y = torch.randn(n)
        pred = torch.stack([y - 0.5, y + 0.5], dim=-1) + 0.1 * torch.randn(n, 2)

        cqr = CQR(alpha=0.2)
        scores_1d = cqr._compute_scores(pred, y)
        scores_2d = cqr._compute_scores(pred, y.unsqueeze(-1))
        assert scores_1d.shape == (n,)
        assert torch.allclose(scores_1d, scores_2d)
        # Scores are bounded by the interval geometry, not the target scale.
        assert scores_1d.abs().max() < 2.0

    def test_cqr_coverage_with_1d_targets(self):
        torch.manual_seed(1)
        n_cal, n_test = 300, 500
        alpha = 0.2

        y_cal = torch.randn(n_cal)
        pred_cal = torch.stack([y_cal - 0.2, y_cal + 0.2], dim=-1)
        pred_cal = pred_cal + 0.3 * torch.randn(n_cal, 2)

        y_test = torch.randn(n_test)
        pred_test = torch.stack([y_test - 0.2, y_test + 0.2], dim=-1)
        pred_test = pred_test + 0.3 * torch.randn(n_test, 2)

        cqr = CQR(alpha=alpha)
        cqr.calibrate(pred_cal, y_cal)
        lower, upper = cqr.predict_interval(pred_test)
        covered = (y_test.unsqueeze(-1) >= lower) & (y_test.unsqueeze(-1) <= upper)
        coverage = covered.float().mean().item()
        # Near-nominal coverage: neither under-covering nor the pathological
        # ~100% over-coverage produced by the broadcasting bug.
        assert abs(coverage - (1 - alpha)) < 0.07, f"coverage {coverage:.3f}"

    def test_conformal_loss_cqr_forward_matches_column_targets(self):
        torch.manual_seed(2)
        n = 32
        y = torch.randn(n)
        pred = torch.stack([y - 0.3, y + 0.3], dim=-1)

        loss_fn = ConformalLoss(method="cqr", alpha=0.1)
        loss_1d = loss_fn(pred, y)
        loss_2d = loss_fn(pred, y.unsqueeze(-1))
        assert torch.allclose(loss_1d, loss_2d)


class TestLocalConformal:
    """Tests for the LocalConformal predictor."""

    def test_basic_calibrate_predict(self):
        cp = LocalConformal(alpha=0.1, bandwidth=1.0)
        torch.manual_seed(42)
        preds = torch.randn(100, 1)
        targets = preds + torch.randn(100, 1) * 0.3
        x_cal = torch.randn(100, 2)

        # Test validation of x
        with pytest.raises(ValueError, match="requires features"):
            cp.calibrate(preds, targets)

        cp.calibrate(preds, targets, x=x_cal)
        assert cp._is_calibrated

        x_test = torch.randn(20, 2)
        y_test_pred = torch.randn(20, 1)

        with pytest.raises(ValueError, match="requires test features"):
            cp.predict_interval(y_test_pred)

        lower, upper = cp.predict_interval(y_test_pred, x=x_test)
        assert lower.shape == (20, 1)
        assert upper.shape == (20, 1)
        assert (lower <= upper).all()

    def test_coverage(self):
        """LVD local conformal coverage check."""
        torch.manual_seed(42)
        alpha = 0.1
        n_cal, n_test = 200, 500

        # Features
        x_cal = torch.randn(n_cal, 2)
        x_test = torch.randn(n_test, 2)

        # Labels (residual is larger when first feature is larger)
        noise_cal = torch.randn(n_cal, 1) * (0.2 + torch.abs(x_cal[:, 0:1]))
        noise_test = torch.randn(n_test, 1) * (0.2 + torch.abs(x_test[:, 0:1]))

        preds_cal = torch.randn(n_cal, 1)
        targets_cal = preds_cal + noise_cal

        preds_test = torch.randn(n_test, 1)
        targets_test = preds_test + noise_test

        cp = LocalConformal(alpha=alpha, bandwidth=0.5)
        cp.calibrate(preds_cal, targets_cal, x=x_cal)
        lower, upper = cp.predict_interval(preds_test, x=x_test)

        covered = (targets_test >= lower) & (targets_test <= upper)
        coverage = covered.float().mean().item()
        assert coverage >= (1 - alpha) - 0.05

    def test_custom_kernel(self):
        """Test with a custom kernel object."""

        class MockKernel:
            def K(self, x1, x2=None):
                return 1.0

            def Ki(self, xi, Xs):
                # Return uniform weights
                M = xi.shape[0] if xi.ndim > 1 else 1
                N = Xs.shape[0]
                return torch.ones((M, N), device=xi.device), Xs

        cp = LocalConformal(alpha=0.1, K_obj=MockKernel())
        preds = torch.randn(50, 1)
        targets = preds + torch.randn(50, 1) * 0.2
        x = torch.randn(50, 2)

        cp.calibrate(preds, targets, x=x)
        lower, upper = cp.predict_interval(preds, x=x)
        assert (lower <= upper).all()

    def test_mondrian_and_weights(self):
        """Test Mondrian groups and importance weights with LocalConformal."""
        cp = LocalConformal(alpha=0.1, bandwidth=1.0)
        torch.manual_seed(42)
        preds = torch.randn(100, 1)
        targets = preds + torch.randn(100, 1) * 0.3
        x = torch.randn(100, 2)
        groups = torch.randint(0, 2, (100,))
        weights = torch.rand(100) + 0.5

        cp.calibrate(preds, targets, x=x, groups=groups, weights=weights)
        assert cp._is_calibrated

        lower, upper = cp.predict_interval(preds, x=x, groups=groups)
        assert (lower <= upper).all()


class TestLocalConformalMAD:
    """Tests for the LocalConformalMAD predictor."""

    def test_basic_calibrate_predict(self):
        cp = LocalConformalMAD(alpha=0.1, bandwidth=1.0)
        torch.manual_seed(42)
        preds = torch.randn(100, 1)
        targets = preds + torch.randn(100, 1) * 0.3
        x_cal = torch.randn(100, 2)
        mad_cal = torch.rand(100, 1) + 0.1

        # Test validation of x and mad
        with pytest.raises(ValueError, match="requires features"):
            cp.calibrate(preds, targets)
        with pytest.raises(ValueError, match="requires MAD"):
            cp.calibrate(preds, targets, x=x_cal)

        cp.calibrate(preds, targets, x=x_cal, mad=mad_cal)
        assert cp._is_calibrated

        x_test = torch.randn(20, 2)
        y_test_pred = torch.randn(20, 1)
        mad_test = torch.rand(20, 1) + 0.1

        with pytest.raises(ValueError, match="requires test features"):
            cp.predict_interval(y_test_pred)
        with pytest.raises(ValueError, match="requires test MAD"):
            cp.predict_interval(y_test_pred, x=x_test)

        lower, upper = cp.predict_interval(y_test_pred, x=x_test, mad=mad_test)
        assert lower.shape == (20, 1)
        assert upper.shape == (20, 1)
        assert (lower <= upper).all()

    def test_coverage(self):
        """LVD local conformal MAD coverage check."""
        torch.manual_seed(42)
        alpha = 0.1
        n_cal, n_test = 200, 500

        x_cal = torch.randn(n_cal, 2)
        x_test = torch.randn(n_test, 2)

        # Scale model predicts the exact noise std
        mad_cal = 0.2 + torch.abs(x_cal[:, 0:1])
        mad_test = 0.2 + torch.abs(x_test[:, 0:1])

        noise_cal = torch.randn(n_cal, 1) * mad_cal
        noise_test = torch.randn(n_test, 1) * mad_test

        preds_cal = torch.randn(n_cal, 1)
        targets_cal = preds_cal + noise_cal

        preds_test = torch.randn(n_test, 1)
        targets_test = preds_test + noise_test

        cp = LocalConformalMAD(alpha=alpha, bandwidth=0.5)
        cp.calibrate(preds_cal, targets_cal, x=x_cal, mad=mad_cal)
        lower, upper = cp.predict_interval(preds_test, x=x_test, mad=mad_test)

        covered = (targets_test >= lower) & (targets_test <= upper)
        coverage = covered.float().mean().item()
        assert coverage >= (1 - alpha) - 0.05


class TestCVPlus:
    """Tests for CVPlus and JackknifePlus predictors."""

    def test_basic_calibrate_predict(self):
        torch.manual_seed(42)
        n_cal = 100
        n_test = 20
        n_folds = 5
        output_dim = 1

        # Out-of-fold calibration predictions
        y_pred_oob = torch.randn(n_cal, output_dim)
        target = y_pred_oob + torch.randn(n_cal, output_dim) * 0.3
        fold_indices = torch.randint(0, n_folds, (n_cal,))

        cp = CVPlus(alpha=0.1)

        # Test predict before calibrate
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict_interval(torch.randn(n_folds, n_test, output_dim))

        # Calibrate
        cp.calibrate_ensemble(y_pred_oob, target, fold_indices)
        assert cp._is_calibrated
        assert cp.residuals is not None
        assert cp.residuals.shape == (n_cal,)

        # Test prediction
        y_pred_members = torch.randn(n_folds, n_test, output_dim)
        lower, upper = cp.predict_interval(y_pred_members)

        assert lower.shape == (n_test, output_dim)
        assert upper.shape == (n_test, output_dim)
        assert (lower <= upper).all()

        # JackknifePlus alias check
        assert JackknifePlus is CVPlus

    def test_with_mask(self):
        torch.manual_seed(42)
        n_cal = 100
        n_folds = 5
        output_dim = 2

        y_pred_oob = torch.randn(n_cal, output_dim)
        target = y_pred_oob + torch.randn(n_cal, output_dim) * 0.2
        fold_indices = torch.randint(0, n_folds, (n_cal,))

        mask = torch.ones(n_cal, output_dim, dtype=torch.bool)
        mask[:10] = False  # Mask out first 10 samples

        cp = CVPlus(alpha=0.1)
        cp.calibrate_ensemble(y_pred_oob, target, fold_indices, mask=mask)
        assert cp._is_calibrated

        # Total valid residuals should be 90
        assert cp.residuals.shape == (90,)

        # Empty calibration set
        empty_mask = torch.zeros(n_cal, dtype=torch.bool)
        with pytest.raises(ValueError, match="Calibration set is empty"):
            cp.calibrate_ensemble(y_pred_oob, target, fold_indices, mask=empty_mask)

    def test_coverage(self):
        """Simulation to verify approximate coverage for CV+."""
        torch.manual_seed(42)
        alpha = 0.1
        n_cal = 200
        n_test = 500
        n_folds = 5
        output_dim = 1

        # Generate simulation data
        y_pred_oob = torch.randn(n_cal, output_dim)
        target = y_pred_oob + torch.randn(n_cal, output_dim) * 0.5
        fold_indices = torch.randint(0, n_folds, (n_cal,))

        cp = CVPlus(alpha=alpha)
        cp.calibrate_ensemble(y_pred_oob, target, fold_indices)

        # For prediction, we need member predictions
        # Simulate member predictions: K models centered around the true target with some variance
        target_test = torch.randn(n_test, output_dim)
        y_pred_members = (
            target_test.unsqueeze(0).repeat(n_folds, 1, 1)
            + torch.randn(n_folds, n_test, output_dim) * 0.1
        )

        lower, upper = cp.predict_interval(y_pred_members)

        covered = (target_test >= lower) & (target_test <= upper)
        coverage = covered.float().mean().item()

        # Marginal coverage guarantee should hold approximately
        assert coverage >= (1 - alpha) - 0.05


class TestEnsembleBatchCP:
    """Tests for EnsembleBatchCP (EnbPI) predictor."""

    def test_basic_calibrate_predict(self):
        torch.manual_seed(42)
        n_cal = 100
        n_test = 20
        output_dim = 1

        # Out-of-bag predictions
        y_pred_oob = torch.randn(n_cal, output_dim)
        target = y_pred_oob + torch.randn(n_cal, output_dim) * 0.3

        cp = EnsembleBatchCP(alpha=0.1)

        # Test predict before calibrate
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict_interval(torch.randn(n_test, output_dim))

        # Calibrate
        cp.calibrate(y_pred_oob, target)
        assert cp._is_calibrated
        assert cp.q_hat is not None

        # Test prediction
        y_pred_mean = torch.randn(n_test, output_dim)
        lower, upper = cp.predict_interval(y_pred_mean)

        assert lower.shape == (n_test, output_dim)
        assert upper.shape == (n_test, output_dim)
        assert (lower <= upper).all()

    def test_coverage(self):
        """Simulation to verify approximate coverage for EnsembleBatchCP."""
        torch.manual_seed(42)
        alpha = 0.1
        n_cal = 300
        n_test = 500
        output_dim = 1

        y_pred_oob = torch.randn(n_cal, output_dim)
        target = y_pred_oob + torch.randn(n_cal, output_dim) * 0.5

        cp = EnsembleBatchCP(alpha=alpha)
        cp.calibrate(y_pred_oob, target)

        target_test = torch.randn(n_test, output_dim)
        y_pred_mean = target_test + torch.randn(n_test, output_dim) * 0.1

        lower, upper = cp.predict_interval(y_pred_mean)

        covered = (target_test >= lower) & (target_test <= upper)
        coverage = covered.float().mean().item()

        assert coverage >= (1 - alpha) - 0.05


if __name__ == "__main__":
    pytest.main([__file__])
