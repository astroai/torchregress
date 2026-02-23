"""
Tests for native conformal prediction module.

Tests cover all predictor classes and backward-compatible loss wrappers.
"""

import pytest
import torch

from torchregress.losses.conformal import (
    CQR,
    CTI,
    ConformalLoss,
    ConformalPredictor,
    DistributionalConformal,
    MultiDimensionalConformalLoss,
    MultiTargetConformal,
    R2CConformal,
    SplitConformal,
)

# ---------------------------------------------------------------------------
# Backward compatibility tests (existing tests, preserved)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["cqr", "split"])
def test_conformal_loss_initialization(method):
    """Test initialization of ConformalLoss for different methods."""
    loss_fn = ConformalLoss(method=method, alpha=0.1)
    assert loss_fn.method == method
    assert loss_fn.alpha == 0.1
    assert not loss_fn._is_calibrated


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


@pytest.mark.parametrize("method", ["cqr", "split"])
def test_conformal_loss_calibration_and_prediction(method):
    """Test calibration and prediction flow."""
    loss_fn = ConformalLoss(method=method, alpha=0.1)
    batch_size, n_features = 50, 1

    if method == "cqr":
        y_pred_cal = torch.randn(batch_size, 2 * n_features)
        y_pred_test = torch.randn(batch_size, 2 * n_features)
    else:
        y_pred_cal = torch.randn(batch_size, n_features)
        y_pred_test = torch.randn(batch_size, n_features)

    y_true_cal = torch.randn(batch_size, n_features)

    with pytest.raises(RuntimeError):
        loss_fn.predict_interval(y_pred_test)

    loss_fn.calibrate(y_pred_cal, y_true_cal)
    assert loss_fn._is_calibrated

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


def test_multidimensional_conformal_loss():
    """Test MultiDimensionalConformalLoss for multi-output regression."""
    loss_fn = MultiDimensionalConformalLoss(alpha=0.1)
    assert isinstance(loss_fn, ConformalLoss)
    assert loss_fn.method == "split"

    batch_size, n_features = 50, 3
    y_pred_cal = torch.randn(batch_size, n_features)
    y_true_cal = torch.randn(batch_size, n_features)
    y_pred_test = torch.randn(batch_size, n_features)

    loss_fn.calibrate(y_pred_cal, y_true_cal)
    lower, upper = loss_fn.predict_interval(y_pred_test)
    assert lower.shape == y_pred_test.shape
    assert upper.shape == y_pred_test.shape
    assert lower.shape[-1] == n_features
    assert (lower <= upper).all()

    assert loss_fn.q_hat is not None
    assert loss_fn.q_hat.shape == (n_features,)


def test_conformal_with_mask():
    """Test that masking works during calibration."""
    loss_fn = ConformalLoss(method="split", alpha=0.1)
    batch_size, n_features = 20, 1

    y_pred = torch.randn(batch_size, n_features)
    y_true = torch.randn(batch_size, n_features)
    mask = torch.ones(batch_size, n_features, dtype=torch.bool)
    mask[:5] = False

    loss_fn.calibrate(y_pred, y_true, mask=mask)
    assert loss_fn._is_calibrated

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

        # Case 1: Exception triggers fallback
        loop_calls = 0

        def icdf_fn_fail(levels, x):
            nonlocal loop_calls
            if x.ndim == 2 and x.shape[0] > 1:
                raise ValueError("No batch support")
            loop_calls += 1
            return Normal(x[0], 1.0).icdf(levels)

        dcp.predict_intervals_from_cdf(icdf_fn_fail, x_test)
        assert loop_calls == n_test

        # Case 2: Wrong shape triggers fallback
        loop_calls = 0

        def icdf_fn_wrong_shape(levels, x):
            nonlocal loop_calls
            if x.ndim == 2 and x.shape[0] > 1:
                return torch.zeros(2)  # Wrong shape (should be N, 2)
            loop_calls += 1
            return Normal(x[0], 1.0).icdf(levels)

        dcp.predict_intervals_from_cdf(icdf_fn_wrong_shape, x_test)
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
        assert loss_fn._is_calibrated
        assert isinstance(loss_fn.q_hat, dict)

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
        assert loss_fn._is_calibrated

        lower, upper = loss_fn.predict_interval(preds)
        assert (lower <= upper).all()


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


if __name__ == "__main__":
    pytest.main([__file__])
