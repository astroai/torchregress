"""Tests for :class:`torchregress.calibration.posthoc.VarianceRestoringCalibrator`.

The scientific point of the calibrator is a trade-off, not a free improvement:
it matches the *marginal* distribution of point predictions while it is not
conditional-mean calibration. The tests below pin down both sides:

* the maps are correct (analytic affine scale, exact 1-D optimal-transport
  marginal matching);
* the diagnostics expose the honest trade-off: a calibrated MSE predictor gets
  worse MSE after stretching, an attenuated predictor gets better MSE.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from torchregress.calibration.posthoc import VarianceRestoringCalibrator


def _calibrated_split(n: int = 4000, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE-optimal Gaussian predictor: ``pred = E[y | x]`` and target noise on top."""
    gen = torch.Generator().manual_seed(seed)
    pred = torch.randn(n, generator=gen)
    target = pred + torch.randn(n, generator=gen)
    return pred, target


def _attenuated_split(
    n: int = 4000, seed: int = 1, rho: float = 0.5, noise: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Under-dispersed predictor: ``pred = rho * target + noise`` with ``rho < 1``."""
    gen = torch.Generator().manual_seed(seed)
    target = torch.randn(n, generator=gen)
    pred = rho * target + noise * torch.randn(n, generator=gen)
    return pred, target


def _var_ratio(values: torch.Tensor, reference: torch.Tensor) -> float:
    return float(values.double().var(unbiased=False) / reference.double().var(unbiased=False))


class TestAffineMap:
    """Moment matching: ``T(pred) = prior_mean + scale * (pred - prior_mean)``."""

    def test_scale_matches_analytic_variance_ratio(self):
        pred, target = _attenuated_split()
        cal = VarianceRestoringCalibrator(method="affine").fit(pred, target)
        target_d, pred_d = target.double(), pred.double()
        expected = math.sqrt(
            float(target_d.var(unbiased=False)) / float(pred_d.var(unbiased=False))
        )
        assert cal.scale_ == pytest.approx(expected, rel=1e-12)
        # Default prior mean is the calibration-target mean.
        assert cal.prior_mean_ == pytest.approx(float(target_d.mean()), rel=1e-12)
        # The transformed predictions have the target's empirical variance.
        assert _var_ratio(cal.transform(pred), target) == pytest.approx(1.0, rel=1e-5)

    def test_prior_mean_override_is_honoured(self):
        pred, target = _attenuated_split(seed=2)
        cal = VarianceRestoringCalibrator().fit(pred, target, prior_mean=0.0)
        assert cal.prior_mean_ == 0.0
        # The map is a pure stretch about zero.
        expected = cal.scale_ * pred
        torch.testing.assert_close(cal.transform(pred), expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(cal.transform(torch.zeros(4)), torch.zeros(4))

    def test_prior_mean_scalar_tensor_accepted(self):
        pred, target = _attenuated_split(seed=3)
        cal = VarianceRestoringCalibrator().fit(pred, target, prior_mean=torch.tensor(0.25))
        assert cal.prior_mean_ == pytest.approx(0.25)

    def test_prior_mean_validation(self):
        pred, target = _attenuated_split(n=100, seed=4)
        with pytest.raises(ValueError, match="scalar"):
            VarianceRestoringCalibrator().fit(pred, target, prior_mean=torch.zeros(2))
        with pytest.raises(ValueError, match="finite"):
            VarianceRestoringCalibrator().fit(pred, target, prior_mean=float("nan"))
        with pytest.raises(ValueError, match="affine"):
            VarianceRestoringCalibrator(method="quantile").fit(pred, target, prior_mean=0.0)

    def test_eps_is_a_variance_denominator_floor(self):
        pred, target = _attenuated_split(n=200, seed=5)
        # A huge eps floors the prediction-variance denominator and biases s down.
        floored = VarianceRestoringCalibrator(eps=1.0).fit(pred, target)
        plain = VarianceRestoringCalibrator(eps=1e-8).fit(pred, target)
        assert floored.scale_ == pytest.approx(
            math.sqrt(float(target.double().var(unbiased=False))), rel=1e-12
        )
        assert floored.scale_ < plain.scale_


class TestQuantileMap:
    """1-D optimal transport: piecewise-linear monotone map between sorted pairs."""

    def _fit(self, n: int = 500, seed: int = 10, out_of_bounds: str = "clip"):
        gen = torch.Generator().manual_seed(seed)
        pred = torch.randn(n, generator=gen)
        target = 2.0 * pred + 0.5 * torch.randn(n, generator=gen)
        cal = VarianceRestoringCalibrator(method="quantile", out_of_bounds=out_of_bounds).fit(
            pred, target
        )
        return cal, pred, target

    def test_reproduces_calibration_marginal(self):
        cal, pred, target = self._fit()
        x_sorted = pred.sort().values
        y_sorted = target.sort().values
        torch.testing.assert_close(cal.transform(x_sorted), y_sorted, rtol=0.0, atol=1e-5)
        # The fitted grids are the sorted calibration samples.
        torch.testing.assert_close(cal._x, pred.double().sort().values)
        torch.testing.assert_close(cal._y, target.double().sort().values)
        # Empirical marginals match exactly on the calibration sample.
        assert abs(float(cal.describe(pred, target)["wasserstein1_after"])) < 1e-6

    def test_map_is_monotone(self):
        cal, pred, _ = self._fit(seed=11)
        grid = torch.linspace(float(pred.min()) - 1.0, float(pred.max()) + 1.0, 501)
        out = cal.transform(grid)
        assert torch.all(out[1:] >= out[:-1])

    def test_out_of_bounds_clip_holds_end_values(self):
        cal, pred, target = self._fit(seed=12)
        x_sorted = pred.sort().values
        y_sorted = target.sort().values
        out = cal.transform(torch.tensor([x_sorted[0] - 5.0, x_sorted[-1] + 5.0]))
        torch.testing.assert_close(out, torch.tensor([y_sorted[0], y_sorted[-1]]))

    def test_out_of_bounds_linear_extrapolates_finitely(self):
        cal, pred, target = self._fit(seed=13, out_of_bounds="linear")
        x_sorted = pred.sort().values
        y_sorted = target.sort().values
        out = cal.transform(torch.tensor([x_sorted[0] - 5.0, x_sorted[-1] + 5.0]))
        assert torch.all(torch.isfinite(out))
        assert float(out[0]) < float(y_sorted[0])
        assert float(out[1]) > float(y_sorted[-1])

    def test_invalid_out_of_bounds_raises(self):
        with pytest.raises(ValueError, match="out_of_bounds"):
            VarianceRestoringCalibrator(method="quantile", out_of_bounds="clamp").fit(
                torch.randn(20), torch.randn(20)
            )


class TestShapesDtypesAndInputValidation:
    """Shape/dtype preservation, non-finite filtering, and error paths."""

    @pytest.mark.parametrize("method", ["affine", "quantile"])
    def test_transform_preserves_shape_and_dtype(self, method):
        pred, target = _attenuated_split(n=100, seed=20)
        cal = VarianceRestoringCalibrator(method=method).fit(pred, target)
        for shape in [(10,), (10, 1), (2, 5)]:
            out = cal.transform(torch.randn(*shape))
            assert out.shape == shape
        out = cal.transform(pred)
        assert out.dtype == torch.float32

        cal64 = VarianceRestoringCalibrator(method=method).fit(pred.double(), target.double())
        assert cal64.transform(pred.double()).dtype == torch.float64

    def test_fit_accepts_equal_numel_different_shape(self):
        pred = torch.randn(50, 1)
        target = torch.randn(50)
        cal = VarianceRestoringCalibrator().fit(pred, target)
        assert cal.scale_ is not None
        assert cal.transform(pred).shape == pred.shape

    @pytest.mark.parametrize("method", ["affine", "quantile"])
    def test_non_finite_calibration_samples_dropped(self, method):
        pred, target = _attenuated_split(n=100, seed=21)
        pred = pred.clone()
        target = target.clone()
        pred[0] = float("nan")
        target[1] = float("inf")
        target[2] = float("-inf")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cal = VarianceRestoringCalibrator(method=method).fit(pred, target)
        if method == "affine":
            assert cal.scale_ is not None and math.isfinite(cal.scale_)
        else:
            assert cal._x is not None and cal._x.numel() == 97
        diagnostics = cal.describe(pred, target)
        assert all(math.isfinite(value) for value in diagnostics.values())

    def test_fit_errors(self):
        cal = VarianceRestoringCalibrator()
        with pytest.raises(ValueError, match="same number of elements"):
            cal.fit(torch.randn(10), torch.randn(15))
        with pytest.raises(ValueError, match="at least two"):
            cal.fit(torch.tensor([1.0, float("nan")]), torch.tensor([1.0, 2.0]))
        with pytest.raises(ValueError, match="non-zero variance"):
            cal.fit(torch.ones(10), torch.randn(10))

    def test_transform_and_describe_before_fit_raise(self):
        with pytest.raises(ValueError, match="fitted before transform"):
            VarianceRestoringCalibrator().transform(torch.randn(10))
        with pytest.raises(ValueError, match="fitted before transform"):
            VarianceRestoringCalibrator().describe(torch.randn(10), torch.randn(10))

    def test_unknown_method_and_eps_raise(self):
        with pytest.raises(ValueError, match="unknown method"):
            VarianceRestoringCalibrator(method="bogus").fit(torch.randn(10), torch.randn(10))
        with pytest.raises(ValueError, match="eps"):
            VarianceRestoringCalibrator(eps=0.0).fit(torch.randn(10), torch.randn(10))

    def test_describe_rejects_mismatched_numel(self):
        pred, target = _attenuated_split(n=20, seed=22)
        cal = VarianceRestoringCalibrator().fit(pred, target)
        with pytest.raises(ValueError, match="same number of elements"):
            cal.describe(pred, target[:-1])


class TestDescribe:
    """The diagnostics returned by :meth:`describe`."""

    EXPECTED_KEYS = {
        "mse_before",
        "mse_after",
        "mse_ratio",
        "wasserstein1_before",
        "wasserstein1_after",
        "var_ratio_before",
        "var_ratio_after",
    }

    @pytest.mark.parametrize("method", ["affine", "quantile"])
    def test_keys_present_and_finite(self, method):
        pred, target = _attenuated_split(n=500, seed=30)
        cal = VarianceRestoringCalibrator(method=method).fit(pred, target)
        diagnostics = cal.describe(pred, target)
        assert set(diagnostics) == self.EXPECTED_KEYS
        for value in diagnostics.values():
            assert isinstance(value, float)
            assert math.isfinite(value)


class TestHonestTradeoff:
    """Marginal matching is not conditional calibration: both sides are measured."""

    def test_calibrated_predictor_fixes_marginal_but_increases_mse(self):
        pred, target = _calibrated_split()
        cal = VarianceRestoringCalibrator(method="affine").fit(pred, target)
        diagnostics = cal.describe(pred, target)

        # The predictor is already MSE-optimal; stretching the spread adds error.
        assert diagnostics["mse_after"] >= diagnostics["mse_before"] - 1e-9
        assert diagnostics["mse_ratio"] > 1.0
        # ... while the marginal variance and the W1 distance are restored.
        assert diagnostics["var_ratio_after"] == pytest.approx(1.0, abs=1e-6)
        assert diagnostics["var_ratio_before"] == pytest.approx(0.5, abs=0.05)
        assert diagnostics["wasserstein1_after"] <= diagnostics["wasserstein1_before"]

    @pytest.mark.parametrize("method", ["affine", "quantile"])
    def test_attenuated_predictor_improves_on_both_axes(self, method):
        pred, target = _attenuated_split()
        cal = VarianceRestoringCalibrator(method=method).fit(pred, target)
        diagnostics = cal.describe(pred, target)

        # Under-dispersed predictions: de-shrinking pays off in point error ...
        assert diagnostics["mse_after"] < diagnostics["mse_before"]
        assert diagnostics["mse_ratio"] < 1.0
        assert diagnostics["wasserstein1_after"] <= diagnostics["wasserstein1_before"]
        # ... and the marginal variance ratio is matched by construction.
        assert diagnostics["var_ratio_before"] < 0.5
        assert diagnostics["var_ratio_after"] == pytest.approx(1.0, abs=1e-6)
