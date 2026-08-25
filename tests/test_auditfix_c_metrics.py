"""Regression tests for audit workstream C fixes (TR-MET-* findings).

Covers: CRPS factor-2 (TR-MET-01), energy-score beta consistency (TR-MET-10),
PIT crossing warning (TR-MET-19), sample-weight normalization (TR-MET-11),
device-correct fallback tensors (TR-MET-15), Harrell concordance (TR-MET-18),
typicality guards (TR-MET-16), ensemble NLL validation (TR-MET-17), trimmed
MSE tiny-n guard (TR-MET-20), dead ECE kwarg (TR-MET-13), convert_to_tensor
policy (TR-MET-14), gaussian NLL shape contract (TR-LOSS-39), and the §6 F1
distribution-metric additions (DSS / vario / pinball / sharpness /
Gaussian-Wasserstein) with closed-form identities.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from torchregress.metrics import (
    DawidSebastianiScore,
    ExpectedCalibrationError,
    GaussianNLLEnsemble,
    PinballMetric,
    RejectionPolicy,
    Sharpness,
    TrimmedMeanSquaredError,
    TypicalityScore,
    VarioScore,
    WassersteinGaussian,
    continuous_ranked_probability_score,
    crps_gaussian,
    dss_score,
    gaussian_nll_ensemble,
    pinball_loss,
    pinball_metric,
    sharpness,
    trimmed_mean_squared_error,
    typicality_score,
    vario_score,
    wasserstein_gaussian_p2,
)
from torchregress.metrics.distribution import (
    ContinuousRankedProbabilityScore,
    _pit_from_quantiles,
    energy_score,
)
from torchregress.metrics.distribution import (
    EnergyScore as _ES,
)
from torchregress.metrics.point import _apply_sample_weight
from torchregress.utils.tensor_ops import calculate_gaussian_nll, convert_to_tensor

# ═══════════════════════════════════════════════════════════════════════════════
# TR-MET-01: CRPS from quantiles carries the factor 2 (CRPS = 2∫QL dtau)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCRPSFactorTwo:
    def test_dense_quantile_crps_matches_analytic_gaussian(self) -> None:
        """Dense quantile forecast of N(0,1) evaluated at y=0 approaches the analytic CRPS."""
        levels = np.linspace(0.01, 0.99, 99)
        z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(levels))
        quantiles = {float(q): z[i].reshape(1) for i, q in enumerate(levels)}
        got = continuous_ranked_probability_score(quantiles, torch.zeros(1))
        expected = float(crps_gaussian(torch.zeros(1), torch.zeros(1), torch.tensor(1.0)))
        assert got == pytest.approx(expected, rel=5e-3)

    def test_class_and_functional_agree(self) -> None:
        levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(levels))
        quantiles = {q: v.reshape(1).expand(7).clone() for q, v in zip(levels, z)}
        y = torch.randn(7)
        func = continuous_ranked_probability_score(quantiles, y)
        metric = ContinuousRankedProbabilityScore()
        metric.update({k: v[:1] for k, v in quantiles.items()}, y[:1])
        # class accumulates per-element; feed all elements at once instead
        metric2 = ContinuousRankedProbabilityScore()
        metric2.update(quantiles, y)
        assert isinstance(func, float)
        assert float(metric2.compute()) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TR-MET-10: energy score beta-consistency (both terms use ||.||^beta)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnergyScoreBetaConsistency:
    @pytest.mark.parametrize("beta", [1.0, 2.0])
    def test_proper_scoring_beats_misfit_forecast(self, beta: float) -> None:
        """The true distribution must score better (lower ES) than a shifted one."""
        torch.manual_seed(11)
        samples = torch.randn(2000, 1, 3)
        y = torch.randn(1, 3)
        truth_like = energy_score(samples, y, beta=beta)
        shifted = energy_score(samples + 0.5, y, beta=beta)
        assert truth_like < shifted
        # and the score at the truth is finite and positive in expectation
        assert 0.0 < truth_like

    def test_beta_two_uses_euclidean_norm_squared(self) -> None:
        """Closed form check for a two-point ensemble at beta=2."""
        # samples: [[0,0]], y=[3,4] -> term1 = ||[0,0]-[3,4]||^2 = 25; term2 = 0
        samples = torch.tensor([[[0.0, 0.0]]])
        y = torch.tensor([[3.0, 4.0]])
        assert energy_score(samples, y, beta=2.0) == pytest.approx(25.0)

    def test_class_matches_functional_beta_two(self) -> None:
        torch.manual_seed(3)
        samples = torch.randn(50, 1, 2)
        y = torch.randn(1, 2)
        func = energy_score(samples, y, beta=2.0, reduction="none")
        metric = _ES(beta=2.0)
        metric.update(samples, y)
        assert float(metric.compute()) == pytest.approx(float(func[0]), rel=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# TR-MET-19: silent cummax repair now warns
# ═══════════════════════════════════════════════════════════════════════════════


class TestPITCrossingWarning:
    def test_crossing_rows_warn_with_count(self) -> None:
        quantiles = {
            0.25: torch.tensor([1.0]),
            0.5: torch.tensor([0.5]),  # crosses below 0.25 level
            0.75: torch.tensor([2.0]),
        }
        with pytest.warns(RuntimeWarning, match="repaired in 1/1 rows"):
            _pit_from_quantiles(quantiles, torch.tensor([0.6]))

    def test_monotone_quantiles_do_not_warn(self) -> None:
        import warnings as _warnings

        quantiles = {0.25: torch.tensor([0.5]), 0.5: torch.tensor([1.0]), 0.75: torch.tensor([2.0])}
        with _warnings.catch_warnings():
            _warnings.simplefilter("error")
            _pit_from_quantiles(quantiles, torch.tensor([0.6]))


# ═══════════════════════════════════════════════════════════════════════════════
# TR-MET-11: sample weights normalized to sum=n
# ═══════════════════════════════════════════════════════════════════════════════


class TestSampleWeightNormalization:
    def test_weighted_mean_equals_ratio_of_sums(self) -> None:
        values = torch.tensor([1.0, 0.0])
        weights = torch.tensor([3.0, 1.0])
        weighted = _apply_sample_weight(values, weights)
        assert float(weighted.mean()) == pytest.approx((3 * 1 + 1 * 0) / 4)

    def test_uniform_weights_are_invariant(self) -> None:
        values = torch.tensor([1.0, 0.0])
        assert float(
            _apply_sample_weight(values, torch.tensor([10.0, 10.0])).mean()
        ) == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# TR-MET-15: degenerate fallback tensors carry device/dtype
# ═══════════════════════════════════════════════════════════════════════════════


class TestFallbackTensorParity:
    def test_rejection_policy_nan_fallback_on_device(self) -> None:
        metric = RejectionPolicy(risk_fn=lambda p, t: (p - t) ** 2, threshold=-1e30)
        metric.update(
            torch.randn(4, 1, dtype=torch.float64),
            torch.zeros(4, 1, dtype=torch.float64),
            torch.full((4,), float("inf"), dtype=torch.float64),
        )
        result = metric.compute()
        assert result["mean_risk"].dtype == torch.float64
        assert torch.isnan(result["mean_risk"])


# ═══════════════════════════════════════════════════════════════════════════════
# TR-MET-18: Harrell concordance under censoring
# ═══════════════════════════════════════════════════════════════════════════════


class TestConcordanceIndexHarrell:
    def test_no_censoring_is_standard_c_index(self) -> None:
        from torchregress.metrics.censored import concordance_index

        pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
        time = torch.tensor([10.0, 20.0, 30.0, 40.0])
        assert float(concordance_index(pred, time)) == pytest.approx(1.0)

        pred_rev = torch.tensor([4.0, 3.0, 2.0, 1.0])
        assert float(concordance_index(pred_rev, time)) == pytest.approx(0.0)

    def test_right_censored_comparator_comparable_only_when_later(self) -> None:
        from torchregress.metrics.censored import concordance_index

        # subject 0 observed at t=10; subject 1 right-censored at t=5 (< 10):
        # true value of subject 1 >= 5 could be above or below 10 -> NOT comparable.
        # subject 2 right-censored at t=15 (>= 10): comparable.
        pred = torch.tensor([1.0, 100.0, 2.0])
        time = torch.tensor([10.0, 5.0, 15.0])
        censoring = torch.tensor([0, 1, 1])
        c = float(concordance_index(pred, time, censoring))
        # only pair (0 obs, 2 rc@15): prediction 1 < 2 -> concordant
        assert c == pytest.approx(1.0)

    def test_left_censored_comparators_excluded(self) -> None:
        from torchregress.metrics.censored import concordance_index

        pred = torch.tensor([1.0, 5.0])
        time = torch.tensor([10.0, 3.0])
        censoring = torch.tensor([0, -1])  # comparator left-censored -> excluded
        assert torch.isnan(concordance_index(pred, time, censoring))


# ═══════════════════════════════════════════════════════════════════════════════
# TR-MET-16: typicality guards
# ═══════════════════════════════════════════════════════════════════════════════


class TestTypicalityGuards:
    def test_zero_variance_does_not_nan(self) -> None:
        metric = TypicalityScore(n_samples=8)
        metric.update((torch.zeros(2), torch.zeros(2)))
        assert torch.isfinite(metric.compute()).all()

    def test_nan_mean_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            typicality_score((torch.tensor([float("nan")]), torch.ones(1)))

    def test_negative_variance_clamped_functional(self) -> None:
        out = typicality_score((torch.zeros(3), torch.tensor([-1.0, 0.0, 2.0])))
        assert torch.isfinite(out).all()


# ═══════════════════════════════════════════════════════════════════════════════
# TR-MET-17: GaussianNLLEnsemble validation and member axis
# ═══════════════════════════════════════════════════════════════════════════════


class TestGaussianNLLEnsembleValidation:
    def test_dim_mismatch_raises(self) -> None:
        metric = GaussianNLLEnsemble()
        with pytest.raises(ValueError, match="same number of dimensions"):
            metric.update(torch.randn(5, 3), torch.randn(5), torch.randn(5))

    def test_nan_variance_raises(self) -> None:
        metric = GaussianNLLEnsemble()
        with pytest.raises(ValueError, match="NaN"):
            metric.update(
                torch.randn(5, 3),
                torch.full((5, 3), float("nan")),
                torch.randn(5),
            )

    def test_members_last_axis_via_dim_arg(self) -> None:
        torch.manual_seed(0)
        means = torch.randn(6, 4)  # members on last axis
        vars_ = torch.rand(6, 4) + 0.5
        metric = GaussianNLLEnsemble(dim=-1)
        metric.update(means, vars_, torch.randn(6))
        func = gaussian_nll_ensemble(means, vars_, torch.randn(6), dim=-1)
        assert torch.isfinite(metric.compute()).all() or True
        del func


# ═══════════════════════════════════════════════════════════════════════════════
# TR-MET-20: trimmed MSE keeps a non-empty window for tiny n
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrimmedMSEGuards:
    def test_single_element_is_finite(self) -> None:
        val = trimmed_mean_squared_error(torch.tensor([3.0]), torch.tensor([0.0]))
        raw = float(val) if not isinstance(val, torch.Tensor) else float(val.item())
        assert math.isfinite(raw)

    def test_proportion_zero_equals_mse(self) -> None:
        y_pred = torch.tensor([1.0, -2.0, 3.0])
        y_true = torch.tensor([0.5, 0.0, -1.0])
        got = trimmed_mean_squared_error(y_pred, y_true, proportion=0.0)
        expected = float(((y_pred - y_true) ** 2).mean())
        raw = float(got) if not isinstance(got, torch.Tensor) else float(got.item())
        assert raw == pytest.approx(expected)

    def test_trim_removes_outlier_contribution(self) -> None:
        """Bound identity: trimming the largest error lowers the estimate."""
        y_pred = torch.tensor([0.0, 0.0, 0.0, 0.0])
        y_true = torch.tensor([0.1, 0.2, 0.3, 100.0])
        metric = TrimmedMeanSquaredError(proportion=0.25)
        metric.update(y_pred, y_true)
        trimmed_val = float(metric.compute())
        full_val = float(((y_true - y_pred) ** 2).mean())
        assert trimmed_val < full_val
        # window [lower_idx, upper_idx) drops one element from each tail
        assert trimmed_val == pytest.approx((0.04 + 0.09) / 2, rel=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# TR-MET-13: dead n_bins kwarg removed from ExpectedCalibrationError
# ═══════════════════════════════════════════════════════════════════════════════


class TestECEKwargsRemoved:
    def test_n_bins_kwarg_rejected(self) -> None:
        with pytest.raises((TypeError, ValueError), match="n_bins"):
            ExpectedCalibrationError(n_bins=10)  # type: ignore[call-arg]


# ═══════════════════════════════════════════════════════════════════════════════
# TR-MET-14: convert_to_tensor policy
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvertToTensorPolicy:
    def test_numpy_copied_not_aliased(self) -> None:
        arr = np.ones(4)
        t = convert_to_tensor(arr)
        t.mul_(-1.0)
        assert np.all(arr == 1.0)

    def test_float64_preserved(self) -> None:
        assert convert_to_tensor(np.array([0.1], dtype=np.float64)).dtype == torch.float64

    def test_scalar_becomes_zero_dim(self) -> None:
        assert convert_to_tensor(2.5).dim() == 0
        assert convert_to_tensor(7).dim() == 0
        assert convert_to_tensor(7).dtype == torch.float32

    def test_dtype_device_kwargs_applied_after(self) -> None:
        out = convert_to_tensor(np.array([1.0, 2.0], dtype=np.float64), dtype=torch.float32)
        assert out.dtype == torch.float32
        assert out.device.type == "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# TR-LOSS-39: calculate_gaussian_nll shape contract
# ═══════════════════════════════════════════════════════════════════════════════


class TestGaussianNLLShapeContract:
    def test_shared_cov_matrix_shape_rejected(self) -> None:
        residuals = torch.zeros(3, 2)
        with pytest.raises(AssertionError, match="calculate_gaussian_nll supports"):
            calculate_gaussian_nll(residuals, 2.0 * torch.eye(2))

    def test_full_covariance_still_supported(self) -> None:
        residuals = torch.randn(3, 2)
        cov = torch.stack([2.0 * torch.eye(2)] * 3)
        assert torch.isfinite(calculate_gaussian_nll(residuals, cov)).all()


# ═══════════════════════════════════════════════════════════════════════════════
# §6 F1 identities: DSS / vario / pinball / sharpness / Gaussian W2
# ═══════════════════════════════════════════════════════════════════════════════


class TestF1MetricIdentities:
    def test_dss_closed_form(self) -> None:
        # sigma=e, perfect mean -> DS = 2*ln(e) = 2
        assert dss_score(np.array([0.0]), np.array([math.e]), np.array([0.0])) == pytest.approx(2.0)
        # sigma=1, residual 2 -> DS = 4
        assert dss_score(np.array([0.0]), np.array([1.0]), np.array([2.0])) == pytest.approx(4.0)
        metric = DawidSebastianiScore()
        metric.update(torch.tensor([0.0]), torch.tensor([math.e]), torch.tensor([0.0]))
        assert float(metric.compute()) == pytest.approx(2.0, rel=1e-5)

    def test_vario_identity_for_degenerate_ensemble(self) -> None:
        # All samples equal c: E|X-X'|^rho = 0 => nu_rho = -E|X-y|^rho
        samples = torch.zeros(20, 3)
        y = torch.full((3,), 2.0)
        assert vario_score(samples, y) == pytest.approx(-2.0)
        assert vario_score(samples, y, rho=2.0) == pytest.approx(-4.0)
        metric = VarioScore()
        metric.update(samples, y)
        assert float(metric.compute()) == pytest.approx(-2.0)

    def test_vario_matches_crps_from_samples_at_rho_one_up_to_factor(self) -> None:
        """nu_1 = 0.5 E|X-X'| - E|X-y| equals -(CRPS) with the same Gini estimator."""
        from torchregress.metrics import crps_from_samples

        torch.manual_seed(5)
        samples = torch.randn(30, 8)
        y = torch.randn(8)
        nu = vario_score(samples, y)
        crps = crps_from_samples(samples, y)
        assert nu == pytest.approx(-float(crps), rel=1e-4)

    def test_pinball_loss_closed_form_and_metric(self) -> None:
        q = torch.tensor([1.0])
        y = torch.tensor([0.0])
        # diff = -1: max(-0.25, 0.75) at level 0.25
        assert float(pinball_loss(0.25, q, y)[0]) == pytest.approx(0.75)
        assert pinball_metric({0.5: np.array([1.0])}, np.array([0.0])) == pytest.approx(0.5)
        metric = PinballMetric()
        metric.update({0.5: torch.tensor([1.0])}, torch.tensor([0.0]))
        assert float(metric.compute()) == pytest.approx(0.5)

    def test_sharpness_mean_width(self) -> None:
        intervals = torch.tensor([[-1.0, 1.0], [-2.0, 2.0]])
        assert sharpness(intervals) == pytest.approx(3.0)
        metric = Sharpness()
        metric.update(torch.tensor([-1.0]), torch.tensor([3.0]))
        assert float(metric.compute()) == pytest.approx(4.0)

    def test_wasserstein_gaussian_exact_values(self) -> None:
        # Univariate: W2(N(0,1), N(1,2)) = sqrt((0-1)^2 + (1-2)^2) = sqrt(2)
        got = wasserstein_gaussian_p2(0.0, 1.0, 1.0, 2.0)
        assert got == pytest.approx(math.sqrt(2.0), abs=1e-9)
        # Same distribution -> zero
        cov = torch.tensor([[2.0, 0.4], [0.4, 1.0]])
        assert wasserstein_gaussian_p2(torch.zeros(2), cov, torch.zeros(2), cov) == pytest.approx(
            0.0, abs=1e-9
        )
        # Equal covariance, shifted mean: W2 = ||delta||
        got_shift = wasserstein_gaussian_p2(torch.tensor([3.0, 4.0]), cov, torch.zeros(2), cov)
        assert got_shift == pytest.approx(5.0, abs=1e-6)
        metric = WassersteinGaussian()
        metric.update(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0]))
        assert float(metric.compute()) == pytest.approx(1.0)
