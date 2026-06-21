"""
Cross-class consistency tests for Gaussian-family losses.

Verifies that all diagonal-Gaussian losses (GaussianNLLLoss, GaussianCRPSLoss,
BetaNLLLoss, FaithfulGaussianLoss) share a common input/mask/weight/reduction
contract, and that the multivariate and Wasserstein families are internally
consistent.

These tests complement (not replace) the per-class unit tests in
``test_gaussian.py``, ``test_beta_nll.py``, and ``test_faithful_gaussian.py``.
"""

import math

import pytest
import torch
from torch import nn

from tests.losses._test_losses_helpers import (
    build_diagonal_losses,
    make_base_kwargs,
    make_none_reduction,
    make_test_data,
)
from torchregress.losses.beta_nll import BetaNLLLoss
from torchregress.losses.faithful_gaussian import FaithfulGaussianLoss
from torchregress.losses.gaussian import (
    GaussianCRPSLoss,
    GaussianNLLLoss,
    LowRankGaussianLoss,
    MultivariateGaussianLoss,
)
from torchregress.losses.gaussian_wasserstein import (
    GaussianWassersteinBoundLoss,
    symmetric_spd_matrix_sqrt,
)

# ── input format contract ─────────────────────────────────────────────


class TestDiagonalInputFormat:
    """Every diagonal Gaussian loss accepts (mean, logvar) tuples and
    concatenated [mean|logvar] tensors identically."""

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    def test_accepts_tuple_input(self, name, loss_fn):
        """Tuple (mean, logvar) produces a finite scalar."""
        mean, log_var, target = make_test_data()
        out = loss_fn((mean, log_var), target)
        assert out.dim() == 0, f"{name}: expected scalar, got shape {out.shape}"
        assert torch.isfinite(out), f"{name}: loss is not finite"

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    def test_accepts_concatenated_input(self, name, loss_fn):
        """Concatenated [mean | logvar] input matches tuple result."""
        mean, log_var, target = make_test_data()
        cat = torch.cat([mean, log_var], dim=-1)
        loss_tuple = loss_fn((mean, log_var), target)
        loss_cat = loss_fn(cat, target)
        torch.testing.assert_close(loss_cat, loss_tuple, msg=f"{name}: concatenated ≠ tuple")

    def test_fixed_variance_supported_by_gaussian_nll_and_crps(self):
        """GaussianNLLLoss and GaussianCRPSLoss accept fixed_variance=…;
        BetaNLLLoss and FaithfulGaussianLoss do not (require learned variance)."""
        mean, _, target = make_test_data()
        for cls in [GaussianNLLLoss, GaussianCRPSLoss]:
            fixed = cls(fixed_variance=1.0, reduction="mean")
            out = fixed(mean, target)
            assert out.dim() == 0, f"{cls.__name__}: expected scalar"
            assert torch.isfinite(out), f"{cls.__name__}: loss not finite"


# ── mask contract ──────────────────────────────────────────────────────


class TestMaskContract:
    """Every diagonal Gaussian loss supports a boolean mask that excludes
    specific elements."""

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    def test_mask_changes_loss(self, name, loss_fn):
        """Masking out one element changes the reduced loss."""
        mean, log_var, target = make_test_data(batch=4)
        loss_full = loss_fn((mean, log_var), target)

        mask = torch.ones_like(mean, dtype=torch.bool)
        mask[0, 0] = False
        loss_masked = loss_fn((mean, log_var), target, mask=mask)

        assert loss_masked != loss_full, f"{name}: mask did not change loss"

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    def test_mask_all_false_no_crash(self, name, loss_fn):
        """Masking all elements should not raise an exception
        (NaN is acceptable — mean of empty set is undefined)."""
        mean, log_var, target = make_test_data(batch=4)
        mask = torch.zeros_like(mean, dtype=torch.bool)
        loss = loss_fn((mean, log_var), target, mask=mask)
        # Just verify it doesn't crash.  NaN is the expected result
        # for reduction='mean' with an empty masked set.
        assert isinstance(loss, torch.Tensor), f"{name}: output is not a tensor"

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    def test_mask_with_reduction_none_returns_only_unmasked(self, name, loss_fn):
        """In 'none' reduction, mask selects which elements are returned."""
        none_fn = make_none_reduction(loss_fn)
        mean, log_var, target = make_test_data(batch=5)
        n_unmasked = 7
        mask = torch.zeros_like(mean, dtype=torch.bool)
        for i in range(n_unmasked):
            row, col = i % 5, i // 5
            mask[row, col] = True

        out = none_fn((mean, log_var), target, mask=mask)
        assert out.numel() == n_unmasked, (
            f"{name}: expected {n_unmasked} elements, got {out.numel()}"
        )


# ── weights contract ───────────────────────────────────────────────────


class TestWeightsContract:
    """Every diagonal Gaussian loss supports per-sample and per-element
    weight tensors."""

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    def test_per_sample_weights_scale_loss(self, name, loss_fn):
        """Doubling a per-sample weight doubles that sample's
        contribution (verified in 'none' reduction)."""
        none_fn = make_none_reduction(loss_fn)
        mean, log_var, target = make_test_data(batch=3)
        w1 = torch.ones(3, 1)
        w2 = w1.clone()
        w2[0, 0] = 2.0
        out1 = none_fn((mean, log_var), target, weights=w1)
        out2 = none_fn((mean, log_var), target, weights=w2)
        # First sample's elements should all be doubled
        ratio = out2[0] / out1[0]
        torch.testing.assert_close(
            ratio, torch.full_like(ratio, 2.0), msg=f"{name}: weight scaling failed"
        )

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    def test_zero_weight_zeros_loss(self, name, loss_fn):
        """Zero weight → zero contribution."""
        none_fn = make_none_reduction(loss_fn)
        mean, log_var, target = make_test_data(batch=3)
        w = torch.ones(3, 1)
        w[0, 0] = 0.0
        out = none_fn((mean, log_var), target, weights=w)
        assert (out[0] == 0.0).all(), f"{name}: zero weight gave non-zero: {out[0]}"


# ── reduction contract ─────────────────────────────────────────────────


class TestReductionContract:
    """Every diagonal Gaussian loss supports 'none', 'mean', and 'sum'
    reductions, and they are internally consistent."""

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    def test_none_shape_is_per_element(self, name, loss_fn):
        none_fn = make_none_reduction(loss_fn)
        mean, log_var, target = make_test_data(batch=4, dim=3)
        out = none_fn((mean, log_var), target)
        assert out.shape == (4, 3), f"{name}: expected (4, 3), got {out.shape}"

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    def test_none_mean_equals_mean_reduction(self, name, loss_fn):
        """The mean of 'none' outputs equals the 'mean' reduction output."""
        mean, log_var, target = make_test_data(batch=4)
        base_kwargs = make_base_kwargs(loss_fn)

        fn_none = type(loss_fn)(**base_kwargs, reduction="none")
        fn_mean = type(loss_fn)(**base_kwargs, reduction="mean")

        none_out = fn_none((mean, log_var), target)
        mean_out = fn_mean((mean, log_var), target)

        # Both should produce the same scalar value.
        torch.testing.assert_close(
            none_out.mean(), mean_out, msg=f"{name}: none.mean() ≠ mean reduction"
        )

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    def test_sum_is_mean_times_element_count(self, name, loss_fn):
        """Sum reduction = mean reduction × total element count (batch × dim)."""
        mean, log_var, target = make_test_data(batch=6, dim=3)
        base_kwargs = make_base_kwargs(loss_fn)

        fn_sum = type(loss_fn)(**base_kwargs, reduction="sum")
        fn_mean = type(loss_fn)(**base_kwargs, reduction="mean")

        sum_out = fn_sum((mean, log_var), target)
        mean_out = fn_mean((mean, log_var), target)

        n_elements = 6 * 3  # batch × dim
        torch.testing.assert_close(
            sum_out / float(n_elements), mean_out, msg=f"{name}: sum/{n_elements} ≠ mean reduction"
        )


# ── gradient contract ──────────────────────────────────────────────────


class TestGradientContract:
    """All diagonal Gaussian losses produce finite gradients under
    typical and extreme input conditions."""

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    def test_finite_gradients_normal_input(self, name, loss_fn):
        mean = torch.randn(4, 2, requires_grad=True)
        log_var = torch.randn(4, 2, requires_grad=True)
        target = torch.randn(4, 2)
        loss = loss_fn((mean, log_var), target)
        loss.backward()
        assert mean.grad is not None, f"{name}: mean.grad is None"
        assert log_var.grad is not None, f"{name}: log_var.grad is None"
        assert torch.isfinite(mean.grad).all(), f"{name}: mean.grad not finite"
        assert torch.isfinite(log_var.grad).all(), f"{name}: log_var.grad not finite"

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    def test_finite_gradients_extreme_logvar(self, name, loss_fn):
        """±20 logvar should still produce finite gradients."""
        mean = torch.zeros(2, 1, requires_grad=True)
        log_var = torch.tensor([[20.0], [-20.0]], requires_grad=True)
        target = torch.ones(2, 1)
        loss = loss_fn((mean, log_var), target)
        assert torch.isfinite(loss), f"{name}: loss is not finite"
        loss.backward()
        assert torch.isfinite(mean.grad).all() if mean.grad is not None else True
        assert torch.isfinite(log_var.grad).all(), f"{name}: log_var.grad not finite"


# ── CRPS numerical stability ───────────────────────────────────────────


class TestCRPSStability:
    """GaussianCRPSLoss at extreme z (tail and center) regimes.

    Verifies the analytic formula holds at z ≈ 0 (center), |z| ≈ 6
    (moderate tail where φ is non-negligible but ndtr nears saturation),
    and |z| ≈ 20 (deep tail where φ underflows and ndtr saturates).
    """

    @staticmethod
    def _analytic_crps(mean, variance, target, eps=1e-8, dtype=torch.float64):
        """Compute the analytic CRPS formula in double precision.

        Reproduces every step of ``GaussianCRPSLoss.forward()``, including
        the double eps (in std and again in the z denominator).

        Args:
            mean, variance: Already-extracted distribution parameters
                (i.e. after ``_extract_distribution_parameters`` has been
                applied — variance is clamped to ``min_variance``).
            target: Observed values.
            eps: Epsilon for numerical stability.
            dtype: Precision for the reference computation.
        """
        variance = torch.as_tensor(variance, dtype=dtype)
        mean = torch.as_tensor(mean, dtype=dtype)
        target = torch.as_tensor(target, dtype=dtype)

        std = torch.sqrt(variance + eps)
        z = (target - mean) / (std + eps)
        cdf = torch.special.ndtr(z)  # Φ(z)
        pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)  # φ(z)
        inv_sqrt_pi = 1.0 / math.sqrt(math.pi)
        return std * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - inv_sqrt_pi)

    @pytest.mark.parametrize(
        "z_target,desc",
        [
            (0.0, "center"),
            (0.01, "near-center"),
            (6.0, "moderate positive tail"),
            (-6.0, "moderate negative tail"),
            (20.0, "deep positive tail"),
            (-20.0, "deep negative tail"),
        ],
    )
    def test_crps_matches_analytic_at_extreme_z(self, z_target, desc):
        """CRPS loss equals the analytic formula at each z regime."""
        eps = 1e-8
        min_variance = 1e-6

        # Construct exact z values: pre-scale target so that
        #   z = (target - 0) / (√(1+eps) + eps) = z_target
        exact_factor = math.sqrt(1.0 + eps) + eps
        mean = torch.zeros(4, 3)
        log_var = torch.zeros(4, 3)  # var = exp(0) = 1 (clamped ≥ min_variance)
        target = torch.full((4, 3), z_target * exact_factor)

        fn = GaussianCRPSLoss(reduction="none", eps=eps)
        crps_loss = fn((mean, log_var), target)

        # Pass actual variance (exp + clamp), not raw log-variance.
        var = torch.exp(log_var).clamp(min=min_variance)
        expected64 = self._analytic_crps(mean, var, target, eps=eps)
        expected32 = expected64.float()

        # All outputs must be finite regardless of z regime.
        assert torch.isfinite(crps_loss).all(), f"z={z_target} ({desc}): non-finite CRPS"
        assert (crps_loss >= 0).all(), f"z={z_target} ({desc}): CRPS negative"

        torch.testing.assert_close(
            crps_loss,
            expected32,
            atol=1e-5,
            rtol=1e-4,
            msg=f"z={z_target} ({desc}): loss ≠ analytic",
        )

    @staticmethod
    def _check_fd_gradients(mean, log_var, target, eps=1e-8, min_variance=1e-6, h=1e-5):
        """Verify that autodiff gradients match central finite-difference
        gradients of the closed-form analytic CRPS formula.

        Args:
            mean, log_var: float64 tensors with ``requires_grad=True``.
            target: float64 tensor.
            eps: Matches ``GaussianCRPSLoss.eps``.
            min_variance: Matches ``GaussianCRPSLoss.min_variance``.
            h: Central-difference step.
        """
        # ── autodiff gradients ────────────────────────────────────────
        fn = GaussianCRPSLoss(reduction="mean", eps=eps)
        loss = fn((mean, log_var), target)
        loss.backward()
        grad_mean_ad = mean.grad.detach().clone()
        grad_logvar_ad = log_var.grad.detach().clone()

        mean.grad = None
        log_var.grad = None

        # ── analytic CRPS as a pure (no-autograd) formula ─────────────
        def analytic_crps_value(m: torch.Tensor, lv: torch.Tensor) -> float:
            with torch.no_grad():
                var = torch.exp(lv).clamp(min=min_variance)
                std = torch.sqrt(var + eps)
                z = (target - m) / (std + eps)
                cdf = torch.special.ndtr(z)
                pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
                inv_sqrt_pi = 1.0 / math.sqrt(math.pi)
                crps = std * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - inv_sqrt_pi)
            return crps.mean().item()

        # ── finite-difference gradients ───────────────────────────────
        grad_mean_fd = torch.zeros_like(mean)
        grad_logvar_fd = torch.zeros_like(log_var)

        mean_base = mean.detach().clone()
        logvar_base = log_var.detach().clone()

        for i in range(mean.numel()):
            # Perturb mean element i
            m_plus = mean_base.clone()
            m_minus = mean_base.clone()
            m_plus.view(-1)[i] += h
            m_minus.view(-1)[i] -= h
            f_plus = analytic_crps_value(m_plus, logvar_base)
            f_minus = analytic_crps_value(m_minus, logvar_base)
            grad_mean_fd.view(-1)[i] = (f_plus - f_minus) / (2.0 * h)

            # Perturb log_var element i
            lv_plus = logvar_base.clone()
            lv_minus = logvar_base.clone()
            lv_plus.view(-1)[i] += h
            lv_minus.view(-1)[i] -= h
            f_plus = analytic_crps_value(mean_base, lv_plus)
            f_minus = analytic_crps_value(mean_base, lv_minus)
            grad_logvar_fd.view(-1)[i] = (f_plus - f_minus) / (2.0 * h)

        torch.testing.assert_close(
            grad_mean_ad,
            grad_mean_fd,
            rtol=1e-5,
            atol=1e-7,
            msg="autodiff mean grad ≠ finite-difference grad",
        )
        torch.testing.assert_close(
            grad_logvar_ad,
            grad_logvar_fd,
            rtol=1e-5,
            atol=1e-7,
            msg="autodiff log_var grad ≠ finite-difference grad",
        )

    @pytest.mark.parametrize(
        "z_target,desc",
        [
            (0.0, "center"),
            (6.0, "moderate positive tail"),
            (-6.0, "moderate negative tail"),
            (20.0, "deep positive tail"),
            (-20.0, "deep negative tail"),
        ],
    )
    def test_crps_gradients_match_finite_difference_at_z(self, z_target: float, desc: str):
        """Autodiff CRPS gradients match finite-difference gradients
        of the analytic formula at specific z regimes.

        Constructs exact z = z_target via pre-scaled targets
        (mean=0, σ=1) so that the finite-difference test runs
        cleanly at center, moderate-tail, and deep-tail values."""
        eps = 1e-8
        min_variance = 1e-6
        dtype = torch.float64
        exact_factor = math.sqrt(1.0 + eps) + eps

        mean = torch.zeros(4, 3, dtype=dtype, requires_grad=True)
        # log_var=0 → var=1, well above min_variance=1e-6 so
        # finite-difference perturbations never hit the clamp boundary.
        log_var = torch.zeros(4, 3, dtype=dtype, requires_grad=True)
        target = torch.full((4, 3), z_target * exact_factor, dtype=dtype)

        self._check_fd_gradients(mean, log_var, target, eps=eps, min_variance=min_variance)

    def test_crps_gradients_match_finite_difference_random(self):
        """Autodiff CRPS gradients match finite-difference gradients
        on random inputs for broad coverage beyond the exact-z cases."""
        torch.manual_seed(123)
        dtype = torch.float64
        eps = 1e-8
        min_variance = 1e-6

        mean = torch.randn(4, 3, dtype=dtype, requires_grad=True)
        # Keep log_var well above log(min_variance) ≈ −13.8.
        log_var = (0.5 * torch.randn(4, 3, dtype=dtype) + 0.5).requires_grad_(True)
        target = torch.randn(4, 3, dtype=dtype)

        self._check_fd_gradients(mean, log_var, target, eps=eps, min_variance=min_variance)

    @pytest.mark.parametrize(
        "z_target,desc",
        [
            (0.0, "center"),
            (6.0, "moderate positive tail"),
            (-6.0, "moderate negative tail"),
            (20.0, "deep positive tail"),
            (-20.0, "deep negative tail"),
        ],
    )
    def test_crps_gradients_are_finite_at_extreme_z(self, z_target, desc):
        """CRPS gradients are finite (not NaN) at all z regimes."""
        eps = 1e-8
        exact_factor = math.sqrt(1.0 + eps) + eps

        mean = torch.zeros(4, 3, requires_grad=True)
        log_var = torch.full((4, 3), 0.0, requires_grad=True)
        target = torch.full((4, 3), z_target * exact_factor)

        fn = GaussianCRPSLoss(reduction="mean", eps=eps)
        loss = fn((mean, log_var), target)
        loss.backward()

        assert torch.isfinite(mean.grad).all(), f"z={z_target} ({desc}): mean.grad not finite"
        assert torch.isfinite(log_var.grad).all(), f"z={z_target} ({desc}): log_var.grad not finite"


# ── numerical stability ────────────────────────────────────────────────


class TestNumericalStability:
    """Variance clamping and epsilon guards prevent NaN/Inf even with
    extreme variance values."""

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    @pytest.mark.parametrize("var_scale", [1e-8, 1e-4, 1.0, 1e4, 1e8])
    def test_extreme_variance_is_finite(self, name, loss_fn, var_scale):
        mean = torch.zeros(2, 1)
        log_var = torch.full((2, 1), math.log(var_scale))
        target = torch.ones(2, 1)
        out = loss_fn((mean, log_var), target)
        assert torch.isfinite(out), f"{name}: loss not finite at variance={var_scale}"

    @pytest.mark.parametrize("name,loss_fn", build_diagonal_losses())
    def test_nan_input_masked_is_finite(self, name, loss_fn):
        """NaN element excluded by mask → finite loss."""
        mean, log_var, target = make_test_data(batch=3)
        mean_nan = mean.clone()
        mean_nan[0, 0] = float("nan")
        mask = torch.ones_like(mean, dtype=torch.bool)
        mask[0, 0] = False
        out = loss_fn((mean_nan, log_var), target, mask=mask)
        assert torch.isfinite(out), f"{name}: loss is NaN despite mask"


# ── family relationships ───────────────────────────────────────────────


class TestFamilyRelationships:
    """Cross-class consistency: BetaNLL(beta=0) ≡ GaussianNLL,
    FaithfulGaussian reduces to decoupled NLL+MSE, etc."""

    def test_beta_zero_equals_gaussian_nll(self):
        """BetaNLLLoss(beta=0) is identical to GaussianNLLLoss."""
        mean, log_var, target = make_test_data()
        b = BetaNLLLoss(beta=0.0, reduction="mean")
        g = GaussianNLLLoss(reduction="mean")
        torch.testing.assert_close(
            b((mean, log_var), target),
            g((mean, log_var), target),
        )

    def test_beta_zero_equals_gaussian_nll_gradients(self):
        """BetaNLLLoss(beta=0) gradients match GaussianNLLLoss gradients."""
        mean = torch.randn(3, 2, requires_grad=True)
        log_var = torch.randn(3, 2, requires_grad=True)
        target = torch.randn(3, 2)

        b = BetaNLLLoss(beta=0.0, reduction="mean")
        g = GaussianNLLLoss(reduction="mean")

        loss_b = b((mean, log_var), target)
        loss_b.backward(retain_graph=True)
        gm_b, gv_b = mean.grad.clone(), log_var.grad.clone()

        mean.grad = None
        log_var.grad = None

        loss_g = g((mean, log_var), target)
        loss_g.backward()
        gm_g, gv_g = mean.grad.clone(), log_var.grad.clone()

        torch.testing.assert_close(gm_b, gm_g, msg="BetaNLL(beta=0) mean grad ≠ NLL")
        torch.testing.assert_close(gv_b, gv_g, msg="BetaNLL(beta=0) logvar grad ≠ NLL")

    def test_faithful_detaches_mean_from_nll_gradient(self):
        """With mean_weight=0, the NLL residual uses a detached mean:
        mean.grad should be None."""
        mean = torch.randn(4, 1, requires_grad=True)
        logvar = torch.randn(4, 1, requires_grad=True)
        target = torch.randn(4, 1)
        loss = FaithfulGaussianLoss(mean_weight=0.0, variance_weight=1.0)((mean, logvar), target)
        loss.backward()
        assert mean.grad is None, "mean should have no gradient from NLL term"

    def test_faithful_mse_gradient_matches_pure_mse(self):
        """With variance_weight=0, FaithfulGaussian gradient on mean
        equals the gradient of nn.MSELoss."""
        mean = torch.tensor([[1.0], [2.0], [0.0]], requires_grad=True)
        logvar = torch.zeros(3, 1, requires_grad=True)
        target = torch.tensor([[0.5], [3.0], [-1.0]])

        loss_f = FaithfulGaussianLoss(mean_weight=1.0, variance_weight=0.0)((mean, logvar), target)
        loss_f.backward()
        grad_f = mean.grad.clone()

        mean.grad = None
        loss_mse = nn.MSELoss(reduction="mean")(mean, target)
        loss_mse.backward()
        grad_mse = mean.grad.clone()

        torch.testing.assert_close(grad_f, grad_mse, msg="Faithful MSE grad ≠ pure MSE grad")

    def test_crps_is_non_negative(self):
        """GaussianCRPSLoss(reduction='none') should be ≥ 0 elementwise."""
        mean, log_var, target = make_test_data()
        fn = GaussianCRPSLoss(reduction="none")
        out = fn((mean, log_var), target)
        assert (out >= 0).all(), f"CRPS negative: min={out.min().item()}"

    def test_faithful_variance_term_equals_gaussian_nll_with_detached_mean(self):
        """FaithfulGaussianLoss(mean_weight=0, variance_weight=1)
        produces the same per-element NLL as GaussianNLLLoss when
        the mean is detached in both — the defining property of
        the 'faithful' decoupling."""
        mean, log_var, target = make_test_data(batch=5, dim=3)

        # FaithfulGaussian with mean_weight=0: NLL uses detached mean
        fg = FaithfulGaussianLoss(mean_weight=0.0, variance_weight=1.0, reduction="none")
        fg_loss = fg((mean, log_var), target)

        # GaussianNLL with manually detached mean
        gn = GaussianNLLLoss(reduction="none")
        gn_loss = gn((mean.detach(), log_var), target)

        # The variance/NLL terms should match — only gradient flow differs
        torch.testing.assert_close(
            fg_loss, gn_loss, msg="Faithful NLL term ≠ Gaussian NLL with detached mean"
        )

    def test_crps_matches_analytic_formula(self):
        """GaussianCRPSLoss equals the closed-form analytic CRPS:
        CRPS = σ · [z · (2Φ(z) − 1) + 2φ(z) − 1/√π]
        where z = (y − μ) / σ, Φ = ndtr(z), φ = standard normal PDF.

        Uses ``TestCRPSStability._analytic_crps`` as a float64
        reference so the comparison catches single-precision drift
        rather than just confirming PyTorch determinism."""
        batch, dim = 33, 5
        eps = 1e-8
        min_variance = 1e-6
        torch.manual_seed(42)
        mean = torch.randn(batch, dim)
        log_var = torch.randn(batch, dim)
        target = torch.randn(batch, dim)

        fn = GaussianCRPSLoss(reduction="none", eps=eps)
        crps_loss = fn((mean, log_var), target)

        var = torch.exp(log_var).clamp(min=min_variance)
        expected64 = TestCRPSStability._analytic_crps(mean, var, target, eps=eps)
        expected32 = expected64.float()

        torch.testing.assert_close(
            crps_loss,
            expected32,
            msg="CRPS loss ≠ analytic formula (float64 reference)",
        )

    def test_beta_nll_differs_from_gaussian_when_heteroscedastic(self):
        """With varying variance, BetaNLLLoss(beta>0) ≠ GaussianNLLLoss."""
        mean = torch.zeros(4, 1)
        log_var = torch.tensor([[0.0], [0.0], [2.0], [2.0]])
        target = torch.ones(4, 1)
        b = BetaNLLLoss(beta=0.5, reduction="mean")((mean, log_var), target)
        g = GaussianNLLLoss(reduction="mean")((mean, log_var), target)
        assert not torch.allclose(b, g), (
            "BetaNLL(0.5) should differ from NLL with heteroscedasticity"
        )


# ── multivariate relationships ─────────────────────────────────────────


def _make_spd_cov(batch, dim, jitter=1e-3):
    """Build a batch of symmetric positive-definite covariance matrices."""
    A = torch.randn(batch, dim, dim)
    return A @ A.transpose(-1, -2) + torch.eye(dim) * jitter


def _make_low_rank_params(batch, dim, rank):
    """Build cov_factor and cov_diag for LowRankGaussianLoss."""
    cov_factor = torch.randn(batch, dim, rank) * 0.5
    cov_diag = torch.rand(batch, dim) + 0.1
    return cov_factor, cov_diag


class TestMultivariateRelationships:
    """Cross-family consistency: LowRank → Multivariate when rank = dim,
    Cholesky fallback matches eigendecomposition, reduction/mask/weight
    contracts, and analytic NLL validation."""

    # ── equivalence ──────────────────────────────────────────────────

    def test_low_rank_equals_multivariate_when_rank_equals_dim(self):
        """When rank = n_features and cov_diag → 0, LowRank ≈ Multivariate."""
        batch, dim, rank = 4, 3, 3
        torch.manual_seed(1)
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)

        cov_factor = torch.randn(batch, dim, rank) * 0.5
        cov_diag = torch.full((batch, dim), 1e-6)

        lr_loss = LowRankGaussianLoss(reduction="mean")(mean, target, cov_factor, cov_diag)
        cov_full = cov_factor @ cov_factor.transpose(-1, -2) + torch.diag_embed(cov_diag)
        mv_loss = MultivariateGaussianLoss(reduction="mean")(mean, target, cov_full)

        torch.testing.assert_close(lr_loss, mv_loss, atol=1e-4, rtol=1e-2)

    def test_multivariate_matches_distribution_nll(self):
        """MultivariateGaussianLoss equals
        -torch.distributions.MultivariateNormal.log_prob."""
        batch, dim = 6, 3
        torch.manual_seed(5)
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cov = _make_spd_cov(batch, dim)

        mv = MultivariateGaussianLoss(jitter=1e-6, reduction="none")
        loss = mv(mean, target, cov)

        dist = torch.distributions.MultivariateNormal(
            mean, covariance_matrix=cov + torch.eye(dim) * 1e-6
        )
        expected = -dist.log_prob(target)

        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-4)

    def test_low_rank_matches_distribution_nll(self):
        """LowRankGaussianLoss equals
        -LowRankMultivariateNormal.log_prob (same clamping/jitter)."""
        from torch.distributions import LowRankMultivariateNormal

        batch, dim, rank = 5, 4, 2
        torch.manual_seed(6)
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cov_factor, cov_diag = _make_low_rank_params(batch, dim, rank)

        lr = LowRankGaussianLoss(reduction="none", jitter=1e-6, min_variance=1e-6)
        loss = lr(mean, target, cov_factor, cov_diag)

        # Reproduce the clamping + jitter that _prepare_low_rank applies
        diag_clamped = cov_diag.clamp(min=1e-6) + 1e-6
        dist = LowRankMultivariateNormal(mean, cov_factor=cov_factor, cov_diag=diag_clamped)
        expected = -dist.log_prob(target)

        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-4)

    # ── ill-conditioned covariance ────────────────────────────────────

    def test_multivariate_handles_ill_conditioned_covariance(self):
        """Ill-conditioned covariance with jitter produces finite
        loss and gradients."""
        batch, dim = 2, 4
        torch.manual_seed(2)
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)

        cov = torch.eye(dim).unsqueeze(0).expand(batch, -1, -1).clone()
        cov[0, 2, 2] = 1e-12

        fn = MultivariateGaussianLoss(jitter=1e-8, reduction="mean")
        loss = fn(mean, target, cov)
        assert torch.isfinite(loss), f"Near-singular cov produced {loss.item()}"
        mean.requires_grad_(True)
        loss2 = fn(mean, target, cov)
        loss2.backward()
        assert torch.isfinite(mean.grad).all()

    def test_multivariate_cholesky_equals_eigh_for_well_conditioned(self):
        """When Cholesky succeeds, forcing the eigh path produces
        the same NLL (within tolerance)."""
        batch, dim = 4, 3
        torch.manual_seed(9)
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cov = _make_spd_cov(batch, dim, jitter=5e-2)

        # Cholesky path via MultivariateGaussianLoss
        mv = MultivariateGaussianLoss(jitter=1e-8, reduction="none")
        loss_chol = mv(mean, target, cov)

        # Manually run eigh path
        cov_jit = cov + torch.eye(dim) * 1e-8
        diff = (target - mean).unsqueeze(-1)
        eigvals, eigvecs = torch.linalg.eigh(cov_jit)
        eigvals = eigvals.clamp(min=1e-8)
        log_det = torch.sum(torch.log(eigvals), dim=-1)
        whitened = torch.matmul(eigvecs.transpose(-1, -2), diff).squeeze(-1)
        quad = torch.sum(whitened**2 / eigvals, dim=-1)
        loss_eigh = 0.5 * (log_det + quad + dim * math.log(2 * math.pi))

        torch.testing.assert_close(loss_chol, loss_eigh, atol=1e-5, rtol=1e-4)

    # ── reduction contract ────────────────────────────────────────────

    def test_multivariate_reduction_consistency(self):
        batch, dim = 6, 3
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cov = _make_spd_cov(batch, dim)

        fn_none = MultivariateGaussianLoss(reduction="none")
        fn_mean = MultivariateGaussianLoss(reduction="mean")
        fn_sum = MultivariateGaussianLoss(reduction="sum")

        none = fn_none(mean, target, cov)
        assert none.shape == (batch,), f"none shape: {none.shape}"
        torch.testing.assert_close(none.mean(), fn_mean(mean, target, cov))
        torch.testing.assert_close(fn_sum(mean, target, cov) / batch, fn_mean(mean, target, cov))

    def test_low_rank_reduction_consistency(self):
        batch, dim, rank = 6, 4, 2
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cf, cd = _make_low_rank_params(batch, dim, rank)

        fn_none = LowRankGaussianLoss(reduction="none")
        fn_mean = LowRankGaussianLoss(reduction="mean")
        fn_sum = LowRankGaussianLoss(reduction="sum")

        none = fn_none(mean, target, cf, cd)
        assert none.shape == (batch,), f"none shape: {none.shape}"
        torch.testing.assert_close(none.mean(), fn_mean(mean, target, cf, cd))
        torch.testing.assert_close(
            fn_sum(mean, target, cf, cd) / batch, fn_mean(mean, target, cf, cd)
        )

    # ── mask contract ─────────────────────────────────────────────────

    def test_multivariate_mask_changes_loss(self):
        batch, dim = 5, 3
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cov = _make_spd_cov(batch, dim)
        mask = torch.ones(batch, dtype=torch.bool)
        mask[0] = False

        fn = MultivariateGaussianLoss(reduction="mean")
        loss_full = fn(mean, target, cov)
        loss_masked = fn(mean, target, cov, mask=mask)
        assert loss_masked != loss_full, "mask did not change loss"

    def test_low_rank_mask_changes_loss(self):
        batch, dim, rank = 5, 3, 2
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cf, cd = _make_low_rank_params(batch, dim, rank)
        mask = torch.ones(batch, dtype=torch.bool)
        mask[0] = False

        fn = LowRankGaussianLoss(reduction="mean")
        loss_full = fn(mean, target, cf, cd)
        loss_masked = fn(mean, target, cf, cd, mask=mask)
        assert loss_masked != loss_full, "mask did not change loss"

    def test_multivariate_mask_all_false_no_crash(self):
        batch, dim = 4, 3
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cov = _make_spd_cov(batch, dim)
        mask = torch.zeros(batch, dtype=torch.bool)

        fn = MultivariateGaussianLoss(reduction="mean")
        loss = fn(mean, target, cov, mask=mask)
        assert isinstance(loss, torch.Tensor), "output is not a tensor"

    def test_low_rank_mask_all_false_no_crash(self):
        batch, dim, rank = 4, 3, 2
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cf, cd = _make_low_rank_params(batch, dim, rank)
        mask = torch.zeros(batch, dtype=torch.bool)

        fn = LowRankGaussianLoss(reduction="mean")
        loss = fn(mean, target, cf, cd, mask=mask)
        assert isinstance(loss, torch.Tensor), "output is not a tensor"

    # ── weight contract ───────────────────────────────────────────────

    def test_multivariate_weights_double_loss(self):
        batch, dim = 4, 3
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cov = _make_spd_cov(batch, dim)
        w1 = torch.ones(batch)
        w2 = w1.clone()
        w2[0] = 2.0

        fn = MultivariateGaussianLoss(reduction="none")
        out1 = fn(mean, target, cov, weights=w1)
        out2 = fn(mean, target, cov, weights=w2)

        torch.testing.assert_close(
            out2[0] / out1[0], torch.tensor(2.0), msg="weight scaling failed"
        )

    def test_multivariate_zero_weight_zeros_loss(self):
        batch, dim = 4, 3
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cov = _make_spd_cov(batch, dim)
        w = torch.ones(batch)
        w[0] = 0.0

        fn = MultivariateGaussianLoss(reduction="none")
        out = fn(mean, target, cov, weights=w)
        assert out[0] == 0.0, f"zero weight gave {out[0].item()}"

    def test_low_rank_weights_double_loss(self):
        batch, dim, rank = 4, 3, 2
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cf, cd = _make_low_rank_params(batch, dim, rank)
        w1 = torch.ones(batch)
        w2 = w1.clone()
        w2[0] = 2.0

        fn = LowRankGaussianLoss(reduction="none")
        out1 = fn(mean, target, cf, cd, weights=w1)
        out2 = fn(mean, target, cf, cd, weights=w2)

        torch.testing.assert_close(
            out2[0] / out1[0], torch.tensor(2.0), msg="weight scaling failed"
        )

    # ── 2D (shared) covariance broadcasting ───────────────────────────

    def test_multivariate_accepts_2d_covariance(self):
        batch, dim = 5, 3
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cov_2d = _make_spd_cov(1, dim).squeeze(0)

        fn = MultivariateGaussianLoss(reduction="mean")
        loss = fn(mean, target, cov_2d)
        assert torch.isfinite(loss)

    def test_low_rank_accepts_2d_params(self):
        batch, dim, rank = 5, 4, 2
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)
        cf_2d = torch.randn(dim, rank) * 0.5
        cd_1d = torch.rand(dim) + 0.5

        fn = LowRankGaussianLoss(reduction="mean")
        loss = fn(mean, target, cf_2d, cd_1d)
        assert torch.isfinite(loss)

    # ── gradient finiteness ───────────────────────────────────────────

    def test_multivariate_finite_gradients(self):
        batch, dim = 4, 3
        mean = torch.randn(batch, dim, requires_grad=True)
        target = torch.randn(batch, dim)
        cov = _make_spd_cov(batch, dim)

        fn = MultivariateGaussianLoss(reduction="mean")
        loss = fn(mean, target, cov)
        loss.backward()
        assert mean.grad is not None
        assert torch.isfinite(mean.grad).all()

    def test_low_rank_finite_gradients(self):
        batch, dim, rank = 4, 3, 2
        mean = torch.randn(batch, dim, requires_grad=True)
        target = torch.randn(batch, dim)
        cf, cd = _make_low_rank_params(batch, dim, rank)

        fn = LowRankGaussianLoss(reduction="mean")
        loss = fn(mean, target, cf, cd)
        loss.backward()
        assert mean.grad is not None
        assert torch.isfinite(mean.grad).all()


# ── wasserstein relationships ──────────────────────────────────────────


class TestWassersteinRelationships:
    """GaussianWassersteinBoundLoss internal consistency."""

    def test_wasserstein_zero_when_perfect_match(self):
        """When pred = target (mean and cov), loss = 0."""
        batch, dim = 3, 2
        mean = torch.randn(batch, dim)
        a = torch.randn(batch, dim, dim)
        cov = a @ a.transpose(-1, -2) + torch.eye(dim) * 1e-2

        fn = GaussianWassersteinBoundLoss(
            covariance_parameterization="covariance", reduction="mean"
        )
        loss = fn(mean, mean, cov, cov)
        torch.testing.assert_close(loss, torch.zeros(()), atol=1e-5, rtol=1e-3)

    def test_wasserstein_diagonal_matches_manual(self):
        """In 'diagonal' mode, covariance term = sum_i (√v̂ᵢ - √vᵢ)²."""
        mean = torch.randn(4, 2)
        target_mean = torch.randn(4, 2)
        pred_var = (torch.rand(4, 2) + 0.1).abs()
        target_var = (torch.rand(4, 2) + 0.1).abs()

        fn = GaussianWassersteinBoundLoss(
            covariance_parameterization="diagonal",
            mean_weight=0.0,
            covariance_weight=1.0,
            reduction="none",
        )
        loss = fn(mean, target_mean, pred_var, target_var)

        expected = ((pred_var.sqrt() - target_var.sqrt()) ** 2).sum(dim=-1)
        torch.testing.assert_close(loss, expected)

    def test_wasserstein_cholesky_mode_handles_shape(self):
        """Cholesky mode accepts lower-triangular L and produces finite output."""
        batch, dim = 3, 2
        mean = torch.randn(batch, dim)

        # Create positive-definite covariances via L @ L.T
        L_pred = torch.randn(batch, dim, dim).tril()
        L_target = torch.randn(batch, dim, dim).tril()

        fn = GaussianWassersteinBoundLoss(covariance_parameterization="cholesky", reduction="mean")
        loss = fn(mean, mean, L_pred, L_target)
        assert torch.isfinite(loss)
        assert loss.dim() == 0

    def test_wasserstein_sqrt_mode_equals_covariance_mode_with_precomputed_roots(self):
        """'sqrt' mode produces the same result as 'covariance' mode
        when given the pre-computed principal matrix square roots.

        Internally, 'covariance' mode applies symmetric_spd_matrix_sqrt
        before the Frobenius comparison; 'sqrt' mode takes the roots
        directly. Passing the same roots to both should be identical."""
        batch, dim = 4, 3
        torch.manual_seed(3)
        mean = torch.randn(batch, dim)
        target_mean = torch.randn(batch, dim)

        # Build SPD covariance matrices: Σ = A @ A^T + ε·I
        a_pred = torch.randn(batch, dim, dim)
        a_target = torch.randn(batch, dim, dim)
        sigma_pred = a_pred @ a_pred.transpose(-1, -2) + torch.eye(dim) * 1e-2
        sigma_target = a_target @ a_target.transpose(-1, -2) + torch.eye(dim) * 1e-2

        # Pre-compute principal square roots (with the same jitter the loss uses)
        sqrt_pred = symmetric_spd_matrix_sqrt(sigma_pred + 1e-6 * torch.eye(dim))
        sqrt_target = symmetric_spd_matrix_sqrt(sigma_target + 1e-6 * torch.eye(dim))

        loss_cov = GaussianWassersteinBoundLoss(
            covariance_parameterization="covariance",
            mean_weight=1.0,
            covariance_weight=1.0,
            reduction="mean",
        )(mean, target_mean, sigma_pred, sigma_target)

        loss_sqrt = GaussianWassersteinBoundLoss(
            covariance_parameterization="sqrt",
            mean_weight=1.0,
            covariance_weight=1.0,
            reduction="mean",
        )(mean, target_mean, sqrt_pred, sqrt_target)

        torch.testing.assert_close(
            loss_sqrt, loss_cov, msg="sqrt mode ≠ covariance mode with pre-computed roots"
        )

    def test_wasserstein_sqrt_mode_equals_cholesky_mode_with_precomputed_roots(self):
        """'sqrt' mode produces the same result as 'cholesky' mode
        when given the pre-computed principal matrix square roots
        derived from the same Cholesky factors.

        'cholesky' mode internally does L → L@L^T → sqrt → Frobenius.
        'sqrt' mode takes the pre-computed roots directly.  Feeding
        the same roots to both paths should yield identical loss."""
        batch, dim = 4, 3
        torch.manual_seed(7)
        mean = torch.randn(batch, dim)
        target_mean = torch.randn(batch, dim)

        # Build lower-triangular Cholesky factors
        L_pred = torch.randn(batch, dim, dim).tril()
        L_target = torch.randn(batch, dim, dim).tril()

        # Reconstruct SPD covariances: Σ = L @ L^T
        sigma_pred = L_pred @ L_pred.transpose(-1, -2)
        sigma_target = L_target @ L_target.transpose(-1, -2)

        # Pre-compute principal square roots with the same jitter the loss uses
        jitter = 1e-6
        sqrt_pred = symmetric_spd_matrix_sqrt(sigma_pred + jitter * torch.eye(dim))
        sqrt_target = symmetric_spd_matrix_sqrt(sigma_target + jitter * torch.eye(dim))

        loss_chol = GaussianWassersteinBoundLoss(
            covariance_parameterization="cholesky",
            mean_weight=1.0,
            covariance_weight=1.0,
            reduction="mean",
        )(mean, target_mean, L_pred, L_target)

        loss_sqrt = GaussianWassersteinBoundLoss(
            covariance_parameterization="sqrt",
            mean_weight=1.0,
            covariance_weight=1.0,
            reduction="mean",
        )(mean, target_mean, sqrt_pred, sqrt_target)

        torch.testing.assert_close(
            loss_sqrt, loss_chol, msg="sqrt mode ≠ cholesky mode with pre-computed roots"
        )
