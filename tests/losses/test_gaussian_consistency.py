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
from typing import Tuple

import pytest
import torch
from torch import nn

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

# ── helpers ───────────────────────────────────────────────────────────

_DIAGONAL_LOSS_FACTORIES = [
    lambda: GaussianNLLLoss(reduction="mean"),
    lambda: GaussianCRPSLoss(reduction="mean"),
    lambda: BetaNLLLoss(beta=0.3, reduction="mean"),
    lambda: FaithfulGaussianLoss(mean_weight=1.0, variance_weight=1.0, reduction="mean"),
]

_DIAGONAL_LOSS_NAMES = ["GaussianNLL", "GaussianCRPS", "BetaNLL", "FaithfulGaussian"]


def _make_base_kwargs(loss_fn):
    """Return the constructor kwargs (excluding reduction) for a loss instance."""
    kwargs: dict = {}
    if hasattr(loss_fn, "beta"):
        kwargs["beta"] = loss_fn.beta
    if hasattr(loss_fn, "mean_weight"):
        kwargs["mean_weight"] = loss_fn.mean_weight
        kwargs["variance_weight"] = loss_fn.variance_weight
    return kwargs


def _make_none_reduction(loss_fn):
    """Return a new instance of the same loss class with reduction='none'."""
    return type(loss_fn)(**_make_base_kwargs(loss_fn), reduction="none")


def _make_test_data(
    batch: int = 8,
    dim: int = 3,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    mean = torch.randn(batch, dim)
    log_var = torch.randn(batch, dim) * 0.5
    target = torch.randn(batch, dim)
    return mean, log_var, target


def _all_diagonal_losses():
    """Yield (name, loss_fn) pairs for parametrized tests."""
    return list(zip(_DIAGONAL_LOSS_NAMES, [f() for f in _DIAGONAL_LOSS_FACTORIES]))


# ── input format contract ─────────────────────────────────────────────


class TestDiagonalInputFormat:
    """Every diagonal Gaussian loss accepts (mean, logvar) tuples and
    concatenated [mean|logvar] tensors identically."""

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
    def test_accepts_tuple_input(self, name, loss_fn):
        """Tuple (mean, logvar) produces a finite scalar."""
        mean, log_var, target = _make_test_data()
        out = loss_fn((mean, log_var), target)
        assert out.dim() == 0, f"{name}: expected scalar, got shape {out.shape}"
        assert torch.isfinite(out), f"{name}: loss is not finite"

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
    def test_accepts_concatenated_input(self, name, loss_fn):
        """Concatenated [mean | logvar] input matches tuple result."""
        mean, log_var, target = _make_test_data()
        cat = torch.cat([mean, log_var], dim=-1)
        loss_tuple = loss_fn((mean, log_var), target)
        loss_cat = loss_fn(cat, target)
        torch.testing.assert_close(loss_cat, loss_tuple, msg=f"{name}: concatenated ≠ tuple")

    def test_fixed_variance_supported_by_gaussian_nll_and_crps(self):
        """GaussianNLLLoss and GaussianCRPSLoss accept fixed_variance=…;
        BetaNLLLoss and FaithfulGaussianLoss do not (require learned variance)."""
        mean, _, target = _make_test_data()
        for cls in [GaussianNLLLoss, GaussianCRPSLoss]:
            fixed = cls(fixed_variance=1.0, reduction="mean")
            out = fixed(mean, target)
            assert out.dim() == 0, f"{cls.__name__}: expected scalar"
            assert torch.isfinite(out), f"{cls.__name__}: loss not finite"


# ── mask contract ──────────────────────────────────────────────────────


class TestMaskContract:
    """Every diagonal Gaussian loss supports a boolean mask that excludes
    specific elements."""

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
    def test_mask_changes_loss(self, name, loss_fn):
        """Masking out one element changes the reduced loss."""
        mean, log_var, target = _make_test_data(batch=4)
        loss_full = loss_fn((mean, log_var), target)

        mask = torch.ones_like(mean, dtype=torch.bool)
        mask[0, 0] = False
        loss_masked = loss_fn((mean, log_var), target, mask=mask)

        assert loss_masked != loss_full, f"{name}: mask did not change loss"

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
    def test_mask_all_false_no_crash(self, name, loss_fn):
        """Masking all elements should not raise an exception
        (NaN is acceptable — mean of empty set is undefined)."""
        mean, log_var, target = _make_test_data(batch=4)
        mask = torch.zeros_like(mean, dtype=torch.bool)
        loss = loss_fn((mean, log_var), target, mask=mask)
        # Just verify it doesn't crash.  NaN is the expected result
        # for reduction='mean' with an empty masked set.
        assert isinstance(loss, torch.Tensor), f"{name}: output is not a tensor"

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
    def test_mask_with_reduction_none_returns_only_unmasked(self, name, loss_fn):
        """In 'none' reduction, mask selects which elements are returned."""
        none_fn = _make_none_reduction(loss_fn)
        mean, log_var, target = _make_test_data(batch=5)
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

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
    def test_per_sample_weights_scale_loss(self, name, loss_fn):
        """Doubling a per-sample weight doubles that sample's
        contribution (verified in 'none' reduction)."""
        none_fn = _make_none_reduction(loss_fn)
        mean, log_var, target = _make_test_data(batch=3)
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

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
    def test_zero_weight_zeros_loss(self, name, loss_fn):
        """Zero weight → zero contribution."""
        none_fn = _make_none_reduction(loss_fn)
        mean, log_var, target = _make_test_data(batch=3)
        w = torch.ones(3, 1)
        w[0, 0] = 0.0
        out = none_fn((mean, log_var), target, weights=w)
        assert (out[0] == 0.0).all(), f"{name}: zero weight gave non-zero: {out[0]}"


# ── reduction contract ─────────────────────────────────────────────────


class TestReductionContract:
    """Every diagonal Gaussian loss supports 'none', 'mean', and 'sum'
    reductions, and they are internally consistent."""

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
    def test_none_shape_is_per_element(self, name, loss_fn):
        none_fn = _make_none_reduction(loss_fn)
        mean, log_var, target = _make_test_data(batch=4, dim=3)
        out = none_fn((mean, log_var), target)
        assert out.shape == (4, 3), f"{name}: expected (4, 3), got {out.shape}"

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
    def test_none_mean_equals_mean_reduction(self, name, loss_fn):
        """The mean of 'none' outputs equals the 'mean' reduction output."""
        mean, log_var, target = _make_test_data(batch=4)
        base_kwargs = _make_base_kwargs(loss_fn)

        fn_none = type(loss_fn)(**base_kwargs, reduction="none")
        fn_mean = type(loss_fn)(**base_kwargs, reduction="mean")

        none_out = fn_none((mean, log_var), target)
        mean_out = fn_mean((mean, log_var), target)

        # Both should produce the same scalar value.
        torch.testing.assert_close(
            none_out.mean(), mean_out, msg=f"{name}: none.mean() ≠ mean reduction"
        )

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
    def test_sum_is_mean_times_element_count(self, name, loss_fn):
        """Sum reduction = mean reduction × total element count (batch × dim)."""
        mean, log_var, target = _make_test_data(batch=6, dim=3)
        base_kwargs = _make_base_kwargs(loss_fn)

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

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
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

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
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


# ── numerical stability ────────────────────────────────────────────────


class TestNumericalStability:
    """Variance clamping and epsilon guards prevent NaN/Inf even with
    extreme variance values."""

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
    @pytest.mark.parametrize("var_scale", [1e-8, 1e-4, 1.0, 1e4, 1e8])
    def test_extreme_variance_is_finite(self, name, loss_fn, var_scale):
        mean = torch.zeros(2, 1)
        log_var = torch.full((2, 1), math.log(var_scale))
        target = torch.ones(2, 1)
        out = loss_fn((mean, log_var), target)
        assert torch.isfinite(out), f"{name}: loss not finite at variance={var_scale}"

    @pytest.mark.parametrize("name,loss_fn", _all_diagonal_losses())
    def test_nan_input_masked_is_finite(self, name, loss_fn):
        """NaN element excluded by mask → finite loss."""
        mean, log_var, target = _make_test_data(batch=3)
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
        mean, log_var, target = _make_test_data()
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
        mean, log_var, target = _make_test_data()
        fn = GaussianCRPSLoss(reduction="none")
        out = fn((mean, log_var), target)
        assert (out >= 0).all(), f"CRPS negative: min={out.min().item()}"

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


class TestMultivariateRelationships:
    """Cross-family consistency: LowRank → Multivariate when rank = dim,
    Cholesky fallback matches eigendecomposition."""

    def test_low_rank_equals_multivariate_when_rank_equals_dim(self):
        """When rank = n_features and cov_diag → 0, LowRank ≈ Multivariate."""
        batch, dim, rank = 4, 3, 3
        torch.manual_seed(1)
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)

        # Build a valid low-rank parameterization that gives a full-rank cov
        cov_factor = torch.randn(batch, dim, rank) * 0.5
        cov_diag = torch.full((batch, dim), 1e-6)

        lr_loss = LowRankGaussianLoss(reduction="mean")(mean, target, cov_factor, cov_diag)

        # Reconstruct full covariance: cov_factor @ cov_factor.T + diag(cov_diag)
        cov_full = cov_factor @ cov_factor.transpose(-1, -2) + torch.diag_embed(cov_diag)
        mv_loss = MultivariateGaussianLoss(reduction="mean")(mean, target, cov_full)

        torch.testing.assert_close(lr_loss, mv_loss, atol=1e-4, rtol=1e-2)

    def test_multivariate_handles_ill_conditioned_covariance(self):
        """Ill-conditioned (near-singular) covariance with jitter
        produces finite loss and gradients."""
        batch, dim = 2, 4
        torch.manual_seed(2)
        mean = torch.randn(batch, dim)
        target = torch.randn(batch, dim)

        # Create a near-singular covariance: one eigenvalue ≈ 0
        cov = torch.eye(dim).unsqueeze(0).expand(batch, -1, -1).clone()
        # Set one eigenvalue to near-zero
        cov[0, 2, 2] = 1e-12

        fn = MultivariateGaussianLoss(jitter=1e-8, reduction="mean")
        loss = fn(mean, target, cov)
        assert torch.isfinite(loss), f"Near-singular cov produced {loss.item()}"
        # Should also backward
        mean.requires_grad_(True)
        loss2 = fn(mean, target, cov)
        loss2.backward()
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
