from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

from torchregress.losses import EvidentialRegressionLoss


class TestEvidentialRegressionLoss:
    def test_init_validation(self) -> None:
        # Valid init
        loss_fn = EvidentialRegressionLoss(coeff_nig=0.05)
        assert loss_fn.coeff_nig == 0.05

        # Invalid coefficient
        with pytest.raises(ValueError, match="coeff_nig must be >= 0"):
            EvidentialRegressionLoss(coeff_nig=-0.1)

    def test_parameter_extraction(self) -> None:
        loss_fn = EvidentialRegressionLoss()

        # Input shape validation
        with pytest.raises(ValueError, match="requires 4 outputs per target dimension"):
            loss_fn(torch.randn(5, 3), torch.randn(5, 1))

        # A4: raw splits are constrained — gamma raw; nu/alpha/beta via
        # softplus(x) + {0.01, 1.01, 0.01}
        y_pred = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        gamma, nu, alpha, beta = loss_fn._extract_nig_parameters(y_pred)
        assert torch.allclose(gamma, torch.tensor([[1.0], [5.0]]))
        assert torch.allclose(nu, F.softplus(torch.tensor([[2.0], [6.0]])) + 0.01)
        assert torch.allclose(alpha, F.softplus(torch.tensor([[3.0], [7.0]])) + 1.01)
        assert torch.allclose(beta, F.softplus(torch.tensor([[4.0], [8.0]])) + 0.01)
        assert (nu > 0).all() and (alpha > 1).all() and (beta > 0).all()

    def test_nig_nll_mathematical_correctness(self) -> None:
        loss_fn = EvidentialRegressionLoss()

        # Test values
        gamma = torch.tensor([[0.5]])
        nu = torch.tensor([[2.0]])
        alpha = torch.tensor([[3.0]])
        beta = torch.tensor([[1.5]])
        target = torch.tensor([[1.0]])

        # Manual student-t NLL calculation:
        # NLL = 0.5 * log(pi / nu) - alpha * log(2 * beta) + (alpha + 0.5) * log(nu * (y - gamma)^2 + 2 * beta)
        #       + lgamma(alpha) - lgamma(alpha + 0.5)
        residual_sq = (1.0 - 0.5) ** 2
        term1 = 0.5 * math.log(math.pi / 2.0)
        term2 = -3.0 * math.log(2.0 * 1.5 + 1e-6)
        term3 = (3.0 + 0.5) * math.log(2.0 * residual_sq + 2.0 * 1.5 + 1e-6)
        term4 = math.lgamma(3.0) - math.lgamma(3.5)
        expected_nll = term1 + term2 + term3 + term4

        actual_nll = loss_fn._nig_nll(target, gamma, nu, alpha, beta)
        assert math.isclose(actual_nll.item(), expected_nll, rel_tol=1e-5)

    def test_nig_regularizer(self) -> None:
        loss_fn = EvidentialRegressionLoss(coeff_nig=0.1)
        gamma = torch.tensor([[1.0]])
        nu = torch.tensor([[2.0]])
        alpha = torch.tensor([[3.0]])
        beta = torch.tensor([[4.0]])
        target = torch.tensor([[3.0]])

        # regularizer = |target - gamma| * (2 * nu + alpha)
        # = |3.0 - 1.0| * (2 * 2.0 + 3.0) = 2.0 * 7.0 = 14.0
        expected_reg = 14.0
        actual_reg = loss_fn._nig_regularizer(target, gamma, nu, alpha, beta)
        assert math.isclose(actual_reg.item(), expected_reg, rel_tol=1e-5)

    def test_forward_with_weights_and_mask(self) -> None:
        loss_fn = EvidentialRegressionLoss(coeff_nig=0.01)

        # Multiple samples
        y_pred = torch.tensor(
            [[0.5, 2.0, 3.0, 1.5], [1.2, 4.0, 2.0, 2.0], [0.0, 1.0, 5.0, 1.0]], dtype=torch.float32
        )
        target = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float32)

        # A4: forward constrains raw outputs before the NLL; mirror that here
        g, nu_, al, be = loss_fn._extract_nig_parameters(y_pred)
        loss0 = loss_fn._nig_nll(target[0:1], g[0:1], nu_[0:1], al[0:1], be[0:1]) + (
            0.01 * loss_fn._nig_regularizer(target[0:1], g[0:1], nu_[0:1], al[0:1], be[0:1])
        )
        loss1 = loss_fn._nig_nll(target[1:2], g[1:2], nu_[1:2], al[1:2], be[1:2]) + (
            0.01 * loss_fn._nig_regularizer(target[1:2], g[1:2], nu_[1:2], al[1:2], be[1:2])
        )
        loss2 = loss_fn._nig_nll(target[2:3], g[2:3], nu_[2:3], al[2:3], be[2:3]) + (
            0.01 * loss_fn._nig_regularizer(target[2:3], g[2:3], nu_[2:3], al[2:3], be[2:3])
        )

        # 1. Reduction None
        loss_fn.reduction = "none"
        actual_none = loss_fn(y_pred, target)
        expected_none = torch.cat([loss0, loss1, loss2], dim=0)
        assert torch.allclose(actual_none, expected_none)

        # 2. Reduction Mean with Weights
        loss_fn.reduction = "mean"
        weights = torch.tensor([0.5, 2.0, 1.0])
        actual_weighted = loss_fn(y_pred, target, weights=weights)
        expected_weighted = (loss0 * 0.5 + loss1 * 2.0 + loss2 * 1.0) / 3.5
        assert torch.allclose(actual_weighted, expected_weighted)

        # 3. Reduction Mean with Mask
        mask = torch.tensor([[True], [False], [True]])
        actual_masked = loss_fn(y_pred, target, mask=mask)
        expected_masked = (loss0 + loss2) / 2.0
        assert torch.allclose(actual_masked, expected_masked)

    def test_predict_with_uncertainty(self) -> None:
        loss_fn = EvidentialRegressionLoss()
        y_pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        # A4: raw outputs are constrained internally; compute closed form with
        # nu = softplus(2)+0.01, alpha = softplus(3)+1.01, beta = softplus(4)+0.01
        nu_c = F.softplus(torch.tensor(2.0)).item() + 0.01
        alpha_c = F.softplus(torch.tensor(3.0)).item() + 1.01
        beta_c = F.softplus(torch.tensor(4.0)).item() + 0.01
        mean, ale, epi = loss_fn.predict_with_uncertainty(y_pred)
        assert mean.item() == pytest.approx(1.0, rel=1e-5)
        assert ale.item() == pytest.approx(beta_c / (alpha_c - 1.0), rel=1e-5)
        assert epi.item() == pytest.approx(beta_c / (nu_c * (alpha_c - 1.0)), rel=1e-5)

    def test_sample_predictions(self) -> None:
        loss_fn = EvidentialRegressionLoss()
        y_pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        samples = loss_fn.sample_predictions(y_pred, n_samples=1000)
        assert samples.shape == (1000, 1, 1)
        # Basic statistic sanity check
        assert 0.0 < samples.mean().item() < 2.0

    def test_predict_interval_student_t(self) -> None:
        loss_fn = EvidentialRegressionLoss()
        y_pred = torch.tensor(
            [[1.5, 3.0, 4.0, 2.0]], dtype=torch.float64
        )  # Use float64 for high accuracy

        # A4: constrained params — gamma = 1.5 (raw),
        # nu = softplus(3)+0.01, alpha = softplus(4)+1.01, beta = softplus(2)+0.01
        nu_c = F.softplus(torch.tensor(3.0, dtype=torch.float64)).item() + 0.01
        alpha_c = F.softplus(torch.tensor(4.0, dtype=torch.float64)).item() + 1.01
        beta_c = F.softplus(torch.tensor(2.0, dtype=torch.float64)).item() + 0.01
        df = 2.0 * alpha_c
        scale = math.sqrt(beta_c * (1.0 + 1.0 / nu_c) / alpha_c)
        conf = 0.90
        t_val = scipy_stats.t.ppf((1.0 + conf) / 2.0, df)

        expected_lower = 1.5 - t_val * scale
        expected_upper = 1.5 + t_val * scale

        lower, upper = loss_fn.predict_interval(y_pred, confidence=conf)
        assert torch.allclose(
            lower, torch.tensor([[expected_lower]], dtype=torch.float64), rtol=1e-6
        )
        assert torch.allclose(
            upper, torch.tensor([[expected_upper]], dtype=torch.float64), rtol=1e-6
        )

    def test_predict_interval_gaussian(self) -> None:
        loss_fn = EvidentialRegressionLoss()
        y_pred = torch.tensor([[1.5, 3.0, 4.0, 2.0]])

        # A4: constrained params (see test above); ale = beta/(alpha-1),
        # epi = beta/(nu*(alpha-1)), std = sqrt(ale + epi)
        nu_c = F.softplus(torch.tensor(3.0)).item() + 0.01
        alpha_c = F.softplus(torch.tensor(4.0)).item() + 1.01
        beta_c = F.softplus(torch.tensor(2.0)).item() + 0.01
        std = math.sqrt(beta_c / (alpha_c - 1.0) * (1.0 + 1.0 / nu_c))
        z_val = scipy_stats.norm.ppf((1.0 + 0.95) / 2.0)  # 0.95 confidence
        expected_lower = 1.5 - z_val * std
        expected_upper = 1.5 + z_val * std

        lower, upper = loss_fn.predict_interval_gaussian(y_pred, confidence=0.95)
        assert torch.allclose(
            lower, torch.tensor([[expected_lower]], dtype=y_pred.dtype), rtol=1e-5
        )
        assert torch.allclose(
            upper, torch.tensor([[expected_upper]], dtype=y_pred.dtype), rtol=1e-5
        )

    def test_default_path_unchanged_fixed_seed(self) -> None:
        # TR-COR-08: the default constructor must reproduce the legacy
        # softplus(+offset) behavior exactly on fixed-seed inputs.
        torch.manual_seed(0)
        y_pred = torch.randn(8, 4)
        target = torch.randn(8, 1)

        loss_fn = EvidentialRegressionLoss(coeff_nig=0.01)
        actual = loss_fn(y_pred, target)

        # Independent reference of the legacy path.
        gamma = y_pred[:, 0:1]
        nu = F.softplus(y_pred[:, 1:2]) + 0.01
        alpha = F.softplus(y_pred[:, 2:3]) + 1.01
        beta = F.softplus(y_pred[:, 3:4]) + 0.01
        residual_sq = (target - gamma) ** 2
        nll = (
            0.5 * torch.log(math.pi / nu)
            - alpha * torch.log(2.0 * beta)
            + (alpha + 0.5) * torch.log(nu * residual_sq + 2.0 * beta)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        reg = torch.abs(target - gamma) * (2.0 * nu + alpha)
        expected = (nll + 0.01 * reg).mean()

        assert torch.allclose(actual, expected, rtol=1e-6)

    def test_unconstrained_inputs_matches_hand_computed_nig_nll(self) -> None:
        # TR-COR-08: unconstrained_inputs=False consumes pre-constrained
        # parameters directly. Hand-computed NIG NLL for
        # gamma=0.5, nu=2, alpha=3, beta=1, target=1.0.
        loss_fn = EvidentialRegressionLoss(coeff_nig=0.01, unconstrained_inputs=False)
        gamma = torch.tensor([[0.5]])
        nu = torch.tensor([[2.0]])
        alpha = torch.tensor([[3.0]])
        beta = torch.tensor([[1.0]])
        target = torch.tensor([[1.0]])

        loss = loss_fn((gamma, nu, alpha, beta), target)

        r = 0.5
        nll = (
            0.5 * math.log(math.pi / 2.0)
            - 3.0 * math.log(2.0 * 1.0)
            + 3.5 * math.log(2.0 * r**2 + 2.0 * 1.0)
            + math.lgamma(3.0)
            - math.lgamma(3.5)
        )
        reg = abs(target.item() - 0.5) * (2.0 * 2.0 + 3.0)
        expected = nll + 0.01 * reg

        assert math.isclose(loss.item(), expected, rel_tol=1e-6)

    def test_unconstrained_inputs_tuple_equals_stacked(self) -> None:
        # TR-COR-08: tuple input must equal stacked-tensor input in both modes.
        torch.manual_seed(1)
        y_pred = torch.randn(6, 8)
        target = torch.randn(6, 2)

        for unconstrained in (True, False):
            loss_fn = EvidentialRegressionLoss(coeff_nig=0.05, unconstrained_inputs=unconstrained)
            stacked = loss_fn(y_pred, target)
            as_tuple = loss_fn(
                (
                    y_pred[..., :2],
                    y_pred[..., 2:4],
                    y_pred[..., 4:6],
                    y_pred[..., 6:],
                ),
                target,
            )
            assert torch.allclose(stacked, as_tuple, rtol=1e-6, atol=1e-8)

    def test_backward_produces_grads_both_modes(self) -> None:
        # TR-COR-08: backward pass produces gradients in both input modes.
        for unconstrained in (True, False):
            loss_fn = EvidentialRegressionLoss(coeff_nig=0.01, unconstrained_inputs=unconstrained)
            if unconstrained:
                y_pred = torch.randn(4, 4, requires_grad=True)
                loss = loss_fn(y_pred, torch.randn(4, 1))
            else:
                gamma = torch.randn(4, 1, requires_grad=True)
                nu = torch.rand(4, 1) + 0.5
                alpha = torch.rand(4, 1) + 1.5
                beta = torch.rand(4, 1) + 0.5
                params = [p.requires_grad_(True) for p in (nu, alpha, beta)]
                loss = loss_fn((gamma, *params), torch.randn(4, 1))

            loss.backward()
            produced = [
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in ([y_pred] if unconstrained else [gamma, *params])
            ]
            assert all(produced)
