from __future__ import annotations

import math

import pytest
import torch
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

        # Check parameter division
        y_pred = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        gamma, nu, alpha, beta = loss_fn._extract_nig_parameters(y_pred)
        assert torch.allclose(gamma, torch.tensor([[1.0], [5.0]]))
        assert torch.allclose(nu, torch.tensor([[2.0], [6.0]]))
        assert torch.allclose(alpha, torch.tensor([[3.0], [7.0]]))
        assert torch.allclose(beta, torch.tensor([[4.0], [8.0]]))

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

        # Expected loss per element (unreduced)
        nll0 = loss_fn._nig_nll(
            target[0:1], y_pred[0:1, :1], y_pred[0:1, 1:2], y_pred[0:1, 2:3], y_pred[0:1, 3:]
        )
        reg0 = loss_fn._nig_regularizer(
            target[0:1], y_pred[0:1, :1], y_pred[0:1, 1:2], y_pred[0:1, 2:3], y_pred[0:1, 3:]
        )
        loss0 = nll0 + 0.01 * reg0

        nll1 = loss_fn._nig_nll(
            target[1:2], y_pred[1:2, :1], y_pred[1:2, 1:2], y_pred[1:2, 2:3], y_pred[1:2, 3:]
        )
        reg1 = loss_fn._nig_regularizer(
            target[1:2], y_pred[1:2, :1], y_pred[1:2, 1:2], y_pred[1:2, 2:3], y_pred[1:2, 3:]
        )
        loss1 = nll1 + 0.01 * reg1

        nll2 = loss_fn._nig_nll(
            target[2:3], y_pred[2:3, :1], y_pred[2:3, 1:2], y_pred[2:3, 2:3], y_pred[2:3, 3:]
        )
        reg2 = loss_fn._nig_regularizer(
            target[2:3], y_pred[2:3, :1], y_pred[2:3, 1:2], y_pred[2:3, 2:3], y_pred[2:3, 3:]
        )
        loss2 = nll2 + 0.01 * reg2

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

        # Mean = gamma = 1.0
        # Aleatoric = beta / (alpha - 1) = 4.0 / (3.0 - 1.0) = 2.0
        # Epistemic = beta / (nu * (alpha - 1)) = 4.0 / (2.0 * (3.0 - 1.0)) = 1.0
        mean, ale, epi = loss_fn.predict_with_uncertainty(y_pred)
        assert mean.item() == pytest.approx(1.0, rel=1e-5)
        assert ale.item() == pytest.approx(2.0, rel=1e-5)
        assert epi.item() == pytest.approx(1.0, rel=1e-5)

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

        # Student-t prediction interval params:
        # gamma = 1.5, nu = 3.0, alpha = 4.0, beta = 2.0
        # df = 2 * alpha = 8
        # scale = sqrt(beta * (1 + 1/nu) / (alpha + 1e-6))
        df = 8.0
        scale = math.sqrt(2.0 * (1.0 + 1.0 / 3.0) / (4.0 + 1e-6))
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

        # total_std = sqrt(aleatoric + epistemic)
        # ale = 2.0 / (4.0 - 1.0) = 2/3
        # epi = 2.0 / (3.0 * (4.0 - 1.0)) = 2/9
        # var = 2/3 + 2/9 = 8/9 ≈ 0.88888888
        # std = sqrt(8/9) ≈ 0.94280904
        # z_val for 95% = 1.95996398
        std = math.sqrt(8.0 / 9.0)
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
