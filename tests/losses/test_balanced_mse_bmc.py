"""Tests for the noise-adaptive Balanced MSE (BMC) loss."""

import math

import numpy as np
import pytest
import torch

from torchregress.losses.balanced_mse_bmc import BalancedMSEBMC, balanced_mse_bmc_loss
from torchregress.losses.loss_registry import create_loss_from_config


def _reference_bmc(
    y_pred: torch.Tensor,
    target: torch.Tensor,
    log_prior: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Independent NumPy reference for the prior-weighted BMC per-sample loss."""
    pred = y_pred.detach().cpu().numpy().astype(np.float64).reshape(-1)
    y = target.detach().cpu().numpy().astype(np.float64).reshape(-1)
    logp = log_prior.detach().cpu().numpy().astype(np.float64).reshape(-1)
    out = np.empty(pred.size, dtype=np.float64)
    for i in range(pred.size):
        terms = logp - (pred[i] - y) ** 2 / (2.0 * sigma**2)
        shift = terms.max()
        log_normaliser = shift + np.log(np.exp(terms - shift).sum())
        residual = (pred[i] - y[i]) ** 2 / (2.0 * sigma**2)
        out[i] = -logp[i] + residual + log_normaliser
    return torch.from_numpy(out)


def _paper_bmc_reference(
    y_pred: torch.Tensor,
    target: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """The paper's BMC loss (uniform prior), independent NumPy implementation."""
    pred = y_pred.detach().cpu().numpy().astype(np.float64).reshape(-1)
    y = target.detach().cpu().numpy().astype(np.float64).reshape(-1)
    out = np.empty(pred.size, dtype=np.float64)
    for i in range(pred.size):
        terms = -((pred[i] - y) ** 2) / (2.0 * sigma**2)
        shift = terms.max()
        log_normaliser = shift + np.log(np.exp(terms - shift).sum())
        out[i] = (pred[i] - y[i]) ** 2 / (2.0 * sigma**2) + log_normaliser
    return torch.from_numpy(out)


class TestHandFormula:
    def test_functional_matches_numpy_reference(self) -> None:
        y_pred = torch.tensor([0.1, -0.4, 1.2, 0.7, -0.9], dtype=torch.float64)
        target = torch.tensor([0.0, -0.5, 1.0, 0.8, -0.8], dtype=torch.float64)
        log_prior = torch.tensor([-0.2, 0.1, -1.3, 0.4, -0.7], dtype=torch.float64)
        sigma = 0.37
        expected = _reference_bmc(y_pred, target, log_prior, sigma)
        got = balanced_mse_bmc_loss(y_pred, target, log_prior, sigma, reduction="none")
        assert torch.allclose(got, expected, rtol=1e-6, atol=1e-8)

    def test_class_matches_numpy_reference_with_fitted_prior(self) -> None:
        target = torch.tensor([0.0, 0.2, 0.4, 0.8, 3.0], dtype=torch.float64)
        y_pred = torch.tensor([0.1, 0.0, 0.5, 0.9, 2.5], dtype=torch.float64)
        sigma = 0.6
        loss_fn = BalancedMSEBMC(sigma=sigma, prior="histogram", num_bins=8, reduction="none").fit(
            target
        )
        log_prior = loss_fn.log_prior(target)
        expected = _reference_bmc(y_pred, target, log_prior, sigma)
        assert torch.allclose(loss_fn(y_pred, target), expected, rtol=1e-6, atol=1e-8)

    def test_uniform_prior_reduces_to_paper_bmc(self) -> None:
        target = torch.tensor([-1.0, -0.2, 0.3, 1.0, 2.5], dtype=torch.float64)
        y_pred = torch.tensor([-0.8, -0.4, 0.1, 1.1, 2.0], dtype=torch.float64)
        sigma = 0.45
        loss_fn = BalancedMSEBMC(sigma=sigma, prior="uniform", reduction="none").fit(target)
        expected = _paper_bmc_reference(y_pred, target, sigma)
        assert torch.allclose(loss_fn(y_pred, target), expected, rtol=1e-6, atol=1e-8)
        assert torch.equal(loss_fn.log_prior(target), torch.zeros_like(target))

    def test_prior_shift_invariance(self) -> None:
        """Only differences of log p matter; adding a constant cannot change L."""
        y_pred = torch.tensor([0.1, 0.4, -0.3, 1.0])
        target = torch.tensor([0.0, 0.5, -0.2, 0.9])
        log_prior = torch.tensor([-0.4, 0.2, -0.7, 0.1])
        base = balanced_mse_bmc_loss(y_pred, target, log_prior, 0.5, reduction="none")
        shifted = balanced_mse_bmc_loss(y_pred, target, log_prior + 2.5, 0.5, reduction="none")
        assert torch.allclose(base, shifted, rtol=1e-6, atol=1e-8)


class TestGradientFlow:
    def test_gradient_flows_to_predictions(self) -> None:
        target = torch.linspace(-1.0, 1.0, 24)
        loss_fn = BalancedMSEBMC(sigma=0.5, prior="histogram", num_bins=16).fit(target)
        y_pred = (target + 0.1 * torch.randn(24)).requires_grad_(True)
        loss = loss_fn(y_pred, target)
        loss.backward()
        assert y_pred.grad is not None
        assert torch.isfinite(y_pred.grad).all()
        assert float(y_pred.grad.abs().sum()) > 0.0

    def test_gradient_flows_through_two_dimensional_input(self) -> None:
        target = torch.linspace(-1.0, 1.0, 16).unsqueeze(-1)
        loss_fn = BalancedMSEBMC(sigma=0.5, prior="histogram", num_bins=8).fit(target)
        y_pred = (target + 0.05).requires_grad_(True)
        loss_fn(y_pred, target).backward()
        assert y_pred.grad is not None
        assert torch.isfinite(y_pred.grad).all()


class TestReductionMaskWeights:
    @pytest.fixture()
    def fitted(self) -> BalancedMSEBMC:
        targets = torch.linspace(-2.0, 2.0, 80)
        return BalancedMSEBMC(sigma=0.8, prior="histogram", num_bins=16).fit(targets)

    def test_reductions(self, fitted: BalancedMSEBMC) -> None:
        target = torch.linspace(-1.5, 1.5, 10)
        y_pred = target + 0.2 * torch.sin(torch.arange(10, dtype=torch.float32))
        none_fn = BalancedMSEBMC(sigma=0.8, prior="histogram", num_bins=16, reduction="none").fit(
            torch.linspace(-2.0, 2.0, 80)
        )
        sum_fn = BalancedMSEBMC(sigma=0.8, prior="histogram", num_bins=16, reduction="sum").fit(
            torch.linspace(-2.0, 2.0, 80)
        )
        per_sample = none_fn(y_pred, target)
        assert per_sample.shape == target.shape
        assert torch.isclose(fitted(y_pred, target), per_sample.mean())
        assert torch.isclose(sum_fn(y_pred, target), per_sample.sum())

    def test_weights_follow_shared_contract(self, fitted: BalancedMSEBMC) -> None:
        target = torch.linspace(-1.5, 1.5, 10)
        y_pred = target + 0.1
        weights = torch.linspace(0.1, 1.0, 10)
        none_fn = BalancedMSEBMC(sigma=0.8, prior="histogram", num_bins=16, reduction="none").fit(
            torch.linspace(-2.0, 2.0, 80)
        )
        per_sample = none_fn(y_pred, target)
        expected = (per_sample * weights).sum() / weights.sum()
        assert torch.isclose(fitted(y_pred, target, weights=weights), expected)

    def test_mask_removes_targets_from_batch_support(self, fitted: BalancedMSEBMC) -> None:
        target = torch.linspace(-1.5, 1.5, 10)
        y_pred = target + 0.15
        mask = torch.tensor([True, False, True, True, True, True, True, True, True, True])
        none_fn = BalancedMSEBMC(sigma=0.8, prior="histogram", num_bins=16, reduction="none").fit(
            torch.linspace(-2.0, 2.0, 80)
        )
        masked = none_fn(y_pred, target, mask=mask)
        # Masked rows are exactly zero and the remaining rows see the batch
        # without the missing target (same fitted prior object).
        assert masked.shape == target.shape
        assert torch.equal(masked[~mask], torch.zeros(1))
        subset = none_fn(y_pred[mask], target[mask])
        assert torch.allclose(masked[mask], subset)
        assert torch.isclose(fitted(y_pred, target, mask=mask), masked[mask].mean())

    def test_weights_with_mask(self, fitted: BalancedMSEBMC) -> None:
        target = torch.linspace(-1.5, 1.5, 10)
        y_pred = target + 0.15
        mask = torch.tensor([True, True, False, True, True, True, True, True, True, True])
        weights = torch.linspace(0.2, 1.0, 10)
        none_fn = BalancedMSEBMC(sigma=0.8, prior="histogram", num_bins=16, reduction="none").fit(
            torch.linspace(-2.0, 2.0, 80)
        )
        masked = none_fn(y_pred, target, mask=mask)
        expected = (masked[mask] * weights[mask]).sum() / weights[mask].sum()
        assert torch.isclose(fitted(y_pred, target, mask=mask, weights=weights), expected)

    def test_all_masked_returns_zero(self, fitted: BalancedMSEBMC) -> None:
        target = torch.linspace(-1.0, 1.0, 6)
        y_pred = target + 0.1
        mask = torch.zeros(6, dtype=torch.bool)
        result = fitted(y_pred, target, mask=mask)
        assert result.ndim == 0
        assert float(result) == 0.0
        assert torch.isfinite(result)

    def test_column_input_shapes(self) -> None:
        target = torch.linspace(-1.0, 1.0, 12).unsqueeze(-1)
        loss_fn = BalancedMSEBMC(sigma=0.6, prior="histogram", num_bins=8, reduction="none").fit(
            target
        )
        y_pred = target + 0.1
        mask = torch.ones_like(target, dtype=torch.bool)
        assert loss_fn(y_pred, target).shape == target.shape
        assert loss_fn(y_pred, target, mask=mask).shape == target.shape
        mask[3] = False
        masked = loss_fn(y_pred, target, mask=mask)
        assert masked[3].item() == 0.0
        assert torch.isfinite(masked).all()

    def test_functional_helper_matches_class_with_mask_and_weights(
        self, fitted: BalancedMSEBMC
    ) -> None:
        target = torch.linspace(-1.5, 1.5, 10)
        y_pred = target + 0.15
        mask = torch.tensor([True, True, False, True, True, True, True, True, True, True])
        weights = torch.linspace(0.2, 1.0, 10)
        log_prior = fitted.log_prior(target)
        expected = fitted(y_pred, target, mask=mask, weights=weights)
        got = balanced_mse_bmc_loss(
            y_pred,
            target,
            log_prior,
            0.8,
            mask=mask,
            weights=weights,
            reduction="mean",
        )
        assert torch.isclose(got, expected)

    def test_nan_target_requires_mask(self, fitted: BalancedMSEBMC) -> None:
        target = torch.linspace(-1.0, 1.0, 6)
        target[2] = float("nan")
        y_pred = torch.linspace(-1.0, 1.0, 6)
        mask = torch.ones(6, dtype=torch.bool)
        mask[2] = False
        assert torch.isfinite(fitted(y_pred, target, mask=mask))
        with pytest.raises(ValueError, match="non-finite"):
            fitted(y_pred, target)


class TestPriorAndSigmaFitting:
    def test_histogram_prior_prefers_dense_region(self) -> None:
        generator = torch.Generator().manual_seed(0)
        dense = -1.0 + 0.1 * torch.randn(600, generator=generator)
        rare = 3.0 + 0.1 * torch.randn(20, generator=generator)
        target = torch.cat([dense, rare])
        loss_fn = BalancedMSEBMC(sigma=0.5, prior="histogram", num_bins=64).fit(target)
        log_prior = loss_fn.log_prior(torch.tensor([-1.0, 3.0]))
        assert float(log_prior[0] - log_prior[1]) > 1.0
        assert torch.isfinite(log_prior).all()
        # Out-of-range queries fall back to the edge density, not -inf.
        assert torch.isfinite(loss_fn.log_prior(torch.tensor([100.0]))).all()

    def test_kde_prior_prefers_dense_region(self) -> None:
        generator = torch.Generator().manual_seed(1)
        dense = -1.0 + 0.1 * torch.randn(500, generator=generator)
        rare = 3.0 + 0.1 * torch.randn(20, generator=generator)
        target = torch.cat([dense, rare])
        loss_fn = BalancedMSEBMC(sigma=0.5, prior="kde", kde_points=256, prior_floor=1e-8).fit(
            target
        )
        log_prior = loss_fn.log_prior(torch.tensor([-1.0, 3.0]))
        assert float(log_prior[0] - log_prior[1]) > 1.0
        assert torch.isfinite(loss_fn.log_prior(torch.tensor([100.0]))).all()

    def test_uniform_prior_is_constant(self) -> None:
        target = torch.linspace(0.0, 5.0, 50)
        loss_fn = BalancedMSEBMC(sigma=1.0, prior="uniform").fit(target)
        assert torch.equal(loss_fn.log_prior(target), torch.zeros_like(target))

    def test_supplied_sigma_is_preserved(self) -> None:
        target = torch.linspace(-1.0, 1.0, 20)
        loss_fn = BalancedMSEBMC(sigma=0.75, prior="uniform").fit(target)
        assert torch.isclose(loss_fn.sigma, torch.tensor(0.75))

    def test_residual_mad_sigma_estimate(self) -> None:
        generator = torch.Generator().manual_seed(2)
        target = 4.0 + 2.0 * torch.randn(500, generator=generator)
        predictions = target - 0.3 * torch.randn(500, generator=generator)
        loss_fn = BalancedMSEBMC(prior="uniform").fit(target, predictions)
        residuals = target - predictions
        expected = 1.4826 * (residuals - residuals.median()).abs().median()
        assert torch.isclose(loss_fn.sigma, expected, rtol=1e-6)

    def test_target_mad_sigma_proxy(self) -> None:
        generator = torch.Generator().manual_seed(3)
        target = 2.0 + 1.5 * torch.randn(300, generator=generator)
        loss_fn = BalancedMSEBMC(prior="uniform").fit(target)
        expected = 1.4826 * (target - target.median()).abs().median()
        assert torch.isclose(loss_fn.sigma, expected, rtol=1e-6)

    def test_constant_targets_get_finite_sigma(self) -> None:
        loss_fn = BalancedMSEBMC(prior="uniform").fit(torch.full((10,), 3.0))
        assert torch.isfinite(loss_fn.sigma)
        assert float(loss_fn.sigma) > 0.0

    def test_nonfinite_fit_targets_are_ignored(self) -> None:
        target = torch.tensor([0.0, 1.0, float("nan"), 2.0])
        loss_fn = BalancedMSEBMC(prior="uniform").fit(target)
        assert torch.isfinite(loss_fn.sigma)
        assert torch.isfinite(loss_fn.log_prior(torch.tensor([1.0]))).all()


class TestValidation:
    def test_uniform_prior_with_supplied_sigma_needs_no_fit(self) -> None:
        loss_fn = BalancedMSEBMC(sigma=1.0, prior="uniform")
        loss = loss_fn(torch.tensor([0.0, 0.2]), torch.tensor([0.1, 0.3]))
        assert torch.isfinite(loss)
        assert torch.equal(loss_fn.log_prior(torch.tensor([0.1, 0.3])), torch.zeros(2))

    def test_histogram_without_fit_warns_and_uses_uniform_fallback(self) -> None:
        unfitted = BalancedMSEBMC(sigma=0.7)
        fallback = BalancedMSEBMC(sigma=0.7, prior="uniform")
        y_pred = torch.tensor([0.0, 0.2, -0.1])
        target = torch.tensor([0.1, 0.3, -0.2])
        with pytest.warns(UserWarning, match="fit"):
            observed = unfitted(y_pred, target)
        assert torch.isclose(observed, fallback(y_pred, target))

    def test_log_prior_requires_fit_for_fitted_prior(self) -> None:
        loss_fn = BalancedMSEBMC(sigma=1.0, prior="histogram")
        with pytest.raises(RuntimeError, match="fit"):
            _ = loss_fn.log_prior(torch.tensor([0.0]))

    def test_rejects_multi_output(self) -> None:
        loss_fn = BalancedMSEBMC(sigma=1.0, prior="uniform").fit(torch.zeros(8))
        with pytest.raises(ValueError, match="single-output"):
            loss_fn(torch.zeros(4, 2), torch.zeros(4, 2))

    def test_rejects_scalar_tensor(self) -> None:
        loss_fn = BalancedMSEBMC(sigma=1.0, prior="uniform").fit(torch.zeros(8))
        with pytest.raises(ValueError, match="batch dimension"):
            loss_fn(torch.tensor(0.0), torch.tensor(0.0))

    def test_rejects_shape_mismatch(self) -> None:
        loss_fn = BalancedMSEBMC(sigma=1.0, prior="uniform").fit(torch.zeros(8))
        with pytest.raises(ValueError, match="match"):
            loss_fn(torch.zeros(4), torch.zeros(5))

    @pytest.mark.parametrize("sigma", [0.0, -1.0, float("inf")])
    def test_rejects_invalid_sigma(self, sigma: float) -> None:
        with pytest.raises(ValueError, match="sigma"):
            BalancedMSEBMC(sigma=sigma)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"prior": "bad"}, "prior"),
            ({"num_bins": 1}, "num_bins"),
            ({"prior_floor": 0.0}, "prior_floor"),
            ({"prior_floor": 1.5}, "prior_floor"),
            ({"min_sigma": 0.0}, "min_sigma"),
            ({"prior_bandwidth": -0.1}, "prior_bandwidth"),
            ({"kde_points": 0}, "kde_points"),
        ],
    )
    def test_rejects_invalid_options(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            BalancedMSEBMC(**kwargs)

    def test_fit_rejects_empty_and_all_nonfinite(self) -> None:
        with pytest.raises(ValueError, match="at least one finite"):
            BalancedMSEBMC().fit(torch.empty(0))
        with pytest.raises(ValueError, match="at least one finite"):
            BalancedMSEBMC().fit(torch.full((5,), float("nan")))

    def test_fit_rejects_mismatched_predictions(self) -> None:
        with pytest.raises(ValueError, match="same number"):
            BalancedMSEBMC().fit(torch.zeros(5), torch.zeros(4))

    def test_functional_helper_validation(self) -> None:
        y_pred = torch.zeros(4)
        target = torch.zeros(4)
        prior = torch.zeros(4)
        with pytest.raises(ValueError, match="sigma"):
            balanced_mse_bmc_loss(y_pred, target, prior, 0.0)
        with pytest.raises(ValueError, match="log_prior"):
            balanced_mse_bmc_loss(y_pred, target, torch.zeros(3), 1.0)
        with pytest.raises(ValueError, match="NaN"):
            bad_prior = prior.clone()
            bad_prior[0] = float("nan")
            balanced_mse_bmc_loss(y_pred, target, bad_prior, 1.0)

    def test_registry_entry(self) -> None:
        loss_fn = create_loss_from_config(
            {"type": "balanced_mse_bmc", "sigma": 0.5, "prior": "uniform"}
        )
        assert isinstance(loss_fn, BalancedMSEBMC)


def _make_paper_synthetic(
    n: int = 512, seed: int = 0, noise_sigma: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small deterministic version of the paper's 1-D synthetic benchmark.

    Training labels are a truncated normal concentrated around 5, then noisy
    values are generated as ``x = y - noise`` so the conditional mean is
    ``y = x``.  The evaluation grid is uniform on ``[0, 10]``, i.e. the
    balanced test distribution of the paper.
    """
    generator = torch.Generator().manual_seed(seed)
    target = 5.0 + 0.5 * torch.randn(n, generator=generator)
    target = target[(target >= 0.0) & (target <= 10.0)]
    while target.numel() < n:
        extra = 5.0 + 0.5 * torch.randn(n, generator=generator)
        target = torch.cat([target, extra[(extra >= 0.0) & (extra <= 10.0)]])
    target = target[:n]
    x = target - noise_sigma * torch.randn(n, generator=generator)
    return x.unsqueeze(-1), target.unsqueeze(-1)


def _train_linear(
    loss_fn: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int = 35,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
) -> torch.Tensor:
    """Train a linear model with full reproducibility (zero init, local RNG)."""
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    generator = torch.Generator().manual_seed(seed + 1)
    y_flat = y.flatten()
    for _ in range(epochs):
        permutation = torch.randperm(x.shape[0], generator=generator)
        for start in range(0, x.shape[0], batch_size):
            idx = permutation[start : start + batch_size]
            optimizer.zero_grad()
            prediction = model(x[idx]).flatten()
            loss = loss_fn(prediction, y_flat[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _regional_errors(model: torch.nn.Module) -> dict[str, float]:
    grid = torch.linspace(0.0, 10.0, 512)
    with torch.no_grad():
        prediction = model(grid.unsqueeze(-1)).flatten()
    error = (prediction - grid) ** 2
    tail = (grid < 1.0) | (grid > 9.0)
    return {
        "all": float(error.mean()),
        "tail": float(error[tail].mean()),
        "worst": float(error.max()),
    }


class TestSyntheticImbalancedRegression:
    def test_bmc_improves_tail_and_worst_region_over_mse(self) -> None:
        x, y = _make_paper_synthetic(n=512, seed=0, noise_sigma=1.0)
        mse_model = _train_linear(torch.nn.MSELoss(), x, y, seed=0)
        bmc_loss = BalancedMSEBMC(sigma=1.0, prior="histogram", num_bins=64).fit(y.flatten())
        bmc_model = _train_linear(bmc_loss, x, y, seed=0)
        mse = _regional_errors(mse_model)
        bmc = _regional_errors(bmc_model)
        assert bmc["all"] < 0.5 * mse["all"], (mse, bmc)
        assert bmc["tail"] < 0.5 * mse["tail"], (mse, bmc)
        assert bmc["worst"] < 0.75 * mse["worst"], (mse, bmc)
        assert math.isfinite(bmc["tail"])
