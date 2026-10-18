"""Tests for forward-generative regression with test-time MLE inversion."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from torchregress.algorithms.forward_inversion import (
    ForwardInversionRegressor,
    build_decoder_mlp,
)

DTYPE = torch.float64


def _linear_decoder(n_labels: int, n_observables: int, seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(n_labels, n_observables).to(DTYPE)


def _linear_problem(
    seed: int = 0,
    n: int = 400,
    n_labels: int = 2,
    n_observables: int = 6,
    sigma: float = 0.05,
) -> tuple[nn.Linear, torch.Tensor, torch.Tensor]:
    """y ~ U(-2, 2), x = A y + b + N(0, sigma^2)."""
    decoder = _linear_decoder(n_labels, n_observables, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    y = torch.rand(n, n_labels, generator=generator, dtype=DTYPE) * 4 - 2
    with torch.no_grad():
        X = decoder(y) + sigma * torch.randn(n, n_observables, generator=generator, dtype=DTYPE)
    return decoder, y, X


def _frozen_regressor(
    decoder: nn.Module,
    sigma: float | torch.Tensor | None,
    *,
    learn_noise: bool = False,
) -> ForwardInversionRegressor:
    return ForwardInversionRegressor(
        decoder=copy.deepcopy(decoder),
        sigma_x=sigma,
        learn_noise=learn_noise,
        epochs=1,
    )


def _analytic_linear_mle(
    decoder: nn.Linear,
    X: torch.Tensor,
    sigma: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Gaussian MLE for the linear forward model with diagonal noise."""
    weight = decoder.weight.detach().to(DTYPE)  # (D_x, D_y)
    bias = decoder.bias.detach().to(DTYPE)  # (D_x,)
    residual = X - bias.reshape(1, -1)
    if isinstance(sigma, torch.Tensor):
        weights = 1.0 / sigma.to(DTYPE).reshape(X.shape) ** 2
        lhs = torch.einsum("nd,dx,dy->nxy", weights, weight, weight)
        rhs = torch.einsum("nd,dx,nd->nx", weights, weight, residual)
        return torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1)
    normal = weight.T @ weight
    rhs = residual @ weight
    return torch.linalg.solve(normal, rhs.T).T


class _QuadraticDecoder(nn.Module):
    """g(y) = [y1, y2, y1^2, y2^2, y1*y2]."""

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y1 = y[:, :1]
        y2 = y[:, 1:2]
        return torch.cat([y1, y2, y1 * y1, y2 * y2, y1 * y2], dim=-1)


# --------------------------------------------------------------------------- #
# Exact linear-decoder inversion
# --------------------------------------------------------------------------- #
def test_linear_decoder_exact_inversion_matches_analytic_mle() -> None:
    decoder, y, X = _linear_problem(sigma=0.05)
    reg = _frozen_regressor(decoder, sigma=0.05).fit(X, y, train_decoder=False)
    y_hat = reg.predict(X, max_iter=10, tol=1e-12)
    expected = _analytic_linear_mle(decoder, X)
    assert torch.allclose(y_hat, expected, atol=1e-6, rtol=0.0)
    # Noise is amplified by the (random) inverse design; compare with y loosely.
    assert torch.allclose(y_hat, y, atol=0.6)


def test_linear_inversion_batch_equals_single_sample() -> None:
    decoder, y, X = _linear_problem(n=16, sigma=0.1)
    reg = _frozen_regressor(decoder, sigma=0.1).fit(X, y, train_decoder=False)
    batched = reg.predict(X, max_iter=10, tol=1e-12)
    single = torch.cat(
        [reg.predict(X[i : i + 1], max_iter=10, tol=1e-12) for i in range(X.shape[0])]
    )
    assert torch.allclose(batched, single, atol=1e-9)


def test_lbfgs_matches_gauss_newton_on_linear_problem() -> None:
    decoder, y, X = _linear_problem(n=32, sigma=0.1)
    reg = _frozen_regressor(decoder, sigma=0.1).fit(X, y, train_decoder=False)
    gn = reg.predict(X, method="gauss_newton", max_iter=10, tol=1e-12)
    lbfgs = reg.predict(X, method="lbfgs", max_iter=60, tol=1e-12)
    assert torch.allclose(gn, lbfgs, atol=1e-4)


# --------------------------------------------------------------------------- #
# Attenuation bias
# --------------------------------------------------------------------------- #
def test_forward_inversion_removes_attenuation_at_target_extremes() -> None:
    """Naive inverse regression shrinks extreme labels; forward inversion does not."""
    torch.manual_seed(7)
    n = 20_000
    y = torch.rand(n, 1, dtype=DTYPE) * 6 - 3  # U(-3, 3)
    sigma = 2.0
    noise = sigma * torch.randn(n, 1, dtype=DTYPE)
    X_obs = 2.0 * y + noise  # true slope 2 (inverse slope 1/2)

    # Naive discriminative regression y ~ x on the noisy features.
    x_mean = X_obs.mean()
    y_mean = y.mean()
    slope = ((X_obs - x_mean) * (y - y_mean)).mean() / (X_obs - x_mean).square().mean()
    intercept = y_mean - slope * x_mean
    y_direct = intercept + slope * X_obs
    assert float(slope) < 0.5 * 2.0  # severe attenuation

    # Forward inversion with the known generative model x = 2 y + eps.
    class _SlopeDecoder(nn.Module):
        def forward(self, labels: torch.Tensor) -> torch.Tensor:
            return 2.0 * labels

    reg = ForwardInversionRegressor(
        decoder=_SlopeDecoder().double(), sigma_x=sigma, learn_noise=False, epochs=1
    ).fit(X_obs, y, train_decoder=False)
    y_forward = reg.predict(X_obs, max_iter=10, tol=1e-12)

    tail = y > 2.0  # one-sided tail: the bias sign does not cancel across signs
    direct_bias = float((y_direct[tail] - y[tail]).mean())
    forward_bias = float((y_forward[tail] - y[tail]).mean())
    assert direct_bias < -0.5  # compressed toward zero
    assert abs(forward_bias) < 0.05  # unbiased apart from the irreducible noise
    assert abs(forward_bias) < abs(direct_bias) / 10.0


# --------------------------------------------------------------------------- #
# Covariance
# --------------------------------------------------------------------------- #
def test_covariance_matches_fisher_information() -> None:
    decoder, y, X = _linear_problem(n=64, sigma=0.2)
    reg = _frozen_regressor(decoder, sigma=0.2).fit(X, y, train_decoder=False)
    mean, covariance = reg.predict_with_covariance(X, max_iter=10, tol=1e-12)

    weight = decoder.weight.detach().to(DTYPE)
    analytic = torch.linalg.inv(weight.T @ weight) * (0.2**2)
    expected = analytic.unsqueeze(0).expand(X.shape[0], -1, -1)
    assert covariance.shape == (X.shape[0], 2, 2)
    assert torch.allclose(covariance, expected, atol=1e-6)
    assert torch.isfinite(mean).all()


def test_covariance_matches_monte_carlo() -> None:
    decoder = _linear_decoder(n_labels=2, n_observables=4, seed=3)
    sigma = 0.3
    y_true = torch.tensor([[1.0, -1.5]], dtype=DTYPE)
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        clean = decoder(y_true)
        samples = clean + sigma * torch.randn(4000, 4, generator=generator, dtype=DTYPE)
    reg = _frozen_regressor(decoder, sigma=sigma).fit(samples, y_true.expand(4000, -1))
    mean, covariance = reg.predict_with_covariance(samples, max_iter=10, tol=1e-12)
    assert covariance.shape == (4000, 2, 2)
    _, analytic = reg.predict_with_covariance(samples[:1], max_iter=10, tol=1e-12)
    empirical = torch.cov(mean.T)
    assert torch.allclose(empirical, analytic[0], atol=0.25 * sigma**2)
    assert torch.allclose(
        torch.diagonal(empirical), torch.diagonal(analytic[0]), rtol=0.15, atol=1e-3
    )


def test_predict_return_std_matches_covariance_diagonal() -> None:
    decoder, y, X = _linear_problem(n=24, sigma=0.1)
    reg = _frozen_regressor(decoder, sigma=0.1).fit(X, y, train_decoder=False)
    mean, std = reg.predict(X, max_iter=10, tol=1e-12, return_std=True)
    _, covariance = reg.predict_with_covariance(X, max_iter=10, tol=1e-12)
    assert torch.allclose(std, torch.diagonal(covariance, dim1=1, dim2=2).sqrt(), atol=1e-6)
    assert mean.shape == std.shape == (X.shape[0], 2)


# --------------------------------------------------------------------------- #
# Nonlinear decoder / inversion robustness
# --------------------------------------------------------------------------- #
def _quadratic_data(n: int = 64, sigma: float = 0.0, seed: int = 5):
    torch.manual_seed(seed)
    y = torch.rand(n, 2, dtype=DTYPE) * 3 - 1.5
    decoder = _QuadraticDecoder().to(DTYPE)
    with torch.no_grad():
        X = decoder(y) + sigma * torch.randn(n, 5, dtype=DTYPE)
    return decoder, y, X


@pytest.mark.parametrize("method", ["gauss_newton", "lbfgs"])
def test_nonlinear_decoder_recovers_labels(method: str) -> None:
    decoder, y, X = _quadratic_data(sigma=0.0)
    reg = _frozen_regressor(decoder, sigma=0.0, learn_noise=False).fit(X, y, train_decoder=False)
    y_hat = reg.predict(X, method=method, max_iter=80, tol=1e-12)
    assert torch.allclose(y_hat, y, atol=1e-4)


def test_jacobian_loop_matches_vmap() -> None:
    decoder, y, X = _quadratic_data(sigma=0.0)
    reg = _frozen_regressor(decoder, sigma=0.0, learn_noise=False).fit(X, y, train_decoder=False)
    vmap_pred = reg.predict(X, max_iter=40, tol=1e-12, jacobian_method="vmap")
    loop_pred = reg.predict(X, max_iter=40, tol=1e-12, jacobian_method="loop")
    assert torch.allclose(vmap_pred, loop_pred, atol=1e-8)


def test_masked_observables_ignore_corrupted_pixels() -> None:
    decoder = _linear_decoder(n_labels=1, n_observables=4, seed=21)
    torch.manual_seed(3)
    y = torch.rand(8, 1, dtype=DTYPE) * 2 - 1
    with torch.no_grad():
        X = decoder(y)
    corrupted = X.clone()
    corrupted[:, 0] += 50.0

    reg = _frozen_regressor(decoder, sigma=0.01).fit(X, y, train_decoder=False)
    mask = torch.ones_like(corrupted, dtype=torch.bool)
    mask[:, 0] = False

    y_masked = reg.predict(corrupted, mask=mask, max_iter=20, tol=1e-12)
    y_unmasked = reg.predict(corrupted, max_iter=20, tol=1e-12)
    assert torch.allclose(y_masked, y, atol=1e-6)
    assert not torch.allclose(y_unmasked, y, atol=0.1)


# --------------------------------------------------------------------------- #
# Per-sample measurement errors
# --------------------------------------------------------------------------- #
def test_per_sample_sigma_weights_observables_correctly() -> None:
    decoder = _linear_decoder(n_labels=1, n_observables=2, seed=9)
    torch.manual_seed(4)
    n = 20
    y = torch.rand(n, 1, dtype=DTYPE) * 2 - 1
    X = decoder(y)
    sigma = torch.full((n, 2), 0.05, dtype=DTYPE)
    sigma[:, 1] = 5.0  # second observable is nearly uninformative

    reg = _frozen_regressor(decoder, sigma=0.05).fit(X, y, train_decoder=False)
    y_hat = reg.predict(X, sigma_x=sigma, max_iter=20, tol=1e-12)

    weight = decoder.weight.detach().to(DTYPE)
    analytic = _analytic_linear_mle(decoder, X, sigma)
    assert torch.allclose(y_hat, analytic, atol=1e-6)
    # A large perturbation of the uninformative pixel must barely move the estimate.
    perturbed = X.clone()
    perturbed[:, 1] += 10.0
    y_perturbed = reg.predict(perturbed, sigma_x=sigma, max_iter=20, tol=1e-12)
    assert float((y_perturbed - y_hat).abs().max()) < 0.02
    assert weight.shape == (2, 1)


# --------------------------------------------------------------------------- #
# Training loops, noise learning, early stopping, determinism
# --------------------------------------------------------------------------- #
def _mlp_problem(seed: int = 0, n: int = 200):
    torch.manual_seed(seed)
    y = torch.rand(n, 1) * 2 - 1
    X = torch.cat([y, torch.sin(2 * y), y**2], dim=1)
    return y, X


def test_default_mlp_training_reduces_loss_and_predicts() -> None:
    y, X = _mlp_problem()
    reg = ForwardInversionRegressor(
        n_labels=1,
        n_observables=3,
        hidden_dims=(16,),
        sigma_x=0.01,
        epochs=60,
        batch_size=64,
        lr=5e-3,
        seed=0,
    )
    reg.fit(X, y)
    assert len(reg.loss_history_) == 60
    assert reg.loss_history_[-1] < reg.loss_history_[0]
    assert reg.decoder_trained_ is True
    y_hat = reg.predict(X, max_iter=20)
    assert y_hat.shape == (200, 1)
    assert float(((y_hat - y) ** 2).mean()) < 0.05


def test_training_is_deterministic_given_seed() -> None:
    y, X = _mlp_problem(seed=1)
    kwargs = dict(
        n_labels=1,
        n_observables=3,
        hidden_dims=(16,),
        sigma_x=0.01,
        epochs=25,
        batch_size=32,
        lr=5e-3,
        seed=123,
    )
    pred_a = ForwardInversionRegressor(**kwargs).fit(X, y).predict(X)
    pred_b = ForwardInversionRegressor(**kwargs).fit(X, y).predict(X)
    assert torch.allclose(pred_a, pred_b)


def test_learned_noise_variance_when_sigma_unknown() -> None:
    y, X = _mlp_problem(seed=2)
    reg = ForwardInversionRegressor(
        n_labels=1,
        n_observables=3,
        hidden_dims=(16,),
        sigma_x=None,
        learn_noise=True,
        epochs=30,
        batch_size=64,
        lr=5e-3,
        seed=0,
    ).fit(X, y)
    assert reg.noise_variance_ is not None
    assert reg.noise_variance_.shape == (3,)
    assert bool((reg.noise_variance_ > 0).all())
    assert torch.isfinite(reg.predict(X)).all()


def test_early_stopping_uses_validation_split() -> None:
    y, X = _mlp_problem(seed=3)
    reg = ForwardInversionRegressor(
        n_labels=1,
        n_observables=3,
        hidden_dims=(16,),
        sigma_x=0.01,
        epochs=40,
        batch_size=64,
        lr=5e-3,
        seed=0,
        validation_fraction=0.2,
        patience=3,
    )
    reg.fit(X, y)
    assert len(reg.validation_history_) == len(reg.loss_history_)
    assert len(reg.loss_history_) <= 40


def test_frozen_decoder_is_not_modified() -> None:
    decoder, y, X = _linear_problem(n=64, sigma=0.1)
    before = {k: v.clone() for k, v in decoder.state_dict().items()}
    reg = _frozen_regressor(decoder, sigma=0.1)
    reg.fit(X, y, train_decoder=False)
    assert reg.decoder_trained_ is False
    for key, value in reg.decoder_.state_dict().items():
        assert torch.equal(value, before[key])


# --------------------------------------------------------------------------- #
# Input validation / error paths
# --------------------------------------------------------------------------- #
def test_predict_before_fit_raises() -> None:
    reg = ForwardInversionRegressor(n_labels=1, n_observables=2)
    with pytest.raises(RuntimeError, match="fitted"):
        reg.predict(torch.zeros(1, 2))


def test_fit_input_validation() -> None:
    reg = ForwardInversionRegressor(n_labels=1, n_observables=2, epochs=1)
    with pytest.raises(ValueError, match="2-D"):
        reg.fit(torch.zeros(4, 2, 2), torch.zeros(4, 1))
    with pytest.raises(ValueError, match="same number of rows"):
        reg.fit(torch.zeros(4, 2), torch.zeros(3, 1))
    with pytest.raises(ValueError, match="n_observables"):
        reg.fit(torch.zeros(4, 3), torch.zeros(4, 1))


def test_unknown_sigma_without_learn_noise_raises() -> None:
    y, X = _mlp_problem()
    reg = ForwardInversionRegressor(
        n_labels=1, n_observables=3, sigma_x=None, learn_noise=False, epochs=1
    )
    with pytest.raises(ValueError, match="sigma_x"):
        reg.fit(X, y)


def test_invalid_method_and_jacobian_and_init() -> None:
    decoder, y, X = _linear_problem(n=8, sigma=0.1)
    reg = _frozen_regressor(decoder, sigma=0.1).fit(X, y, train_decoder=False)
    with pytest.raises(ValueError, match="method"):
        reg.predict(X, method="newton")
    with pytest.raises(ValueError, match="jacobian_method"):
        reg.predict(X, jacobian_method="finite_difference")
    with pytest.raises(ValueError, match="init"):
        reg.predict(X, init=torch.zeros(3))
    with pytest.raises(ValueError, match="mask"):
        reg.predict(X, mask=torch.ones(8, 1, dtype=torch.bool))


def test_negative_sigma_rejected() -> None:
    y, X = _mlp_problem()
    reg = ForwardInversionRegressor(
        n_labels=1, n_observables=3, sigma_x=-1.0, learn_noise=True, epochs=1
    )
    with pytest.raises(ValueError, match="non-negative"):
        reg.fit(X, y)


def test_build_decoder_mlp_contract() -> None:
    decoder = build_decoder_mlp(2, 5, hidden_dims=(8, 4))
    out = decoder(torch.zeros(7, 2))
    assert out.shape == (7, 5)
    with pytest.raises(ValueError, match="activation"):
        build_decoder_mlp(2, 5, activation="gelu")
    with pytest.raises(ValueError, match="positive"):
        build_decoder_mlp(0, 5)


# --------------------------------------------------------------------------- #
# Training likelihood options (full NLL, robust Student-t, WLS)
# --------------------------------------------------------------------------- #
def test_default_likelihood_is_gaussian_and_elementwise_nll_formulas() -> None:
    reg = ForwardInversionRegressor(n_labels=1, n_observables=3, sigma_x=0.5)
    assert reg.likelihood == "gaussian"
    assert reg.degrees_of_freedom == 5.0

    residual = torch.tensor([0.1, -2.0, 0.5], dtype=DTYPE)
    total_var = torch.tensor([0.25, 1.0, 0.04], dtype=DTYPE)

    gaussian = reg._elementwise_nll(residual, total_var)  # noqa: SLF001
    expected_gaussian = 0.5 * (residual**2 / total_var + torch.log(total_var))
    assert torch.allclose(gaussian, expected_gaussian)

    wls = ForwardInversionRegressor(
        n_labels=1, n_observables=3, sigma_x=0.5, likelihood="wls", learn_noise=False
    )
    expected_wls = 0.5 * residual**2 / total_var
    assert torch.allclose(wls._elementwise_nll(residual, total_var), expected_wls)  # noqa: SLF001

    student = ForwardInversionRegressor(
        n_labels=1,
        n_observables=3,
        sigma_x=0.5,
        likelihood="student_t",
        degrees_of_freedom=4.0,
    )
    expected_t = 0.5 * torch.log(total_var) + 0.5 * 5.0 * torch.log1p(
        residual**2 / (4.0 * total_var)
    )
    assert torch.allclose(student._elementwise_nll(residual, total_var), expected_t)  # noqa: SLF001


def test_likelihood_validation() -> None:
    with pytest.raises(ValueError, match="likelihood"):
        ForwardInversionRegressor(n_labels=1, n_observables=2, likelihood="poisson")
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        ForwardInversionRegressor(
            n_labels=1, n_observables=2, likelihood="student_t", degrees_of_freedom=2.0
        )
    with pytest.raises(ValueError, match="learn_noise"):
        ForwardInversionRegressor(
            n_labels=1, n_observables=2, sigma_x=0.1, likelihood="wls", learn_noise=True
        )
    # WLS needs known sigma_x at fit time.
    wls = ForwardInversionRegressor(
        n_labels=1, n_observables=2, likelihood="wls", learn_noise=False, epochs=1
    )
    with pytest.raises(ValueError, match="known sigma_x"):
        wls.fit(torch.zeros(4, 2), torch.zeros(4, 1))


@pytest.mark.parametrize("likelihood", ["gaussian", "student_t", "wls"])
def test_training_with_each_likelihood(likelihood: str) -> None:
    y, X = _mlp_problem(seed=5)
    learn_noise = likelihood != "wls"
    reg = ForwardInversionRegressor(
        n_labels=1,
        n_observables=3,
        hidden_dims=(16,),
        sigma_x=0.01,
        learn_noise=learn_noise,
        likelihood=likelihood,
        epochs=20,
        batch_size=64,
        lr=5e-3,
        seed=0,
    )
    reg.fit(X, y)
    assert len(reg.loss_history_) == 20
    assert torch.isfinite(torch.tensor(reg.loss_history_)).all()
    assert torch.isfinite(reg.predict(X)).all()
    if likelihood == "wls":
        assert torch.equal(reg.noise_variance_, torch.zeros(3, dtype=reg.noise_variance_.dtype))


def test_student_t_is_robust_to_defective_pixels() -> None:
    """Heavy-tailed likelihood should ignore injected bad pixels, Gaussian should not."""
    torch.manual_seed(0)
    n = 800
    y = torch.rand(n, 1, dtype=DTYPE) * 2 - 1
    X = 2.0 * y + 0.05 * torch.randn(n, 1, dtype=DTYPE)
    X[torch.arange(0, n, 10)] += 20.0  # 10% catastrophic pixel defects

    def fit(likelihood: str) -> float:
        torch.manual_seed(0)
        reg = ForwardInversionRegressor(
            decoder=nn.Linear(1, 1).to(DTYPE),
            sigma_x=0.05,
            learn_noise=likelihood != "wls",
            likelihood=likelihood,
            degrees_of_freedom=3.0,
            epochs=150,
            batch_size=128,
            lr=1e-2,
            seed=0,
        )
        reg.fit(X, y)
        return float(reg.decoder_.weight.item())

    gaussian_slope = fit("gaussian")
    student_slope = fit("student_t")
    assert abs(gaussian_slope - 2.0) > 0.3  # dragged by the defects
    assert abs(student_slope - 2.0) < 0.2  # robust


# --------------------------------------------------------------------------- #
# Test-time optimisation diagnostics (non-linear / non-convex awareness)
# --------------------------------------------------------------------------- #
def test_predict_return_info_reports_stationarity_and_conditioning() -> None:
    decoder, y, X = _linear_problem(n=32, sigma=0.05)
    reg = _frozen_regressor(decoder, sigma=0.05).fit(X, y, train_decoder=False)
    mean, info = reg.predict(X, max_iter=10, tol=1e-12, return_info=True)
    assert info["max_iter"] == 10
    for key in (
        "final_weighted_sse",
        "gradient_norm",
        "relative_gradient_norm",
        "fisher_condition",
    ):
        assert info[key].shape == (X.shape[0],)
        assert torch.isfinite(info[key]).all()
    # Exact linear problem: the solution is a stationary point.
    assert float(info["relative_gradient_norm"].max()) < 1e-6
    # The weighted SSE is the residual of the noisy data at the MLE, not zero.
    residual = X - decoder(mean)
    expected_sse = (residual**2 / 0.05**2).sum(dim=1)
    assert torch.allclose(info["final_weighted_sse"], expected_sse, rtol=1e-6, atol=1e-6)

    weight = decoder.weight.detach().to(DTYPE)
    expected_condition = float(torch.linalg.cond(weight.T @ weight))
    assert torch.allclose(
        info["fisher_condition"],
        torch.full((X.shape[0],), expected_condition, dtype=DTYPE),
        rtol=1e-6,
    )
    assert mean.shape == (X.shape[0], 2)

    # Combined return contract.
    mean_both, std_both, info_both = reg.predict(
        X, max_iter=10, tol=1e-12, return_std=True, return_info=True
    )
    assert mean_both.shape == std_both.shape == (X.shape[0], 2)
    assert set(info_both) == set(info)


def test_inversion_diagnostics_flag_rank_deficient_decoder() -> None:
    torch.manual_seed(0)
    # Two label directions with identical observable signatures -> singular Fisher.
    decoder = nn.Linear(2, 3).to(DTYPE)
    with torch.no_grad():
        decoder.weight[:, 1].copy_(decoder.weight[:, 0])
    y = torch.rand(16, 2, dtype=DTYPE)
    with torch.no_grad():
        X = decoder(y)
    reg = _frozen_regressor(decoder, sigma=0.01).fit(X, y, train_decoder=False)
    _, info = reg.predict(X, max_iter=10, tol=1e-12, return_info=True)
    condition = info["fisher_condition"]
    assert bool((~torch.isfinite(condition)).any()) or float(condition.max()) > 1e8
