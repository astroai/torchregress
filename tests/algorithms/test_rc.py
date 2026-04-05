import pytest
import torch

from torchregress.algorithms.rc import RegressionCalibration


def test_rc_initialization():
    # Scalar
    rc = RegressionCalibration(sigma_u=0.1)
    assert rc.sigma_u_input == 0.1

    # Vector
    rc = RegressionCalibration(sigma_u=torch.tensor([0.1, 0.2]))
    assert torch.equal(rc.sigma_u_input, torch.tensor([0.1, 0.2]))


def test_rc_synthetic_correction():
    # Generate data similar to the prompt
    torch.manual_seed(42)
    n_samples = 5000
    noise_std = 0.5

    # True latent X ~ N(0, 1)
    X_true = torch.randn(n_samples, 1)

    # Target Y = 3*X + epsilon
    Y = 3 * X_true + torch.randn(n_samples, 1) * 0.1

    # Observed W = X + error
    W_obs = X_true + torch.randn(n_samples, 1) * noise_std

    # Naive slope (attenuation bias)
    # Slope = Cov(W, Y) / Var(W)
    # Var(W) = Var(X) + Var(U) = 1 + 0.25 = 1.25
    # Cov(W, Y) = Cov(X+U, 3X+E) = 3Var(X) = 3
    # Expected slope = 3 / 1.25 = 2.4

    # Fit RC
    rc = RegressionCalibration(sigma_u=noise_std)
    rc.fit(W_obs)
    X_cal = rc.transform(W_obs)

    # Check if calibrated data restores slope
    # We can check simple linear regression slope

    # Naive regression
    w_centered = W_obs - W_obs.mean()
    y_centered = Y - Y.mean()
    naive_slope = (w_centered * y_centered).sum() / (w_centered**2).sum()

    # Calibrated regression
    x_cal_centered = X_cal - X_cal.mean()
    cal_slope = (x_cal_centered * y_centered).sum() / (x_cal_centered**2).sum()

    print(f"Naive Slope: {naive_slope.item():.4f}")
    print(f"RC Slope: {cal_slope.item():.4f}")

    assert naive_slope.item() < 2.6  # Expect attenuation (approx 2.4)
    assert cal_slope.item() > 2.8 and cal_slope.item() < 3.2  # Expect correction (approx 3.0)


def test_rc_multivariate():
    torch.manual_seed(42)
    n_samples = 1000
    n_features = 2

    # X ~ N(0, I)
    X_true = torch.randn(n_samples, n_features)

    # Noise covariance
    sigma_u = torch.tensor([[0.1, 0.05], [0.05, 0.2]])
    L = torch.linalg.cholesky(sigma_u)
    noise = torch.randn(n_samples, n_features) @ L.T

    W_obs = X_true + noise

    rc = RegressionCalibration(sigma_u=sigma_u)
    X_cal = rc.fit_transform(W_obs)

    assert X_cal.shape == W_obs.shape

    # Verify reliability matrix shape
    assert rc.reliability_matrix.shape == (n_features, n_features)


def test_rc_error_handling():
    rc = RegressionCalibration(sigma_u=0.1)
    with pytest.raises(RuntimeError):
        rc.transform(torch.randn(10, 1))

    with pytest.raises(ValueError):
        rc.fit(torch.randn(10, 1, 1))  # Wrong dim


def test_rc_falls_back_to_stable_diagonal_reliability_when_full_matrix_is_unstable():
    torch.manual_seed(0)
    X = torch.randn(512, 3)
    # Use an unrealistically large error covariance to force the fallback path.
    rc = RegressionCalibration(sigma_u=torch.eye(3) * 50.0)
    rc.fit(X)
    diag = torch.diagonal(rc.reliability_matrix)
    assert torch.isfinite(rc.reliability_matrix).all()
    assert torch.all(diag >= 0.0)
    assert torch.all(diag <= 1.0)


def test_rc_posterior_returns_mean_and_covariance():
    torch.manual_seed(0)
    x_true = torch.randn(128, 2)
    x_obs = x_true + 0.2 * torch.randn_like(x_true)
    rc = RegressionCalibration(sigma_u=0.2).fit(x_obs)
    post_mean, post_cov = rc.posterior(x_obs)
    assert post_mean.shape == x_obs.shape
    assert post_cov.shape == (2, 2)
    assert torch.isfinite(post_mean).all()
    assert torch.isfinite(post_cov).all()
    assert float(torch.linalg.eigvalsh(post_cov).min()) >= 0.0


def test_rc_posterior_supports_per_sample_diagonal_noise():
    torch.manual_seed(1)
    x_true = torch.randn(64, 3)
    x_obs = x_true + 0.1 * torch.randn_like(x_true)
    rc = RegressionCalibration(sigma_u=0.1).fit(x_obs)
    sigma_u = torch.full_like(x_obs, 0.15)
    post_mean, post_cov = rc.posterior(x_obs, sigma_u=sigma_u)
    assert post_mean.shape == x_obs.shape
    assert post_cov.shape == (64, 3, 3)
    assert torch.isfinite(post_cov).all()
    min_eigs = torch.linalg.eigvalsh(post_cov).min(dim=-1).values
    assert torch.all(min_eigs >= 0.0)
