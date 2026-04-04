"""
Regression Calibration implementation.

Regression Calibration is a method for correcting measurement error in inputs
(Errors-in-Variables) by estimating the true values of the inputs before training.
"""

from typing import Optional, Union

import torch

from ..utils.validation import check_tensor


class RegressionCalibration:
    """
    Regression Calibration (RC) for correcting measurement error.

    RC estimates the conditional expectation E[X|W] where X is the true input
    and W is the observed noisy input. This implementation assumes Gaussian distributions
    for both the signal and the noise.

    The calibration formula is:
    X_cal = mu_w + (Sigma_x @ (Sigma_x + Sigma_u)^-1) @ (W - mu_w)

    where:
    - W is the observed noisy input
    - Sigma_u is the measurement error covariance (assumed known)
    - Sigma_w is the observed covariance of W
    - Sigma_x = Sigma_w - Sigma_u is the estimated signal covariance
    - mu_w is the mean of W

    Args:
        sigma_u: Standard deviation (scalar/vector) or covariance matrix of measurement error.
    """

    def __init__(self, sigma_u: Union[float, torch.Tensor]):
        self.sigma_u_input = sigma_u
        self.mu_w: Optional[torch.Tensor] = None
        self.sigma_w: Optional[torch.Tensor] = None
        self.sigma_u: Optional[torch.Tensor] = None
        self.signal_covariance: Optional[torch.Tensor] = None
        self.reliability_matrix: Optional[torch.Tensor] = None
        self.device: Optional[torch.device] = None

    def _prepare_sigma_u(self, n_features: int, device: torch.device) -> torch.Tensor:
        """Converts input sigma_u to a full covariance matrix."""
        sigma = self.sigma_u_input
        if isinstance(sigma, (int, float)):
            return torch.eye(n_features, device=device) * float(sigma) ** 2

        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device)
            if sigma.numel() == 1:
                return torch.eye(n_features, device=device) * float(sigma.item()) ** 2
            if sigma.ndim == 1:
                if sigma.shape[0] != n_features:
                    raise ValueError(
                        f"sigma_u vector shape {sigma.shape} doesn't match features {n_features}"
                    )
                return torch.diag(sigma**2)
            if sigma.ndim == 2:
                if sigma.shape != (n_features, n_features):
                    raise ValueError(
                        f"sigma_u matrix shape {sigma.shape} "
                        f"doesn't match ({n_features}, {n_features})"
                    )
                return sigma  # Assumed to be covariance matrix already, not std
            raise ValueError(f"sigma_u must be scalar, vector, or matrix, got {sigma.ndim}D tensor")

        raise TypeError(f"sigma_u must be float or tensor, got {type(sigma).__name__}")

    def fit(self, X_observed: torch.Tensor) -> "RegressionCalibration":
        """
        Fit the calibration parameters using the observed noisy data.

        Args:
            X_observed: Noisy input tensor of shape (N, D)

        Returns:
            self
        """
        check_tensor(X_observed, "X_observed")

        if X_observed.ndim != 2:
            raise ValueError("X_observed must be a 2D tensor (N, D)")

        n_samples, n_features = X_observed.shape
        self.device = X_observed.device

        # 1. Estimate properties of the observed data
        self.mu_w = torch.mean(X_observed, dim=0)

        # Compute sample covariance of W
        # (N, D) -> centered (N, D)
        X_centered = X_observed - self.mu_w
        # (D, N) @ (N, D) -> (D, D)
        self.sigma_w = (X_centered.T @ X_centered) / (n_samples - 1)

        # 2. Prepare noise covariance Sigma_u
        self.sigma_u = self._prepare_sigma_u(n_features, self.device)

        # 3. Estimate variance of TRUE X (Sigma_x = Sigma_w - Sigma_u)
        sigma_x = self.sigma_w - self.sigma_u

        # Ensure Sigma_x is Positive Semi-Definite
        # Simple approach: Eigen-decomposition and clipping negative eigenvalues
        L, Q = torch.linalg.eigh(sigma_x)
        L_clipped = torch.clamp(L, min=1e-6)  # Clip small/negative eigenvalues
        sigma_x_psd = Q @ torch.diag(L_clipped) @ Q.T

        # 4. Calculate Reliability Ratio/Matrix
        # Lambda = Sigma_x @ (Sigma_x + Sigma_u)^-1
        # Note: Sigma_x + Sigma_u approximates Sigma_w (reconstructed)
        # We use the PSD version of Sigma_x plus Sigma_u
        denominator = sigma_x_psd + self.sigma_u

        # Robust inverse
        try:
            denom_inv = torch.linalg.inv(denominator)
        except RuntimeError:
            # Fallback to pseudoinverse or add jitter
            denom_inv = torch.linalg.pinv(denominator)

        reliability = sigma_x_psd @ denom_inv

        # In high-dimensional or badly conditioned tabular settings, the full
        # multivariate reliability matrix can become numerically unstable and
        # cease to behave like an attenuation operator. Fall back to a clipped
        # diagonal reliability ratio in those cases.
        reliability_absmax = float(reliability.abs().max())
        reliability_diag = torch.diagonal(reliability)
        unstable = (
            not torch.isfinite(reliability).all()
            or reliability_absmax > 10.0
            or float(reliability_diag.min()) < -1.0e-3
            or float(reliability_diag.max()) > 1.5
        )
        if unstable:
            sigma_x_diag = torch.diagonal(sigma_x_psd).clamp_min(1.0e-6)
            sigma_u_diag = torch.diagonal(self.sigma_u).clamp_min(1.0e-6)
            reliability_ratio = (sigma_x_diag / (sigma_x_diag + sigma_u_diag)).clamp(0.0, 1.0)
            reliability = torch.diag(reliability_ratio)

        self.signal_covariance = sigma_x_psd
        self.reliability_matrix = reliability

        return self

    def transform(self, X_observed: torch.Tensor) -> torch.Tensor:
        """
        Apply regression calibration to the observed data.

        Args:
            X_observed: Noisy input tensor of shape (N, D)

        Returns:
            Calibrated input tensor of shape (N, D)
        """
        check_tensor(X_observed, "X_observed")

        if self.reliability_matrix is None or self.mu_w is None:
            raise RuntimeError("RegressionCalibration must be fit before calling transform")

        if X_observed.device != self.device:
            # Handle device mismatch gracefully
            X_observed = X_observed.to(self.device)

        # X_cal = mu_w + (W - mu_w) @ Lambda.T
        # Note: In the formula Lambda is typically applied from the left to column vectors:
        # x_cal = mu + Lambda @ (w - mu)
        # For batch row vectors: X_cal = Mu + (W - Mu) @ Lambda^T

        X_centered = X_observed - self.mu_w
        X_cal = self.mu_w + X_centered @ self.reliability_matrix.T

        return X_cal

    def fit_transform(self, X_observed: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step."""
        return self.fit(X_observed).transform(X_observed)

    def posterior(
        self,
        X_observed: torch.Tensor,
        sigma_u: Optional[Union[float, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and covariance for latent clean inputs.

        When ``sigma_u`` is omitted, this uses the noise specification stored during ``fit``.
        If ``sigma_u`` is a ``(N, D)`` tensor it is interpreted as per-sample diagonal
        standard deviations and a batched posterior covariance is returned.
        """
        check_tensor(X_observed, "X_observed")
        if self.mu_w is None or self.signal_covariance is None or self.device is None:
            raise RuntimeError("RegressionCalibration must be fit before calling posterior")

        X_observed = X_observed.to(self.device)
        signal = self.signal_covariance
        n_features = X_observed.shape[-1]

        sigma_value = self.sigma_u_input if sigma_u is None else sigma_u
        if isinstance(sigma_value, torch.Tensor) and sigma_value.ndim == 2 and sigma_value.shape == X_observed.shape:
            sigma_diag = sigma_value.to(self.device, dtype=X_observed.dtype).clamp_min(1.0e-6)
            sigma_u_cov = torch.diag_embed(sigma_diag.pow(2))
            denom = signal.unsqueeze(0) + sigma_u_cov
            gain = signal.unsqueeze(0) @ torch.linalg.pinv(denom)
            centered = (X_observed - self.mu_w).unsqueeze(-1)
            post_mean = self.mu_w.unsqueeze(0) + (gain @ centered).squeeze(-1)
            post_cov = signal.unsqueeze(0) - gain @ signal.unsqueeze(0)
            post_cov = 0.5 * (post_cov + post_cov.transpose(-1, -2))
            return post_mean, post_cov

        sigma_u_cov = self.sigma_u
        if sigma_value is not None and sigma_u is not None:
            sigma_u_cov = self._prepare_sigma_u(n_features, self.device)
        if sigma_u_cov is None:
            raise RuntimeError("sigma_u is not available; fit the calibrator first")
        gain = signal @ torch.linalg.pinv(signal + sigma_u_cov)
        post_mean = self.mu_w + (X_observed - self.mu_w) @ gain.T
        post_cov = signal - gain @ signal
        post_cov = 0.5 * (post_cov + post_cov.T)
        return post_mean, post_cov
