"""
Simulation Extrapolation (SIMEX) implementation.

SIMEX is a simulation-based method for correcting measurement error in inputs.
It works by adding additional measurement error to the data, establishing a trend
of how the error affects predictions, and extrapolating back to the case of no error.
"""

from typing import Callable, List, Optional, Union, cast

import torch
import torch.nn as nn

from ..utils.validation import check_tensor


class SIMEX:
    """
    Simulation Extrapolation (SIMEX) algorithm.

    SIMEX estimates the effect of measurement error by adding simulated noise
    of varying magnitudes to the data, re-training the model, and extrapolating
    the predictions to the case of zero measurement error.

    Args:
        model_factory: A callable that returns a new instance of the model to be trained.
        train_func: A callable that takes (model, X, y) and trains the model.
                    It should return the trained model.
        sigma_u: Standard deviation (scalar/vector) or covariance matrix of measurement error.
        lambdas: List of noise multipliers to simulate. Default is [0.5, 1.0, 1.5, 2.0].
                 Lambda represents the added variance ratio: Var_added = lambda * Sigma_u.
        n_simulations: Number of Monte Carlo replicates per lambda. Public SIMEX
                 implementations typically average over many perturbation replicates;
                 values larger than 1 materially reduce extrapolation variance.
        extrapolation_order: Order of the polynomial for extrapolation (1 for linear,
                             2 for quadratic). Default is 2.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        train_func: Callable[[nn.Module, torch.Tensor, torch.Tensor], nn.Module],
        sigma_u: Union[float, torch.Tensor],
        lambdas: Optional[List[float]] = None,
        n_simulations: int = 1,
        extrapolation_order: int = 2,
    ):
        self.model_factory = model_factory
        self.train_func = train_func
        self.sigma_u_input = sigma_u
        self.lambdas = lambdas if lambdas is not None else [0.5, 1.0, 1.5, 2.0]
        self.n_simulations = int(n_simulations)
        self.extrapolation_order = extrapolation_order

        self.trained_models: List[nn.Module] = []
        self.models_by_lambda: List[List[nn.Module]] = []
        self.device: Optional[torch.device] = None
        self.sigma_u: Optional[torch.Tensor] = None
        self.extrapolation_weights: Optional[torch.Tensor] = None

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
                return sigma
            raise ValueError(f"sigma_u must be scalar, vector, or matrix, got {sigma.ndim}D tensor")

        raise TypeError(f"sigma_u must be float or tensor, got {type(sigma).__name__}")

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> "SIMEX":
        """
        Fit the SIMEX models.

        Args:
            X_train: Noisy input tensor of shape (N, D)
            y_train: Target tensor

        Returns:
            self
        """

        check_tensor(X_train, "X_train")
        check_tensor(y_train, "y_train")

        self.device = X_train.device
        n_features = X_train.shape[1]
        self.sigma_u = self._prepare_sigma_u(n_features, self.device)

        if self.n_simulations < 1:
            raise ValueError("n_simulations must be >= 1")

        self.trained_models = []
        self.models_by_lambda = []

        # Train base model (lambda=0) if not included in lambdas, but usually SIMEX
        # uses the original data as lambda=0 point.
        # However, standard SIMEX workflow often treats the original data as one point
        # and added noise as others.
        # We will explicitly include lambda=0 (original data) in our list for simulation
        # to ensure the curve is anchored at the observed data.

        all_lambdas = [0.0] + [lam for lam in self.lambdas if lam > 0.0]
        # Remove duplicates and sort
        all_lambdas = sorted(list(set(all_lambdas)))
        self.lambdas_used = all_lambdas  # Store for prediction

        # Pre-calculate Cholesky for noise generation
        # Add small epsilon for stability
        L = torch.linalg.cholesky(self.sigma_u + torch.eye(n_features, device=self.device) * 1e-6)

        for lam in self.lambdas_used:
            lambda_models: List[nn.Module] = []
            for _ in range(self.n_simulations):
                model = self.model_factory().to(self.device)

                if lam == 0.0:
                    X_sim = X_train
                else:
                    # Add noise: Variance_added = lambda * Sigma_u
                    # Noise = N(0, lambda * Sigma_u) = sqrt(lambda) * N(0, Sigma_u)
                    #       = sqrt(lambda) * epsilon @ L.T
                    noise = torch.randn_like(X_train) @ L.T
                    X_sim = X_train + torch.sqrt(torch.tensor(lam, device=self.device)) * noise

                trained_model = self.train_func(model, X_sim, y_train)
                lambda_models.append(trained_model)
                self.trained_models.append(trained_model)
            self.models_by_lambda.append(lambda_models)

        # Precompute extrapolation weights
        # We solve A * Beta = Y for Beta, then predict y_target = target_vec @ Beta.
        # This is equivalent to y_target = (target_vec @ pinv(A)) @ Y
        # So weights = target_vec @ pinv(A)
        lambdas = torch.tensor(self.lambdas_used, device=self.device, dtype=X_train.dtype)

        # Design matrix A for polynomial fit
        A_cols = [torch.ones_like(lambdas)]
        for order in range(1, self.extrapolation_order + 1):
            A_cols.append(lambdas**order)

        A = torch.stack(A_cols, dim=1)  # (M, order+1)
        A_pinv = torch.linalg.pinv(A)

        lambda_target = -1.0
        target_vec = torch.tensor(
            [lambda_target**i for i in range(self.extrapolation_order + 1)],
            device=self.device,
            dtype=X_train.dtype,
        )  # (order+1,)

        self.extrapolation_weights = target_vec @ A_pinv

        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict using SIMEX extrapolation.

        Args:
            X: Input tensor

        Returns:
            Extrapolated predictions
        """

        check_tensor(X, "X")

        if not self.models_by_lambda:
            raise RuntimeError("SIMEX must be fit before predicting")

        if X.device != self.device:
            X = X.to(self.device)

        # Collect predictions from all models
        # Shape: (n_lambdas, n_samples, n_outputs)
        preds_list = []
        with torch.no_grad():
            for lambda_models in self.models_by_lambda:
                lambda_preds = []
                for model in lambda_models:
                    model.eval()
                    lambda_preds.append(model(X))
                preds_list.append(torch.stack(lambda_preds, dim=0).mean(dim=0))

        # Stack predictions
        # (M, N, K) where M is num lambdas
        Y_stack = torch.stack(preds_list, dim=0)

        # Use precomputed weights to combine predictions
        # weights: (M,)
        # Y_stack: (M, N, K)
        # result: (N, K)
        return cast(
            torch.Tensor,
            torch.tensordot(self.extrapolation_weights, Y_stack, dims=([0], [0])),
        )
