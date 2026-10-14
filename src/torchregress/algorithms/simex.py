"""
Simulation Extrapolation (SIMEX) implementation.

SIMEX is a simulation-based method for correcting measurement error in inputs.
It works by adding additional measurement error to the data, establishing a trend
of how the error affects predictions, and extrapolating back to the case of no error.
"""

import math
from collections.abc import Callable

import torch
import torch.nn as nn

from ..utils.validation import check_tensor

# Relative diagonal jitters tried (in order) when ``Sigma_u`` is only positive
# semi-definite; each is a fraction of the mean variance so the noise stays
# scale-equivariant (an absolute jitter swamps features with variance << jitter).
_CHOLESKY_JITTERS = (1e-10, 1e-8, 1e-6)


def _noise_cholesky(sigma_u: torch.Tensor) -> torch.Tensor:
    """Cholesky factor of ``sigma_u`` for simulating measurement noise.

    The factor of the covariance itself is used whenever it exists, so the simulated
    noise has exactly ``lambda * Sigma_u`` variance.  Only if the factorisation fails
    (singular PSD matrix) is a jitter proportional to the mean variance added.

    Raises
    ------
    ValueError
        If ``sigma_u`` is not positive semi-definite.
    """
    try:
        return torch.linalg.cholesky(sigma_u)
    except RuntimeError as exc:
        error = exc
    if not bool((sigma_u != 0).any()):
        return torch.zeros_like(sigma_u)  # no measurement error: a valid (zero) factor
    scale = torch.diagonal(sigma_u).mean().clamp_min(torch.finfo(sigma_u.dtype).tiny)
    eye = torch.eye(sigma_u.shape[0], device=sigma_u.device, dtype=sigma_u.dtype)
    for jitter in _CHOLESKY_JITTERS:
        try:
            return torch.linalg.cholesky(sigma_u + jitter * scale * eye)
        except RuntimeError as exc:
            error = exc
    raise ValueError(f"sigma_u is not PSD (cholesky failed): {error}") from error


class SIMEX:
    """
    Simulation Extrapolation (SIMEX) algorithm.

    SIMEX estimates the effect of measurement error by adding simulated noise
    of varying magnitudes to the data, re-training the model, and extrapolating
    estimands or predictions to the case of zero measurement error (lambda = -1).

    Note on Extrapolation Levels:
    - **Prediction SIMEX**: Extrapolates the final model predictions at test time. This is
      a heuristic supported via the `predict` method.

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

    References
    ----------
    .. [1] Cook, J. R., & Stefanski, L. A. (1994). Simulation-Extrapolation Estimation
       in Parametric Measurement Error Models. In *JASA*, 89(428), 1314-1328.
       https://www.jstor.org/stable/2290994
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        train_func: Callable[[nn.Module, torch.Tensor, torch.Tensor], nn.Module],
        sigma_u: float | torch.Tensor,
        lambdas: list[float] | None = None,
        n_simulations: int = 5,
        extrapolation_order: int = 2,
    ):
        self.model_factory = model_factory
        self.train_func = train_func
        self.sigma_u_input = sigma_u
        self.lambdas = lambdas if lambdas is not None else [0.5, 1.0, 1.5, 2.0]
        self.n_simulations = int(n_simulations)
        self.extrapolation_order = extrapolation_order

        self.trained_models: list[nn.Module] = []
        self.models_by_lambda: list[list[nn.Module]] = []
        self.device: torch.device | None = None
        self.sigma_u: torch.Tensor | None = None
        self.extrapolation_weights: torch.Tensor | None = None

    def _prepare_sigma_u(
        self, n_features: int, device: torch.device, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Converts input sigma_u to a full covariance matrix in ``dtype``.

        ``dtype`` should be the dtype of the data; it defaults to the dtype of a
        tensor ``sigma_u`` and to the torch default dtype for python scalars.
        """
        sigma = self.sigma_u_input
        if isinstance(sigma, (int, float)):
            eye_dtype = dtype if dtype is not None else torch.get_default_dtype()
            return torch.eye(n_features, device=device, dtype=eye_dtype) * float(sigma) ** 2

        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device)
            if dtype is not None:
                sigma = sigma.to(dtype)
            if sigma.numel() == 1:
                return torch.eye(n_features, device=device, dtype=sigma.dtype) * (
                    float(sigma.item()) ** 2
                )
            if sigma.ndim == 1:
                if sigma.shape[0] != n_features:
                    raise ValueError(
                        f"sigma_u vector shape {sigma.shape} doesn't match features {n_features}"
                    )
                # Coverage invariants (TOR003): chain .to() on torch.diag because
                # torch.diag does not accept device=/dtype= kwargs natively.
                return torch.diag(sigma**2).to(device=device, dtype=sigma.dtype)
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
        self.sigma_u = self._prepare_sigma_u(n_features, self.device, X_train.dtype)

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
        # Pre-calculate Cholesky for noise generation (with PSD validation)
        L = _noise_cholesky(self.sigma_u)

        for lam in self.lambdas_used:
            lambda_models: list[nn.Module] = []
            for _ in range(self.n_simulations):
                model = self.model_factory().to(self.device)
                if lam == 0.0:
                    X_sim = X_train
                else:
                    # Add noise: Variance_added = lambda * Sigma_u
                    # Noise = N(0, lambda * Sigma_u) = sqrt(lambda) * N(0, Sigma_u)
                    #       = sqrt(lambda) * epsilon @ L.T
                    noise = torch.randn_like(X_train) @ L.T
                    X_sim = X_train + math.sqrt(lam) * noise

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
        )
        self.extrapolation_weights = target_vec @ A_pinv
        # ponytail: warn if Vandermonde conditioning amplifies Monte Carlo error
        if bool((self.extrapolation_weights.abs() > 5).any()):
            import warnings

            warnings.warn(
                f"SIMEX extrapolation weights {self.extrapolation_weights.tolist()} have |w|>5; "
                "small Monte Carlo error will be amplified (consider order=1 or fewer lambdas)",
                UserWarning,
                stacklevel=2,
            )

        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict using SIMEX extrapolation with matched noise.

        To maintain consistency between training and test domains, test
        inputs are perturbed with matched noise at each λ level before being
        fed to the models trained at that level, each model receiving its own
        independent noise draw so that the level estimate averages over all
        ``n_simulations`` remeasurements. Predictions are then extrapolated to
        λ = -1 (zero measurement error).

        Args:
            X: Input tensor (raw observed test data with measurement error)

        Returns:
            Extrapolated predictions (N, K)
        """

        check_tensor(X, "X")

        if not self.models_by_lambda:
            raise RuntimeError("SIMEX must be fit before predicting")

        if X.device != self.device:
            X = X.to(self.device)

        if self.sigma_u is None:
            raise RuntimeError("SIMEX must be fit before predicting")

        L = _noise_cholesky(self.sigma_u.to(dtype=X.dtype))
        # Collect predictions from all models WITH matched input noise.  Each of the
        # B models at level lambda_i receives its OWN remeasurement
        # X + sqrt(lambda_i) * N(0, Sigma_u): the lambda-level estimate is the average
        # over B independent remeasurements, so sharing one draw would leave the
        # Monte Carlo noise of a single replicate in the prediction.
        preds_list = []
        with torch.no_grad():
            for lambda_models, lam in zip(self.models_by_lambda, self.lambdas_used):
                lambda_preds = []
                for model in lambda_models:
                    model.eval()
                    if lam == 0.0:
                        X_lam = X
                    else:
                        X_lam = X + math.sqrt(lam) * (torch.randn_like(X) @ L.T)
                    lambda_preds.append(model(X_lam))
                preds_list.append(torch.stack(lambda_preds, dim=0).mean(dim=0))

        # Stack predictions (M, N, K) where M is num lambdas
        Y_stack = torch.stack(preds_list, dim=0)

        # Use precomputed weights to extrapolate from λ≥0 to λ=-1
        # weights: (M,)  Y_stack: (M, N, K)  result: (N, K)
        return torch.tensordot(self.extrapolation_weights, Y_stack, dims=([0], [0]))
