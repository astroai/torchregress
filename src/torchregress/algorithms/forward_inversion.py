"""Forward generative regression with test-time maximum-likelihood inversion.

Most of ``torchregress`` is *discriminative*: a model maps a (possibly noisy)
feature vector ``x`` to a target ``y``.  When the features are much noisier than
the labels, that direction of regression suffers from **attenuation bias**: the
learned map is systematically compressed toward the mean because the regression
treats the noisy ``x`` as a fixed design and the label as random.

A generative surrogate inverts the causal direction.  A decoder
``g_theta(y) -> x`` is trained to reproduce the *observables* from the labels
using the known per-pixel measurement errors, and a new object is solved by
maximum likelihood:

.. math::

    \\hat{y} = \\arg\\min_y
        \\sum_d \\frac{(x_d - g_{\\theta,d}(y))^2}{\\sigma_{x,d}^2 + s_d^2}

The construction is classical, not astronomy-specific:

* **Inverse regression / statistical calibration.**  Regressing the *response*
  on the *covariate* and then inverting the fitted relation is the "inverse
  estimator" of Krutchkoff (1967); see Berkson (1950) for why the two
  regressions differ, and Osborne (1991) and Fuller (1987) for the measurement
  error theory.  Inverting a forward model avoids the ``Var(w) = Var(x) +
  sigma_x^2`` dilution of the direct regression, and the training label
  distribution only shapes the design of the surrogate (not the per-object
  inversion).
* **Latent-variable and inverse-problem inference.**  The labels are latent
  variables of a forward model; maximising their likelihood is a MAP inverse
  problem (Tarantola 2005).  For a nonlinear decoder the test-time problem is
  non-convex: Gauss-Newton and L-BFGS are local methods, sensible
  initialisation and convergence diagnostics matter (Bates & Watts 1988;
  Nocedal & Wright 2006; Raue et al. 2009).
* **Energy-based and auto-decoder inference.**  A decoder trained with the
  labels as observed codes, whose labels are recovered by minimising an energy
  at test time, is exactly an *auto-decoder* (Park et al. 2019) and a special
  case of energy-based prediction (LeCun et al. 2006); it is the
  *non-amortised* counterpart of an auto-encoder whose encoder is an explicit
  network.
* **Likelihood-based event reconstruction.**  Physics analyses routinely
  evaluate a forward model's likelihood and infer parameters or per-event
  quantities from it, e.g. the matrix element method in high-energy physics
  (Gainer et al. 2013) and simulation-based inference more broadly
  (Cranmer et al. 2020).

The frequentist covariance of the estimate follows from the Fisher information:

.. math::

    \\mathrm{Cov}(\\hat{y}) =
        \\left(J_y^\\top \\Sigma_x^{-1} J_y\\right)^{-1},
    \\qquad J_y = \\nabla_y g_\\theta(y).

The module is deliberately a small, self-contained estimator: it wraps any
``torch.nn.Module`` decoder and the usual ``fit`` / ``predict`` contract.

References
----------
.. [1] Krutchkoff, R. G. (1967). Classical and inverse regression methods of
       calibration. *Technometrics*, 9(3), 425-439.
       https://doi.org/10.1080/00401706.1967.10490486
.. [2] Berkson, J. (1950). Are there two regressions? *Journal of the American
       Statistical Association*, 45(250), 164-180.
.. [3] Osborne, C. (1991). Statistical calibration: a review. *International
       Statistical Review*, 59(3), 309-336.
.. [4] Fuller, W. A. (1987). *Measurement Error Models*. Wiley.
.. [5] Bates, D. M., & Watts, D. G. (1988). *Nonlinear Regression Analysis and
       Its Applications*. Wiley.  (Gauss-Newton convergence, curvature and
       parameter-effects nonlinearity.)
.. [6] Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.).
       Springer.  (Levenberg-Marquardt / L-BFGS are local methods; no global
       convergence guarantee.)
.. [7] Raue, A., Kreutz, C., Maiwald, T., et al. (2009). Structural and
       practical identifiability analysis of partially observed dynamical
       models by exploiting the profile likelihood. *Bioinformatics*, 25(15),
       1923-1929. https://doi.org/10.1093/bioinformatics/btp358
.. [8] Tarantola, A. (2005). *Inverse Problem Theory and Methods for Model
       Parameter Estimation*. SIAM. https://doi.org/10.1137/1.9780898717921
.. [9] LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006).
       A tutorial on energy-based learning. In *Predicting Structured Data*.
       MIT Press.
.. [10] Park, J. J., Florence, P., Straub, J., Newcombe, R., & Lovegrove, S.
        (2019). DeepSDF: Learning continuous signed distance functions for
        shape representation. *CVPR 2019*. https://arxiv.org/abs/1901.05103
.. [11] Ng, A. Y., & Jordan, M. I. (2002). On discriminative vs. generative
        classifiers: a comparison of logistic regression and naive Bayes.
        *NeurIPS 2001*.
.. [12] Gainer, J. S., Lykken, J., Matchev, K. T., Mrenna, S., & Park, M.
        (2013). The matrix element method: past, present, and future.
        Snowmass white paper. https://arxiv.org/abs/1307.3546
.. [13] Cranmer, K., Brehmer, J., & Louppe, G. (2020). The frontier of
        simulation-based inference. *PNAS*, 117(48), 30055-30062.
        https://doi.org/10.1073/pnas.1912789117
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from ..utils.validation import check_tensor

__all__ = ["ForwardInversionRegressor", "build_decoder_mlp"]


def build_decoder_mlp(
    n_labels: int,
    n_observables: int,
    hidden_dims: Sequence[int] = (128, 128),
    activation: str = "silu",
) -> nn.Sequential:
    """Build a simple fully-connected decoder ``y -> x``.

    Args:
        n_labels: Number of label (target) dimensions.
        n_observables: Number of observable (feature) dimensions.
        hidden_dims: Hidden layer widths.
        activation: ``"silu"``, ``"relu"`` or ``"tanh"``.

    Returns:
        A ``torch.nn.Sequential`` mapping ``(N, n_labels)`` to
        ``(N, n_observables)``.
    """
    n_labels = int(n_labels)
    n_observables = int(n_observables)
    if n_labels < 1 or n_observables < 1:
        raise ValueError("n_labels and n_observables must be positive")
    activations = {
        "silu": nn.SiLU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    activation_key = activation.lower()
    if activation_key not in activations:
        raise ValueError(f"activation must be one of {sorted(activations)}, got {activation!r}")
    act = activations[activation_key]

    layers: list[nn.Module] = []
    in_dim = n_labels
    for width in hidden_dims:
        width = int(width)
        if width < 1:
            raise ValueError("hidden_dims entries must be positive")
        layers.extend([nn.Linear(in_dim, width), act()])
        in_dim = width
    layers.append(nn.Linear(in_dim, n_observables))
    return nn.Sequential(*layers)


class ForwardInversionRegressor:
    """Generative ``y -> x`` surrogate with test-time maximum-likelihood inversion.

    The decoder is trained to predict the observables from the labels.  By
    default the training objective is the **full heteroscedastic Gaussian
    likelihood** with per-observable measurement error ``sigma_x`` and an
    optional learnable model-mismatch variance ``s_d``:

    .. math::

        \\mathcal{L}(\\theta) = \\frac{1}{N D} \\sum_{i=1}^N \\sum_{d=1}^D
            \\left[
                \\frac{(x_{i,d} - g_{\\theta,d}(y_i))^2}
                     {\\sigma_{x,i,d}^2 + s_d^2}
                + \\ln\\!\\left(\\sigma_{x,i,d}^2 + s_d^2\\right)
            \\right].

    ``likelihood="student_t"`` replaces the Gaussian with a heavy-tailed
    Student-t likelihood (robust to unmodelled outliers and defective pixels
    during *training*), and ``likelihood="wls"`` drops the log-determinant term,
    giving
    heteroskedastic weighted least squares with fixed ``sigma_x``
    (``learn_noise=False`` required).  Plain unweighted MSE is recovered by a
    constant ``sigma_x`` with ``learn_noise=False``; it is the least efficient
    choice when per-pixel errors are known.

    At prediction time the labels of a new object are recovered by minimising
    the weighted residual (the Gaussian/Student-t MLE objective for fixed
    scale) with a damped Gauss-Newton method or L-BFGS, and the frequentist
    covariance is ``(J^T Sigma_x^{-1} J)^{-1}``.  The inversion itself uses
    fixed Gaussian weights, so defective *test* observables should be masked
    (``predict(..., mask=...)``) or down-weighted; the Student-t likelihood only
    makes the decoder robust during training.

    Structurally this is an **auto-decoder** (Park et al. 2019) / energy-based
    predictor (LeCun et al. 2006): the decoder is trained with the labels as
    observed codes, and a new label vector is found by test-time optimisation
    rather than by an explicit encoder network.  It is the non-amortised
    counterpart of an auto-encoder and the classical inverse-regression
    estimator of Krutchkoff (1967) in modern (neural) form.  It avoids the
    regression-dilution (attenuation) that a discriminative ``x -> y`` model
    inherits when ``sigma_x`` is large.

    Args:
        decoder: Optional decoder module mapping ``(N, n_labels)`` to
            ``(N, n_observables)``.  It is **cloned** before fitting, so the
            caller's module is never modified in place.  When ``None``, a
            fully-connected MLP with ``hidden_dims`` is built at ``fit`` time.
        n_labels: Number of label dimensions.  Optional when ``decoder`` is
            given (inferred from ``y`` at fit time).
        n_observables: Number of observable dimensions.  Optional when
            ``decoder`` is given (inferred from ``X`` at fit time).
        hidden_dims: Hidden widths of the default MLP decoder.
        activation: Activation of the default MLP decoder.
        sigma_x: Known measurement standard deviation of the observables.  May
            be a scalar, a ``(D,)`` feature vector or a per-sample ``(N, D)``
            matrix.  ``None`` means "unknown": with ``learn_noise=True`` the
            per-observable residual variance is learned instead.
        learn_noise: Learn an extra per-observable model-mismatch variance
            ``s_d^2`` on top of ``sigma_x^2``.
        likelihood: Training objective: ``"gaussian"`` (full heteroscedastic
            Gaussian NLL, default), ``"student_t"`` (heavy-tailed robust NLL)
            or ``"wls"`` (heteroskedastic weighted least squares, no
            log-determinant; requires ``learn_noise=False`` and known
            ``sigma_x``).
        degrees_of_freedom: Degrees of freedom of the Student-t likelihood
            (must be > 2 for a finite variance).  Ignored for the other
            likelihoods.
        noise_floor: Numerical floor on ``s_d^2`` (and on the total variance).
        epochs: Number of training epochs.
        batch_size: Mini-batch size (the full training set when ``None``).
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay (does not apply to ``sigma_x``).
        seed: Optional seed for mini-batch shuffling and the validation split.
        device: Optional device; defaults to the device of ``X``.
        validation_fraction: Hold out this fraction of rows for early stopping.
            Ignored when explicit ``X_val``/``y_val`` are passed to ``fit``.
        patience: Early-stopping patience in epochs.  ``None`` uses 50 when
            validation data is available and disables early stopping otherwise.
        verbose: Print the training loss every epoch.

    Attributes:
        decoder_: The fitted decoder (``eval`` mode after ``fit``).
        noise_variance_: Fitted ``(D,)`` extra variance ``s_d^2``.
        label_center_: Mean training label, used as the default inversion
            initialisation.
        loss_history_: Training loss per epoch.
        validation_history_: Validation loss per epoch (empty without a
            validation split).

    Example:
        >>> import torch
        >>> from torchregress.algorithms import ForwardInversionRegressor
        >>>
        >>> torch.manual_seed(0)
        >>> y = torch.rand(256, 2) * 4 - 2                      # labels
        >>> A = torch.randn(2, 8)                              # y -> 8 pixels
        >>> X = y @ A + 0.05 * torch.randn(256, 8)             # noisy fluxes
        >>>
        >>> reg = ForwardInversionRegressor(
        ...     n_labels=2, n_observables=8, hidden_dims=(32,),
        ...     sigma_x=0.05, epochs=100, batch_size=64, seed=0,
        ... )
        >>> reg.fit(X, y)
        ForwardInversionRegressor(...)
        >>> y_hat = reg.predict(X)
        >>> y_hat.shape
        torch.Size([256, 2])
    """

    def __init__(
        self,
        decoder: nn.Module | None = None,
        *,
        n_labels: int | None = None,
        n_observables: int | None = None,
        hidden_dims: Sequence[int] = (128, 128),
        activation: str = "silu",
        sigma_x: float | torch.Tensor | None = None,
        learn_noise: bool = True,
        likelihood: str = "gaussian",
        degrees_of_freedom: float = 5.0,
        noise_floor: float = 1.0e-8,
        epochs: int = 200,
        batch_size: int | None = 64,
        lr: float = 1.0e-3,
        weight_decay: float = 0.0,
        seed: int | None = None,
        device: str | torch.device | None = None,
        validation_fraction: float = 0.0,
        patience: int | None = None,
        verbose: bool = False,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not 0.0 <= float(validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if batch_size is not None and int(batch_size) < 1:
            raise ValueError("batch_size must be >= 1")
        likelihood_key = str(likelihood).lower()
        if likelihood_key not in {"gaussian", "student_t", "wls"}:
            raise ValueError(
                f"likelihood must be 'gaussian', 'student_t' or 'wls', got {likelihood!r}"
            )
        if likelihood_key == "student_t" and float(degrees_of_freedom) <= 2.0:
            raise ValueError(
                "student_t likelihood requires degrees_of_freedom > 2 "
                f"(finite variance), got {degrees_of_freedom}"
            )
        if likelihood_key == "wls" and learn_noise:
            raise ValueError(
                "likelihood='wls' uses fixed heteroskedastic weights; set learn_noise=False "
                "(the weighted least-squares objective has no variance parameter)"
            )
        self.decoder = decoder
        self.n_labels = None if n_labels is None else int(n_labels)
        self.n_observables = None if n_observables is None else int(n_observables)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.activation = activation
        self.sigma_x = sigma_x
        self.learn_noise = bool(learn_noise)
        self.likelihood = likelihood_key
        self.degrees_of_freedom = float(degrees_of_freedom)
        self.noise_floor = float(noise_floor)
        self.epochs = int(epochs)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = seed
        self.device = device
        self.validation_fraction = float(validation_fraction)
        self.patience = patience
        self.verbose = bool(verbose)

        # Fitted attributes.
        self.decoder_: nn.Module | None = None
        self.noise_variance_: torch.Tensor | None = None
        self.label_center_: torch.Tensor | None = None
        self.loss_history_: list[float] = []
        self.validation_history_: list[float] = []
        self.n_labels_: int | None = None
        self.n_observables_: int | None = None
        self._sigma_template_: torch.Tensor | None = None
        self._fitted = False
        self.decoder_trained_ = False

    # ------------------------------------------------------------------ #
    # sigma_x handling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_sigma_var(
        sigma: float | torch.Tensor | None,
        reference: torch.Tensor,
        *,
        name: str = "sigma_x",
    ) -> torch.Tensor:
        """Expand a standard-deviation spec to a variance tensor like ``reference``."""
        if sigma is None:
            return torch.zeros_like(reference)
        if isinstance(sigma, (int, float)):
            value = float(sigma)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            return torch.full_like(reference, value * value)
        if not isinstance(sigma, torch.Tensor):
            raise TypeError(f"{name} must be a float or a tensor, got {type(sigma).__name__}")
        sigma_t = sigma.to(device=reference.device, dtype=reference.dtype)
        if (sigma_t < 0).any():
            raise ValueError(f"{name} must be non-negative")
        if sigma_t.ndim == 0:
            return torch.full_like(reference, float(sigma_t.item()) ** 2)
        if sigma_t.ndim == 1:
            if sigma_t.shape[0] != reference.shape[-1]:
                raise ValueError(
                    f"{name} vector must have length {reference.shape[-1]}, got {sigma_t.shape[0]}"
                )
            return (sigma_t**2).reshape(1, -1).expand_as(reference)
        if sigma_t.ndim == 2:
            if sigma_t.shape == reference.shape:
                return sigma_t**2
            if sigma_t.shape[0] == reference.shape[0] and sigma_t.shape[1] == 1:
                return sigma_t**2
            raise ValueError(
                f"{name} matrix must match {tuple(reference.shape)}, got {tuple(sigma_t.shape)}"
            )
        raise ValueError(f"{name} must be scalar, 1-D or 2-D, got {sigma_t.ndim}-D")

    def _elementwise_nll(self, residual: torch.Tensor, total_var: torch.Tensor) -> torch.Tensor:
        """Per-observable negative log-likelihood for the selected objective.

        All three objectives are minimised by the same weighted least-squares
        inversion at prediction time; the likelihood only changes how the
        decoder and the noise floor are *trained*.
        """
        if self.likelihood == "gaussian":
            return 0.5 * (residual * residual / total_var + torch.log(total_var))
        if self.likelihood == "wls":
            return 0.5 * (residual * residual / total_var)
        dof = self.degrees_of_freedom
        return 0.5 * torch.log(total_var) + 0.5 * (dof + 1.0) * torch.log1p(
            residual * residual / (dof * total_var)
        )

    # ------------------------------------------------------------------ #
    # fitting
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        sigma_x: float | torch.Tensor | None = None,
        train_decoder: bool = True,
    ) -> "ForwardInversionRegressor":
        """Fit the forward decoder on ``(x, y)`` pairs.

        Args:
            X: Observed features (noisy fluxes/photometry), shape ``(N, D_x)``.
            y: Labels (high-precision targets), shape ``(N,)``, ``(N, 1)`` or
                ``(N, D_y)``.
            X_val, y_val: Optional explicit validation tensors.
            sigma_x: Known measurement standard deviation of ``X``; overrides
                the value passed to the constructor.
            train_decoder: When ``False`` the decoder is frozen (e.g. a
                pre-trained surrogate) and only the noise floor / label centre
                are estimated.

        Returns:
            ``self``, fitted.
        """
        check_tensor(X, "X")
        check_tensor(y, "y")
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D tensor (N, D), got {tuple(X.shape)}")
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        elif y.ndim != 2:
            raise ValueError(f"y must be 1-D or 2-D, got {tuple(y.shape)}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows, got {X.shape[0]} and {y.shape[0]}"
            )
        if not X.is_floating_point() or not y.is_floating_point():
            raise TypeError("X and y must be floating-point tensors")

        device = torch.device(self.device) if self.device is not None else X.device
        X = X.to(device)
        y = y.to(device)
        dtype = X.dtype
        y = y.to(dtype)

        n_samples, n_observables = X.shape
        n_labels = y.shape[1]
        self._check_dims(n_labels=n_labels, n_observables=n_observables)

        # Decoder: clone a user-supplied module so the original is untouched.
        if self.decoder is None:
            if self.seed is not None:
                # Local RNG scope: deterministic init without perturbing global state.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(int(self.seed))
                    decoder = build_decoder_mlp(
                        n_labels, n_observables, self.hidden_dims, self.activation
                    )
            else:
                decoder = build_decoder_mlp(
                    n_labels, n_observables, self.hidden_dims, self.activation
                )
            decoder = decoder.to(device=device, dtype=dtype)
        else:
            decoder = copy.deepcopy(self.decoder).to(device=device, dtype=dtype)

        # Measurement-error variance, resolved once on the full training set.
        sigma_spec = self.sigma_x if sigma_x is None else sigma_x
        if self.likelihood == "wls" and sigma_spec is None:
            raise ValueError("likelihood='wls' requires known sigma_x")
        if sigma_spec is None and not self.learn_noise:
            raise ValueError("provide sigma_x or set learn_noise=True")
        sigma_var = self._as_sigma_var(sigma_spec, X)

        # Template used by predict() when no sigma_x is given at inference time.
        if sigma_spec is None:
            self._sigma_template_ = None
        elif isinstance(sigma_spec, torch.Tensor) and sigma_spec.ndim == 2:
            # Per-sample errors cannot be reused at inference: require sigma_x.
            self._sigma_template_ = None
        else:
            self._sigma_template_ = sigma_var.mean(dim=0).detach().clone()

        # Optional explicit validation split / hold-out.
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be provided together")
        sigma_var_val: torch.Tensor | None = None
        if X_val is None and self.validation_fraction > 0.0 and n_samples > 2:
            generator = torch.Generator(device=device)
            if self.seed is not None:
                generator.manual_seed(int(self.seed))
            perm = torch.randperm(n_samples, generator=generator, device=device)
            n_val = max(1, int(round(self.validation_fraction * n_samples)))
            n_val = min(n_val, n_samples - 1)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
            X_val, y_val = X[val_idx], y[val_idx]
            sigma_var_val = sigma_var[val_idx]
            X, y, sigma_var = X[train_idx], y[train_idx], sigma_var[train_idx]
        elif X_val is not None:
            assert y_val is not None
            X_val = X_val.to(device=device, dtype=dtype)
            y_val = y_val.to(device=device, dtype=dtype)
            if y_val.ndim == 1:
                y_val = y_val.unsqueeze(-1)
            if X_val.ndim != 2 or X_val.shape[1] != n_observables:
                raise ValueError("X_val must have shape (N_val, D_x)")
            if y_val.shape[1] != n_labels:
                raise ValueError("y_val label dimension does not match y")
            sigma_var_val = torch.zeros_like(X_val)

        n_train = X.shape[0]
        if n_train < 1:
            raise ValueError("need at least one training sample")

        # Trainable model-mismatch variance s_d^2 (optional).
        noise_var: torch.Tensor
        log_noise: torch.Tensor | None = None
        if self.learn_noise:
            x_var = X.var(dim=0, unbiased=False).mean().clamp_min(self.noise_floor)
            init = (0.01 * x_var).clamp_min(self.noise_floor)
            log_noise = nn.Parameter(torch.log(init).expand(n_observables).clone())
            noise_var = torch.exp(log_noise).clamp_min(self.noise_floor)
        else:
            noise_var = torch.zeros(n_observables, dtype=dtype, device=device)

        decoder_was_trained = bool(train_decoder and self.epochs > 0)
        if train_decoder:
            params: list[nn.Parameter] = list(decoder.parameters())
            if log_noise is not None:
                params.append(log_noise)
            train_decoder = bool(params)
            optimizer = (
                torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
                if params
                else None
            )
        else:
            optimizer = None

        generator = torch.Generator(device=device)
        if self.seed is not None:
            generator.manual_seed(int(self.seed))
        batch_size = n_train if self.batch_size is None else min(self.batch_size, n_train)

        self.loss_history_ = []
        self.validation_history_ = []
        best_state: dict[str, Any] | None = None
        best_val = float("inf")
        wait = 0
        patience = self.patience if self.patience is not None else 50

        for epoch in range(self.epochs if train_decoder else 0):
            if optimizer is None:
                break
            decoder.train()
            perm = torch.randperm(n_train, generator=generator, device=device)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                if log_noise is not None:
                    noise_var = torch.exp(log_noise).clamp_min(self.noise_floor)
                pred = decoder(y[idx])
                total_var = (sigma_var[idx] + noise_var.reshape(1, -1)).clamp_min(self.noise_floor)
                residual = X[idx] - pred
                loss = self._elementwise_nll(residual, total_var).mean()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                n_batches += 1
            epoch_loss = total_loss / max(n_batches, 1)
            self.loss_history_.append(epoch_loss)
            if self.verbose:
                print(f"epoch {epoch + 1:4d}/{self.epochs}  loss={epoch_loss:.6g}")

            if X_val is not None and y_val is not None:
                decoder.eval()
                with torch.no_grad():
                    val_pred = decoder(y_val)
                    if log_noise is not None:
                        noise_var = torch.exp(log_noise).clamp_min(self.noise_floor)
                    val_sigma = (
                        sigma_var_val if sigma_var_val is not None else torch.zeros_like(X_val)
                    )
                    val_total = (val_sigma + noise_var.reshape(1, -1)).clamp_min(self.noise_floor)
                    val_loss = float(self._elementwise_nll(X_val - val_pred, val_total).mean())
                self.validation_history_.append(val_loss)
                if val_loss < best_val - 1e-12:
                    best_val = val_loss
                    best_state = copy.deepcopy(decoder.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if train_decoder and wait >= patience:
                        break

        if best_state is not None:
            decoder.load_state_dict(best_state)

        decoder.eval()
        with torch.no_grad():
            final_noise = (
                torch.exp(log_noise).clamp_min(self.noise_floor)
                if log_noise is not None
                else torch.zeros(n_observables, dtype=dtype, device=device)
            )
            if not train_decoder:
                # Frozen decoder: calibrate the residual variance directly.
                residual = X - decoder(y)
                resid_var = (residual * residual).mean(dim=0)
                if self.learn_noise:
                    final_noise = (resid_var - sigma_var.mean(dim=0)).clamp_min(self.noise_floor)
                else:
                    final_noise = torch.zeros_like(resid_var)
            noise_var = final_noise

        self.decoder_ = decoder
        self.noise_variance_ = noise_var.detach().clone()
        self.label_center_ = y.mean(dim=0).detach().clone()
        self.n_labels_ = n_labels
        self.n_observables_ = n_observables
        self._fitted = True
        # Keep the frozen/training distinction visible for diagnostics.
        self.decoder_trained_ = decoder_was_trained
        return self

    def _check_dims(self, *, n_labels: int, n_observables: int) -> None:
        if self.n_labels is not None and self.n_labels != n_labels:
            raise ValueError(f"n_labels={self.n_labels} does not match y dimension {n_labels}")
        if self.n_observables is not None and self.n_observables != n_observables:
            raise ValueError(
                f"n_observables={self.n_observables} does not match X dimension {n_observables}"
            )

    # ------------------------------------------------------------------ #
    # inversion internals
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if not self._fitted or self.decoder_ is None:
            raise RuntimeError("ForwardInversionRegressor must be fitted before predicting")
        assert self.n_labels_ is not None
        assert self.noise_variance_ is not None

    def _decoder_forward(self, y: torch.Tensor) -> torch.Tensor:
        assert self.decoder_ is not None
        return self.decoder_(y)

    def _weighted_sse(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        weights: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = X - self._decoder_forward(y)
        w = weights
        if mask is not None:
            residual = residual * mask
            w = w * mask
        return (residual * residual * w).sum(dim=1)

    def _observation_jacobian(
        self,
        y: torch.Tensor,
        *,
        chunk_size: int | None,
        jacobian_method: str,
    ) -> torch.Tensor:
        """Jacobian ``d g / d y`` with shape ``(N, D_x, D_y)``."""
        if jacobian_method not in {"vmap", "loop"}:
            raise ValueError(f"jacobian_method must be 'vmap' or 'loop', got {jacobian_method!r}")
        decoder = self.decoder_
        assert decoder is not None

        def f_single(y_single: torch.Tensor) -> torch.Tensor:
            return decoder(y_single.unsqueeze(0)).squeeze(0)

        def one_chunk(values: torch.Tensor) -> torch.Tensor:
            if jacobian_method == "vmap":
                return torch.func.vmap(torch.func.jacfwd(f_single))(values)
            return torch.stack([torch.func.jacfwd(f_single)(row) for row in values], dim=0)

        n = y.shape[0]
        if chunk_size is None or chunk_size >= n:
            return one_chunk(y)
        chunks = [one_chunk(y[i : i + int(chunk_size)]) for i in range(0, n, int(chunk_size))]
        return torch.cat(chunks, dim=0)

    def _invert(
        self,
        X: torch.Tensor,
        weights: torch.Tensor,
        *,
        method: str,
        init: torch.Tensor | None,
        max_iter: int,
        tol: float,
        chunk_size: int | None,
        mask: torch.Tensor | None,
        jacobian_method: str,
    ) -> torch.Tensor:
        assert self.label_center_ is not None
        assert self.decoder_ is not None
        # Inversion is a deterministic single-object problem: batch-dependent
        # layers (dropout, batch-norm batch statistics) must be in eval mode.
        self.decoder_.eval()
        n, _ = X.shape
        if init is None:
            y0 = self.label_center_.reshape(1, -1).expand(n, -1).clone()
        else:
            init_t = torch.as_tensor(init, dtype=X.dtype, device=X.device)
            if init_t.ndim == 1:
                if init_t.shape[-1] != self.label_center_.shape[-1]:
                    raise ValueError(
                        f"init must have length {self.label_center_.shape[-1]}, "
                        f"got {init_t.shape[-1]}"
                    )
                y0 = init_t.reshape(1, -1).expand(n, -1).clone()
            elif init_t.ndim == 2 and init_t.shape == (n, self.label_center_.shape[-1]):
                y0 = init_t.clone()
            else:
                raise ValueError(
                    f"init must have shape ({self.label_center_.shape[-1]},) or "
                    f"({n}, {self.label_center_.shape[-1]}), got {tuple(init_t.shape)}"
                )
        y0 = y0.detach()
        if method == "gauss_newton":
            with torch.no_grad():
                return self._gauss_newton(
                    X,
                    weights,
                    y0,
                    max_iter=max_iter,
                    tol=tol,
                    chunk_size=chunk_size,
                    mask=mask,
                    jacobian_method=jacobian_method,
                ).detach()
        if method == "lbfgs":
            with torch.enable_grad():
                return self._lbfgs(X, weights, y0, max_iter=max_iter, tol=tol, mask=mask).detach()
        raise ValueError(f"method must be 'gauss_newton' or 'lbfgs', got {method!r}")

    def _gauss_newton(
        self,
        X: torch.Tensor,
        weights: torch.Tensor,
        y0: torch.Tensor,
        *,
        max_iter: int,
        tol: float,
        chunk_size: int | None,
        mask: torch.Tensor | None,
        jacobian_method: str,
    ) -> torch.Tensor:
        n, n_labels = y0.shape
        y = y0.clone()
        eye = torch.eye(n_labels, dtype=X.dtype, device=X.device).expand(n, -1, -1)
        damping = torch.full((n,), 1.0e-3, dtype=X.dtype, device=X.device)
        loss = self._weighted_sse(X, y, weights, mask)
        best_y = y.clone()
        best_loss = loss.clone()
        if not torch.isfinite(loss).all():
            raise RuntimeError("non-finite initial inversion loss; check inputs for NaN/Inf")

        for _ in range(int(max_iter)):
            jacobian = self._observation_jacobian(
                y, chunk_size=chunk_size, jacobian_method=jacobian_method
            ).detach()
            residual = X - self._decoder_forward(y)
            w_eff = weights if mask is None else weights * mask
            if mask is not None:
                residual = residual * mask
            jw = jacobian * w_eff.unsqueeze(-1)
            normal = torch.einsum("ndk,ndl->nkl", jw, jacobian)
            rhs = torch.einsum("ndk,nd->nk", jw, residual)
            diag_scale = (
                torch.diagonal(normal, dim1=1, dim2=2).mean(dim=1).clamp_min(self.noise_floor)
            )
            normal_damped = normal + (damping * diag_scale).reshape(-1, 1, 1) * eye
            try:
                delta = torch.linalg.solve(normal_damped, rhs.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                delta = (torch.linalg.pinv(normal_damped) @ rhs.unsqueeze(-1)).squeeze(-1)

            accepted = torch.zeros(n, dtype=torch.bool, device=X.device)
            for backtrack in range(8):
                step = delta * (0.5**backtrack)
                candidate = y + step
                candidate_loss = self._weighted_sse(X, candidate, weights, mask)
                newly = (~accepted) & (candidate_loss < loss)
                if bool(newly.any()):
                    y = torch.where(newly.unsqueeze(-1), candidate, y)
                    loss = torch.where(newly, candidate_loss, loss)
                    accepted = accepted | newly
                if bool(accepted.all()):
                    break

            previous_best = best_loss.clone()
            improved = loss < best_loss
            best_y = torch.where(improved.unsqueeze(-1), y, best_y)
            best_loss = torch.where(improved, loss, best_loss)
            damping = torch.where(accepted, damping / 3.0, damping * 2.0).clamp(1e-10, 1e10)

            improvement = float((previous_best - best_loss).max().detach())
            if improvement < tol and bool(accepted.all()):
                break
            if bool((~accepted).all()):
                # No sample could improve: the damping has saturated the step.
                break
        return best_y

    def _lbfgs(
        self,
        X: torch.Tensor,
        weights: torch.Tensor,
        y0: torch.Tensor,
        *,
        max_iter: int,
        tol: float,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        y = y0.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [y],
            lr=1.0,
            max_iter=int(max_iter),
            tolerance_grad=float(tol),
            tolerance_change=float(tol),
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            objective = self._weighted_sse(X, y, weights, mask).sum()
            objective.backward()
            return objective

        optimizer.step(closure)
        return y.detach()

    # ------------------------------------------------------------------ #
    # public prediction API
    # ------------------------------------------------------------------ #
    def _prepare_inputs(
        self,
        X: torch.Tensor,
        sigma_x: float | torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self._check_fitted()
        check_tensor(X, "X")
        if X.ndim == 1:
            X = X.unsqueeze(0)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (N, D), got {tuple(X.shape)}")
        if self.n_observables_ is not None and X.shape[1] != self.n_observables_:
            raise ValueError(f"X has {X.shape[1]} observables, expected {self.n_observables_}")
        device = torch.device(self.device) if self.device is not None else X.device
        X = X.to(device)
        if not X.is_floating_point():
            raise TypeError("X must be floating-point")
        dtype = X.dtype
        if self.noise_variance_ is None:
            raise RuntimeError("model is not fitted")
        noise = self.noise_variance_.to(device=device, dtype=dtype)
        if sigma_x is None:
            if self._sigma_template_ is None:
                sigma_var = torch.zeros_like(X)
            else:
                sigma_var = (
                    self._sigma_template_.to(device=device, dtype=dtype).reshape(1, -1).expand_as(X)
                )
        else:
            sigma_var = self._as_sigma_var(sigma_x, X)
        total = (sigma_var + noise.reshape(1, -1)).clamp_min(self.noise_floor)
        weights = 1.0 / total
        if mask is not None:
            mask = mask.to(device=device, dtype=torch.bool)
            if mask.shape != X.shape:
                raise ValueError(
                    f"mask must match X shape {tuple(X.shape)}, got {tuple(mask.shape)}"
                )
        return X, weights, mask

    def predict(
        self,
        X: torch.Tensor,
        *,
        sigma_x: float | torch.Tensor | None = None,
        method: str = "gauss_newton",
        init: torch.Tensor | None = None,
        max_iter: int = 50,
        tol: float = 1.0e-8,
        chunk_size: int | None = None,
        mask: torch.Tensor | None = None,
        return_std: bool = False,
        return_info: bool = False,
        jacobian_method: str = "vmap",
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, dict[str, Any]]
        | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]
    ):
        """Recover labels by maximum-likelihood inversion.

        Args:
            X: Observed features, shape ``(N, D_x)`` (or ``(D_x,)`` for one
                object).
            sigma_x: Measurement standard deviation of ``X``.  Defaults to the
                value used at fit time (when it was a scalar / ``(D,)`` vector)
                or to zero.
            method: ``"gauss_newton"`` (damped Levenberg-Marquardt) or
                ``"lbfgs"``.
            init: Optional initial labels ``(D_y,)`` or ``(N, D_y)``.  Defaults
                to the mean training label.
            max_iter: Maximum number of inversion iterations.
            tol: Convergence tolerance on the weighted residual.
            chunk_size: Optional batch chunking for Jacobian evaluation.
            mask: Optional boolean ``(N, D_x)`` mask of *observed* entries.
                Masked entries do not contribute to the objective.
            return_std: When ``True`` also return the marginal standard
                deviations.
            return_info: When ``True`` also return a diagnostics dictionary
                with per-object ``final_weighted_sse``, ``gradient_norm``,
                ``relative_gradient_norm``, ``fisher_condition`` (condition
                number of ``J^T Sigma_x^{-1} J``, a local identifiability
                indicator) and the ``max_iter`` cap.  Test-time inversion is a
                local, non-convex optimisation: these diagnostics should be
                inspected before trusting the estimates.
            jacobian_method: ``"vmap"`` (fast, requires vmap-safe decoder) or
                ``"loop"`` (per-sample forward-mode autodiff fallback).

        Returns:
            ``(N, D_y)`` label estimates.  With ``return_std=True`` returns
            ``(mean, std)``; with ``return_info=True`` returns
            ``(mean, info)``; with both returns ``(mean, std, info)``.
        """
        X, weights, mask = self._prepare_inputs(X, sigma_x, mask)
        mean = self._invert(
            X,
            weights,
            method=method,
            init=init,
            max_iter=max_iter,
            tol=tol,
            chunk_size=chunk_size,
            mask=mask,
            jacobian_method=jacobian_method,
        )
        std: torch.Tensor | None = None
        if return_std:
            covariance = self._covariance(
                X,
                weights,
                mean,
                mask=mask,
                chunk_size=chunk_size,
                jacobian_method=jacobian_method,
            )
            std = torch.diagonal(covariance, dim1=1, dim2=2).clamp_min(0.0).sqrt()
        info: dict[str, Any] | None = None
        if return_info:
            info = self._inversion_diagnostics(
                X,
                weights,
                mean,
                mask=mask,
                chunk_size=chunk_size,
                jacobian_method=jacobian_method,
                max_iter=max_iter,
            )
        if std is not None and info is not None:
            return mean, std, info
        if std is not None:
            return mean, std
        if info is not None:
            return mean, info
        return mean

    def predict_with_covariance(
        self,
        X: torch.Tensor,
        *,
        sigma_x: float | torch.Tensor | None = None,
        method: str = "gauss_newton",
        init: torch.Tensor | None = None,
        max_iter: int = 50,
        tol: float = 1.0e-8,
        chunk_size: int | None = None,
        mask: torch.Tensor | None = None,
        jacobian_method: str = "vmap",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover labels and their frequentist covariance.

        Returns:
            ``(mean, covariance)`` with shapes ``(N, D_y)`` and
            ``(N, D_y, D_y)``.  The covariance is
            ``(J^T Sigma_x^{-1} J)^{-1}`` evaluated at the solution.
        """
        X, weights, mask = self._prepare_inputs(X, sigma_x, mask)
        mean = self._invert(
            X,
            weights,
            method=method,
            init=init,
            max_iter=max_iter,
            tol=tol,
            chunk_size=chunk_size,
            mask=mask,
            jacobian_method=jacobian_method,
        )
        covariance = self._covariance(
            X, weights, mean, mask=mask, chunk_size=chunk_size, jacobian_method=jacobian_method
        )
        return mean, covariance

    def _inversion_diagnostics(
        self,
        X: torch.Tensor,
        weights: torch.Tensor,
        y: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        chunk_size: int | None,
        jacobian_method: str,
        max_iter: int,
    ) -> dict[str, Any]:
        """Per-object local diagnostics of a test-time inversion.

        The objective is a non-convex function of ``y`` for a nonlinear
        decoder.  These quantities are cheap, local sanity checks, not
        certificates of a global optimum:

        * ``final_weighted_sse``: value of the inversion objective.
        * ``gradient_norm`` / ``relative_gradient_norm``: stationarity of the
          solution (the relative version divides by ``1 + |sse|``, so it is
          scale-free).
        * ``fisher_condition``: condition number of the Fisher/normal matrix
          ``J^T Sigma_x^{-1} J``; very large values indicate weak
          identifiability in at least one label direction (and make the
          linearised covariance unreliable).
        """
        jacobian = self._observation_jacobian(
            y, chunk_size=chunk_size, jacobian_method=jacobian_method
        ).detach()
        w_eff = weights if mask is None else weights * mask
        residual = X - self._decoder_forward(y)
        if mask is not None:
            residual = residual * mask
        jw = jacobian * w_eff.unsqueeze(-1)
        fisher = torch.einsum("ndk,ndl->nkl", jw, jacobian)
        fisher = 0.5 * (fisher + fisher.transpose(1, 2))
        # d/dy sum_i w_i r_i^2 = -2 J^T W r
        gradient = -2.0 * torch.einsum("ndk,nd->nk", jw, residual)
        sse = (residual * residual * w_eff).sum(dim=1)
        gradient_norm = torch.linalg.vector_norm(gradient, dim=1)
        return {
            "final_weighted_sse": sse.detach(),
            "gradient_norm": gradient_norm.detach(),
            "relative_gradient_norm": (gradient_norm / (1.0 + sse.abs())).detach(),
            "fisher_condition": torch.linalg.cond(fisher).detach(),
            "max_iter": int(max_iter),
        }

    def _covariance(
        self,
        X: torch.Tensor,
        weights: torch.Tensor,
        y: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        chunk_size: int | None,
        jacobian_method: str = "vmap",
    ) -> torch.Tensor:
        jacobian = self._observation_jacobian(
            y, chunk_size=chunk_size, jacobian_method=jacobian_method
        ).detach()
        w_eff = weights if mask is None else weights * mask
        jw = jacobian * w_eff.unsqueeze(-1)
        fisher = torch.einsum("ndk,ndl->nkl", jw, jacobian)
        n, n_labels = y.shape
        eye = torch.eye(n_labels, dtype=X.dtype, device=X.device).expand(n, -1, -1)
        jitter = (
            10.0
            * torch.finfo(X.dtype).eps
            * torch.diagonal(fisher, dim1=1, dim2=2).mean(dim=1, keepdim=True)
        )
        fisher = fisher + jitter.reshape(-1, 1, 1) * eye
        try:
            chol = torch.linalg.cholesky(fisher)
            covariance = torch.cholesky_inverse(chol)
        except RuntimeError:
            covariance = torch.linalg.pinv(fisher)
        covariance = 0.5 * (covariance + covariance.transpose(1, 2))
        return covariance
