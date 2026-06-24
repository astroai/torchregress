"""
Conjugate Bayesian linear regression on **fixed features** (Gaussian likelihood).

``BayesianLinearHead`` exposes batch ``fit``; ``RecursiveBayesianHead`` adds
``partial_fit`` with optional forgetting. Multi-output uses independent BLR per
column of ``y`` (shared design :math:`\\Phi`, separate canonical vectors :math:`h`).

See ``docs/test_time/bayesian_linear_regression.md``.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from torchregress.prediction import PredictiveBatch


def _as_tensor(
    x: Union[torch.Tensor, np.ndarray],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.as_tensor(x, device=device, dtype=dtype)
    return x.to(device=device, dtype=dtype)


def _augment_features(phi: torch.Tensor, *, fit_intercept: bool) -> torch.Tensor:
    if not fit_intercept:
        return phi
    ones = torch.ones(phi.shape[0], 1, device=phi.device, dtype=phi.dtype)
    return torch.cat([phi, ones], dim=-1)


def _posterior_covariance_from_precision(
    precision: torch.Tensor,
    jitter: float,
) -> torch.Tensor:
    d = precision.shape[-1]
    eye = torch.eye(d, device=precision.device, dtype=precision.dtype)
    return torch.linalg.solve(precision + jitter * eye, eye)


class BayesianLinearHead(nn.Module):
    r"""
    Batch conjugate Gaussian linear regression on fixed features.

    Posterior precision :math:`\Lambda` and canonical :math:`h = \Lambda m` satisfy
    :math:`\Lambda = \Lambda_0 + \sigma^{-2}\Phi^\top W \Phi` and
    :math:`h = h_0 + \sigma^{-2}\Phi^\top (W y)` after ``fit``.

    Args:
        in_features: Input dimension before optional intercept column.
        out_features: Independent scalar outputs (separate :math:`h` rows).
        fit_intercept: Append a column of ones.
        prior_mean: Scalar broadcast or vector of length ``d_eff``.
        prior_precision: Diagonal prior precision :math:`\tau` with
            :math:`\Lambda_0=\tau I`.
        noise_variance: Homoscedastic :math:`\sigma^2`.  When ``auto_noise=True``
            this is overridden in :meth:`fit` with an estimate derived from
            the training targets.
        auto_noise: If True, :meth:`fit` estimates ``noise_variance`` from
            the training targets as ``max((0.2·σ_y)², 1e-4)``, providing
            a data-driven prior that adapts to the target scale.
        rbf_centers: If set, apply a radial basis function (RBF) feature
            expansion using ``rbf_centers`` random training points as centres.
            This gives the linear model nonlinear capacity (kernel trick).
            Centres and bandwidth (see ``rbf_gamma``) are selected in
            :meth:`fit` and reused during :meth:`predict` / :meth:`partial_fit`.
        rbf_gamma: RBF kernel bandwidth :math:`\gamma`.  If ``None`` (default),
            :meth:`fit` estimates ``1 / (2·median(pairwise-distances)²)``
            from a random subsample of training points.
        jitter: Diagonal jitter on :math:`\Lambda` before Cholesky solves.
    """

    _Lambda0: torch.Tensor
    _h0: torch.Tensor
    _Lambda: torch.Tensor
    _h: torch.Tensor
    _fitted: torch.Tensor
    _n_obs: torch.Tensor
    _rbf_centers: torch.Tensor
    _rbf_gamma: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        *,
        fit_intercept: bool = True,
        prior_mean: Union[float, torch.Tensor] = 0.0,
        prior_precision: float = 1.0,
        noise_variance: float = 1.0,
        auto_noise: bool = False,
        rbf_centers: Optional[int] = None,
        rbf_gamma: Optional[float] = None,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if noise_variance <= 0 or prior_precision <= 0:
            raise ValueError("noise_variance and prior_precision must be positive")
        if not isinstance(auto_noise, bool):
            raise TypeError("auto_noise must be a boolean")
        if rbf_centers is not None and rbf_centers <= 0:
            raise ValueError("rbf_centers must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.fit_intercept = bool(fit_intercept)
        self.auto_noise = bool(auto_noise)
        self.noise_variance = float(noise_variance)
        self.jitter = float(jitter)
        self.prior_precision = float(prior_precision)

        # RBF expansion
        self.rbf_centers = rbf_centers  # None → no expansion; int → num centres
        self._rbf_gamma_user = rbf_gamma  # None → auto (median heuristic)

        # Effective feature dim: RBF centres (if set) + optional intercept
        d_eff = (rbf_centers if rbf_centers is not None else in_features) + int(fit_intercept)
        self._d_eff = d_eff
        lam0 = torch.eye(d_eff, dtype=torch.float32) * prior_precision
        if isinstance(prior_mean, torch.Tensor):
            m0 = prior_mean.reshape(-1).float()
            if m0.numel() not in (1, d_eff):
                raise ValueError(f"prior_mean must have length 1 or {d_eff}, got {m0.numel()}")
            if m0.numel() == 1:
                m0 = m0.expand(d_eff).clone()
        else:
            m0 = torch.full((d_eff,), float(prior_mean), dtype=torch.float32)
        h0 = lam0 @ m0
        self.register_buffer("_Lambda0", lam0.clone())
        self.register_buffer("_h0", h0.clone())
        self.register_buffer("_Lambda", lam0.clone())
        self.register_buffer("_h", torch.zeros(out_features, d_eff, dtype=torch.float32))
        self.register_buffer("_fitted", torch.tensor(0, dtype=torch.uint8))
        self.register_buffer("_n_obs", torch.tensor(0, dtype=torch.long))
        self._h.copy_(h0.unsqueeze(0).expand(out_features, d_eff))

        # RBF buffers — populated lazily in fit() or first partial_fit()
        self.register_buffer(
            "_rbf_centers",
            torch.empty(0, in_features, dtype=torch.float32),
        )
        self.register_buffer(
            "_rbf_gamma",
            torch.tensor(1.0, dtype=torch.float32),
        )

    @property
    def is_fitted(self) -> bool:
        return bool(self._fitted.item())

    def reset_posterior(self) -> None:
        self._Lambda.copy_(self._Lambda0)
        self._h.copy_(self._h0.unsqueeze(0).expand(self.out_features, self._d_eff))
        self._fitted.zero_()
        self._n_obs.zero_()
        if self.rbf_centers is not None:
            self._rbf_centers.resize_(0, self.in_features)
            self._rbf_gamma.fill_(1.0)

    # ------------------------------------------------------------------
    # RBF feature expansion
    # ------------------------------------------------------------------

    def _apply_rbf(self, phi0: torch.Tensor) -> torch.Tensor:
        """Transform raw features through the RBF layer (no-op if not enabled)."""
        if self.rbf_centers is None:
            return phi0
        if phi0.shape[1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} raw features, got {phi0.shape[1]}")
        # exp(-γ · ‖x − c‖²)
        dists_sq = torch.cdist(phi0, self._rbf_centers).pow(2)
        return torch.exp(-self._rbf_gamma * dists_sq)

    def _init_rbf(self, phi0: torch.Tensor, generator: Optional[torch.Generator] = None) -> None:
        """Select RBF centres and bandwidth from training data.

        Idempotent — only runs if ``rbf_centers`` is set and centres have
        not been populated yet (supports lazy initialisation from
        ``partial_fit``).
        """
        if self.rbf_centers is None or self._rbf_centers.numel() > 0:
            return
        n_train = phi0.shape[0]
        n_centers = min(self.rbf_centers, n_train)
        idx = torch.randperm(n_train, generator=generator)[:n_centers].to(phi0.device)
        centers = phi0[idx].clone()
        self._rbf_centers.resize_(centers.shape)
        self._rbf_centers.copy_(centers)

        # Gamma: user-supplied or median-heuristic estimate
        if self._rbf_gamma_user is not None:
            gamma = float(self._rbf_gamma_user)
        else:
            # Median pairwise distance on a subsample (avoid O(N²))
            n_sub = min(n_train, 1000)
            idx_sub = torch.randperm(n_train, device=phi0.device)[:n_sub]
            sub = phi0[idx_sub]
            pdists = torch.pdist(sub)
            if pdists.numel() == 0:
                gamma = 1.0
            else:
                median_sq = pdists.median().item() ** 2
                gamma = 1.0 / max(2.0 * median_sq, 1e-12)
        self._rbf_gamma.fill_(gamma)

    @torch.no_grad()
    def _accumulate(
        self,
        phi: torch.Tensor,
        y: torch.Tensor,
        sample_weight: Optional[torch.Tensor],
    ) -> None:
        n = phi.shape[0]
        inv_sig2 = 1.0 / self.noise_variance
        if sample_weight is None:
            w = torch.ones(n, device=phi.device, dtype=phi.dtype)
        else:
            w = sample_weight.reshape(-1).to(device=phi.device, dtype=phi.dtype)
            if torch.any(w < 0):
                raise ValueError("sample_weight must be non-negative")
        sw = torch.sqrt(w.clamp(min=0.0) + 1e-18)
        phi_w = phi * sw.unsqueeze(-1)
        self._Lambda.addmm_(phi_w.T, phi_w, beta=1.0, alpha=inv_sig2)
        wy = w.unsqueeze(-1) * y
        self._h.addmm_(wy.T, phi, beta=1.0, alpha=inv_sig2)
        self._n_obs.add_(n)

    @torch.no_grad()
    def fit(
        self,
        features: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        sample_weight: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> BayesianLinearHead:
        device = self._Lambda.device
        dtype = self._Lambda.dtype
        phi0 = _as_tensor(features, device=device, dtype=dtype)
        y0 = _as_tensor(y, device=device, dtype=dtype)
        if y0.dim() == 1:
            y0 = y0.unsqueeze(-1)
        if y0.shape[0] != phi0.shape[0]:
            raise ValueError("features and y must have the same number of rows")
        if y0.shape[1] != self.out_features:
            raise ValueError(f"y must have {self.out_features} columns, got {y0.shape[1]}")
        if phi0.shape[1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} features, got {phi0.shape[1]}")
        sw = (
            None if sample_weight is None else _as_tensor(sample_weight, device=device, dtype=dtype)
        )
        self.reset_posterior()

        # --- auto_noise: estimate noise_variance from training targets ---
        if self.auto_noise:
            noise_std = y0.std().item() * 0.2
            self.noise_variance = max(noise_std**2, 1e-4)

        # --- RBF feature expansion (select centres / gamma, then transform) ---
        self._init_rbf(phi0, generator=generator)
        phi0 = self._apply_rbf(phi0)
        phi = _augment_features(phi0, fit_intercept=self.fit_intercept)

        self._accumulate(phi, y0, sw)
        self._fitted.fill_(1)
        return self

    @property
    def posterior_precision(self) -> torch.Tensor:
        return self._Lambda.clone()

    @property
    def posterior_mean(self) -> torch.Tensor:
        lam = self._Lambda + self.jitter * torch.eye(
            self._d_eff, device=self._Lambda.device, dtype=self._Lambda.dtype
        )
        sol = torch.linalg.solve(lam, self._h.T)
        return sol.T

    @property
    def posterior_covariance(self) -> torch.Tensor:
        return _posterior_covariance_from_precision(self._Lambda, self.jitter)

    @torch.no_grad()
    def predict(
        self,
        features: Union[torch.Tensor, np.ndarray],
        *,
        return_std: bool = False,
        include_noise: bool = True,
    ) -> dict[str, torch.Tensor]:
        if not self.is_fitted:
            raise RuntimeError("Call fit before predict.")
        device = self._Lambda.device
        dtype = self._Lambda.dtype
        phi0 = _as_tensor(features, device=device, dtype=dtype)
        phi0 = self._apply_rbf(phi0)
        phi = _augment_features(phi0, fit_intercept=self.fit_intercept)
        lam = self._Lambda + self.jitter * torch.eye(self._d_eff, device=device, dtype=dtype)
        chol = torch.linalg.cholesky(lam)
        m = torch.cholesky_solve(self._h.unsqueeze(-1), chol).squeeze(-1)
        mean = phi @ m.T
        v = torch.cholesky_solve(phi.T, chol).T
        epi = (phi * v).sum(dim=-1, keepdim=True).expand(-1, self.out_features)
        var = epi + (self.noise_variance if include_noise else 0.0)
        std = torch.sqrt(torch.clamp(var, min=0.0))
        out: dict[str, torch.Tensor] = {"mean": mean}
        if return_std:
            out["variance"] = var
            out["std"] = std
        return out

    @torch.no_grad()
    def predictive_batch(
        self,
        features: Union[torch.Tensor, np.ndarray],
        *,
        include_noise: bool = True,
    ) -> PredictiveBatch:
        pred = self.predict(features, return_std=True, include_noise=include_noise)
        mean = pred["mean"]
        std = pred["std"]
        epi = pred["variance"] - self.noise_variance if include_noise else pred["variance"]
        tr_val = float(torch.trace(self.posterior_covariance).item())
        extra = {
            "epistemic_variance": epi,
            "aleatoric_variance": (
                torch.full_like(epi, self.noise_variance)
                if include_noise
                else torch.zeros_like(epi)
            ),
            "posterior_trace": torch.full(mean.shape, tr_val, device=mean.device, dtype=mean.dtype),
            "n_observations_seen": torch.full(
                mean.shape, float(self._n_obs.item()), device=mean.device, dtype=mean.dtype
            ),
        }
        return PredictiveBatch(
            point=mean,
            mean=mean,
            std=std,
            extra=extra,
        )

    @torch.no_grad()
    def sample_weights(
        self,
        n_samples: int,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw weight vectors :math:`w \\sim \\mathcal{N}(m, S)` (independent noise per output)."""
        if not self.is_fitted:
            raise RuntimeError("Call fit before sample_weights.")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        m = self.posterior_mean
        s = self.posterior_covariance
        chol = torch.linalg.cholesky(
            s + self.jitter * torch.eye(self._d_eff, device=s.device, dtype=s.dtype)
        )
        z = torch.randn(
            n_samples,
            self.out_features,
            self._d_eff,
            device=s.device,
            dtype=s.dtype,
            generator=generator,
        )
        return m.unsqueeze(0) + torch.matmul(z, chol.T)


class RecursiveBayesianHead(BayesianLinearHead):
    """
    Incremental conjugate updates with optional forgetting on the precision.

    ``fit`` performs a full reset plus one-shot batch update (same as the parent).
    ``partial_fit`` applies ``forgetting_factor`` to ``Lambda`` (if ``< 1``) then
    adds the new sufficient statistics without clearing prior mass from earlier
    batches (commutative over disjoint batches).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        *,
        fit_intercept: bool = True,
        prior_mean: Union[float, torch.Tensor] = 0.0,
        prior_precision: float = 1.0,
        noise_variance: float = 1.0,
        auto_noise: bool = False,
        rbf_centers: Optional[int] = None,
        rbf_gamma: Optional[float] = None,
        forgetting_factor: float = 1.0,
        jitter: float = 1e-6,
    ) -> None:
        if not 0 < forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must be in (0, 1]")
        super().__init__(
            in_features,
            out_features,
            fit_intercept=fit_intercept,
            prior_mean=prior_mean,
            prior_precision=prior_precision,
            noise_variance=noise_variance,
            auto_noise=auto_noise,
            rbf_centers=rbf_centers,
            rbf_gamma=rbf_gamma,
            jitter=jitter,
        )
        self.forgetting_factor = float(forgetting_factor)

    @torch.no_grad()
    def partial_fit(
        self,
        features: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        sample_weight: Optional[Union[torch.Tensor, np.ndarray]] = None,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> RecursiveBayesianHead:
        device = self._Lambda.device
        dtype = self._Lambda.dtype
        phi0 = _as_tensor(features, device=device, dtype=dtype)
        y0 = _as_tensor(y, device=device, dtype=dtype)
        if y0.dim() == 1:
            y0 = y0.unsqueeze(-1)
        if y0.shape[0] != phi0.shape[0]:
            raise ValueError("features and y must have the same number of rows")
        if y0.shape[1] != self.out_features:
            raise ValueError(f"y must have {self.out_features} columns, got {y0.shape[1]}")
        sw = (
            None if sample_weight is None else _as_tensor(sample_weight, device=device, dtype=dtype)
        )
        if self.forgetting_factor < 1.0:
            self._Lambda.copy_(
                self._Lambda0 + self.forgetting_factor * (self._Lambda - self._Lambda0)
            )
            self._h.copy_(
                self._h0.unsqueeze(0) + self.forgetting_factor * (self._h - self._h0.unsqueeze(0))
            )
            n_obs_scaled = (self._n_obs.float() * self.forgetting_factor).round().to(torch.long)
            self._n_obs.copy_(n_obs_scaled)
        # Lazy RBF init + apply (idempotent — only runs on first call)
        self._init_rbf(phi0, generator=generator)
        phi0 = self._apply_rbf(phi0)
        phi = _augment_features(phi0, fit_intercept=self.fit_intercept)
        self._accumulate(phi, y0, sw)
        self._fitted.fill_(1)
        return self
