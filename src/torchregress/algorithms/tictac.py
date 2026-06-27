"""
Taylor-Induced Covariance (TIC) parameterization for deep heteroscedastic regression.

Reference: Shukla et al., "TIC-TAC: A Framework For Improved Covariance Estimation
In Deep Heteroscedastic Regression" (ICML 2024).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class TaylorInducedCovarianceHead(nn.Module):
    """
    Taylor-Induced Covariance (TIC) head.

    Ties predicted covariance to the gradient (Jacobian) and curvature (Hessian)
    of the mean prediction backbone function with respect to the input features.

    Specifically:
    cov = k1(x) * J(x) J(x)^T + k2(x) * H(x) + diag(k3(x)) + jitter * I
    where H(x)_ij = Tr(H_i(x) H_j(x)).

    References
    ----------
    .. [1] Shukla, S., et al. (2024). TIC-TAC: A Framework For Improved Covariance
       Estimation In Deep Heteroscedastic Regression. In *ICML 2024*.
       https://arxiv.org/abs/2310.18953
    """

    def __init__(
        self,
        base_model: nn.Module,
        target_dim: int,
        input_dim: Optional[int] = None,
        k1_init: float = 1.0,
        k2_init: float = 1.0,
        k3_init: float = 1.0,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.target_dim = target_dim
        self.jitter = jitter

        if input_dim is not None:
            # Input-dependent parameters
            self.k1_net = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            self.k2_net = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
            self.k3_net = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, target_dim),
            )
            self.is_input_dependent = True
        else:
            # Global learnable parameters
            self.log_k1 = nn.Parameter(torch.tensor(math.log(k1_init)))
            self.log_k2 = nn.Parameter(torch.tensor(math.log(k2_init)))
            self.log_k3 = nn.Parameter(torch.full((target_dim,), math.log(k3_init)))
            self.is_input_dependent = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and Taylor-Induced Covariance.

        Args:
            x: Input features of shape [B, input_dim]

        Returns:
            Tuple of (mean, covariance_matrix) of shapes:
            - mean: [B, target_dim]
            - covariance_matrix: [B, target_dim, target_dim]
        """
        # Ensure input has batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 1. Compute mean prediction
        mean = self.base_model(x)

        # 2. Compute Jacobian and Hessian of mean w.r.t input features
        # using the torch.func functional API
        params = dict(self.base_model.named_parameters())
        buffers = dict(self.base_model.named_buffers())

        def functional_model_fn(
            p: dict[str, torch.Tensor], b: dict[str, torch.Tensor], x_val: torch.Tensor
        ) -> torch.Tensor:
            return torch.func.functional_call(
                self.base_model, (p, b), (x_val.unsqueeze(0),)
            ).squeeze(0)

        # Autograd w.r.t input features (argnums=2)
        jac_fn = torch.func.jacrev(functional_model_fn, argnums=2)
        hess_fn = torch.func.hessian(functional_model_fn, argnums=2)

        # Batch vectorization using vmap
        jac_batch_fn = torch.vmap(jac_fn, in_dims=(None, None, 0))
        hess_batch_fn = torch.vmap(hess_fn, in_dims=(None, None, 0))

        # Shapes: jacs -> [B, target_dim, input_dim]
        # hesses -> [B, target_dim, input_dim, input_dim]
        jacs = jac_batch_fn(params, buffers, x)
        hesses = hess_batch_fn(params, buffers, x)

        # 3. Compute Jacobian and Hessian covariance matrices
        # Jacobian term: J J^T of shape [B, target_dim, target_dim]
        jj_t = torch.bmm(jacs, jacs.transpose(1, 2))

        # Hessian term: H_ij = Tr(H_i H_j) of shape [B, target_dim, target_dim]
        # ponytail: Tr(H_i @ H_j) = Σ_k Σ_m H_i[k,m] * H_j[m,k]; for symmetric
        # Hessians the Frobenius inner product (bikm,bjkm) gives the same result,
        # but the correct trace contraction is bikm,bjmk.
        h_matrix = torch.einsum("bikm,bjmk->bij", hesses, hesses)

        # 4. Predict scaling parameters k1, k2, and residual diagonal k3
        if self.is_input_dependent:
            k1 = torch.exp(self.k1_net(x)).unsqueeze(-1)  # [B, 1, 1]
            k2 = torch.exp(self.k2_net(x)).unsqueeze(-1)  # [B, 1, 1]
            k3 = torch.exp(self.k3_net(x))  # [B, target_dim]
        else:
            k1 = torch.exp(self.log_k1)
            k2 = torch.exp(self.log_k2)
            k3 = torch.exp(self.log_k3).expand(mean.shape[0], -1)

        # 5. Add stabilizer jitter to ensure positive definiteness
        eye = torch.eye(self.target_dim, device=x.device, dtype=x.dtype).unsqueeze(0)
        cov = (
            k1 * jj_t
            + k2 * h_matrix
            + torch.diag_embed(k3).to(device=x.device, dtype=x.dtype)
            + eye * self.jitter
        )

        return mean, cov
