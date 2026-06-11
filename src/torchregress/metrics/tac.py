"""
Task-Agnostic Correlations (TAC) metric for evaluating predicted covariance.

Reference: Shukla et al., "TIC-TAC: A Framework For Improved Covariance Estimation
In Deep Heteroscedastic Regression" (ICML 2024).
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import torch
from torchmetrics import Metric

from .utils import convert_to_tensor, metric_state_tensor, validate_inputs


class TaskAgnosticCorrelations(Metric):
    """
    Task-Agnostic Correlations (TAC) metric.

    Measures covariance accuracy by iteratively masking one dimension of target
    variables and predicting it from the remaining observed dimensions using
    conditional normal updates.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_tac_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        covariance: torch.Tensor,
    ) -> None:
        """
        Update state with predictions, targets, and covariance.

        Args:
            y_pred: Predicted mean of shape [B, D]
            y_true: True targets of shape [B, D]
            covariance: Predicted covariance matrices of shape [B, D, D]
        """
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)
        covariance = convert_to_tensor(covariance)

        validate_inputs(y_pred, y_true)

        B, d = y_pred.shape
        if covariance.shape != (B, d, d):
            raise ValueError(
                f"covariance shape {list(covariance.shape)} must match [batch, dim, dim] "
                f"where dim is {d}"
            )

        device = y_pred.device
        dtype = y_pred.dtype
        tac_errors = []

        # Iterate over each target dimension
        for i in range(d):
            obs_idx = [j for j in range(d) if j != i]

            # Extract sub-matrices
            sigma_12 = covariance[:, i, obs_idx].unsqueeze(1)  # [B, 1, d-1]
            sigma_22 = covariance[:, obs_idx][:, :, obs_idx]  # [B, d-1, d-1]

            # Add jitter to Sigma_22 for stable inversion
            eye_jitter = torch.eye(d - 1, device=device, dtype=dtype).unsqueeze(0) * 1e-6
            sigma_22 = sigma_22 + eye_jitter

            diff_obs = (y_true[:, obs_idx] - y_pred[:, obs_idx]).unsqueeze(-1)  # [B, d-1, 1]

            # Solve Sigma_22 \\ diff_obs
            sol = torch.linalg.solve(sigma_22, diff_obs)  # [B, d-1, 1]

            # Conditional update: mean_i + Sigma_12 @ sol
            update = torch.bmm(sigma_12, sol).squeeze(-1).squeeze(-1)  # [B]
            y_pred_i_updated = y_pred[:, i] + update

            err = torch.abs(y_pred_i_updated - y_true[:, i])  # [B]
            tac_errors.append(err)

        # Stack along dimensions (B, d) and sum the total absolute error
        tac_errors = torch.stack(tac_errors, dim=1)  # [B, d]
        sum_error = torch.sum(tac_errors)

        metric_state_tensor(self.sum_tac_error).add_(sum_error)
        metric_state_tensor(self.total).add_(torch.as_tensor(B * d, device=device))

    def compute(self) -> torch.Tensor:
        """Compute the average TAC error."""
        return metric_state_tensor(self.sum_tac_error) / metric_state_tensor(self.total)


def task_agnostic_correlations(
    y_pred: Union[torch.Tensor, np.ndarray],
    y_true: Union[torch.Tensor, np.ndarray],
    covariance: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Functional wrapper for :class:`TaskAgnosticCorrelations`."""
    metric = TaskAgnosticCorrelations()
    metric.update(
        convert_to_tensor(y_pred),
        convert_to_tensor(y_true),
        convert_to_tensor(covariance),
    )
    return metric.compute()
