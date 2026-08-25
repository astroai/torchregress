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

    References
    ----------
    .. [1] Shukla, S., et al. (2024). TIC-TAC: A Framework For Improved Covariance
       Estimation In Deep Heteroscedastic Regression. In *ICML 2024*.
       https://arxiv.org/abs/2310.18953
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

        # Add jitter to full covariance for stable inversion
        eye_jitter = torch.eye(d, device=device, dtype=dtype).unsqueeze(0) * 1e-6
        cov_jittered = covariance + eye_jitter

        diff = (y_true - y_pred).unsqueeze(-1)  # [B, d, 1]

        # Compute precision matrix
        P = torch.linalg.inv(cov_jittered)  # [B, d, d]

        # Calculate errors simultaneously using Schur complement properties
        u = torch.bmm(P, diff).squeeze(-1)  # [B, d]
        P_diag = P.diagonal(dim1=-2, dim2=-1)  # [B, d]

        err = torch.abs(u / P_diag)  # [B, d]
        sum_error = torch.sum(err)

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
        convert_to_tensor(y_pred),  # ty: ignore[invalid-argument-type]  # torchmetrics update/compute overrides confuse ty
        convert_to_tensor(y_true),
        convert_to_tensor(covariance),
    )
    return metric.compute()  # ty: ignore[missing-argument]  # torchmetrics update/compute overrides confuse ty
