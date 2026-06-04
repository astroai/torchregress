"""
Decision support metrics for evaluating risk-coverage trade-offs and selective prediction.
"""

from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor
from torchmetrics import Metric

from torchregress.metrics.utils import convert_to_tensor, metric_state_list, validate_inputs


class RiskCoverageCurve(Metric):
    """
    Computes the Risk-Coverage Curve (RCC).

    The RCC shows how a performance metric (risk, e.g., MSE) changes as we
    gradually reject samples with the highest uncertainty (selective prediction).
    A well-calibrated uncertainty score should result in a curve where risk
    decreases as coverage decreases.

    Args:
        risk_fn: Callable that takes (y_pred, y_true) and returns per-sample risk.
                Defaults to squared error: (y_pred - y_true)**2.
        n_points: Number of points to evaluate on the curve (coverage levels).
                Default is 100.

    Example:
        >>> rcc = RiskCoverageCurve()
        >>> rcc.update(y_pred, y_true, uncertainty)
        >>> curve = rcc.compute()
        >>> # curve contains 'coverage' and 'risk' tensors.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        risk_fn: Optional[Any] = None,
        n_points: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.risk_fn = risk_fn if risk_fn is not None else lambda p, t: (p - t) ** 2
        self.n_points = n_points

        self.add_state("risks", default=[], dist_reduce_fx=None)
        self.add_state("uncertainties", default=[], dist_reduce_fx=None)

    def update(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        uncertainty: Tensor,
    ) -> None:
        """
        Update state with predictions, targets, and uncertainty scores.

        Args:
            y_pred: Model predictions [N, ...]
            y_true: Ground truth targets [N, ...]
            uncertainty: Uncertainty scores [N]. Higher means less certain.
        """
        y_pred = convert_to_tensor(y_pred)
        y_true = convert_to_tensor(y_true)
        uncertainty = convert_to_tensor(uncertainty).view(-1)

        validate_inputs(y_pred, y_true)
        if uncertainty.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"Uncertainty shape {uncertainty.shape} must match batch size {y_pred.shape[0]}"
            )

        # Compute per-sample risk
        risk = self.risk_fn(y_pred, y_true)
        if risk.dim() > 1:
            risk = risk.mean(dim=list(range(1, risk.dim())))

        metric_state_list[Tensor](self.risks).append(risk)
        metric_state_list[Tensor](self.uncertainties).append(uncertainty)

    def compute(self) -> Dict[str, Tensor]:
        """
        Compute the Risk-Coverage Curve.

        Returns:
            Dictionary containing:
                - 'coverage': Tensor of coverage levels [0 to 1]
                - 'risk': Tensor of mean risk at each coverage level
                - 'aurc': Area Under the Risk-Coverage Curve
        """
        risks_state = metric_state_list[Tensor](self.risks)
        uncertainties_state = metric_state_list[Tensor](self.uncertainties)
        if not risks_state:
            return {
                "coverage": torch.tensor([]),
                "risk": torch.tensor([]),
                "aurc": torch.tensor(0.0),
            }

        risks = torch.cat(risks_state)
        uncertainties = torch.cat(uncertainties_state)

        n_samples = risks.shape[0]
        if n_samples == 0:
            return {
                "coverage": torch.tensor([]),
                "risk": torch.tensor([]),
                "aurc": torch.tensor(0.0),
            }

        # Sort by uncertainty (ascending)
        sorted_indices = torch.argsort(uncertainties)
        sorted_risks = risks[sorted_indices]

        # Define coverage levels
        coverage_levels = torch.linspace(1.0 / n_samples, 1.0, self.n_points, device=risks.device)

        # Cumulative sum of risks for efficient mean calculation
        risk_cumsum = torch.cumsum(sorted_risks, dim=0)

        # Vectorized calculation of n_kept
        n_kept_vec = torch.round(coverage_levels * n_samples).long()
        n_kept_vec = torch.clamp(n_kept_vec, min=1)

        # Vectorized risk at coverage
        risk_at_coverage = risk_cumsum[n_kept_vec - 1] / n_kept_vec

        # Compute AURC (Area Under Risk-Coverage Curve)
        # We use a simple trapezoidal rule over the computed points
        aurc = torch.trapz(risk_at_coverage, coverage_levels)

        return {
            "coverage": coverage_levels,
            "risk": risk_at_coverage,
            "aurc": aurc,
        }


class RejectionPolicy(Metric):
    """
    Evaluates a model's performance under a fixed rejection threshold or fraction.

    Args:
        risk_fn: Callable that takes (y_pred, y_true).
        threshold: Optional fixed uncertainty threshold.
        fraction: Optional fixed fraction of samples to reject.
                 If both are provided, 'fraction' takes precedence.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        risk_fn: Optional[Any] = None,
        threshold: Optional[float] = None,
        fraction: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.risk_fn = risk_fn if risk_fn is not None else lambda p, t: (p - t) ** 2
        self.threshold = threshold
        self.fraction = fraction

        self.add_state("y_pred", default=[], dist_reduce_fx=None)
        self.add_state("y_true", default=[], dist_reduce_fx=None)
        self.add_state("uncertainty", default=[], dist_reduce_fx=None)

    def update(
        self,
        y_pred: Tensor,
        y_true: Tensor,
        uncertainty: Tensor,
    ) -> None:
        metric_state_list[Tensor](self.y_pred).append(convert_to_tensor(y_pred))
        metric_state_list[Tensor](self.y_true).append(convert_to_tensor(y_true))
        metric_state_list[Tensor](self.uncertainty).append(convert_to_tensor(uncertainty).view(-1))

    def compute(self) -> Dict[str, Tensor]:
        y_pred_state = metric_state_list[Tensor](self.y_pred)
        y_true_state = metric_state_list[Tensor](self.y_true)
        unc_state = metric_state_list[Tensor](self.uncertainty)
        if not y_pred_state:
            return {}

        y_pred = torch.cat(y_pred_state)
        y_true = torch.cat(y_true_state)
        uncertainty = torch.cat(unc_state)

        n_samples = y_pred.shape[0]
        if n_samples == 0:
            return {}

        if self.fraction is not None:
            # Determine threshold from fraction
            n_keep = max(1, int(round((1 - self.fraction) * n_samples)))
            sorted_unc, _ = torch.sort(uncertainty)
            effective_threshold = sorted_unc[n_keep - 1]
        elif self.threshold is not None:
            effective_threshold = torch.tensor(self.threshold, device=y_pred.device)
        else:
            raise ValueError("Either 'threshold' or 'fraction' must be provided.")

        keep_mask = uncertainty <= effective_threshold

        if not keep_mask.any():
            return {
                "mean_risk": torch.tensor(float("nan")),
                "coverage": torch.tensor(0.0),
                "n_rejected": torch.tensor(float(n_samples)),
            }

        y_pred_kept = y_pred[keep_mask]
        y_true_kept = y_true[keep_mask]

        risk = self.risk_fn(y_pred_kept, y_true_kept)
        mean_risk = torch.mean(risk)
        coverage = keep_mask.float().mean()
        n_rejected = (~keep_mask).sum()

        return {
            "mean_risk": mean_risk,
            "coverage": coverage,
            "n_rejected": n_rejected.float(),
        }


def risk_coverage_curve(
    y_pred: Union[Tensor, Any],
    y_true: Union[Tensor, Any],
    uncertainty: Union[Tensor, Any],
    risk_fn: Optional[Any] = None,
    n_points: int = 100,
) -> Dict[str, Tensor]:
    """Functional interface for Risk-Coverage Curve."""
    metric = RiskCoverageCurve(risk_fn=risk_fn, n_points=n_points)
    metric.update(y_pred, y_true, uncertainty)
    return metric.compute()
